"""Tests for the calibration helpers and the public `calibrate` entry."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.almgren_chriss.calibration import (
    MODEL_NAME,
    average_daily_volume,
    calibrate,
    eta_almgren_2005,
    eta_half_spread,
    gamma_from_eta,
    realized_log_return_vol,
)
from src.models.almgren_chriss.types import (
    AlmgrenChrissInputs,
    ImpactParams,
    LiquidationProblem,
)


def test_realized_log_return_vol_recovers_known_sigma() -> None:
    """Simulate 1000 daily returns with sigma_annual = 0.20."""

    rng = np.random.default_rng(seed=123)
    n = 1000
    sigma_annual = 0.20
    sigma_daily = sigma_annual / math.sqrt(252)
    rets = rng.normal(loc=0.0, scale=sigma_daily, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    s = pd.Series(prices, index=pd.date_range("2018-01-02", periods=n, freq="B"))
    est = realized_log_return_vol(s, window=500)
    assert est == pytest.approx(sigma_annual, rel=0.10)


def test_realized_log_return_vol_falls_back_for_short_series() -> None:
    s = pd.Series(np.linspace(100.0, 110.0, 8), index=pd.date_range("2024-01-02", periods=8))
    # Window of 30 forces fallback to len-1 = 7 returns.
    est = realized_log_return_vol(s, window=30)
    assert est > 0.0


def test_average_daily_volume_uses_tail() -> None:
    vals = np.concatenate([np.full(50, 1.0e6), np.full(20, 3.0e6)])
    s = pd.Series(vals, index=pd.date_range("2024-01-02", periods=70, freq="B"))
    adv = average_daily_volume(s, window=20)
    assert adv == pytest.approx(3.0e6, rel=1.0e-9)


def test_average_daily_volume_rejects_all_zero() -> None:
    s = pd.Series(np.zeros(30), index=pd.date_range("2024-01-02", periods=30, freq="B"))
    with pytest.raises(ValueError, match="non-positive"):
        average_daily_volume(s)


def test_eta_almgren_2005_formula() -> None:
    sigma_ann, V, P = 0.25, 5.0e7, 100.0
    eta = eta_almgren_2005(
        sigma_annual=sigma_ann, daily_volume=V, last_price=P
    )
    sigma_daily = sigma_ann / math.sqrt(252)
    expected = 0.6 * 0.142 * sigma_daily * P / V
    assert eta == pytest.approx(expected, rel=1.0e-12)


def test_eta_half_spread_formula() -> None:
    eta = eta_half_spread(half_spread_bps=2.0, daily_volume=1.0e7, last_price=100.0)
    expected = (2.0 / 1.0e4 * 100.0) / 1.0e7
    assert eta == pytest.approx(expected, rel=1.0e-12)


def test_gamma_from_eta_inverse_of_halflife() -> None:
    eta = 1.0e-6
    g1 = gamma_from_eta(eta, halflife_days=1.0)
    g10 = gamma_from_eta(eta, halflife_days=10.0)
    assert g1 == pytest.approx(eta / 10.0)
    assert g10 == pytest.approx(eta / 100.0)


def test_calibrate_end_to_end_returns_impact_params() -> None:
    rng = np.random.default_rng(seed=7)
    n = 60
    sigma_annual = 0.25
    sigma_daily = sigma_annual / math.sqrt(252)
    rets = rng.normal(loc=0.0, scale=sigma_daily, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    inputs = AlmgrenChrissInputs(
        ticker="TEST",
        close=pd.Series(prices, index=dates),
        volume=pd.Series(np.full(n, 5.0e7), index=dates),
        intraday_volume_profile=None,
        last_price=float(prices[-1]),
        problem=LiquidationProblem(
            ticker="TEST", X=1.0e5, T=1.0, N=10, side="sell", lam=1.0e-6
        ),
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
    )
    result = calibrate(inputs)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == MODEL_NAME
    impact = result.parameters["impact"]
    assert isinstance(impact, ImpactParams)
    # sigma should be in the ballpark of 0.25 (small-sample noise allowed).
    assert 0.10 < impact.sigma < 0.50
    assert impact.eta > 0.0
    assert impact.gamma > 0.0
    assert impact.last_price == pytest.approx(prices[-1])
    # Metrics populated.
    assert "sigma_annual" in result.fit_metrics
    assert "eta" in result.fit_metrics
    assert "gamma" in result.fit_metrics


def test_calibrate_supports_half_spread_rule() -> None:
    rng = np.random.default_rng(seed=8)
    n = 60
    rets = rng.normal(loc=0.0, scale=0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    inputs = AlmgrenChrissInputs(
        ticker="TEST",
        close=pd.Series(prices, index=dates),
        volume=pd.Series(np.full(n, 1.0e7), index=dates),
        intraday_volume_profile=None,
        last_price=float(prices[-1]),
        problem=LiquidationProblem(
            ticker="TEST", X=1.0e5, T=1.0, N=10, side="sell", lam=1.0e-6
        ),
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
    )
    result = calibrate(inputs, eta_rule="half_spread", half_spread_bps=2.0)
    impact = result.parameters["impact"]
    expected = (2.0 / 1.0e4 * inputs.last_price) / 1.0e7
    assert impact.eta == pytest.approx(expected, rel=1.0e-12)
    assert result.parameters["eta_rule"] == "half_spread"


def test_calibrate_rejects_unknown_eta_rule() -> None:
    inputs = AlmgrenChrissInputs(
        ticker="TEST",
        close=pd.Series([100.0, 101.0, 102.0]),
        volume=pd.Series([1.0e6, 1.0e6, 1.0e6]),
        intraday_volume_profile=None,
        last_price=102.0,
        problem=LiquidationProblem(
            ticker="TEST", X=1.0e5, T=1.0, N=10, side="sell", lam=1.0e-6
        ),
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="eta_rule"):
        calibrate(inputs, eta_rule="invalid_rule")  # type: ignore[arg-type]
