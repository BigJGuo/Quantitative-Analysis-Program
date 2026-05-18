"""Unit tests for the model-local dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.expected_shortfall.types import (
    FRTB_ALPHA,
    ESFit,
    ESInputs,
    ESResult,
)


def _make_returns(n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    rng = np.random.default_rng(seed=1)
    return pd.DataFrame(rng.standard_normal((n, 2)) * 0.01, index=idx, columns=["A", "B"])


def test_frtb_alpha_default_is_0_975() -> None:
    assert FRTB_ALPHA == 0.975


def test_es_inputs_rejects_unknown_method() -> None:
    rets = _make_returns()
    with pytest.raises(ValueError, match="method"):
        ESInputs(
            tickers=("A", "B"),
            returns=rets,
            dollar_positions=np.array([1.0, 1.0]),
            alpha=0.975,
            method="bogus",  # type: ignore[arg-type]
            lookback_N=500,
            timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        )


def test_es_inputs_rejects_bad_alpha() -> None:
    rets = _make_returns()
    with pytest.raises(ValueError, match="alpha"):
        ESInputs(
            tickers=("A", "B"),
            returns=rets,
            dollar_positions=np.array([1.0, 1.0]),
            alpha=1.5,
            method="historical",
            lookback_N=500,
            timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        )


def test_es_inputs_rejects_short_window() -> None:
    rets = _make_returns(n=100)
    with pytest.raises(ValueError, match="lookback_N"):
        ESInputs(
            tickers=("A", "B"),
            returns=rets,
            dollar_positions=np.array([1.0, 1.0]),
            alpha=0.975,
            method="historical",
            lookback_N=500,
            timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        )


def test_es_inputs_rejects_dim_mismatch() -> None:
    rets = _make_returns()
    with pytest.raises(ValueError, match="dollar_positions"):
        ESInputs(
            tickers=("A", "B"),
            returns=rets,
            dollar_positions=np.array([1.0, 1.0, 1.0]),
            alpha=0.975,
            method="historical",
            lookback_N=500,
            timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        )


def test_es_fit_requires_t_params_for_parametric_t() -> None:
    with pytest.raises(ValueError, match="parametric_t"):
        ESFit(
            mu=np.array([0.0]),
            Sigma=np.array([[1.0]]),
            sigma_p=1.0,
            method="parametric_t",
            alpha=0.975,
            n_obs=500,
        )


def test_es_fit_rejects_low_nu() -> None:
    with pytest.raises(ValueError, match="nu"):
        ESFit(
            mu=np.array([0.0]),
            Sigma=np.array([[1.0]]),
            sigma_p=1.0,
            method="parametric_t",
            alpha=0.975,
            n_obs=500,
            nu=1.5,
            loc=0.0,
            scale=1.0,
        )


def test_es_result_rejects_es_below_var() -> None:
    with pytest.raises(ValueError, match="ES"):
        ESResult(
            VaR=10.0,
            ES=5.0,
            method="historical",
            alpha=0.975,
            tail_losses=np.array([]),
            timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        )


def test_es_result_accepts_normal_relationship() -> None:
    r = ESResult(
        VaR=5.0,
        ES=8.0,
        method="historical",
        alpha=0.975,
        tail_losses=np.array([6.0, 7.0, 11.0]),
        timestamp=datetime(2026, 5, 18, tzinfo=UTC),
    )
    assert r.VaR < r.ES
