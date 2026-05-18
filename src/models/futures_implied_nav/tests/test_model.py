"""Unit tests for `FuturesImpliedNAV` using `InMemoryProvider` fixtures.

These exercise the full fetch_data -> calibrate -> predict -> validate path
without touching the network.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import Signal
from src.models.futures_implied_nav.model import (
    FuturesImpliedNAV,
    build_panel_from_daily,
)
from src.models.futures_implied_nav.types import (
    CostParameters,
    HedgeSpec,
    NAVAnchor,
)


def _make_daily(
    *,
    ticker: str,
    start: datetime,
    n_days: int,
    base: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a synthetic daily OHLC frame mimicking yfinance output."""

    dates = [start + timedelta(days=i) for i in range(n_days)]
    rets = rng.normal(scale=0.005, size=n_days)
    prices = base * np.cumprod(1.0 + rets)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.999,
            "High": prices * 1.001,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.full(n_days, 1_000_000),
        }
    )


def _make_correlated_daily(
    *,
    base_returns: np.ndarray,
    beta: float,
    base_price: float,
    start: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """ETF price series whose returns are beta * hedge_returns + tiny noise."""

    n = len(base_returns)
    rets = beta * base_returns + rng.normal(scale=1e-5, size=n)
    dates = [start + timedelta(days=i) for i in range(n)]
    prices = base_price * np.cumprod(1.0 + rets)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.999,
            "High": prices * 1.001,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.full(n, 1_000_000),
        }
    )


def _provider_for_test(
    *,
    etf_ticker: str = "EWJ",
    hedge_ticker: str = "^N225",
    fx_ticker: str = "JPYUSD=X",
    n_days: int = 90,
    history_days: int = 90,
    beta: float = 0.9,
    seed: int = 7,
) -> InMemoryProvider:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2)

    # Drive everything off a single hedge-return series so the relationship is recoverable.
    hedge_rets = rng.normal(scale=0.01, size=n_days)
    hedge_prices = 30_000.0 * np.cumprod(1.0 + hedge_rets)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    hedge_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": hedge_prices * 0.999,
            "High": hedge_prices * 1.001,
            "Low": hedge_prices * 0.998,
            "Close": hedge_prices,
            "Volume": np.full(n_days, 1_000_000),
        }
    )
    etf_df = _make_correlated_daily(
        base_returns=hedge_rets,
        beta=beta,
        base_price=70.0,
        start=start,
        rng=rng,
    )
    fx_df = _make_daily(
        ticker=fx_ticker, start=start, n_days=n_days, base=0.0065, rng=rng
    )

    period = f"{history_days}d"
    prices = {
        (etf_ticker, period, "1d"): etf_df,
        (hedge_ticker, period, "1d"): hedge_df,
        (fx_ticker, period, "1d"): fx_df,
    }
    return InMemoryProvider(prices=prices)


def test_build_panel_from_daily_aligns_lengths() -> None:
    rng = np.random.default_rng(1)
    start = datetime(2024, 1, 2)
    etf = _make_daily(ticker="EWJ", start=start, n_days=10, base=70.0, rng=rng)
    hedge = _make_daily(ticker="^N225", start=start, n_days=10, base=30_000.0, rng=rng)
    panel = build_panel_from_daily(
        etf_ticker="EWJ",
        hedges=(HedgeSpec(ticker="^N225", kind="futures", currency="JPY"),),
        daily={"EWJ": etf, "^N225": hedge},
    )
    # n - 1 overnight windows.
    assert len(panel.dates) == 9
    assert len(panel.nav_close) == 9
    assert len(panel.hedge_close["^N225"]) == 9


def test_model_endtoend_with_in_memory_provider() -> None:
    provider = _provider_for_test()
    anchor_override = NAVAnchor(
        nav_close=70.0,
        strike_time=datetime(2024, 1, 1, 16, 0),
        hedge_close={"^N225": 30_000.0},
        fx_close={"JPYUSD=X": 0.0065},
    )
    model = FuturesImpliedNAV(
        etf_ticker="EWJ",
        hedges=(HedgeSpec(ticker="^N225", kind="futures", currency="JPY"),),
        fx_tickers=("JPYUSD=X",),
        history_days=90,
        cost_params=CostParameters(half_spread_bps=2.0, cost_band_bps=3.0),
        nav_anchor_override=anchor_override,
    )

    data = model.fetch_data(provider)
    cal = model.calibrate(data)
    assert cal.model_name == "futures_implied_nav"
    beta = cal.parameters["beta"]
    # Synthetic beta = 0.9; should recover to within 0.1.
    assert beta.betas[0] == pytest.approx(0.9, abs=0.1)

    signal = model.predict(data)
    assert isinstance(signal, Signal)
    assert signal.ticker == "EWJ"
    assert signal.direction in {"long", "short", "flat"}
    assert 0.0 <= signal.strength <= 1.0
    assert "fair_value" in signal.metadata
    assert "premium" in signal.metadata

    diagnostics = model.validate(data)
    assert diagnostics["status"] == "ok"
    assert diagnostics["n_obs"] > 0
    assert "r_squared" in diagnostics
    # Synthetic data is almost noiseless -> R^2 near 1.
    assert diagnostics["r_squared"] > 0.99


def test_predict_without_calibration_raises() -> None:
    provider = _provider_for_test()
    model = FuturesImpliedNAV(
        etf_ticker="EWJ",
        hedges=(HedgeSpec(ticker="^N225", kind="futures", currency="JPY"),),
        fx_tickers=("JPYUSD=X",),
        history_days=90,
    )
    data = model.fetch_data(provider)
    with pytest.raises(RuntimeError, match="calibrate"):
        model.predict(data)


def test_validate_runs_diagnostics_per_spec() -> None:
    """Diagnostic test corresponding to the spec's Validation and diagnostics section."""

    provider = _provider_for_test()
    model = FuturesImpliedNAV(
        etf_ticker="EWJ",
        hedges=(HedgeSpec(ticker="^N225", kind="futures", currency="JPY"),),
        fx_tickers=("JPYUSD=X",),
        history_days=90,
    )
    data = model.fetch_data(provider)
    model.calibrate(data)
    diagnostics = model.validate(data)
    # Per spec point 2: R^2 acceptance threshold is >= 0.85; synthetic data passes easily.
    assert diagnostics["r_squared_acceptable"] is True
    # Per spec point 4: beta box bounds 0 <= beta <= 1.5.
    assert diagnostics["beta_within_bounds"] is True
    # Residual sanity: zero-mean small variance.
    assert abs(diagnostics["residual_mean"]) < 0.005
