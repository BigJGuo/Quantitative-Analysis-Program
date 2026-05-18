"""Unit tests for `CostOfCarry` using `InMemoryProvider` fixtures.

Exercises the full fetch_data -> calibrate -> predict -> validate path without
touching the network.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, Signal
from src.models.cost_of_carry_dividends.model import (
    CostOfCarry,
    build_default_expiry,
)
from src.models.cost_of_carry_dividends.signal import (
    fv_dividend_stream,
    theoretical_futures_discrete,
    time_to_maturity,
)
from src.models.cost_of_carry_dividends.types import (
    CostParameters,
    Dividend,
    DividendStream,
    IndexConstituent,
)


def _bars(timestamp: datetime, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Datetime": [pd.Timestamp(timestamp)],
            "Open": [close],
            "High": [close * 1.001],
            "Low": [close * 0.999],
            "Close": [close],
            "Volume": [1000],
        }
    )


def _daily_bars(timestamp: datetime, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp(timestamp)],
            "Open": [close],
            "High": [close * 1.001],
            "Low": [close * 0.999],
            "Close": [close],
            "Volume": [1000],
        }
    )


def _build_provider(
    *,
    now: datetime,
    spot: float = 5000.0,
    futures: float = 5050.0,
    short_rate_pct: float = 5.25,
    mid_rate_pct: float = 4.20,
    constituent_tickers: tuple[str, ...] = ("AAPL", "MSFT"),
) -> InMemoryProvider:
    intraday = {
        ("^GSPC", "1m", 1): _bars(now, spot),
        ("ES=F", "1m", 1): _bars(now, futures),
    }
    prices = {
        ("^IRX", "5d", "1d"): _daily_bars(now, short_rate_pct),
        ("^TNX", "5d", "1d"): _daily_bars(now, mid_rate_pct),
    }
    # Add a couple of dividends per constituent — one in window, one outside.
    div_series: dict[str, pd.Series] = {}
    for t in constituent_tickers:
        div_series[t] = pd.Series(
            [0.20, 0.20, 0.24, 0.24],
            index=pd.to_datetime(
                [
                    now - timedelta(days=270),
                    now - timedelta(days=180),
                    now - timedelta(days=90),
                    now + timedelta(days=10),
                ]
            ),
        )
    return InMemoryProvider(intraday=intraday, prices=prices, dividends=div_series)


def test_fetch_data_assembles_inputs() -> None:
    now = datetime(2026, 5, 18, 14, 30, tzinfo=UTC)
    expiry = now + timedelta(days=90)
    provider = _build_provider(now=now)
    model = CostOfCarry(
        expiry=expiry,
        constituents=(
            IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5),
            IndexConstituent(ticker="MSFT", weight=0.06, shares_factor=1.5),
        ),
        now_func=lambda: now,
    )
    data = model.fetch_data(provider)
    assert data.spot == pytest.approx(5000.0)
    assert data.futures == pytest.approx(5050.0)
    # 13-week and 10-year proxies should make it into the curve.
    assert 0.25 in data.rate_curve.rates
    assert 10.0 in data.rate_curve.rates
    # Dividends per share should be present for both constituents.
    assert set(data.dividends_per_share.keys()) == {"AAPL", "MSFT"}


def test_predict_emits_signal_with_required_metadata() -> None:
    now = datetime(2026, 5, 18, 14, 30, tzinfo=UTC)
    expiry = now + timedelta(days=90)
    provider = _build_provider(now=now, spot=5000.0, futures=5050.0)
    model = CostOfCarry(
        expiry=expiry,
        constituents=(
            IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5),
        ),
        cost_params=CostParameters(futures_band_points=1.0, etf_cost_band_bps=2.0),
        now_func=lambda: now,
    )
    data = model.fetch_data(provider)
    cal = model.calibrate(data)
    assert isinstance(cal, CalibrationResult)
    sig = model.predict(data)
    assert isinstance(sig, Signal)
    assert sig.ticker == "ES=F"
    assert sig.direction in {"long", "short", "flat"}
    assert 0.0 <= sig.strength <= 1.0
    for key in ("spot", "futures", "theoretical", "basis", "implied_repo", "tau_years"):
        assert key in sig.metadata


def test_predict_recovers_signal_when_basis_is_pricing_error() -> None:
    """Construct futures = F_theo + 5 points so the action is unambiguous."""

    now = datetime(2026, 5, 18, 14, 30, tzinfo=UTC)
    expiry = now + timedelta(days=60)
    # Synthetic dividend stream (one payment of 5 pts in window).
    div_stream = DividendStream(
        dividends=(
            Dividend(
                ex_date=now + timedelta(days=20), amount_index_points=5.0, source="announced"
            ),
        )
    )
    spot = 5000.0
    rate = 0.0525 + 0.0010  # 5.25% short rate + 10 bps stock loan default
    tau = time_to_maturity(now, expiry)
    fv = fv_dividend_stream(div_stream, rate, now, expiry)
    f_theo = theoretical_futures_discrete(spot, rate, tau, fv)
    market = f_theo + 5.0

    provider = _build_provider(now=now, spot=spot, futures=market)
    model = CostOfCarry(
        expiry=expiry,
        cost_params=CostParameters(futures_band_points=1.0, etf_cost_band_bps=2.0),
        dividend_stream_override=div_stream,
        now_func=lambda: now,
    )
    data = model.fetch_data(provider)
    sig = model.predict(data)
    assert sig.direction == "short"
    assert sig.metadata["basis"] == pytest.approx(5.0, abs=0.1)


def test_validate_runs_diagnostics_per_spec() -> None:
    now = datetime(2026, 5, 18, 14, 30, tzinfo=UTC)
    expiry = now + timedelta(days=90)
    provider = _build_provider(now=now, spot=5000.0, futures=5060.0)
    model = CostOfCarry(
        expiry=expiry,
        constituents=(
            IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5),
            IndexConstituent(ticker="MSFT", weight=0.06, shares_factor=1.5),
        ),
        now_func=lambda: now,
    )
    data = model.fetch_data(provider)
    model.calibrate(data)
    diag = model.validate(data)
    # Spec-required diagnostic keys.
    assert diag["status"] == "ok"
    assert "implied_repo" in diag
    assert "basis" in diag
    assert "tau_years" in diag
    assert "method_cross_check_bps" in diag
    # The two methods should agree to within a small tolerance (spec: 1-2 bps).
    assert abs(diag["method_cross_check_bps"]) < 5.0


def test_validate_implied_repo_matches_input_when_market_is_theoretical() -> None:
    """When the market F equals F_theo, validate() should report repo_ok=True."""

    now = datetime(2026, 5, 18, 14, 30, tzinfo=UTC)
    expiry = now + timedelta(days=60)
    spot = 5000.0
    rate = 0.0525 + 0.0010
    div_stream = DividendStream(
        dividends=(
            Dividend(ex_date=now + timedelta(days=20), amount_index_points=3.0),
        )
    )
    tau = time_to_maturity(now, expiry)
    fv = fv_dividend_stream(div_stream, rate, now, expiry)
    f_theo = theoretical_futures_discrete(spot, rate, tau, fv)
    provider = _build_provider(now=now, spot=spot, futures=f_theo)
    model = CostOfCarry(
        expiry=expiry,
        cost_params=CostParameters(futures_band_points=0.5, etf_cost_band_bps=2.0),
        dividend_stream_override=div_stream,
        now_func=lambda: now,
    )
    data = model.fetch_data(provider)
    diag = model.validate(data)
    assert diag["repo_ok"] is True
    assert math.isclose(diag["basis"], 0.0, abs_tol=1e-6)


def test_build_default_expiry_returns_third_friday() -> None:
    today = datetime(2026, 5, 18, tzinfo=UTC)
    expiry = build_default_expiry(today, months_out=3)
    # The third Friday of August 2026 is the 21st.
    assert expiry.year == 2026
    assert expiry.month == 8
    assert expiry.day == 21
    assert expiry.weekday() == 4  # Friday


def test_repeated_predict_records_basis_history() -> None:
    now = datetime(2026, 5, 18, 14, 30, tzinfo=UTC)
    expiry = now + timedelta(days=90)
    provider = _build_provider(now=now, spot=5000.0, futures=5060.0)
    model = CostOfCarry(
        expiry=expiry,
        now_func=lambda: now,
    )
    data = model.fetch_data(provider)
    for _ in range(5):
        model.predict(data)
    assert len(model._basis_history) == 5
    # Second calibration should observe these and feed into the band quantile;
    # 5 obs is below the threshold so we expect the fallback band.
    cal = model.calibrate(data)
    assert "band_points" in cal.fit_metrics
