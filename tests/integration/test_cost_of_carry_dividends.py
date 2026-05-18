"""End-to-end integration test for `CostOfCarry` against live yfinance.

Skipped automatically when yfinance is unreachable (offline CI agent) so the
suite does not produce spurious red builds. When connectivity is available
this exercises the full DataProvider -> fetch_data -> calibrate -> predict ->
validate pipeline on the S&P 500 front-month future (`ES=F`) and a handful of
representative S&P 500 constituents.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, Signal
from src.models.cost_of_carry_dividends import (
    CostOfCarry,
    CostParameters,
    IndexConstituent,
)
from src.models.cost_of_carry_dividends.model import build_default_expiry


def _skip_if_offline(provider: YFinanceProvider, ticker: str) -> None:
    try:
        df = provider.fetch_prices(ticker, period="5d", interval="1d")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance unreachable: {exc}")
    if df is None or len(df) == 0:
        pytest.skip(f"yfinance returned no data for {ticker}")


def test_endtoend_spx_pipeline() -> None:
    provider = YFinanceProvider()
    _skip_if_offline(provider, "ES=F")
    _skip_if_offline(provider, "^GSPC")

    today = datetime.now(UTC)
    expiry = build_default_expiry(today, months_out=3)

    # A short list of mega-cap dividend payers; index weights are illustrative
    # rather than authoritative.
    constituents = (
        IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5),
        IndexConstituent(ticker="MSFT", weight=0.07, shares_factor=1.5),
        IndexConstituent(ticker="JPM", weight=0.013, shares_factor=1.5),
    )

    model = CostOfCarry(
        futures_ticker="ES=F",
        spot_ticker="^GSPC",
        expiry=expiry,
        constituents=constituents,
        cost_params=CostParameters(futures_band_points=2.0, etf_cost_band_bps=5.0),
    )

    data = model.fetch_data(provider)
    assert data.spot > 0
    assert data.futures > 0
    assert data.rate_curve.tenors  # at least one pillar
    assert set(data.dividends_per_share.keys()) == {"AAPL", "MSFT", "JPM"}

    cal = model.calibrate(data)
    assert isinstance(cal, CalibrationResult)
    assert cal.model_name == "cost_of_carry_dividends"
    stream = cal.parameters["dividend_stream"]
    # We expect at least one dividend in the next quarter for these tickers,
    # but skip the assertion if all upstream feeds were empty (offline degrade).
    if not stream.dividends:
        pytest.skip("yfinance returned no dividends for any constituent")
    assert all(d.amount_index_points >= 0 for d in stream.dividends)

    sig = model.predict(data)
    assert isinstance(sig, Signal)
    assert sig.ticker == "ES=F"
    assert sig.direction in {"long", "short", "flat"}
    assert 0.0 <= sig.strength <= 1.0
    # Sanity bounds on the structural quantities.
    assert sig.metadata["spot"] > 0
    assert sig.metadata["futures"] > 0
    assert sig.metadata["theoretical"] > 0
    # SPX futures basis is typically < a few percent in absolute terms.
    relative_basis = sig.metadata["basis"] / sig.metadata["theoretical"]
    assert abs(relative_basis) < 0.05

    diag = model.validate(data)
    assert diag["status"] == "ok"
    assert 0.0 < diag["tau_years"] < 1.0
    # The implied repo should be in a reasonable range vs the input rate, even
    # if quarter-end pressure occasionally pushes it outside the strict 50 bp
    # spec band — keep this assertion loose for the live test.
    assert abs(diag["repo_spread_bps"]) < 500.0
    # Continuous vs discrete formulas should agree closely for short tau.
    assert abs(diag["method_cross_check_bps"]) < 50.0
