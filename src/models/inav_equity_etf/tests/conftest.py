"""Shared synthetic fixtures for iNAV unit tests.

A deliberately small two-constituent ETF (one USD stock, one EUR stock) so
worked-out hand calculations stay tractable and the assertions stay readable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.models.inav_equity_etf.types import (
    BasketHolding,
    CashLedgerSeed,
    ConstituentQuote,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 18, 14, 30, tzinfo=UTC)


@pytest.fixture
def eod_release_time(now: datetime) -> datetime:
    # Prior NAV strike was at 4 PM ET the previous business day, ~22.5 hours ago.
    return now - timedelta(hours=22, minutes=30)


@pytest.fixture
def basket() -> tuple[BasketHolding, ...]:
    return (
        BasketHolding(ticker="AAPL", shares=1000.0, currency="USD", weight=0.6),
        BasketHolding(ticker="SAP.DE", shares=200.0, currency="EUR", weight=0.4),
    )


@pytest.fixture
def seed(eod_release_time: datetime) -> CashLedgerSeed:
    return CashLedgerSeed(
        eod_nav=100.0,
        eod_shares_out=10_000.0,
        eod_release_time=eod_release_time,
        prior_cash=50_000.0,
        prior_liabilities=200.0,
    )


@pytest.fixture
def quotes(now: datetime) -> dict[str, ConstituentQuote]:
    return {
        "AAPL": ConstituentQuote(
            ticker="AAPL", price=200.0, timestamp=now - timedelta(seconds=30)
        ),
        "SAP.DE": ConstituentQuote(
            ticker="SAP.DE", price=150.0, timestamp=now - timedelta(seconds=45)
        ),
    }


@pytest.fixture
def fx_rates() -> dict[str, float]:
    return {"EUR": 1.10}


def make_bars(
    *,
    timestamp: datetime,
    close: float,
    high: float | None = None,
    low: float | None = None,
) -> pd.DataFrame:
    hi = high if high is not None else close * 1.001
    lo = low if low is not None else close * 0.999
    return pd.DataFrame(
        {
            "Datetime": [pd.Timestamp(timestamp)],
            "Open": [close],
            "High": [hi],
            "Low": [lo],
            "Close": [close],
            "Volume": [1000],
        }
    )


@pytest.fixture
def provider(
    now: datetime,
    basket: tuple[BasketHolding, ...],
    quotes: dict[str, ConstituentQuote],
    fx_rates: dict[str, float],
) -> InMemoryProvider:
    intraday: dict[tuple[str, str, int], pd.DataFrame] = {}
    for holding in basket:
        q = quotes[holding.ticker]
        intraday[(holding.ticker, "1m", 1)] = make_bars(timestamp=q.timestamp, close=q.price)
    for currency, rate in fx_rates.items():
        sym = f"{currency}USD=X"
        intraday[(sym, "1m", 1)] = make_bars(
            timestamp=now, close=rate, high=rate, low=rate
        )
    # ETF own quote: choose a mid above iNAV to land in CREATE territory.
    intraday[("ETFX", "1m", 1)] = make_bars(timestamp=now, close=23.50)

    short_rate_df = pd.DataFrame(
        {"Date": [pd.Timestamp(now.date())], "Open": [5.25], "High": [5.30],
         "Low": [5.20], "Close": [5.25], "Volume": [0]}
    )
    prices = {("^IRX", "5d", "1d"): short_rate_df}

    fundamentals = {"ETFX": {"annualReportExpenseRatio": 0.0009}}
    dividends = {
        "AAPL": pd.Series(dtype="float64"),
        "SAP.DE": pd.Series(dtype="float64"),
    }
    holdings_df = pd.DataFrame(
        {"symbol": [h.ticker for h in basket], "weight": [h.weight for h in basket]}
    )

    return InMemoryProvider(
        intraday=intraday,
        prices=prices,
        fundamentals=fundamentals,
        dividends=dividends,
        holdings={"ETFX": holdings_df},
    )
