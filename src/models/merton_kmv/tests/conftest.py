"""Shared synthetic fixtures for Merton-KMV unit tests.

A geometric-Brownian-motion synthetic firm with hand-checkable parameters so
worked-out assertions stay tractable.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.models.merton_kmv.signal import merton_call_price
from src.models.merton_kmv.types import BalanceSheetSnapshot


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 18, 14, 30, tzinfo=UTC)


@pytest.fixture
def balance_sheet() -> BalanceSheetSnapshot:
    # Default point D = 30 + 0.5*50 = 55 (in $M).
    return BalanceSheetSnapshot(
        short_term_debt=30.0,
        long_term_debt=50.0,
        cash_and_equivalents=5.0,
        total_debt=80.0,
    )


@pytest.fixture
def true_asset_value() -> float:
    return 150.0  # $M


@pytest.fixture
def true_asset_volatility() -> float:
    return 0.25


@pytest.fixture
def risk_free_rate() -> float:
    return 0.04


@pytest.fixture
def horizon_years() -> float:
    return 1.0


@pytest.fixture
def synthetic_asset_path(
    true_asset_value: float,
    true_asset_volatility: float,
    risk_free_rate: float,
) -> pd.Series:
    """Geometric Brownian motion of `V_t` over 252 trading days.

    Drift is set to `r` so the path is risk-neutral and the realized vol over
    the window will be close to `true_asset_volatility`.
    """

    n = 252
    rng = np.random.default_rng(seed=42)
    dt = 1.0 / 252.0
    shocks = rng.standard_normal(n)
    log_returns = (
        (risk_free_rate - 0.5 * true_asset_volatility**2) * dt
        + true_asset_volatility * math.sqrt(dt) * shocks
    )
    log_v0 = math.log(true_asset_value)
    log_v = np.concatenate([[log_v0], log_v0 + np.cumsum(log_returns)])
    dates = pd.date_range("2025-05-01", periods=len(log_v), freq="B")
    return pd.Series(np.exp(log_v), index=dates)


@pytest.fixture
def synthetic_equity_path(
    synthetic_asset_path: pd.Series,
    balance_sheet: BalanceSheetSnapshot,
    risk_free_rate: float,
    horizon_years: float,
    true_asset_volatility: float,
) -> pd.Series:
    """E_s = BS_call(V_s, D, r, T, sigma_V) — the model-consistent equity path."""

    d = balance_sheet.short_term_debt + 0.5 * balance_sheet.long_term_debt
    values = [
        merton_call_price(v, d, risk_free_rate, horizon_years, true_asset_volatility)
        for v in synthetic_asset_path.to_numpy(dtype=float)
    ]
    return pd.Series(values, index=synthetic_asset_path.index)


@pytest.fixture
def shares_outstanding() -> float:
    return 1.0  # `equity_market_value` already in $M, so 1 share = $1M equity


@pytest.fixture
def equity_market_value(synthetic_equity_path: pd.Series) -> float:
    return float(synthetic_equity_path.iloc[-1])


@pytest.fixture
def in_memory_provider(
    now: datetime,
    synthetic_equity_path: pd.Series,
    balance_sheet: BalanceSheetSnapshot,
    equity_market_value: float,
    shares_outstanding: float,
    risk_free_rate: float,
) -> InMemoryProvider:
    # Equity price history shaped like a yfinance OHLCV reset_index() frame.
    bars = pd.DataFrame(
        {
            "Date": [pd.Timestamp(d) for d in synthetic_equity_path.index],
            "Open": synthetic_equity_path.to_numpy(dtype=float),
            "High": synthetic_equity_path.to_numpy(dtype=float) * 1.01,
            "Low": synthetic_equity_path.to_numpy(dtype=float) * 0.99,
            "Close": synthetic_equity_path.to_numpy(dtype=float),
            "Volume": np.full(len(synthetic_equity_path), 1_000_000),
        }
    )
    prices = {("ACME", "1y", "1d"): bars}

    # Risk-free rate proxy (^IRX quotes percent).
    rate_bars = pd.DataFrame(
        {
            "Date": [pd.Timestamp(now.date()) - timedelta(days=i) for i in range(5)],
            "Open": np.full(5, risk_free_rate * 100.0),
            "High": np.full(5, risk_free_rate * 100.0),
            "Low": np.full(5, risk_free_rate * 100.0),
            "Close": np.full(5, risk_free_rate * 100.0),
            "Volume": np.zeros(5, dtype=int),
        }
    )
    prices[("^IRX", "5d", "1d")] = rate_bars

    fundamentals = {
        "ACME": {
            "marketCap": equity_market_value,
            "sharesOutstanding": shares_outstanding,
            "currentDebt": balance_sheet.short_term_debt,
            "longTermDebt": balance_sheet.long_term_debt,
            "totalDebt": balance_sheet.short_term_debt + balance_sheet.long_term_debt,
            "cash": balance_sheet.cash_and_equivalents,
            "industry": "Test Industry",
            "sector": "Industrials",
        }
    }

    return InMemoryProvider(
        prices=prices,
        fundamentals=fundamentals,
    )
