"""Shared synthetic fixtures for Liquidity-Adjusted VaR unit tests.

The fixtures encode a tiny three-ticker portfolio whose ADV, spread,
returns std, and price path are all chosen so the resulting L-VaR
quantities can be hand-checked against the spec formulas.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.models.liquidity_adjusted_var.types import (
    LiquidityVaRInputs,
    Position,
    TickerLiquidityStats,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 18, 14, 30, tzinfo=UTC)


@pytest.fixture
def synthetic_tickers() -> tuple[str, ...]:
    return ("LARGE", "MID", "SMALL")


@pytest.fixture
def synthetic_positions() -> tuple[Position, ...]:
    """Long 1M of large-cap, 500k of mid-cap, 200k of small-cap."""

    return (
        Position(ticker="LARGE", dollar_value=1_000_000.0),
        Position(ticker="MID", dollar_value=500_000.0),
        Position(ticker="SMALL", dollar_value=200_000.0),
    )


@pytest.fixture
def synthetic_returns_panel(synthetic_tickers: tuple[str, ...]) -> pd.DataFrame:
    """500 trading days of synthetic returns with known volatilities.

    Each column has a different std so the price-risk VaR is informative
    when the portfolio is weighted unevenly. Correlations are mild.
    """

    rng = np.random.default_rng(seed=20260518)
    t = 500
    base = rng.standard_normal(size=(t, 3))
    # Inject a market factor for mild cross-correlation.
    market = rng.standard_normal(size=t)
    base[:, 0] += 0.6 * market
    base[:, 1] += 0.5 * market
    base[:, 2] += 0.4 * market
    # Per-asset daily-vol targets: 1%, 1.5%, 2.5%.
    scale = np.array([0.01, 0.015, 0.025])
    rets = base * scale
    index = pd.date_range("2023-01-03", periods=t, freq="B")
    return pd.DataFrame(rets, index=index, columns=list(synthetic_tickers))


@pytest.fixture
def synthetic_prices_and_volumes(
    synthetic_returns_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cumulative prices, volumes, highs, lows consistent with the returns."""

    cum = (1.0 + synthetic_returns_panel).cumprod()
    # Anchor each ticker at a different starting price so ADV_dollars vary.
    anchors = pd.Series({"LARGE": 200.0, "MID": 50.0, "SMALL": 25.0})
    prices = cum.multiply(anchors, axis=1)
    # Prepend a starting row so pct_change inside compute_arithmetic_returns
    # is exercised consistently with downstream code paths.
    first_row = pd.DataFrame(
        [anchors.values], index=[prices.index[0] - pd.Timedelta(days=1)], columns=prices.columns
    )
    prices_full = pd.concat([first_row, prices])
    # Volumes: chosen so position-to-ADV ratios are well-separated.
    # LARGE: 10M shares/day at $200 = $2B/day -> 1M position is 0.05% of ADV.
    # MID:    1M shares/day at $50  = $50M/day -> 500k is 1% of ADV.
    # SMALL:  20k shares/day at $25 = $500k/day -> 200k is 40% of ADV (tier 3+).
    volumes_data = {
        "LARGE": np.full(len(prices_full), 10_000_000.0),
        "MID": np.full(len(prices_full), 1_000_000.0),
        "SMALL": np.full(len(prices_full), 20_000.0),
    }
    volumes = pd.DataFrame(volumes_data, index=prices_full.index)
    highs = prices_full * 1.005
    lows = prices_full * 0.995
    return prices_full, volumes, highs, lows


@pytest.fixture
def synthetic_fundamentals() -> dict[str, dict[str, float]]:
    """Bid/ask + market cap. LARGE is large-cap, MID is mid-cap, SMALL is micro-cap."""

    return {
        "LARGE": {"bid": 199.95, "ask": 200.05, "marketCap": 50e9},
        "MID": {"bid": 49.95, "ask": 50.05, "marketCap": 5e9},
        # SMALL: missing bid/ask so calibration falls back to the H/L proxy.
        "SMALL": {"bid": 0.0, "ask": 0.0, "marketCap": 500e6},
    }


@pytest.fixture
def synthetic_ticker_stats(
    synthetic_positions: tuple[Position, ...],
) -> tuple[TickerLiquidityStats, ...]:
    """Hand-built `TickerLiquidityStats` used by tests that don't need a fit."""

    return (
        TickerLiquidityStats(
            ticker="LARGE",
            last_price=200.0,
            shares_held=5_000.0,
            adv_shares=10_000_000.0,
            adv_dollar=2_000_000_000.0,
            daily_return_std=0.01,
            spread_relative=0.0005,  # 5 bps
            spread_source="bid_ask",
            market_cap=50e9,
            position_horizon_days=1.0,
            frtb_bucket_days=10,
            tier=1,
            ac_eta=1.0e-12,
        ),
        TickerLiquidityStats(
            ticker="MID",
            last_price=50.0,
            shares_held=10_000.0,
            adv_shares=1_000_000.0,
            adv_dollar=50_000_000.0,
            daily_return_std=0.015,
            spread_relative=0.002,  # 20 bps
            spread_source="bid_ask",
            market_cap=5e9,
            position_horizon_days=1.0,
            frtb_bucket_days=20,
            tier=2,
            ac_eta=1.0e-10,
        ),
        TickerLiquidityStats(
            ticker="SMALL",
            last_price=25.0,
            shares_held=8_000.0,
            adv_shares=20_000.0,
            adv_dollar=500_000.0,
            daily_return_std=0.025,
            spread_relative=0.01,  # 100 bps
            spread_source="hl_proxy",
            market_cap=500e6,
            position_horizon_days=2.6667,
            frtb_bucket_days=60,
            tier=4,
            ac_eta=2.5e-8,
        ),
    )


@pytest.fixture
def synthetic_inputs(
    synthetic_positions: tuple[Position, ...],
    synthetic_ticker_stats: tuple[TickerLiquidityStats, ...],
    synthetic_returns_panel: pd.DataFrame,
    now: datetime,
) -> LiquidityVaRInputs:
    return LiquidityVaRInputs(
        positions=synthetic_positions,
        alpha=0.975,
        max_participation=0.15,
        returns_panel=synthetic_returns_panel,
        ticker_stats=synthetic_ticker_stats,
        timestamp=now,
        metadata={"source": "synthetic"},
    )


@pytest.fixture
def in_memory_provider(
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> InMemoryProvider:
    """`InMemoryProvider` exposing the synthetic 3-ticker portfolio."""

    prices, volumes, highs, lows = synthetic_prices_and_volumes
    bars_by_ticker: dict[tuple[str, str, str], pd.DataFrame] = {}
    for ticker in prices.columns:
        bars_by_ticker[(ticker, "2y", "1d")] = pd.DataFrame(
            {
                "Date": [pd.Timestamp(d) for d in prices.index],
                "Open": prices[ticker].astype(float).to_numpy(),
                "High": highs[ticker].astype(float).to_numpy(),
                "Low": lows[ticker].astype(float).to_numpy(),
                "Close": prices[ticker].astype(float).to_numpy(),
                "Volume": volumes[ticker].astype(float).to_numpy(),
            }
        )
    return InMemoryProvider(
        prices=bars_by_ticker,
        fundamentals=synthetic_fundamentals,
    )
