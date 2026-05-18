"""Validation tests for the L-VaR dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.models.liquidity_adjusted_var.types import (
    FRTB_HORIZON_ENDPOINTS,
    FRTB_T_BASE,
    LiquidityVaRInputs,
    Position,
    TickerLiquidityStats,
)


def test_frtb_constants_match_spec() -> None:
    assert FRTB_HORIZON_ENDPOINTS == (10, 20, 40, 60, 120)
    assert FRTB_T_BASE == 10


def test_position_rejects_empty_ticker() -> None:
    with pytest.raises(ValueError):
        Position(ticker="", dollar_value=1.0)


def test_position_accepts_negative_dollar_value() -> None:
    pos = Position(ticker="X", dollar_value=-500.0)
    assert pos.dollar_value == -500.0


def test_ticker_stats_rejects_non_endpoint_bucket() -> None:
    with pytest.raises(ValueError):
        TickerLiquidityStats(
            ticker="X",
            last_price=10.0,
            shares_held=100.0,
            adv_shares=10_000.0,
            adv_dollar=100_000.0,
            daily_return_std=0.01,
            spread_relative=0.001,
            spread_source="bid_ask",
            market_cap=1e9,
            position_horizon_days=1.0,
            frtb_bucket_days=15,  # not a valid FRTB endpoint
            tier=1,
            ac_eta=1.0e-9,
        )


def test_ticker_stats_rejects_negative_spread() -> None:
    with pytest.raises(ValueError):
        TickerLiquidityStats(
            ticker="X",
            last_price=10.0,
            shares_held=100.0,
            adv_shares=10_000.0,
            adv_dollar=100_000.0,
            daily_return_std=0.01,
            spread_relative=-0.001,
            spread_source="bid_ask",
            market_cap=1e9,
            position_horizon_days=1.0,
            frtb_bucket_days=10,
            tier=1,
            ac_eta=1.0e-9,
        )


def test_ticker_stats_rejects_invalid_tier() -> None:
    with pytest.raises(ValueError):
        TickerLiquidityStats(
            ticker="X",
            last_price=10.0,
            shares_held=100.0,
            adv_shares=10_000.0,
            adv_dollar=100_000.0,
            daily_return_std=0.01,
            spread_relative=0.001,
            spread_source="bid_ask",
            market_cap=1e9,
            position_horizon_days=1.0,
            frtb_bucket_days=10,
            tier=5,
            ac_eta=1.0e-9,
        )


def test_ticker_stats_rejects_horizon_below_floor() -> None:
    with pytest.raises(ValueError):
        TickerLiquidityStats(
            ticker="X",
            last_price=10.0,
            shares_held=100.0,
            adv_shares=10_000.0,
            adv_dollar=100_000.0,
            daily_return_std=0.01,
            spread_relative=0.001,
            spread_source="bid_ask",
            market_cap=1e9,
            position_horizon_days=0.5,
            frtb_bucket_days=10,
            tier=1,
            ac_eta=1.0e-9,
        )


def test_liquidity_var_inputs_rejects_mismatched_lengths(
    synthetic_positions: tuple[Position, ...],
    synthetic_ticker_stats: tuple[TickerLiquidityStats, ...],
    synthetic_returns_panel: pd.DataFrame,
    now: datetime,
) -> None:
    with pytest.raises(ValueError):
        LiquidityVaRInputs(
            positions=synthetic_positions,
            alpha=0.975,
            max_participation=0.15,
            returns_panel=synthetic_returns_panel,
            ticker_stats=synthetic_ticker_stats[:-1],
            timestamp=now,
        )


def test_liquidity_var_inputs_rejects_alpha_out_of_range(
    synthetic_positions: tuple[Position, ...],
    synthetic_ticker_stats: tuple[TickerLiquidityStats, ...],
    synthetic_returns_panel: pd.DataFrame,
    now: datetime,
) -> None:
    with pytest.raises(ValueError):
        LiquidityVaRInputs(
            positions=synthetic_positions,
            alpha=1.2,
            max_participation=0.15,
            returns_panel=synthetic_returns_panel,
            ticker_stats=synthetic_ticker_stats,
            timestamp=now,
        )


def test_liquidity_var_inputs_rejects_panel_column_mismatch(
    synthetic_positions: tuple[Position, ...],
    synthetic_ticker_stats: tuple[TickerLiquidityStats, ...],
    synthetic_returns_panel: pd.DataFrame,
    now: datetime,
) -> None:
    bad_panel = synthetic_returns_panel.rename(columns={"LARGE": "OTHER"})
    with pytest.raises(ValueError):
        LiquidityVaRInputs(
            positions=synthetic_positions,
            alpha=0.975,
            max_participation=0.15,
            returns_panel=bad_panel,
            ticker_stats=synthetic_ticker_stats,
            timestamp=now,
        )


def test_liquidity_var_inputs_rejects_short_panel(
    synthetic_positions: tuple[Position, ...],
    synthetic_ticker_stats: tuple[TickerLiquidityStats, ...],
) -> None:
    tiny_panel = pd.DataFrame(
        {"LARGE": [0.01] * 5, "MID": [0.01] * 5, "SMALL": [0.01] * 5},
        index=pd.date_range("2024-01-01", periods=5, freq="B"),
    )
    with pytest.raises(ValueError):
        LiquidityVaRInputs(
            positions=synthetic_positions,
            alpha=0.975,
            max_participation=0.15,
            returns_panel=tiny_panel,
            ticker_stats=synthetic_ticker_stats,
            timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        )
