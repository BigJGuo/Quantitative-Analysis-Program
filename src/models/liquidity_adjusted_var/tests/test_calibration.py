"""Unit tests for `calibrate`."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.liquidity_adjusted_var.calibration import calibrate
from src.models.liquidity_adjusted_var.types import (
    Position,
    TickerLiquidityStats,
)


def test_calibrate_returns_per_ticker_stats(
    synthetic_positions: tuple[Position, ...],
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> None:
    prices, volumes, highs, lows = synthetic_prices_and_volumes
    result = calibrate(
        positions=synthetic_positions,
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        fundamentals=synthetic_fundamentals,
        max_participation=0.15,
        adv_lookback=60,
        timestamp=datetime(2026, 5, 18, tzinfo=UTC),
    )
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "liquidity_adjusted_var"
    stats = result.parameters["ticker_stats"]
    assert isinstance(stats, tuple)
    assert len(stats) == 3
    by_ticker = {s.ticker: s for s in stats}
    # All present in position order.
    assert tuple(s.ticker for s in stats) == ("LARGE", "MID", "SMALL")
    # LARGE: tight bid/ask captured.
    assert by_ticker["LARGE"].spread_source == "bid_ask"
    assert by_ticker["LARGE"].spread_relative == pytest.approx(
        (200.05 - 199.95) / 200.0, rel=1e-3
    )
    # LARGE is large-cap -> FRTB bucket 10d.
    assert by_ticker["LARGE"].frtb_bucket_days == 10
    # MID is mid-cap.
    assert by_ticker["MID"].frtb_bucket_days == 20
    # SMALL falls back to H/L proxy and uses 60d bucket override.
    assert by_ticker["SMALL"].spread_source == "hl_proxy"
    assert by_ticker["SMALL"].frtb_bucket_days == 60


def test_calibrate_tier_assignment_matches_adv_ratio(
    synthetic_positions: tuple[Position, ...],
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> None:
    prices, volumes, highs, lows = synthetic_prices_and_volumes
    result = calibrate(
        positions=synthetic_positions,
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        fundamentals=synthetic_fundamentals,
        max_participation=0.15,
    )
    by_ticker = {s.ticker: s for s in result.parameters["ticker_stats"]}
    # LARGE: 1M / 2B = 0.0005 days -> tier 1.
    assert by_ticker["LARGE"].tier == 1
    # MID: 500k / 50M = 0.01 days -> tier 1.
    assert by_ticker["MID"].tier == 1
    # SMALL: 200k / 500k = 0.4 days -> tier 2.
    assert by_ticker["SMALL"].tier == 2


def test_calibrate_position_horizon_for_thin_name(
    synthetic_positions: tuple[Position, ...],
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> None:
    prices, volumes, highs, lows = synthetic_prices_and_volumes
    result = calibrate(
        positions=synthetic_positions,
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        fundamentals=synthetic_fundamentals,
        max_participation=0.15,
    )
    by_ticker = {s.ticker: s for s in result.parameters["ticker_stats"]}
    # SMALL: shares_held = 200k / last_price; ADV = 20k shares; kappa = 0.15.
    # horizon = shares / (kappa * ADV) — should exceed 1 trading day.
    assert by_ticker["SMALL"].position_horizon_days > 1.0


def test_calibrate_metadata_reports_proxy_count(
    synthetic_positions: tuple[Position, ...],
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> None:
    prices, volumes, highs, lows = synthetic_prices_and_volumes
    result = calibrate(
        positions=synthetic_positions,
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        fundamentals=synthetic_fundamentals,
        max_participation=0.15,
    )
    assert result.metadata["n_hl_proxy_spreads"] == 1


def test_calibrate_raises_on_missing_volume_column(
    synthetic_positions: tuple[Position, ...],
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> None:
    prices, volumes, highs, lows = synthetic_prices_and_volumes
    volumes_bad = volumes.drop(columns=["SMALL"])
    with pytest.raises(ValueError):
        calibrate(
            positions=synthetic_positions,
            prices=prices,
            volumes=volumes_bad,
            highs=highs,
            lows=lows,
            fundamentals=synthetic_fundamentals,
            max_participation=0.15,
        )


def test_calibrate_rejects_empty_positions(
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> None:
    prices, volumes, highs, lows = synthetic_prices_and_volumes
    with pytest.raises(ValueError):
        calibrate(
            positions=(),
            prices=prices,
            volumes=volumes,
            highs=highs,
            lows=lows,
            fundamentals=synthetic_fundamentals,
            max_participation=0.15,
        )


def test_calibrate_rejects_bad_max_participation(
    synthetic_positions: tuple[Position, ...],
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    synthetic_fundamentals: dict[str, dict[str, float]],
) -> None:
    prices, volumes, highs, lows = synthetic_prices_and_volumes
    with pytest.raises(ValueError):
        calibrate(
            positions=synthetic_positions,
            prices=prices,
            volumes=volumes,
            highs=highs,
            lows=lows,
            fundamentals=synthetic_fundamentals,
            max_participation=1.5,
        )


def test_calibrate_handles_missing_fundamentals(
    synthetic_positions: tuple[Position, ...],
    synthetic_prices_and_volumes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Ticker missing from fundamentals dict should still calibrate via H/L proxy."""

    prices, volumes, highs, lows = synthetic_prices_and_volumes
    result = calibrate(
        positions=synthetic_positions,
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        fundamentals={},  # no fundamentals at all
        max_participation=0.15,
    )
    stats: tuple[TickerLiquidityStats, ...] = result.parameters["ticker_stats"]
    for s in stats:
        assert s.spread_source == "hl_proxy"
        assert s.market_cap is None
        # No market cap -> conservative 60d bucket (unless position horizon snaps higher).
        assert s.frtb_bucket_days >= 60
