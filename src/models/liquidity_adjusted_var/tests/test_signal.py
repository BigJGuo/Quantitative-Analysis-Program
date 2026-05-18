"""Math-layer tests for the L-VaR signal module.

Each function is checked against a hand-worked value so the spec
formulas are auditable from the test alone.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.liquidity_adjusted_var.signal import (
    almgren_chriss_eta,
    almgren_chriss_simple_cost,
    assign_frtb_bucket,
    bangia_liquidity_cost,
    compute_arithmetic_returns,
    compute_l_var,
    concentration_ratio,
    fallback_spread_from_high_low,
    frtb_horizon_scaled_es,
    historical_es,
    historical_var,
    horizon_coverage_violations,
    liquidation_horizon_days,
    market_cap_bucket_days,
    relative_spread_from_bid_ask,
    snap_to_frtb_endpoint,
    stress_spread_l_var,
    tier_for_adv_ratio,
)
from src.models.liquidity_adjusted_var.types import (
    FRTB_HORIZON_ENDPOINTS,
    LiquidityVaRInputs,
    Position,
    TickerLiquidityStats,
)

# ---------------------------------------------------------------------------
# Spread helpers
# ---------------------------------------------------------------------------


def test_relative_spread_from_bid_ask_simple() -> None:
    # bid=100, ask=100.1 -> mid=100.05 -> spread/mid = 0.1 / 100.05
    assert relative_spread_from_bid_ask(100.0, 100.1) == pytest.approx(
        0.1 / 100.05, rel=1e-9
    )


def test_relative_spread_from_bid_ask_returns_none_for_invalid() -> None:
    assert relative_spread_from_bid_ask(0.0, 100.0) is None
    assert relative_spread_from_bid_ask(100.0, 0.0) is None
    assert relative_spread_from_bid_ask(101.0, 100.0) is None
    assert relative_spread_from_bid_ask(float("nan"), 100.0) is None


def test_fallback_spread_from_high_low_uses_0_25_factor() -> None:
    # High/Low ratio constant at 1.01 -> mean ratio - 1 = 0.01 -> 0.0025.
    highs = pd.Series([100.1] * 20)
    lows = pd.Series([99.108910] * 20)
    expected = (100.1 / 99.108910 - 1.0) * 0.25
    assert fallback_spread_from_high_low(highs, lows, lookback=20) == pytest.approx(
        expected, rel=1e-9
    )


def test_fallback_spread_clipped_to_unit_interval() -> None:
    # Pathological data: high = 5x low -> raw ratio 4 -> 0.25 * 4 = 1.0 (clip).
    highs = pd.Series([500.0] * 10)
    lows = pd.Series([100.0] * 10)
    assert fallback_spread_from_high_low(highs, lows) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Liquidation horizon and bucket assignment
# ---------------------------------------------------------------------------


def test_liquidation_horizon_floored_at_one_day() -> None:
    # 1000 shares at 100k ADV with 0.2 participation -> raw 1000 / 20000 = 0.05.
    assert liquidation_horizon_days(1000.0, 100_000.0, 0.2) == 1.0


def test_liquidation_horizon_scales_linearly() -> None:
    # 100k shares at 200k ADV at 25% participation -> 100k / 50k = 2.0 days.
    assert liquidation_horizon_days(100_000.0, 200_000.0, 0.25) == pytest.approx(2.0)


def test_liquidation_horizon_absolute_value_for_shorts() -> None:
    # Short 100k shares should clear in the same time as long 100k.
    assert liquidation_horizon_days(-100_000.0, 200_000.0, 0.25) == pytest.approx(2.0)


def test_market_cap_bucket() -> None:
    assert market_cap_bucket_days(50e9) == 10
    assert market_cap_bucket_days(5e9) == 20
    assert market_cap_bucket_days(1e9) == 60
    assert market_cap_bucket_days(None) == 60
    assert market_cap_bucket_days(0.0) == 60


def test_snap_to_frtb_endpoint() -> None:
    assert snap_to_frtb_endpoint(0.5) == 10
    assert snap_to_frtb_endpoint(10) == 10
    assert snap_to_frtb_endpoint(11) == 20
    assert snap_to_frtb_endpoint(30) == 40
    assert snap_to_frtb_endpoint(50) == 60
    assert snap_to_frtb_endpoint(80) == 120
    # Past the top endpoint, saturate.
    assert snap_to_frtb_endpoint(500) == 120


def test_assign_frtb_bucket_takes_max_then_snaps() -> None:
    # Large-cap (10d) but illiquid position needing 30d -> 40d FRTB.
    assert assign_frtb_bucket(market_cap=50e9, position_horizon_days=30.0) == 40
    # Small-cap (60d) overrides a short position horizon.
    assert assign_frtb_bucket(market_cap=500e6, position_horizon_days=1.0) == 60
    # Large-cap with a 1-day horizon stays at 10.
    assert assign_frtb_bucket(market_cap=50e9, position_horizon_days=1.0) == 10


def test_tier_assignment_matches_spec_table() -> None:
    # 1M position vs 50M ADV -> 0.02 days, tier 1.
    assert tier_for_adv_ratio(1_000_000.0, 50_000_000.0) == 1
    # 1M vs 5M ADV -> 0.2 days, tier 2.
    assert tier_for_adv_ratio(1_000_000.0, 5_000_000.0) == 2
    # 1M vs 1M ADV -> 1 day, tier 3.
    assert tier_for_adv_ratio(1_000_000.0, 1_000_000.0) == 3
    # 1M vs 200k ADV -> 5 days, tier 4.
    assert tier_for_adv_ratio(1_000_000.0, 200_000.0) == 4
    # Zero ADV -> tier 4 (defensive).
    assert tier_for_adv_ratio(1_000_000.0, 0.0) == 4


# ---------------------------------------------------------------------------
# Almgren-Chriss
# ---------------------------------------------------------------------------


def test_almgren_chriss_eta_zero_when_inputs_invalid() -> None:
    assert almgren_chriss_eta(daily_return_std=0.0, last_price=100.0, adv_dollar=1e6) == 0.0
    assert almgren_chriss_eta(daily_return_std=0.01, last_price=0.0, adv_dollar=1e6) == 0.0
    assert almgren_chriss_eta(daily_return_std=0.01, last_price=100.0, adv_dollar=0.0) == 0.0


def test_almgren_chriss_eta_formula() -> None:
    # c * sigma * P / ADV_dollar = 0.1 * 0.02 * 50 / 1e6 = 1e-7
    eta = almgren_chriss_eta(
        daily_return_std=0.02, last_price=50.0, adv_dollar=1_000_000.0, impact_coef=0.1
    )
    assert eta == pytest.approx(0.1 * 0.02 * 50.0 / 1_000_000.0, rel=1e-12)


def test_almgren_chriss_simple_cost_formula() -> None:
    cost = almgren_chriss_simple_cost(eta=1e-7, shares_held=10_000.0, horizon_days=2.0)
    assert cost == pytest.approx(1e-7 * 10_000.0**2 / 2.0)


def test_almgren_chriss_cost_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError):
        almgren_chriss_simple_cost(eta=1.0, shares_held=10.0, horizon_days=0.0)


# ---------------------------------------------------------------------------
# Bangia spread cost
# ---------------------------------------------------------------------------


def test_bangia_liquidity_cost_half_spread() -> None:
    # 50 bps spread on a 1M position -> 0.5 * 0.005 * 1e6 = 2500.
    assert bangia_liquidity_cost(0.005, 1_000_000.0) == pytest.approx(2500.0)


def test_bangia_liquidity_cost_uses_absolute_position() -> None:
    assert bangia_liquidity_cost(0.005, -1_000_000.0) == pytest.approx(2500.0)


def test_bangia_liquidity_cost_rejects_negative_spread() -> None:
    with pytest.raises(ValueError):
        bangia_liquidity_cost(-0.001, 1_000.0)


# ---------------------------------------------------------------------------
# Historical-sim VaR / ES
# ---------------------------------------------------------------------------


def test_historical_var_matches_numpy_quantile() -> None:
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # 90th percentile -> linear interp at index 9 = 9.1
    expected = float(np.quantile(losses, 0.9))
    assert historical_var(losses, 0.9) == expected


def test_historical_es_is_mean_above_cutoff() -> None:
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # Cutoff at 80% = 8.2 -> tail is {9, 10} -> mean 9.5.
    assert historical_es(losses, 0.80) == pytest.approx(9.5)


def test_historical_es_falls_back_to_max_when_empty_tail() -> None:
    # All losses equal -> quantile equals the max -> no strict >, fall back to max.
    losses = np.array([5.0] * 10)
    assert historical_es(losses, 0.95) == pytest.approx(5.0)


def test_historical_var_validates_alpha() -> None:
    with pytest.raises(ValueError):
        historical_var(np.array([1.0, 2.0]), 1.0)


# ---------------------------------------------------------------------------
# FRTB horizon-scaled ES
# ---------------------------------------------------------------------------


def test_frtb_scaling_factors_match_increments() -> None:
    # Build a panel where all losses are concentrated in a single bucket so
    # we can verify the scale factor mechanically.
    n = 200
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(size=(n, 2)) * 0.01
    panel = pd.DataFrame(rets, columns=["A", "B"])
    weights = np.array([1.0, 1.0])
    # Both names bucketed at 40 days -> only endpoints up to 40 contribute.
    horizons = np.array([40, 40])
    _, components = frtb_horizon_scaled_es(
        returns_panel=panel, weights=weights, horizons=horizons, alpha=0.95
    )
    by_endpoint = {c["endpoint"]: c for c in components}
    # Scale factors per the spec: sqrt((LH_j - LH_{j-1}) / 10).
    assert by_endpoint[10]["scale"] == pytest.approx(math.sqrt(10 / 10))
    assert by_endpoint[20]["scale"] == pytest.approx(math.sqrt(10 / 10))
    assert by_endpoint[40]["scale"] == pytest.approx(math.sqrt(20 / 10))
    assert by_endpoint[60]["scale"] == pytest.approx(math.sqrt(20 / 10))
    assert by_endpoint[120]["scale"] == pytest.approx(math.sqrt(60 / 10))
    # Buckets above 40 are empty.
    assert by_endpoint[60]["n_tickers_in_bucket"] == 0
    assert by_endpoint[120]["n_tickers_in_bucket"] == 0


def test_frtb_es_aggregates_with_l2_norm() -> None:
    # Hand-built panel with a single dominant loss day so ES is predictable.
    rets = np.zeros((100, 2))
    rets[0] = [-0.05, -0.10]  # large losses
    panel = pd.DataFrame(rets, columns=["A", "B"])
    weights = np.array([1_000_000.0, 1_000_000.0])
    horizons = np.array([10, 10])  # both in the smallest bucket
    es_agg, components = frtb_horizon_scaled_es(
        returns_panel=panel, weights=weights, horizons=horizons, alpha=0.95
    )
    # Only j=1 (endpoint 10) contributes; scale = 1 -> es_agg == ES_one_day.
    j1 = next(c for c in components if c["endpoint"] == 10.0)
    assert es_agg == pytest.approx(abs(j1["es_one_day"]))
    # Buckets 20/40/60/120 are restricted to horizons >= those endpoints.
    for c in components:
        if c["endpoint"] > 10.0:
            assert c["n_tickers_in_bucket"] == 0


def test_frtb_es_uses_subset_for_higher_buckets() -> None:
    # Two tickers, one in 10d bucket, one in 60d bucket.
    n = 300
    rng = np.random.default_rng(42)
    rets = rng.standard_normal(size=(n, 2)) * np.array([0.01, 0.02])
    panel = pd.DataFrame(rets, columns=["A", "B"])
    weights = np.array([1_000_000.0, 200_000.0])
    horizons = np.array([10, 60])
    _, components = frtb_horizon_scaled_es(
        returns_panel=panel, weights=weights, horizons=horizons, alpha=0.95
    )
    by_endpoint = {c["endpoint"]: c for c in components}
    assert by_endpoint[10]["n_tickers_in_bucket"] == 2
    assert by_endpoint[20]["n_tickers_in_bucket"] == 1
    assert by_endpoint[40]["n_tickers_in_bucket"] == 1
    assert by_endpoint[60]["n_tickers_in_bucket"] == 1
    assert by_endpoint[120]["n_tickers_in_bucket"] == 0


# ---------------------------------------------------------------------------
# Top-level compute
# ---------------------------------------------------------------------------


def test_compute_l_var_combines_es_and_spread(synthetic_inputs: LiquidityVaRInputs) -> None:
    result = compute_l_var(synthetic_inputs)
    # Liquidity cost = sum of 0.5 * spread * |position|.
    expected_spread = (
        0.5 * 0.0005 * 1_000_000.0
        + 0.5 * 0.002 * 500_000.0
        + 0.5 * 0.01 * 200_000.0
    )
    assert result.liquidity_cost_spread == pytest.approx(expected_spread)
    assert result.l_var == pytest.approx(result.es_frtb + result.liquidity_cost_spread)
    # L-VaR is strictly positive and exceeds the headline price-risk VaR.
    assert result.l_var > 0.0
    # FRTB ES under reasonable conditions exceeds 1-day historical VaR.
    assert result.es_frtb > 0.0


def test_compute_l_var_per_name_carries_diagnostics(
    synthetic_inputs: LiquidityVaRInputs,
) -> None:
    result = compute_l_var(synthetic_inputs)
    assert set(result.per_name.keys()) == {"LARGE", "MID", "SMALL"}
    for ticker, block in result.per_name.items():
        assert block["position_dollar"] == pytest.approx(
            next(p.dollar_value for p in synthetic_inputs.positions if p.ticker == ticker)
        )
        assert "spread_relative" in block
        assert "ac_cost" in block
        assert "frtb_bucket_days" in block
        assert "tier" in block


def test_compute_l_var_price_var_matches_historical_quantile(
    synthetic_inputs: LiquidityVaRInputs,
) -> None:
    weights = np.array([p.dollar_value for p in synthetic_inputs.positions], dtype=float)
    rets = synthetic_inputs.returns_panel.to_numpy(dtype=float)
    losses = -(rets @ weights)
    expected_var = float(np.quantile(losses, synthetic_inputs.alpha))
    assert compute_l_var(synthetic_inputs).price_var_1d == pytest.approx(expected_var)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_concentration_ratio_counts_oversize_positions(
    synthetic_inputs: LiquidityVaRInputs,
) -> None:
    # SMALL is 200k vs 500k ADV -> 40%, above 20% threshold.
    count = concentration_ratio(synthetic_inputs.positions, synthetic_inputs.ticker_stats)
    assert count == 1


def test_concentration_ratio_threshold_validation(
    synthetic_inputs: LiquidityVaRInputs,
) -> None:
    with pytest.raises(ValueError):
        concentration_ratio(
            synthetic_inputs.positions,
            synthetic_inputs.ticker_stats,
            max_participation_threshold=1.5,
        )


def test_horizon_coverage_no_violations_when_buckets_dominate(
    synthetic_inputs: LiquidityVaRInputs,
) -> None:
    # Fixture has SMALL at position_horizon ~2.67 days, FRTB bucket 60 -> no violation.
    assert horizon_coverage_violations(
        synthetic_inputs.positions, synthetic_inputs.ticker_stats
    ) == []


def test_horizon_coverage_flags_under_bucketed_name() -> None:
    pos = (Position(ticker="X", dollar_value=1.0),)
    stats = (
        TickerLiquidityStats(
            ticker="X",
            last_price=10.0,
            shares_held=1.0,
            adv_shares=10.0,
            adv_dollar=100.0,
            daily_return_std=0.01,
            spread_relative=0.01,
            spread_source="bid_ask",
            market_cap=None,
            position_horizon_days=15.0,
            frtb_bucket_days=10,  # under-bucketed: 10 < 15
            tier=4,
            ac_eta=0.0,
        ),
    )
    assert horizon_coverage_violations(pos, stats) == ["X"]


def test_stress_spread_l_var_scales_spread_component(
    synthetic_inputs: LiquidityVaRInputs,
) -> None:
    base = compute_l_var(synthetic_inputs)
    stressed = stress_spread_l_var(synthetic_inputs, shock_multiplier=5.0)
    # The price-risk piece is unchanged; only spread cost rises 5x.
    expected = base.es_frtb + 5.0 * base.liquidity_cost_spread
    assert stressed == pytest.approx(expected)


def test_stress_spread_l_var_rejects_nonpositive_multiplier(
    synthetic_inputs: LiquidityVaRInputs,
) -> None:
    with pytest.raises(ValueError):
        stress_spread_l_var(synthetic_inputs, shock_multiplier=0.0)


# ---------------------------------------------------------------------------
# compute_arithmetic_returns
# ---------------------------------------------------------------------------


def test_compute_arithmetic_returns_drops_leading_row() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.01], "B": [50.0, 49.5, 50.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    out = compute_arithmetic_returns(prices)
    assert len(out) == 2
    assert out.iloc[0]["A"] == pytest.approx(0.01)
    assert out.iloc[0]["B"] == pytest.approx(-0.01)


def test_compute_arithmetic_returns_empty_input_returns_empty() -> None:
    assert compute_arithmetic_returns(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Sanity check: all FRTB endpoints are recognised
# ---------------------------------------------------------------------------


def test_every_frtb_endpoint_is_an_acceptable_bucket() -> None:
    for ep in FRTB_HORIZON_ENDPOINTS:
        assert snap_to_frtb_endpoint(float(ep)) == ep
