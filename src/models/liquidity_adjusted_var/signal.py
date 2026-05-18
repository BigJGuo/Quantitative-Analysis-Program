"""Pure-math layer for the Liquidity-Adjusted VaR model.

Every function in this module is `numpy`/`pandas`-only — no I/O, no class
state, no DataProvider calls. `LiquidityAdjustedVaR` composes these into a
single L-VaR computation.

Numerical conventions:

- ``returns_panel`` carries **arithmetic** daily returns (decimal, not
  percent). VaR / ES are reported in **dollars** so they aggregate cleanly
  with the additive Bangia spread cost.
- The FRTB liquidity-horizon combination follows BCBS d457 / d518: cumulative
  endpoints ``(10, 20, 40, 60, 120)`` trading days with implicit ``LH_0 = 0``
  and a base horizon ``T = 10`` days.
- All currency values share the position currency (typically USD).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.models.liquidity_adjusted_var.types import (
    FRTB_HORIZON_ENDPOINTS,
    FRTB_T_BASE,
    LiquidityVaRInputs,
    LVaRResult,
    Position,
    TickerLiquidityStats,
)

# Tier thresholds: number of days at 100% of ADV needed to clear the position.
# These match the spec's tier table (index futures, mid-cap, small-cap, exotic).
_TIER_THRESHOLDS_DAYS: tuple[float, ...] = (0.05, 0.5, 2.0)

# Market-cap → FRTB bucket from the spec's STEP 3.
_LARGE_CAP_USD: float = 10e9
_MID_CAP_USD: float = 2e9


# ---------------------------------------------------------------------------
# Return preprocessing
# ---------------------------------------------------------------------------


def compute_arithmetic_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide Close-price panel to arithmetic daily returns.

    NaNs introduced by `pct_change` (the leading row, plus any missing
    bars per ticker) are dropped row-wise so the resulting panel is rectangular
    and ready for matrix-multiply against a position vector.
    """

    if prices is None or prices.empty:
        return pd.DataFrame()
    rets = prices.astype(float).sort_index().pct_change()
    return rets.dropna(how="any")


# ---------------------------------------------------------------------------
# Liquidity horizon and bucket assignment
# ---------------------------------------------------------------------------


def liquidation_horizon_days(
    shares_held: float, adv_shares: float, max_participation: float
) -> float:
    """``|shares| / (kappa * ADV_shares)``, floored at 1 trading day.

    The floor encodes the spec's assumption that no position is liquidated
    in less than a single trading day; smaller raw values reflect a position
    that can be cleared inside a day at the chosen participation rate.
    """

    if adv_shares <= 0.0:
        raise ValueError(f"adv_shares must be > 0, got {adv_shares}")
    if not 0.0 < max_participation <= 1.0:
        raise ValueError(
            f"max_participation must lie in (0, 1], got {max_participation}"
        )
    raw = abs(shares_held) / (max_participation * adv_shares)
    return max(1.0, float(raw))


def market_cap_bucket_days(market_cap: float | None) -> int:
    """Spec STEP 3 bucketing by market cap.

    >10B → 10d (large-cap), >2B → 20d (mid/small-cap), else 60d
    (micro-cap / EM-like). Missing market cap is treated as micro-cap
    (60d) — the conservative default.
    """

    if market_cap is None or not math.isfinite(market_cap) or market_cap <= 0.0:
        return 60
    if market_cap > _LARGE_CAP_USD:
        return 10
    if market_cap > _MID_CAP_USD:
        return 20
    return 60


def snap_to_frtb_endpoint(days: float) -> int:
    """Round a horizon (in days) **up** to the nearest FRTB endpoint.

    FRTB capital is computed at discrete endpoints; a stock whose
    position-implied horizon is 35 days falls into the 40-day bucket.
    """

    if days <= 0.0 or not math.isfinite(days):
        return FRTB_HORIZON_ENDPOINTS[0]
    for endpoint in FRTB_HORIZON_ENDPOINTS:
        if days <= float(endpoint):
            return endpoint
    return FRTB_HORIZON_ENDPOINTS[-1]


def assign_frtb_bucket(
    *,
    market_cap: float | None,
    position_horizon_days: float,
) -> int:
    """Combine market-cap and position-based horizons, snapped to FRTB endpoints.

    Spec STEP 3: ``h_t = max(market_cap_bucket, ceil(position_horizon))``,
    then snapped up to the next standardized endpoint.
    """

    base = market_cap_bucket_days(market_cap)
    overridden = max(float(base), math.ceil(position_horizon_days))
    return snap_to_frtb_endpoint(overridden)


def tier_for_adv_ratio(position_dollar: float, adv_dollar: float) -> int:
    """Spec tier table on ADV-to-position ratio.

    Days-to-clear at 100% participation are
    ``|position| / ADV_dollar``. Thresholds (0.05, 0.5, 2.0) split into
    tiers 1, 2, 3 respectively; everything above is tier 4. Matches the
    spec table (index futures, mid-caps, small-caps, exotic).
    """

    if adv_dollar <= 0.0:
        return 4
    days = abs(position_dollar) / adv_dollar
    for tier_idx, threshold in enumerate(_TIER_THRESHOLDS_DAYS, start=1):
        if days <= threshold:
            return tier_idx
    return 4


# ---------------------------------------------------------------------------
# Spread estimation
# ---------------------------------------------------------------------------


def relative_spread_from_bid_ask(bid: float, ask: float) -> float | None:
    """``(ask - bid) / mid`` when both quotes are valid; ``None`` otherwise.

    ``None`` is the explicit "no usable quote" sentinel; callers should
    fall back to `fallback_spread_from_high_low` in that case.
    """

    if not math.isfinite(bid) or not math.isfinite(ask):
        return None
    if bid <= 0.0 or ask <= 0.0:
        return None
    if ask < bid:
        return None
    mid = 0.5 * (ask + bid)
    if mid <= 0.0:
        return None
    return float((ask - bid) / mid)


def fallback_spread_from_high_low(
    highs: pd.Series,
    lows: pd.Series,
    lookback: int = 20,
) -> float:
    """High-low range proxy for the relative spread.

    Spec STEP 2 fallback when ``info["bid"]`` / ``info["ask"]`` are zero
    or stale: ``mean(High/Low - 1) * 0.25`` over the last ``lookback``
    sessions. The 0.25 factor reflects the empirical
    Corwin-Schultz-style scaling between intraday range and effective
    spread.
    """

    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    h = highs.tail(lookback).astype(float).to_numpy()
    l_ = lows.tail(lookback).astype(float).to_numpy()
    if h.size == 0 or l_.size == 0:
        return 0.0
    mask = np.isfinite(h) & np.isfinite(l_) & (l_ > 0.0)
    if not mask.any():
        return 0.0
    ratio = h[mask] / l_[mask] - 1.0
    return float(np.clip(ratio.mean() * 0.25, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Almgren-Chriss simplified execution-cost model
# ---------------------------------------------------------------------------


def almgren_chriss_eta(
    *,
    daily_return_std: float,
    last_price: float,
    adv_dollar: float,
    impact_coef: float = 0.1,
) -> float:
    """Spec STEP 7 heuristic ``eta = c * sigma_dollar / ADV_dollar``.

    ``sigma_dollar = std(returns) * last_price`` is the per-share daily
    dollar volatility; dividing by ADV (in dollars) gives an impact
    coefficient that is small for liquid names and large for illiquid
    ones. The units are loose (the spec acknowledges this); replace with
    a TCA-fitted η in production.
    """

    if adv_dollar <= 0.0 or not math.isfinite(adv_dollar):
        return 0.0
    if daily_return_std <= 0.0 or not math.isfinite(daily_return_std):
        return 0.0
    if last_price <= 0.0 or not math.isfinite(last_price):
        return 0.0
    return float(impact_coef * daily_return_std * last_price / adv_dollar)


def almgren_chriss_simple_cost(
    *,
    eta: float,
    shares_held: float,
    horizon_days: float,
) -> float:
    """``eta * x0^2 / T`` per the spec's STEP 7 simplification.

    The full Almgren-Chriss optimal-trajectory cost depends on the
    risk-aversion parameter; this is the closed form for the constant-rate
    schedule a desk would use in practice when the regulator only cares
    about expected cost (not the variance) of execution.
    """

    if horizon_days <= 0.0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days}")
    return float(eta * shares_held * shares_held / horizon_days)


# ---------------------------------------------------------------------------
# Bangia-Diebold liquidity cost
# ---------------------------------------------------------------------------


def bangia_liquidity_cost(spread_relative: float, position_dollar: float) -> float:
    """``0.5 * spread * |position|``.

    The half-spread captures the round-trip cost of moving from mid to bid
    (or ask) on exit. ``position_dollar`` may be signed; we take the absolute
    value so a short pays the same exit cost as a long.
    """

    if spread_relative < 0.0:
        raise ValueError(
            f"spread_relative must be >= 0, got {spread_relative}"
        )
    return 0.5 * float(spread_relative) * abs(float(position_dollar))


# ---------------------------------------------------------------------------
# Historical-simulation VaR / ES
# ---------------------------------------------------------------------------


def historical_var(losses: np.ndarray, alpha: float) -> float:
    """Quantile-based historical-simulation VaR.

    ``losses`` is the vector of one-day portfolio losses (signs flipped from
    P&L). Returns ``quantile(losses, alpha)``.
    """

    if losses.size == 0:
        raise ValueError("historical_var requires a non-empty loss vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return float(np.quantile(losses, alpha))


def historical_es(losses: np.ndarray, alpha: float) -> float:
    """Average of losses strictly above the alpha-quantile.

    Empty tail (no observed loss above the threshold) returns the maximum
    loss as a safe upper bound — preferable to NaN in a risk pipeline.
    """

    if losses.size == 0:
        raise ValueError("historical_es requires a non-empty loss vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    cutoff = np.quantile(losses, alpha)
    tail = losses[losses > cutoff]
    if tail.size == 0:
        return float(losses.max())
    return float(tail.mean())


# ---------------------------------------------------------------------------
# FRTB liquidity-horizon scaled ES
# ---------------------------------------------------------------------------


def frtb_horizon_scaled_es(
    *,
    returns_panel: pd.DataFrame,
    weights: np.ndarray,
    horizons: np.ndarray,
    alpha: float,
    endpoints: tuple[int, ...] = FRTB_HORIZON_ENDPOINTS,
    base_horizon: int = FRTB_T_BASE,
) -> tuple[float, list[dict[str, float]]]:
    """BCBS d457 / d518 liquidity-horizon ES combination.

    For each cumulative endpoint ``LH_j`` (and implicit ``LH_0 = 0``):

    1. Restrict the portfolio to tickers with assigned horizon ``>= LH_j``
       (other weights zeroed).
    2. Compute the one-day historical-simulation ES of the restricted
       portfolio.
    3. Scale by ``sqrt((LH_j - LH_{j-1}) / base_horizon)``.

    The aggregated ES is the L2 norm of the scaled component ESs.

    Returns
    -------
    es_aggregate:
        ``sqrt(sum_j (ES_j * scale_j)^2)``, in the same units as the
        portfolio P&L.
    components:
        Per-endpoint diagnostics list: each item has ``endpoint``,
        ``increment_days``, ``scale``, ``n_tickers_in_bucket``,
        ``es_one_day``, ``es_scaled``.
    """

    n_assets = weights.size
    if horizons.shape != (n_assets,):
        raise ValueError(
            f"horizons shape {horizons.shape} must match weights ({n_assets},)"
        )
    if returns_panel.shape[1] != n_assets:
        raise ValueError(
            f"returns_panel has {returns_panel.shape[1]} columns, expected {n_assets}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if base_horizon <= 0:
        raise ValueError(f"base_horizon must be > 0, got {base_horizon}")

    rets = returns_panel.to_numpy(dtype=float)
    components: list[dict[str, float]] = []
    prev = 0
    accum_sq = 0.0
    for endpoint in endpoints:
        increment = endpoint - prev
        scale = math.sqrt(increment / base_horizon)
        mask = horizons >= endpoint
        bucket_count = int(mask.sum())
        if bucket_count == 0:
            components.append(
                {
                    "endpoint": float(endpoint),
                    "increment_days": float(increment),
                    "scale": float(scale),
                    "n_tickers_in_bucket": 0.0,
                    "es_one_day": 0.0,
                    "es_scaled": 0.0,
                }
            )
            prev = endpoint
            continue
        bucket_weights = np.where(mask, weights, 0.0)
        bucket_pnl = rets @ bucket_weights
        bucket_loss = -bucket_pnl
        es_one_day = historical_es(bucket_loss, alpha)
        scaled = es_one_day * scale
        accum_sq += scaled * scaled
        components.append(
            {
                "endpoint": float(endpoint),
                "increment_days": float(increment),
                "scale": float(scale),
                "n_tickers_in_bucket": float(bucket_count),
                "es_one_day": float(es_one_day),
                "es_scaled": float(scaled),
            }
        )
        prev = endpoint

    return math.sqrt(accum_sq), components


# ---------------------------------------------------------------------------
# Top-level computation
# ---------------------------------------------------------------------------


def compute_l_var(inputs: LiquidityVaRInputs) -> LVaRResult:
    """Run the spec's STEP 4 – STEP 8 pipeline on the prepared inputs.

    The expensive STEP 1 – STEP 3 work (price / volume / fundamentals
    download + per-name liquidity stats) lives in `calibration.calibrate`;
    this function is pure and deterministic on its inputs.
    """

    positions = inputs.positions
    stats_by_ticker = {s.ticker: s for s in inputs.ticker_stats}

    tickers = [p.ticker for p in positions]
    weights = np.asarray([p.dollar_value for p in positions], dtype=float)
    horizons = np.asarray(
        [stats_by_ticker[t].frtb_bucket_days for t in tickers], dtype=int
    )

    panel = inputs.returns_panel
    rets_arr = panel.to_numpy(dtype=float)
    portfolio_pnl = rets_arr @ weights
    losses = -portfolio_pnl
    var_1d = historical_var(losses, inputs.alpha)

    es_frtb, _components = frtb_horizon_scaled_es(
        returns_panel=panel,
        weights=weights,
        horizons=horizons,
        alpha=inputs.alpha,
    )

    liquidity_cost_spread = 0.0
    liquidity_cost_ac = 0.0
    per_name: dict[str, dict[str, object]] = {}
    for pos in positions:
        stats = stats_by_ticker[pos.ticker]
        spread_cost = bangia_liquidity_cost(stats.spread_relative, pos.dollar_value)
        ac_cost = almgren_chriss_simple_cost(
            eta=stats.ac_eta,
            shares_held=stats.shares_held,
            horizon_days=stats.position_horizon_days,
        )
        liquidity_cost_spread += spread_cost
        liquidity_cost_ac += ac_cost
        per_name[pos.ticker] = {
            "position_dollar": float(pos.dollar_value),
            "shares_held": float(stats.shares_held),
            "last_price": float(stats.last_price),
            "adv_shares": float(stats.adv_shares),
            "adv_dollar": float(stats.adv_dollar),
            "spread_relative": float(stats.spread_relative),
            "spread_source": stats.spread_source,
            "market_cap": float(stats.market_cap) if stats.market_cap is not None else None,
            "position_horizon_days": float(stats.position_horizon_days),
            "frtb_bucket_days": int(stats.frtb_bucket_days),
            "tier": int(stats.tier),
            "ac_eta": float(stats.ac_eta),
            "ac_cost": float(ac_cost),
            "spread_cost": float(spread_cost),
        }

    l_var = es_frtb + liquidity_cost_spread

    return LVaRResult(
        price_var_1d=var_1d,
        es_frtb=es_frtb,
        liquidity_cost_spread=liquidity_cost_spread,
        liquidity_cost_ac=liquidity_cost_ac,
        l_var=l_var,
        per_name=per_name,
        alpha=inputs.alpha,
        timestamp=inputs.timestamp,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def concentration_ratio(
    positions: Iterable[Position],
    ticker_stats: Iterable[TickerLiquidityStats],
    *,
    max_participation_threshold: float = 0.20,
) -> int:
    """Number of positions that cannot be cleared in one day at the threshold.

    Spec validation block 4: the count of positions whose dollar value
    exceeds ``threshold * ADV_dollar``. Persistent values above the
    book-level limit are a risk-management red flag.
    """

    if not 0.0 < max_participation_threshold <= 1.0:
        raise ValueError(
            f"max_participation_threshold must lie in (0, 1], "
            f"got {max_participation_threshold}"
        )
    stats_map = {s.ticker: s for s in ticker_stats}
    count = 0
    for pos in positions:
        stats = stats_map.get(pos.ticker)
        if stats is None or stats.adv_dollar <= 0.0:
            count += 1
            continue
        if abs(pos.dollar_value) > max_participation_threshold * stats.adv_dollar:
            count += 1
    return count


def stress_spread_l_var(inputs: LiquidityVaRInputs, *, shock_multiplier: float) -> float:
    """Recompute L-VaR with every relative spread scaled by ``shock_multiplier``.

    Spec validation block 5: scales all bid-ask spreads by 5× (or any
    user-chosen factor) to reveal exposure to a liquidity-regime shift.
    Price-risk ES is unchanged; only the additive spread cost moves.
    """

    if shock_multiplier <= 0.0:
        raise ValueError(
            f"shock_multiplier must be > 0, got {shock_multiplier}"
        )
    shocked_cost = 0.0
    stats_map = {s.ticker: s for s in inputs.ticker_stats}
    for pos in inputs.positions:
        stats = stats_map[pos.ticker]
        shocked_cost += bangia_liquidity_cost(
            stats.spread_relative * shock_multiplier, pos.dollar_value
        )
    weights = np.asarray([p.dollar_value for p in inputs.positions], dtype=float)
    horizons = np.asarray(
        [stats_map[p.ticker].frtb_bucket_days for p in inputs.positions], dtype=int
    )
    es_frtb, _ = frtb_horizon_scaled_es(
        returns_panel=inputs.returns_panel,
        weights=weights,
        horizons=horizons,
        alpha=inputs.alpha,
    )
    return es_frtb + shocked_cost


def horizon_coverage_violations(
    positions: Iterable[Position],
    ticker_stats: Iterable[TickerLiquidityStats],
) -> list[str]:
    """Names whose assigned FRTB bucket is shorter than the ADV-implied horizon.

    Spec validation block 3: the assigned horizon should always be >=
    ``position / (kappa * ADV)``. Violations indicate the FRTB endpoint
    snapping has under-bucketed a thinly-traded position.
    """

    stats_map = {s.ticker: s for s in ticker_stats}
    violations: list[str] = []
    for pos in positions:
        stats = stats_map.get(pos.ticker)
        if stats is None:
            continue
        if float(stats.frtb_bucket_days) < stats.position_horizon_days:
            violations.append(pos.ticker)
    return violations


__all__ = [
    "almgren_chriss_eta",
    "almgren_chriss_simple_cost",
    "assign_frtb_bucket",
    "bangia_liquidity_cost",
    "compute_arithmetic_returns",
    "compute_l_var",
    "concentration_ratio",
    "fallback_spread_from_high_low",
    "frtb_horizon_scaled_es",
    "historical_es",
    "historical_var",
    "horizon_coverage_violations",
    "liquidation_horizon_days",
    "market_cap_bucket_days",
    "relative_spread_from_bid_ask",
    "snap_to_frtb_endpoint",
    "stress_spread_l_var",
    "tier_for_adv_ratio",
]
