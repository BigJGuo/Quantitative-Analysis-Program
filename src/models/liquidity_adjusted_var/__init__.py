"""Public interface for the Liquidity-Adjusted VaR model.

See ``models/layer6_risk/20_liquidity_adjusted_var.md`` for the full
specification.

Public surface:

- ``LiquidityAdjustedVaR``      — the `BaseModel` subclass registered as
                                  ``"liquidity_adjusted_var"``.
- ``Position``, ``TickerLiquidityStats``, ``LiquidityVaRInputs``,
  ``LVaRResult``                 — dataclasses passed across the model
                                  boundary.
- ``calibrate``                  — assemble per-ticker liquidity stats
                                  from prices / volumes / fundamentals.
- Pure-math helpers              — historical VaR / ES, FRTB
                                  horizon-scaled ES, Bangia-Diebold spread
                                  cost, Almgren-Chriss eta / cost,
                                  liquidation horizon, FRTB bucketing,
                                  tier assignment, and the diagnostic
                                  helpers (concentration ratio, stressed
                                  L-VaR, horizon coverage violations).
"""

from __future__ import annotations

from src.models.liquidity_adjusted_var.calibration import MODEL_NAME, calibrate
from src.models.liquidity_adjusted_var.model import LiquidityAdjustedVaR
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
    FRTB_T_BASE,
    VALID_TIERS,
    LiquidityVaRInputs,
    LVaRResult,
    Position,
    TickerLiquidityStats,
)

__all__ = [
    "FRTB_HORIZON_ENDPOINTS",
    "FRTB_T_BASE",
    "LVaRResult",
    "LiquidityAdjustedVaR",
    "LiquidityVaRInputs",
    "MODEL_NAME",
    "Position",
    "TickerLiquidityStats",
    "VALID_TIERS",
    "almgren_chriss_eta",
    "almgren_chriss_simple_cost",
    "assign_frtb_bucket",
    "bangia_liquidity_cost",
    "calibrate",
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
