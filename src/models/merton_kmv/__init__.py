"""Public interface for the Merton-KMV structural credit model.

See `models/layer2_structural/03_merton_kmv.md` for the full specification.
"""

from __future__ import annotations

from src.models.merton_kmv.calibration import (
    MODEL_NAME,
    calibrate,
    iterative_kmv,
    joint_solve,
)
from src.models.merton_kmv.model import MertonKMV
from src.models.merton_kmv.signal import (
    cap_struct_arb_stance,
    classify_credit_quality,
    compute_merton_result,
    credit_spread,
    default_point,
    diagnostic_residuals,
    distance_to_default,
    equity_price_to_market_cap,
    equity_vol_from_asset_vol,
    estimate_drift_from_assets,
    kmv_summary_table,
    merton_call_price,
    merton_d1_d2,
    norm_cdf,
    prob_default,
    realized_equity_volatility,
    select_recent_prices,
    signal_direction_from_quality,
    signal_strength_from_dd,
    solve_for_v_given_e,
)
from src.models.merton_kmv.types import (
    ArbStance,
    BalanceSheetSnapshot,
    CreditQuality,
    KMVSolution,
    MertonInputs,
    MertonResult,
)

__all__ = [
    "MODEL_NAME",
    "ArbStance",
    "BalanceSheetSnapshot",
    "CreditQuality",
    "KMVSolution",
    "MertonInputs",
    "MertonKMV",
    "MertonResult",
    "calibrate",
    "cap_struct_arb_stance",
    "classify_credit_quality",
    "compute_merton_result",
    "credit_spread",
    "default_point",
    "diagnostic_residuals",
    "distance_to_default",
    "equity_price_to_market_cap",
    "equity_vol_from_asset_vol",
    "estimate_drift_from_assets",
    "iterative_kmv",
    "joint_solve",
    "kmv_summary_table",
    "merton_call_price",
    "merton_d1_d2",
    "norm_cdf",
    "prob_default",
    "realized_equity_volatility",
    "select_recent_prices",
    "signal_direction_from_quality",
    "signal_strength_from_dd",
    "solve_for_v_given_e",
]
