"""Public interface for the Almgren-Chriss optimal-execution model.

See ``models/layer5_execution/16_almgren_chriss.md`` for the full
specification.

Public surface
--------------
- ``AlmgrenChriss``           — the ``BaseModel`` subclass registered as
                                ``"almgren_chriss"``.
- ``calibrate``               — ``AlmgrenChrissInputs -> CalibrationResult``.
- Schedule builders           — ``optimal_schedule`` (Algorithm A/B),
                                ``vwap_shaped_schedule`` (Algorithm C),
                                ``multi_asset_schedule`` (Algorithm D),
                                ``efficient_frontier``.
- Pure-math helpers           — ``compute_kappa``, ``sinh_ratio``,
                                ``inventory_path_continuous``,
                                ``inventory_path_discrete``,
                                ``expected_temporary_cost``,
                                ``cost_variance_closed_form``,
                                ``permanent_cost``,
                                ``annual_return_vol_to_dollar_per_day``.
- Calibration helpers         — ``realized_log_return_vol``,
                                ``average_daily_volume``,
                                ``eta_almgren_2005``,
                                ``eta_half_spread``,
                                ``gamma_from_eta``.
- Diagnostics                 — ``schedule_diagnostics``,
                                ``stress_schedule``.
- Dataclasses                 — ``AlmgrenChrissInputs``, ``ImpactParams``,
                                ``LiquidationProblem``, ``ExecutionSchedule``,
                                ``EfficientFrontier``, ``EfficientFrontierPoint``,
                                ``MultiAssetProblem``, ``MultiAssetSchedule``.
"""

from __future__ import annotations

from src.models.almgren_chriss.calibration import (
    MODEL_NAME,
    VALID_ETA_RULES,
    EtaRule,
    average_daily_volume,
    calibrate,
    eta_almgren_2005,
    eta_half_spread,
    gamma_from_eta,
    realized_log_return_vol,
)
from src.models.almgren_chriss.model import AlmgrenChriss
from src.models.almgren_chriss.signal import (
    annual_return_vol_to_dollar_per_day,
    compute_kappa,
    cost_variance_closed_form,
    discrete_kappa,
    efficient_frontier,
    expected_temporary_cost,
    intraday_volume_profile,
    inventory_path_continuous,
    inventory_path_discrete,
    multi_asset_schedule,
    optimal_schedule,
    permanent_cost,
    resample_profile,
    schedule_diagnostics,
    sinh_ratio,
    stress_schedule,
    vwap_shaped_schedule,
)
from src.models.almgren_chriss.types import (
    VALID_DISCRETIZATIONS,
    VALID_SIDES,
    AlmgrenChrissInputs,
    DiscretizationMode,
    EfficientFrontier,
    EfficientFrontierPoint,
    ExecutionSchedule,
    ImpactParams,
    LiquidationProblem,
    MultiAssetProblem,
    MultiAssetSchedule,
    Side,
)

__all__ = [
    "MODEL_NAME",
    "VALID_DISCRETIZATIONS",
    "VALID_ETA_RULES",
    "VALID_SIDES",
    "AlmgrenChriss",
    "AlmgrenChrissInputs",
    "DiscretizationMode",
    "EfficientFrontier",
    "EfficientFrontierPoint",
    "EtaRule",
    "ExecutionSchedule",
    "ImpactParams",
    "LiquidationProblem",
    "MultiAssetProblem",
    "MultiAssetSchedule",
    "Side",
    "annual_return_vol_to_dollar_per_day",
    "average_daily_volume",
    "calibrate",
    "compute_kappa",
    "cost_variance_closed_form",
    "discrete_kappa",
    "efficient_frontier",
    "eta_almgren_2005",
    "eta_half_spread",
    "expected_temporary_cost",
    "gamma_from_eta",
    "intraday_volume_profile",
    "inventory_path_continuous",
    "inventory_path_discrete",
    "multi_asset_schedule",
    "optimal_schedule",
    "permanent_cost",
    "realized_log_return_vol",
    "resample_profile",
    "schedule_diagnostics",
    "sinh_ratio",
    "stress_schedule",
    "vwap_shaped_schedule",
]
