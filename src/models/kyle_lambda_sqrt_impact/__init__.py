"""Public interface for the Kyle-lambda / square-root impact model.

See ``models/layer5_execution/15_kyle_lambda_sqrt_impact.md`` for the full
specification.

Public surface:

- ``KyleSqrtImpact``    — the ``BaseModel`` subclass registered as
                          ``"kyle_lambda_sqrt_impact"``.
- ``KyleLambdaFit``, ``KyleInputs``, ``SqrtImpactPrediction``,
  ``CombinedImpactPrediction``
                         — dataclasses passed across the model boundary.
- ``calibrate``         — ``pd.DataFrame -> CalibrationResult`` OLS+HC1 entry.
- ``calibrate_daily``, ``calibrate_intraday``, ``estimate_intraday_pooled``
                         — frequency-specific calibrators.
- Pure-math helpers     — ``kyle_equilibrium_lambda``, ``kyle_equilibrium_beta``,
                          ``compute_signed_volume``, ``compute_log_returns``,
                          ``ols_slope_no_intercept``, ``hc1_se_no_intercept``,
                          ``r_squared_no_intercept``, ``wald_ci_95``,
                          ``sqrt_law_impact``, ``crossover_size``,
                          ``linear_impact``, ``participation_ratio_exceeds_threshold``,
                          ``realized_daily_vol``, ``adv``.
"""

from __future__ import annotations

from src.models.kyle_lambda_sqrt_impact.calibration import (
    MODEL_NAME,
    calibrate,
    calibrate_daily,
    calibrate_intraday,
    estimate_intraday_pooled,
)
from src.models.kyle_lambda_sqrt_impact.model import KyleSqrtImpact
from src.models.kyle_lambda_sqrt_impact.signal import (
    adv,
    compute_log_returns,
    compute_signed_volume,
    crossover_size,
    hc1_se_no_intercept,
    kyle_equilibrium_beta,
    kyle_equilibrium_lambda,
    linear_impact,
    ols_slope_no_intercept,
    participation_ratio_exceeds_threshold,
    r_squared_no_intercept,
    realized_daily_vol,
    sqrt_law_impact,
    wald_ci_95,
)
from src.models.kyle_lambda_sqrt_impact.types import (
    VALID_FREQUENCIES,
    CombinedImpactPrediction,
    KyleFrequency,
    KyleInputs,
    KyleLambdaFit,
    SqrtImpactPrediction,
)

__all__ = [
    "MODEL_NAME",
    "VALID_FREQUENCIES",
    "CombinedImpactPrediction",
    "KyleFrequency",
    "KyleInputs",
    "KyleLambdaFit",
    "KyleSqrtImpact",
    "SqrtImpactPrediction",
    "adv",
    "calibrate",
    "calibrate_daily",
    "calibrate_intraday",
    "compute_log_returns",
    "compute_signed_volume",
    "crossover_size",
    "estimate_intraday_pooled",
    "hc1_se_no_intercept",
    "kyle_equilibrium_beta",
    "kyle_equilibrium_lambda",
    "linear_impact",
    "ols_slope_no_intercept",
    "participation_ratio_exceeds_threshold",
    "r_squared_no_intercept",
    "realized_daily_vol",
    "sqrt_law_impact",
    "wald_ci_95",
]
