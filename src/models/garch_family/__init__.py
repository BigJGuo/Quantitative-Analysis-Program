"""Public interface for the GARCH-family conditional-volatility model.

See `models/layer4_signals_ml/08_garch_family.md` for the full specification.

Public surface:

- ``GARCHModel``                — the `BaseModel` subclass registered as
                                  ``"garch_family"``.
- ``GARCHParams``, ``GARCHFit``, ``GARCHInputs``, ``GARCHForecastResult``
                                — dataclasses passed across the model boundary.
- ``calibrate``                 — `pd.Series -> CalibrationResult` MLE entry.
- Pure-math helpers              — `compute_sigma2_path`, `negloglik`,
                                  `forecast_variance`, plus all diagnostic
                                  primitives (Ljung-Box, ARCH-LM, sign-bias,
                                  Mincer-Zarnowitz, Kupiec, QLIKE, VaR).
"""

from __future__ import annotations

from src.models.garch_family.calibration import MODEL_NAME, calibrate
from src.models.garch_family.model import GARCHModel
from src.models.garch_family.signal import (
    annualize_vol_pct,
    arch_lm_test,
    compute_sigma2_path,
    expected_abs_z_gaussian,
    expected_abs_z_student_t,
    forecast_variance,
    kupiec_pof_test,
    ljung_box,
    log_returns_pct,
    mincer_zarnowitz_regression,
    negloglik,
    normal_cdf,
    normal_ppf,
    qlike_loss,
    sign_bias_test,
    standardized_residuals,
    student_t_logpdf,
    student_t_ppf,
    value_at_risk,
)
from src.models.garch_family.types import (
    VALID_DISTS,
    VALID_MEAN_MODELS,
    VALID_SPECS,
    GARCHFit,
    GARCHForecastResult,
    GARCHInputs,
    GARCHParams,
    GARCHSpec,
    InnovationDist,
    MeanModel,
)

__all__ = [
    "MODEL_NAME",
    "VALID_DISTS",
    "VALID_MEAN_MODELS",
    "VALID_SPECS",
    "GARCHFit",
    "GARCHForecastResult",
    "GARCHInputs",
    "GARCHModel",
    "GARCHParams",
    "GARCHSpec",
    "InnovationDist",
    "MeanModel",
    "annualize_vol_pct",
    "arch_lm_test",
    "calibrate",
    "compute_sigma2_path",
    "expected_abs_z_gaussian",
    "expected_abs_z_student_t",
    "forecast_variance",
    "kupiec_pof_test",
    "ljung_box",
    "log_returns_pct",
    "mincer_zarnowitz_regression",
    "negloglik",
    "normal_cdf",
    "normal_ppf",
    "qlike_loss",
    "sign_bias_test",
    "standardized_residuals",
    "student_t_logpdf",
    "student_t_ppf",
    "value_at_risk",
]
