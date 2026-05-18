"""Public interface for the HAR-RV realized-volatility model.

See `models/layer4_signals_ml/09_har_rv.md` for the full specification.
"""

from __future__ import annotations

from src.models.har_rv.calibration import MODEL_NAME, calibrate
from src.models.har_rv.model import HARRVModel
from src.models.har_rv.signal import (
    DEFAULT_M_WINDOW,
    DEFAULT_MIN_RV,
    DEFAULT_W_WINDOW,
    aggregate_har_components,
    arch_lm_pvalue,
    build_har_design_matrix,
    compute_daily_rv,
    compute_garman_klass_rv,
    compute_intraday_log_returns,
    compute_yang_zhang_rv,
    diebold_mariano_pvalue,
    fit_har,
    forecast_one_step,
    in_sample_forecasts,
    latest_har_state,
    ljung_box_pvalue,
    mincer_zarnowitz,
    newey_west_covariance,
    nw_optimal_lag,
    ols_fit,
    qlike_loss,
)
from src.models.har_rv.types import (
    HAR_COEF_NAMES,
    VALID_SOURCES,
    VALID_SPECS,
    HARFit,
    HARRVInputs,
    HARSpec,
    RVComponents,
    RVSource,
)

__all__ = [
    "DEFAULT_MIN_RV",
    "DEFAULT_M_WINDOW",
    "DEFAULT_W_WINDOW",
    "HARFit",
    "HARRVInputs",
    "HARRVModel",
    "HARSpec",
    "HAR_COEF_NAMES",
    "MODEL_NAME",
    "RVComponents",
    "RVSource",
    "VALID_SOURCES",
    "VALID_SPECS",
    "aggregate_har_components",
    "arch_lm_pvalue",
    "build_har_design_matrix",
    "calibrate",
    "compute_daily_rv",
    "compute_garman_klass_rv",
    "compute_intraday_log_returns",
    "compute_yang_zhang_rv",
    "diebold_mariano_pvalue",
    "fit_har",
    "forecast_one_step",
    "in_sample_forecasts",
    "latest_har_state",
    "ljung_box_pvalue",
    "mincer_zarnowitz",
    "newey_west_covariance",
    "nw_optimal_lag",
    "ols_fit",
    "qlike_loss",
]
