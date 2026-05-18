"""Public interface for the cointegration / pair-trading model.

See `models/layer4_signals_ml/06_cointegration_pairs.md` for the full
specification.
"""

from __future__ import annotations

from src.models.cointegration_pairs.calibration import MODEL_NAME, calibrate
from src.models.cointegration_pairs.model import CointegrationPairs
from src.models.cointegration_pairs.signal import (
    adf_test,
    align_pair,
    compute_log_prices,
    engle_granger_test,
    fit_ar1,
    fit_ou,
    half_life_from_phi,
    kalman_dynamic_beta,
    ljung_box_pvalue,
    ols_intercept_slope,
    pair_position_path,
    realized_half_life,
    rolling_eg_residual_tstat,
    rolling_zscore_moments,
    select_dependent_orientation,
    static_zscore,
)
from src.models.cointegration_pairs.types import (
    VALID_METHODS,
    ADFResult,
    EngleGrangerFit,
    KalmanFit,
    OUFit,
    PairFit,
    PairInputs,
    PairMethod,
    PairStatus,
    TradingRule,
)

__all__ = [
    "ADFResult",
    "CointegrationPairs",
    "EngleGrangerFit",
    "KalmanFit",
    "MODEL_NAME",
    "OUFit",
    "PairFit",
    "PairInputs",
    "PairMethod",
    "PairStatus",
    "TradingRule",
    "VALID_METHODS",
    "adf_test",
    "align_pair",
    "calibrate",
    "compute_log_prices",
    "engle_granger_test",
    "fit_ar1",
    "fit_ou",
    "half_life_from_phi",
    "kalman_dynamic_beta",
    "ljung_box_pvalue",
    "ols_intercept_slope",
    "pair_position_path",
    "realized_half_life",
    "rolling_eg_residual_tstat",
    "rolling_zscore_moments",
    "select_dependent_orientation",
    "static_zscore",
]
