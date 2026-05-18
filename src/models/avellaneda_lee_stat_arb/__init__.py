"""Public interface for the Avellaneda-Lee PCA-residual stat-arb model.

See `models/layer4_signals_ml/07_avellaneda_lee_stat_arb.md` for the full
specification.
"""

from __future__ import annotations

from src.models.avellaneda_lee_stat_arb.calibration import (
    DEFAULT_HALF_LIFE_BAND,
    DEFAULT_MIN_R_SQUARED,
    MODEL_NAME,
    calibrate,
)
from src.models.avellaneda_lee_stat_arb.model import (
    DEFAULT_SECTOR_ETFS,
    AvellanedaLeeStatArb,
)
from src.models.avellaneda_lee_stat_arb.signal import (
    POSITION_FLAT,
    POSITION_LONG,
    POSITION_SHORT,
    AR1Fit,
    FactorRegressionResult,
    OUParameters,
    adf_test_pvalue,
    cumulative_residual,
    factor_neutral_hedge,
    factor_regression,
    fit_ou_ar1,
    ou_parameters_from_ar1,
    pca_eigenportfolio_returns,
    position_decision,
    s_score,
    s_score_modified,
)
from src.models.avellaneda_lee_stat_arb.types import (
    VALID_FACTOR_MODES,
    DropReason,
    FactorMode,
    OUFit,
    SignalThresholds,
    StatArbFit,
    StatArbInputs,
)

__all__ = [
    "AR1Fit",
    "AvellanedaLeeStatArb",
    "DEFAULT_HALF_LIFE_BAND",
    "DEFAULT_MIN_R_SQUARED",
    "DEFAULT_SECTOR_ETFS",
    "DropReason",
    "FactorMode",
    "FactorRegressionResult",
    "MODEL_NAME",
    "OUFit",
    "OUParameters",
    "POSITION_FLAT",
    "POSITION_LONG",
    "POSITION_SHORT",
    "SignalThresholds",
    "StatArbFit",
    "StatArbInputs",
    "VALID_FACTOR_MODES",
    "adf_test_pvalue",
    "calibrate",
    "cumulative_residual",
    "factor_neutral_hedge",
    "factor_regression",
    "fit_ou_ar1",
    "ou_parameters_from_ar1",
    "pca_eigenportfolio_returns",
    "position_decision",
    "s_score",
    "s_score_modified",
]
