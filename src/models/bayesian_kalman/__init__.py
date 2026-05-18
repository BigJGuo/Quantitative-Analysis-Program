"""Public interface for the Bayesian-hierarchical / Kalman model.

See `models/layer4_signals_ml/12_bayesian_kalman.md` for the full specification.
"""

from __future__ import annotations

from src.models.bayesian_kalman.calibration import MODEL_NAME, calibrate
from src.models.bayesian_kalman.model import BayesianKalman
from src.models.bayesian_kalman.signal import (
    dynamic_factor_em,
    hierarchical_gibbs,
    hierarchical_posterior_mean,
    innovation_normality_kurtosis,
    kalman_filter,
    kalman_log_likelihood,
    kalman_smoother,
    ljung_box_pvalue,
    mle_time_varying_beta,
    standardized_innovations,
    time_varying_beta_filter,
)
from src.models.bayesian_kalman.types import (
    VALID_MODES,
    BayesianKalmanInputs,
    DynamicFactorFit,
    HierarchicalFit,
    KalmanFit,
    ModelMode,
)

__all__ = [
    "MODEL_NAME",
    "VALID_MODES",
    "BayesianKalman",
    "BayesianKalmanInputs",
    "DynamicFactorFit",
    "HierarchicalFit",
    "KalmanFit",
    "ModelMode",
    "calibrate",
    "dynamic_factor_em",
    "hierarchical_gibbs",
    "hierarchical_posterior_mean",
    "innovation_normality_kurtosis",
    "kalman_filter",
    "kalman_log_likelihood",
    "kalman_smoother",
    "ljung_box_pvalue",
    "mle_time_varying_beta",
    "standardized_innovations",
    "time_varying_beta_filter",
]
