"""Public interface for the factor-model / PCA cross-sectional risk model.

See `models/layer2_structural/04_factor_models_pca.md` for the full specification.
"""

from __future__ import annotations

from src.models.factor_models_pca.calibration import MODEL_NAME, calibrate
from src.models.factor_models_pca.model import FactorModelPCA
from src.models.factor_models_pca.signal import (
    align_returns_panel,
    compute_full_covariance,
    compute_log_returns,
    cross_sectional_factor_regression,
    diagonal_shrinkage,
    ljung_box_pvalue,
    marchenko_pastur_upper_edge,
    pca_decompose,
    portfolio_risk,
    residual_autocorrelation,
    residual_max_cross_corr,
    residual_returns,
    residualize,
    time_series_factor_regression,
    variance_explained_curve,
)
from src.models.factor_models_pca.types import (
    VALID_METHODS,
    FactorFit,
    FactorInputs,
    FactorMethod,
    PortfolioRisk,
)

__all__ = [
    "MODEL_NAME",
    "VALID_METHODS",
    "FactorFit",
    "FactorInputs",
    "FactorMethod",
    "FactorModelPCA",
    "PortfolioRisk",
    "align_returns_panel",
    "calibrate",
    "compute_full_covariance",
    "compute_log_returns",
    "cross_sectional_factor_regression",
    "diagonal_shrinkage",
    "ljung_box_pvalue",
    "marchenko_pastur_upper_edge",
    "pca_decompose",
    "portfolio_risk",
    "residual_autocorrelation",
    "residual_max_cross_corr",
    "residual_returns",
    "residualize",
    "time_series_factor_regression",
    "variance_explained_curve",
]
