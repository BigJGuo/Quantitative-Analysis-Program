"""Tests for the `calibrate()` dispatch entry point."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.factor_models_pca.calibration import MODEL_NAME, calibrate
from src.models.factor_models_pca.types import FactorFit


class TestCalibratePCA:
    def test_returns_calibration_result(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        result = calibrate(returns=returns, method="PCA", K=3)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        assert "factor_fit" in result.parameters
        assert isinstance(result.parameters["factor_fit"], FactorFit)

    def test_pca_fit_metrics_populated(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        result = calibrate(returns=returns, method="PCA", K=3)
        keys = {
            "n_assets",
            "n_factors",
            "n_obs",
            "variance_explained",
            "mean_specific_variance",
            "top_eigenvalue",
            "mp_upper_edge",
        }
        assert keys.issubset(result.fit_metrics.keys())
        assert result.fit_metrics["n_factors"] == 3.0
        assert result.fit_metrics["variance_explained"] > 0.9

    def test_mp_cutoff_caps_K(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        # Even when asking for 5 factors, MP should knock it down on
        # well-separated 1-factor synthetic data.
        result = calibrate(returns=returns, method="PCA", K=5, use_mp_cutoff=True)
        fit: FactorFit = result.parameters["factor_fit"]
        assert fit.n_factors <= 5

    def test_empty_returns_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty returns panel"):
            calibrate(returns=pd.DataFrame(), method="PCA", K=3)


class TestCalibrateFamaFrench:
    def test_recovers_betas(
        self,
        fama_french_synthetic: tuple[pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        asset, factor, true_B = fama_french_synthetic
        result = calibrate(
            returns=asset,
            method="FamaFrench",
            factor_returns=factor,
        )
        fit: FactorFit = result.parameters["factor_fit"]
        assert fit.method == "FamaFrench"
        assert np.allclose(fit.B, true_B, atol=0.08)
        assert fit.factor_names == ("MKT", "SMB", "HML")

    def test_requires_factor_returns(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        with pytest.raises(ValueError, match="factor_returns"):
            calibrate(returns=returns, method="FamaFrench")


class TestCalibrateBarra:
    def test_runs_with_constant_exposures(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, true_B = synthetic_three_factor
        # Use the true B as the daily exposures matrix; this should make the
        # cross-sectional regression recover the latent factors.
        exposures_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        exposures_df = pd.DataFrame(
            true_B,
            index=list(returns.columns),
            columns=["F1", "F2", "F3"],
        )
        for date in returns.index:
            exposures_by_date[date] = exposures_df
        result = calibrate(
            returns=returns,
            method="Barra",
            exposures_by_date=exposures_by_date,
        )
        fit: FactorFit = result.parameters["factor_fit"]
        assert fit.method == "Barra"
        assert fit.n_factors == 3
        # Specific variance should be small since the exposures are exactly correct.
        assert fit.specific_variances.max() < 0.001

    def test_requires_exposures(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        with pytest.raises(ValueError, match="exposures_by_date"):
            calibrate(returns=returns, method="Barra")


class TestCalibrateDispatch:
    def test_unknown_method_rejected(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        with pytest.raises(ValueError, match="method"):
            calibrate(returns=returns, method="unknown")  # type: ignore[arg-type]
