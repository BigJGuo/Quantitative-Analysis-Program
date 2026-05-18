"""Unit tests for `stress_tests/calibration.py`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.stress_tests.calibration import calibrate
from src.models.stress_tests.types import StressFit


class TestCalibrate:
    def test_returns_calibration_result(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> None:
        result = calibrate(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            capital=1_000_000.0,
        )
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "stress_tests"
        fit = result.parameters["fit"]
        assert isinstance(fit, StressFit)
        assert fit.n_assets == 3
        assert fit.n_factors == 2

    def test_betas_in_parameters_match_fit(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> None:
        result = calibrate(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            capital=1.0,
        )
        betas = result.parameters["betas"]
        assert isinstance(betas, pd.DataFrame)
        np.testing.assert_allclose(betas.to_numpy(), result.parameters["fit"].betas)

    def test_factor_covariance_psd(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> None:
        result = calibrate(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            capital=1.0,
        )
        Sigma = result.parameters["fit"].factor_covariance
        assert np.allclose(Sigma, Sigma.T)
        assert np.linalg.eigvalsh(Sigma).min() > -1e-12

    def test_recovers_known_equity_beta(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> None:
        # asset_returns was built with betas [0.6, 1.0, 1.4] onto the first factor.
        result = calibrate(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            capital=1.0,
        )
        betas = result.parameters["fit"].betas
        np.testing.assert_allclose(betas[:, 0], [0.6, 1.0, 1.4], atol=0.1)

    def test_rejects_nonpositive_capital(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match="capital"):
            calibrate(
                asset_returns=asset_returns,
                factor_returns=factor_returns,
                capital=0.0,
            )

    def test_rejects_empty_factor_returns(
        self,
        asset_returns: pd.DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match="factor_returns"):
            calibrate(
                asset_returns=asset_returns,
                factor_returns=pd.DataFrame(),
                capital=1.0,
            )

    def test_fit_metrics_populated(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> None:
        result = calibrate(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            capital=1.0,
        )
        for key in (
            "n_assets",
            "n_factors",
            "n_obs",
            "mean_abs_beta",
            "max_abs_beta",
            "factor_cov_trace",
        ):
            assert key in result.fit_metrics
