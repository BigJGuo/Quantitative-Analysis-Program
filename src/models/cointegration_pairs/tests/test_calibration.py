"""Unit tests for the `calibrate()` entry point."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.cointegration_pairs.calibration import MODEL_NAME, calibrate
from src.models.cointegration_pairs.types import PairFit


class TestCalibrateBasics:
    def test_model_name_constant(self) -> None:
        assert MODEL_NAME == "cointegration_pairs"

    def test_returns_calibration_result(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, _ = synthetic_cointegrated_pair
        result = calibrate(a, b, method="static")
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        fit = result.parameters["pair_fit"]
        assert isinstance(fit, PairFit)

    def test_metrics_include_required_keys(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, _ = synthetic_cointegrated_pair
        result = calibrate(a, b)
        for key in (
            "alpha",
            "beta",
            "adf_leg_a_tstat",
            "adf_leg_b_tstat",
            "eg_residual_tstat",
            "ou_phi",
            "half_life",
            "accepted",
        ):
            assert key in result.fit_metrics


class TestCalibrateAccepted:
    def test_cointegrated_pair_accepted(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, truth = synthetic_cointegrated_pair
        result = calibrate(a, b, method="static", significance="5%")
        fit = result.parameters["pair_fit"]
        assert fit.is_cointegrated
        assert math.isclose(fit.eg_fit.beta, truth["beta"], abs_tol=0.05)
        # Half-life should be inside the default band.
        lo, hi = fit.half_life_band
        assert lo <= fit.ou_fit.half_life <= hi

    def test_kalman_method_returns_kalman_fit(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, _ = synthetic_cointegrated_pair
        result = calibrate(a, b, method="kalman")
        fit = result.parameters["pair_fit"]
        assert fit.is_cointegrated
        assert fit.kalman_fit is not None
        assert fit.kalman_fit.beta_path.shape == (len(a),)


class TestCalibrateRejected:
    def test_independent_walks_rejected(
        self, synthetic_independent_walks: tuple[pd.Series, pd.Series]
    ) -> None:
        a, b = synthetic_independent_walks
        result = calibrate(a, b, method="static", significance="5%")
        fit = result.parameters["pair_fit"]
        assert fit.status == "rejected"
        assert fit.rejection_reason is not None

    def test_half_life_too_long_rejected(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        # Set the band so the synthetic pair (half_life ~ 4.3) is too low.
        a, b, _ = synthetic_cointegrated_pair
        result = calibrate(a, b, half_life_min=50.0, half_life_max=100.0)
        fit = result.parameters["pair_fit"]
        assert fit.status == "rejected"
        assert "half-life" in (fit.rejection_reason or "")

    def test_invalid_method_raises(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, _ = synthetic_cointegrated_pair
        with pytest.raises(ValueError, match="method"):
            calibrate(a, b, method="cubic")  # type: ignore[arg-type]

    def test_invalid_half_life_band_raises(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, _ = synthetic_cointegrated_pair
        with pytest.raises(ValueError, match="half-life band"):
            calibrate(a, b, half_life_min=10.0, half_life_max=5.0)
