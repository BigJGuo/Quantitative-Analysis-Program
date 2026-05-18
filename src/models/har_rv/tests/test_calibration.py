"""Unit tests for the public `calibrate()` entry point."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.har_rv.calibration import MODEL_NAME, calibrate
from src.models.har_rv.types import HARFit


class TestCalibrate:
    def test_returns_calibration_result(self, persistent_log_rv: pd.Series) -> None:
        result = calibrate(rv_d=persistent_log_rv)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME

    def test_parameters_payload_includes_har_fit(
        self, persistent_log_rv: pd.Series
    ) -> None:
        result = calibrate(rv_d=persistent_log_rv)
        assert "har_fit" in result.parameters
        assert isinstance(result.parameters["har_fit"], HARFit)
        for key in ("c", "beta_d", "beta_w", "beta_m"):
            assert key in result.parameters
            assert isinstance(result.parameters[key], float)
        for key in ("se_c", "se_beta_d", "se_beta_w", "se_beta_m"):
            assert key in result.parameters

    def test_fit_metrics_have_required_keys(self, persistent_log_rv: pd.Series) -> None:
        result = calibrate(rv_d=persistent_log_rv)
        for key in (
            "r_squared",
            "residual_variance",
            "n_obs",
            "hac_lag",
            "persistence_sum",
        ):
            assert key in result.fit_metrics
            assert isinstance(result.fit_metrics[key], float)

    def test_metadata_includes_spec_and_horizon(
        self, persistent_log_rv: pd.Series
    ) -> None:
        result = calibrate(rv_d=persistent_log_rv, spec="level", horizon=2)
        assert result.metadata["spec"] == "level"
        assert result.metadata["horizon"] == 2

    def test_timestamp_honored(self, persistent_log_rv: pd.Series) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        result = calibrate(rv_d=persistent_log_rv, timestamp=ts)
        assert result.timestamp == ts

    def test_rejects_bad_spec(self, persistent_log_rv: pd.Series) -> None:
        with pytest.raises(ValueError, match="spec"):
            calibrate(rv_d=persistent_log_rv, spec="quadratic")  # type: ignore[arg-type]

    def test_rejects_empty_rv(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            calibrate(rv_d=pd.Series(dtype="float64"))

    def test_rejects_zero_horizon(self, persistent_log_rv: pd.Series) -> None:
        with pytest.raises(ValueError, match="horizon"):
            calibrate(rv_d=persistent_log_rv, horizon=0)

    def test_log_persistence_sum_is_close_to_one_for_high_phi(
        self, persistent_log_rv: pd.Series
    ) -> None:
        result = calibrate(rv_d=persistent_log_rv)
        beta_sum = result.fit_metrics["persistence_sum"]
        # Strong-persistence DGP => sum of three lag coefficients near 1.
        assert 0.7 < beta_sum < 1.05

    def test_se_matches_fit_object(self, persistent_log_rv: pd.Series) -> None:
        result = calibrate(rv_d=persistent_log_rv)
        fit = result.parameters["har_fit"]
        assert isinstance(fit, HARFit)
        np.testing.assert_allclose(
            [
                result.parameters["se_c"],
                result.parameters["se_beta_d"],
                result.parameters["se_beta_w"],
                result.parameters["se_beta_m"],
            ],
            fit.standard_errors,
            atol=1e-12,
        )
