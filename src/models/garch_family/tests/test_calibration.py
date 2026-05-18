"""Unit tests for GARCH-family MLE calibration."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.garch_family.calibration import MODEL_NAME, calibrate
from src.models.garch_family.types import GARCHFit


class TestCalibrateContract:
    def test_returns_calibration_result_with_expected_shape(
        self, synthetic_garch_returns: pd.Series
    ) -> None:
        result = calibrate(returns=synthetic_garch_returns, spec="GARCH", distribution="Gaussian")
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        # parameters bundle
        assert isinstance(result.parameters["fit"], GARCHFit)
        assert result.parameters["spec"] == "GARCH"
        assert result.parameters["distribution"] == "Gaussian"
        params_dict = result.parameters["params"]
        for key in ("omega", "alpha", "beta", "gamma", "nu", "mu"):
            assert key in params_dict
        # fit_metrics has the keys the spec validation expects
        for key in (
            "log_likelihood",
            "aic",
            "bic",
            "persistence",
            "n_obs",
            "n_params",
            "mean_sigma_pct",
            "final_sigma_pct",
            "converged",
        ):
            assert key in result.fit_metrics

    def test_rejects_unknown_spec(self, synthetic_garch_returns: pd.Series) -> None:
        with pytest.raises(ValueError, match="spec"):
            calibrate(returns=synthetic_garch_returns, spec="ARCH")  # type: ignore[arg-type]

    def test_rejects_unknown_distribution(self, synthetic_garch_returns: pd.Series) -> None:
        with pytest.raises(ValueError, match="distribution"):
            calibrate(returns=synthetic_garch_returns, distribution="GED")  # type: ignore[arg-type]

    def test_rejects_short_series(self) -> None:
        r = pd.Series(np.random.default_rng(0).standard_normal(10))
        with pytest.raises(ValueError, match=">= 30"):
            calibrate(returns=r, spec="GARCH")


class TestParameterRecovery:
    """Synthetic-data recovery tests. Tolerances loose because MLE is finite-sample noisy."""

    def test_garch11_recovers_persistence(
        self,
        synthetic_garch_returns: pd.Series,
        true_garch_params: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=synthetic_garch_returns,
            spec="GARCH",
            distribution="Gaussian",
            mean_model="Zero",
        )
        p = result.parameters["params"]
        # true persistence = 0.98
        true_persistence = true_garch_params["alpha"] + true_garch_params["beta"]
        recovered = p["alpha"] + p["beta"]
        assert recovered == pytest.approx(true_persistence, abs=0.03)
        # Stationarity holds.
        assert recovered < 1.0
        # Omega in the right ballpark (broad — sample noise is high).
        assert 0.01 < p["omega"] < 0.2

    def test_gjr_recovers_positive_leverage(
        self,
        synthetic_gjr_returns: pd.Series,
        true_gjr_params: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=synthetic_gjr_returns,
            spec="GJR",
            distribution="Gaussian",
            mean_model="Zero",
        )
        p = result.parameters["params"]
        # gamma must come out positive (leverage effect).
        assert p["gamma"] > 0.0
        # Persistence reasonable (true alpha + gamma/2 + beta = 0.05 + 0.04 + 0.88 = 0.97).
        persistence = p["alpha"] + 0.5 * p["gamma"] + p["beta"]
        assert persistence == pytest.approx(0.97, abs=0.05)

    def test_student_t_recovers_fat_tail_nu(self) -> None:
        """Generate fat-tailed GARCH and check the MLE pulls nu below 10."""

        rng = np.random.default_rng(seed=20260520)
        omega, alpha, beta = 0.05, 0.1, 0.85
        nu_true = 5.0
        t = 2500
        eps = np.empty(t)
        s2 = np.empty(t)
        s2[0] = omega / (1.0 - alpha - beta)
        for i in range(1, t):
            s2[i] = omega + alpha * eps[i - 1] ** 2 + beta * s2[i - 1]
            raw_t = rng.standard_t(nu_true)
            z = raw_t / math.sqrt(nu_true / (nu_true - 2.0))
            eps[i] = float(math.sqrt(s2[i]) * z)
        result = calibrate(
            returns=pd.Series(eps),
            spec="GARCH",
            distribution="Student-t",
            mean_model="Zero",
            max_iter=4000,
        )
        nu_hat = result.parameters["params"]["nu"]
        assert nu_hat is not None
        assert 3.0 < nu_hat < 10.0

    def test_aic_prefers_student_t_on_fat_tailed_data(self) -> None:
        rng = np.random.default_rng(seed=20260521)
        omega, alpha, beta = 0.05, 0.1, 0.85
        nu_true = 5.0
        t = 2500
        eps = np.empty(t)
        s2 = np.empty(t)
        s2[0] = omega / (1.0 - alpha - beta)
        for i in range(1, t):
            s2[i] = omega + alpha * eps[i - 1] ** 2 + beta * s2[i - 1]
            raw = rng.standard_t(nu_true)
            z = raw / math.sqrt(nu_true / (nu_true - 2.0))
            eps[i] = float(math.sqrt(s2[i]) * z)
        r = pd.Series(eps)
        res_g = calibrate(returns=r, spec="GARCH", distribution="Gaussian", mean_model="Zero")
        res_t = calibrate(returns=r, spec="GARCH", distribution="Student-t", mean_model="Zero")
        assert res_t.fit_metrics["aic"] < res_g.fit_metrics["aic"]


class TestSigma2PathAttachment:
    def test_fit_carries_sigma2_path_same_length_as_returns(
        self, synthetic_garch_returns: pd.Series
    ) -> None:
        result = calibrate(
            returns=synthetic_garch_returns, spec="GARCH", distribution="Gaussian"
        )
        fit = result.parameters["fit"]
        assert isinstance(fit, GARCHFit)
        # We drop non-finite entries in `calibrate`; synthetic data is all finite,
        # so the fit length matches the input length exactly.
        assert fit.sigma2_path.shape == (len(synthetic_garch_returns),)
        assert fit.eps_path.shape == (len(synthetic_garch_returns),)
        assert fit.std_resid.shape == (len(synthetic_garch_returns),)
        assert np.all(fit.sigma2_path > 0)


class TestNelderMeadConvergence:
    @pytest.mark.parametrize("spec", ["GARCH", "GJR", "EGARCH"])
    def test_converges_on_synthetic_data(
        self, synthetic_garch_returns: pd.Series, spec: str
    ) -> None:
        result = calibrate(
            returns=synthetic_garch_returns,
            spec=spec,  # type: ignore[arg-type]
            distribution="Gaussian",
            mean_model="Zero",
            max_iter=3000,
        )
        assert result.fit_metrics["converged"] == 1.0
        assert result.fit_metrics["n_iter"] < 3000
