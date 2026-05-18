"""Unit tests for `src.models.var.calibration`."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.var.calibration import MODEL_NAME, calibrate
from src.models.var.types import VaRFit


@pytest.fixture
def returns_two_asset() -> pd.DataFrame:
    rng = np.random.default_rng(seed=100)
    n = 1000
    r1 = rng.standard_normal(n) * 0.01
    r2 = rng.standard_normal(n) * 0.02
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.DataFrame({"A": r1, "B": r2}, index=idx)


@pytest.fixture
def positions_two_asset() -> dict[str, float]:
    return {"A": 1_000_000.0, "B": 500_000.0}


class TestCalibrateBasic:
    def test_returns_calibration_result(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=returns_two_asset,
            positions=positions_two_asset,
            method="historical",
            alpha=0.99,
            horizon_days=10,
        )
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        assert "fit" in result.parameters
        fit = result.parameters["fit"]
        assert isinstance(fit, VaRFit)
        assert fit.method == "historical"
        assert fit.alpha == 0.99
        assert fit.horizon_days == 10
        assert fit.n_assets == 2

    def test_metrics_populated(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=returns_two_asset,
            positions=positions_two_asset,
            method="historical",
            alpha=0.99,
            horizon_days=10,
        )
        for k in (
            "var_1d_dollar",
            "var_horizon_dollar",
            "horizon_days",
            "alpha",
            "n_obs",
            "n_assets",
            "portfolio_value",
            "gross_exposure",
            "pnl_mean",
            "pnl_std",
            "var_1d_pct_gross",
        ):
            assert k in result.fit_metrics
            assert isinstance(result.fit_metrics[k], float)


class TestCalibrateMethods:
    def test_historical_matches_signal(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=returns_two_asset,
            positions=positions_two_asset,
            method="historical",
            alpha=0.99,
            horizon_days=1,
        )
        fit = result.parameters["fit"]
        assert isinstance(fit, VaRFit)
        # HS VaR should equal np.quantile of -PnL.
        pnl = returns_two_asset.to_numpy() @ np.array([1_000_000.0, 500_000.0])
        expected = float(np.quantile(-pnl, 0.99))
        assert fit.var_1d_dollar == pytest.approx(expected, abs=1e-6)
        assert fit.var_horizon_dollar == pytest.approx(fit.var_1d_dollar)

    def test_parametric_records_sigma_matrix(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=returns_two_asset,
            positions=positions_two_asset,
            method="parametric",
            alpha=0.99,
            horizon_days=10,
        )
        fit = result.parameters["fit"]
        assert isinstance(fit, VaRFit)
        assert fit.sigma_matrix is not None
        assert fit.sigma_matrix.shape == (2, 2)
        assert fit.sigma_p_dollar is not None
        assert fit.sigma_p_dollar > 0

    def test_monte_carlo_records_sample_count(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=returns_two_asset,
            positions=positions_two_asset,
            method="monte_carlo",
            alpha=0.99,
            horizon_days=10,
            mc_samples=10_000,
            mc_seed=7,
        )
        fit = result.parameters["fit"]
        assert isinstance(fit, VaRFit)
        assert fit.mc_samples == 10_000
        assert fit.pnl_vector.shape == (10_000,)


class TestCalibrateValidation:
    def test_rejects_mismatched_columns(
        self,
        returns_two_asset: pd.DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match="must match"):
            calibrate(
                returns=returns_two_asset,
                positions={"X": 1.0, "Y": 2.0},
                method="historical",
            )

    def test_rejects_short_history(self) -> None:
        small = pd.DataFrame(
            {"A": np.zeros(10), "B": np.zeros(10)},
            index=pd.date_range("2020-01-01", periods=10, freq="B"),
        )
        with pytest.raises(ValueError, match=">= 30"):
            calibrate(
                returns=small,
                positions={"A": 1.0, "B": 1.0},
                method="historical",
            )

    def test_rejects_invalid_alpha(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        with pytest.raises(ValueError, match="alpha"):
            calibrate(
                returns=returns_two_asset,
                positions=positions_two_asset,
                alpha=1.5,
            )

    def test_rejects_unknown_method(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        with pytest.raises(ValueError, match="method"):
            calibrate(
                returns=returns_two_asset,
                positions=positions_two_asset,
                method="bogus",  # type: ignore[arg-type]
            )

    def test_uses_supplied_timestamp(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        ts = datetime(2026, 5, 18, 14, 30, tzinfo=UTC)
        result = calibrate(
            returns=returns_two_asset,
            positions=positions_two_asset,
            method="historical",
            timestamp=ts,
        )
        assert result.timestamp == ts


class TestCalibrateHorizonScaling:
    def test_ten_day_is_sqrt_ten_times_one_day(
        self,
        returns_two_asset: pd.DataFrame,
        positions_two_asset: dict[str, float],
    ) -> None:
        result = calibrate(
            returns=returns_two_asset,
            positions=positions_two_asset,
            method="historical",
            alpha=0.99,
            horizon_days=10,
        )
        fit = result.parameters["fit"]
        assert isinstance(fit, VaRFit)
        ratio = fit.var_horizon_dollar / fit.var_1d_dollar
        assert ratio == pytest.approx(np.sqrt(10), rel=1e-9)
