"""Validation invariants on the HAR-RV dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.har_rv.types import HARFit, HARRVInputs, RVComponents


def _valid_components() -> RVComponents:
    idx = pd.date_range("2025-01-02", periods=10, freq="B")
    return RVComponents(rv_d=pd.Series(np.full(10, 1e-4), index=idx, name="rv_d"))


def _valid_fit(n_obs: int = 50) -> HARFit:
    idx = pd.date_range("2024-01-02", periods=n_obs, freq="B")
    return HARFit(
        coefficients=np.array([0.0, 0.5, 0.3, 0.2]),
        hac_covariance=np.eye(4) * 1e-4,
        residuals=np.zeros(n_obs),
        residual_variance=0.04,
        r_squared=0.7,
        n_obs=n_obs,
        hac_lag=3,
        spec="log",
        horizon=1,
        fit_index=pd.DatetimeIndex(idx),
    )


class TestRVComponents:
    def test_constructs_with_only_rv_d(self) -> None:
        rc = _valid_components()
        assert isinstance(rc.rv_d, pd.Series)
        assert rc.bipower is None
        assert rc.rv_minus is None
        assert rc.rv_plus is None
        assert rc.source == "intraday"

    def test_rejects_bad_source(self) -> None:
        idx = pd.date_range("2025-01-02", periods=5, freq="B")
        with pytest.raises(ValueError, match="source"):
            RVComponents(
                rv_d=pd.Series(np.ones(5), index=idx),
                source="bogus",  # type: ignore[arg-type]
            )

    def test_rejects_non_series_rv_d(self) -> None:
        with pytest.raises(TypeError, match="rv_d"):
            RVComponents(rv_d=np.ones(5))  # type: ignore[arg-type]

    def test_rejects_non_series_bipower(self) -> None:
        idx = pd.date_range("2025-01-02", periods=5, freq="B")
        with pytest.raises(TypeError, match="bipower"):
            RVComponents(
                rv_d=pd.Series(np.ones(5), index=idx),
                bipower=np.ones(5),  # type: ignore[arg-type]
            )


class TestHARFit:
    def test_constructs(self) -> None:
        fit = _valid_fit()
        assert fit.coefficients.shape == (4,)
        assert fit.hac_covariance.shape == (4, 4)
        assert fit.standard_errors.shape == (4,)

    def test_standard_errors_are_sqrt_of_diagonal(self) -> None:
        fit = _valid_fit()
        np.testing.assert_allclose(fit.standard_errors, np.full(4, 1e-2))

    def test_t_statistics_handle_zero_se(self) -> None:
        # Make the second diagonal zero; the corresponding t-stat must be NaN.
        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        cov = np.eye(4) * 1e-4
        cov[1, 1] = 0.0
        fit = HARFit(
            coefficients=np.array([0.0, 0.5, 0.3, 0.2]),
            hac_covariance=cov,
            residuals=np.zeros(10),
            residual_variance=0.04,
            r_squared=0.7,
            n_obs=10,
            hac_lag=2,
            spec="log",
            horizon=1,
            fit_index=pd.DatetimeIndex(idx),
        )
        t_stats = fit.t_statistics()
        assert np.isnan(t_stats[1])
        assert np.isfinite(t_stats[0])

    def test_rejects_bad_coef_shape(self) -> None:
        idx = pd.date_range("2024-01-02", periods=5, freq="B")
        with pytest.raises(ValueError, match="length-4"):
            HARFit(
                coefficients=np.zeros(3),
                hac_covariance=np.eye(4),
                residuals=np.zeros(5),
                residual_variance=0.0,
                r_squared=0.0,
                n_obs=5,
                hac_lag=1,
                spec="log",
                horizon=1,
                fit_index=pd.DatetimeIndex(idx),
            )

    def test_rejects_bad_horizon(self) -> None:
        idx = pd.date_range("2024-01-02", periods=5, freq="B")
        with pytest.raises(ValueError, match="horizon"):
            HARFit(
                coefficients=np.zeros(4),
                hac_covariance=np.eye(4),
                residuals=np.zeros(5),
                residual_variance=0.0,
                r_squared=0.0,
                n_obs=5,
                hac_lag=1,
                spec="log",
                horizon=0,
                fit_index=pd.DatetimeIndex(idx),
            )

    def test_rejects_negative_residual_variance(self) -> None:
        idx = pd.date_range("2024-01-02", periods=5, freq="B")
        with pytest.raises(ValueError, match="residual_variance"):
            HARFit(
                coefficients=np.zeros(4),
                hac_covariance=np.eye(4),
                residuals=np.zeros(5),
                residual_variance=-1.0,
                r_squared=0.0,
                n_obs=5,
                hac_lag=1,
                spec="log",
                horizon=1,
                fit_index=pd.DatetimeIndex(idx),
            )


class TestHARRVInputs:
    def test_constructs(self) -> None:
        inputs = HARRVInputs(
            ticker="SPY",
            rv_components=_valid_components(),
            spec="log",
            horizon=1,
            timestamp=datetime.now(UTC),
        )
        assert inputs.ticker == "SPY"
        assert inputs.trading_days_per_year == 252

    def test_rejects_bad_spec(self) -> None:
        with pytest.raises(ValueError, match="spec"):
            HARRVInputs(
                ticker="SPY",
                rv_components=_valid_components(),
                spec="weird",  # type: ignore[arg-type]
                horizon=1,
                timestamp=datetime.now(UTC),
            )

    def test_rejects_bad_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            HARRVInputs(
                ticker="SPY",
                rv_components=_valid_components(),
                spec="log",
                horizon=0,
                timestamp=datetime.now(UTC),
            )
