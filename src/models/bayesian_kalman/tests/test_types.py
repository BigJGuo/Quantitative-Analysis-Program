"""Validation checks for the model-internal dataclasses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.bayesian_kalman.types import (
    BayesianKalmanInputs,
    DynamicFactorFit,
    HierarchicalFit,
    KalmanFit,
)


def _kalman_fit(n: int = 2, T: int = 50) -> KalmanFit:
    return KalmanFit(
        F=np.eye(n),
        Q=np.eye(n) * 1e-6,
        R=1e-4,
        x0=np.zeros(n),
        P0=np.eye(n) * 0.1,
        states=np.zeros((T, n)),
        covariances=np.tile(np.eye(n) * 0.05, (T, 1, 1)),
        innovations=np.zeros(T),
        innovation_variances=np.ones(T) * 1e-4,
        log_likelihood=-100.0,
        index=pd.RangeIndex(T),
        state_names=tuple(f"x{i}" for i in range(n)),
    )


def test_kalman_fit_validates_shape() -> None:
    with pytest.raises(ValueError, match="F must be square"):
        KalmanFit(
            F=np.zeros((2, 3)),
            Q=np.eye(2),
            R=1e-4,
            x0=np.zeros(2),
            P0=np.eye(2),
            states=np.zeros((10, 2)),
            covariances=np.zeros((10, 2, 2)),
            innovations=np.zeros(10),
            innovation_variances=np.ones(10),
            log_likelihood=0.0,
            index=pd.RangeIndex(10),
        )


def test_kalman_fit_negative_R_rejected() -> None:
    with pytest.raises(ValueError, match="R must be non-negative"):
        KalmanFit(
            F=np.eye(2),
            Q=np.eye(2),
            R=-0.1,
            x0=np.zeros(2),
            P0=np.eye(2),
            states=np.zeros((10, 2)),
            covariances=np.zeros((10, 2, 2)),
            innovations=np.zeros(10),
            innovation_variances=np.ones(10),
            log_likelihood=0.0,
            index=pd.RangeIndex(10),
        )


def test_kalman_fit_latest_state_and_cov() -> None:
    fit = _kalman_fit()
    assert fit.latest_state().shape == (2,)
    assert fit.latest_covariance().shape == (2, 2)


def test_hierarchical_fit_validates() -> None:
    N, p = 4, 2
    fit = HierarchicalFit(
        mu_mean=np.zeros(p),
        mu_cov=np.eye(p),
        Sigma_mean=np.eye(p),
        theta_means=np.zeros((N, p)),
        theta_covs=np.tile(np.eye(p), (N, 1, 1)),
        sigma_sq_means=np.ones(N),
        tickers=tuple(f"A{i}" for i in range(N)),
        coef_names=("c", "beta"),
        n_iter=100,
        n_burn=50,
    )
    assert fit.n_assets == N
    assert fit.n_coefs == p


def test_hierarchical_fit_burn_in_invariant() -> None:
    with pytest.raises(ValueError, match="n_burn"):
        HierarchicalFit(
            mu_mean=np.zeros(2),
            mu_cov=np.eye(2),
            Sigma_mean=np.eye(2),
            theta_means=np.zeros((2, 2)),
            theta_covs=np.tile(np.eye(2), (2, 1, 1)),
            sigma_sq_means=np.ones(2),
            tickers=("A", "B"),
            coef_names=("c", "b"),
            n_iter=10,
            n_burn=10,
        )


def test_dynamic_factor_fit_validates() -> None:
    N, r = 5, 2
    fit = DynamicFactorFit(
        Lambda=np.zeros((N, r)),
        Phi=np.eye(r) * 0.5,
        Q=np.eye(r),
        Psi=np.ones(N),
        factors=np.zeros((20, r)),
        tickers=tuple(f"S{i}" for i in range(N)),
        factor_names=("F1", "F2"),
        log_likelihood=0.0,
        n_iter=5,
        index=pd.RangeIndex(20),
    )
    assert fit.n_assets == N
    assert fit.n_factors == r


def test_bayesian_kalman_inputs_mode_dispatch() -> None:
    with pytest.raises(ValueError, match="kalman_beta"):
        BayesianKalmanInputs(
            mode="kalman_beta",
            timestamp=pd.Timestamp("2024-01-01").to_pydatetime(),
        )
    with pytest.raises(ValueError, match="hierarchical"):
        BayesianKalmanInputs(
            mode="hierarchical",
            timestamp=pd.Timestamp("2024-01-01").to_pydatetime(),
        )
    with pytest.raises(ValueError, match="dynamic_factor"):
        BayesianKalmanInputs(
            mode="dynamic_factor",
            timestamp=pd.Timestamp("2024-01-01").to_pydatetime(),
        )


def test_bayesian_kalman_inputs_accepts_valid_kalman_beta() -> None:
    s = pd.Series([0.0, 0.01, -0.005])
    inputs = BayesianKalmanInputs(
        mode="kalman_beta",
        timestamp=pd.Timestamp("2024-01-01").to_pydatetime(),
        asset_returns=s,
        market_returns=s,
    )
    assert inputs.mode == "kalman_beta"
