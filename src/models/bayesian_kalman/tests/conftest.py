"""Shared synthetic fixtures for the bayesian_kalman unit tests.

Synthetic data is built from known generative parameters so the recovery
tests (MLE picks the right `(q, R)`, Gibbs shrinks toward the right mu,
DFM recovers PCA-style loadings up to a sign) become exact-ish checks rather
than vibe checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


@pytest.fixture
def constant_beta_series(rng: np.random.Generator) -> tuple[pd.Series, pd.Series, float]:
    """`r_asset = alpha + beta * r_market + eps` with constant beta = 1.2.

    The Kalman filter with appropriate `q_beta` should track beta close to
    1.2 once enough observations land.
    """

    T = 500
    market = pd.Series(rng.normal(0.0, 0.01, size=T), name="MKT")
    alpha = 0.0002
    beta = 1.2
    eps = rng.normal(0.0, 0.005, size=T)
    asset = pd.Series(alpha + beta * market.to_numpy() + eps, name="ASSET")
    return asset, market, beta


@pytest.fixture
def drifting_beta_series(rng: np.random.Generator) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Beta drifts linearly from 0.8 to 1.6 across the sample."""

    T = 600
    market = pd.Series(rng.normal(0.0, 0.01, size=T), name="MKT")
    beta_path = np.linspace(0.8, 1.6, T)
    alpha = 0.0
    eps = rng.normal(0.0, 0.003, size=T)
    asset = pd.Series(alpha + beta_path * market.to_numpy() + eps, name="ASSET")
    return asset, market, beta_path


@pytest.fixture
def hierarchical_panel(
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """N assets each with their own `theta_i` drawn from a known population.

    Returns `(designs, targets, true_thetas)`. The Gibbs sampler should
    recover posterior means close to the true thetas, and the population mu
    should land near the true population mean.
    """

    N = 8
    p = 2
    T = 200
    true_mu = np.array([0.0005, 1.1])
    true_Sigma = np.diag([1e-6, 0.04])
    Sigma_chol = np.linalg.cholesky(true_Sigma)
    true_thetas = true_mu + rng.standard_normal(size=(N, p)) @ Sigma_chol.T
    designs: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for i in range(N):
        market = rng.normal(0.0, 0.01, size=T)
        X = np.column_stack([np.ones(T), market])
        eps = rng.normal(0.0, 0.004, size=T)
        y = X @ true_thetas[i] + eps
        designs[f"A{i}"] = X
        targets[f"A{i}"] = y
    return designs, targets, true_thetas


@pytest.fixture
def dfm_panel(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Two-factor latent-factor returns. Used to check DFM EM recovery."""

    T = 400
    N = 10
    r = 2
    Lambda = rng.normal(0.0, 0.5, size=(N, r))
    Phi = np.diag([0.7, 0.5])
    Q = np.diag([0.0004, 0.0002])
    Psi = np.full(N, 1e-5)
    f = np.zeros((T, r))
    eta = rng.standard_normal(size=(T, r)) @ np.linalg.cholesky(Q).T
    for t in range(1, T):
        f[t] = Phi @ f[t - 1] + eta[t]
    eps = rng.standard_normal(size=(T, N)) * np.sqrt(Psi)
    Y = f @ Lambda.T + eps
    panel = pd.DataFrame(
        Y,
        index=pd.date_range("2024-01-01", periods=T, freq="B"),
        columns=[f"S{i}" for i in range(N)],
    )
    return panel, Lambda, f


@pytest.fixture
def market_log_returns(rng: np.random.Generator) -> pd.Series:
    """Plain market return series used by integration-ish unit tests."""

    T = 300
    idx = pd.date_range("2024-01-01", periods=T, freq="B")
    return pd.Series(rng.normal(0.0003, 0.012, size=T), index=idx, name="SPY")
