"""Shared synthetic fixtures for the cointegration / pair-trading unit tests.

The fixtures build price series with a *known* cointegrating relationship so
worked-out checks (recovered beta, AR(1) phi, half-life) reduce to "does the
estimator hit the true value within noise?".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


@pytest.fixture
def synthetic_cointegrated_pair(
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """A simulated cointegrated pair with known parameters.

    Construction:
        p_B is a random walk (I(1)).
        Z is an OU/AR(1) process with phi = 0.85.
        p_A = alpha + beta * p_B + Z is therefore I(1) and cointegrates with
        p_B at vector (1, -beta).

    Returns
    -------
    (p_a, p_b, truth) where `truth` maps the names ``alpha``, ``beta``,
    ``phi``, ``sigma_eps``, ``half_life`` to their generating values.
    """

    n_obs = 800
    alpha_true = 0.30
    beta_true = 1.20
    phi_true = 0.85
    sigma_walk = 0.012
    sigma_z = 0.02

    eps_b = rng.normal(0.0, sigma_walk, size=n_obs)
    p_b_arr = np.cumsum(eps_b) + np.log(50.0)

    z = np.empty(n_obs)
    z[0] = 0.0
    eps_z = rng.normal(0.0, sigma_z, size=n_obs)
    for t in range(1, n_obs):
        z[t] = phi_true * z[t - 1] + eps_z[t]

    p_a_arr = alpha_true + beta_true * p_b_arr + z
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    p_a = pd.Series(p_a_arr, index=idx, name="A")
    p_b = pd.Series(p_b_arr, index=idx, name="B")
    truth = {
        "alpha": alpha_true,
        "beta": beta_true,
        "phi": phi_true,
        "sigma_eps": sigma_z,
        "half_life": -math.log(2.0) / math.log(phi_true),
    }
    return p_a, p_b, truth


@pytest.fixture
def synthetic_independent_walks(
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """Two independent random walks; should NOT cointegrate."""

    n_obs = 600
    a = np.cumsum(rng.normal(0.0, 0.01, size=n_obs)) + np.log(50.0)
    b = np.cumsum(rng.normal(0.0, 0.01, size=n_obs)) + np.log(75.0)
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


@pytest.fixture
def synthetic_ar1_path(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Long AR(1) sample so phi / sigma can be checked against ground truth."""

    n_obs = 5000
    phi_true = 0.9
    c_true = 0.05
    sigma_true = 0.1
    z = np.empty(n_obs)
    z[0] = c_true / (1.0 - phi_true)
    eps = rng.normal(0.0, sigma_true, size=n_obs)
    for t in range(1, n_obs):
        z[t] = c_true + phi_true * z[t - 1] + eps[t]
    return z, {
        "phi": phi_true,
        "c": c_true,
        "sigma_eps": sigma_true,
        "mu": c_true / (1.0 - phi_true),
    }


@pytest.fixture
def synthetic_kalman_drift(
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """A pair where beta drifts linearly through time.

    Used to check that the Kalman filter actually tracks a moving hedge ratio,
    which the static-OLS path cannot.
    """

    n_obs = 600
    sigma_walk = 0.01
    sigma_eps = 0.003
    eps_b = rng.normal(0.0, sigma_walk, size=n_obs)
    p_b_arr = np.cumsum(eps_b) + np.log(80.0)
    beta_path = np.linspace(1.0, 1.5, n_obs)
    eps = rng.normal(0.0, sigma_eps, size=n_obs)
    p_a_arr = beta_path * p_b_arr + eps
    idx = pd.date_range("2022-01-03", periods=n_obs, freq="B")
    return (
        pd.Series(p_a_arr, index=idx, name="A"),
        pd.Series(p_b_arr, index=idx, name="B"),
        beta_path,
    )
