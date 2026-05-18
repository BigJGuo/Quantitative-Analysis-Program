"""Synthetic fixtures for Avellaneda-Lee unit tests.

We generate returns from a *known* low-rank factor structure with mean-
reverting idiosyncratic components, so the AR(1) -> OU mapping and the
filter logic can be checked against ground truth rather than empirical
approximations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


@pytest.fixture
def ou_known_params() -> dict[str, float]:
    """Ground-truth OU parameters used by the AR(1) -> OU recovery test."""
    return {"kappa": 0.10, "m": 0.0, "sigma": 0.04, "T": 5000}


@pytest.fixture
def ou_series(rng: np.random.Generator, ou_known_params: dict[str, float]) -> np.ndarray:
    """Long simulated OU path with known parameters (dt=1, T=5000)."""

    kappa = ou_known_params["kappa"]
    m = ou_known_params["m"]
    sigma = ou_known_params["sigma"]
    T = int(ou_known_params["T"])
    dt = 1.0

    x = np.zeros(T, dtype=float)
    b = np.exp(-kappa * dt)
    # Exact-discretization SD of the AR(1) innovations.
    sigma_zeta = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    z = rng.normal(0.0, sigma_zeta, size=T)
    for t in range(1, T):
        x[t] = m * (1.0 - b) + b * x[t - 1] + z[t]
    return x


@pytest.fixture
def synthetic_panel(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-factor synthetic panel with mean-reverting residual prices.

    Returns ``(asset_returns, factor_returns)``. The "factor" is the equally-
    weighted average of high-beta assets, and each asset's idiosyncratic
    component is an OU residual return with a known half-life of ~7 days.
    Useful for end-to-end calibration checks without hitting yfinance.
    """

    n_obs = 600
    n_assets = 12
    factor_vol = 0.012
    f = rng.normal(0.0, factor_vol, size=n_obs)

    # Build OU residual returns for each asset. Discrete OU with kappa=0.1.
    kappa = 0.1
    b = np.exp(-kappa)
    eq_var = 0.02**2  # equilibrium variance of residual *price* X
    sigma_zeta = float(np.sqrt(eq_var * (1.0 - b**2)))
    eps_returns = np.zeros((n_obs, n_assets), dtype=float)
    X_prev = rng.normal(0.0, np.sqrt(eq_var), size=n_assets)
    for t in range(n_obs):
        z = rng.normal(0.0, sigma_zeta, size=n_assets)
        X_now = b * X_prev + z
        eps_returns[t] = X_now - X_prev
        X_prev = X_now

    betas = rng.uniform(0.7, 1.3, size=n_assets)
    R = betas[None, :] * f[:, None] + eps_returns
    idx = pd.date_range("2024-01-01", periods=n_obs, freq="B")
    asset_returns = pd.DataFrame(R, index=idx, columns=[f"S{i}" for i in range(n_assets)])
    factor_returns = pd.DataFrame({"MKT": f}, index=idx)
    return asset_returns, factor_returns
