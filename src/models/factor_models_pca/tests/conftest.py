"""Shared synthetic fixtures for the factor-model unit tests.

Synthetic returns are generated from a known low-rank factor structure so
worked-out checks (variance explained, recovered loadings up to sign, near-
zero residual cross-correlation) become exact rather than approximate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


@pytest.fixture
def synthetic_one_factor(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """One latent factor (the market) + tiny idiosyncratic noise.

    Returns `(panel, true_loadings)`. The true B is `(N, 1)` and is what PCA
    should recover up to a column sign.
    """

    n_obs = 500
    n_assets = 8
    true_B = rng.uniform(0.6, 1.4, size=(n_assets, 1))
    factor_vol = 0.012
    spec_vol = 0.003  # tiny on purpose so eigenvalues separate cleanly
    f = rng.normal(0.0, factor_vol, size=(n_obs, 1))
    eps = rng.normal(0.0, spec_vol, size=(n_obs, n_assets))
    R = f @ true_B.T + eps
    cols = [f"S{i}" for i in range(n_assets)]
    index = pd.date_range("2024-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(R, index=index, columns=cols), true_B


@pytest.fixture
def synthetic_three_factor(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Three orthogonal latent factors of decaying magnitude + noise."""

    n_obs = 600
    n_assets = 12
    true_B = rng.normal(0.0, 1.0, size=(n_assets, 3))
    factor_vols = np.array([0.02, 0.012, 0.006])
    spec_vol = 0.004
    f = rng.normal(0.0, 1.0, size=(n_obs, 3)) * factor_vols
    eps = rng.normal(0.0, spec_vol, size=(n_obs, n_assets))
    R = f @ true_B.T + eps
    cols = [f"S{i}" for i in range(n_assets)]
    index = pd.date_range("2024-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(R, index=index, columns=cols), true_B


@pytest.fixture
def fama_french_synthetic(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Returns built from known factor returns + known betas.

    Returns `(asset_returns, factor_returns, true_betas)` so the
    Fama-French time-series-regression path can be checked exactly.
    """

    n_obs = 400
    n_assets = 6
    factor_names = ["MKT", "SMB", "HML"]
    factor_returns = pd.DataFrame(
        rng.normal(0.0, [0.01, 0.005, 0.004], size=(n_obs, 3)),
        index=pd.date_range("2024-01-01", periods=n_obs, freq="B"),
        columns=factor_names,
    )
    true_betas = rng.uniform(-1.0, 1.5, size=(n_assets, 3))
    spec_vol = 0.002
    eps = rng.normal(0.0, spec_vol, size=(n_obs, n_assets))
    R = factor_returns.to_numpy() @ true_betas.T + eps
    asset_returns = pd.DataFrame(
        R,
        index=factor_returns.index,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    return asset_returns, factor_returns, true_betas
