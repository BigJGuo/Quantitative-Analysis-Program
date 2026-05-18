"""Shared synthetic fixtures for the anomaly-detection unit tests.

We generate small panels with a known clean Gaussian core so the detectors'
in-sample false-positive rate matches the configured contamination level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


@pytest.fixture
def synthetic_yf_bars(rng: np.random.Generator) -> pd.DataFrame:
    """yfinance-shaped OHLCV frame from a 1-factor log-return process."""

    n = 400
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    log_rets = rng.normal(0.0, 0.012, size=n)
    log_prices = np.log(100.0) + log_rets.cumsum()
    prices = np.exp(log_prices)
    volume = rng.integers(500_000, 5_000_000, size=n)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1.0 + rng.normal(0.0, 0.001, size=n)),
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Adj Close": prices,
            "Volume": volume,
        }
    )


@pytest.fixture
def gaussian_panel(rng: np.random.Generator) -> np.ndarray:
    """A (500, 6) standard Gaussian panel — clean, no anomalies."""

    return rng.normal(0.0, 1.0, size=(500, 6))


@pytest.fixture
def panel_with_outliers(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian panel + 5 obvious outliers; returns (X, outlier_indices)."""

    n_clean = 200
    p = 5
    X_clean = rng.normal(0.0, 1.0, size=(n_clean, p))
    # Place outliers far outside the Gaussian ball.
    outliers = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -12.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 11.0, 0.0, 0.0],
            [8.0, 8.0, 8.0, 0.0, 0.0],
            [-9.0, -9.0, 0.0, 0.0, 9.0],
        ]
    )
    X = np.vstack([X_clean, outliers])
    return X, np.arange(n_clean, n_clean + outliers.shape[0])
