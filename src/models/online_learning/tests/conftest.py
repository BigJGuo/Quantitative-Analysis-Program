"""Shared fixtures for the Online Learning unit tests.

All fixtures are deterministic — fixed seeds throughout — so the math
checks (Hedge regret bound, FTRL closed-form recovery, weight collapse onto
a dominant expert) are reproducible without depending on yfinance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.models.online_learning.types import ExpertStream


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_prices(rng: np.random.Generator) -> pd.Series:
    """500 days of synthetic geometric-Brownian-motion close prices."""

    n = 500
    mu = 0.0001
    sigma = 0.01
    log_returns = rng.normal(mu, sigma, size=n)
    prices = 100.0 * np.exp(log_returns.cumsum())
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def synthetic_close_frame(synthetic_prices: pd.Series) -> pd.DataFrame:
    """yfinance-style daily OHLC frame with 'Date' and 'Close' columns."""

    frame = pd.DataFrame(
        {
            "Open": synthetic_prices.shift(1).bfill().to_numpy(dtype=float),
            "High": synthetic_prices.to_numpy(dtype=float) * 1.001,
            "Low": synthetic_prices.to_numpy(dtype=float) * 0.999,
            "Close": synthetic_prices.to_numpy(dtype=float),
            "Volume": np.full(synthetic_prices.shape[0], 1_000_000.0),
        }
    )
    frame.insert(0, "Date", synthetic_prices.index)
    return frame


@pytest.fixture
def in_memory_provider(synthetic_close_frame: pd.DataFrame) -> InMemoryProvider:
    return InMemoryProvider(
        prices={("FAKE", "5y", "1d"): synthetic_close_frame},
    )


@pytest.fixture
def dominant_expert_stream() -> ExpertStream:
    """Two-expert panel where expert 0 is always perfect, expert 1 always off by 1.

    Used to verify Hedge collapses weight onto the dominant expert.
    """

    n = 200
    rng = np.random.default_rng(0)
    target = rng.uniform(0.0, 1.0, size=n)
    preds = pd.DataFrame(
        {"good": target, "bad": target + 1.0},
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )
    tgt = pd.Series(target, index=preds.index, name="target")
    return ExpertStream(predictions=preds, targets=tgt)


@pytest.fixture
def linear_regression_stream() -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    np.ndarray,
]:
    """Sparse stream where y_t = sum_j theta*_j * x_{t,j} + noise.

    Used to verify FTRL-Proximal recovers a sparse weight vector close to
    the data-generating process.
    """

    rng = np.random.default_rng(7)
    dim = 20
    n = 800
    theta_star = np.zeros(dim, dtype=float)
    theta_star[[2, 5, 11]] = (1.5, -2.0, 0.7)

    features: list[tuple[np.ndarray, np.ndarray]] = []
    targets = np.empty(n, dtype=float)
    for t in range(n):
        # Each example activates 5 random coordinates.
        idx = rng.choice(dim, size=5, replace=False)
        vals = rng.normal(size=5)
        y = float(np.dot(theta_star[idx], vals) + 0.1 * rng.normal())
        features.append((idx.astype(np.int64), vals))
        targets[t] = y
    return features, targets


@pytest.fixture
def logistic_regression_stream() -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    np.ndarray,
]:
    """Sparse binary-classification stream with logistic data-generating process."""

    rng = np.random.default_rng(11)
    dim = 30
    n = 1500
    theta_star = np.zeros(dim, dtype=float)
    theta_star[[1, 7, 19]] = (2.0, -1.5, 1.0)

    features: list[tuple[np.ndarray, np.ndarray]] = []
    targets = np.empty(n, dtype=float)
    for t in range(n):
        idx = rng.choice(dim, size=6, replace=False)
        vals = rng.normal(size=6)
        logit = float(np.dot(theta_star[idx], vals))
        p = 1.0 / (1.0 + np.exp(-logit))
        y = float(rng.binomial(1, p))
        features.append((idx.astype(np.int64), vals))
        targets[t] = y
    return features, targets
