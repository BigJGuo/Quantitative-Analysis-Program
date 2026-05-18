"""Shared synthetic fixtures for the SAE+MLP unit tests.

We generate small panels with a known latent factor structure so the
ensemble can hit non-trivial AUC > 0.5 even with the tiny epoch count used
to keep tests fast.
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
    """Build a yfinance-shaped OHLCV frame from a 1-factor log-return process."""

    n = 400
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    log_rets = rng.normal(0.0, 0.012, size=n)
    log_prices = np.log(100.0) + log_rets.cumsum()
    prices = np.exp(log_prices)
    volume = rng.integers(500_000, 5_000_000, size=n)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": prices,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Adj Close": prices,
            "Volume": volume,
        }
    )


@pytest.fixture
def synthetic_feature_panel(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """`(features, targets, dates)` with a deliberately learnable structure.

    Feature column 0 is correlated with target column 0 so even a tiny
    training run can drive the validation AUC above 0.55.
    """

    n_rows = 240
    n_features = 8
    X = rng.normal(0.0, 1.0, size=(n_rows, n_features))
    # Inject a learnable signal: target 0 = 1 if x_0 + 0.5*x_1 > 0.
    signal = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 0.3, size=n_rows)
    y0 = (signal > 0).astype(float)
    y1 = (X[:, 2] > 0).astype(float)
    y2 = (X[:, 0] - X[:, 3] > 0).astype(float)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    targets = pd.DataFrame({"pos_1d": y0, "pos_med_5d": y1, "pos_sum_5d": y2})
    dates = np.array(
        pd.date_range("2024-01-02", periods=n_rows, freq="B").to_numpy()
    )
    return features, targets, dates
