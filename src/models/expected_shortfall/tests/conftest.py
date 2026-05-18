"""Shared synthetic fixtures for the ExpectedShortfall unit tests.

The fixtures generate a deterministic two-asset return panel from a known
multivariate Gaussian; this lets the parametric-normal ES be checked against
its closed form to machine precision and lets the historical / monte-carlo
estimators be checked for convergence to the same value as `N` grows.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 18, 14, 30, tzinfo=UTC)


@pytest.fixture
def true_mvn_params() -> dict[str, np.ndarray]:
    """Hand-checkable multivariate-normal return parameters for two assets."""

    mu = np.array([0.0005, 0.0008], dtype=float)
    Sigma = np.array(
        [
            [0.0004, 0.0001],
            [0.0001, 0.0009],
        ],
        dtype=float,
    )
    return {"mu": mu, "Sigma": Sigma}


@pytest.fixture
def synthetic_returns(true_mvn_params: dict[str, np.ndarray]) -> pd.DataFrame:
    """3000 daily i.i.d. multivariate-normal returns for two tickers."""

    rng = np.random.default_rng(seed=20260518)
    mu = true_mvn_params["mu"]
    Sigma = true_mvn_params["Sigma"]
    L = np.linalg.cholesky(Sigma)
    t = 3000
    z = rng.standard_normal(size=(t, 2))
    r = mu[np.newaxis, :] + z @ L.T
    idx = pd.date_range("2014-01-02", periods=t, freq="B")
    return pd.DataFrame(r, index=idx, columns=["AAA", "BBB"])


@pytest.fixture
def synthetic_prices(synthetic_returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative price panel consistent with the synthetic returns."""

    init = 100.0
    cum = (1.0 + synthetic_returns).cumprod() * init
    return cum


@pytest.fixture
def in_memory_provider(synthetic_prices: pd.DataFrame) -> InMemoryProvider:
    """`InMemoryProvider` exposing two tickers (`AAA`, `BBB`) with synthetic prices."""

    prices: dict[tuple[str, str, str], pd.DataFrame] = {}
    for ticker in synthetic_prices.columns:
        s = synthetic_prices[ticker]
        bars = pd.DataFrame(
            {
                "Date": [pd.Timestamp(d) for d in s.index],
                "Open": s.to_numpy(dtype=float),
                "High": s.to_numpy(dtype=float) * 1.001,
                "Low": s.to_numpy(dtype=float) * 0.999,
                "Close": s.to_numpy(dtype=float),
                "Volume": np.ones(len(s), dtype=int) * 1_000_000,
            }
        )
        prices[(ticker, "3y", "1d")] = bars
    return InMemoryProvider(prices=prices)
