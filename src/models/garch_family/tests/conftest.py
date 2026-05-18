"""Shared synthetic fixtures for GARCH-family unit tests.

Hand-checkable parameters so worked-out assertions stay tractable.
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
def true_garch_params() -> dict[str, float]:
    """GARCH(1,1) parameters with comfortable persistence and ARCH effect."""

    return {"omega": 0.05, "alpha": 0.08, "beta": 0.9, "mu": 0.0}


@pytest.fixture
def true_gjr_params() -> dict[str, float]:
    """GJR(1,1) parameters with a non-trivial leverage term."""

    return {"omega": 0.05, "alpha": 0.05, "gamma": 0.08, "beta": 0.88, "mu": 0.0}


@pytest.fixture
def synthetic_garch_returns(true_garch_params: dict[str, float]) -> pd.Series:
    """2500 daily returns drawn from a GARCH(1,1) with Gaussian innovations.

    Returns are in **percent** units (the arch-package convention used
    throughout this model). 2500 obs ~= 10 years of daily data.
    """

    p = true_garch_params
    rng = np.random.default_rng(seed=20260518)
    t = 2500
    eps = np.empty(t, dtype=float)
    s2 = np.empty(t, dtype=float)
    s2[0] = p["omega"] / (1.0 - p["alpha"] - p["beta"])
    for i in range(1, t):
        s2[i] = p["omega"] + p["alpha"] * eps[i - 1] ** 2 + p["beta"] * s2[i - 1]
        eps[i] = float(np.sqrt(s2[i]) * rng.standard_normal())
    return pd.Series(p["mu"] + eps, index=pd.date_range("2016-01-04", periods=t, freq="B"))


@pytest.fixture
def synthetic_gjr_returns(true_gjr_params: dict[str, float]) -> pd.Series:
    p = true_gjr_params
    rng = np.random.default_rng(seed=20260519)
    t = 2500
    eps = np.empty(t, dtype=float)
    s2 = np.empty(t, dtype=float)
    s2[0] = p["omega"] / (1.0 - p["alpha"] - 0.5 * p["gamma"] - p["beta"])
    for i in range(1, t):
        ind = 1.0 if eps[i - 1] < 0.0 else 0.0
        s2[i] = (
            p["omega"]
            + p["alpha"] * eps[i - 1] ** 2
            + p["gamma"] * ind * eps[i - 1] ** 2
            + p["beta"] * s2[i - 1]
        )
        eps[i] = float(np.sqrt(s2[i]) * rng.standard_normal())
    return pd.Series(p["mu"] + eps, index=pd.date_range("2016-01-04", periods=t, freq="B"))


@pytest.fixture
def synthetic_close_series(synthetic_garch_returns: pd.Series) -> pd.Series:
    """Cumulative price series consistent with the synthetic GARCH returns.

    Returns are stored in percent units, so we divide by 100 before exponentiating.
    """

    log_r = synthetic_garch_returns.to_numpy(dtype=float) / 100.0
    return pd.Series(
        100.0 * np.exp(np.cumsum(log_r)),
        index=synthetic_garch_returns.index,
        name="Close",
    )


@pytest.fixture
def in_memory_provider(synthetic_close_series: pd.Series) -> InMemoryProvider:
    """`InMemoryProvider` exposing one ticker, ``"SYN"``, with the synthetic prices."""

    close = synthetic_close_series
    bars = pd.DataFrame(
        {
            "Date": [pd.Timestamp(d) for d in close.index],
            "Open": close.to_numpy(dtype=float),
            "High": close.to_numpy(dtype=float) * 1.001,
            "Low": close.to_numpy(dtype=float) * 0.999,
            "Close": close.to_numpy(dtype=float),
            "Volume": np.ones(len(close), dtype=int) * 1_000_000,
        }
    )
    return InMemoryProvider(prices={("SYN", "10y", "1d"): bars})
