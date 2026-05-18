"""Shared synthetic fixtures for the HAR-RV unit tests.

The synthetic-RV generators emit daily series with a known autocorrelation
structure (AR(1) in log RV, with a slow-decay tail) so worked-out checks on
the regression (coefficient signs, R^2 ranges, persistence sum) become
tight rather than vacuous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


def _ar1_log_rv(
    rng: np.random.Generator,
    *,
    n_obs: int,
    mu: float = -9.5,
    phi: float = 0.95,
    sigma: float = 0.30,
) -> pd.Series:
    """Generate `n_obs` daily log-RV values from an AR(1) with mean `mu`.

    With `phi=0.95` and `sigma=0.3` the level series mimics the persistence
    and dispersion of liquid US equity RV (RV ~ 1e-4 daily on average).
    """

    log_rv = np.empty(n_obs)
    log_rv[0] = mu + rng.normal(0.0, sigma / np.sqrt(1.0 - phi**2))
    for t in range(1, n_obs):
        log_rv[t] = mu + phi * (log_rv[t - 1] - mu) + rng.normal(0.0, sigma)
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.Series(np.exp(log_rv), index=index, name="rv_d")


@pytest.fixture
def persistent_log_rv(rng: np.random.Generator) -> pd.Series:
    """Strongly-persistent daily RV: AR(1) with phi=0.95 in log space."""

    return _ar1_log_rv(rng, n_obs=900)


@pytest.fixture
def low_persistence_rv(rng: np.random.Generator) -> pd.Series:
    """Weakly-persistent daily RV: phi=0.2, dominated by noise."""

    return _ar1_log_rv(rng, n_obs=500, phi=0.2)


@pytest.fixture
def synthetic_intraday_bars() -> pd.DataFrame:
    """Two 78-bar trading days of 5-minute prices for `compute_daily_rv`.

    The first day has a small constant per-bar log return so the manual RV
    is exactly `M * r^2`. The second day uses negative-then-positive blocks
    so the semivariance split is exact.
    """

    timestamps_day1 = pd.date_range("2026-01-05 09:30", periods=78, freq="5min")
    timestamps_day2 = pd.date_range("2026-01-06 09:30", periods=78, freq="5min")
    timestamps = timestamps_day1.append(timestamps_day2)

    # Day 1: log return per bar = +log(1.001) ≈ 0.0009995
    p_day1 = 100.0 * (1.001 ** np.arange(78))

    # Day 2: 39 bars of -0.001 then 39 bars of +0.001, starting from 100.
    n_half = 39
    factors = np.concatenate(
        [np.full(n_half, 0.999), np.full(78 - n_half, 1.001)]
    )
    p_day2 = 100.0 * np.cumprod(factors)

    prices = np.concatenate([p_day1, p_day2])
    return pd.DataFrame(
        {"Close": prices, "Open": prices, "High": prices, "Low": prices},
        index=timestamps,
    ).reset_index().rename(columns={"index": "Datetime"})


@pytest.fixture
def synthetic_daily_ohlc() -> pd.DataFrame:
    """Daily OHLC frame with a known Garman-Klass variance per day.

    Each day has H/L = exp(0.02) and C/O = exp(0.005) so the per-day GK
    value is exactly:

        0.5 * 0.02^2 - (2 log 2 - 1) * 0.005^2
    """

    n = 30
    closes = 100.0 * (1.0 + 0.001 * np.arange(n))
    opens = closes / np.exp(0.005)
    highs = opens * np.exp(0.02)
    lows = opens
    index = pd.date_range("2025-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=index,
    ).reset_index().rename(columns={"index": "Date"})
