"""Shared synthetic fixtures for VaR unit tests.

We build deterministic price panels so analytic VaR assertions stay
tractable. ``SYN1`` is a single asset with Gaussian iid returns at
known sigma; ``SYN2`` is a second iid asset, uncorrelated with the first,
used to exercise the multi-asset code paths.
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
def true_sigma_syn1() -> float:
    """1% daily decimal vol — typical for a slow equity index ETF."""

    return 0.01


@pytest.fixture
def true_sigma_syn2() -> float:
    """2% daily decimal vol — moderately volatile single-name."""

    return 0.02


@pytest.fixture
def synthetic_returns_two_asset(
    true_sigma_syn1: float, true_sigma_syn2: float
) -> pd.DataFrame:
    """1000 daily decimal returns, two zero-correlated Gaussian assets."""

    rng = np.random.default_rng(seed=20260518)
    n = 1000
    r1 = rng.standard_normal(n) * true_sigma_syn1
    r2 = rng.standard_normal(n) * true_sigma_syn2
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.DataFrame({"SYN1": r1, "SYN2": r2}, index=idx)


@pytest.fixture
def synthetic_close_two_asset(
    synthetic_returns_two_asset: pd.DataFrame,
) -> pd.DataFrame:
    """Cumulative close-price panel consistent with the synthetic returns."""

    rets = synthetic_returns_two_asset
    # Simple-return compounding -> P_t = P_0 * prod(1 + r_t)
    prices = (1.0 + rets).cumprod() * 100.0
    return prices


@pytest.fixture
def in_memory_provider(
    synthetic_close_two_asset: pd.DataFrame,
) -> InMemoryProvider:
    """`InMemoryProvider` exposing SYN1 and SYN2 with synthetic prices."""

    panel = synthetic_close_two_asset

    def _bars_for(col: str) -> pd.DataFrame:
        closes = panel[col].astype(float)
        return pd.DataFrame(
            {
                "Date": [pd.Timestamp(d) for d in closes.index],
                "Open": closes.to_numpy(dtype=float),
                "High": closes.to_numpy(dtype=float) * 1.001,
                "Low": closes.to_numpy(dtype=float) * 0.999,
                "Close": closes.to_numpy(dtype=float),
                "Volume": np.ones(len(closes), dtype=int) * 1_000_000,
            }
        )

    return InMemoryProvider(
        prices={
            ("SYN1", "3y", "1d"): _bars_for("SYN1"),
            ("SYN2", "3y", "1d"): _bars_for("SYN2"),
        }
    )


@pytest.fixture
def two_asset_positions() -> dict[str, float]:
    """$1M long SYN1, $500k long SYN2 — a representative test book."""

    return {"SYN1": 1_000_000.0, "SYN2": 500_000.0}
