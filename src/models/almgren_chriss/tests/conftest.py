"""Shared fixtures for Almgren-Chriss unit tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.almgren_chriss.types import (
    AlmgrenChrissInputs,
    ImpactParams,
    LiquidationProblem,
)


@pytest.fixture
def default_params() -> ImpactParams:
    """Canonical impact parameters used across the unit tests.

    Numbers chosen so that ``κ T`` lands near 1 (the "balanced" regime).
    With ``T = 1`` day, ``λ = 1e-6``, ``σ = 0.25``, ``S_0 = 100``:

      ``σ_d = 100 · 0.25 / sqrt(252) ≈ 1.575``
      ``κ²  = λ σ_d² / η``
      For ``κ T ≈ 1``: ``η = λ σ_d² T² ≈ 2.48 · 10⁻⁶``.

    The ``η`` chosen here is intentionally far above the Almgren-2005
    default — this keeps the unit-tests focused on the math regime in
    the spec's "production" range.
    """

    return ImpactParams(
        sigma=0.25,
        eta=2.5e-6,
        gamma=2.5e-7,
        last_price=100.0,
        daily_volume=5.0e7,
    )


@pytest.fixture
def default_problem() -> LiquidationProblem:
    return LiquidationProblem(
        ticker="TEST",
        X=1.0e6,
        T=1.0,
        N=50,
        side="sell",
        lam=1.0e-6,
    )


@pytest.fixture
def synthetic_inputs() -> AlmgrenChrissInputs:
    """A deterministic synthetic AlmgrenChrissInputs bundle.

    Builds 60 days of close + volume with a known annualized vol (~20%),
    plus an artificial U-shaped intraday volume profile.
    """

    rng = np.random.default_rng(seed=42)
    n_days = 60
    sigma_annual = 0.20
    sigma_daily = sigma_annual / np.sqrt(252)
    rets = rng.normal(loc=0.0, scale=sigma_daily, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    close = pd.Series(prices, index=dates, name="Close")
    volume = pd.Series(
        rng.lognormal(mean=np.log(5.0e7), sigma=0.1, size=n_days),
        index=dates,
        name="Volume",
    )
    bars_per_day = 12
    # U-shaped profile
    pos = np.arange(bars_per_day) / (bars_per_day - 1)
    profile = 0.6 + 0.4 * (1.0 - 4.0 * pos * (1.0 - pos))  # min at midday
    profile = profile / profile.sum()

    problem = LiquidationProblem(
        ticker="TEST",
        X=1.0e5,
        T=1.0,
        N=bars_per_day,
        side="sell",
        lam=1.0e-6,
    )
    return AlmgrenChrissInputs(
        ticker="TEST",
        close=close,
        volume=volume,
        intraday_volume_profile=profile,
        last_price=float(close.iloc[-1]),
        problem=problem,
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
    )
