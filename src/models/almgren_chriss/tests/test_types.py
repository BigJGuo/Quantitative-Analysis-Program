"""Validation tests for the model-internal dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.almgren_chriss.types import (
    ImpactParams,
    LiquidationProblem,
    MultiAssetProblem,
)


def test_impact_params_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        ImpactParams(
            sigma=0.0, eta=1.0e-6, gamma=1.0e-7, last_price=100.0,
            daily_volume=1.0e6,
        )


def test_impact_params_rejects_nonpositive_eta() -> None:
    with pytest.raises(ValueError, match="eta"):
        ImpactParams(
            sigma=0.2, eta=-1.0, gamma=1.0e-7, last_price=100.0,
            daily_volume=1.0e6,
        )


def test_impact_params_allows_zero_gamma() -> None:
    p = ImpactParams(
        sigma=0.2, eta=1.0e-6, gamma=0.0, last_price=100.0,
        daily_volume=1.0e6,
    )
    assert p.gamma == 0.0


def test_impact_params_rejects_negative_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        ImpactParams(
            sigma=0.2, eta=1.0e-6, gamma=-1.0, last_price=100.0,
            daily_volume=1.0e6,
        )


def test_liquidation_problem_rejects_invalid_side() -> None:
    with pytest.raises(ValueError, match="side"):
        LiquidationProblem(
            ticker="A", X=1.0, T=1.0, N=10, side="hold", lam=1e-6  # type: ignore[arg-type]
        )


def test_liquidation_problem_rejects_nonpositive_X() -> None:
    with pytest.raises(ValueError, match="X"):
        LiquidationProblem(ticker="A", X=0.0, T=1.0, N=10, side="sell", lam=1e-6)


def test_liquidation_problem_rejects_zero_N() -> None:
    with pytest.raises(ValueError, match="N"):
        LiquidationProblem(ticker="A", X=1.0, T=1.0, N=0, side="sell", lam=1e-6)


def test_multi_asset_problem_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        MultiAssetProblem(
            tickers=("A", "B"),
            X=np.asarray([1.0]),  # wrong shape
            T=1.0,
            N=10,
            sides=("sell", "sell"),
            lam=1e-6,
        )


def test_multi_asset_problem_validates_side_strings() -> None:
    with pytest.raises(ValueError, match="sides"):
        MultiAssetProblem(
            tickers=("A", "B"),
            X=np.asarray([1.0, 2.0]),
            T=1.0,
            N=10,
            sides=("sell", "hold"),  # type: ignore[arg-type]
            lam=1e-6,
        )
