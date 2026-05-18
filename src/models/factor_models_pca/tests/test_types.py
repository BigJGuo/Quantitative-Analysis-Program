"""Tests for the dataclasses defined in `types.py`."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.factor_models_pca.types import (
    FactorFit,
    FactorInputs,
    PortfolioRisk,
)


def _ok_fit() -> FactorFit:
    return FactorFit(
        B=np.ones((3, 2)),
        F=np.eye(2),
        specific_variances=np.full(3, 1e-3),
        tickers=("A", "B", "C"),
        factor_names=("PC1", "PC2"),
        method="PCA",
    )


class TestFactorFit:
    def test_shape_properties(self) -> None:
        fit = _ok_fit()
        assert fit.n_assets == 3
        assert fit.n_factors == 2

    def test_asset_index(self) -> None:
        fit = _ok_fit()
        assert fit.asset_index("B") == 1
        with pytest.raises(KeyError):
            fit.asset_index("Z")


class TestFactorInputs:
    def test_valid_construction(self) -> None:
        rets = pd.DataFrame(np.zeros((10, 2)), columns=["A", "B"])
        inputs = FactorInputs(
            returns=rets,
            tickers=("A", "B"),
            method="PCA",
            K=1,
            timestamp=datetime.now(UTC),
        )
        assert inputs.method == "PCA"

    def test_rejects_mismatched_tickers(self) -> None:
        rets = pd.DataFrame(np.zeros((10, 2)), columns=["A", "B"])
        with pytest.raises(ValueError, match="returns has 2 columns"):
            FactorInputs(
                returns=rets,
                tickers=("A",),
                method="PCA",
                K=1,
                timestamp=datetime.now(UTC),
            )

    def test_rejects_mismatched_weights(self) -> None:
        rets = pd.DataFrame(np.zeros((10, 2)), columns=["A", "B"])
        with pytest.raises(ValueError, match="weights shape"):
            FactorInputs(
                returns=rets,
                tickers=("A", "B"),
                method="PCA",
                K=1,
                timestamp=datetime.now(UTC),
                weights=np.ones(3),
            )


class TestPortfolioRisk:
    def test_rejects_negative_vols(self) -> None:
        with pytest.raises(ValueError):
            PortfolioRisk(
                total_vol=-1.0,
                systematic_vol=0.0,
                specific_vol=0.0,
                factor_exposures=np.zeros(2),
                factor_contributions=np.zeros(2),
            )
