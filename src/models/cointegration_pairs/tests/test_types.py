"""Unit tests for the model-internal dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.cointegration_pairs.types import (
    ADFResult,
    KalmanFit,
    PairInputs,
    TradingRule,
)


class TestADFResult:
    def test_rejects_unit_root(self) -> None:
        result = ADFResult(
            t_stat=-3.0,
            n_obs=100,
            n_lags=1,
            critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
            regression="c",
            test_type="adf",
        )
        assert result.rejects_unit_root("5%")
        assert not result.rejects_unit_root("1%")

    def test_unknown_level_raises(self) -> None:
        result = ADFResult(
            t_stat=-2.0,
            n_obs=10,
            n_lags=0,
            critical_values={"5%": -3.0},
            regression="c",
            test_type="adf",
        )
        with pytest.raises(KeyError, match="bogus"):
            result.rejects_unit_root("bogus")


class TestPairInputs:
    def test_mismatched_indices_raise(self) -> None:
        a = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
        b = pd.Series([1.0, 2.0], index=pd.RangeIndex(1, 3))
        with pytest.raises(ValueError, match="index"):
            PairInputs(
                ticker_a="A",
                ticker_b="B",
                log_price_a=a,
                log_price_b=b,
                timestamp=datetime.now(UTC),
            )

    def test_same_ticker_raises(self) -> None:
        a = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="differ"):
            PairInputs(
                ticker_a="X",
                ticker_b="X",
                log_price_a=a,
                log_price_b=a.copy(),
                timestamp=datetime.now(UTC),
            )


class TestKalmanFit:
    def test_shape_check(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            KalmanFit(
                beta_path=np.zeros(5),
                variance_path=np.zeros(4),
                innovations=np.zeros(5),
                innovation_variance=np.zeros(5),
                standardized_innovations=np.zeros(5),
                Q=0.0,
                R=1.0,
                beta0=0.0,
                P0=1.0,
            )

    def test_R_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="Q >= 0"):
            KalmanFit(
                beta_path=np.zeros(3),
                variance_path=np.zeros(3),
                innovations=np.zeros(3),
                innovation_variance=np.zeros(3),
                standardized_innovations=np.zeros(3),
                Q=0.0,
                R=0.0,
                beta0=0.0,
                P0=1.0,
            )


class TestTradingRule:
    def test_defaults(self) -> None:
        rule = TradingRule()
        assert 0 < rule.s_out < rule.s_in < rule.s_stop

    def test_ordering_enforced(self) -> None:
        with pytest.raises(ValueError, match="TradingRule"):
            TradingRule(s_in=1.0, s_out=2.0, s_stop=3.0)
