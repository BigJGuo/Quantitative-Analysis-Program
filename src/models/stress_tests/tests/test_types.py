"""Unit tests for the dataclasses in `stress_tests/types.py`."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.stress_tests.types import (
    CrisisWindow,
    HypotheticalScenario,
    ReverseStressResult,
    ScenarioPnL,
    StressFit,
    StressInputs,
    StressReport,
)


class TestCrisisWindow:
    def test_constructs_ok(self) -> None:
        w = CrisisWindow(name="X", t1="2020-02-19", t2="2020-03-23")
        assert w.name == "X"

    def test_rejects_inverted_window(self) -> None:
        with pytest.raises(ValueError, match="t1 .* > t2"):
            CrisisWindow(name="X", t1="2020-03-23", t2="2020-02-19")


class TestScenarioPnL:
    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="scenario_type"):
            ScenarioPnL(
                scenario_name="x",
                scenario_type="bogus",  # type: ignore[arg-type]
                total_pnl=0.0,
                contributions={},
                factor_changes={},
            )


class TestReverseStressResult:
    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="dF_star"):
            ReverseStressResult(
                dF_star=np.zeros(3),
                std_devs=np.zeros(2),
                mahalanobis_distance=1.0,
                loss_target=100.0,
                factor_names=("a", "b", "c"),
            )

    def test_rejects_nonpositive_target(self) -> None:
        with pytest.raises(ValueError, match="loss_target"):
            ReverseStressResult(
                dF_star=np.zeros(1),
                std_devs=np.zeros(1),
                mahalanobis_distance=1.0,
                loss_target=0.0,
                factor_names=("a",),
            )


class TestStressFit:
    def test_basic_construction(self) -> None:
        fit = StressFit(
            tickers=("AAA", "BBB"),
            factor_names=("equity", "rates"),
            betas=np.array([[1.0, 0.1], [1.2, -0.05]]),
            factor_covariance=np.array([[0.04, 0.0], [0.0, 0.0001]]),
            capital=1_000_000.0,
        )
        assert fit.n_assets == 2
        assert fit.n_factors == 2

    def test_rejects_mismatched_betas(self) -> None:
        with pytest.raises(ValueError, match="betas"):
            StressFit(
                tickers=("AAA",),
                factor_names=("equity", "rates"),
                betas=np.zeros((2, 2)),
                factor_covariance=np.eye(2),
                capital=1.0,
            )


class TestStressInputs:
    def test_requires_positions(self) -> None:
        with pytest.raises(ValueError, match="positions"):
            StressInputs(
                positions={},
                capital=1.0,
                crisis_windows=(),
                hypotheticals=(),
                loss_target_fraction=0.3,
                timestamp=datetime.now(UTC),
                crisis_prices={},
                factor_returns=pd.DataFrame(),
                sector_map={},
            )

    def test_rejects_bad_loss_fraction(self) -> None:
        with pytest.raises(ValueError, match="loss_target_fraction"):
            StressInputs(
                positions={"AAA": 1.0},
                capital=1.0,
                crisis_windows=(),
                hypotheticals=(),
                loss_target_fraction=1.5,
                timestamp=datetime.now(UTC),
                crisis_prices={},
                factor_returns=pd.DataFrame(),
                sector_map={},
            )


class TestStressReport:
    def test_rejects_unknown_worst_type(self) -> None:
        with pytest.raises(ValueError, match="worst_case_type"):
            StressReport(
                timestamp=datetime.now(UTC),
                capital=1.0,
                historical={},
                hypothetical={},
                reverse=None,
                worst_case=0.0,
                worst_case_name="x",
                worst_case_type="bogus",  # type: ignore[arg-type]
            )


def test_hypothetical_scenario_dataclass() -> None:
    scen = HypotheticalScenario(name="X", shocks={"equity": -0.2})
    assert scen.shocks["equity"] == pytest.approx(-0.2)
