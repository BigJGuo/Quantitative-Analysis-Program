"""End-to-end tests for the `StressTests` orchestration class.

Uses an `InMemoryProvider` populated with synthetic long-history price panels
so the fetch -> calibrate -> predict -> validate flow runs without network.
"""

from __future__ import annotations

import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.stress_tests import StressTests
from src.models.stress_tests.types import (
    CrisisWindow,
    HypotheticalScenario,
    StressInputs,
)


class TestStressTestsClass:
    def test_registered_in_global_registry(self) -> None:
        assert get_model("stress_tests") is StressTests

    def test_base_model_class_attrs(self) -> None:
        assert StressTests.name == "stress_tests"
        assert StressTests.layer == 6
        assert StressTests.refit_frequency == "daily"

    def test_constructor_rejects_empty_positions(self) -> None:
        with pytest.raises(ValueError, match="at least one position"):
            StressTests(positions={}, capital=1_000_000.0)

    def test_constructor_rejects_nonpositive_capital(self) -> None:
        with pytest.raises(ValueError, match="capital"):
            StressTests(positions={"AAA": 1.0}, capital=0.0)

    def test_constructor_rejects_bad_loss_fraction(self) -> None:
        with pytest.raises(ValueError, match="loss_target_fraction"):
            StressTests(
                positions={"AAA": 1.0},
                capital=1.0,
                loss_target_fraction=2.0,
            )


class TestFetchData:
    def test_returns_stress_inputs(
        self,
        positions: dict[str, float],
        capital: float,
        crisis_windows: tuple[CrisisWindow, ...],
        hypotheticals: tuple[HypotheticalScenario, ...],
        sector_map: dict[str, str],
        provider: InMemoryProvider,
    ) -> None:
        model = StressTests(
            positions=positions,
            capital=capital,
            crisis_windows=crisis_windows,
            hypotheticals=hypotheticals,
            sector_map=sector_map,
            calibration_period="2y",
        )
        data = model.fetch_data(provider)
        assert isinstance(data, StressInputs)
        # Each (window, ticker) we have data for should appear in crisis_prices.
        # The synthetic panel covers 1985-2024, so every window in the fixture
        # is in-range.
        for window in crisis_windows:
            for ticker in positions:
                assert (window.name, ticker) in data.crisis_prices
        assert not data.factor_returns.empty
        assert "price_panel" in data.metadata


class TestCalibrate:
    def test_round_trip_calibrate(
        self,
        positions: dict[str, float],
        capital: float,
        crisis_windows: tuple[CrisisWindow, ...],
        hypotheticals: tuple[HypotheticalScenario, ...],
        sector_map: dict[str, str],
        provider: InMemoryProvider,
    ) -> None:
        model = StressTests(
            positions=positions,
            capital=capital,
            crisis_windows=crisis_windows,
            hypotheticals=hypotheticals,
            sector_map=sector_map,
            calibration_period="2y",
        )
        data = model.fetch_data(provider)
        result = model.calibrate(data)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "stress_tests"
        assert model.fit.n_assets == len(positions)


class TestPredict:
    def test_returns_risk_metric(
        self,
        positions: dict[str, float],
        capital: float,
        crisis_windows: tuple[CrisisWindow, ...],
        hypotheticals: tuple[HypotheticalScenario, ...],
        sector_map: dict[str, str],
        provider: InMemoryProvider,
    ) -> None:
        model = StressTests(
            positions=positions,
            capital=capital,
            crisis_windows=crisis_windows,
            hypotheticals=hypotheticals,
            sector_map=sector_map,
            calibration_period="2y",
        )
        data = model.fetch_data(provider)
        model.calibrate(data)
        risk = model.predict(data)
        assert isinstance(risk, RiskMetric)
        assert risk.ticker == "PORTFOLIO"
        assert risk.metric_name == "stress_worst_case_pnl"
        for key in (
            "worst_case_name",
            "worst_case_type",
            "historical_pnls",
            "hypothetical_pnls",
            "capital",
        ):
            assert key in risk.metadata

    def test_predict_without_calibrate_still_runs_historical(
        self,
        positions: dict[str, float],
        capital: float,
        crisis_windows: tuple[CrisisWindow, ...],
        hypotheticals: tuple[HypotheticalScenario, ...],
        sector_map: dict[str, str],
        provider: InMemoryProvider,
    ) -> None:
        # Historical replay does not require a fit; predict should still return
        # a RiskMetric, just with empty hypothetical / reverse metadata.
        model = StressTests(
            positions=positions,
            capital=capital,
            crisis_windows=crisis_windows,
            hypotheticals=hypotheticals,
            sector_map=sector_map,
        )
        data = model.fetch_data(provider)
        risk = model.predict(data)
        assert risk.metadata["hypothetical_pnls"] == {}
        assert risk.metadata["reverse_mahalanobis"] is None


class TestValidate:
    def test_validate_emits_diagnostics(
        self,
        positions: dict[str, float],
        capital: float,
        crisis_windows: tuple[CrisisWindow, ...],
        hypotheticals: tuple[HypotheticalScenario, ...],
        sector_map: dict[str, str],
        provider: InMemoryProvider,
    ) -> None:
        model = StressTests(
            positions=positions,
            capital=capital,
            crisis_windows=crisis_windows,
            hypotheticals=hypotheticals,
            sector_map=sector_map,
            calibration_period="2y",
        )
        data = model.fetch_data(provider)
        model.calibrate(data)
        diag = model.validate(data)
        for key in (
            "worst_case",
            "worst_case_name",
            "worst_case_type",
            "n_historical",
            "n_hypothetical",
            "coverage_empty_factors",
            "coverage_empty_scenarios",
            "endpoint_window",
            "endpoint_pnls",
            "endpoint_range",
        ):
            assert key in diag

    def test_validate_top3_contributors_present(
        self,
        positions: dict[str, float],
        capital: float,
        crisis_windows: tuple[CrisisWindow, ...],
        hypotheticals: tuple[HypotheticalScenario, ...],
        sector_map: dict[str, str],
        provider: InMemoryProvider,
    ) -> None:
        model = StressTests(
            positions=positions,
            capital=capital,
            crisis_windows=crisis_windows,
            hypotheticals=hypotheticals,
            sector_map=sector_map,
            calibration_period="2y",
        )
        data = model.fetch_data(provider)
        model.calibrate(data)
        diag = model.validate(data)
        assert "worst_historical_top3" in diag
        top = diag["worst_historical_top3"]
        # Each entry is (ticker, pnl); should be at most 3.
        assert 1 <= len(top) <= 3


class TestPublicConveniences:
    def test_run_full_report(
        self,
        positions: dict[str, float],
        capital: float,
        crisis_windows: tuple[CrisisWindow, ...],
        hypotheticals: tuple[HypotheticalScenario, ...],
        sector_map: dict[str, str],
        provider: InMemoryProvider,
    ) -> None:
        model = StressTests(
            positions=positions,
            capital=capital,
            crisis_windows=crisis_windows,
            hypotheticals=hypotheticals,
            sector_map=sector_map,
            calibration_period="2y",
        )
        data = model.fetch_data(provider)
        model.calibrate(data)
        report = model.run_full_report(data)
        assert len(report.historical) == len(crisis_windows)
        assert len(report.hypothetical) == len(hypotheticals)
        assert report.reverse is not None
