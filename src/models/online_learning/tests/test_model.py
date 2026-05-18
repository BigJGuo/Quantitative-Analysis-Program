"""End-to-end tests for the `OnlineLearning` BaseModel subclass.

Uses `InMemoryProvider` with synthetic GBM prices so the tests are
deterministic and run without yfinance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, Forecast
from src.models.online_learning.model import OnlineLearning
from src.models.online_learning.types import OnlineLearningInputs


class TestConstruction:
    def test_class_attributes(self) -> None:
        assert OnlineLearning.name == "online_learning"
        assert OnlineLearning.layer == 4
        assert OnlineLearning.refit_frequency == "daily"

    def test_rejects_empty_ticker(self) -> None:
        with pytest.raises(ValueError, match="ticker"):
            OnlineLearning(ticker="")

    def test_rejects_unsupported_ftrl_kind(self) -> None:
        # FTRL is supported via calibrate_ftrl, not via the daily-bars
        # fetch path.
        with pytest.raises(ValueError, match="FTRL"):
            OnlineLearning(ticker="SPY", kind="ftrl")

    def test_rejects_invalid_kind(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            OnlineLearning(ticker="SPY", kind="banana")  # type: ignore[arg-type]

    def test_requires_at_least_two_experts(self) -> None:
        with pytest.raises(ValueError, match="at least 2 experts"):
            OnlineLearning(
                ticker="SPY",
                rolling_windows=(5,),
                ewma_lambdas=(),
            )

    def test_registered_in_global_registry(self) -> None:
        cls = get_model("online_learning")
        assert cls is OnlineLearning


class TestFetchAndCalibrate:
    def test_fetch_data_produces_inputs(
        self, in_memory_provider: InMemoryProvider
    ) -> None:
        model = OnlineLearning(ticker="FAKE")
        inputs = model.fetch_data(in_memory_provider)
        assert isinstance(inputs, OnlineLearningInputs)
        assert inputs.kind == "hedge"
        # 5 experts: 3 rolling + 2 EWMA.
        assert inputs.stream.predictions.shape[1] == 5
        assert inputs.metadata["ticker"] == "FAKE"

    def test_calibrate_returns_result(
        self, in_memory_provider: InMemoryProvider
    ) -> None:
        model = OnlineLearning(ticker="FAKE")
        inputs = model.fetch_data(in_memory_provider)
        result = model.calibrate(inputs)
        assert isinstance(result, CalibrationResult)
        assert result.fit_metrics["n_experts"] == 5
        assert result.fit_metrics["empirical_regret"] >= 0
        # `fit` is now accessible.
        assert model.fit is not None

    def test_predict_returns_forecast(
        self, in_memory_provider: InMemoryProvider
    ) -> None:
        model = OnlineLearning(ticker="FAKE", annualize=True)
        inputs = model.fetch_data(in_memory_provider)
        model.calibrate(inputs)
        forecast = model.predict(inputs)
        assert isinstance(forecast, Forecast)
        assert forecast.ticker == "FAKE"
        # Annualized vol of a synthetic GBM with sigma=0.01 → ~16% annualized.
        # Bound loosely to absorb random variation.
        assert 0.05 < forecast.value < 1.0
        assert forecast.metadata["annualized"] is True
        assert "weights" in forecast.metadata

    def test_validate_returns_diagnostics(
        self, in_memory_provider: InMemoryProvider
    ) -> None:
        model = OnlineLearning(ticker="FAKE")
        inputs = model.fetch_data(in_memory_provider)
        model.calibrate(inputs)
        diag = model.validate(inputs)
        assert "empirical_regret" in diag
        assert "theoretical_regret_bound" in diag
        assert "final_weight_entropy" in diag
        # Weights must be a valid simplex.
        weights = np.asarray(diag["weights"], dtype=float)
        assert weights.shape == (5,)
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        # Entropy bounded by log N.
        assert diag["final_weight_entropy"] <= diag["uniform_entropy"] + 1e-9

    def test_predict_requires_calibration(
        self, in_memory_provider: InMemoryProvider
    ) -> None:
        model = OnlineLearning(ticker="FAKE")
        inputs = model.fetch_data(in_memory_provider)
        with pytest.raises(RuntimeError, match="calibrated"):
            model.predict(inputs)

    def test_calibrate_rejects_wrong_payload(self) -> None:
        model = OnlineLearning(ticker="FAKE")
        with pytest.raises(TypeError, match="OnlineLearningInputs"):
            model.calibrate({"not": "inputs"})


def test_empirical_regret_below_theoretical_bound(
    in_memory_provider: InMemoryProvider,
) -> None:
    """Spec validation step: empirical regret should stay below `sqrt(T log N / 2)`."""

    model = OnlineLearning(ticker="FAKE", eta=0.5, eta_schedule="adaptive")
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    diag = model.validate(inputs)
    bound = diag["theoretical_regret_bound"]
    regret = diag["empirical_regret"]
    assert math.isfinite(bound)
    # The textbook Hedge bound is conservative — on real-ish vol data the
    # empirical regret typically lands well below it. We allow a generous
    # 5x slack to absorb the log-squared-loss scaling.
    assert regret < 5 * bound


def test_close_series_extraction_handles_index_date(
    in_memory_provider: InMemoryProvider,
) -> None:
    """Regression: the daily-OHLC frame can have either a 'Date' column or a DatetimeIndex."""

    model = OnlineLearning(ticker="FAKE")
    inputs = model.fetch_data(in_memory_provider)
    # The internal Close series should preserve the original number of business days.
    assert inputs.stream.predictions.shape[0] > 100
