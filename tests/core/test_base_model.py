"""Tests for `BaseModel`, the `@register_model` decorator, and shared types."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from src.core.base_model import BaseModel
from src.core.data_provider import DataProvider, InMemoryProvider
from src.core.registry import (
    clear_registry,
    get_model,
    list_models,
    models_by_layer,
    register_model,
)
from src.core.types import CalibrationResult, Forecast, RiskMetric, Signal


@pytest.fixture(autouse=True)
def _isolated_registry() -> None:
    clear_registry()


def _now() -> datetime:
    return datetime(2024, 1, 1, 9, 30)


class _DummyModel(BaseModel):
    name = "dummy"
    layer = 4
    refit_frequency = "daily"

    def fetch_data(self, provider: DataProvider) -> Any:
        return provider.fetch_prices("AAPL", "1y", "1d")

    def calibrate(self, data: Any) -> CalibrationResult:
        return CalibrationResult(
            model_name=self.name, parameters={"alpha": 0.1}, fit_metrics={"r2": 0.5},
            timestamp=_now(),
        )

    def predict(self, data: Any) -> Signal:
        return Signal(
            ticker="AAPL", direction="long", strength=0.7, timestamp=_now(), horizon="1d"
        )

    def validate(self, data: Any) -> dict[str, Any]:
        return {"ok": True}


class TestRegistry:
    def test_register_and_retrieve(self) -> None:
        register_model(_DummyModel)
        assert get_model("dummy") is _DummyModel
        assert "dummy" in list_models()

    def test_duplicate_name_raises(self) -> None:
        register_model(_DummyModel)

        class Conflict(BaseModel):
            name = "dummy"
            layer = 4
            refit_frequency = "daily"

            def fetch_data(self, provider: DataProvider) -> Any: ...
            def calibrate(self, data: Any) -> CalibrationResult:
                return CalibrationResult("dummy", {}, {}, _now())
            def predict(self, data: Any) -> Signal:
                return Signal("AAPL", "flat", 0.0, _now(), "1d")
            def validate(self, data: Any) -> dict[str, Any]:
                return {}

        with pytest.raises(ValueError, match="already registered"):
            register_model(Conflict)

    def test_re_registering_same_class_is_idempotent(self) -> None:
        register_model(_DummyModel)
        register_model(_DummyModel)
        assert list_models() == ["dummy"]

    def test_models_by_layer(self) -> None:
        register_model(_DummyModel)

        class L6(BaseModel):
            name = "risk_x"
            layer = 6
            refit_frequency = "weekly"

            def fetch_data(self, provider: DataProvider) -> Any: ...
            def calibrate(self, data: Any) -> CalibrationResult:
                return CalibrationResult("risk_x", {}, {}, _now())
            def predict(self, data: Any) -> RiskMetric:
                return RiskMetric("AAPL", "vol", 0.2, _now())
            def validate(self, data: Any) -> dict[str, Any]:
                return {}

        register_model(L6)
        assert {c.name for c in models_by_layer(4)} == {"dummy"}
        assert {c.name for c in models_by_layer(6)} == {"risk_x"}

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            get_model("nope")

    def test_register_rejects_non_basemodel(self) -> None:
        class NotAModel:
            name = "x"

        with pytest.raises(TypeError, match="BaseModel"):
            register_model(NotAModel)  # type: ignore[arg-type]


class TestBaseModelContract:
    def test_missing_class_attributes_raises_on_class_creation(self) -> None:
        with pytest.raises(TypeError, match="must define class attributes"):

            class Broken(BaseModel):
                # missing name / layer / refit_frequency
                def fetch_data(self, provider: DataProvider) -> Any: ...
                def calibrate(self, data: Any) -> CalibrationResult:
                    return CalibrationResult("x", {}, {}, _now())
                def predict(self, data: Any) -> Signal:
                    return Signal("AAPL", "flat", 0.0, _now(), "1d")
                def validate(self, data: Any) -> dict[str, Any]:
                    return {}

    def test_invalid_refit_frequency_rejected(self) -> None:
        with pytest.raises(ValueError, match="refit_frequency"):

            class BadFreq(BaseModel):
                name = "bad"
                layer = 4
                refit_frequency = "yearly"  # type: ignore[assignment]

                def fetch_data(self, provider: DataProvider) -> Any: ...
                def calibrate(self, data: Any) -> CalibrationResult:
                    return CalibrationResult("bad", {}, {}, _now())
                def predict(self, data: Any) -> Signal:
                    return Signal("AAPL", "flat", 0.0, _now(), "1d")
                def validate(self, data: Any) -> dict[str, Any]:
                    return {}

    def test_non_int_layer_rejected(self) -> None:
        with pytest.raises(TypeError, match="layer must be int"):

            class BadLayer(BaseModel):
                name = "bl"
                layer = "four"  # type: ignore[assignment]
                refit_frequency = "daily"

                def fetch_data(self, provider: DataProvider) -> Any: ...
                def calibrate(self, data: Any) -> CalibrationResult:
                    return CalibrationResult("bl", {}, {}, _now())
                def predict(self, data: Any) -> Signal:
                    return Signal("AAPL", "flat", 0.0, _now(), "1d")
                def validate(self, data: Any) -> dict[str, Any]:
                    return {}

    def test_abstract_methods_enforced(self) -> None:
        class Partial(BaseModel):
            name = "partial"
            layer = 4
            refit_frequency = "daily"

            def fetch_data(self, provider: DataProvider) -> Any: ...
            # missing calibrate/predict/validate

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates_and_runs(self) -> None:
        provider = InMemoryProvider(
            prices={("AAPL", "1y", "1d"): _fake_prices()},
        )
        m = _DummyModel()
        data = m.fetch_data(provider)
        cal = m.calibrate(data)
        sig = m.predict(data)
        diag = m.validate(data)

        assert isinstance(cal, CalibrationResult)
        assert isinstance(sig, Signal)
        assert diag == {"ok": True}


class TestSharedTypes:
    def test_signal_strength_bounded(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            Signal("AAPL", "long", 1.5, _now(), "1d")

    def test_forecast_interval_ordering(self) -> None:
        with pytest.raises(ValueError, match="lower"):
            Forecast(ticker="AAPL", horizon="1d", value=100.0, timestamp=_now(),
                     lower=110.0, upper=90.0)

    def test_signal_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        s = Signal("AAPL", "long", 0.5, _now(), "1d")
        with pytest.raises(FrozenInstanceError):
            s.strength = 0.9  # type: ignore[misc]


def _fake_prices() -> Any:
    import pandas as pd

    return pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]}
    )
