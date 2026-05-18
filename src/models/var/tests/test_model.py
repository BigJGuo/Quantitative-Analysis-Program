"""End-to-end unit tests for `VaRModel` against `InMemoryProvider`.

Exercises the full BaseModel contract (`fetch_data` -> `calibrate` ->
`predict` -> `validate`) using deterministic synthetic Gaussian
two-asset inputs. The diagnostic test corresponds to the spec's
"Validation and diagnostics" section.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from typing import cast

import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.var.model import VaRModel
from src.models.var.types import VaRFit, VaRInputs

ModelFactory = Callable[..., VaRModel]


@pytest.fixture
def model_factory(now: datetime) -> ModelFactory:
    """Factory for fresh `VaRModel` instances pinned to the test clock."""

    def _factory(
        method: str = "historical",
        alpha: float = 0.99,
        positions: dict[str, float] | None = None,
        backtest_window: int = 250,
        lookback_days: int = 500,
    ) -> VaRModel:
        positions = positions or {"SYN1": 1_000_000.0, "SYN2": 500_000.0}
        return VaRModel(
            positions=positions,
            method=method,  # type: ignore[arg-type]
            alpha=alpha,
            lookback_days=lookback_days,
            horizon_days=10,
            history_period="3y",
            backtest_window=backtest_window,
            now_func=lambda: now,
            mc_seed=42,
            mc_samples=5_000,
        )

    return _factory


class TestClassAttributes:
    def test_name_layer_refit(self) -> None:
        assert VaRModel.name == "var"
        assert VaRModel.layer == 6
        assert VaRModel.refit_frequency == "daily"


class TestFetchData:
    def test_returns_inputs(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        model = model_factory()
        inputs = model.fetch_data(in_memory_provider)
        assert isinstance(inputs, VaRInputs)
        assert list(inputs.positions.keys()) == ["SYN1", "SYN2"]
        assert list(inputs.returns.columns) == ["SYN1", "SYN2"]
        assert len(inputs.returns) >= 500

    def test_rejects_unavailable_ticker(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        # MISSING is not in the provider — fetch raises.
        model = model_factory(positions={"SYN1": 100.0, "MISSING": 100.0})
        with pytest.raises((RuntimeError, KeyError)):
            model.fetch_data(in_memory_provider)


class TestCalibrate:
    def test_persists_fit_on_instance(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        model = model_factory(method="historical")
        inputs = model.fetch_data(in_memory_provider)
        result = model.calibrate(inputs)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "var"
        assert isinstance(model._fit, VaRFit)

    def test_calibrate_rejects_wrong_inputs(self, model_factory: ModelFactory) -> None:
        model = model_factory()
        with pytest.raises(TypeError):
            model.calibrate({"not": "inputs"})


class TestPredict:
    @pytest.mark.parametrize("method", ["parametric", "historical", "monte_carlo"])
    def test_returns_risk_metric(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider, method: str
    ) -> None:
        model = model_factory(method=method)
        inputs = model.fetch_data(in_memory_provider)
        risk = model.predict(inputs)
        assert isinstance(risk, RiskMetric)
        assert risk.ticker == "PORTFOLIO"
        assert risk.metric_name == "value_at_risk"
        assert risk.horizon == "1d"
        assert risk.confidence_level == 0.99
        assert risk.value > 0
        # SYN1 has sigma=1%, SYN2 has sigma=2%, w=(1M, 0.5M).
        # sigma_p = sqrt(1M^2 * 1e-4 + 0.5M^2 * 4e-4) = sqrt(2e8) ~ 14,142
        # parametric VaR@99 ~ 32,900. Allow a broad band for all methods.
        assert 10_000 < risk.value < 80_000
        assert risk.metadata["var_horizon_dollar"] == pytest.approx(
            risk.value * math.sqrt(10), rel=1e-9
        )
        assert risk.metadata["method"] == method

    def test_predict_lazy_calibrates(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        model = model_factory(method="historical")
        inputs = model.fetch_data(in_memory_provider)
        risk = model.predict(inputs)
        assert isinstance(risk, RiskMetric)
        assert model._fit is not None


class TestValidate:
    """Diagnostic test corresponding to the spec's "Validation and
    diagnostics" section: Kupiec POF, Christoffersen independence, Basel
    traffic-light.
    """

    def test_validate_emits_backtest_block(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        model = model_factory(
            method="historical", alpha=0.99, backtest_window=250, lookback_days=500
        )
        inputs = model.fetch_data(in_memory_provider)
        model.calibrate(inputs)
        diag = model.validate(inputs)
        # All spec diagnostic keys present.
        for k in (
            "method",
            "alpha",
            "var_1d_dollar",
            "backtest_n",
            "backtest_breaches",
            "backtest_breach_rate",
            "backtest_expected_breaches",
            "kupiec_lr",
            "kupiec_p",
            "christoffersen_lr",
            "christoffersen_p",
            "traffic_light",
            "var_path",
            "loss_path",
        ):
            assert k in diag, k
        assert diag["traffic_light"] in {"green", "yellow", "red"}
        # Under Gaussian iid simulated returns at alpha=0.99 the long-run
        # breach rate is ~1%; on ~500 obs that's roughly 5 breaches with
        # binomial uncertainty. Allow a generous band.
        assert 0.0 <= cast(float, diag["backtest_breach_rate"]) <= 0.10

    def test_validate_kupiec_passes_for_well_calibrated_method(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        """Under Gaussian iid data, HS-VaR should pass the Kupiec POF test."""

        model = model_factory(method="historical", alpha=0.95, backtest_window=250)
        inputs = model.fetch_data(in_memory_provider)
        model.calibrate(inputs)
        diag = model.validate(inputs)
        # p-value should be above any reasonable rejection level. The data
        # is mild Gaussian iid so the test should not reject.
        assert cast(float, diag["kupiec_p"]) > 0.01

    def test_validate_rejects_wrong_input_type(self, model_factory: ModelFactory) -> None:
        model = model_factory()
        with pytest.raises(TypeError):
            model.validate({"foo": "bar"})


class TestHorizonHelper:
    def test_horizon_var_requires_calibration(self, model_factory: ModelFactory) -> None:
        model = model_factory()
        with pytest.raises(RuntimeError, match="calibrate"):
            model.horizon_var(10)

    def test_horizon_var_scales_with_sqrt_h(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        model = model_factory(method="historical")
        inputs = model.fetch_data(in_memory_provider)
        model.calibrate(inputs)
        v_1 = model.horizon_var(1)
        v_10 = model.horizon_var(10)
        assert v_10 == pytest.approx(v_1 * math.sqrt(10), rel=1e-12)


class TestConstructorValidation:
    def test_empty_positions_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            VaRModel(positions={})

    def test_invalid_alpha_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            VaRModel(positions={"A": 1.0}, alpha=1.5)

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="method"):
            VaRModel(positions={"A": 1.0}, method="bogus")  # type: ignore[arg-type]

    def test_non_finite_position_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            VaRModel(positions={"A": float("nan")})

    def test_tickers_uppercased(self) -> None:
        model = VaRModel(positions={"spy": 100.0})
        assert "SPY" in model.positions
        assert "spy" not in model.positions


class TestPredictMetadata:
    def test_var_pct_gross_when_positive_exposure(
        self, model_factory: ModelFactory, in_memory_provider: InMemoryProvider
    ) -> None:
        model = model_factory(method="historical")
        inputs = model.fetch_data(in_memory_provider)
        risk = model.predict(inputs)
        # gross = 1_500_000; var_1d_pct_gross should be in (0, 1).
        assert "var_1d_pct_gross" in risk.metadata
        pct = cast(float, risk.metadata["var_1d_pct_gross"])
        assert 0.0 < pct < 0.1  # well-behaved synthetic data
