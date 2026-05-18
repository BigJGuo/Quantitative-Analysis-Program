"""End-to-end unit tests for `LiquidityAdjustedVaR` against `InMemoryProvider`."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.liquidity_adjusted_var.model import LiquidityAdjustedVaR
from src.models.liquidity_adjusted_var.types import LiquidityVaRInputs


@pytest.fixture
def positions_dict() -> dict[str, float]:
    return {"LARGE": 1_000_000.0, "MID": 500_000.0, "SMALL": 200_000.0}


@pytest.fixture
def model(positions_dict: dict[str, float], now: datetime) -> LiquidityAdjustedVaR:
    return LiquidityAdjustedVaR(
        positions_dict,
        alpha=0.975,
        max_participation=0.15,
        history_period="2y",
        now_func=lambda: now,
    )


def test_class_attributes() -> None:
    assert LiquidityAdjustedVaR.name == "liquidity_adjusted_var"
    assert LiquidityAdjustedVaR.layer == 6
    assert LiquidityAdjustedVaR.refit_frequency == "daily"


def test_init_rejects_empty_positions() -> None:
    with pytest.raises(ValueError):
        LiquidityAdjustedVaR({})


def test_init_rejects_alpha_out_of_range(positions_dict: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LiquidityAdjustedVaR(positions_dict, alpha=0.4)


def test_init_rejects_bad_max_participation(positions_dict: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LiquidityAdjustedVaR(positions_dict, max_participation=0.0)


def test_fetch_data_returns_inputs(
    model: LiquidityAdjustedVaR, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    assert isinstance(inputs, LiquidityVaRInputs)
    assert tuple(p.ticker for p in inputs.positions) == ("LARGE", "MID", "SMALL")
    assert tuple(s.ticker for s in inputs.ticker_stats) == ("LARGE", "MID", "SMALL")
    assert inputs.returns_panel.shape[1] == 3
    assert len(inputs.returns_panel) > 100


def test_calibrate_returns_calibration_result(
    model: LiquidityAdjustedVaR, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    cal = model.calibrate(inputs)
    assert isinstance(cal, CalibrationResult)
    assert cal.model_name == "liquidity_adjusted_var"
    assert cal.fit_metrics["n_positions"] == 3
    assert cal.fit_metrics["gross_exposure"] == pytest.approx(
        1_000_000.0 + 500_000.0 + 200_000.0
    )


def test_predict_returns_risk_metric(
    model: LiquidityAdjustedVaR, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    risk = model.predict(inputs)
    assert isinstance(risk, RiskMetric)
    assert risk.ticker == "PORTFOLIO"
    assert risk.metric_name == "liquidity_adjusted_var"
    assert risk.horizon == "1d"
    assert risk.confidence_level == pytest.approx(0.975)
    assert risk.value > 0.0
    md: dict[str, Any] = dict(risk.metadata)
    assert md["price_var_1d"] > 0.0
    assert md["es_frtb"] > 0.0
    assert md["liquidity_cost_spread"] > 0.0
    # L-VaR is the sum of FRTB ES and Bangia spread cost.
    assert md["l_var"] == pytest.approx(md["es_frtb"] + md["liquidity_cost_spread"])
    # Per-name diagnostics cover every position.
    assert set(md["per_name"].keys()) == {"LARGE", "MID", "SMALL"}


def test_validate_reports_full_diagnostic_block(
    model: LiquidityAdjustedVaR, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    diag = model.validate(inputs)
    # Spec validation block 3: horizon coverage check.
    assert "horizon_coverage_violations" in diag
    # Spec validation block 4: concentration ratio.
    assert "concentration_ratio" in diag
    # Spec validation block 5: stressed L-VaR with 5x spreads.
    assert diag["stress_multiplier"] == pytest.approx(5.0)
    assert diag["stressed_l_var"] >= diag["l_var"]
    # Stress uplift must be exactly 4x the base spread cost (5x - 1x).
    assert diag["stress_uplift"] == pytest.approx(4.0 * diag["liquidity_cost_spread"])
    # Other sanity fields.
    assert diag["n_positions"] == 3
    assert diag["gross_exposure"] == pytest.approx(1_700_000.0)


def test_predict_then_validate_consistent(
    model: LiquidityAdjustedVaR, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    risk = model.predict(inputs)
    diag = model.validate(inputs)
    assert diag["l_var"] == pytest.approx(risk.metadata["l_var"])
    assert diag["es_frtb"] == pytest.approx(risk.metadata["es_frtb"])


def test_short_position_uses_absolute_for_spread(
    in_memory_provider: InMemoryProvider, now: datetime
) -> None:
    """A net short of -500k in MID should still incur a positive spread cost."""

    model = LiquidityAdjustedVaR(
        {"LARGE": 1_000_000.0, "MID": -500_000.0, "SMALL": 200_000.0},
        now_func=lambda: now,
    )
    inputs = model.fetch_data(in_memory_provider)
    risk = model.predict(inputs)
    assert risk.metadata["liquidity_cost_spread"] > 0.0
    # Net exposure reflects the short.
    assert risk.metadata["net_exposure"] == pytest.approx(700_000.0)


def test_predict_type_check() -> None:
    model = LiquidityAdjustedVaR({"X": 1.0})
    with pytest.raises(TypeError):
        model.predict({"not": "inputs"})


def test_validate_type_check() -> None:
    model = LiquidityAdjustedVaR({"X": 1.0})
    with pytest.raises(TypeError):
        model.validate({"not": "inputs"})


def test_compute_helper_matches_predict(
    model: LiquidityAdjustedVaR, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    structured = model.compute(inputs)
    risk = model.predict(inputs)
    assert risk.metadata["l_var"] == pytest.approx(structured.l_var)
    assert risk.metadata["es_frtb"] == pytest.approx(structured.es_frtb)
    assert risk.metadata["price_var_1d"] == pytest.approx(structured.price_var_1d)
