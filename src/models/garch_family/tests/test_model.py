"""End-to-end unit tests for `GARCHModel` against `InMemoryProvider`.

Exercises the full BaseModel contract (`fetch_data` -> `calibrate` ->
`predict` -> `validate`) using deterministic synthetic GARCH inputs.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.garch_family.model import GARCHModel
from src.models.garch_family.types import GARCHFit, GARCHInputs


@pytest.fixture
def model(now: datetime) -> GARCHModel:
    return GARCHModel(
        ticker="SYN",
        spec="GARCH",
        distribution="Gaussian",
        mean_model="Zero",
        history_period="10y",
        now_func=lambda: now,
    )


def test_class_attributes() -> None:
    assert GARCHModel.name == "garch_family"
    assert GARCHModel.layer == 4
    assert GARCHModel.refit_frequency == "monthly"


def test_fetch_data_returns_inputs(
    model: GARCHModel, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    assert isinstance(inputs, GARCHInputs)
    assert inputs.ticker == "SYN"
    assert inputs.spec == "GARCH"
    assert inputs.distribution == "Gaussian"
    assert len(inputs.returns_pct) >= 1000


def test_calibrate_persists_fit_on_instance(
    model: GARCHModel, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    result = model.calibrate(inputs)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "garch_family"
    assert isinstance(model._fit, GARCHFit)


def test_predict_returns_risk_metric(
    model: GARCHModel, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    risk = model.predict(inputs)
    assert isinstance(risk, RiskMetric)
    assert risk.ticker == "SYN"
    assert risk.metric_name == "conditional_volatility"
    assert risk.horizon == "1d"
    # Annualized vol for synthetic GARCH(1,1) with omega=0.05, alpha=0.08,
    # beta=0.9, persistence 0.98: unconditional vol_pct ~ sqrt(2.5) ~ 1.58%,
    # annualized ~ 0.25. Allow a broad band.
    assert 0.05 < risk.value < 1.0
    # Metadata carries multi-horizon forecasts and parameters.
    assert "forecast_horizons" in risk.metadata
    assert "params" in risk.metadata
    assert "var_pct" in risk.metadata


def test_predict_lazy_calibrates_when_uncalibrated(
    in_memory_provider: InMemoryProvider, now: datetime
) -> None:
    fresh = GARCHModel(
        ticker="SYN",
        spec="GARCH",
        distribution="Gaussian",
        mean_model="Zero",
        now_func=lambda: now,
    )
    inputs = fresh.fetch_data(in_memory_provider)
    risk = fresh.predict(inputs)
    assert isinstance(risk, RiskMetric)
    assert fresh._fit is not None


def test_validate_reports_full_diagnostic_block(
    model: GARCHModel, in_memory_provider: InMemoryProvider
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    diag = model.validate(inputs)
    assert diag["ticker"] == "SYN"
    assert diag["converged"] is True
    # Spec validation 1 & 2: Ljung-Box on z and z^2 should not reject for a
    # well-specified fit on synthetic data.
    assert diag["ljung_box_z2_lag10_p"] > 0.01
    # Spec validation 3: ARCH-LM should not reject.
    assert diag["arch_lm_lag10_p"] > 0.01
    # Spec validation 4: sign-bias joint test should not reject for symmetric
    # GARCH(1,1) data.
    assert diag["sign_bias_joint_p"] > 0.01
    # VaR backtest: breach rate should be within ~3x of nominal alpha=0.05.
    assert 0.02 < diag["var_backtest_breach_rate"] < 0.10
    # Persistence < 1 for stationary GARCH.
    assert diag["persistence"] < 1.0


def test_forecast_horizons_match_configuration(
    in_memory_provider: InMemoryProvider, now: datetime
) -> None:
    horizons = (1, 3, 7)
    m = GARCHModel(
        ticker="SYN",
        spec="GARCH",
        distribution="Gaussian",
        mean_model="Zero",
        forecast_horizons=horizons,
        now_func=lambda: now,
    )
    inputs = m.fetch_data(in_memory_provider)
    m.calibrate(inputs)
    fc = m.forecast(inputs)
    assert fc.horizons == horizons
    assert fc.variances.shape == (3,)


def test_gjr_runs_end_to_end(
    in_memory_provider: InMemoryProvider, now: datetime
) -> None:
    m = GARCHModel(
        ticker="SYN",
        spec="GJR",
        distribution="Gaussian",
        mean_model="Zero",
        now_func=lambda: now,
    )
    inputs = m.fetch_data(in_memory_provider)
    m.calibrate(inputs)
    risk = m.predict(inputs)
    assert risk.metadata["spec"] == "GJR"


def test_egarch_runs_end_to_end(
    in_memory_provider: InMemoryProvider, now: datetime
) -> None:
    m = GARCHModel(
        ticker="SYN",
        spec="EGARCH",
        distribution="Gaussian",
        mean_model="Zero",
        now_func=lambda: now,
    )
    inputs = m.fetch_data(in_memory_provider)
    m.calibrate(inputs)
    risk = m.predict(inputs)
    assert risk.metadata["spec"] == "EGARCH"


def test_student_t_runs_end_to_end(
    in_memory_provider: InMemoryProvider, now: datetime
) -> None:
    m = GARCHModel(
        ticker="SYN",
        spec="GARCH",
        distribution="Student-t",
        mean_model="Constant",
        now_func=lambda: now,
    )
    inputs = m.fetch_data(in_memory_provider)
    m.calibrate(inputs)
    risk = m.predict(inputs)
    assert risk.metadata["distribution"] == "Student-t"
    # Student-t fits store nu.
    assert m._fit is not None
    assert m._fit.params.nu is not None


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        GARCHModel(ticker="")
    with pytest.raises(ValueError):
        GARCHModel(ticker="SYN", spec="ARCH")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GARCHModel(ticker="SYN", distribution="GED")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GARCHModel(ticker="SYN", forecast_horizons=())
    with pytest.raises(ValueError):
        GARCHModel(ticker="SYN", forecast_horizons=(0, 1))
    with pytest.raises(ValueError):
        GARCHModel(ticker="SYN", var_alpha=1.5)


def test_registry_contains_garch_family() -> None:
    from src.core.registry import list_models

    assert "garch_family" in list_models()


def test_fetch_data_raises_on_missing_close_data(now: datetime) -> None:
    import pandas as pd

    provider = InMemoryProvider(prices={("SYN", "10y", "1d"): pd.DataFrame()})
    m = GARCHModel(ticker="SYN", now_func=lambda: now)
    with pytest.raises(RuntimeError, match="No daily price history"):
        m.fetch_data(provider)


def test_fetch_data_raises_on_too_few_obs(now: datetime) -> None:
    import pandas as pd

    # 100 obs only — below the 250 floor.
    closes = pd.Series(range(100, 200), dtype=float)
    bars = pd.DataFrame(
        {"Date": pd.date_range("2024-01-01", periods=100), "Close": closes.to_numpy()}
    )
    provider = InMemoryProvider(prices={("SYN", "10y", "1d"): bars})
    m = GARCHModel(ticker="SYN", now_func=lambda: now)
    with pytest.raises(RuntimeError, match=">= 250"):
        m.fetch_data(provider)
