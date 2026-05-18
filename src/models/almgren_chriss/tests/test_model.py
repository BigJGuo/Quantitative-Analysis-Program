"""Tests for the `AlmgrenChriss` BaseModel orchestration shell.

Uses an `InMemoryProvider` so the tests stay deterministic and offline.
The yfinance end-to-end check lives in `tests/integration/`.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.almgren_chriss.model import AlmgrenChriss
from src.models.almgren_chriss.types import (
    AlmgrenChrissInputs,
    ExecutionSchedule,
)


def _make_provider(n_days: int = 60, ticker: str = "TEST") -> InMemoryProvider:
    rng = np.random.default_rng(seed=11)
    sigma_annual = 0.20
    sigma_daily = sigma_annual / math.sqrt(252)
    rets = rng.normal(loc=0.0, scale=sigma_daily, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "Volume": rng.lognormal(mean=math.log(5.0e7), sigma=0.1, size=n_days),
        }
    )
    # Build a synthetic 5m intraday frame (5 days × 78 bars).
    intraday_rows = []
    for d in range(5):
        for b in range(78):
            intraday_rows.append(
                {
                    "Datetime": pd.Timestamp("2024-04-01")
                    + pd.Timedelta(days=d)
                    + pd.Timedelta(minutes=5 * b),
                    "Volume": 5.0e5 + 1.0e5 * math.cos(b / 78.0 * math.pi),
                }
            )
    intraday = pd.DataFrame(intraday_rows)
    return InMemoryProvider(
        prices={(ticker, "60d", "1d"): df},
        intraday={(ticker, "5m", 30): intraday},
    )


def test_model_class_attributes() -> None:
    assert AlmgrenChriss.name == "almgren_chriss"
    assert AlmgrenChriss.layer == 5
    assert AlmgrenChriss.refit_frequency == "daily"


def test_model_constructor_validation() -> None:
    with pytest.raises(ValueError, match="ticker"):
        AlmgrenChriss(ticker="", X=1.0, T=1.0, N=10)
    with pytest.raises(ValueError, match="X"):
        AlmgrenChriss(ticker="A", X=0.0, T=1.0, N=10)
    with pytest.raises(ValueError, match="T"):
        AlmgrenChriss(ticker="A", X=1.0, T=0.0, N=10)
    with pytest.raises(ValueError, match="N"):
        AlmgrenChriss(ticker="A", X=1.0, T=1.0, N=0)
    with pytest.raises(ValueError, match="lam"):
        AlmgrenChriss(ticker="A", X=1.0, T=1.0, N=10, lam=0.0)
    with pytest.raises(ValueError, match="side"):
        AlmgrenChriss(
            ticker="A", X=1.0, T=1.0, N=10, side="hold"  # type: ignore[arg-type]
        )


def test_model_fetch_data_assembles_inputs() -> None:
    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    assert isinstance(inputs, AlmgrenChrissInputs)
    assert inputs.ticker == "TEST"
    assert not inputs.close.empty
    assert not inputs.volume.empty
    assert inputs.intraday_volume_profile is not None
    assert inputs.intraday_volume_profile.shape == (78,)
    assert inputs.intraday_volume_profile.sum() == pytest.approx(1.0)
    assert inputs.problem.X == 1.0e5
    assert inputs.problem.N == 20


def test_model_fetch_data_raises_on_empty_history() -> None:
    provider = InMemoryProvider(prices={("X", "60d", "1d"): pd.DataFrame()})
    model = AlmgrenChriss(ticker="X", X=1.0, T=1.0, N=10)
    with pytest.raises(RuntimeError, match="No daily history"):
        model.fetch_data(provider)


def test_model_calibrate_and_predict_flow() -> None:
    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    calib = model.calibrate(inputs)
    assert isinstance(calib, CalibrationResult)
    assert calib.model_name == "almgren_chriss"

    risk = model.predict(inputs)
    assert isinstance(risk, RiskMetric)
    assert risk.ticker == "TEST"
    assert risk.metric_name == "implementation_shortfall_bps"
    assert risk.value > 0.0
    # Metadata: schedule arrays consistent.
    inv = risk.metadata["inventory"]
    n_orders = risk.metadata["child_orders"]
    assert len(inv) == 21
    assert len(n_orders) == 20
    assert abs(sum(n_orders) - 1.0e5) < 1.0e-6


def test_model_validate_returns_diagnostics() -> None:
    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    model.calibrate(inputs)
    diag = model.validate(inputs)
    assert diag["monotone_inventory"] is True
    assert diag["positive_children"] is True
    assert diag["sum_n_minus_X_relative"] < 1.0e-9
    # Stress test should accelerate.
    assert diag["stressed_kappa_T"] > diag["kappa_T"]
    # Frontier sanity.
    assert diag["frontier_monotone"] is True
    # Numerical-stability flag false for sensible params.
    assert diag["numerical_stability_warning"] is False


def test_model_schedule_helper_returns_execution_schedule() -> None:
    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    model.calibrate(inputs)
    sched = model.schedule(inputs)
    assert isinstance(sched, ExecutionSchedule)
    assert sched.X == 1.0e5
    assert sched.N == 20


def test_model_vwap_schedule_uses_profile() -> None:
    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    model.calibrate(inputs)
    sched = model.vwap_schedule(inputs)
    assert isinstance(sched, ExecutionSchedule)
    assert sched.inventory[0] == pytest.approx(1.0e5)
    assert sched.inventory[-1] == pytest.approx(0.0)


def test_model_with_problem_swaps_parameters() -> None:
    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    new_inputs = model.with_problem(inputs, X=2.0e5, lam=1.0e-5)
    assert new_inputs.problem.X == 2.0e5
    assert new_inputs.problem.lam == 1.0e-5
    # Other fields preserved.
    assert new_inputs.problem.T == inputs.problem.T
    assert new_inputs.problem.N == inputs.problem.N


def test_model_predict_uses_default_calibration_lazily() -> None:
    """Calling predict before calibrate should auto-calibrate, not crash."""

    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    risk = model.predict(inputs)
    assert isinstance(risk, RiskMetric)


def test_model_horizon_string_formats() -> None:
    m1 = AlmgrenChriss(ticker="A", X=1.0, T=2.0, N=10)
    assert m1._horizon_string().endswith("d")
    m2 = AlmgrenChriss(ticker="A", X=1.0, T=0.5, N=10)
    assert m2._horizon_string().endswith("h")
    m3 = AlmgrenChriss(ticker="A", X=1.0, T=0.01, N=10)
    assert m3._horizon_string().endswith("m")


def test_model_type_errors_on_wrong_input_kind() -> None:
    model = AlmgrenChriss(ticker="A", X=1.0, T=1.0, N=10)
    with pytest.raises(TypeError, match="AlmgrenChrissInputs"):
        model.calibrate({"foo": "bar"})
    with pytest.raises(TypeError, match="AlmgrenChrissInputs"):
        model.predict({"foo": "bar"})
    with pytest.raises(TypeError, match="AlmgrenChrissInputs"):
        model.validate({"foo": "bar"})


def test_model_frontier_helper() -> None:
    provider = _make_provider()
    model = AlmgrenChriss(ticker="TEST", X=1.0e5, T=1.0, N=20)
    inputs = model.fetch_data(provider)
    model.calibrate(inputs)
    front = model.frontier(inputs)
    assert len(front.points) == 7
    kts = front.kappa_T_values()
    assert np.all(np.diff(kts) > 0)


@pytest.fixture
def now_at_fixed() -> datetime:
    return datetime(2024, 5, 1, tzinfo=UTC)
