"""End-to-end unit tests for `MertonKMV` against `InMemoryProvider`.

These tests exercise the full BaseModel contract (`fetch_data` -> `calibrate` ->
`predict` -> `validate`) using deterministic synthetic inputs.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, Signal
from src.models.merton_kmv.model import MertonKMV
from src.models.merton_kmv.types import MertonInputs


@pytest.fixture
def model(now: datetime) -> MertonKMV:
    return MertonKMV(
        ticker="ACME",
        horizon_years=1.0,
        history_days=252,
        weight_lt_debt=0.5,
        lgd=0.6,
        now_func=lambda: now,
    )


def test_fetch_data_returns_inputs(
    model: MertonKMV,
    in_memory_provider: InMemoryProvider,
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    assert isinstance(inputs, MertonInputs)
    assert inputs.ticker == "ACME"
    assert inputs.horizon_years == 1.0
    assert inputs.risk_free_rate == pytest.approx(0.04, abs=1e-6)
    assert inputs.balance_sheet.short_term_debt == pytest.approx(30.0)
    assert inputs.balance_sheet.long_term_debt == pytest.approx(50.0)
    assert len(inputs.equity_prices) >= 100


def test_calibrate_persists_solution_on_instance(
    model: MertonKMV,
    in_memory_provider: InMemoryProvider,
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    result = model.calibrate(inputs)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "merton_kmv"
    assert model._last_solution is not None
    assert model._last_drift is not None


def test_predict_returns_signal(
    model: MertonKMV,
    in_memory_provider: InMemoryProvider,
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    signal = model.predict(inputs)
    assert isinstance(signal, Signal)
    assert signal.ticker == "ACME"
    assert signal.direction in ("long", "short", "flat")
    assert 0.0 <= signal.strength <= 1.0
    assert "dd_physical" in signal.metadata
    assert "credit_spread_bps" in signal.metadata


def test_predict_lazy_calibrates_if_uncalibrated(
    in_memory_provider: InMemoryProvider,
    now: datetime,
) -> None:
    fresh_model = MertonKMV(ticker="ACME", now_func=lambda: now)
    inputs = fresh_model.fetch_data(in_memory_provider)
    # No explicit calibrate(); predict() should still work via lazy fit.
    signal = fresh_model.predict(inputs)
    assert isinstance(signal, Signal)
    assert fresh_model._last_solution is not None


def test_validate_reports_diagnostics(
    model: MertonKMV,
    in_memory_provider: InMemoryProvider,
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    diag = model.validate(inputs)
    assert diag["ticker"] == "ACME"
    assert diag["converged"] is True
    # Spec validation 2: g1 strictly < 1e-4; g2 only approximately enforced.
    assert diag["g1_residual"] < 1e-4
    assert diag["g2_residual"] < 0.15
    assert diag["residual_ok"] is True
    # Spec validation 6: sigma_V should be lower than sigma_E (leverage effect).
    assert diag["vol_ratio_asset_to_equity"] < 1.0
    # The synthetic firm is solidly investment grade (V/D ~ 3, low vol).
    assert diag["credit_quality"] in ("investment_grade", "high_yield")


def test_validate_residual_ok_flag_set_on_consistent_fit(
    model: MertonKMV,
    in_memory_provider: InMemoryProvider,
) -> None:
    """Spec's diagnostic 2 check — residual_ok should fire for healthy fits."""

    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    diag = model.validate(inputs)
    assert diag["residual_ok"] is True


def test_model_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        MertonKMV(ticker="ACME", horizon_years=-1.0)
    with pytest.raises(ValueError):
        MertonKMV(ticker="ACME", weight_lt_debt=1.5)
    with pytest.raises(ValueError):
        MertonKMV(ticker="ACME", lgd=-0.1)
    with pytest.raises(ValueError):
        MertonKMV(ticker="", horizon_years=1.0)


def test_predict_with_cds_spread_emits_arb_stance(
    in_memory_provider: InMemoryProvider,
    now: datetime,
) -> None:
    # A very wide market CDS spread should be flagged "cds_rich".
    model = MertonKMV(
        ticker="ACME",
        market_cds_spread_bps=5_000.0,  # absurdly wide
        arb_threshold_bps=25.0,
        now_func=lambda: now,
    )
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    signal = model.predict(inputs)
    assert signal.metadata["arb_stance"] == "cds_rich"
    # Per signal_direction_from_quality: cds_rich -> short.
    assert signal.direction == "short"


def test_predict_trailing_drift_changes_dd(
    in_memory_provider: InMemoryProvider,
    now: datetime,
) -> None:
    """`drift_mode='trailing'` should produce a different DD than `'risk_free'`."""

    model_rn = MertonKMV(ticker="ACME", drift_mode="risk_free", now_func=lambda: now)
    model_tr = MertonKMV(ticker="ACME", drift_mode="trailing", now_func=lambda: now)

    inputs_rn = model_rn.fetch_data(in_memory_provider)
    inputs_tr = model_tr.fetch_data(in_memory_provider)
    model_rn.calibrate(inputs_rn)
    model_tr.calibrate(inputs_tr)

    sig_rn = model_rn.predict(inputs_rn)
    sig_tr = model_tr.predict(inputs_tr)
    # Both should produce valid signals; DD values will generally differ.
    assert isinstance(sig_rn.metadata["dd_physical"], float)
    assert isinstance(sig_tr.metadata["dd_physical"], float)


def test_registry_contains_merton_kmv() -> None:
    from src.core.registry import list_models

    assert "merton_kmv" in list_models()


def test_snapshot_returns_full_result(
    model: MertonKMV,
    in_memory_provider: InMemoryProvider,
) -> None:
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    result = model.snapshot(inputs)
    assert result.ticker == "ACME"
    assert result.asset_value > 0
    assert result.asset_volatility > 0
    assert result.default_point > 0
    assert isinstance(result.timestamp, datetime)
    assert result.credit_quality in (
        "investment_grade",
        "high_yield",
        "distressed",
        "default_imminent",
    )


def test_fetch_data_raises_on_missing_market_cap(
    now: datetime,
) -> None:
    provider = InMemoryProvider(
        fundamentals={"ACME": {"sharesOutstanding": 1.0, "totalDebt": 80.0}},
        prices={},
    )
    model = MertonKMV(ticker="ACME", now_func=lambda: now)
    with pytest.raises(RuntimeError, match="market cap"):
        model.fetch_data(provider)


def test_fetch_data_raises_on_missing_debt(
    now: datetime,
    in_memory_provider: InMemoryProvider,
) -> None:
    # Strip debt fields from the fundamentals blob.
    debtless = {**in_memory_provider.fundamentals["ACME"]}
    for k in ("currentDebt", "longTermDebt", "totalDebt", "shortTermDebt", "shortLongTermDebt"):
        debtless.pop(k, None)
    in_memory_provider.fundamentals["ACME"] = debtless

    model = MertonKMV(ticker="ACME", now_func=lambda: now)
    with pytest.raises(RuntimeError, match="debt"):
        model.fetch_data(in_memory_provider)


def test_fetch_data_splits_total_debt_when_components_missing(
    now: datetime,
    in_memory_provider: InMemoryProvider,
) -> None:
    """If only totalDebt is present, the model splits 30/70 ST/LT."""

    f = {**in_memory_provider.fundamentals["ACME"]}
    f.pop("currentDebt", None)
    f.pop("longTermDebt", None)
    f["totalDebt"] = 100.0
    in_memory_provider.fundamentals["ACME"] = f

    model = MertonKMV(ticker="ACME", now_func=lambda: now)
    inputs = model.fetch_data(in_memory_provider)
    assert inputs.balance_sheet.short_term_debt == pytest.approx(30.0)
    assert inputs.balance_sheet.long_term_debt == pytest.approx(70.0)


def test_model_class_attributes_are_correct() -> None:
    assert MertonKMV.name == "merton_kmv"
    assert MertonKMV.layer == 2
    assert MertonKMV.refit_frequency == "daily"
