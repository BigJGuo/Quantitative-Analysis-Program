"""End-to-end integration test for `FuturesImpliedNAV` against live yfinance.

Skipped automatically when yfinance is unreachable (offline CI agent) so the
suite does not produce spurious red builds. When connectivity is available,
this exercises the full DataProvider -> fetch_data -> calibrate -> predict ->
validate pipeline on a representative international ETF (EWJ).

Path-based selection: run only this directory with
`pytest tests/integration/`. Reside it outside `tests/integration/` to skip.
"""

from __future__ import annotations

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, Signal
from src.models.futures_implied_nav import (
    CostParameters,
    FuturesImpliedNAV,
    HedgeSpec,
)


def _skip_if_offline(provider: YFinanceProvider, ticker: str) -> None:
    try:
        df = provider.fetch_prices(ticker, period="5d", interval="1d")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance unreachable: {exc}")
    if df is None or len(df) == 0:
        pytest.skip(f"yfinance returned no data for {ticker}")


def test_endtoend_ewj_pipeline() -> None:
    provider = YFinanceProvider()
    _skip_if_offline(provider, "EWJ")

    model = FuturesImpliedNAV(
        etf_ticker="EWJ",
        hedges=(
            HedgeSpec(ticker="^N225", kind="futures", currency="JPY"),
            HedgeSpec(ticker="EWY", kind="equity", currency="USD"),
        ),
        fx_tickers=("JPYUSD=X",),
        history_days=90,
        cost_params=CostParameters(half_spread_bps=2.0, cost_band_bps=5.0),
    )

    data = model.fetch_data(provider)
    assert data.etf_ticker == "EWJ"
    assert data.etf_mid > 0
    assert "^N225" in data.daily_history
    assert "JPYUSD=X" in data.daily_history

    cal_result = model.calibrate(data)
    assert isinstance(cal_result, CalibrationResult)
    assert cal_result.model_name == "futures_implied_nav"
    beta = cal_result.parameters["beta"]
    # Sanity: betas should sit in the constraint box.
    assert all(0.0 <= b <= 1.5 for b in beta.betas)
    # Real EWJ ~ Nikkei beta is ~0.85-1.0; the constrained ridge should reflect that.
    assert sum(beta.betas) <= 1.2 + 1e-6

    signal = model.predict(data)
    assert isinstance(signal, Signal)
    assert signal.ticker == "EWJ"
    assert signal.direction in {"long", "short", "flat"}
    assert 0.0 <= signal.strength <= 1.0
    fair_value = signal.metadata["fair_value"]
    assert fair_value > 0
    premium = signal.metadata["premium"]
    # The model is not allowed to produce nonsense: |premium| under 5% is a sanity bound.
    assert abs(premium) < 0.05

    diagnostics = model.validate(data)
    assert diagnostics["status"] == "ok"
    assert diagnostics["n_obs"] >= 30
    # On real EWJ data the calibration R^2 should be respectable, but we don't
    # want the integration test to be brittle to market regime — just sanity check.
    assert 0.0 <= diagnostics["r_squared"] <= 1.0
