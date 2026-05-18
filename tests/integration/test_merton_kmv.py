"""End-to-end integration test for Merton-KMV.

Pulls real yfinance data for a representative ticker and runs the full
`fetch_data -> calibrate -> predict -> validate` pipeline. The whole module
self-skips when yfinance is unavailable or the network call fails, so
unit-test runs stay deterministic.
"""

from __future__ import annotations

import importlib.util

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, Signal
from src.models.merton_kmv.model import MertonKMV

_TICKER: str = "GS"  # Goldman Sachs — large, well-instrumented, has debt fields


def _yfinance_available() -> bool:
    return importlib.util.find_spec("yfinance") is not None


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
def test_merton_kmv_end_to_end_on_real_ticker() -> None:
    """Run the model against a real ticker. Tolerates network/data outages."""

    provider = YFinanceProvider(cache=None)
    model = MertonKMV(
        ticker=_TICKER,
        horizon_years=1.0,
        history_days=252,
        weight_lt_debt=0.5,
        lgd=0.6,
    )

    try:
        inputs = model.fetch_data(provider)
    except RuntimeError as exc:
        pytest.skip(f"yfinance data unavailable for {_TICKER}: {exc}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed for {_TICKER}: {exc}")

    calibration = model.calibrate(inputs)
    assert isinstance(calibration, CalibrationResult)
    assert calibration.model_name == "merton_kmv"

    signal = model.predict(inputs)
    assert isinstance(signal, Signal)
    assert signal.ticker == _TICKER
    assert signal.direction in ("long", "short", "flat")
    assert 0.0 <= signal.strength <= 1.0

    diag = model.validate(inputs)
    # The iteration should have terminated cleanly; deeply distressed names
    # might fail this, but GS-class IG firms always converge.
    assert diag["converged"] is True
    # g1 strictly enforced by the BS inversion.
    assert diag["g1_residual"] < 1e-3
    # g2 only approximately enforced by the iterative procedure — real-data
    # firms can show 5-15% residuals here because sigma_E is taken as observed.
    assert diag["g2_residual"] < 0.30
    # IG-class firms typically have DD between 3 and 6 (spec validation 7).
    assert diag["dd_physical"] > 0.0
    # PD should be a probability.
    assert 0.0 <= diag["pd_physical"] <= 1.0
    assert 0.0 <= diag["pd_risk_neutral"] <= 1.0
    # Credit spread should be non-negative and finite.
    assert diag["credit_spread_bps"] >= 0.0
    assert diag["credit_spread_bps"] < 1e6


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
def test_merton_kmv_diagnostics_match_spec_thresholds() -> None:
    """Spec validation section: residuals at convergence and vol-ratio sanity.

    Spec validation 6 says sigma_V should typically be lower than sigma_E for a
    levered firm. For a financial-services name (GS), this still holds; for
    qualitatively-different non-financials the spec also caveats — we use the
    cleaner GS for the assertion.
    """

    provider = YFinanceProvider(cache=None)
    model = MertonKMV(ticker=_TICKER)
    try:
        inputs = model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed for {_TICKER}: {exc}")

    model.calibrate(inputs)
    diag = model.validate(inputs)

    # Spec validation 2: g1 strict, g2 loose for iterative-KMV.
    assert diag["g1_residual"] < 1e-3
    assert diag["g2_residual"] < 0.30
    # Spec validation 6 (sigma_V < sigma_E for leveraged firms).
    assert diag["vol_ratio_asset_to_equity"] < 1.0
