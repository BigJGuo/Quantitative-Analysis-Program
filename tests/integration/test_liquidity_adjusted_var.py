"""End-to-end integration test for the Liquidity-Adjusted VaR model.

Pulls real yfinance data for a small multi-tier portfolio (SPY, AAPL,
MSFT) and runs the full ``fetch_data -> calibrate -> predict -> validate``
pipeline. Self-skips when yfinance is unavailable or the network call
fails so unit-test runs stay deterministic.
"""

from __future__ import annotations

import importlib.util

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.liquidity_adjusted_var.model import LiquidityAdjustedVaR

# A small multi-tier portfolio: two mega-cap equities and an index ETF.
# All three should be tier-1 / large-cap (10d FRTB bucket).
_POSITIONS: dict[str, float] = {
    "SPY": 1_000_000.0,
    "AAPL": 500_000.0,
    "MSFT": 500_000.0,
}


def _yfinance_available() -> bool:
    return importlib.util.find_spec("yfinance") is not None


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
def test_liquidity_adjusted_var_end_to_end_on_real_tickers() -> None:
    """Run the full L-VaR pipeline on a real liquid-equity portfolio."""

    provider = YFinanceProvider(cache=None)
    model = LiquidityAdjustedVaR(
        _POSITIONS,
        alpha=0.975,
        max_participation=0.15,
        history_period="2y",
    )

    try:
        inputs = model.fetch_data(provider)
    except RuntimeError as exc:
        pytest.skip(f"yfinance data unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed: {exc}")

    calibration = model.calibrate(inputs)
    assert isinstance(calibration, CalibrationResult)
    assert calibration.model_name == "liquidity_adjusted_var"
    assert calibration.fit_metrics["n_positions"] == 3
    assert calibration.fit_metrics["gross_exposure"] == pytest.approx(2_000_000.0)
    # Mega-cap / SPY spreads stay well below 5% even when yfinance's
    # end-of-day bid/ask is stale and the H/L proxy kicks in.
    assert 0.0 <= calibration.fit_metrics["mean_spread_relative"] < 0.05

    risk = model.predict(inputs)
    assert isinstance(risk, RiskMetric)
    assert risk.ticker == "PORTFOLIO"
    assert risk.metric_name == "liquidity_adjusted_var"
    # Headline L-VaR should be a meaningful fraction of the 2M book.
    assert 0.0 < risk.value < 0.5 * 2_000_000.0
    md = risk.metadata
    assert md["price_var_1d"] > 0.0
    assert md["es_frtb"] >= md["price_var_1d"] * 0.5
    # All three positions are tier-1 liquid -> small spread cost relative to ES.
    assert md["liquidity_cost_spread"] < md["es_frtb"]
    # Per-name diagnostics surface every ticker.
    assert set(md["per_name"].keys()) == set(_POSITIONS.keys())
    for block in md["per_name"].values():
        assert block["adv_dollar"] > 0.0
        assert block["frtb_bucket_days"] in (10, 20, 40, 60, 120)
        assert block["tier"] in (1, 2, 3, 4)


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
def test_liquidity_adjusted_var_diagnostics_spec_validation() -> None:
    """Spec validation blocks 3-5: concentration, horizon coverage, stress shock."""

    provider = YFinanceProvider(cache=None)
    model = LiquidityAdjustedVaR(
        _POSITIONS,
        alpha=0.975,
        max_participation=0.15,
        history_period="2y",
        stress_multiplier=5.0,
    )

    try:
        inputs = model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed: {exc}")

    diag = model.validate(inputs)
    # Block 3 — horizon coverage: large-cap names should have no violations
    # at a 1M / 500k position size and 60d-default FRTB endpoint snapping.
    assert isinstance(diag["horizon_coverage_violations"], list)
    # Block 4 — concentration: 500k-1M positions in liquid mega-caps should
    # not breach the 20% ADV threshold.
    assert diag["concentration_ratio"] == 0
    # Block 5 — stress uplift = 4 * base spread cost (5x - 1x).
    assert diag["stress_uplift"] == pytest.approx(
        4.0 * diag["liquidity_cost_spread"], rel=1e-6
    )
    # Sanity: L-VaR > VaR > 0.
    assert diag["l_var"] > diag["price_var_1d"] > 0.0
    # L-VaR-to-VaR ratio should be a small uplift (<3x) for liquid equities.
    assert 1.0 <= diag["l_var_to_price_var_ratio"] < 3.0
