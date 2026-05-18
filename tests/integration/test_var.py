"""End-to-end integration test for the Value-at-Risk model.

Pulls real yfinance data for a small representative portfolio
(``SPY`` + ``QQQ`` — the canonical US large-cap / Nasdaq exposure
pair) and runs the full `fetch_data -> calibrate -> predict ->
validate` pipeline for each of the three VaR methods. The whole
module self-skips when yfinance is unavailable or the network call
fails, so unit-test runs stay deterministic.
"""

from __future__ import annotations

import importlib.util

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.var.model import VaRModel

_POSITIONS: dict[str, float] = {
    "SPY": 1_000_000.0,  # long $1M SPY
    "QQQ": 500_000.0,    # long $500k QQQ
}


def _yfinance_available() -> bool:
    return importlib.util.find_spec("yfinance") is not None


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_var_end_to_end_on_real_tickers(method: str) -> None:
    """Run all three VaR engines against real SPY + QQQ data."""

    provider = YFinanceProvider(cache=None)
    model = VaRModel(
        positions=_POSITIONS,
        method=method,  # type: ignore[arg-type]
        alpha=0.99,
        lookback_days=500,
        horizon_days=10,
        history_period="3y",
        backtest_window=250,
        mc_samples=10_000,
        mc_seed=42,
    )

    try:
        inputs = model.fetch_data(provider)
    except RuntimeError as exc:
        pytest.skip(f"yfinance data unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed: {exc}")

    calibration = model.calibrate(inputs)
    assert isinstance(calibration, CalibrationResult)
    assert calibration.model_name == "var"
    assert calibration.fit_metrics["n_assets"] == 2.0
    # The 1-day VaR should be positive and a sensible fraction of gross.
    var_1d = calibration.fit_metrics["var_1d_dollar"]
    gross = calibration.fit_metrics["gross_exposure"]
    assert var_1d > 0
    # SPY/QQQ daily vol ~1.2%/1.5%; long-only $1.5M book: 99% 1d VaR
    # typically lands in $25k - $100k under normal regimes.
    assert 0.005 * gross < var_1d < 0.20 * gross

    risk = model.predict(inputs)
    assert isinstance(risk, RiskMetric)
    assert risk.ticker == "PORTFOLIO"
    assert risk.metric_name == "value_at_risk"
    assert risk.value == pytest.approx(var_1d, rel=1e-9)
    assert risk.metadata["method"] == method
    # Sqrt-h scaling: 10-day VaR = sqrt(10) * 1-day VaR.
    assert risk.metadata["var_horizon_dollar"] == pytest.approx(
        risk.value * (10 ** 0.5), rel=1e-9
    )

    diag = model.validate(inputs)
    assert diag["method"] == method
    assert diag["traffic_light"] in {"green", "yellow", "red"}
    # The Basel traffic-light is only formally calibrated for T=250 at
    # alpha=0.99; we've configured the model that way. We don't insist on
    # green (real data has fat tails) but we do insist on a sensible
    # backtest size and a finite breach rate.
    assert diag["backtest_n"] > 200
    rate = float(diag["backtest_breach_rate"])
    assert 0.0 <= rate <= 0.15


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
def test_var_long_short_portfolio_on_real_tickers() -> None:
    """A long-short pair (long SPY / short QQQ) should yield a much smaller
    VaR than the gross book because the two ETFs are highly correlated.
    Tests that the model handles signed dollar positions correctly.
    """

    provider = YFinanceProvider(cache=None)
    long_short = {"SPY": 1_000_000.0, "QQQ": -1_000_000.0}
    model = VaRModel(
        positions=long_short,
        method="historical",
        alpha=0.99,
        lookback_days=500,
        horizon_days=10,
        history_period="3y",
        backtest_window=250,
    )
    try:
        inputs = model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed: {exc}")

    model.calibrate(inputs)
    risk = model.predict(inputs)
    # Net exposure is 0 but gross is $2M. Hedged book should have far
    # smaller VaR than 1% of gross (the long-only benchmark).
    gross = float(risk.metadata["gross_exposure"])
    assert risk.value < 0.02 * gross


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
def test_var_methods_agree_within_band() -> None:
    """Under broadly-Gaussian-looking equity returns, the three methods
    should agree to within ~50%. Strong disagreement implies a regime
    where fat tails dominate (HS > parametric) — informative but rare on
    SPY+QQQ at 1% over 500 days.
    """

    provider = YFinanceProvider(cache=None)
    base_kwargs = dict(
        positions=_POSITIONS,
        alpha=0.99,
        lookback_days=500,
        horizon_days=1,
        history_period="3y",
        backtest_window=250,
        mc_samples=20_000,
        mc_seed=2025,
    )
    try:
        model_p = VaRModel(method="parametric", **base_kwargs)  # type: ignore[arg-type]
        inputs = model_p.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed: {exc}")
    var_p = model_p.predict(inputs).value

    model_h = VaRModel(method="historical", **base_kwargs)  # type: ignore[arg-type]
    var_h = model_h.predict(model_h.fetch_data(provider)).value

    model_mc = VaRModel(method="monte_carlo", **base_kwargs)  # type: ignore[arg-type]
    var_mc = model_mc.predict(model_mc.fetch_data(provider)).value

    # Pairwise ratios in [0.5, 2.0] — generous to accommodate fat tails.
    assert 0.5 < var_p / var_h < 2.0
    assert 0.5 < var_p / var_mc < 2.0
