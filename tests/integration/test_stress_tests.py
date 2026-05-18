"""End-to-end integration test for the stress-tests risk model.

Hits yfinance via `YFinanceProvider` for a small US-equity book and runs the
full pipeline (fetch -> calibrate -> predict -> validate). Skipped when
yfinance is unavailable or the fetch fails so offline CI stays green.

The portfolio is two large-caps (AAPL, JPM) with a short-equity hedge through
SPY; this gives both a meaningful equity-factor sensitivity and enough proxy
coverage (XLK, XLF) for the historical-replay sector fallback.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yfinance")

from src.core.data_provider import YFinanceProvider  # noqa: E402
from src.core.types import CalibrationResult, RiskMetric  # noqa: E402
from src.models.stress_tests import StressTests  # noqa: E402
from src.models.stress_tests.types import StressInputs  # noqa: E402

_POSITIONS: dict[str, float] = {
    "AAPL": 1_000_000.0,
    "JPM": 500_000.0,
    "SPY": -500_000.0,
}
_CAPITAL: float = 5_000_000.0


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model() -> StressTests:
    return StressTests(
        positions=_POSITIONS,
        capital=_CAPITAL,
        calibration_period="2y",
    )


def _fetch_or_skip(model: StressTests, provider: YFinanceProvider) -> StressInputs:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_pipeline_against_yfinance(
    provider: YFinanceProvider, model: StressTests
) -> None:
    data = _fetch_or_skip(model, provider)
    assert isinstance(data, StressInputs)
    # We should have some coverage of recent crises (Lehman 2008 onwards is
    # within AAPL / JPM / SPY trading history).
    recent_windows = ["Lehman 2008", "COVID crash", "GameStop unwind"]
    for w in recent_windows:
        # At least the benchmark `^GSPC` should be present for each window.
        assert any(
            key for key in data.crisis_prices if key[0] == w
        ), f"no crisis_prices for window {w!r}"

    result = model.calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "stress_tests"

    risk = model.predict(data)
    assert isinstance(risk, RiskMetric)
    assert risk.ticker == "PORTFOLIO"
    assert risk.metric_name == "stress_worst_case_pnl"
    # Worst case must be non-positive (a "loss") in any realistic crisis run.
    assert risk.value <= 0.0
    # Worst-case loss as a fraction of capital should be in a sane range.
    loss_pct = risk.metadata["loss_pct_of_capital"]
    assert 0.0 <= loss_pct <= 1.0


@pytest.mark.integration
def test_diagnostics_present(
    provider: YFinanceProvider, model: StressTests
) -> None:
    """Spec validation block: top-3 contributors and scenario coverage."""

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    diag = model.validate(data)
    assert diag["n_historical"] > 0
    assert diag["n_hypothetical"] > 0
    assert "worst_historical_top3" in diag
    # Reverse-stress section should be present once a fit exists.
    assert "reverse_mahalanobis" in diag


@pytest.mark.integration
def test_lehman_loss_is_negative_for_long_only(
    provider: YFinanceProvider,
) -> None:
    """Sanity check: a long-only equity book should lose money in Lehman 2008.

    The shape of the book differs from the module-level model fixture, so we
    build a fresh long-only model here.
    """

    model = StressTests(
        positions={"AAPL": 1_000_000.0, "JPM": 1_000_000.0},
        capital=5_000_000.0,
        calibration_period="2y",
    )
    try:
        data = model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed: {exc!r}")
    report = model.run_full_report(data)
    lehman = report.historical.get("Lehman 2008")
    assert lehman is not None
    assert lehman.total_pnl < 0
