"""End-to-end integration test for the cointegration / pair-trading model.

Hits yfinance via `YFinanceProvider` for KO / PEP — the textbook cointegrated
pair from the spec. Skipped automatically when yfinance is missing or the
network fetch fails, so unit-test CI without internet stays green.
"""

from __future__ import annotations

import pytest

from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, Signal
from src.models.cointegration_pairs import CointegrationPairs
from src.models.cointegration_pairs.types import PairInputs

pytest.importorskip("yfinance")


_TICKER_A = "KO"
_TICKER_B = "PEP"


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def static_model() -> CointegrationPairs:
    return CointegrationPairs(
        ticker_a=_TICKER_A,
        ticker_b=_TICKER_B,
        method="static",
        period="2y",
        significance="5%",
    )


@pytest.fixture(scope="module")
def kalman_model() -> CointegrationPairs:
    return CointegrationPairs(
        ticker_a=_TICKER_A,
        ticker_b=_TICKER_B,
        method="kalman",
        period="2y",
        significance="5%",
    )


def _fetch_or_skip(
    model: CointegrationPairs, provider: YFinanceProvider
) -> PairInputs:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_static_pipeline_against_yfinance(
    provider: YFinanceProvider, static_model: CointegrationPairs
) -> None:
    data = _fetch_or_skip(static_model, provider)
    assert isinstance(data, PairInputs)
    assert data.ticker_a == _TICKER_A
    assert data.ticker_b == _TICKER_B
    assert len(data.log_price_a) > 200  # ~ 2 years of trading days

    result = static_model.calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "cointegration_pairs"
    # Cointegration outcome on KO/PEP varies window-to-window. We only assert
    # that the pipeline runs through and produces sane metric values.
    assert "alpha" in result.fit_metrics
    assert "beta" in result.fit_metrics
    assert "half_life" in result.fit_metrics

    signal = static_model.predict(data)
    assert isinstance(signal, Signal)
    assert signal.ticker == f"{_TICKER_A}/{_TICKER_B}"
    assert signal.direction in ("long", "short", "flat")
    assert 0.0 <= signal.strength <= 1.0


@pytest.mark.integration
def test_kalman_pipeline_against_yfinance(
    provider: YFinanceProvider, kalman_model: CointegrationPairs
) -> None:
    data = _fetch_or_skip(kalman_model, provider)
    kalman_model.calibrate(data)
    fit = kalman_model.fit
    # Kalman fit should always exist when calibrate ran cleanly, even if the
    # pair was rejected — but we only require it on accepted pairs.
    if fit.is_cointegrated:
        assert fit.kalman_fit is not None
        # Filtered beta path should span all observations.
        assert fit.kalman_fit.beta_path.shape[0] == len(data.log_price_a)

    signal = kalman_model.predict(data)
    assert signal.metadata["method"] == "kalman"
    assert signal.metadata["ticker_a"] == _TICKER_A
    assert signal.metadata["ticker_b"] == _TICKER_B


@pytest.mark.integration
def test_diagnostics_emitted(
    provider: YFinanceProvider, static_model: CointegrationPairs
) -> None:
    """Spec validation step 2 (rolling EG t-stat) and step 6 (realized half-life)."""

    data = _fetch_or_skip(static_model, provider)
    static_model.calibrate(data)
    diag = static_model.validate(data)
    assert "realized_half_life" in diag
    assert "ou_half_life" in diag
    assert "eg_residual_tstat" in diag
    # Rolling diagnostics should appear for any pull with enough history (2y).
    assert "rolling_eg_tstat_last" in diag
