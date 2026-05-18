"""End-to-end integration test for the Bayesian-hierarchical / Kalman model.

Hits yfinance via `YFinanceProvider` for AAPL vs SPY and runs the full
Kalman-beta pipeline. Skipped automatically when yfinance is missing or the
network fetch fails, so unit-test CI without internet stays green.

The Kalman-beta path is the spec's primary use case ("Rolling beta of AAPL
to SPY"); the diagnostic checks correspond to the spec's "Validation and
diagnostics" section.
"""

from __future__ import annotations

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, Forecast
from src.models.bayesian_kalman import BayesianKalman
from src.models.bayesian_kalman.types import BayesianKalmanInputs, KalmanFit

pytest.importorskip("yfinance")


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model() -> BayesianKalman:
    return BayesianKalman(
        ticker="AAPL",
        market_ticker="SPY",
        mode="kalman_beta",
        period="2y",
        interval="1d",
    )


def _fetch_or_skip(model: BayesianKalman, provider: YFinanceProvider) -> BayesianKalmanInputs:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_kalman_pipeline_against_yfinance(
    provider: YFinanceProvider, model: BayesianKalman
) -> None:
    data = _fetch_or_skip(model, provider)
    assert isinstance(data, BayesianKalmanInputs)
    assert data.mode == "kalman_beta"
    assert data.asset_returns is not None and data.market_returns is not None
    # ~2y of trading days, give or take a weekend pull.
    assert data.asset_returns.shape[0] > 200
    assert data.market_returns.shape[0] > 200

    result = model.calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "bayesian_kalman"
    fit = model.fit
    assert isinstance(fit, KalmanFit)
    assert fit.n_states == 2

    pred = model.predict(data)
    assert isinstance(pred, Forecast)
    assert pred.ticker == "AAPL"
    # Mega-cap tech vs SPY: historical beta sits comfortably in 0.7 - 1.8.
    # Allow a wide envelope for any single 2y window.
    assert 0.3 < pred.value < 2.5
    assert pred.lower is not None and pred.upper is not None
    assert pred.lower < pred.value < pred.upper


@pytest.mark.integration
def test_kalman_diagnostic_block(
    provider: YFinanceProvider, model: BayesianKalman
) -> None:
    """Spec validation block: innovation whiteness + finite log-likelihood."""

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    diag = model.validate(data)
    assert diag["n_obs"] >= 200
    # Standardized innovations should be roughly mean-zero unit-variance.
    assert abs(diag["mean_standardized_innovation"]) < 0.5
    assert 0.5 < diag["std_standardized_innovation"] < 2.0
    # MLE-fit parameters must be strictly positive.
    assert diag["q_alpha"] > 0
    assert diag["q_beta"] > 0
    assert diag["R"] > 0
