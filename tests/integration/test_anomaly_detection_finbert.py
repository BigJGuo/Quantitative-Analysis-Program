"""End-to-end integration test for the anomaly-detection + FinBERT model.

Hits yfinance via `YFinanceProvider` for a small universe and runs the full
`fetch_data -> calibrate -> predict -> validate` pipeline. Skipped if
`yfinance` is missing or the network fetch fails — keeps offline CI green.

We force `SentimentConfig(backend="stub")` so the test never depends on the
optional `transformers` / `torch` stack, but still exercises the
end-to-end sentiment / news path with real yfinance news payloads.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yfinance")

from src.core.data_provider import YFinanceProvider  # noqa: E402
from src.core.types import CalibrationResult, Signal  # noqa: E402
from src.models.anomaly_detection_finbert import (  # noqa: E402
    AnomalyConfig,
    AnomalyDetectionFinBERT,
    AnomalyInputs,
    SentimentConfig,
)

_UNIVERSE = ["SPY", "QQQ", "IWM"]


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model() -> AnomalyDetectionFinBERT:
    return AnomalyDetectionFinBERT(
        universe=_UNIVERSE,
        period="1y",
        anomaly_config=AnomalyConfig(
            iforest_n_trees=30,
            iforest_subsample=128,
            bottleneck_dim=4,
        ),
        sentiment_config=SentimentConfig(backend="stub"),
    )


def _fetch_or_skip(
    model: AnomalyDetectionFinBERT, provider: YFinanceProvider
) -> AnomalyInputs:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_pipeline_against_yfinance(
    provider: YFinanceProvider, model: AnomalyDetectionFinBERT
) -> None:
    data = _fetch_or_skip(model, provider)
    assert isinstance(data, AnomalyInputs)
    assert data.features.shape[0] > 50
    assert data.features.shape[1] == 9

    result = model.calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "anomaly_detection_finbert"
    assert set(model.fit.thresholds.keys()) == {"ae", "iforest", "mahalanobis"}

    sig = model.predict(data)
    assert isinstance(sig, Signal)
    assert sig.direction in {"long", "short", "flat"}
    assert 0.0 <= sig.strength <= 1.0
    assert "anomaly_flag" in sig.metadata
    assert "sentiment_score" in sig.metadata
    assert sig.metadata["sentiment_backend"] in {"stub", "empty"}


@pytest.mark.integration
def test_validation_diagnostics(
    provider: YFinanceProvider, model: AnomalyDetectionFinBERT
) -> None:
    """Spec validation: injected anomalies recovered at >= 0.9 recall."""

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    diag = model.validate(data)
    assert diag["n_rows"] > 0
    assert diag["injected_recall"] >= 0.9
    for key in ("ae", "iforest", "mahalanobis"):
        assert key in diag["flag_rate_per_detector"]
        # In-sample false-positive rate should be near the contamination level.
        assert diag["flag_rate_per_detector"][key] <= 0.02
