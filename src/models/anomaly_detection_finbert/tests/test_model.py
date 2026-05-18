"""Unit tests for `AnomalyDetectionFinBERT` against an `InMemoryProvider`.

Exercises the full `fetch_data -> calibrate -> predict -> validate` cycle
with synthetic OHLCV bars. The FinBERT backend is forced to `"stub"` so
these tests never require `transformers` / `torch`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, Signal
from src.models.anomaly_detection_finbert import (
    AnomalyConfig,
    AnomalyDetectionFinBERT,
    AnomalyInputs,
    SentimentConfig,
)
from src.models.anomaly_detection_finbert.calibration import MODEL_NAME


def _make_bars(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    log_rets = rng.normal(0.0, 0.012, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + rng.normal(0.0, 0.001, size=n)),
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n),
        }
    )


@pytest.fixture
def provider() -> InMemoryProvider:
    rng = np.random.default_rng(20260518)
    bars_a = _make_bars(rng, n=400)
    bars_b = _make_bars(rng, n=400)
    bars_c = _make_bars(rng, n=400)
    return InMemoryProvider(
        prices={
            ("AAA", "2y", "1d"): bars_a,
            ("BBB", "2y", "1d"): bars_b,
            ("CCC", "2y", "1d"): bars_c,
        },
        news={
            "AAA": [
                {
                    "title": "Strong earnings beat. Bullish guidance.",
                    "summary": "Revenue grew 25%. The stock rallied.",
                    "providerPublishTime": int(
                        (datetime.now(UTC) - timedelta(hours=4)).timestamp()
                    ),
                }
            ],
            "BBB": [],
            "CCC": [],
        },
    )


@pytest.fixture
def model() -> AnomalyDetectionFinBERT:
    return AnomalyDetectionFinBERT(
        universe=["AAA", "BBB", "CCC"],
        period="2y",
        anomaly_config=AnomalyConfig(iforest_n_trees=20, iforest_subsample=64),
        sentiment_config=SentimentConfig(backend="stub"),
    )


class TestRegistry:
    def test_model_is_registered(self) -> None:
        cls = get_model(MODEL_NAME)
        assert cls is AnomalyDetectionFinBERT
        assert cls.layer == 4
        assert cls.refit_frequency == "monthly"


class TestFetchData:
    def test_returns_anomaly_inputs(
        self, model: AnomalyDetectionFinBERT, provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(provider)
        assert isinstance(data, AnomalyInputs)
        # 3 tickers * ~400 rows after rolling-window NaN drop.
        assert data.features.shape[0] > 100
        assert data.features.shape[1] == 9
        assert data.feature_names[-1] == "cross_sectional_rank"

    def test_news_payload_present(
        self, model: AnomalyDetectionFinBERT, provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(provider)
        assert "AAA" in data.news
        assert len(data.news["AAA"]) == 1


class TestCalibrate:
    def test_returns_calibration_result(
        self, model: AnomalyDetectionFinBERT, provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(provider)
        result = model.calibrate(data)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        assert model.fit.thresholds.keys() == {"ae", "iforest", "mahalanobis"}

    def test_rejects_wrong_type(self, model: AnomalyDetectionFinBERT) -> None:
        with pytest.raises(TypeError):
            model.calibrate({"not": "AnomalyInputs"})


class TestPredict:
    def test_emits_well_formed_signal(
        self, model: AnomalyDetectionFinBERT, provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(provider)
        model.calibrate(data)
        sig = model.predict(data)
        assert isinstance(sig, Signal)
        assert sig.direction in {"long", "short", "flat"}
        assert 0.0 <= sig.strength <= 1.0
        # Stub backend ⇒ neutral sentiment ⇒ flat direction.
        assert sig.direction == "flat"
        assert sig.metadata["sentiment_backend"] == "stub"
        assert "anomaly_flag" in sig.metadata
        assert isinstance(sig.metadata["anomaly_scores"], dict)
        assert set(sig.metadata["anomaly_scores"].keys()) == {
            "ae",
            "iforest",
            "mahalanobis",
        }

    def test_uncalibrated_raises(self, model: AnomalyDetectionFinBERT) -> None:
        # Build a tiny valid AnomalyInputs to feed predict.
        rng = np.random.default_rng(0)
        X = rng.normal(0.0, 1.0, size=(3, 9))
        from src.models.anomaly_detection_finbert.types import DEFAULT_FEATURE_NAMES

        inp = AnomalyInputs(
            features=pd.DataFrame(X, columns=list(DEFAULT_FEATURE_NAMES)),
            feature_names=DEFAULT_FEATURE_NAMES,
            dates=np.arange(3).astype("datetime64[D]"),
            ticker_ids=np.zeros(3, dtype=int),
            tickers=("AAA",),
            news={},
        )
        with pytest.raises(RuntimeError):
            model.predict(inp)


class TestValidate:
    def test_injected_anomaly_recall_is_perfect(
        self, model: AnomalyDetectionFinBERT, provider: InMemoryProvider
    ) -> None:
        """Spec validation: injected synthetic anomalies (±20σ) MUST be flagged.

        With 3-out-of-3 majority vote on extremely perturbed rows, recall
        should be at or near 1.0 — the spec's PR-AUC diagnostic at the
        easiest possible level.
        """

        data = model.fetch_data(provider)
        model.calibrate(data)
        diag = model.validate(data)
        assert diag["injected_recall"] >= 0.9
        # Each per-detector flag rate at the threshold approximates
        # contamination = 0.005.
        for key, rate in diag["flag_rate_per_detector"].items():
            assert rate <= 0.02, f"detector {key} false-positive rate too high"
        assert diag["combined_flag_rate"] <= diag["flag_rate_per_detector"]["ae"]


class TestSentiment:
    def test_stub_backend_returns_neutral(
        self, model: AnomalyDetectionFinBERT
    ) -> None:
        articles = [
            {
                "title": "Stock soars on great news.",
                "providerPublishTime": int(datetime.now(UTC).timestamp()),
            }
        ]
        score = model.score_sentiment(
            ticker="AAA",
            news=articles,
            as_of=datetime.now(UTC),
        )
        assert score.score == 0.0
        assert score.backend == "stub"
        assert score.coverage == 1

    def test_empty_news_returns_empty_backend(
        self, model: AnomalyDetectionFinBERT
    ) -> None:
        score = model.score_sentiment(
            ticker="AAA", news=[], as_of=datetime.now(UTC)
        )
        assert score.backend == "empty"
        assert score.coverage == 0
        assert score.score == 0.0
