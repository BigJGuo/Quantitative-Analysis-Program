"""`AnomalyDetectionFinBERT` — Layer 4 numerical anomaly + sentiment model.

Thin orchestration shell around the pure helpers in `signal.py` and the
`calibrate` entry point in `calibration.py`. Two jobs in one class:

- **Numerical anomaly detection** via an ensemble of three detectors
  (PCA reconstruction error / isolation forest / shrinkage Mahalanobis).
  Inference returns a per-row flag plus top-feature attribution.
- **Sentiment scoring** via FinBERT (lazy `transformers` import) over the
  `provider.fetch_news` payload. Falls back to a deterministic stub when
  `transformers` is not installed so the model still returns a well-formed
  `Signal`.

The model's `Signal` payload reports the latest ticker's sentiment as the
primary alpha feature; the anomaly flag is attached in `metadata` for
monitoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Signal, SignalDirection
from src.models.anomaly_detection_finbert.calibration import MODEL_NAME
from src.models.anomaly_detection_finbert.calibration import calibrate as _calibrate_impl
from src.models.anomaly_detection_finbert.signal import (
    build_anomaly_feature_panel,
    cross_sectional_rank,
    exp_decay_weight,
    impute_and_clip,
    majority_vote_flags,
    median_mad_standardize,
    pca_residual_attribution,
    score_all_detectors,
    sentiment_scalar,
    split_sentences,
    top_feature_attribution,
    weighted_sentiment,
)
from src.models.anomaly_detection_finbert.types import (
    DEFAULT_FEATURE_NAMES,
    AnomalyAlert,
    AnomalyConfig,
    AnomalyFit,
    AnomalyInputs,
    SentimentConfig,
    SentimentScore,
)

_DEFAULT_PERIOD: str = "2y"
_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_HORIZON_LABEL: str = "1d"


@register_model
class AnomalyDetectionFinBERT(BaseModel):
    """Anomaly-ensemble + FinBERT sentiment for a configured universe.

    Parameters
    ----------
    universe:
        Tickers feeding the pooled feature panel. The cross-sectional rank
        feature is computed within this list, so a meaningful universe is
        at least 5 tickers.
    period:
        yfinance ``period`` string for `fetch_prices`. Defaults to `"2y"`.
    anomaly_config:
        `AnomalyConfig` overrides (bottleneck dim, iForest hyperparams, etc.).
    sentiment_config:
        `SentimentConfig` overrides. `backend="stub"` disables FinBERT even
        when `transformers` is installed (handy for unit tests).
    max_news_per_ticker:
        Cap on news items pulled per ticker via `provider.fetch_news`.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "monthly"

    def __init__(
        self,
        universe: Sequence[str],
        *,
        period: str = _DEFAULT_PERIOD,
        anomaly_config: AnomalyConfig | None = None,
        sentiment_config: SentimentConfig | None = None,
        max_news_per_ticker: int = 20,
    ) -> None:
        if len(universe) < 1:
            raise ValueError(
                f"AnomalyDetectionFinBERT requires at least 1 ticker, got {len(universe)}"
            )
        if max_news_per_ticker < 0:
            raise ValueError(
                f"max_news_per_ticker must be >= 0, got {max_news_per_ticker}"
            )
        self.universe: tuple[str, ...] = tuple(universe)
        self.period: str = period
        self.anomaly_config: AnomalyConfig = anomaly_config or AnomalyConfig()
        self.sentiment_config: SentimentConfig = (
            sentiment_config or SentimentConfig()
        )
        self.max_news_per_ticker: int = max_news_per_ticker
        self._fit: AnomalyFit | None = None
        self._finbert_cache: _FinBERTBackend | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> AnomalyInputs:
        feature_frames: list[pd.DataFrame] = []
        ticker_id_chunks: list[np.ndarray] = []
        returns_chunks: dict[str, pd.Series] = {}
        news_payload: dict[str, list[dict[str, Any]]] = {}

        for tid, ticker in enumerate(self.universe):
            bars = provider.fetch_prices(ticker, self.period, _DEFAULT_INTERVAL)
            indexed = _index_bars(bars)
            if indexed is None or indexed.empty:
                continue
            panel = build_anomaly_feature_panel(indexed)
            panel = panel.assign(_ticker=ticker)
            feature_frames.append(panel)
            ticker_id_chunks.append(np.full(len(panel), tid, dtype=int))
            returns_chunks[ticker] = panel["log_return_1d"].rename(ticker)
            if self.max_news_per_ticker > 0:
                try:
                    news_payload[ticker] = provider.fetch_news(
                        ticker, max_items=self.max_news_per_ticker
                    )
                except Exception:  # noqa: BLE001
                    # News fetch is best-effort; sentiment branch handles
                    # missing payloads as `coverage = 0`.
                    news_payload[ticker] = []

        if not feature_frames:
            raise RuntimeError(
                f"fetch_data produced no usable rows for universe {self.universe!r}"
            )

        panel = pd.concat(feature_frames, axis=0, ignore_index=False)
        ticker_ids = np.concatenate(ticker_id_chunks)

        # Cross-sectional rank feature: align all returns into a wide frame
        # then look up each (date, ticker)'s rank.
        wide_returns = pd.concat(returns_chunks.values(), axis=1)
        ranks = cross_sectional_rank(wide_returns)
        rank_values: list[float] = []
        for idx, ticker in zip(panel.index, panel["_ticker"], strict=False):
            try:
                rank_values.append(float(ranks.loc[idx, ticker]))
            except (KeyError, TypeError):
                rank_values.append(float("nan"))
        panel = panel.copy()
        panel["cross_sectional_rank"] = rank_values

        feature_cols = list(DEFAULT_FEATURE_NAMES)
        # Sanity: make sure every expected column is present.
        missing = [c for c in feature_cols if c not in panel.columns]
        if missing:
            raise RuntimeError(
                f"feature panel missing expected columns: {missing}"
            )
        features = panel.loc[:, feature_cols]
        # Drop rows where every feature is NaN (typically the first 60 bars
        # before the rolling windows have warmed up).
        valid = ~features.isna().all(axis=1)
        features = features.loc[valid].reset_index(drop=False)
        ticker_ids = ticker_ids[valid.to_numpy()]
        # `reset_index(drop=False)` produced an "index" column we want as dates.
        dates_col = features.columns[0]
        dates = features[dates_col].to_numpy()
        features = features.drop(columns=[dates_col])

        return AnomalyInputs(
            features=features,
            feature_names=DEFAULT_FEATURE_NAMES,
            dates=dates,
            ticker_ids=ticker_ids,
            tickers=self.universe,
            news=news_payload,
            timestamp=datetime.now(UTC),
            metadata={
                "period": self.period,
                "n_tickers": float(len(self.universe)),
                "n_rows": float(features.shape[0]),
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, AnomalyInputs):
            raise TypeError(
                f"AnomalyDetectionFinBERT.calibrate expects AnomalyInputs, got "
                f"{type(data).__name__}"
            )
        result = _calibrate_impl(data, config=self.anomaly_config)
        self._fit = cast(AnomalyFit, result.parameters["anomaly_fit"])
        return result

    def predict(self, data: Any) -> Signal:
        if not isinstance(data, AnomalyInputs):
            raise TypeError(
                f"AnomalyDetectionFinBERT.predict expects AnomalyInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        latest_idx = int(np.argmax(data.dates))
        x_raw = data.features.iloc[latest_idx].to_numpy(dtype=float)
        ticker_id = int(data.ticker_ids[latest_idx])
        ticker = (
            data.tickers[ticker_id]
            if ticker_id < len(data.tickers)
            else "POOLED"
        )
        ts = _to_aware(data.dates[latest_idx])

        alert = self._score_row(x_raw, fit, ticker=ticker, ts=ts)
        sentiment = self.score_sentiment(
            ticker=ticker, news=data.news.get(ticker, []), as_of=ts
        )

        direction: SignalDirection
        strength: float
        if sentiment.score > 0:
            direction = "long"
            strength = float(min(abs(sentiment.score), 1.0))
        elif sentiment.score < 0:
            direction = "short"
            strength = float(min(abs(sentiment.score), 1.0))
        else:
            direction = "flat"
            strength = 0.0

        return Signal(
            ticker=ticker,
            direction=direction,
            strength=strength,
            timestamp=ts,
            horizon=_DEFAULT_HORIZON_LABEL,
            metadata={
                "sentiment_score": sentiment.score,
                "sentiment_coverage": float(sentiment.coverage),
                "sentiment_backend": sentiment.backend,
                "anomaly_flag": float(alert.combined),
                "anomaly_scores": dict(alert.scores),
                "anomaly_flags": {k: float(v) for k, v in alert.flags.items()},
                "top_features": [
                    [name, value] for name, value in alert.top_features
                ],
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, AnomalyInputs):
            raise TypeError(
                f"AnomalyDetectionFinBERT.validate expects AnomalyInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        X_raw = data.features.to_numpy(dtype=float)
        X_std, _, _ = median_mad_standardize(
            X_raw,
            median=fit.detector.feature_median,
            scale=fit.detector.feature_scale,
        )
        X_clean = impute_and_clip(X_std, fit.config.clip_z)
        scores = score_all_detectors(X_clean, fit.detector)
        flags = {key: scores[key] > fit.thresholds[key] for key in scores}
        flag_rate = {key: float(np.mean(flags[key])) for key in flags}
        combined = (
            sum(flags[key].astype(int) for key in flags) >= 2
        )
        combined_rate = float(np.mean(combined))

        # Injected-anomaly diagnostic: perturb a random subset by ±20σ
        # (spec validation item — "synthetic anomalies"). Measure recall.
        rng = np.random.default_rng(42)
        n = X_clean.shape[0]
        n_inject = max(int(0.01 * n), 1)
        inject_idx = rng.choice(n, size=n_inject, replace=False)
        X_perturbed = X_clean.copy()
        # Push the chosen rows well past the clip boundary.
        perturb_direction = rng.choice([-1.0, 1.0], size=X_perturbed.shape[1])
        X_perturbed[inject_idx, :] += 20.0 * perturb_direction
        injected_scores = score_all_detectors(X_perturbed, fit.detector)
        injected_flags = {
            key: injected_scores[key][inject_idx] > fit.thresholds[key]
            for key in injected_scores
        }
        injected_combined = (
            sum(injected_flags[key].astype(int) for key in injected_flags) >= 2
        )
        recall = float(np.mean(injected_combined))

        return {
            "n_rows": int(n),
            "flag_rate_per_detector": flag_rate,
            "combined_flag_rate": combined_rate,
            "score_means": {
                key: float(np.mean(arr)) for key, arr in scores.items()
            },
            "score_p99": {
                key: float(np.quantile(arr, 0.99)) for key, arr in scores.items()
            },
            "injected_recall": recall,
            "n_injected": int(n_inject),
            "thresholds": dict(fit.thresholds),
        }

    # ----- Public conveniences -----------------------------------------------

    @property
    def fit(self) -> AnomalyFit:
        return self._require_fit()

    def score_row(self, x_raw: np.ndarray, ticker: str = "UNKNOWN") -> AnomalyAlert:
        """Public scoring for an external row in *raw feature units*."""

        fit = self._require_fit()
        return self._score_row(x_raw, fit, ticker=ticker, ts=datetime.now(UTC))

    def score_sentiment(
        self,
        *,
        ticker: str,
        news: Sequence[dict[str, Any]],
        as_of: datetime,
    ) -> SentimentScore:
        """Aggregate FinBERT sentiment for `news` items at time `as_of`."""

        if not news:
            return SentimentScore(
                ticker=ticker,
                date=as_of,
                score=0.0,
                coverage=0,
                backend="empty",
            )
        backend = self._get_finbert_backend()
        article_scores: list[float] = []
        weights: list[float] = []
        for item in news:
            body = _article_text(item)
            if not body:
                continue
            sentences = split_sentences(body)
            if not sentences:
                continue
            probs_per_sentence = backend.score_sentences(sentences)  # (n, 3)
            sentence_scalars = [
                sentiment_scalar(probs_per_sentence[i], backend.label_to_index)
                for i in range(probs_per_sentence.shape[0])
            ]
            article_score = float(np.mean(sentence_scalars)) if sentence_scalars else 0.0
            age_hours = _age_hours(item, as_of)
            weight = exp_decay_weight(age_hours, self.sentiment_config.decay_hours)
            article_scores.append(article_score)
            weights.append(weight)
        scalar = weighted_sentiment(
            article_scores, weights, eps=self.sentiment_config.eps
        )
        # Clip into [-1, 1] to satisfy `SentimentScore.__post_init__`. The
        # weighted average is *mathematically* in [-1, 1] but float error can
        # push it slightly beyond.
        scalar_clipped = float(max(-1.0, min(1.0, scalar)))
        return SentimentScore(
            ticker=ticker,
            date=as_of,
            score=scalar_clipped,
            coverage=len(article_scores),
            backend=backend.name,
            metadata={
                "n_news_items": float(len(news)),
                "decay_hours": float(self.sentiment_config.decay_hours),
            },
        )

    # ----- Internal helpers --------------------------------------------------

    def _require_fit(self) -> AnomalyFit:
        if self._fit is None:
            raise RuntimeError(
                "AnomalyDetectionFinBERT has not been calibrated. "
                "Call calibrate(data) first."
            )
        return self._fit

    def _score_row(
        self,
        x_raw: np.ndarray,
        fit: AnomalyFit,
        *,
        ticker: str,
        ts: datetime,
    ) -> AnomalyAlert:
        x = x_raw.reshape(1, -1)
        x_std, _, _ = median_mad_standardize(
            x, median=fit.detector.feature_median, scale=fit.detector.feature_scale
        )
        x_clean = impute_and_clip(x_std, fit.config.clip_z)
        scores = score_all_detectors(x_clean, fit.detector)
        single_scores = {key: float(arr[0]) for key, arr in scores.items()}
        flags = {key: single_scores[key] > fit.thresholds[key] for key in single_scores}
        combined = majority_vote_flags(flags, min_votes=2)
        residuals = pca_residual_attribution(
            x_clean[0], fit.detector.pca_mean, fit.detector.pca_components
        )
        top = top_feature_attribution(residuals, fit.feature_names, k=3)
        return AnomalyAlert(
            ticker=ticker,
            date=ts,
            scores=single_scores,
            flags=flags,
            combined=combined,
            top_features=top,
        )

    def _get_finbert_backend(self) -> _FinBERTBackend:
        if self._finbert_cache is not None:
            return self._finbert_cache
        cfg = self.sentiment_config
        if cfg.backend == "stub":
            self._finbert_cache = _StubBackend()
            return self._finbert_cache
        try:
            self._finbert_cache = _TransformersBackend(
                model_id=cfg.finbert_model_id,
                max_seq_len=cfg.max_seq_len,
            )
        except _BackendUnavailable:
            self._finbert_cache = _StubBackend()
        return self._finbert_cache


# ---- module-level helpers ---------------------------------------------------


def _index_bars(bars: pd.DataFrame) -> pd.DataFrame | None:
    """Pick a DatetimeIndex from a yfinance frame; return None if unusable."""

    if bars is None or bars.empty:
        return None
    df = bars.copy()
    if "Date" in df.columns:
        df.index = pd.DatetimeIndex(pd.to_datetime(df["Date"]))
        df = df.drop(columns="Date")
    elif "Datetime" in df.columns:
        df.index = pd.DatetimeIndex(pd.to_datetime(df["Datetime"]))
        df = df.drop(columns="Datetime")
    else:
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _to_aware(value: Any) -> datetime:
    ts = pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _article_text(item: dict[str, Any]) -> str:
    """Concatenate `title` and `summary` per the spec's pipeline pseudocode."""

    title = str(item.get("title", "")).strip()
    summary = str(item.get("summary", "")).strip()
    if title and summary:
        return f"{title}. {summary}"
    return title or summary


def _age_hours(item: dict[str, Any], as_of: datetime) -> float:
    """Compute article age in hours given `providerPublishTime` (epoch s)."""

    raw = item.get("providerPublishTime") or item.get("published")
    if raw is None:
        return 0.0
    try:
        if isinstance(raw, int | float):
            published = datetime.fromtimestamp(float(raw), tz=UTC)
        else:
            published = pd.Timestamp(raw).to_pydatetime()
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
    except Exception:  # noqa: BLE001
        return 0.0
    as_of_aware = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)
    delta_seconds = (as_of_aware - published).total_seconds()
    return max(delta_seconds, 0.0) / 3600.0


# ----- FinBERT backend abstraction -------------------------------------------


class _BackendUnavailable(RuntimeError):
    """Raised when `transformers`/`torch` cannot be imported."""


class _FinBERTBackend:
    """Common interface: `score_sentences` returns an (N, 3) probs array.

    Subclasses populate `name` and `label_to_index`. `label_to_index` MUST
    contain `"positive"` and `"negative"` (case sensitive) — the spec's
    label-order surprise is handled by reading the model config at startup.
    """

    name: str = "abstract"
    label_to_index: dict[str, int] = {}

    def score_sentences(self, sentences: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class _StubBackend(_FinBERTBackend):
    """Deterministic placeholder used when `transformers` is missing.

    Always emits a neutral probability vector — sentiment scalar = 0.
    Useful for offline CI and unit tests; production must install
    `transformers` to get real sentiment.
    """

    name = "stub"
    label_to_index = {"positive": 0, "negative": 1, "neutral": 2}

    def score_sentences(self, sentences: Sequence[str]) -> np.ndarray:
        n = len(sentences)
        probs = np.zeros((n, 3), dtype=float)
        probs[:, self.label_to_index["neutral"]] = 1.0
        return probs


class _TransformersBackend(_FinBERTBackend):
    """Real FinBERT-via-Transformers backend. Lazy-loads the checkpoint.

    Reads `model.config.label2id` at startup so the scalar conversion is
    robust to the label-order surprise documented in the spec.
    """

    def __init__(self, model_id: str, max_seq_len: int) -> None:
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise _BackendUnavailable(
                "transformers not installed; falling back to stub backend"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            raise _BackendUnavailable(
                f"FinBERT checkpoint {model_id!r} unavailable: {exc!r}"
            ) from exc
        self._model.eval()
        self.max_seq_len = max_seq_len
        self.name = "finbert"
        # Read the runtime label order — *do not* trust alphabetical order.
        label2id = getattr(self._model.config, "label2id", None) or {}
        normalized: dict[str, int] = {}
        for label, idx in label2id.items():
            normalized[str(label).strip().lower()] = int(idx)
        # FinBERT's id2label is sometimes `{0: 'positive', 1: 'negative', 2: 'neutral'}`
        # and sometimes alphabetical. Either way we look up by name.
        if "positive" not in normalized or "negative" not in normalized:
            # Fallback to the documented ProsusAI order if config is missing.
            normalized = {"positive": 0, "negative": 1, "neutral": 2}
        self.label_to_index = normalized

    def score_sentences(self, sentences: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs transformers
        if not sentences:
            return np.zeros((0, 3), dtype=float)
        import torch

        inputs = self._tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        return cast(np.ndarray, probs.cpu().numpy().astype(float))


__all__ = ["AnomalyDetectionFinBERT"]
