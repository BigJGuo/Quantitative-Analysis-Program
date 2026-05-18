"""Validation tests for the dataclasses in `types.py`."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detection_finbert.types import (
    DEFAULT_FEATURE_NAMES,
    AnomalyConfig,
    AnomalyFit,
    AnomalyInputs,
    DetectorFit,
    ITreeRecord,
    SentimentConfig,
    SentimentScore,
)


class TestAnomalyConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = AnomalyConfig()
        assert cfg.bottleneck_dim == 8
        assert cfg.iforest_n_trees == 100
        assert cfg.contamination == 0.005

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bottleneck_dim": 0},
            {"iforest_n_trees": 0},
            {"iforest_subsample": 1},
            {"contamination": 0.0},
            {"contamination": 0.6},
            {"mahal_shrinkage": -1.0},
            {"clip_z": 0.0},
        ],
    )
    def test_rejects_invalid(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            AnomalyConfig(**kwargs)  # type: ignore[arg-type]


class TestSentimentConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = SentimentConfig()
        assert cfg.decay_hours == 24.0
        assert cfg.backend == "auto"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"decay_hours": 0.0},
            {"eps": 0.0},
            {"max_seq_len": 8},
            {"backend": "bogus"},
        ],
    )
    def test_rejects_invalid(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            SentimentConfig(**kwargs)  # type: ignore[arg-type]


class TestAnomalyInputs:
    def _make(self, n: int = 4, p: int = 9) -> AnomalyInputs:
        features = pd.DataFrame(
            np.zeros((n, p)), columns=list(DEFAULT_FEATURE_NAMES)
        )
        return AnomalyInputs(
            features=features,
            feature_names=DEFAULT_FEATURE_NAMES,
            dates=np.arange(n).astype("datetime64[D]"),
            ticker_ids=np.zeros(n, dtype=int),
            tickers=("AAA",),
            news={},
        )

    def test_round_trip(self) -> None:
        inp = self._make()
        assert inp.features.shape == (4, 9)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            AnomalyInputs(
                features=pd.DataFrame(columns=list(DEFAULT_FEATURE_NAMES)),
                feature_names=DEFAULT_FEATURE_NAMES,
                dates=np.array([], dtype="datetime64[D]"),
                ticker_ids=np.array([], dtype=int),
                tickers=(),
                news={},
            )

    def test_rejects_misaligned_dates(self) -> None:
        with pytest.raises(ValueError):
            AnomalyInputs(
                features=pd.DataFrame(
                    np.zeros((3, 9)), columns=list(DEFAULT_FEATURE_NAMES)
                ),
                feature_names=DEFAULT_FEATURE_NAMES,
                dates=np.arange(2).astype("datetime64[D]"),
                ticker_ids=np.zeros(3, dtype=int),
                tickers=("AAA",),
                news={},
            )


class TestSentimentScore:
    def test_round_trip(self) -> None:
        s = SentimentScore(
            ticker="AAA",
            date=datetime(2024, 1, 2, tzinfo=UTC),
            score=0.42,
            coverage=3,
            backend="stub",
        )
        assert s.score == 0.42
        assert s.backend == "stub"

    @pytest.mark.parametrize("score", [1.1, -1.5])
    def test_rejects_out_of_range_score(self, score: float) -> None:
        with pytest.raises(ValueError):
            SentimentScore(
                ticker="AAA",
                date=datetime(2024, 1, 2, tzinfo=UTC),
                score=score,
                coverage=1,
                backend="stub",
            )

    def test_rejects_negative_coverage(self) -> None:
        with pytest.raises(ValueError):
            SentimentScore(
                ticker="AAA",
                date=datetime(2024, 1, 2, tzinfo=UTC),
                score=0.0,
                coverage=-1,
                backend="stub",
            )

    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError):
            SentimentScore(
                ticker="AAA",
                date=datetime(2024, 1, 2, tzinfo=UTC),
                score=0.0,
                coverage=0,
                backend="bogus",
            )


class TestITreeRecord:
    def test_validates_array_alignment(self) -> None:
        with pytest.raises(ValueError):
            ITreeRecord(
                feature=np.zeros(3, dtype=int),
                split=np.zeros(2, dtype=float),
                left=np.full(3, -1, dtype=int),
                right=np.full(3, -1, dtype=int),
                size=np.ones(3, dtype=int),
                max_depth=2,
            )


class TestDetectorFit:
    def _make(self, p: int = 4) -> DetectorFit:
        tree = ITreeRecord(
            feature=np.array([-1], dtype=int),
            split=np.array([0.0]),
            left=np.array([-1]),
            right=np.array([-1]),
            size=np.array([1]),
            max_depth=1,
        )
        return DetectorFit(
            pca_mean=np.zeros(p),
            pca_components=np.eye(p)[:1],
            pca_explained_var=np.ones(1),
            iforest_trees=(tree,),
            iforest_psi=4,
            iforest_c_psi=1.0,
            mahal_mean=np.zeros(p),
            mahal_inv_cov=np.eye(p),
            feature_median=np.zeros(p),
            feature_scale=np.ones(p),
        )

    def test_round_trip(self) -> None:
        fit = self._make()
        assert fit.pca_components.shape == (1, 4)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            DetectorFit(
                pca_mean=np.zeros(4),
                pca_components=np.eye(4)[:1],
                pca_explained_var=np.ones(1),
                iforest_trees=(),  # empty -> rejected
                iforest_psi=4,
                iforest_c_psi=1.0,
                mahal_mean=np.zeros(4),
                mahal_inv_cov=np.eye(4),
                feature_median=np.zeros(4),
                feature_scale=np.ones(4),
            )


class TestAnomalyFit:
    def test_rejects_missing_threshold_key(self) -> None:
        # Re-uses the minimal DetectorFit fixture.
        tree = ITreeRecord(
            feature=np.array([-1], dtype=int),
            split=np.array([0.0]),
            left=np.array([-1]),
            right=np.array([-1]),
            size=np.array([1]),
            max_depth=1,
        )
        detector = DetectorFit(
            pca_mean=np.zeros(4),
            pca_components=np.eye(4)[:1],
            pca_explained_var=np.ones(1),
            iforest_trees=(tree,),
            iforest_psi=4,
            iforest_c_psi=1.0,
            mahal_mean=np.zeros(4),
            mahal_inv_cov=np.eye(4),
            feature_median=np.zeros(4),
            feature_scale=np.ones(4),
        )
        with pytest.raises(ValueError):
            AnomalyFit(
                detector=detector,
                thresholds={"ae": 1.0, "iforest": 0.5},  # missing 'mahalanobis'
                in_sample_scores={
                    "ae": np.zeros(2),
                    "iforest": np.zeros(2),
                    "mahalanobis": np.zeros(2),
                },
                feature_names=DEFAULT_FEATURE_NAMES,
                config=AnomalyConfig(),
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            )
