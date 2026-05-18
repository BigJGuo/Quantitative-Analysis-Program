"""Public interface for the Anomaly Detection + FinBERT model.

See `models/layer4_signals_ml/14_anomaly_detection_finbert.md` for the full
specification.

Heavier optional backends (`torch`, `sklearn`, `transformers`) are imported
lazily — importing this package does *not* pull them in, so the rest of the
ecosystem can introspect the model without the heavy ML stack installed.
"""

from __future__ import annotations

from src.models.anomaly_detection_finbert.calibration import MODEL_NAME, calibrate
from src.models.anomaly_detection_finbert.model import AnomalyDetectionFinBERT
from src.models.anomaly_detection_finbert.signal import (
    build_anomaly_feature_panel,
    c_psi,
    cross_sectional_rank,
    exp_decay_weight,
    fit_iforest,
    fit_mahalanobis,
    iforest_score,
    impute_and_clip,
    mahalanobis_score,
    majority_vote_flags,
    median_mad_standardize,
    pca_fit,
    pca_reconstruct,
    pca_reconstruction_score,
    pca_residual_attribution,
    score_all_detectors,
    sentiment_scalar,
    split_sentences,
    threshold_from_quantile,
    top_feature_attribution,
    weighted_sentiment,
)
from src.models.anomaly_detection_finbert.types import (
    DEFAULT_FEATURE_NAMES,
    AnomalyAlert,
    AnomalyConfig,
    AnomalyFit,
    AnomalyInputs,
    DetectorFit,
    ITreeRecord,
    SentimentConfig,
    SentimentScore,
)

__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "MODEL_NAME",
    "AnomalyAlert",
    "AnomalyConfig",
    "AnomalyDetectionFinBERT",
    "AnomalyFit",
    "AnomalyInputs",
    "DetectorFit",
    "ITreeRecord",
    "SentimentConfig",
    "SentimentScore",
    "build_anomaly_feature_panel",
    "c_psi",
    "calibrate",
    "cross_sectional_rank",
    "exp_decay_weight",
    "fit_iforest",
    "fit_mahalanobis",
    "iforest_score",
    "impute_and_clip",
    "mahalanobis_score",
    "majority_vote_flags",
    "median_mad_standardize",
    "pca_fit",
    "pca_reconstruct",
    "pca_reconstruction_score",
    "pca_residual_attribution",
    "score_all_detectors",
    "sentiment_scalar",
    "split_sentences",
    "threshold_from_quantile",
    "top_feature_attribution",
    "weighted_sentiment",
]
