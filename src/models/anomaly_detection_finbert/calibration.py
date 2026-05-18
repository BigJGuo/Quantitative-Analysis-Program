"""Calibration entry point for the Anomaly Detection + FinBERT model.

`calibrate(data, **params)` trains the three detectors on the supplied
feature panel and returns an `AnomalyFit` wrapped in a `CalibrationResult`:

1. Robust-standardize the feature panel (median / MAD, clip ±clip_z).
2. Fit a PCA reconstruction model (linear AE) at the requested bottleneck.
3. Fit an isolation forest of `n_trees` iTrees on subsamples of size ψ.
4. Fit a shrinkage Mahalanobis distance (Gaussian one-class baseline).
5. Score the *training* rows under each detector and set the empirical
   `1 - contamination` quantile as that detector's flag threshold.

The function is pure data-in / parameters-out; the orchestration shell in
`model.py` handles I/O and signal emission.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from src.core.types import CalibrationResult
from src.models.anomaly_detection_finbert.signal import c_psi as _c_psi
from src.models.anomaly_detection_finbert.signal import (
    fit_iforest,
    fit_mahalanobis,
    iforest_score,
    impute_and_clip,
    mahalanobis_score,
    median_mad_standardize,
    pca_fit,
    pca_reconstruction_score,
    threshold_from_quantile,
)
from src.models.anomaly_detection_finbert.types import (
    AnomalyConfig,
    AnomalyFit,
    AnomalyInputs,
    DetectorFit,
)

MODEL_NAME: str = "anomaly_detection_finbert"


def calibrate(
    data: AnomalyInputs,
    *,
    config: AnomalyConfig | None = None,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Train all three detectors and assemble an `AnomalyFit`.

    Parameters
    ----------
    data:
        Panel produced by the model's `fetch_data`.
    config:
        `AnomalyConfig` overrides. Defaults to `AnomalyConfig()`.
    timestamp:
        Calibration timestamp; defaults to `datetime.now(UTC)`.
    """

    cfg = config or AnomalyConfig()
    ts = timestamp or datetime.now(UTC)

    X_raw = data.features.to_numpy(dtype=float)
    if X_raw.shape[0] < 2:
        raise ValueError(
            f"calibrate requires N >= 2 rows, got {X_raw.shape[0]}"
        )

    # ---- Robust preprocessing -------------------------------------------
    X_std, feature_median, feature_scale = median_mad_standardize(X_raw)
    X_clean = impute_and_clip(X_std, cfg.clip_z)

    # ---- Detector 1: PCA reconstruction (linear AE) --------------------
    p = X_clean.shape[1]
    bottleneck = min(cfg.bottleneck_dim, p)
    pca_mean, pca_components, pca_var = pca_fit(X_clean, n_components=bottleneck)
    ae_scores = pca_reconstruction_score(X_clean, pca_mean, pca_components)

    # ---- Detector 2: Isolation forest ----------------------------------
    psi = min(cfg.iforest_subsample, X_clean.shape[0])
    trees = fit_iforest(
        X_clean,
        n_trees=cfg.iforest_n_trees,
        psi=psi,
        seed=cfg.iforest_seed,
    )
    iforest_scores = iforest_score(X_clean, trees, psi)
    iforest_normalizer = _c_psi(psi)

    # ---- Detector 3: Mahalanobis (OCSVM baseline) ----------------------
    mahal_mean, mahal_inv_cov = fit_mahalanobis(X_clean, cfg.mahal_shrinkage)
    mahal_scores = mahalanobis_score(X_clean, mahal_mean, mahal_inv_cov)

    detector = DetectorFit(
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_explained_var=pca_var,
        iforest_trees=trees,
        iforest_psi=psi,
        iforest_c_psi=float(iforest_normalizer),
        mahal_mean=mahal_mean,
        mahal_inv_cov=mahal_inv_cov,
        feature_median=feature_median,
        feature_scale=feature_scale,
    )

    thresholds = {
        "ae": threshold_from_quantile(ae_scores, cfg.contamination),
        "iforest": threshold_from_quantile(iforest_scores, cfg.contamination),
        "mahalanobis": threshold_from_quantile(mahal_scores, cfg.contamination),
    }
    in_sample = {
        "ae": ae_scores,
        "iforest": iforest_scores,
        "mahalanobis": mahal_scores,
    }
    fit = AnomalyFit(
        detector=detector,
        thresholds=thresholds,
        in_sample_scores=in_sample,
        feature_names=data.feature_names,
        config=cfg,
        timestamp=ts,
    )

    fit_metrics = _fit_metrics(fit)
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={
            "anomaly_fit": fit,
            "config": cfg,
        },
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata={
            "n_rows": float(X_clean.shape[0]),
            "n_features": float(X_clean.shape[1]),
            "bottleneck_dim": float(bottleneck),
            "n_trees": float(cfg.iforest_n_trees),
            "iforest_psi": float(psi),
            "contamination": float(cfg.contamination),
        },
    )


def _fit_metrics(fit: AnomalyFit) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, scores in fit.in_sample_scores.items():
        metrics[f"{key}_mean"] = float(np.mean(scores))
        metrics[f"{key}_median"] = float(np.median(scores))
        metrics[f"{key}_p99"] = float(np.quantile(scores, 0.99))
        metrics[f"{key}_threshold"] = float(fit.thresholds[key])
    # Per-detector flag rate at the configured threshold (should equal
    # contamination, up to discreteness).
    for key, scores in fit.in_sample_scores.items():
        flag_rate = float(np.mean(scores > fit.thresholds[key]))
        metrics[f"{key}_flag_rate"] = flag_rate
    return metrics


__all__ = ["MODEL_NAME", "calibrate"]
