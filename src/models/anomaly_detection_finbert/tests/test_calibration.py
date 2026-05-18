"""Unit tests for `calibration.calibrate`.

We feed `AnomalyInputs` built directly from numpy panels and check:

- `AnomalyFit` has the three required threshold keys.
- The in-sample flag rate at each threshold matches `contamination`
  (up to the discreteness of `np.quantile`).
- The fit's preprocessing stats reproduce the standardize -> clip -> score
  pipeline if we re-apply them by hand.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.anomaly_detection_finbert.calibration import MODEL_NAME, calibrate
from src.models.anomaly_detection_finbert.signal import (
    impute_and_clip,
    median_mad_standardize,
    score_all_detectors,
)
from src.models.anomaly_detection_finbert.types import (
    DEFAULT_FEATURE_NAMES,
    AnomalyConfig,
    AnomalyFit,
    AnomalyInputs,
)


def _make_inputs(rng: np.random.Generator, n: int = 400) -> AnomalyInputs:
    p = len(DEFAULT_FEATURE_NAMES)
    X = rng.normal(0.0, 1.0, size=(n, p))
    features = pd.DataFrame(X, columns=list(DEFAULT_FEATURE_NAMES))
    return AnomalyInputs(
        features=features,
        feature_names=DEFAULT_FEATURE_NAMES,
        dates=np.arange(n).astype("datetime64[D]"),
        ticker_ids=np.zeros(n, dtype=int),
        tickers=("AAA",),
        news={"AAA": []},
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_returns_calibration_result(rng: np.random.Generator) -> None:
    data = _make_inputs(rng)
    result = calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == MODEL_NAME
    fit = result.parameters["anomaly_fit"]
    assert isinstance(fit, AnomalyFit)
    assert set(fit.thresholds.keys()) == {"ae", "iforest", "mahalanobis"}


def test_in_sample_flag_rate_matches_contamination(rng: np.random.Generator) -> None:
    contamination = 0.05
    data = _make_inputs(rng, n=500)
    result = calibrate(data, config=AnomalyConfig(contamination=contamination))
    fit = result.parameters["anomaly_fit"]
    for key, scores in fit.in_sample_scores.items():
        rate = float(np.mean(scores > fit.thresholds[key]))
        # `np.quantile` interpolates so we accept a small discretization band.
        assert rate <= contamination + 0.01, (
            f"{key} flag_rate={rate} exceeds contamination + 0.01"
        )


def test_threshold_matches_quantile(rng: np.random.Generator) -> None:
    data = _make_inputs(rng, n=400)
    cfg = AnomalyConfig(contamination=0.005)
    result = calibrate(data, config=cfg)
    fit = result.parameters["anomaly_fit"]
    for key, scores in fit.in_sample_scores.items():
        expected = float(np.quantile(scores, 1.0 - cfg.contamination))
        assert fit.thresholds[key] == expected


def test_preprocessing_stats_round_trip(rng: np.random.Generator) -> None:
    """Standardize / clip / score using the fit's stats reproduces in-sample scores."""

    data = _make_inputs(rng, n=300)
    result = calibrate(data, config=AnomalyConfig(iforest_n_trees=20, iforest_subsample=64))
    fit = result.parameters["anomaly_fit"]
    X_raw = data.features.to_numpy(dtype=float)
    X_std, _, _ = median_mad_standardize(
        X_raw,
        median=fit.detector.feature_median,
        scale=fit.detector.feature_scale,
    )
    X_clean = impute_and_clip(X_std, fit.config.clip_z)
    scores = score_all_detectors(X_clean, fit.detector)
    for key, arr in scores.items():
        assert np.allclose(arr, fit.in_sample_scores[key])
