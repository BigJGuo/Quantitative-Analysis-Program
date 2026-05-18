"""Dataclasses specific to the Anomaly Detection + FinBERT model.

The model produces three kinds of artifacts:

- `AnomalyInputs`  — feature panel + news payload (built by `fetch_data`).
- `DetectorFit`    — the trained AE / iForest / Mahalanobis parameters.
- `AnomalyFit`     — `DetectorFit` plus per-detector quantile thresholds.

Plus two diagnostic shapes used at inference:

- `AnomalyAlert`   — flag, per-detector scores, and top-feature attribution.
- `SentimentScore` — scalar sentiment for a (ticker, date), with provenance.

Heavy backends (torch, sklearn, transformers) are not imported here; the
fit dataclasses store opaque arrays plus enough metadata to recreate any
state at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "log_return_1d",
    "intraday_range",
    "open_close_gap",
    "log_volume",
    "vol_ratio_20d",
    "realized_vol_20d",
    "ma_drift_50",
    "abs_return_zscore",
    "cross_sectional_rank",
)


@dataclass(frozen=True)
class AnomalyConfig:
    """Hyperparameters for the three numerical detectors.

    Defaults follow the spec's "Architecture / model design" section:

    - PCA-AE bottleneck = 8 (spec uses a 16-d bottleneck for a deep AE; PCA
      is a linear AE and needs fewer dimensions for an equivalent fit).
    - iForest with 100 trees, subsample 256, sklearn default.
    - Mahalanobis with a small ridge shrinkage so singular covariance does
      not explode on tiny panels.
    - Contamination = 0.005 (the 99.5th in-sample quantile is flagged).
    """

    bottleneck_dim: int = 8
    iforest_n_trees: int = 100
    iforest_subsample: int = 256
    iforest_seed: int = 0
    mahal_shrinkage: float = 1e-3
    contamination: float = 0.005
    clip_z: float = 5.0

    def __post_init__(self) -> None:
        if self.bottleneck_dim < 1:
            raise ValueError(f"bottleneck_dim must be >= 1, got {self.bottleneck_dim}")
        if self.iforest_n_trees < 1:
            raise ValueError(f"iforest_n_trees must be >= 1, got {self.iforest_n_trees}")
        if self.iforest_subsample < 2:
            raise ValueError(
                f"iforest_subsample must be >= 2, got {self.iforest_subsample}"
            )
        if not 0.0 < self.contamination < 0.5:
            raise ValueError(
                f"contamination must lie in (0, 0.5), got {self.contamination}"
            )
        if self.mahal_shrinkage < 0:
            raise ValueError(
                f"mahal_shrinkage must be >= 0, got {self.mahal_shrinkage}"
            )
        if self.clip_z <= 0:
            raise ValueError(f"clip_z must be > 0, got {self.clip_z}")


@dataclass(frozen=True)
class SentimentConfig:
    """FinBERT sentence-aggregation hyperparameters.

    `decay_hours` is the time constant of the exponential weight on article
    age — half-life of `decay_hours * ln 2`. Default 24h means a 1-day-old
    article gets weight 1/e ≈ 0.37 of a same-day one.
    """

    decay_hours: float = 24.0
    eps: float = 1e-9
    finbert_model_id: str = "ProsusAI/finbert"
    max_seq_len: int = 256
    backend: str = "auto"  # "auto" | "finbert" | "stub"

    def __post_init__(self) -> None:
        if self.decay_hours <= 0:
            raise ValueError(f"decay_hours must be > 0, got {self.decay_hours}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.max_seq_len < 16:
            raise ValueError(f"max_seq_len must be >= 16, got {self.max_seq_len}")
        if self.backend not in {"auto", "finbert", "stub"}:
            raise ValueError(
                f"backend must be one of 'auto'|'finbert'|'stub', got {self.backend!r}"
            )


@dataclass(frozen=True)
class AnomalyInputs:
    """Inputs to `calibrate`, `predict`, and `validate`.

    `features` is a (N, p) panel — one row per (ticker, date) — indexed by
    ``date``. `news` is keyed by ticker so the sentiment branch can look up
    news independently from the price panel.
    """

    features: pd.DataFrame
    feature_names: tuple[str, ...]
    dates: np.ndarray
    ticker_ids: np.ndarray
    tickers: tuple[str, ...]
    news: dict[str, list[dict[str, Any]]]
    timestamp: datetime = field(default_factory=lambda: datetime.fromtimestamp(0))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if n == 0:
            raise ValueError("AnomalyInputs.features must be non-empty")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features has {self.features.shape[1]} columns but "
                f"{len(self.feature_names)} feature_names were supplied"
            )
        if self.dates.shape != (n,):
            raise ValueError(
                f"dates shape {self.dates.shape} does not match N={n}"
            )
        if self.ticker_ids.shape != (n,):
            raise ValueError(
                f"ticker_ids shape {self.ticker_ids.shape} does not match N={n}"
            )


@dataclass(frozen=True)
class ITreeRecord:
    """One isolation tree stored as parallel arrays for fast scoring.

    Each node has either two children (an internal split) or is a leaf with
    `size` recording how many training points it isolated. We store the
    nodes as flat arrays so a `dict`/torch is not needed.
    """

    feature: np.ndarray  # (n_nodes,) int; -1 for leaf
    split: np.ndarray  # (n_nodes,) float
    left: np.ndarray  # (n_nodes,) int; -1 for leaf
    right: np.ndarray  # (n_nodes,) int; -1 for leaf
    size: np.ndarray  # (n_nodes,) int; meaningful only at leaves
    max_depth: int

    def __post_init__(self) -> None:
        n = self.feature.shape[0]
        for name, arr in (
            ("split", self.split),
            ("left", self.left),
            ("right", self.right),
            ("size", self.size),
        ):
            if arr.shape != (n,):
                raise ValueError(
                    f"ITreeRecord arrays must share length; {name} has {arr.shape}"
                )
        if self.max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {self.max_depth}")


@dataclass(frozen=True)
class DetectorFit:
    """Concrete parameters of the three numerical detectors.

    - PCA-AE: ``pca_mean`` (p,) and ``pca_components`` (d, p).
    - iForest: tuple of `ITreeRecord`s plus the path-length normalizer.
    - Mahalanobis: ``mahal_mean`` (p,) and ``mahal_inv_cov`` (p, p).

    `feature_median` / `feature_scale` capture the train-time robust
    standardization stats so test-time rows go through the same affine map.
    """

    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained_var: np.ndarray
    iforest_trees: tuple[ITreeRecord, ...]
    iforest_psi: int
    iforest_c_psi: float
    mahal_mean: np.ndarray
    mahal_inv_cov: np.ndarray
    feature_median: np.ndarray
    feature_scale: np.ndarray

    def __post_init__(self) -> None:
        if self.pca_mean.ndim != 1:
            raise ValueError(
                f"pca_mean must be 1-D, got shape {self.pca_mean.shape}"
            )
        p = self.pca_mean.shape[0]
        if self.pca_components.ndim != 2 or self.pca_components.shape[1] != p:
            raise ValueError(
                f"pca_components must be (d, {p}), got {self.pca_components.shape}"
            )
        if self.mahal_mean.shape != (p,):
            raise ValueError(
                f"mahal_mean shape {self.mahal_mean.shape} != ({p},)"
            )
        if self.mahal_inv_cov.shape != (p, p):
            raise ValueError(
                f"mahal_inv_cov shape {self.mahal_inv_cov.shape} != ({p}, {p})"
            )
        if self.feature_median.shape != (p,) or self.feature_scale.shape != (p,):
            raise ValueError(
                "feature_median / feature_scale must have shape (p,) matching pca_mean"
            )
        if not self.iforest_trees:
            raise ValueError("iforest_trees must contain at least one tree")
        if self.iforest_psi < 2:
            raise ValueError(f"iforest_psi must be >= 2, got {self.iforest_psi}")


@dataclass(frozen=True)
class AnomalyFit:
    """`DetectorFit` plus thresholds and reference distributions.

    `thresholds` keys: `"ae"`, `"iforest"`, `"mahalanobis"` — the empirical
    `1 - contamination` quantile of each detector's in-sample score.

    `in_sample_scores` stores the full training-set score arrays so callers
    can re-derive thresholds at different contamination levels without
    refitting.
    """

    detector: DetectorFit
    thresholds: dict[str, float]
    in_sample_scores: dict[str, np.ndarray]
    feature_names: tuple[str, ...]
    config: AnomalyConfig
    timestamp: datetime

    def __post_init__(self) -> None:
        for key in ("ae", "iforest", "mahalanobis"):
            if key not in self.thresholds:
                raise ValueError(
                    f"thresholds missing required key {key!r}; got "
                    f"{sorted(self.thresholds)}"
                )
            if key not in self.in_sample_scores:
                raise ValueError(
                    f"in_sample_scores missing required key {key!r}"
                )


@dataclass(frozen=True)
class AnomalyAlert:
    """One row's alert payload.

    `combined` is the majority-vote flag (`>= 2 of 3 detectors fired`). The
    `top_features` list ranks the standardized residuals from the PCA-AE
    reconstruction so the operator can see *which* features drove the alert.
    """

    ticker: str
    date: datetime
    scores: dict[str, float]
    flags: dict[str, bool]
    combined: bool
    top_features: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SentimentScore:
    """Aggregated sentiment for a (ticker, date).

    `score ∈ [-1, 1]`; `backend` is one of `"finbert"`, `"stub"`, or
    `"empty"` (no articles found). `coverage` is the number of articles
    actually scored.
    """

    ticker: str
    date: datetime
    score: float
    coverage: int
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 - 1e-9 <= self.score <= 1.0 + 1e-9:
            raise ValueError(f"score must lie in [-1, 1], got {self.score}")
        if self.coverage < 0:
            raise ValueError(f"coverage must be >= 0, got {self.coverage}")
        if self.backend not in {"finbert", "stub", "empty"}:
            raise ValueError(
                f"backend must be 'finbert'|'stub'|'empty', got {self.backend!r}"
            )


__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "AnomalyAlert",
    "AnomalyConfig",
    "AnomalyFit",
    "AnomalyInputs",
    "DetectorFit",
    "ITreeRecord",
    "SentimentConfig",
    "SentimentScore",
]
