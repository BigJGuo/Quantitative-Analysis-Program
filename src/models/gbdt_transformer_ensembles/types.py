"""Dataclasses for the GBDT + Transformer ensemble (Model 11).

The ensemble is a stack of heterogeneous base learners — gradient-boosted
decision trees plus a time-series-aware linear "sequence" model — whose
out-of-fold predictions feed a ridge meta-learner. These dataclasses are
the shared shape every module in this package agrees on.

See `models/layer4_signals_ml/11_gbdt_transformer_ensembles.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

LossName = Literal["mse", "huber"]
BaseLearnerName = Literal["gbdt", "linear_seq"]

VALID_LOSSES: frozenset[str] = frozenset({"mse", "huber"})
VALID_BASE_LEARNERS: frozenset[str] = frozenset({"gbdt", "linear_seq"})


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for the per-ticker / per-date feature engineering pass.

    The defaults match the spec's "yfinance feature matrix" section closely
    while staying tractable on the universe sizes used in tests.
    """

    return_horizons: tuple[int, ...] = (1, 5, 10, 21, 63)
    lag_count: int = 10
    vol_windows: tuple[int, ...] = (5, 21, 63)
    mean_windows: tuple[int, ...] = (5, 21, 63)
    target_horizon: int = 5
    clip_target: float | None = 0.2
    include_cross_sectional_ranks: bool = True
    include_calendar: bool = True
    benchmark_ticker: str | None = "SPY"
    min_history_days: int = 252  # ~1y; spec calls for 2y on real data
    nan_sentinel: float = -1.0

    def __post_init__(self) -> None:
        if self.lag_count < 0:
            raise ValueError(f"lag_count must be >= 0, got {self.lag_count}")
        if self.target_horizon < 1:
            raise ValueError(
                f"target_horizon must be >= 1, got {self.target_horizon}"
            )
        if any(h < 1 for h in self.return_horizons):
            raise ValueError("return_horizons must all be >= 1")
        if any(w < 2 for w in self.vol_windows):
            raise ValueError("vol_windows must all be >= 2")
        if any(w < 1 for w in self.mean_windows):
            raise ValueError("mean_windows must all be >= 1")
        if self.clip_target is not None and self.clip_target <= 0:
            raise ValueError(
                f"clip_target must be positive or None, got {self.clip_target}"
            )


@dataclass(frozen=True)
class PanelFrame:
    """Long-format feature panel with target and sample weights.

    `X` has columns = feature names, rows aligned to `meta` (date, ticker).
    `y` and `w` are 1-D series of the same length. `feature_names` is the
    canonical ordering used by every downstream learner.
    """

    X: pd.DataFrame
    y: pd.Series
    w: pd.Series
    meta: pd.DataFrame  # columns: date, ticker
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.X)
        if not (len(self.y) == n == len(self.w) == len(self.meta)):
            raise ValueError(
                "PanelFrame: X, y, w, meta lengths must match — "
                f"got {len(self.X)}, {len(self.y)}, {len(self.w)}, {len(self.meta)}"
            )
        if list(self.X.columns) != list(self.feature_names):
            raise ValueError(
                "PanelFrame.X.columns must equal feature_names in order"
            )
        for col in ("date", "ticker"):
            if col not in self.meta.columns:
                raise ValueError(f"PanelFrame.meta missing column {col!r}")

    @property
    def n_rows(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class FoldSpec:
    """One purged-K-fold split over a date axis.

    Indices are positions into the *date* array (not row positions in the
    panel) — the panel-row mapping is materialized when training a base
    learner. This keeps the CV machinery symmetric with the spec, which
    purges/embargoes on the date axis.
    """

    fold_id: int
    train_dates: np.ndarray  # 1-D int array of date positions
    val_dates: np.ndarray


@dataclass(frozen=True)
class TreeNode:
    """One node of a regression tree.

    Leaf nodes have `feature_index == -1` and `value` set. Internal nodes
    use `feature_index` and `threshold`: left child = `samples <= threshold`,
    right child = `samples > threshold`.
    """

    feature_index: int
    threshold: float
    value: float
    left: int  # child node index in the tree's flat list (-1 for leaf)
    right: int


@dataclass(frozen=True)
class RegressionTree:
    """Flat-array regression tree. `predict()` walks `nodes` from index 0.

    Pure data — the fitting algorithm lives in `gbdt.py`.
    """

    nodes: tuple[TreeNode, ...]

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self.nodes if n.feature_index == -1)

    @property
    def depth(self) -> int:
        # Iterative DFS — depth is the longest root-to-leaf walk.
        if not self.nodes:
            return 0
        stack: list[tuple[int, int]] = [(0, 0)]
        max_d = 0
        while stack:
            idx, d = stack.pop()
            node = self.nodes[idx]
            if node.feature_index == -1:
                max_d = max(max_d, d)
                continue
            stack.append((node.left, d + 1))
            stack.append((node.right, d + 1))
        return max_d


@dataclass(frozen=True)
class GBDTFit:
    """Fitted GBDT: an initialization plus an ordered ensemble of trees.

    `learning_rate` and `loss` are stored so `predict()` can reproduce the
    additive expansion deterministically.
    """

    trees: tuple[RegressionTree, ...]
    learning_rate: float
    init_value: float
    loss: LossName
    feature_names: tuple[str, ...]
    feature_importance_gain: np.ndarray
    feature_importance_split: np.ndarray
    train_loss_history: tuple[float, ...]
    val_loss_history: tuple[float, ...] | None
    best_iteration: int

    @property
    def n_features(self) -> int:
        return int(self.feature_importance_gain.shape[0])

    @property
    def n_trees(self) -> int:
        return len(self.trees)


@dataclass(frozen=True)
class LinearSeqFit:
    """Fitted linear sequence model.

    The sequence model expands each feature column into a small bank of
    rolling aggregates per ticker, then fits a ridge regression on the
    augmented column set. `aggregates` records the order of aggregates so
    `predict()` can recreate the exact same column layout.
    """

    coefficients: np.ndarray  # (n_expanded_features,)
    intercept: float
    feature_names: tuple[str, ...]  # original feature names
    aggregates: tuple[str, ...]  # e.g., ("last", "mean_5", "std_5")
    expanded_feature_names: tuple[str, ...]
    alpha: float


@dataclass(frozen=True)
class StackingFit:
    """Linear stacker over base-learner OOF predictions."""

    coefficients: np.ndarray  # (n_base_learners,)
    intercept: float
    base_learner_names: tuple[str, ...]
    alpha: float


@dataclass(frozen=True)
class EnsembleFit:
    """Top-level fitted ensemble — everything `predict` needs at inference."""

    base_fits: dict[str, Any]
    stacker: StackingFit | None
    feature_names: tuple[str, ...]
    base_learner_names: tuple[str, ...]
    target_horizon: int
    clip_target: float | None
    oof_predictions: pd.DataFrame  # columns = base learner names + ("y", "w", "date", "ticker")
    cv_metrics: dict[str, float]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def has_stacker(self) -> bool:
        return self.stacker is not None


@dataclass(frozen=True)
class EnsemblePrediction:
    """Cross-sectional prediction at a single timestamp.

    `values` is the final blended prediction (stacked or averaged across
    base learners). `per_learner` retains the individual base-learner
    predictions for diagnostics.
    """

    timestamp: datetime
    tickers: tuple[str, ...]
    values: np.ndarray
    per_learner: dict[str, np.ndarray]
    horizon: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.tickers)
        if self.values.shape != (n,):
            raise ValueError(
                f"values shape {self.values.shape} must be ({n},)"
            )
        for name, arr in self.per_learner.items():
            if arr.shape != (n,):
                raise ValueError(
                    f"per_learner[{name!r}] shape {arr.shape} must be ({n},)"
                )


def coerce_base_learners(
    names: Sequence[str] | None,
) -> tuple[BaseLearnerName, ...]:
    """Validate and dedupe a sequence of base-learner names."""

    if names is None:
        return ("gbdt", "linear_seq")
    seen: list[BaseLearnerName] = []
    for n in names:
        if n not in VALID_BASE_LEARNERS:
            raise ValueError(
                f"Unknown base learner {n!r}; must be in "
                f"{sorted(VALID_BASE_LEARNERS)}"
            )
        typed = cast(BaseLearnerName, n)
        if typed not in seen:
            seen.append(typed)
    if not seen:
        raise ValueError("base_learners must be non-empty")
    return tuple(seen)
