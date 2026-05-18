"""Pure-numpy gradient-boosted regression trees (Friedman, 2001).

This is the math from the spec: at each stage `m`, fit a regression tree
to the negative-gradient pseudo-residuals of the loss, with shrinkage
`eta`, row subsampling `subsample`, and feature subsampling `colsample`.

Supports MSE (least squares) and Huber loss. The Huber gradient and
threshold-based pseudo-residuals follow Friedman (2001) §4.4. Tree splits
maximize the SSR-reduction gain analytically.

When `lightgbm`/`xgboost`/`catboost` are installed this module's
`OptionalLibBackend` shim can be wired in later (see README); the default
backend is the numpy implementation so the ensemble works in any env
that ships only the project's declared core dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.models.gbdt_transformer_ensembles.types import (
    VALID_LOSSES,
    GBDTFit,
    LossName,
    RegressionTree,
    TreeNode,
)

DEFAULT_HUBER_DELTA: float = 1.0


@dataclass(frozen=True)
class GBDTParams:
    """Hyperparameters for the numpy GBDT — defaults mirror the spec table."""

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 5
    min_samples_leaf: int = 20
    subsample: float = 0.8
    colsample: float = 0.8
    l2_leaf_reg: float = 1.0
    loss: LossName = "huber"
    huber_delta: float = DEFAULT_HUBER_DELTA
    early_stopping_rounds: int | None = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}"
            )
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError(
                f"subsample must be in (0, 1], got {self.subsample}"
            )
        if not 0.0 < self.colsample <= 1.0:
            raise ValueError(
                f"colsample must be in (0, 1], got {self.colsample}"
            )
        if self.l2_leaf_reg < 0:
            raise ValueError(
                f"l2_leaf_reg must be >= 0, got {self.l2_leaf_reg}"
            )
        if self.loss not in VALID_LOSSES:
            raise ValueError(
                f"loss must be in {sorted(VALID_LOSSES)}, got {self.loss!r}"
            )
        if self.huber_delta <= 0:
            raise ValueError(
                f"huber_delta must be > 0, got {self.huber_delta}"
            )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def loss_value(
    y: np.ndarray, pred: np.ndarray, w: np.ndarray, loss: LossName, delta: float
) -> float:
    """Weighted loss value (`sum w_i * ell_i / sum w_i`)."""

    r = y - pred
    if loss == "mse":
        per = 0.5 * r * r
    else:  # huber
        abs_r = np.abs(r)
        quad = 0.5 * r * r
        lin = delta * (abs_r - 0.5 * delta)
        per = np.where(abs_r <= delta, quad, lin)
    s_w = float(w.sum())
    if s_w <= 0:
        return float(per.mean())
    return float((w * per).sum() / s_w)


def negative_gradient(
    y: np.ndarray, pred: np.ndarray, loss: LossName, delta: float
) -> np.ndarray:
    """Pseudo-residuals = -dL/dF.

    - MSE: y - F.
    - Huber: y - F when |y - F| <= delta, else delta * sign(y - F).
    """

    r = np.asarray(y - pred, dtype=float)
    if loss == "mse":
        return r
    abs_r = np.abs(r)
    return np.asarray(np.where(abs_r <= delta, r, delta * np.sign(r)), dtype=float)


def initial_prediction(y: np.ndarray, w: np.ndarray, loss: LossName) -> float:
    """Loss-minimizing constant for the initialization F_0.

    - MSE: weighted mean.
    - Huber: weighted median (matches Friedman 2001 §4.4 initialization).
    """

    s_w = float(w.sum())
    if s_w <= 0:
        return float(np.mean(y))
    if loss == "mse":
        return float((w * y).sum() / s_w)
    return float(_weighted_median(y, w))


def _weighted_median(y: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(y, kind="mergesort")
    y_s = y[order]
    w_s = w[order]
    cw = np.cumsum(w_s)
    half = 0.5 * w_s.sum()
    idx = int(np.searchsorted(cw, half))
    idx = min(idx, len(y_s) - 1)
    return float(y_s[idx])


# ---------------------------------------------------------------------------
# Regression tree fitting
# ---------------------------------------------------------------------------


@dataclass
class _BuildState:
    """Mutable scratch space passed to the recursive split routine."""

    nodes: list[TreeNode]
    feature_gain: np.ndarray  # accumulator updated per split
    feature_split: np.ndarray  # accumulator updated per split


def _leaf_value(
    sum_g: float, sum_w: float, l2: float
) -> float:
    """Newton-step leaf value `-sum(g) / (sum(w) + lambda)` per spec eq."""

    denom = sum_w + l2
    if denom <= 0:
        return 0.0
    # Note: the optimal leaf value for the MSE/Huber pseudo-residual fit is
    # the weighted mean of the residuals — sum(w * r) / sum(w + lambda) — to
    # which the L2 leaf regularization shrinks the magnitude.
    return sum_g / denom


def _best_split_for_feature(
    x: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    min_samples_leaf: int,
    l2: float,
) -> tuple[float, float] | None:
    """Find the best threshold for one feature.

    Returns `(threshold, gain)` if a valid split exists, else `None`.
    Gain is computed from the SSR reduction formula

        gain = (S_L^2 / W_L + S_R^2 / W_R - (S_L + S_R)^2 / (W_L + W_R))

    with the leaf-L2 reg shifting each denominator by `l2`. We sort once
    by feature value and sweep left-to-right.
    """

    n = x.shape[0]
    if n < 2 * min_samples_leaf:
        return None
    order = np.argsort(x, kind="mergesort")
    x_s = x[order]
    g_s = g[order]
    w_s = w[order]

    total_g = float(g_s.sum())
    total_w = float(w_s.sum())
    parent_score = (total_g * total_g) / (total_w + l2) if (total_w + l2) > 0 else 0.0

    cum_g = np.cumsum(g_s)
    cum_w = np.cumsum(w_s)

    best_gain = -math.inf
    best_threshold = math.nan

    # Iterate candidate split points where the feature actually changes.
    # `i` is the count of elements going to the LEFT child.
    for i in range(min_samples_leaf, n - min_samples_leaf + 1):
        if x_s[i - 1] == x_s[i]:
            continue
        s_l = float(cum_g[i - 1])
        w_l = float(cum_w[i - 1])
        s_r = total_g - s_l
        w_r = total_w - w_l
        denom_l = w_l + l2
        denom_r = w_r + l2
        if denom_l <= 0 or denom_r <= 0:
            continue
        gain = (s_l * s_l) / denom_l + (s_r * s_r) / denom_r - parent_score
        if gain > best_gain:
            best_gain = gain
            best_threshold = 0.5 * (x_s[i - 1] + x_s[i])
    if best_gain <= 0 or not math.isfinite(best_threshold):
        return None
    return best_threshold, best_gain


def _grow_tree(
    X: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    feature_subset: np.ndarray,
    *,
    max_depth: int,
    min_samples_leaf: int,
    l2: float,
    feature_gain: np.ndarray,
    feature_split: np.ndarray,
) -> RegressionTree:
    """Fit one regression tree to the pseudo-residuals `g` with sample weights `w`.

    Recursive split-best-first depth-limited growth. Returns a flat-array
    `RegressionTree`. Internal nodes carry (feature_index, threshold);
    leaves carry the leaf value with `feature_index == -1`.
    """

    nodes: list[TreeNode] = []

    def build(indices: np.ndarray, depth: int) -> int:
        node_idx = len(nodes)
        sub_g = g[indices]
        sub_w = w[indices]
        leaf_val = _leaf_value(float(sub_g.sum()), float(sub_w.sum()), l2)
        # Reserve a leaf slot; we'll overwrite if we end up splitting.
        nodes.append(
            TreeNode(
                feature_index=-1,
                threshold=math.nan,
                value=leaf_val,
                left=-1,
                right=-1,
            )
        )
        if depth >= max_depth or indices.shape[0] < 2 * min_samples_leaf:
            return node_idx

        best_gain = -math.inf
        best_feat = -1
        best_thresh = math.nan
        for feat in feature_subset:
            split = _best_split_for_feature(
                X[indices, feat],
                sub_g,
                sub_w,
                min_samples_leaf=min_samples_leaf,
                l2=l2,
            )
            if split is None:
                continue
            thresh, gain = split
            if gain > best_gain:
                best_gain = gain
                best_feat = int(feat)
                best_thresh = thresh
        if best_feat < 0 or best_gain <= 0:
            return node_idx

        left_mask = X[indices, best_feat] <= best_thresh
        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]
        if left_idx.size < min_samples_leaf or right_idx.size < min_samples_leaf:
            return node_idx

        feature_gain[best_feat] += float(best_gain)
        feature_split[best_feat] += 1.0

        left_node = build(left_idx, depth + 1)
        right_node = build(right_idx, depth + 1)
        nodes[node_idx] = TreeNode(
            feature_index=best_feat,
            threshold=float(best_thresh),
            value=0.0,  # internal nodes don't have a leaf value
            left=left_node,
            right=right_node,
        )
        return node_idx

    n = X.shape[0]
    build(np.arange(n, dtype=np.int64), depth=0)
    return RegressionTree(nodes=tuple(nodes))


def predict_tree(tree: RegressionTree, X: np.ndarray) -> np.ndarray:
    """Vectorized tree prediction."""

    n = X.shape[0]
    out = np.zeros(n, dtype=float)
    if not tree.nodes:
        return out
    # Per-row walk. Trees here are small (<= depth 8) so the python loop is
    # fine; this could be JIT-compiled later if profiling demands it.
    for i in range(n):
        idx = 0
        while True:
            node = tree.nodes[idx]
            if node.feature_index == -1:
                out[i] = node.value
                break
            idx = node.left if X[i, node.feature_index] <= node.threshold else node.right
    return out


# ---------------------------------------------------------------------------
# Boosting loop
# ---------------------------------------------------------------------------


def fit_gbdt(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    feature_names: Sequence[str],
    *,
    params: GBDTParams,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
) -> GBDTFit:
    """Stagewise gradient-boosting fit.

    Pseudocode (matches the spec's "Algorithm outline"):

        F_0 := argmin_c sum_i w_i * ell(y_i, c)
        for m in 1..M:
            g_i := -dL/dF |_{F = F_{m-1}}
            (optionally subsample rows and columns)
            f_m := regression-tree fit to g_i with sample weights w_i
            F_m := F_{m-1} + eta * f_m
            track train (and val) loss; early-stop on val plateau
    """

    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y shape {y.shape} does not match X rows {n}")
    if w.shape != (n,):
        raise ValueError(f"w shape {w.shape} does not match X rows {n}")
    if len(feature_names) != p:
        raise ValueError(
            f"feature_names length {len(feature_names)} != X columns {p}"
        )
    has_val = X_val is not None
    if has_val:
        assert X_val is not None and y_val is not None and w_val is not None
        if X_val.shape[1] != p:
            raise ValueError(
                f"X_val has {X_val.shape[1]} features, expected {p}"
            )

    rng = np.random.default_rng(params.seed)
    init = initial_prediction(y, w, params.loss)
    train_pred = np.full(n, init, dtype=float)
    val_pred = (
        np.full(X_val.shape[0], init, dtype=float)
        if has_val and X_val is not None
        else None
    )

    trees: list[RegressionTree] = []
    feature_gain = np.zeros(p, dtype=float)
    feature_split = np.zeros(p, dtype=float)
    train_history: list[float] = []
    val_history: list[float] = []
    best_iter = 0
    best_val = math.inf
    rounds_no_improve = 0

    n_subsample = max(1, int(round(params.subsample * n)))
    n_colsample = max(1, int(round(params.colsample * p)))

    for m in range(params.n_estimators):
        g = negative_gradient(y, train_pred, params.loss, params.huber_delta)
        if params.subsample < 1.0:
            row_idx = rng.choice(n, size=n_subsample, replace=False)
        else:
            row_idx = np.arange(n, dtype=np.int64)
        if params.colsample < 1.0:
            feat_idx = rng.choice(p, size=n_colsample, replace=False)
        else:
            feat_idx = np.arange(p, dtype=np.int64)

        tree = _grow_tree(
            X[row_idx],
            g[row_idx],
            w[row_idx],
            feat_idx,
            max_depth=params.max_depth,
            min_samples_leaf=params.min_samples_leaf,
            l2=params.l2_leaf_reg,
            feature_gain=feature_gain,
            feature_split=feature_split,
        )
        trees.append(tree)
        increment = params.learning_rate * predict_tree(tree, X)
        train_pred = train_pred + increment
        train_history.append(
            loss_value(y, train_pred, w, params.loss, params.huber_delta)
        )

        if has_val and X_val is not None and y_val is not None and w_val is not None:
            assert val_pred is not None
            val_pred = val_pred + params.learning_rate * predict_tree(tree, X_val)
            v = loss_value(y_val, val_pred, w_val, params.loss, params.huber_delta)
            val_history.append(v)
            if v < best_val - 1e-12:
                best_val = v
                best_iter = m + 1
                rounds_no_improve = 0
            else:
                rounds_no_improve += 1
                if (
                    params.early_stopping_rounds is not None
                    and rounds_no_improve >= params.early_stopping_rounds
                ):
                    break
        else:
            best_iter = m + 1

    return GBDTFit(
        trees=tuple(trees),
        learning_rate=params.learning_rate,
        init_value=init,
        loss=params.loss,
        feature_names=tuple(feature_names),
        feature_importance_gain=feature_gain,
        feature_importance_split=feature_split,
        train_loss_history=tuple(train_history),
        val_loss_history=tuple(val_history) if val_history else None,
        best_iteration=best_iter,
    )


def predict_gbdt(fit: GBDTFit, X: np.ndarray, n_trees: int | None = None) -> np.ndarray:
    """Sum of tree contributions through `n_trees` (default: all)."""

    if X.shape[1] != fit.n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, expected {fit.n_features}"
        )
    cap = fit.n_trees if n_trees is None else min(n_trees, fit.n_trees)
    out = np.full(X.shape[0], fit.init_value, dtype=float)
    for tree in fit.trees[:cap]:
        out = out + fit.learning_rate * predict_tree(tree, X)
    return out


__all__ = [
    "GBDTParams",
    "fit_gbdt",
    "initial_prediction",
    "loss_value",
    "negative_gradient",
    "predict_gbdt",
    "predict_tree",
]
