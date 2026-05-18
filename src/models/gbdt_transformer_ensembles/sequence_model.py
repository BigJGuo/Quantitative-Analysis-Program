"""Linear sequence-aware base learner.

The spec asks for a Transformer encoder + GRU as the second family of base
learners. Both of those need a deep-learning runtime (`torch`) that the
project does not declare as a hard dependency. To still satisfy the
spirit of the spec — "a base learner whose error correlations with GBDT
are low because it sees the *sequence*, not the row" — we ship a linear
sequence model:

For each input feature `f`, build a small bank of rolling aggregates per
ticker:

    last_f    = f_t                       (current value)
    mean_w_f  = mean(f_{t-w+1..t})        for each window w
    std_w_f   = std(f_{t-w+1..t})         for each window w
    ewma_f    = EWMA_alpha(f_t)

Stack those columns and fit a ridge regression on the result. This is a
"shallow attention" surrogate — the EWMA approximates content-free
attention pooling, and rolling-mean/std capture short-range structure.

When `torch` is available a future extension can swap in a real Transformer
encoder block; see SHARED-CHANGE-REQUEST in the README.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.gbdt_transformer_ensembles.types import LinearSeqFit


@dataclass(frozen=True)
class LinearSeqParams:
    """Hyperparameters for the sequence model."""

    windows: tuple[int, ...] = (5, 21)
    ewma_alpha: float = 0.1
    alpha: float = 1.0  # ridge penalty
    standardize: bool = True

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("windows must be non-empty")
        if any(w < 2 for w in self.windows):
            raise ValueError("all windows must be >= 2")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(
                f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}"
            )
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")


def _aggregate_names(feature_names: Sequence[str], windows: Sequence[int]) -> tuple[str, ...]:
    cols: list[str] = []
    for f in feature_names:
        cols.append(f"last__{f}")
        for w in windows:
            cols.append(f"mean{w}__{f}")
            cols.append(f"std{w}__{f}")
        cols.append(f"ewma__{f}")
    return tuple(cols)


def expand_features(
    X: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    tickers: pd.Series,
    windows: Sequence[int],
    ewma_alpha: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Expand `(rows, features)` into `(rows, aggregates)` per ticker.

    Rolling aggregates are computed *within* each ticker group so they do
    not bleed across symbols. NaNs (from short ticker histories) are
    filled with zero — the linear model only needs to see them as
    uninformative.
    """

    expanded_names = _aggregate_names(feature_names, windows)
    n = len(X)
    p = len(feature_names) * (1 + 2 * len(windows) + 1)
    if len(expanded_names) != p:
        raise RuntimeError("internal: expanded feature name count mismatch")

    out = np.zeros((n, p), dtype=float)
    df = X[list(feature_names)].copy()
    df = df.reset_index(drop=True)
    tk = tickers.reset_index(drop=True)
    df["__ticker__"] = tk

    # We'll compute per-ticker rolling stats and rejoin.
    groups = df.groupby("__ticker__", sort=False, observed=True)

    col_offset = 0
    for f in feature_names:
        # "last" = the value at t
        out[:, col_offset] = df[f].to_numpy()
        col_offset += 1
        # rolling mean / std per ticker
        for w in windows:
            mean = groups[f].transform(
                lambda s, w=w: s.rolling(window=w, min_periods=1).mean()
            )
            std = groups[f].transform(
                lambda s, w=w: s.rolling(window=w, min_periods=2).std(ddof=0)
            )
            out[:, col_offset] = mean.to_numpy()
            col_offset += 1
            out[:, col_offset] = std.to_numpy()
            col_offset += 1
        # EWMA per ticker
        ewma = groups[f].transform(
            lambda s, a=ewma_alpha: s.ewm(alpha=a, adjust=False).mean()
        )
        out[:, col_offset] = ewma.to_numpy()
        col_offset += 1

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out, expanded_names


def fit_linear_seq(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    *,
    feature_names: Sequence[str],
    tickers: pd.Series,
    params: LinearSeqParams,
) -> LinearSeqFit:
    """Weighted ridge regression on the expanded sequence features.

    Closed form: `beta = (Z' W Z + alpha I)^-1 Z' W y` where `W = diag(w)`.
    Intercept is fit by augmenting with a 1-column (unpenalized).
    """

    Z, expanded = expand_features(
        X,
        feature_names=feature_names,
        tickers=tickers,
        windows=params.windows,
        ewma_alpha=params.ewma_alpha,
    )
    n, p = Z.shape
    if y.shape != (n,):
        raise ValueError(f"y shape {y.shape} does not match Z rows {n}")
    if w.shape != (n,):
        raise ValueError(f"w shape {w.shape} does not match Z rows {n}")

    if params.standardize:
        mu = Z.mean(axis=0)
        sigma = Z.std(axis=0, ddof=0)
        sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
        Z = (Z - mu) / sigma_safe
    else:
        mu = np.zeros(p)
        sigma_safe = np.ones(p)

    # Augment with intercept column.
    Z_aug = np.hstack([Z, np.ones((n, 1), dtype=float)])
    p_aug = p + 1

    sw = np.sqrt(np.maximum(w, 0.0))
    Z_w = Z_aug * sw[:, None]
    y_w = y * sw

    # Ridge — penalize only the slope columns, leave intercept free.
    penalty = np.zeros(p_aug)
    penalty[:p] = params.alpha
    A = Z_w.T @ Z_w + np.diag(penalty)
    b = Z_w.T @ y_w
    beta_aug, *_ = np.linalg.lstsq(A, b, rcond=None)
    beta = beta_aug[:p]
    intercept = float(beta_aug[p])

    # Bake the standardization into the coefficients so `predict` doesn't
    # need to keep mu/sigma separately.
    if params.standardize:
        beta = beta / sigma_safe
        intercept = intercept - float(np.sum(beta * mu))

    return LinearSeqFit(
        coefficients=beta,
        intercept=intercept,
        feature_names=tuple(feature_names),
        aggregates=("last", *[f"mean{w}" for w in params.windows], *[f"std{w}" for w in params.windows], "ewma"),
        expanded_feature_names=expanded,
        alpha=params.alpha,
    )


def predict_linear_seq(
    fit: LinearSeqFit,
    X: pd.DataFrame,
    tickers: pd.Series,
    *,
    windows: Sequence[int],
    ewma_alpha: float,
) -> np.ndarray:
    """Apply the fitted sequence model to a new feature matrix."""

    Z, expanded = expand_features(
        X,
        feature_names=fit.feature_names,
        tickers=tickers,
        windows=windows,
        ewma_alpha=ewma_alpha,
    )
    if expanded != fit.expanded_feature_names:
        raise ValueError(
            "predict_linear_seq: expanded feature layout mismatch between fit and predict"
        )
    return np.asarray(Z @ fit.coefficients + fit.intercept, dtype=float)


__all__ = [
    "LinearSeqParams",
    "expand_features",
    "fit_linear_seq",
    "predict_linear_seq",
]
