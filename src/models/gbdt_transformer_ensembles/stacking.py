"""Ridge stacking meta-learner over base-learner OOF predictions.

The stacker takes the `M` columns of out-of-fold predictions (one per
base learner) and fits a small ridge regression to predict the realized
target. The result is the spec's "level-2 LightGBM" stand-in — linear
instead of GBDT, but that's the right shape when M is small (M = 2 here
by default).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.models.gbdt_transformer_ensembles.types import StackingFit


def fit_stacker(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    base_learner_names: Sequence[str],
    alpha: float = 1.0,
    non_negative: bool = False,
) -> StackingFit:
    """Fit `y ~ intercept + sum_m b_m * yhat_m`.

    `non_negative=True` projects coefficients onto the non-negative orthant
    after the ridge step — useful when you want a true "blend" interpretation,
    not a regression with sign-flipped weights.
    """

    n, m = oof_predictions.shape
    if y.shape != (n,):
        raise ValueError(f"y shape {y.shape} does not match oof rows {n}")
    if w.shape != (n,):
        raise ValueError(f"w shape {w.shape} does not match oof rows {n}")
    if len(base_learner_names) != m:
        raise ValueError(
            f"base_learner_names length {len(base_learner_names)} != columns {m}"
        )
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    Z = np.hstack([oof_predictions, np.ones((n, 1), dtype=float)])
    sw = np.sqrt(np.maximum(w, 0.0))
    Z_w = Z * sw[:, None]
    y_w = y * sw

    penalty = np.zeros(m + 1)
    penalty[:m] = alpha
    A = Z_w.T @ Z_w + np.diag(penalty)
    b = Z_w.T @ y_w
    beta_aug, *_ = np.linalg.lstsq(A, b, rcond=None)
    coefs = beta_aug[:m]
    intercept = float(beta_aug[m])

    if non_negative:
        coefs = np.clip(coefs, 0.0, None)
        s = coefs.sum()
        if s > 0:
            coefs = coefs / s  # normalize to a convex blend

    return StackingFit(
        coefficients=coefs,
        intercept=intercept,
        base_learner_names=tuple(base_learner_names),
        alpha=alpha,
    )


def predict_stacker(fit: StackingFit, oof_predictions: np.ndarray) -> np.ndarray:
    """Apply the stacker. `oof_predictions` columns must align with `fit.base_learner_names`."""

    if oof_predictions.shape[1] != len(fit.base_learner_names):
        raise ValueError(
            f"oof_predictions has {oof_predictions.shape[1]} columns; "
            f"expected {len(fit.base_learner_names)}"
        )
    return np.asarray(oof_predictions @ fit.coefficients + fit.intercept, dtype=float)


__all__ = ["fit_stacker", "predict_stacker"]
