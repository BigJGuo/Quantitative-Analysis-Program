"""Unit tests for the ridge stacker."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.gbdt_transformer_ensembles.stacking import (
    fit_stacker,
    predict_stacker,
)


def test_stacker_recovers_identity_when_perfect_predictions() -> None:
    """If yhat_1 == y, the stacker should give weight ~1 to learner 1."""

    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    yhat1 = y.copy()  # perfect
    yhat2 = rng.normal(size=n)  # noise
    Z = np.column_stack([yhat1, yhat2])
    w = np.ones(n)
    fit = fit_stacker(
        Z, y, w, base_learner_names=("perf", "noise"), alpha=0.01,
    )
    assert fit.coefficients[0] > 0.7
    assert abs(fit.coefficients[1]) < 0.3
    pred = predict_stacker(fit, Z)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.95


def test_stacker_non_negative_blend() -> None:
    """non_negative=True must produce a convex blend (non-negative + sums to 1)."""

    rng = np.random.default_rng(1)
    n = 100
    y = rng.normal(size=n)
    Z = np.column_stack([y + rng.normal(0, 0.5, size=n), y + rng.normal(0, 1.0, size=n)])
    w = np.ones(n)
    fit = fit_stacker(
        Z, y, w, base_learner_names=("a", "b"), alpha=0.1, non_negative=True
    )
    assert (fit.coefficients >= 0).all()
    assert fit.coefficients.sum() == pytest.approx(1.0)


def test_stacker_input_validation() -> None:
    Z = np.zeros((10, 2))
    y = np.zeros(10)
    w = np.ones(10)
    with pytest.raises(ValueError):
        fit_stacker(Z, y[:5], w, base_learner_names=("a", "b"))
    with pytest.raises(ValueError):
        fit_stacker(Z, y, w, base_learner_names=("a",))
    with pytest.raises(ValueError):
        fit_stacker(Z, y, w, base_learner_names=("a", "b"), alpha=-1.0)


def test_predict_stacker_shape_check() -> None:
    Z = np.zeros((5, 3))
    y = np.zeros(5)
    w = np.ones(5)
    fit = fit_stacker(Z, y, w, base_learner_names=("a", "b", "c"))
    with pytest.raises(ValueError):
        predict_stacker(fit, np.zeros((5, 2)))
