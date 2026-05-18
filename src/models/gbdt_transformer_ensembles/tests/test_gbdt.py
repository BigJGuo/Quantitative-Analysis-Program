"""Unit tests for the pure-numpy GBDT.

These tests verify the gradient/loss math against the spec's worked-out
formulas with hand-rolled synthetic data, plus end-to-end behavior on a
known regression problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.gbdt_transformer_ensembles.gbdt import (
    GBDTParams,
    fit_gbdt,
    initial_prediction,
    loss_value,
    negative_gradient,
    predict_gbdt,
    predict_tree,
)
from src.models.gbdt_transformer_ensembles.types import RegressionTree, TreeNode


def test_mse_negative_gradient_is_residual() -> None:
    """Per the spec eq: g_i = y - F for squared loss."""

    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([0.5, 2.5, 2.0])
    g = negative_gradient(y, pred, loss="mse", delta=1.0)
    np.testing.assert_allclose(g, y - pred)


def test_huber_negative_gradient_quadratic_region() -> None:
    """|y - F| <= delta -> Huber gradient == MSE gradient."""

    y = np.array([1.0, 2.0])
    pred = np.array([1.4, 2.5])  # residuals 0.6, -0.5; both within delta=1
    g = negative_gradient(y, pred, loss="huber", delta=1.0)
    np.testing.assert_allclose(g, y - pred)


def test_huber_negative_gradient_linear_region() -> None:
    """|y - F| > delta -> Huber gradient saturates at delta * sign(y - F)."""

    y = np.array([5.0, -5.0])
    pred = np.array([1.0, 1.0])  # |r| = 4 > delta=1
    g = negative_gradient(y, pred, loss="huber", delta=1.0)
    np.testing.assert_allclose(g, np.array([1.0, -1.0]))


def test_loss_value_mse() -> None:
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    w = np.ones(3)
    assert loss_value(y, pred, w, loss="mse", delta=1.0) == pytest.approx(0.0)
    pred2 = np.array([0.0, 0.0, 0.0])
    expected = 0.5 * np.mean(y**2)
    assert loss_value(y, pred2, w, loss="mse", delta=1.0) == pytest.approx(expected)


def test_loss_value_huber_matches_formula() -> None:
    """Spec eq: 0.5 r^2 if |r| <= delta else delta*(|r| - 0.5*delta)."""

    y = np.array([0.0, 0.0])
    pred = np.array([0.5, 2.0])  # residuals -0.5 (quad), -2.0 (lin)
    w = np.ones(2)
    delta = 1.0
    expected = ((0.5 * 0.25) + (delta * (2.0 - 0.5 * delta))) / 2.0
    assert loss_value(y, pred, w, loss="huber", delta=delta) == pytest.approx(expected)


def test_initial_prediction_mse_is_weighted_mean() -> None:
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 2.0, 3.0])
    expected = (1.0 + 4.0 + 9.0) / 6.0
    assert initial_prediction(y, w, loss="mse") == pytest.approx(expected)


def test_initial_prediction_huber_is_weighted_median() -> None:
    """Weighted median: cumulative weight crosses half at the median value."""

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert initial_prediction(y, w, loss="huber") == pytest.approx(3.0)


def test_predict_tree_leaf_only() -> None:
    """A tree with a single leaf must return that leaf's value for every row."""

    tree = RegressionTree(
        nodes=(
            TreeNode(feature_index=-1, threshold=float("nan"), value=0.42, left=-1, right=-1),
        )
    )
    X = np.zeros((4, 3))
    np.testing.assert_allclose(predict_tree(tree, X), np.full(4, 0.42))


def test_predict_tree_threshold() -> None:
    """Root splits on feature 0 at 0.5; left value -1, right value +1."""

    tree = RegressionTree(
        nodes=(
            TreeNode(feature_index=0, threshold=0.5, value=0.0, left=1, right=2),
            TreeNode(feature_index=-1, threshold=float("nan"), value=-1.0, left=-1, right=-1),
            TreeNode(feature_index=-1, threshold=float("nan"), value=+1.0, left=-1, right=-1),
        )
    )
    X = np.array([[0.0], [0.4], [0.6], [1.0]])
    out = predict_tree(tree, X)
    np.testing.assert_allclose(out, np.array([-1.0, -1.0, 1.0, 1.0]))


def test_fit_gbdt_recovers_step_function() -> None:
    """y = sign(x_0) -- a depth-1 tree should give near-perfect fit in 1-2 trees."""

    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    y = np.where(X[:, 0] > 0, 1.0, -1.0)
    w = np.ones(n)
    params = GBDTParams(
        n_estimators=20,
        learning_rate=0.3,
        max_depth=2,
        min_samples_leaf=10,
        subsample=1.0,
        colsample=1.0,
        l2_leaf_reg=0.0,
        loss="mse",
        early_stopping_rounds=None,
        seed=1,
    )
    fit = fit_gbdt(
        X, y, w, feature_names=("x0", "x1", "x2"), params=params
    )
    pred = predict_gbdt(fit, X)
    # The sign of the prediction should match y on > 95% of rows.
    accuracy = float(np.mean(np.sign(pred) == y))
    assert accuracy > 0.95


def test_fit_gbdt_train_loss_monotone_nonincreasing() -> None:
    """With learning_rate small enough, the training loss must not increase."""

    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 4))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, size=n)
    w = np.ones(n)
    params = GBDTParams(
        n_estimators=15,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        subsample=1.0,
        colsample=1.0,
        l2_leaf_reg=0.1,
        loss="mse",
        early_stopping_rounds=None,
        seed=1,
    )
    fit = fit_gbdt(X, y, w, feature_names=tuple(f"x{i}" for i in range(4)), params=params)
    hist = np.array(fit.train_loss_history)
    # Allow tiny numerical jitter.
    diffs = np.diff(hist)
    assert (diffs <= 1e-9).all()


def test_fit_gbdt_early_stopping_triggers() -> None:
    """When validation loss plateaus quickly, early stopping should fire."""

    rng = np.random.default_rng(2)
    n = 200
    X_tr = rng.normal(size=(n, 3))
    y_tr = X_tr[:, 0] + rng.normal(0, 0.1, size=n)
    X_val = rng.normal(size=(50, 3))
    y_val = X_val[:, 0] + rng.normal(0, 0.1, size=50)
    w_tr = np.ones(n)
    w_val = np.ones(50)
    params = GBDTParams(
        n_estimators=200,
        learning_rate=0.5,  # large -> early plateau
        max_depth=2,
        min_samples_leaf=10,
        subsample=1.0,
        colsample=1.0,
        l2_leaf_reg=0.0,
        loss="mse",
        early_stopping_rounds=5,
        seed=1,
    )
    fit = fit_gbdt(
        X_tr, y_tr, w_tr, feature_names=("x0", "x1", "x2"),
        params=params,
        X_val=X_val, y_val=y_val, w_val=w_val,
    )
    assert fit.val_loss_history is not None
    assert fit.n_trees < params.n_estimators
    assert fit.best_iteration <= fit.n_trees


def test_predict_gbdt_n_trees_cap() -> None:
    """Passing n_trees should produce the same value as initial when n=0."""

    rng = np.random.default_rng(3)
    n = 80
    X = rng.normal(size=(n, 2))
    y = X[:, 0]
    w = np.ones(n)
    fit = fit_gbdt(
        X, y, w, feature_names=("a", "b"),
        params=GBDTParams(
            n_estimators=10, learning_rate=0.1, max_depth=2, min_samples_leaf=5,
            subsample=1.0, colsample=1.0, l2_leaf_reg=0.0, loss="mse",
            early_stopping_rounds=None, seed=1,
        ),
    )
    pred_zero = predict_gbdt(fit, X, n_trees=0)
    np.testing.assert_allclose(pred_zero, np.full(n, fit.init_value))


def test_feature_importance_accumulates() -> None:
    """Splits on a highly-predictive feature accumulate gain there."""

    rng = np.random.default_rng(4)
    n = 200
    X = rng.normal(size=(n, 3))
    y = X[:, 1]  # only feature 1 matters
    w = np.ones(n)
    fit = fit_gbdt(
        X, y, w, feature_names=("a", "b", "c"),
        params=GBDTParams(
            n_estimators=10, learning_rate=0.1, max_depth=2, min_samples_leaf=5,
            subsample=1.0, colsample=1.0, l2_leaf_reg=0.0, loss="mse",
            early_stopping_rounds=None, seed=1,
        ),
    )
    gains = fit.feature_importance_gain
    assert gains[1] > gains[0]
    assert gains[1] > gains[2]


def test_gbdt_params_validation() -> None:
    with pytest.raises(ValueError):
        GBDTParams(n_estimators=0)
    with pytest.raises(ValueError):
        GBDTParams(learning_rate=0.0)
    with pytest.raises(ValueError):
        GBDTParams(subsample=0.0)
    with pytest.raises(ValueError):
        GBDTParams(loss="logloss")  # type: ignore[arg-type]
