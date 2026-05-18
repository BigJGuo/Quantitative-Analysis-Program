"""Unit tests for the linear sequence base learner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gbdt_transformer_ensembles.sequence_model import (
    LinearSeqParams,
    expand_features,
    fit_linear_seq,
    predict_linear_seq,
)


def _toy_data(n_per_ticker: int = 40) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rng = np.random.default_rng(0)
    n_tickers = 3
    X_blocks = []
    y_blocks = []
    tickers = []
    for t in range(n_tickers):
        x1 = rng.normal(size=n_per_ticker)
        x2 = rng.normal(size=n_per_ticker)
        # Target depends on the current value of x1 and a 5-step rolling mean of x2.
        rolling_mean_x2 = pd.Series(x2).rolling(window=5, min_periods=1).mean().to_numpy()
        y = 0.5 * x1 + 1.0 * rolling_mean_x2 + rng.normal(0, 0.1, size=n_per_ticker)
        X_blocks.append(np.column_stack([x1, x2]))
        y_blocks.append(y)
        tickers.extend([f"T{t}"] * n_per_ticker)
    X = pd.DataFrame(np.vstack(X_blocks), columns=["x1", "x2"])
    y = np.concatenate(y_blocks)
    return X, y, pd.Series(tickers)


def test_expand_features_column_count() -> None:
    """Each feature -> last + 2*windows + ewma = 1 + 2*|windows| + 1 columns."""

    X, _, tickers = _toy_data()
    Z, names = expand_features(
        X,
        feature_names=("x1", "x2"),
        tickers=tickers,
        windows=(3, 5),
        ewma_alpha=0.1,
    )
    # 2 features * (1 + 2*2 + 1) = 2 * 6 = 12 columns.
    assert Z.shape == (len(X), 12)
    assert len(names) == 12


def test_expand_features_per_ticker_rolling() -> None:
    """The rolling mean of `x` for a ticker must equal pandas' rolling mean."""

    X, _, tickers = _toy_data()
    Z, names = expand_features(
        X,
        feature_names=("x1", "x2"),
        tickers=tickers,
        windows=(5,),
        ewma_alpha=0.1,
    )
    mean_col = names.index("mean5__x1")
    expected = (
        X["x1"]
        .groupby(tickers, sort=False)
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        .to_numpy()
    )
    np.testing.assert_allclose(Z[:, mean_col], expected)


def test_fit_linear_seq_recovers_signal() -> None:
    """The toy target depends on rolling features; the model should fit well."""

    X, y, tickers = _toy_data(n_per_ticker=120)
    w = np.ones_like(y)
    params = LinearSeqParams(windows=(5,), ewma_alpha=0.2, alpha=0.1)
    fit = fit_linear_seq(
        X, y, w, feature_names=("x1", "x2"), tickers=tickers, params=params
    )
    pred = predict_linear_seq(
        fit, X, tickers, windows=params.windows, ewma_alpha=params.ewma_alpha
    )
    # R^2 should be high.
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.7


def test_predict_linear_seq_shape_consistency() -> None:
    X, y, tickers = _toy_data(n_per_ticker=30)
    w = np.ones_like(y)
    params = LinearSeqParams(windows=(3,), ewma_alpha=0.2, alpha=1.0)
    fit = fit_linear_seq(X, y, w, feature_names=("x1", "x2"), tickers=tickers, params=params)
    pred = predict_linear_seq(
        fit, X, tickers, windows=params.windows, ewma_alpha=params.ewma_alpha
    )
    assert pred.shape == (len(X),)


def test_linear_seq_params_validation() -> None:
    with pytest.raises(ValueError):
        LinearSeqParams(windows=())
    with pytest.raises(ValueError):
        LinearSeqParams(windows=(1,))
    with pytest.raises(ValueError):
        LinearSeqParams(ewma_alpha=0.0)
    with pytest.raises(ValueError):
        LinearSeqParams(alpha=-1.0)
