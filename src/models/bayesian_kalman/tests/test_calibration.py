"""Tests for `calibration.calibrate` — the public dispatch entry point."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.bayesian_kalman.calibration import calibrate
from src.models.bayesian_kalman.types import (
    BayesianKalmanInputs,
    DynamicFactorFit,
    HierarchicalFit,
    KalmanFit,
)


def _now() -> datetime:
    return datetime.now(UTC)


def test_calibrate_kalman_beta_fixed_hyperparameters(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset, market, true_beta = constant_beta_series
    inputs = BayesianKalmanInputs(
        mode="kalman_beta",
        timestamp=_now(),
        asset_returns=asset,
        market_returns=market,
    )
    result = calibrate(
        inputs=inputs,
        fix_hyperparameters=(1e-12, 1e-8, 0.005 ** 2),
    )
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "bayesian_kalman"
    fit = result.parameters["kalman_fit"]
    assert isinstance(fit, KalmanFit)
    assert fit.n_states == 2
    # Latest beta near the truth.
    assert abs(fit.latest_state()[1] - true_beta) < 0.20


def test_calibrate_kalman_beta_mle(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset, market, _ = constant_beta_series
    inputs = BayesianKalmanInputs(
        mode="kalman_beta",
        timestamp=_now(),
        asset_returns=asset,
        market_returns=market,
    )
    result = calibrate(inputs=inputs, mle_max_iter=100)
    metrics = result.fit_metrics
    assert metrics["q_alpha"] > 0
    assert metrics["q_beta"] > 0
    assert metrics["R"] > 0
    assert metrics["n_obs"] >= 400


def test_calibrate_hierarchical(
    hierarchical_panel: tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]
) -> None:
    designs, targets, _true_thetas = hierarchical_panel
    tickers = tuple(sorted(designs.keys()))
    inputs = BayesianKalmanInputs(
        mode="hierarchical",
        timestamp=_now(),
        panel_design=designs,
        panel_targets=targets,
        coef_names=("intercept", "market"),
        tickers=tickers,
    )
    result = calibrate(inputs=inputs, n_iter=200, n_burn=100)
    fit = result.parameters["hierarchical_fit"]
    assert isinstance(fit, HierarchicalFit)
    assert fit.n_assets == len(tickers)
    assert fit.n_coefs == 2
    assert result.fit_metrics["mean_sigma_sq"] > 0


def test_calibrate_dynamic_factor(
    dfm_panel: tuple[pd.DataFrame, np.ndarray, np.ndarray]
) -> None:
    panel, _Lambda, _factors = dfm_panel
    inputs = BayesianKalmanInputs(
        mode="dynamic_factor",
        timestamp=_now(),
        panel_returns=panel,
        tickers=tuple(str(c) for c in panel.columns),
    )
    result = calibrate(inputs=inputs, n_factors=2, n_em_iter=20)
    fit = result.parameters["dynamic_factor_fit"]
    assert isinstance(fit, DynamicFactorFit)
    assert fit.n_factors == 2
    assert result.fit_metrics["variance_explained"] > 0.5


def test_calibrate_kalman_beta_short_history_rejected() -> None:
    """Fewer than 30 aligned observations should fail loudly rather than fit."""
    short = pd.Series(np.arange(10).astype(float) / 100)
    inputs = BayesianKalmanInputs(
        mode="kalman_beta",
        timestamp=_now(),
        asset_returns=short,
        market_returns=short,
    )
    try:
        calibrate(inputs=inputs, fix_hyperparameters=(1e-10, 1e-8, 1e-4))
    except ValueError as exc:
        assert "30" in str(exc)
    else:
        raise AssertionError("Expected ValueError for short kalman_beta history")
