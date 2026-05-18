"""Unit tests for `src.models.futures_implied_nav.calibration`.

The strategy: build synthetic basket returns from a known true beta plus a
small i.i.d. residual, fit, and check the recovered coefficients match.
Edge cases (constraints binding, degenerate panel, CV behavior) get their own
focused tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.models.futures_implied_nav.calibration import (
    _constrained_ridge,
    _cv_lambda,
    _overnight_returns,
    _project_box_and_sum,
    _r_squared,
    calibrate,
)
from src.models.futures_implied_nav.types import HedgePanel


def _build_synthetic_panel(
    *,
    n: int = 60,
    true_betas: tuple[float, ...] = (0.6, 0.4),
    noise_sd: float = 1e-4,
    seed: int = 0,
) -> tuple[HedgePanel, np.ndarray]:
    rng = np.random.default_rng(seed)
    j = len(true_betas)
    # Hedge returns ~ N(0, 1%); basket return = sum(beta_j * r_H_j) + noise.
    r_h = rng.normal(scale=0.01, size=(n, j))
    eps = rng.normal(scale=noise_sd, size=n)
    r_b = r_h @ np.asarray(true_betas) + eps

    # Convert returns to "now/close" price pairs. Pick nav_close=100, hedge_close=50.
    nav_close = np.full(n, 100.0)
    nav_open = nav_close * (1.0 + r_b)
    hedge_tickers = tuple(f"H{i}" for i in range(j))
    hedge_close: dict[str, tuple[float, ...]] = {
        t: tuple(float(x) for x in np.full(n, 50.0)) for t in hedge_tickers
    }
    hedge_now: dict[str, tuple[float, ...]] = {
        t: tuple(float(50.0 * (1.0 + r)) for r in r_h[:, i])
        for i, t in enumerate(hedge_tickers)
    }
    dates = tuple(datetime(2024, 1, 1) + timedelta(days=k) for k in range(n))
    panel = HedgePanel(
        dates=dates,
        nav_close=tuple(float(x) for x in nav_close),
        nav_open=tuple(float(x) for x in nav_open),
        hedge_close=hedge_close,
        hedge_now=hedge_now,
    )
    return panel, np.asarray(true_betas)


def test_overnight_returns_matches_panel_arithmetic() -> None:
    panel, _ = _build_synthetic_panel(n=10)
    r_b, r_h, tickers = _overnight_returns(panel)
    assert tickers == ("H0", "H1")
    assert r_b.shape == (10,)
    assert r_h.shape == (10, 2)
    # Spot-check: r_b[0] = nav_open[0]/nav_close[0] - 1
    expected = panel.nav_open[0] / panel.nav_close[0] - 1.0
    assert r_b[0] == pytest.approx(expected)


def test_project_box_only_when_within_sum() -> None:
    b = np.array([0.5, -0.1, 2.0])
    out = _project_box_and_sum(b, beta_max=1.5, sum_max=1.2)
    # Clip to [0, 1.5]; raw sum after clip = 0.5 + 0 + 1.5 = 2.0 > 1.2 -> reduce.
    assert out.sum() <= 1.2 + 1e-6
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.5 + 1e-6)


def test_project_box_no_change_when_feasible() -> None:
    b = np.array([0.3, 0.2, 0.4])
    out = _project_box_and_sum(b, beta_max=1.5, sum_max=1.2)
    np.testing.assert_allclose(out, b)


def test_constrained_ridge_recovers_true_beta() -> None:
    panel, true_b = _build_synthetic_panel(n=200, true_betas=(0.7, 0.3), noise_sd=1e-5)
    r_b, r_h, _ = _overnight_returns(panel)
    beta_hat = _constrained_ridge(r_h, r_b, lam=1e-4)
    np.testing.assert_allclose(beta_hat, true_b, atol=1e-2)


def test_constrained_ridge_enforces_box() -> None:
    # Build a panel where unconstrained OLS would want beta > 1.5 for hedge 0.
    rng = np.random.default_rng(42)
    n = 200
    r_h = np.column_stack(
        [rng.normal(scale=0.01, size=n), rng.normal(scale=0.01, size=n)]
    )
    # True relationship has beta_0 = 3.0 (above box upper bound 1.5).
    r_b = 3.0 * r_h[:, 0] + 0.05 * r_h[:, 1] + rng.normal(scale=1e-5, size=n)
    beta_hat = _constrained_ridge(r_h, r_b, lam=1e-4, beta_max=1.5, sum_max=1.2)
    assert np.all(beta_hat >= -1e-9)
    assert np.all(beta_hat <= 1.5 + 1e-6)
    assert beta_hat.sum() <= 1.2 + 1e-6
    # First coefficient should saturate at or near the upper sum/box bound.
    assert beta_hat[0] >= 1.0


def test_r_squared_perfect_fit() -> None:
    rng = np.random.default_rng(0)
    r_h = rng.normal(scale=0.01, size=(50, 2))
    beta = np.array([0.6, 0.4])
    r_b = r_h @ beta
    assert _r_squared(r_b, r_h, beta) == pytest.approx(1.0, abs=1e-10)


def test_cv_lambda_picks_one_of_the_grid() -> None:
    panel, _ = _build_synthetic_panel(n=80)
    r_b, r_h, _ = _overnight_returns(panel)
    lambdas = (1e-3, 1e-1, 10.0)
    chosen = _cv_lambda(r_h, r_b, lambdas, n_folds=4)
    assert chosen in lambdas


def test_calibrate_endtoend_returns_calibration_result() -> None:
    panel, true_b = _build_synthetic_panel(n=120, true_betas=(0.5, 0.4), noise_sd=1e-5)
    result = calibrate(panel)
    assert result.model_name == "futures_implied_nav"
    beta = result.parameters["beta"]
    variances = result.parameters["variances"]
    # Recovered beta close to truth.
    np.testing.assert_allclose(
        np.asarray(beta.betas), true_b, atol=1e-2
    )
    # R^2 should be very high on synthetic data with tiny noise.
    assert beta.r_squared > 0.99
    # Variances positive.
    assert variances.sigma2_hedge > 0
    assert variances.sigma2_nav > 0
    assert variances.sigma2_etf > 0
    # Metrics surface the basics.
    assert result.fit_metrics["n_obs"] == 120.0
    assert result.fit_metrics["n_hedges"] == 2.0


def test_calibrate_rejects_short_panel() -> None:
    panel, _ = _build_synthetic_panel(n=4)
    with pytest.raises(ValueError, match="at least"):
        calibrate(panel)


def test_calibrate_rejects_non_panel() -> None:
    with pytest.raises(TypeError):
        calibrate({"not": "a panel"})  # type: ignore[arg-type]
