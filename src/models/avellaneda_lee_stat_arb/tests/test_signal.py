"""Unit tests for the pure math layer of the Avellaneda-Lee model.

Each test pins down one block of the spec's Algorithm outline against a
worked-out formula or a synthetic ground truth.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.avellaneda_lee_stat_arb.signal import (
    POSITION_FLAT,
    POSITION_LONG,
    POSITION_SHORT,
    adf_test_pvalue,
    cumulative_residual,
    factor_neutral_hedge,
    factor_regression,
    fit_ou_ar1,
    ou_parameters_from_ar1,
    pca_eigenportfolio_returns,
    position_decision,
    s_score,
    s_score_modified,
)


def test_factor_regression_recovers_known_beta() -> None:
    """Spec eq. 6 — OLS of r on F recovers known (alpha, beta)."""
    rng = np.random.default_rng(0)
    T = 500
    F = rng.normal(0.0, 0.01, size=(T, 2))
    true_alpha = 0.0005
    true_beta = np.array([0.9, -0.3])
    eps = rng.normal(0.0, 0.002, size=T)
    r = true_alpha + F @ true_beta + eps
    reg = factor_regression(r, F)
    assert reg.alpha == pytest.approx(true_alpha, abs=2e-4)
    assert np.allclose(reg.beta, true_beta, atol=0.02)
    assert 0.0 <= reg.r_squared <= 1.0
    # With sigma_eps=2bp and sigma_F*beta ~ 9bp, R^2 should be >> 0.5.
    assert reg.r_squared > 0.7


def test_cumulative_residual_is_running_sum() -> None:
    eps = np.array([0.1, -0.2, 0.05, 0.0, 0.3])
    X = cumulative_residual(eps)
    np.testing.assert_allclose(X, np.array([0.1, -0.1, -0.05, -0.05, 0.25]))


def test_fit_ou_ar1_recovers_known_params(
    ou_series: np.ndarray, ou_known_params: dict[str, float]
) -> None:
    """AR(1) on a long simulated OU path recovers (b, a, sigma_zeta)."""
    fit = fit_ou_ar1(ou_series)
    expected_b = math.exp(-ou_known_params["kappa"])
    assert fit.b == pytest.approx(expected_b, abs=0.02)
    # m=0 -> a should be near 0.
    assert abs(fit.a) < 0.02
    # The OU innovation SD from exact discretization.
    expected_sigma_zeta = ou_known_params["sigma"] * math.sqrt(
        (1.0 - math.exp(-2.0 * ou_known_params["kappa"])) / (2.0 * ou_known_params["kappa"])
    )
    assert fit.sigma_zeta == pytest.approx(expected_sigma_zeta, rel=0.05)


def test_ou_parameters_from_ar1_match_spec_formulas() -> None:
    """Worked-out check of spec eqs. 10-13 for a hand-picked AR(1) fit."""
    from src.models.avellaneda_lee_stat_arb.signal import AR1Fit

    fit = AR1Fit(a=0.01, b=0.9, sigma_zeta=0.05, n_obs=60)
    params = ou_parameters_from_ar1(fit, dt=1.0)
    assert params.kappa == pytest.approx(-math.log(0.9))
    assert params.m == pytest.approx(0.01 / 0.1)
    # sigma_eq = sigma_zeta / sqrt(1 - b^2)
    assert params.sigma_eq == pytest.approx(0.05 / math.sqrt(1 - 0.81))
    # half-life = ln(2) / kappa
    assert params.half_life == pytest.approx(math.log(2) / params.kappa)
    # sigma^2 = sigma_zeta^2 * 2 kappa / (1 - b^2)
    expected_sigma = math.sqrt(0.05**2 * 2 * params.kappa / 0.19)
    assert params.sigma == pytest.approx(expected_sigma, rel=1e-6)


def test_ou_parameters_rejects_non_stationary() -> None:
    """b outside (0, 1) is non-stationary and should raise."""
    from src.models.avellaneda_lee_stat_arb.signal import AR1Fit

    for bad_b in (-0.1, 0.0, 1.0, 1.2):
        with pytest.raises(ValueError):
            ou_parameters_from_ar1(AR1Fit(a=0.0, b=bad_b, sigma_zeta=0.01, n_obs=60))


def test_s_score_formula() -> None:
    """Spec eq. 14: s = (X - m) / sigma_eq."""
    assert s_score(2.0, 1.0, 0.5) == pytest.approx(2.0)
    assert s_score(0.0, 0.0, 1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        s_score(1.0, 0.0, 0.0)


def test_s_score_modified_subtracts_drift_term() -> None:
    """Spec eq. 15: s_mod = s - alpha / (kappa * sigma_eq)."""
    base = s_score(1.5, 0.5, 0.5)
    mod = s_score_modified(base, alpha=0.02, kappa=0.1, sigma_eq=0.5)
    assert mod == pytest.approx(base - 0.02 / (0.1 * 0.5))


def test_position_decision_state_machine() -> None:
    """Spec table 1: open/close/hold rules with default thresholds."""
    kw = dict(open_long=-1.25, open_short=1.25, close_long=-0.5, close_short=0.5)
    # Flat -> open short on extreme positive.
    assert position_decision(2.0, POSITION_FLAT, **kw) == POSITION_SHORT
    # Flat -> open long on extreme negative.
    assert position_decision(-2.0, POSITION_FLAT, **kw) == POSITION_LONG
    # Flat -> stays flat in the no-trade zone.
    assert position_decision(0.5, POSITION_FLAT, **kw) == POSITION_FLAT
    # Short -> close when s drops below close_short.
    assert position_decision(0.3, POSITION_SHORT, **kw) == POSITION_FLAT
    # Short -> hold while still above close threshold.
    assert position_decision(1.0, POSITION_SHORT, **kw) == POSITION_SHORT
    # Long -> close when s rises above close_long.
    assert position_decision(-0.3, POSITION_LONG, **kw) == POSITION_FLAT
    # Long -> hold below close threshold.
    assert position_decision(-1.0, POSITION_LONG, **kw) == POSITION_LONG


def test_position_decision_rejects_invalid_prev() -> None:
    with pytest.raises(ValueError):
        position_decision(0.0, 7, open_long=-1, open_short=1, close_long=-0.5, close_short=0.5)


def test_pca_eigenportfolio_returns_one_factor() -> None:
    """A panel with one dominant factor should put ~all eigen-mass on PC1."""
    rng = np.random.default_rng(1)
    T, N = 400, 6
    f = rng.normal(0.0, 0.01, size=T)
    betas = rng.uniform(0.8, 1.2, size=N)
    eps = rng.normal(0.0, 0.0005, size=(T, N))
    R = betas[None, :] * f[:, None] + eps
    panel = pd.DataFrame(R, columns=[f"S{i}" for i in range(N)])

    factor_returns, eigvecs, eigvals = pca_eigenportfolio_returns(panel, K=2)
    assert factor_returns.shape == (T, 2)
    assert eigvecs.shape == (N, N)
    assert eigvals.shape == (N,)
    # Top eigenvalue should dominate.
    assert eigvals[0] > 0.7 * eigvals.sum()
    # PC1 should correlate strongly with f.
    rho = np.corrcoef(factor_returns.iloc[:, 0].to_numpy(), f)[0, 1]
    assert abs(rho) > 0.95


def test_pca_eigenportfolio_returns_rejects_invalid_K() -> None:
    panel = pd.DataFrame(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        pca_eigenportfolio_returns(panel, K=0)
    with pytest.raises(ValueError):
        pca_eigenportfolio_returns(panel, K=5)


def test_factor_neutral_hedge_accumulates_exposure() -> None:
    """Spec step 7: sum_i q_i * beta_i = aggregate factor exposure."""
    betas = {
        "A": np.array([1.0, 0.5]),
        "B": np.array([0.8, -0.2]),
        "C": np.array([1.2, 0.0]),
    }
    positions = {"A": +1, "B": -1, "C": +1}
    exposure = factor_neutral_hedge(positions, betas)
    expected = np.array([1.0 - 0.8 + 1.2, 0.5 + 0.2 + 0.0])
    np.testing.assert_allclose(exposure, expected)


def test_factor_neutral_hedge_skips_flat_positions() -> None:
    betas = {"A": np.array([1.0]), "B": np.array([2.0])}
    positions = {"A": 0, "B": +1}
    exposure = factor_neutral_hedge(positions, betas)
    np.testing.assert_allclose(exposure, np.array([2.0]))


def test_adf_pvalue_returns_finite_for_random_walk() -> None:
    rng = np.random.default_rng(2)
    rw = np.cumsum(rng.normal(0.0, 1.0, size=200))
    p = adf_test_pvalue(rw)
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_adf_pvalue_handles_short_series() -> None:
    assert math.isnan(adf_test_pvalue(np.array([1.0, 2.0])))
