"""Unit tests for the pure-math `signal.py` layer.

The Kalman filter / smoother / log-likelihood are exercised against synthetic
data with known generative parameters so the worked-out formulas in the spec
become exact-ish recovery checks.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.bayesian_kalman.signal import (
    dynamic_factor_em,
    hierarchical_gibbs,
    hierarchical_posterior_mean,
    innovation_normality_kurtosis,
    kalman_log_likelihood,
    kalman_smoother,
    ljung_box_pvalue,
    mle_time_varying_beta,
    standardized_innovations,
    time_varying_beta_filter,
)

# ---------------------------------------------------------------------------
# Kalman filter math
# ---------------------------------------------------------------------------


def test_kalman_filter_constant_beta_tracks_truth(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset, market, true_beta = constant_beta_series
    run = time_varying_beta_filter(
        asset.to_numpy(),
        market.to_numpy(),
        q_alpha=1e-12,
        q_beta=1e-8,
        R=0.005 ** 2,
    )
    # After ~250 observations the filter should settle near the truth.
    final_beta = run.states[-1, 1]
    assert abs(final_beta - true_beta) < 0.10


def test_kalman_filter_drifting_beta_tracks_truth(
    drifting_beta_series: tuple[pd.Series, pd.Series, np.ndarray]
) -> None:
    asset, market, beta_path = drifting_beta_series
    run = time_varying_beta_filter(
        asset.to_numpy(),
        market.to_numpy(),
        q_alpha=1e-12,
        q_beta=1e-5,
        R=0.003 ** 2,
    )
    # End-of-sample beta should be near the final true beta (1.6 ± 0.2).
    assert abs(run.states[-1, 1] - beta_path[-1]) < 0.20
    # Filter should pick up *direction* of drift: end > start.
    assert run.states[-1, 1] > run.states[20, 1]


def test_kalman_filter_log_likelihood_matches_step_sum(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset, market, _ = constant_beta_series
    asset_arr = asset.to_numpy()
    market_arr = market.to_numpy()
    ll = kalman_log_likelihood(
        asset_arr,
        np.column_stack([np.ones(len(asset_arr)), market_arr]),
        np.eye(2),
        np.diag([1e-12, 1e-8]),
        0.005 ** 2,
        np.array([0.0, 1.0]),
        np.diag([0.01, 0.1]),
    )
    # Recompute via filter directly and confirm.
    run = time_varying_beta_filter(
        asset_arr, market_arr, q_alpha=1e-12, q_beta=1e-8, R=0.005 ** 2
    )
    assert math.isfinite(ll)
    assert ll == pytest.approx(run.log_likelihood, rel=1e-10, abs=1e-10)


def test_kalman_filter_innovation_step_one(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    """Worked formula check: step-1 innovation = y_0 - H_0 x_0_pred,
    with x_0_pred = F x0 = x0 (because F = I)."""
    asset, market, _ = constant_beta_series
    y_arr = asset.to_numpy()
    m_arr = market.to_numpy()
    x0 = np.array([0.0, 1.0])
    expected_nu = y_arr[0] - (x0[0] + x0[1] * m_arr[0])
    run = time_varying_beta_filter(
        y_arr, m_arr, q_alpha=1e-12, q_beta=1e-8, R=0.005 ** 2
    )
    assert run.innovations[0] == pytest.approx(expected_nu, rel=1e-12, abs=1e-12)


def test_kalman_filter_covariance_symmetric_psd(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset, market, _ = constant_beta_series
    run = time_varying_beta_filter(
        asset.to_numpy(),
        market.to_numpy(),
        q_alpha=1e-12,
        q_beta=1e-8,
        R=0.005 ** 2,
    )
    for t in range(0, run.covariances.shape[0], 50):
        P = run.covariances[t]
        assert np.allclose(P, P.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals >= -1e-10)


def test_kalman_smoother_matches_filter_at_end(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    """At the last timestamp the smoother coincides with the filter."""
    asset, market, _ = constant_beta_series
    run = time_varying_beta_filter(
        asset.to_numpy(),
        market.to_numpy(),
        q_alpha=1e-12,
        q_beta=1e-8,
        R=0.005 ** 2,
    )
    sm_states, sm_covs = kalman_smoother(run, np.eye(2), np.diag([1e-12, 1e-8]))
    assert np.allclose(sm_states[-1], run.states[-1])
    assert np.allclose(sm_covs[-1], run.covariances[-1])


def test_kalman_smoother_reduces_uncertainty(
    drifting_beta_series: tuple[pd.Series, pd.Series, np.ndarray]
) -> None:
    """The smoother should at least weakly reduce the trace of `P` mid-sample."""
    asset, market, _ = drifting_beta_series
    run = time_varying_beta_filter(
        asset.to_numpy(),
        market.to_numpy(),
        q_alpha=1e-12,
        q_beta=1e-5,
        R=0.003 ** 2,
    )
    sm_states, sm_covs = kalman_smoother(run, np.eye(2), np.diag([1e-12, 1e-5]))
    # Strict containment is hard to guarantee with a single sample, but the
    # *mean* trace over the middle of the sample should be lower for smoother.
    mid = slice(50, -50)
    mean_filter_trace = float(np.mean([np.trace(P) for P in run.covariances[mid]]))
    mean_smoother_trace = float(np.mean([np.trace(P) for P in sm_covs[mid]]))
    assert mean_smoother_trace <= mean_filter_trace + 1e-12


def test_standardized_innovations_are_unit_variance(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset, market, _ = constant_beta_series
    run = time_varying_beta_filter(
        asset.to_numpy(),
        market.to_numpy(),
        q_alpha=1e-12,
        q_beta=1e-8,
        R=0.005 ** 2,
    )
    std = standardized_innovations(run)
    # Skip the burn-in transient.
    std = std[100:]
    assert abs(std.mean()) < 0.2
    assert abs(std.std(ddof=1) - 1.0) < 0.25


def test_ljung_box_pvalue_white_noise_is_large(rng: np.random.Generator) -> None:
    # iid Gaussian is white, p > 0.05 most of the time.
    x = rng.standard_normal(size=500)
    p = ljung_box_pvalue(x, max_lag=10)
    assert p > 0.05


def test_ljung_box_pvalue_ar1_is_tiny(rng: np.random.Generator) -> None:
    # Strong AR(1) is not white; p should be tiny.
    T = 500
    x = np.zeros(T)
    eps = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = 0.7 * x[t - 1] + eps[t]
    p = ljung_box_pvalue(x, max_lag=10)
    assert p < 0.01


def test_innovation_kurtosis_gaussian_is_near_zero(rng: np.random.Generator) -> None:
    x = rng.standard_normal(size=5000)
    k = innovation_normality_kurtosis(x)
    assert abs(k) < 0.4


# ---------------------------------------------------------------------------
# MLE for Kalman hyperparameters
# ---------------------------------------------------------------------------


def test_mle_converges_to_positive_finite(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset, market, _ = constant_beta_series
    q_a, q_b, R, ll = mle_time_varying_beta(asset.to_numpy(), market.to_numpy())
    assert q_a > 0 and q_b > 0 and R > 0
    assert math.isfinite(ll)


def test_mle_R_close_to_residual_var(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    """For a constant-beta DGP with observation noise of 0.005, the MLE R
    should land in the right order of magnitude."""
    asset, market, true_beta = constant_beta_series
    _q_a, _q_b, R, _ll = mle_time_varying_beta(asset.to_numpy(), market.to_numpy())
    # True R is 0.005^2 = 2.5e-5. MLE shouldn't be more than ~5x off.
    assert 1e-6 < R < 5e-4


def test_mle_likelihood_at_least_as_good_as_init(
    drifting_beta_series: tuple[pd.Series, pd.Series, np.ndarray]
) -> None:
    asset, market, _ = drifting_beta_series
    # Compare MLE to a deliberately-bad fixed hyperparameter set.
    bad_run = time_varying_beta_filter(
        asset.to_numpy(),
        market.to_numpy(),
        q_alpha=1.0,
        q_beta=1.0,
        R=1.0,
    )
    _q_a, _q_b, _R, ll = mle_time_varying_beta(asset.to_numpy(), market.to_numpy())
    assert ll > bad_run.log_likelihood


# ---------------------------------------------------------------------------
# Hierarchical Bayes: closed-form posterior + Gibbs
# ---------------------------------------------------------------------------


def test_hierarchical_posterior_mean_matches_ols_with_diffuse_prior() -> None:
    """With a diffuse Sigma the posterior mean should collapse to OLS."""
    rng = np.random.default_rng(0)
    T = 200
    X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
    true_theta = np.array([0.5, -0.3])
    y = X @ true_theta + rng.normal(0, 0.1, T)
    mu = np.zeros(2)
    big_Sigma = np.eye(2) * 1e6
    m, _V = hierarchical_posterior_mean(X, y, mu, big_Sigma, sigma_sq=0.01)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(m, ols, rtol=1e-3, atol=1e-4)


def test_hierarchical_posterior_mean_shrinks_with_tight_prior() -> None:
    """With a very tight prior + high observation noise, posterior collapses
    to the prior mean."""
    rng = np.random.default_rng(1)
    T = 50
    X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
    y = X @ np.array([5.0, -5.0]) + rng.normal(0, 1, T)
    tight_Sigma = np.eye(2) * 1e-4
    mu = np.array([0.0, 0.0])
    m, _V = hierarchical_posterior_mean(X, y, mu, tight_Sigma, sigma_sq=10.0)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    # Posterior must shrink heavily toward mu — far closer to mu than to OLS.
    dist_to_mu = float(np.linalg.norm(m - mu))
    dist_to_ols = float(np.linalg.norm(m - ols))
    assert dist_to_mu < 0.2 * dist_to_ols


def test_gibbs_recovers_population_mean_and_thetas(
    hierarchical_panel: tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]
) -> None:
    designs_dict, targets_dict, true_thetas = hierarchical_panel
    tickers = sorted(designs_dict.keys())
    designs = [designs_dict[t] for t in tickers]
    targets = [targets_dict[t] for t in tickers]
    out = hierarchical_gibbs(designs, targets, n_iter=400, n_burn=200, seed=42)

    # Posterior mu should be near the empirical mean of the true thetas.
    empirical_pop_mean = true_thetas.mean(axis=0)
    np.testing.assert_allclose(out["mu_mean"], empirical_pop_mean, atol=0.15)

    # Per-asset posterior means should be close to the true thetas.
    err = np.linalg.norm(out["theta_means"] - true_thetas, axis=1)
    assert err.max() < 0.30


def test_gibbs_posterior_covariance_is_psd(
    hierarchical_panel: tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]
) -> None:
    designs_dict, targets_dict, _ = hierarchical_panel
    tickers = sorted(designs_dict.keys())
    designs = [designs_dict[t] for t in tickers]
    targets = [targets_dict[t] for t in tickers]
    out = hierarchical_gibbs(designs, targets, n_iter=200, n_burn=100, seed=7)
    for V in out["theta_covs"]:
        eigvals = np.linalg.eigvalsh(0.5 * (V + V.T))
        assert eigvals.min() > -1e-6


# ---------------------------------------------------------------------------
# Dynamic factor model (EM)
# ---------------------------------------------------------------------------


def test_dfm_recovers_two_factor_structure(
    dfm_panel: tuple[pd.DataFrame, np.ndarray, np.ndarray]
) -> None:
    panel, true_Lambda, _true_factors = dfm_panel
    out = dynamic_factor_em(panel, r=2, n_iter=40)
    Lambda = out["Lambda"]
    # Loadings recovered up to a sign / orthogonal rotation. Compare via
    # subspace alignment: the column space of Lambda should match the
    # column space of true_Lambda.
    Q_est, _ = np.linalg.qr(Lambda)
    Q_true, _ = np.linalg.qr(true_Lambda)
    # principal angles via singular values of Q_est^T Q_true
    s = np.linalg.svd(Q_est.T @ Q_true, compute_uv=False)
    # Both columns should align reasonably well (s_i close to 1).
    assert s.min() > 0.7


def test_dfm_log_likelihood_finite(
    dfm_panel: tuple[pd.DataFrame, np.ndarray, np.ndarray]
) -> None:
    panel, _Lambda, _factors = dfm_panel
    out = dynamic_factor_em(panel, r=2, n_iter=20)
    assert math.isfinite(out["log_likelihood"])
    assert out["iters"] >= 1


def test_dfm_variance_explained_high(
    dfm_panel: tuple[pd.DataFrame, np.ndarray, np.ndarray]
) -> None:
    """Two latent factors explain most of cross-sectional variance in the DGP."""
    panel, _Lambda, _factors = dfm_panel
    out = dynamic_factor_em(panel, r=2, n_iter=30)
    Y = panel.to_numpy(dtype=float)
    Y_c = Y - Y.mean(axis=0)
    fitted = out["factors"] @ out["Lambda"].T
    resid = Y_c - fitted
    total_var = Y_c.var(axis=0, ddof=1).sum()
    resid_var = resid.var(axis=0, ddof=1).sum()
    assert resid_var / total_var < 0.30


# ---------------------------------------------------------------------------
# Spec validation: parameter recovery on simulated state-space
# ---------------------------------------------------------------------------


def test_kalman_parameter_recovery_on_simulated_data(rng: np.random.Generator) -> None:
    """Spec diagnostic: 'Parameter recovery on simulation. Generate synthetic
    data with known (F, H, Q, R); check that MLE recovers them.'"""
    T = 1000
    true_q_beta = 1e-6
    true_R = 1e-4
    # Random-walk beta with small Q.
    market = rng.normal(0, 0.012, size=T)
    alpha_path = np.zeros(T)
    beta_path = np.zeros(T)
    beta_path[0] = 1.0
    for t in range(1, T):
        beta_path[t] = beta_path[t - 1] + rng.normal(0, np.sqrt(true_q_beta))
    asset = alpha_path + beta_path * market + rng.normal(0, np.sqrt(true_R), size=T)
    q_a, q_b, R, _ll = mle_time_varying_beta(asset, market, max_iter=400)
    # Within ~one order of magnitude is the realistic expectation for short-T MLE.
    assert true_q_beta / 50.0 < q_b < true_q_beta * 50.0
    assert true_R / 5.0 < R < true_R * 5.0
    assert q_a > 0
