"""Pure math layer of the Bayesian-hierarchical / Kalman model.

Nothing in this module performs I/O. Every function is a deterministic
transformation of arrays (or pandas Series at the data-loading boundary),
which keeps the math unit-testable without a mock provider.

The functions split into three clusters:

- Kalman: ``kalman_filter``, ``kalman_smoother``, ``kalman_log_likelihood``,
  ``time_varying_beta_filter``, plus helpers ``standardized_innovations``
  and ``ljung_box_pvalue``.
- Hierarchical Bayes: ``hierarchical_gibbs`` (conjugate Normal-Inverse-Wishart
  sampler) and ``hierarchical_posterior_mean`` (closed-form Gaussian-Gaussian
  conjugate solution used as a sanity check).
- Dynamic factor model: ``dynamic_factor_em`` (Stock-Watson via EM with PCA
  initialization), plus the multivariate Kalman primitives it relies on.

The MLE for the Kalman hyperparameters is solved with a hand-rolled Nelder-
Mead simplex — scipy is not a project dependency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

# ---- numerical constants ----------------------------------------------------

_DEFAULT_MIN_VARIANCE: float = 1e-12
_DEFAULT_DAILY_BETA_DRIFT: float = (0.05 / 252.0) ** 2  # spec heuristic for q_beta
_DEFAULT_DAILY_ALPHA_DRIFT: float = (0.001 / 252.0) ** 2
_LOG_2PI: float = math.log(2.0 * math.pi)


# =============================================================================
# Kalman filter / smoother (scalar observations)
# =============================================================================


@dataclass(frozen=True)
class _KalmanRun:
    """Internal carrier — what ``kalman_filter`` returns before being lifted
    into a ``KalmanFit`` dataclass."""

    states: np.ndarray
    covariances: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray
    log_likelihood: float


def kalman_filter(
    y: np.ndarray,
    H_seq: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: float,
    x0: np.ndarray,
    P0: np.ndarray,
) -> _KalmanRun:
    """Linear-Gaussian Kalman filter for scalar observations.

    Parameters
    ----------
    y:
        ``(T,)`` observation series.
    H_seq:
        ``(T, n)`` time-varying row-vector observation matrices `H_t`. For a
        constant `H`, pass ``np.tile(H, (T, 1))`` upstream.
    F:
        ``(n, n)`` transition matrix.
    Q:
        ``(n, n)`` state-innovation covariance.
    R:
        Scalar observation-noise variance.
    x0:
        ``(n,)`` initial state mean.
    P0:
        ``(n, n)`` initial state covariance.

    Returns
    -------
    `_KalmanRun` holding the filtered states, covariances, innovations,
    innovation variances, and total Gaussian log-likelihood.
    """

    T = y.shape[0]
    n = x0.shape[0]
    if H_seq.shape != (T, n):
        raise ValueError(f"H_seq shape {H_seq.shape} != ({T}, {n})")
    if F.shape != (n, n) or Q.shape != (n, n) or P0.shape != (n, n):
        raise ValueError(
            f"F/Q/P0 must be ({n}, {n}); got {F.shape}, {Q.shape}, {P0.shape}"
        )
    if R < 0:
        raise ValueError(f"R must be non-negative, got {R}")

    states = np.zeros((T, n))
    covariances = np.zeros((T, n, n))
    innovations = np.zeros(T)
    innovation_vars = np.zeros(T)
    I_n = np.eye(n)

    x = x0.copy()
    P = P0.copy()
    log_lik = 0.0
    for t in range(T):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # Symmetrize (small numerical drift)
        P_pred = 0.5 * (P_pred + P_pred.T)

        H_t = H_seq[t]
        nu = float(y[t] - H_t @ x_pred)
        S = float(H_t @ P_pred @ H_t.T + R)
        if S <= 0:
            S = _DEFAULT_MIN_VARIANCE
        K = (P_pred @ H_t) / S  # (n,)
        x_new = x_pred + K * nu
        # Joseph form for stability
        A = I_n - np.outer(K, H_t)
        P_new = A @ P_pred @ A.T + R * np.outer(K, K)
        P_new = 0.5 * (P_new + P_new.T)

        states[t] = x_new
        covariances[t] = P_new
        innovations[t] = nu
        innovation_vars[t] = S
        log_lik += -0.5 * (_LOG_2PI + math.log(S) + nu * nu / S)

        x = x_new
        P = P_new

    return _KalmanRun(
        states=states,
        covariances=covariances,
        innovations=innovations,
        innovation_variances=innovation_vars,
        log_likelihood=log_lik,
    )


def kalman_smoother(
    run: _KalmanRun,
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """RTS smoother (backward pass).

    Returns ``(smoothed_states, smoothed_covariances)``, both shaped the same
    as ``run.states`` / ``run.covariances``.
    """

    T, n = run.states.shape
    smoothed_states = run.states.copy()
    smoothed_covs = run.covariances.copy()

    for t in range(T - 2, -1, -1):
        P_t = run.covariances[t]
        x_t = run.states[t]
        x_pred_next = F @ x_t
        P_pred_next = F @ P_t @ F.T + Q
        P_pred_next = 0.5 * (P_pred_next + P_pred_next.T)
        # C_t = P_t F^T P_pred_next^{-1}
        try:
            C = P_t @ F.T @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            C = P_t @ F.T @ np.linalg.pinv(P_pred_next)
        smoothed_states[t] = x_t + C @ (smoothed_states[t + 1] - x_pred_next)
        smoothed_covs[t] = P_t + C @ (smoothed_covs[t + 1] - P_pred_next) @ C.T
        smoothed_covs[t] = 0.5 * (smoothed_covs[t] + smoothed_covs[t].T)
    return smoothed_states, smoothed_covs


def kalman_log_likelihood(
    y: np.ndarray,
    H_seq: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: float,
    x0: np.ndarray,
    P0: np.ndarray,
) -> float:
    """Marginal Gaussian log-likelihood of `y` under the state-space."""

    return kalman_filter(y, H_seq, F, Q, R, x0, P0).log_likelihood


def time_varying_beta_filter(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    *,
    q_alpha: float,
    q_beta: float,
    R: float,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> _KalmanRun:
    """Spec algorithm: state ``(alpha_t, beta_t)`` random walk, observation
    ``r_asset = alpha + beta * r_market + eps``.

    All variances are in *daily* return units. `q_beta` near
    ``(0.05/252)^2`` corresponds to a half-life of several years for the beta
    drift (the spec's "slow drift" heuristic).
    """

    T = asset_returns.shape[0]
    if market_returns.shape != (T,):
        raise ValueError(
            f"market_returns length {market_returns.shape[0]} != asset T={T}"
        )
    if q_alpha < 0 or q_beta < 0 or R < 0:
        raise ValueError(
            f"variances must be non-negative; got q_alpha={q_alpha}, "
            f"q_beta={q_beta}, R={R}"
        )
    H_seq = np.column_stack([np.ones(T), market_returns])
    F = np.eye(2)
    Q = np.diag([q_alpha, q_beta])
    if x0 is None:
        x0 = np.array([0.0, 1.0])
    if P0 is None:
        P0 = np.diag([0.01, 0.1])
    return kalman_filter(asset_returns, H_seq, F, Q, R, x0, P0)


def standardized_innovations(run: _KalmanRun) -> np.ndarray:
    """`nu_t / sqrt(S_t)` — should be iid N(0,1) under a correctly-specified
    Kalman model."""

    return cast(
        np.ndarray,
        run.innovations / np.sqrt(np.clip(run.innovation_variances, _DEFAULT_MIN_VARIANCE, None)),
    )


# =============================================================================
# MLE for Kalman hyperparameters via Nelder-Mead simplex
# =============================================================================


def mle_time_varying_beta(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    *,
    log_init: tuple[float, float, float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[float, float, float, float]:
    """Maximize the Kalman marginal log-likelihood w.r.t. `(q_alpha, q_beta, R)`.

    Variances are reparameterized as ``v = exp(ell)`` so the unconstrained
    Nelder-Mead simplex stays in the positive orthant.

    Returns ``(q_alpha_hat, q_beta_hat, R_hat, log_likelihood)``.
    """

    if log_init is None:
        log_init = (
            math.log(max(_DEFAULT_DAILY_ALPHA_DRIFT, 1e-16)),
            math.log(max(_DEFAULT_DAILY_BETA_DRIFT, 1e-16)),
            math.log(max(float(np.var(asset_returns, ddof=1)), 1e-12)),
        )

    def neg_ll(theta: np.ndarray) -> float:
        q_a = math.exp(theta[0])
        q_b = math.exp(theta[1])
        rr = math.exp(theta[2])
        try:
            run = time_varying_beta_filter(
                asset_returns,
                market_returns,
                q_alpha=q_a,
                q_beta=q_b,
                R=rr,
            )
        except (ValueError, np.linalg.LinAlgError):
            return 1e18
        if not math.isfinite(run.log_likelihood):
            return 1e18
        return -run.log_likelihood

    best_theta, neg_best = _nelder_mead(
        neg_ll,
        np.array(log_init, dtype=float),
        max_iter=max_iter,
        tol=tol,
    )
    return (
        math.exp(best_theta[0]),
        math.exp(best_theta[1]),
        math.exp(best_theta[2]),
        -neg_best,
    )


def _nelder_mead(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
    step: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Plain Nelder-Mead simplex minimization.

    Returns ``(best_x, best_f)``. Standard parameter values per Nelder & Mead
    (1965); the simplex is initialized by adding ``step`` to each coordinate.
    """

    n = x0.size
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += step
    fvals = np.array([func(p) for p in simplex])

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if float(np.std(fvals)) < tol:
            break
        x_centroid = simplex[:-1].mean(axis=0)
        x_worst = simplex[-1]
        f_worst = fvals[-1]
        f_second_worst = fvals[-2]
        f_best = fvals[0]
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = func(x_r)
        if f_best <= f_r < f_second_worst:
            simplex[-1] = x_r
            fvals[-1] = f_r
            continue
        if f_r < f_best:
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
            continue
        x_c = x_centroid + rho * (x_worst - x_centroid)
        f_c = func(x_c)
        if f_c < f_worst:
            simplex[-1] = x_c
            fvals[-1] = f_c
            continue
        # Shrink toward best
        best = simplex[0].copy()
        for i in range(1, n + 1):
            simplex[i] = best + sigma * (simplex[i] - best)
            fvals[i] = func(simplex[i])

    order = np.argsort(fvals)
    simplex = simplex[order]
    fvals = fvals[order]
    return simplex[0], float(fvals[0])


# =============================================================================
# Innovation diagnostics (shared with HAR-RV style)
# =============================================================================


def ljung_box_pvalue(residual_series: np.ndarray, max_lag: int = 10) -> float:
    """Ljung-Box Q-statistic p-value against H0 = no autocorrelation.

    Implemented inline (no scipy) using the Wilson-Hilferty chi^2 survival
    approximation. Accurate to better than 0.01 in the tail region we care
    about (p < 0.1). Mirrors the implementation in `factor_models_pca`.
    """

    x = np.asarray(residual_series, dtype=float)
    n = x.size
    if n <= max_lag + 1:
        return float("nan")
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float("nan")
    q_stat = 0.0
    for lag in range(1, max_lag + 1):
        rho = float((x[lag:] * x[:-lag]).sum()) / denom
        q_stat += rho * rho / (n - lag)
    q_stat *= n * (n + 2)
    k = max_lag
    if q_stat <= 0:
        return 1.0
    z = ((q_stat / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def innovation_normality_kurtosis(standardized: np.ndarray) -> float:
    """Excess kurtosis of standardized innovations.

    A Gaussian-noise Kalman fit should yield ~0. Large positive values are a
    classic symptom of fat-tailed shocks and motivate the spec's Student-t
    extension.
    """

    x = np.asarray(standardized, dtype=float)
    n = x.size
    if n < 4:
        return float("nan")
    x = x - x.mean()
    var = float((x * x).sum()) / max(n - 1, 1)
    if var <= 0:
        return float("nan")
    m4 = float((x ** 4).sum()) / n
    return m4 / (var * var) - 3.0


# =============================================================================
# Hierarchical Bayesian regression (Gibbs sampler)
# =============================================================================


def hierarchical_posterior_mean(
    X_i: np.ndarray,
    y_i: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    sigma_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form conjugate Normal-Normal posterior of `theta_i`.

    Spec formula:
        V_i^-1 = Sigma^-1 + (X^T X) / sigma^2
        m_i   = V_i (Sigma^-1 mu + X^T y / sigma^2)

    Returns ``(m_i, V_i)``.
    """

    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be positive, got {sigma_sq}")
    Sigma_inv = np.linalg.inv(Sigma)
    XtX = X_i.T @ X_i
    Xty = X_i.T @ y_i
    V_inv = Sigma_inv + XtX / sigma_sq
    V = np.linalg.inv(V_inv)
    m = V @ (Sigma_inv @ mu + Xty / sigma_sq)
    return m, V


def _sample_mvn(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a single MVNormal sample via Cholesky (jittered if needed)."""

    cov_sym = 0.5 * (cov + cov.T)
    jitter = 0.0
    eye = np.eye(cov_sym.shape[0])
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov_sym + jitter * eye)
            break
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-12)
    else:
        # Fall back to eigendecomposition (slow but bulletproof).
        w, V = np.linalg.eigh(cov_sym)
        w = np.clip(w, 0.0, None)
        L = V * np.sqrt(w)
    return mean + L @ rng.standard_normal(mean.size)


def _sample_inverse_wishart(
    nu: float, S: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample from IW(nu, S) via Bartlett decomposition of W ~ W(nu, S^-1).

    If X ~ IW(nu, S) then X^-1 ~ W(nu, S^-1). We build a Wishart draw and
    invert it.
    """

    p = S.shape[0]
    S_sym = 0.5 * (S + S.T)
    S_inv = np.linalg.inv(S_sym)
    L = np.linalg.cholesky(0.5 * (S_inv + S_inv.T) + 1e-12 * np.eye(p))
    # Bartlett: A lower-triangular with chi2 on diag, N(0,1) off-diag.
    A = np.zeros((p, p))
    for i in range(p):
        df = nu - i
        if df <= 0:
            df = 1.0
        A[i, i] = math.sqrt(rng.chisquare(df))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    LA = L @ A
    W = LA @ LA.T  # Wishart draw with scale S^-1
    W = 0.5 * (W + W.T)
    return cast(np.ndarray, np.linalg.inv(W + 1e-12 * np.eye(p)))


def _sample_inv_gamma(a: float, b: float, rng: np.random.Generator) -> float:
    """Inverse-Gamma(a, b) sample via Gamma reciprocal."""

    if a <= 0 or b <= 0:
        raise ValueError(f"InvGamma requires a, b > 0; got a={a}, b={b}")
    return float(b / rng.gamma(shape=a, scale=1.0))


def hierarchical_gibbs(
    designs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    n_iter: int = 1000,
    n_burn: int = 500,
    prior_mu_mean: np.ndarray | None = None,
    prior_mu_cov: np.ndarray | None = None,
    prior_iw_nu: float | None = None,
    prior_iw_scale: np.ndarray | None = None,
    prior_ig_a: float = 2.0,
    prior_ig_b: float = 0.001,
    seed: int = 20260518,
) -> dict[str, np.ndarray]:
    """Conjugate Gibbs sampler for the hierarchical-Bayesian regression.

    Implements the spec's "hand-rolled Gibbs alternative" pseudocode. Returns
    posterior-mean summaries (not the full chain) for compactness.

    Parameters
    ----------
    designs:
        ``N`` per-asset design matrices ``X_i`` with shapes ``(T_i, p)``.
    targets:
        ``N`` per-asset response vectors ``y_i`` of length ``T_i``. Each
        ``y_i`` must align with ``designs[i]``.
    n_iter / n_burn:
        Total iterations and burn-in count discarded.
    prior_*:
        Sensible Normal/IW/InvGamma defaults are applied when ``None``.

    Returns
    -------
    Dict containing posterior means / covariances of ``mu``, ``Sigma``, the
    per-asset ``theta_i``, and ``sigma_sq_i``.
    """

    if n_iter <= 0 or n_burn < 0 or n_burn >= n_iter:
        raise ValueError("Require 0 <= n_burn < n_iter")
    N = len(designs)
    if N == 0:
        raise ValueError("hierarchical_gibbs requires at least one asset")
    if len(targets) != N:
        raise ValueError(f"designs/targets length mismatch: {N} vs {len(targets)}")
    p = designs[0].shape[1]
    for i, (X_i, y_i) in enumerate(zip(designs, targets, strict=True)):
        if X_i.ndim != 2 or X_i.shape[1] != p:
            raise ValueError(
                f"designs[{i}] must be 2-D with {p} columns; got shape {X_i.shape}"
            )
        if y_i.shape[0] != X_i.shape[0]:
            raise ValueError(
                f"targets[{i}] length {y_i.shape[0]} != designs[{i}].rows "
                f"{X_i.shape[0]}"
            )

    if prior_mu_mean is None:
        prior_mu_mean = np.zeros(p)
    if prior_mu_cov is None:
        prior_mu_cov = np.eye(p)  # tau_0^2 I with tau_0 = 1
    if prior_iw_nu is None:
        prior_iw_nu = float(p + 2)
    if prior_iw_scale is None:
        prior_iw_scale = np.eye(p) * 0.01

    rng = np.random.default_rng(seed)

    # OLS initialization of theta_i
    theta = np.zeros((N, p))
    sigma_sq = np.ones(N) * float(np.var(np.concatenate(list(targets)), ddof=1) or 1.0)
    for i, (X_i, y_i) in enumerate(zip(designs, targets, strict=True)):
        XtX = X_i.T @ X_i
        try:
            theta[i] = np.linalg.solve(XtX + 1e-8 * np.eye(p), X_i.T @ y_i)
        except np.linalg.LinAlgError:
            theta[i] = np.zeros(p)

    mu = theta.mean(axis=0)
    Sigma = np.cov(theta.T) + 0.01 * np.eye(p) if N > 1 else np.eye(p)

    prior_mu_cov_inv = np.linalg.inv(prior_mu_cov)
    n_kept = n_iter - n_burn

    mu_sum = np.zeros(p)
    mu_sq_sum = np.zeros((p, p))
    Sigma_sum = np.zeros((p, p))
    theta_sum = np.zeros((N, p))
    theta_sq_sum = np.zeros((N, p, p))
    sigma_sq_sum = np.zeros(N)

    for it in range(n_iter):
        Sigma_inv = np.linalg.inv(Sigma + 1e-12 * np.eye(p))

        # 1. theta_i | rest
        for i in range(N):
            X_i = designs[i]
            y_i = targets[i]
            V_inv = Sigma_inv + (X_i.T @ X_i) / sigma_sq[i]
            V = np.linalg.inv(V_inv + 1e-12 * np.eye(p))
            m = V @ (Sigma_inv @ mu + (X_i.T @ y_i) / sigma_sq[i])
            theta[i] = _sample_mvn(m, V, rng)

        # 2. mu | rest
        V_mu_inv = prior_mu_cov_inv + N * Sigma_inv
        V_mu = np.linalg.inv(V_mu_inv + 1e-12 * np.eye(p))
        m_mu = V_mu @ (prior_mu_cov_inv @ prior_mu_mean + Sigma_inv @ theta.sum(axis=0))
        mu = _sample_mvn(m_mu, V_mu, rng)

        # 3. Sigma | rest
        diff = theta - mu
        S_sum_data = diff.T @ diff
        Sigma = _sample_inverse_wishart(
            prior_iw_nu + N, prior_iw_scale + S_sum_data, rng
        )

        # 4. sigma_sq_i | rest
        for i in range(N):
            X_i = designs[i]
            y_i = targets[i]
            T_i = X_i.shape[0]
            r = y_i - X_i @ theta[i]
            a = prior_ig_a + 0.5 * T_i
            b = prior_ig_b + 0.5 * float(r @ r)
            sigma_sq[i] = _sample_inv_gamma(a, b, rng)

        if it >= n_burn:
            mu_sum += mu
            mu_sq_sum += np.outer(mu, mu)
            Sigma_sum += Sigma
            theta_sum += theta
            for i in range(N):
                theta_sq_sum[i] += np.outer(theta[i], theta[i])
            sigma_sq_sum += sigma_sq

    mu_mean = mu_sum / n_kept
    mu_cov = mu_sq_sum / n_kept - np.outer(mu_mean, mu_mean)
    mu_cov = 0.5 * (mu_cov + mu_cov.T)
    Sigma_mean = Sigma_sum / n_kept
    theta_means = theta_sum / n_kept
    theta_covs = np.zeros((N, p, p))
    for i in range(N):
        theta_covs[i] = theta_sq_sum[i] / n_kept - np.outer(theta_means[i], theta_means[i])
        theta_covs[i] = 0.5 * (theta_covs[i] + theta_covs[i].T)
    sigma_sq_means = sigma_sq_sum / n_kept

    return {
        "mu_mean": mu_mean,
        "mu_cov": mu_cov,
        "Sigma_mean": Sigma_mean,
        "theta_means": theta_means,
        "theta_covs": theta_covs,
        "sigma_sq_means": sigma_sq_means,
    }


# =============================================================================
# Dynamic factor model (EM, PCA-initialized)
# =============================================================================


def _pca_init(
    returns_centered: np.ndarray, r: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(Lambda0, factors0, Psi0)`` from a top-r PCA of the centered
    returns panel."""

    T, N = returns_centered.shape
    cov = (returns_centered.T @ returns_centered) / max(T - 1, 1)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[idx], 0.0, None)
    eigvecs = eigvecs[:, idx]
    Lambda0 = eigvecs[:, :r] * np.sqrt(eigvals[:r])
    # factor scores f_t = (R Lambda) (Lambda^T Lambda)^-1
    LtL = Lambda0.T @ Lambda0
    factors0 = returns_centered @ Lambda0 @ np.linalg.inv(LtL + 1e-12 * np.eye(r))
    residual = returns_centered - factors0 @ Lambda0.T
    Psi0 = np.maximum(residual.var(axis=0, ddof=1), _DEFAULT_MIN_VARIANCE)
    return Lambda0, factors0, Psi0


def _ar1_fit(factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-factor AR(1) OLS — returns ``(Phi, Q)`` with diagonal `Phi`."""

    r = factors.shape[1]
    phi = np.zeros(r)
    resid_var = np.zeros(r)
    for k in range(r):
        x = factors[:-1, k]
        y = factors[1:, k]
        denom = float(x @ x)
        if denom <= 0:
            phi[k] = 0.0
            resid_var[k] = float(np.var(y, ddof=1) or _DEFAULT_MIN_VARIANCE)
            continue
        phi[k] = float(x @ y) / denom
        resid = y - phi[k] * x
        resid_var[k] = float(np.var(resid, ddof=1) or _DEFAULT_MIN_VARIANCE)
    return np.diag(phi), np.diag(np.maximum(resid_var, _DEFAULT_MIN_VARIANCE))


def _multivariate_kalman(
    Y: np.ndarray,
    Lambda: np.ndarray,
    Phi: np.ndarray,
    Q: np.ndarray,
    Psi: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Multivariate Kalman filter with diagonal observation noise ``Psi``.

    State equation: ``f_t = Phi f_{t-1} + eta_t``, ``eta ~ N(0, Q)``.
    Observation:    ``y_t = Lambda f_t + eps_t``, ``eps ~ N(0, diag(Psi))``.

    Returns ``(states, covariances, states_pred, covariances_pred, log_lik)``
    where the ``_pred`` arrays hold the *predicted* `x_{t|t-1}` / `P_{t|t-1}`
    (used by the smoother) and the un-suffixed arrays hold the filtered
    `x_{t|t}` / `P_{t|t}`.
    """

    T, N = Y.shape
    r = Lambda.shape[1]
    R = np.diag(Psi)
    states = np.zeros((T, r))
    covs = np.zeros((T, r, r))
    states_pred = np.zeros((T, r))
    covs_pred = np.zeros((T, r, r))
    I_r = np.eye(r)
    log_lik = 0.0
    x = x0.copy()
    P = P0.copy()
    for t in range(T):
        x_pred = Phi @ x
        P_pred = Phi @ P @ Phi.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        states_pred[t] = x_pred
        covs_pred[t] = P_pred
        nu = Y[t] - Lambda @ x_pred
        S = Lambda @ P_pred @ Lambda.T + R
        S = 0.5 * (S + S.T)
        try:
            S_inv = np.linalg.inv(S)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                logdet = float("nan")
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            logdet = float("nan")
        K = P_pred @ Lambda.T @ S_inv
        x = x_pred + K @ nu
        A = I_r - K @ Lambda
        P = A @ P_pred @ A.T + K @ R @ K.T
        P = 0.5 * (P + P.T)
        states[t] = x
        covs[t] = P
        if math.isfinite(logdet):
            log_lik += -0.5 * (N * _LOG_2PI + logdet + float(nu @ S_inv @ nu))
    return states, covs, states_pred, covs_pred, log_lik


def _multivariate_smoother(
    states: np.ndarray,
    covs: np.ndarray,
    states_pred: np.ndarray,
    covs_pred: np.ndarray,
    Phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RTS smoother for the multivariate Kalman.

    Returns ``(smoothed_states, smoothed_covs, lag1_covs)`` — the lag-1
    cross-covariance ``P_{t, t-1 | T}`` is needed for the EM M-step.
    """

    T, r = states.shape
    sm_states = states.copy()
    sm_covs = covs.copy()
    lag1 = np.zeros((T, r, r))
    for t in range(T - 2, -1, -1):
        try:
            J = covs[t] @ Phi.T @ np.linalg.inv(covs_pred[t + 1] + 1e-12 * np.eye(r))
        except np.linalg.LinAlgError:
            J = covs[t] @ Phi.T @ np.linalg.pinv(covs_pred[t + 1])
        sm_states[t] = states[t] + J @ (sm_states[t + 1] - states_pred[t + 1])
        sm_covs[t] = covs[t] + J @ (sm_covs[t + 1] - covs_pred[t + 1]) @ J.T
        sm_covs[t] = 0.5 * (sm_covs[t] + sm_covs[t].T)
        lag1[t + 1] = sm_covs[t + 1] @ J.T
    return sm_states, sm_covs, lag1


def dynamic_factor_em(
    returns: pd.DataFrame,
    *,
    r: int,
    n_iter: int = 30,
    tol: float = 1e-5,
    min_psi: float = _DEFAULT_MIN_VARIANCE,
) -> dict[str, Any]:
    """Stock-Watson dynamic factor model via EM with PCA initialization.

    Parameters
    ----------
    returns:
        ``(T, N)`` returns panel (already centered or not — we center
        internally before fitting).
    r:
        Number of latent factors to extract.
    n_iter:
        Maximum EM iterations.
    tol:
        Convergence tolerance on the log-likelihood gain.

    Returns
    -------
    Dict with keys ``Lambda``, ``Phi``, ``Q``, ``Psi``, ``factors``,
    ``log_likelihood``, and ``iters``.
    """

    if returns.empty:
        raise ValueError("dynamic_factor_em requires a non-empty returns panel")
    T, N = returns.shape
    if r < 1 or r > N:
        raise ValueError(f"r must be in [1, N={N}], got {r}")
    if r + 2 > T:
        raise ValueError(f"need at least r+2 observations, got T={T}")

    Y = returns.to_numpy(dtype=float)
    Y_centered = Y - Y.mean(axis=0)

    Lambda, factors0, Psi = _pca_init(Y_centered, r)
    Phi, Q = _ar1_fit(factors0)
    x0 = factors0[0].copy()
    P0 = np.eye(r) * 1.0
    prev_ll = -np.inf
    iters_used = 0

    for _it in range(n_iter):
        iters_used = _it + 1
        states, covs, states_pred, covs_pred, ll = _multivariate_kalman(
            Y_centered, Lambda, Phi, Q, Psi, x0, P0
        )
        sm_states, sm_covs, lag1 = _multivariate_smoother(
            states, covs, states_pred, covs_pred, Phi
        )

        # Sufficient statistics
        # E[f_t f_t^T] = sm_covs[t] + sm_states[t] sm_states[t]^T
        # E[f_t f_{t-1}^T] = lag1[t] + sm_states[t] sm_states[t-1]^T  (t>=1)
        Eff = sm_covs + np.einsum("ti,tj->tij", sm_states, sm_states)
        S_xx = Eff.sum(axis=0)
        if T >= 2:
            S_xx_lag = Eff[:-1].sum(axis=0)  # sum over t = 0..T-2 of E[f_t f_t^T]
            cross = lag1[1:].sum(axis=0) + np.einsum(
                "ti,tj->ij", sm_states[1:], sm_states[:-1]
            )
        else:
            S_xx_lag = np.eye(r)
            cross = np.zeros((r, r))

        # M-step
        Y_states = Y_centered.T @ sm_states  # (N, r)
        Lambda = Y_states @ np.linalg.inv(S_xx + 1e-12 * np.eye(r))
        # Psi diagonal
        resid = Y_centered - sm_states @ Lambda.T
        # Var per asset, with correction term for sm_covs
        # Var_i = mean( (y_it - Lambda_i sm_state_t)^2 + Lambda_i sm_cov_t Lambda_i^T )
        var_correction = np.einsum("ij,tjk,ik->ti", Lambda, sm_covs, Lambda)
        Psi = np.maximum(
            np.mean(resid ** 2 + var_correction, axis=0), min_psi
        )
        if T >= 2:
            Phi = cross @ np.linalg.inv(S_xx_lag + 1e-12 * np.eye(r))
            Q_new = (S_xx - Eff[0] - Phi @ cross.T) / max(T - 1, 1)
            Q_new = 0.5 * (Q_new + Q_new.T)
            # Floor eigenvalues
            w, V = np.linalg.eigh(Q_new)
            w = np.clip(w, min_psi, None)
            Q = (V * w) @ V.T
        x0 = sm_states[0]
        P0 = sm_covs[0]

        if math.isfinite(ll) and abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            prev_ll = ll
            break
        prev_ll = ll

    return {
        "Lambda": Lambda,
        "Phi": Phi,
        "Q": Q,
        "Psi": Psi,
        "factors": sm_states,
        "log_likelihood": float(prev_ll),
        "iters": float(iters_used),
    }
