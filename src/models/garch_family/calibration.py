"""Maximum-likelihood calibration of GARCH-family models.

This module is the only place where the optimizer lives. It exposes one
public function — `calibrate()` — that takes a returns panel (in **percent**
units, the arch-package convention) plus the spec / distribution choice
and returns a `CalibrationResult` whose `parameters["fit"]` carries a
fully-populated `GARCHFit`.

We roll our own Nelder-Mead since scipy is not available in this environment.
Nelder-Mead is derivative-free (no scipy.optimize, no autograd needed) and
handles the soft constraints introduced by the reparameterization below
without trouble. The simplex algorithm follows Nelder & Mead (1965) with
the standard expansion / contraction / shrink coefficients.

Reparameterization to enforce positivity + stationarity:
  - omega = exp(omega_tilde)                                  (> 0)
  - GARCH:  (alpha, beta) via 2-way softmax on (alpha_t, beta_t):
        s = 1 + exp(alpha_t) + exp(beta_t)
        alpha = exp(alpha_t) / s,  beta = exp(beta_t) / s
        alpha + beta = (s - 1) / s in (0, 1)                  (stationarity)
  - GJR:    (alpha, gamma/2, beta) via 3-way softmax — keeps
        alpha + gamma/2 + beta < 1, all >= 0.
  - EGARCH: omega unconstrained; alpha, gamma unconstrained;
        beta = tanh(beta_t) in (-1, 1)                        (stationarity)
  - Student-t: nu = 2.5 + exp(nu_tilde) (>= 2.5; lower bound keeps the
    second moment well-defined and the optimizer away from the
    nu -> 2 singularity in E|z|).

The mean parameter `mu` is fitted as a free real, no transform.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.garch_family.signal import (
    compute_sigma2_path,
    negloglik,
)
from src.models.garch_family.types import (
    VALID_DISTS,
    VALID_MEAN_MODELS,
    VALID_SPECS,
    GARCHFit,
    GARCHParams,
    GARCHSpec,
    InnovationDist,
    MeanModel,
)

MODEL_NAME: str = "garch_family"

# Lower floor on nu so Student-t MLE never grazes the nu -> 2 singularity
# (where the standardized variance and E|z| blow up).
_NU_FLOOR: float = 2.5
# Upper bound on nu: above 200 the Student-t is statistically Gaussian to
# eight digits, so this caps optimizer drift toward `nu -> infinity` (which
# otherwise produces overflow inside math.exp).
_NU_RANGE: float = 200.0
# Hard upper bound on persistence in the unconstrained -> constrained map
# (prevents `alpha + beta = 0.99999999`; that region is near-IGARCH and
# numerically unstable).
_PERSISTENCE_CEILING: float = 0.999


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------


def _softmax2(a_t: float, b_t: float) -> tuple[float, float]:
    """Map two reals -> (alpha, beta) with alpha + beta < 1, both >= 0."""

    # Subtract a constant for numerical stability.
    m = max(0.0, a_t, b_t)
    ea = math.exp(a_t - m)
    eb = math.exp(b_t - m)
    one = math.exp(-m)
    denom = one + ea + eb
    alpha = ea / denom
    beta = eb / denom
    # Cap persistence to keep us strictly inside the stationarity boundary.
    s = alpha + beta
    if s > _PERSISTENCE_CEILING:
        scale = _PERSISTENCE_CEILING / s
        alpha *= scale
        beta *= scale
    return alpha, beta


def _softmax3(a_t: float, g_t: float, b_t: float) -> tuple[float, float, float]:
    """Map three reals -> (alpha, gamma/2, beta) with their sum < 1, all >= 0.

    Returns the *gamma* (not gamma/2) for the caller's convenience.
    """

    m = max(0.0, a_t, g_t, b_t)
    ea = math.exp(a_t - m)
    eg = math.exp(g_t - m)
    eb = math.exp(b_t - m)
    one = math.exp(-m)
    denom = one + ea + eg + eb
    alpha = ea / denom
    gamma_half = eg / denom
    beta = eb / denom
    s = alpha + gamma_half + beta
    if s > _PERSISTENCE_CEILING:
        scale = _PERSISTENCE_CEILING / s
        alpha *= scale
        gamma_half *= scale
        beta *= scale
    return alpha, 2.0 * gamma_half, beta


def _params_from_theta(
    theta: np.ndarray,
    *,
    spec: GARCHSpec,
    distribution: InnovationDist,
    mean_model: MeanModel,
) -> GARCHParams:
    """Inverse of the parameter packing used by `_initial_theta`."""

    i = 0
    mu = 0.0
    if mean_model == "Constant":
        mu = float(theta[i])
        i += 1

    # Clip the exponent so log-omega drift can't overflow math.exp.
    omega_t = max(-50.0, min(50.0, float(theta[i])))
    omega = math.exp(omega_t)
    i += 1

    if spec == "GARCH":
        alpha, beta = _softmax2(float(theta[i]), float(theta[i + 1]))
        gamma = 0.0
        i += 2
    elif spec == "GJR":
        alpha, gamma, beta = _softmax3(
            float(theta[i]), float(theta[i + 1]), float(theta[i + 2])
        )
        i += 3
    elif spec == "EGARCH":
        alpha = float(theta[i])
        gamma = float(theta[i + 1])
        # beta = tanh(beta_t) in (-1, 1)
        beta = math.tanh(float(theta[i + 2]))
        i += 3
    else:
        raise ValueError(f"Unknown spec: {spec!r}")

    nu: float | None = None
    if distribution == "Student-t":
        # nu = floor + range * sigmoid(nu_tilde) -- bounded in (floor, floor+range).
        nu = _NU_FLOOR + _NU_RANGE * _sigmoid(float(theta[i]))
        i += 1

    return GARCHParams(
        omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu, mu=mu
    )


def _initial_theta(
    returns: np.ndarray,
    *,
    spec: GARCHSpec,
    distribution: InnovationDist,
    mean_model: MeanModel,
) -> np.ndarray:
    """Spec-recommended starting values, pushed through the inverse transform."""

    sample_var = float(np.var(returns, ddof=1))
    if not math.isfinite(sample_var) or sample_var <= 0:
        sample_var = 1.0
    sample_mean = float(np.mean(returns))

    # Targets: alpha = 0.05, beta = 0.9, gamma = 0.05 (GJR), omega = 0.01 * var.
    omega_t = math.log(max(0.01 * sample_var, 1e-8))
    # For softmax2: solve for (a_t, b_t) given alpha, beta in (0, 1) with
    # alpha + beta < 1. Use log of the (unnormalized) weights with the
    # "one" baseline implicit at zero:
    #   alpha = exp(a_t) / (1 + exp(a_t) + exp(b_t)),
    #   beta  = exp(b_t) / (1 + exp(a_t) + exp(b_t)).
    # => one = 1 - alpha - beta, exp(a_t) = alpha / one, exp(b_t) = beta / one.

    parts: list[float] = []
    if mean_model == "Constant":
        parts.append(sample_mean)
    parts.append(omega_t)

    if spec == "GARCH":
        alpha_init, beta_init = 0.05, 0.9
        one = 1.0 - alpha_init - beta_init
        parts.extend([math.log(alpha_init / one), math.log(beta_init / one)])
    elif spec == "GJR":
        alpha_init, gamma_init, beta_init = 0.03, 0.05, 0.9
        gh = 0.5 * gamma_init
        one = 1.0 - alpha_init - gh - beta_init
        parts.extend(
            [
                math.log(alpha_init / one),
                math.log(gh / one),
                math.log(beta_init / one),
            ]
        )
    elif spec == "EGARCH":
        # EGARCH has wider tolerance — start at alpha=0.1, gamma=-0.05, beta=0.9.
        alpha_init, gamma_init, beta_init = 0.1, -0.05, 0.9
        beta_t = math.atanh(beta_init) if abs(beta_init) < 1.0 else 0.0
        parts.extend([alpha_init, gamma_init, beta_t])
    else:
        raise ValueError(f"Unknown spec: {spec!r}")

    if distribution == "Student-t":
        # nu = floor + range * sigmoid(nu_t); target nu = 8 ->
        # sigmoid(nu_t) = (8 - floor) / range -> nu_t = logit(...)
        target = (8.0 - _NU_FLOOR) / _NU_RANGE
        parts.append(math.log(target / (1.0 - target)))

    return np.asarray(parts, dtype=float)


# ---------------------------------------------------------------------------
# Nelder-Mead optimizer
# ---------------------------------------------------------------------------


def _nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    max_iter: int = 2000,
    xtol: float = 1e-6,
    ftol: float = 1e-7,
    init_step: float = 0.2,
) -> tuple[np.ndarray, float, int, bool]:
    """Standard Nelder-Mead simplex search. Returns (x, f(x), n_iter, converged)."""

    n = x0.size
    simplex = np.empty((n + 1, n), dtype=float)
    simplex[0] = x0
    for i in range(n):
        v = x0.copy()
        delta = init_step if x0[i] == 0 else init_step * abs(x0[i])
        if delta < 1e-4:
            delta = 1e-4
        v[i] = x0[i] + delta
        simplex[i + 1] = v

    fvals = np.asarray([f(v) for v in simplex], dtype=float)

    alpha_r = 1.0  # reflection
    gamma_e = 2.0  # expansion
    rho_c = 0.5  # contraction
    sigma_s = 0.5  # shrinkage

    n_iter = 0
    converged = False
    for it in range(1, max_iter + 1):
        n_iter = it
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        # Convergence check.
        f_spread = fvals[-1] - fvals[0]
        x_spread = float(np.max(np.linalg.norm(simplex[1:] - simplex[0], axis=1)))
        if f_spread < ftol and x_spread < xtol:
            converged = True
            break

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        x_r = centroid + alpha_r * (centroid - worst)
        f_r = f(x_r)

        if fvals[0] <= f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
            continue

        if f_r < fvals[0]:
            x_e = centroid + gamma_e * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
            continue

        # f_r >= fvals[-2]: contract.
        if f_r < fvals[-1]:
            # outside contraction
            x_oc = centroid + rho_c * (x_r - centroid)
            f_oc = f(x_oc)
            if f_oc <= f_r:
                simplex[-1] = x_oc
                fvals[-1] = f_oc
                continue
        else:
            # inside contraction
            x_ic = centroid + rho_c * (worst - centroid)
            f_ic = f(x_ic)
            if f_ic < fvals[-1]:
                simplex[-1] = x_ic
                fvals[-1] = f_ic
                continue

        # Shrink toward the best.
        best = simplex[0]
        for k in range(1, n + 1):
            simplex[k] = best + sigma_s * (simplex[k] - best)
            fvals[k] = f(simplex[k])

    order = np.argsort(fvals)
    return simplex[order[0]].copy(), float(fvals[order[0]]), n_iter, converged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def calibrate(
    *,
    returns: pd.Series,
    spec: GARCHSpec = "GARCH",
    distribution: InnovationDist = "Student-t",
    mean_model: MeanModel = "Constant",
    timestamp: datetime | None = None,
    max_iter: int = 2000,
) -> CalibrationResult:
    """Fit a GARCH-family model by MLE.

    Parameters
    ----------
    returns:
        `pd.Series` of daily log returns in **percent** units (i.e.
        `100 * log diff`). Index should be a DatetimeIndex; that's not
        enforced here.
    spec / distribution / mean_model:
        Model configuration. See `GARCHSpec`, `InnovationDist`, `MeanModel`.
    timestamp:
        Optional fit-time stamp; defaults to `datetime.now(UTC)`.
    max_iter:
        Cap on Nelder-Mead iterations. Typical convergence on 2500 obs is
        300-800 iterations.
    """

    if spec not in VALID_SPECS:
        raise ValueError(f"spec must be one of {sorted(VALID_SPECS)}, got {spec!r}")
    if distribution not in VALID_DISTS:
        raise ValueError(
            f"distribution must be one of {sorted(VALID_DISTS)}, got {distribution!r}"
        )
    if mean_model not in VALID_MEAN_MODELS:
        raise ValueError(
            f"mean_model must be one of {sorted(VALID_MEAN_MODELS)}, got {mean_model!r}"
        )
    if len(returns) < 30:
        raise ValueError(
            f"calibrate requires >= 30 observations, got {len(returns)}"
        )

    ts = timestamp or datetime.now(UTC)
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError(
            f"calibrate requires >= 30 finite returns, got {r.size}"
        )

    theta0 = _initial_theta(
        r, spec=spec, distribution=distribution, mean_model=mean_model
    )

    def objective(theta: np.ndarray) -> float:
        params = _params_from_theta(
            theta, spec=spec, distribution=distribution, mean_model=mean_model
        )
        return negloglik(r, params, spec, distribution)

    theta_hat, neg_ll, n_iter, converged = _nelder_mead(
        objective, theta0, max_iter=max_iter
    )
    params_hat = _params_from_theta(
        theta_hat, spec=spec, distribution=distribution, mean_model=mean_model
    )
    sigma2_path, eps_path = compute_sigma2_path(r, params_hat, spec)
    sigma_path = np.sqrt(sigma2_path)
    std_resid = eps_path / np.where(sigma_path > 0, sigma_path, 1.0)

    persistence: float
    unconditional_variance: float | None
    if spec == "GARCH":
        persistence = params_hat.alpha + params_hat.beta
        unconditional_variance = (
            params_hat.omega / (1.0 - persistence) if persistence < 1.0 else None
        )
    elif spec == "GJR":
        persistence = params_hat.alpha + 0.5 * params_hat.gamma + params_hat.beta
        unconditional_variance = (
            params_hat.omega / (1.0 - persistence) if persistence < 1.0 else None
        )
    else:  # EGARCH
        persistence = params_hat.beta
        unconditional_variance = None

    log_likelihood = -neg_ll
    k = theta_hat.size
    n = r.size
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = k * math.log(n) - 2.0 * log_likelihood

    fit = GARCHFit(
        params=params_hat,
        spec=spec,
        distribution=distribution,
        mean_model=mean_model,
        sigma2_path=sigma2_path,
        eps_path=eps_path,
        std_resid=std_resid,
        log_likelihood=log_likelihood,
        converged=converged,
        n_iter=n_iter,
        n_obs=n,
        unconditional_variance=unconditional_variance,
    )

    fit_metrics: dict[str, float] = {
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "n_obs": float(n),
        "n_params": float(k),
        "persistence": persistence,
        "mean_sigma_pct": float(sigma_path.mean()),
        "final_sigma_pct": float(sigma_path[-1]),
        "n_iter": float(n_iter),
        "converged": 1.0 if converged else 0.0,
    }
    if unconditional_variance is not None:
        fit_metrics["unconditional_variance"] = unconditional_variance
        fit_metrics["unconditional_vol_pct"] = math.sqrt(unconditional_variance)

    parameters: dict[str, object] = {
        "fit": fit,
        "params": params_hat.to_dict(),
        "spec": spec,
        "distribution": distribution,
        "mean_model": mean_model,
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata={
            "n_iter": n_iter,
            "converged": converged,
            "persistence": persistence,
        },
    )


__all__ = ["MODEL_NAME", "calibrate"]
