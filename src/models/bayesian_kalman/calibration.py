"""Calibration entry point for the Bayesian-Hierarchical / Kalman model.

`calibrate()` dispatches on ``mode`` to one of three estimators:

- ``"kalman_beta"``     — MLE of `(q_alpha, q_beta, R)` then filter +
                          smoother.
- ``"hierarchical"``    — Gibbs sampler over the conjugate Normal-Inverse-
                          Wishart hierarchy.
- ``"dynamic_factor"``  — EM (PCA-init) for the Stock-Watson DFM.

All three return the same `CalibrationResult` shape so the orchestration
layer doesn't need to branch.

Data fetching is the model class's job — keeping I/O out of this module makes
the math unit-testable with synthetic inputs.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.bayesian_kalman.signal import (
    _DEFAULT_DAILY_BETA_DRIFT,
    dynamic_factor_em,
    hierarchical_gibbs,
    kalman_smoother,
    mle_time_varying_beta,
    time_varying_beta_filter,
)
from src.models.bayesian_kalman.types import (
    VALID_MODES,
    BayesianKalmanInputs,
    DynamicFactorFit,
    HierarchicalFit,
    KalmanFit,
    ModelMode,
)

MODEL_NAME: str = "bayesian_kalman"


def calibrate(
    *,
    inputs: BayesianKalmanInputs,
    n_iter: int = 1000,
    n_burn: int = 500,
    n_em_iter: int = 30,
    n_factors: int = 3,
    mle_max_iter: int = 200,
    fix_hyperparameters: tuple[float, float, float] | None = None,
    seed: int = 20260518,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Fit one of the three model families.

    Parameters
    ----------
    inputs:
        Payload from `BayesianKalman.fetch_data`. ``inputs.mode`` selects the
        estimator.
    n_iter / n_burn:
        Gibbs iterations (hierarchical mode only).
    n_em_iter:
        EM iterations (dynamic_factor mode only).
    n_factors:
        Number of latent factors (dynamic_factor mode only).
    mle_max_iter:
        Maximum Nelder-Mead iterations for the Kalman MLE.
    fix_hyperparameters:
        If provided, skip MLE and use ``(q_alpha, q_beta, R)`` directly
        (kalman_beta mode only). Useful for testing or fast incremental refits.
    seed:
        Seed for the Gibbs sampler.
    timestamp:
        Override the wall-clock fit timestamp.
    """

    mode = inputs.mode
    if mode not in VALID_MODES:
        raise ValueError(
            f"calibrate: mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
        )
    ts = timestamp or datetime.now(UTC)

    if mode == "kalman_beta":
        return _calibrate_kalman(
            inputs=inputs,
            mle_max_iter=mle_max_iter,
            fix_hyperparameters=fix_hyperparameters,
            timestamp=ts,
        )
    if mode == "hierarchical":
        return _calibrate_hierarchical(
            inputs=inputs,
            n_iter=n_iter,
            n_burn=n_burn,
            seed=seed,
            timestamp=ts,
        )
    return _calibrate_dynamic_factor(
        inputs=inputs,
        n_factors=n_factors,
        n_em_iter=n_em_iter,
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# Kalman-beta path
# ---------------------------------------------------------------------------


def _calibrate_kalman(
    *,
    inputs: BayesianKalmanInputs,
    mle_max_iter: int,
    fix_hyperparameters: tuple[float, float, float] | None,
    timestamp: datetime,
) -> CalibrationResult:
    assert inputs.asset_returns is not None and inputs.market_returns is not None
    asset = inputs.asset_returns.dropna()
    market = inputs.market_returns.reindex(asset.index).dropna()
    joined = pd.concat([asset, market], axis=1, join="inner").dropna()
    if joined.shape[0] < 30:
        raise ValueError(
            f"kalman_beta calibration requires >=30 aligned observations, got "
            f"{joined.shape[0]}"
        )
    asset_arr = joined.iloc[:, 0].to_numpy(dtype=float)
    market_arr = joined.iloc[:, 1].to_numpy(dtype=float)
    index = joined.index

    if fix_hyperparameters is not None:
        q_alpha, q_beta, R = fix_hyperparameters
        if q_alpha < 0 or q_beta < 0 or R < 0:
            raise ValueError(
                f"fix_hyperparameters must be non-negative, got {fix_hyperparameters}"
            )
        run = time_varying_beta_filter(
            asset_arr, market_arr, q_alpha=q_alpha, q_beta=q_beta, R=R
        )
        log_lik = run.log_likelihood
    else:
        q_alpha, q_beta, R, log_lik = mle_time_varying_beta(
            asset_arr, market_arr, max_iter=mle_max_iter
        )
        run = time_varying_beta_filter(
            asset_arr, market_arr, q_alpha=q_alpha, q_beta=q_beta, R=R
        )

    F = np.eye(2)
    Q = np.diag([q_alpha, q_beta])
    sm_states, sm_covs = kalman_smoother(run, F, Q)

    fit = KalmanFit(
        F=F,
        Q=Q,
        R=R,
        x0=np.array([0.0, 1.0]),
        P0=np.diag([0.01, 0.1]),
        states=run.states,
        covariances=run.covariances,
        innovations=run.innovations,
        innovation_variances=run.innovation_variances,
        log_likelihood=log_lik,
        index=index,
        smoothed_states=sm_states,
        smoothed_covariances=sm_covs,
        state_names=("alpha", "beta"),
        metadata={"hyperparameters_source": "fixed" if fix_hyperparameters else "mle"},
    )

    parameters: dict[str, object] = {
        "kalman_fit": fit,
        "q_alpha": float(q_alpha),
        "q_beta": float(q_beta),
        "R": float(R),
        "mode": "kalman_beta",
    }
    fit_metrics: dict[str, float] = {
        "log_likelihood": float(log_lik),
        "n_obs": float(run.states.shape[0]),
        "q_alpha": float(q_alpha),
        "q_beta": float(q_beta),
        "R": float(R),
        "latest_alpha": float(run.states[-1, 0]),
        "latest_beta": float(run.states[-1, 1]),
        "mean_innovation": float(run.innovations.mean()),
        "innovation_variance": float(run.innovations.var(ddof=1)),
        "q_beta_drift_units": float(q_beta / max(_DEFAULT_DAILY_BETA_DRIFT, 1e-30)),
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=timestamp,
        metadata={"mode": "kalman_beta"},
    )


# ---------------------------------------------------------------------------
# Hierarchical-Bayes path
# ---------------------------------------------------------------------------


def _calibrate_hierarchical(
    *,
    inputs: BayesianKalmanInputs,
    n_iter: int,
    n_burn: int,
    seed: int,
    timestamp: datetime,
) -> CalibrationResult:
    assert inputs.panel_design is not None and inputs.panel_targets is not None
    tickers = inputs.tickers
    if not tickers:
        raise ValueError("hierarchical mode requires inputs.tickers")
    designs = [inputs.panel_design[t] for t in tickers]
    targets = [inputs.panel_targets[t] for t in tickers]
    if any(d.shape[0] < 5 for d in designs):
        raise ValueError(
            "hierarchical_gibbs requires at least 5 observations per asset"
        )

    posterior = hierarchical_gibbs(
        designs,
        targets,
        n_iter=n_iter,
        n_burn=n_burn,
        seed=seed,
    )
    p = posterior["mu_mean"].shape[0]
    coef_names = inputs.coef_names or tuple(f"x{i}" for i in range(p))

    fit = HierarchicalFit(
        mu_mean=posterior["mu_mean"],
        mu_cov=posterior["mu_cov"],
        Sigma_mean=posterior["Sigma_mean"],
        theta_means=posterior["theta_means"],
        theta_covs=posterior["theta_covs"],
        sigma_sq_means=posterior["sigma_sq_means"],
        tickers=tickers,
        coef_names=coef_names,
        n_iter=n_iter,
        n_burn=n_burn,
    )

    fit_metrics: dict[str, float] = {
        "n_assets": float(fit.n_assets),
        "n_coefs": float(fit.n_coefs),
        "n_iter": float(n_iter),
        "n_burn": float(n_burn),
        "mean_sigma_sq": float(fit.sigma_sq_means.mean()),
        "max_theta_norm": float(np.linalg.norm(fit.theta_means, axis=1).max()),
        "mu_norm": float(np.linalg.norm(fit.mu_mean)),
        "Sigma_trace": float(np.trace(fit.Sigma_mean)),
    }
    parameters: dict[str, object] = {
        "hierarchical_fit": fit,
        "mode": "hierarchical",
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=timestamp,
        metadata={"mode": "hierarchical"},
    )


# ---------------------------------------------------------------------------
# Dynamic-factor path
# ---------------------------------------------------------------------------


def _calibrate_dynamic_factor(
    *,
    inputs: BayesianKalmanInputs,
    n_factors: int,
    n_em_iter: int,
    timestamp: datetime,
) -> CalibrationResult:
    assert inputs.panel_returns is not None
    panel = inputs.panel_returns
    if panel.empty:
        raise ValueError("dynamic_factor calibration requires a non-empty panel")
    if n_factors < 1 or n_factors > panel.shape[1]:
        raise ValueError(
            f"n_factors must be in [1, {panel.shape[1]}], got {n_factors}"
        )

    out = dynamic_factor_em(panel, r=n_factors, n_iter=n_em_iter)
    tickers = tuple(str(c) for c in panel.columns)
    factor_names = tuple(f"F{i + 1}" for i in range(n_factors))
    fit = DynamicFactorFit(
        Lambda=out["Lambda"],
        Phi=out["Phi"],
        Q=out["Q"],
        Psi=out["Psi"],
        factors=out["factors"],
        tickers=tickers,
        factor_names=factor_names,
        log_likelihood=float(out["log_likelihood"]),
        n_iter=int(out["iters"]),
        index=panel.index,
    )
    fit_metrics: dict[str, float] = {
        "log_likelihood": fit.log_likelihood,
        "n_iter": float(fit.n_iter),
        "n_assets": float(fit.n_assets),
        "n_factors": float(fit.n_factors),
        "mean_psi": float(np.mean(fit.Psi)),
        "phi_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(fit.Phi)))),
        "variance_explained": _dfm_variance_explained(panel, fit),
    }
    parameters: dict[str, object] = {
        "dynamic_factor_fit": fit,
        "mode": "dynamic_factor",
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=timestamp,
        metadata={"mode": "dynamic_factor"},
    )


def _dfm_variance_explained(panel: pd.DataFrame, fit: DynamicFactorFit) -> float:
    """Average per-asset R^2: 1 - var(resid_i) / var(y_i)."""
    Y = panel.to_numpy(dtype=float)
    Y_c = Y - Y.mean(axis=0)
    fitted = fit.factors @ fit.Lambda.T
    resid = Y_c - fitted
    total = Y_c.var(axis=0, ddof=1)
    total = np.where(total > 0, total, 1.0)
    r2 = 1.0 - resid.var(axis=0, ddof=1) / total
    return float(np.mean(np.clip(r2, 0.0, 1.0)))


__all__ = ["MODEL_NAME", "ModelMode", "calibrate"]
