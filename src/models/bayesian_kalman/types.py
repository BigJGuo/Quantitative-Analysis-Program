"""Dataclasses specific to the Bayesian-Hierarchical / Kalman model.

Shared `Forecast` / `Signal` / `RiskMetric` / `CalibrationResult` live in
`src.core.types`; this module holds the model-internal shapes that flow
between `fetch_data`, `calibrate`, and `predict`.

The module exposes three loosely related fits because the spec covers three
related families:

- ``KalmanFit``        — linear-Gaussian state-space (time-varying beta etc.).
- ``HierarchicalFit``  — partial-pooling Bayesian regression posterior summary.
- ``DynamicFactorFit`` — Stock-Watson latent-factor decomposition.

Plus the ``BayesianKalmanInputs`` payload that `fetch_data` produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

ModelMode = Literal["kalman_beta", "hierarchical", "dynamic_factor"]
VALID_MODES: frozenset[str] = frozenset({"kalman_beta", "hierarchical", "dynamic_factor"})


@dataclass(frozen=True)
class KalmanFit:
    """Fitted linear-Gaussian state-space model and the filtered/smoothed states.

    The state convention follows the spec's time-varying-beta special case:
    ``x_t = (alpha_t, beta_t)`` for the 2-D case, but the dataclass is generic
    and supports any ``n``-dimensional state.

    Attributes
    ----------
    F:
        ``(n, n)`` transition matrix. Held constant over the sample.
    Q:
        ``(n, n)`` state-innovation covariance. PSD.
    R:
        Scalar observation-noise variance (we only support scalar observations
        for now — multivariate `R` would only appear in the dynamic-factor
        path, which has its own `DynamicFactorFit`).
    x0 / P0:
        Initial state mean / covariance prior.
    states:
        ``(T, n)`` filtered state means `x_{t|t}`.
    covariances:
        ``(T, n, n)`` filtered state covariances `P_{t|t}`.
    smoothed_states / smoothed_covariances:
        Same shapes — RTS smoother output. May be ``None`` when the filter is
        run in online-only mode.
    innovations:
        Length-``T`` innovation series `nu_t = y_t - H_t x_{t|t-1}`.
    innovation_variances:
        Length-``T`` innovation variances `S_t`.
    log_likelihood:
        Sum of per-step log-likelihoods (Gaussian).
    index:
        The pandas index aligned to the observations (kept for plotting /
        downstream slicing).
    """

    F: np.ndarray
    Q: np.ndarray
    R: float
    x0: np.ndarray
    P0: np.ndarray
    states: np.ndarray
    covariances: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray
    log_likelihood: float
    index: pd.Index
    smoothed_states: np.ndarray | None = None
    smoothed_covariances: np.ndarray | None = None
    state_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.F.shape[0]
        if self.F.shape != (n, n):
            raise ValueError(f"F must be square, got shape {self.F.shape}")
        if self.Q.shape != (n, n):
            raise ValueError(f"Q shape {self.Q.shape} does not match n={n}")
        if self.x0.shape != (n,):
            raise ValueError(f"x0 shape {self.x0.shape} does not match n={n}")
        if self.P0.shape != (n, n):
            raise ValueError(f"P0 shape {self.P0.shape} does not match n={n}")
        T = self.states.shape[0]
        if self.states.shape != (T, n):
            raise ValueError(
                f"states shape {self.states.shape} inconsistent with n={n}"
            )
        if self.covariances.shape != (T, n, n):
            raise ValueError(
                f"covariances shape {self.covariances.shape} inconsistent with "
                f"T={T}, n={n}"
            )
        if self.innovations.shape != (T,):
            raise ValueError(
                f"innovations length {self.innovations.shape[0]} does not match T={T}"
            )
        if self.innovation_variances.shape != (T,):
            raise ValueError(
                f"innovation_variances length {self.innovation_variances.shape[0]} "
                f"does not match T={T}"
            )
        if self.R < 0:
            raise ValueError(f"R must be non-negative, got {self.R}")
        if self.state_names and len(self.state_names) != n:
            raise ValueError(
                f"state_names length {len(self.state_names)} does not match n={n}"
            )

    @property
    def n_states(self) -> int:
        return int(self.F.shape[0])

    @property
    def n_obs(self) -> int:
        return int(self.states.shape[0])

    def latest_state(self) -> np.ndarray:
        """Most recent filtered state estimate `x_{T|T}`."""
        return cast(np.ndarray, self.states[-1].copy())

    def latest_covariance(self) -> np.ndarray:
        """Most recent filtered state covariance `P_{T|T}`."""
        return cast(np.ndarray, self.covariances[-1].copy())


@dataclass(frozen=True)
class HierarchicalFit:
    """Posterior summary of a Gaussian hierarchical regression.

    Stored summaries — not the full chain — because the orchestration layer
    only consumes posterior means + variances. Callers that need full draws
    can re-run the sampler with ``return_draws=True``.

    Attributes
    ----------
    mu_mean / mu_cov:
        Posterior mean and covariance of the population-level mean
        ``mu in R^p``.
    Sigma_mean:
        Posterior mean of the population-level covariance ``Sigma in R^{p,p}``.
    theta_means:
        ``(N, p)`` per-asset posterior means of ``theta_i``.
    theta_covs:
        ``(N, p, p)`` per-asset posterior covariances of ``theta_i``.
    sigma_sq_means:
        Length-``N`` per-asset posterior mean of the residual variance.
    tickers / coef_names:
        Row / column labels.
    n_iter / n_burn:
        Total Gibbs iterations and burn-in count actually used.
    convergence_diagnostics:
        Free-form keys (e.g. mean ``R_hat`` if multi-chain).
    """

    mu_mean: np.ndarray
    mu_cov: np.ndarray
    Sigma_mean: np.ndarray
    theta_means: np.ndarray
    theta_covs: np.ndarray
    sigma_sq_means: np.ndarray
    tickers: tuple[str, ...]
    coef_names: tuple[str, ...]
    n_iter: int
    n_burn: int
    convergence_diagnostics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        N, p = self.theta_means.shape
        if len(self.tickers) != N:
            raise ValueError(
                f"theta_means has {N} rows but {len(self.tickers)} tickers given"
            )
        if p != len(self.coef_names):
            raise ValueError(
                f"theta_means has {p} cols but {len(self.coef_names)} coef names given"
            )
        if self.theta_covs.shape != (N, p, p):
            raise ValueError(
                f"theta_covs shape {self.theta_covs.shape} != ({N}, {p}, {p})"
            )
        if self.mu_mean.shape != (p,):
            raise ValueError(f"mu_mean shape {self.mu_mean.shape} != ({p},)")
        if self.mu_cov.shape != (p, p):
            raise ValueError(f"mu_cov shape {self.mu_cov.shape} != ({p}, {p})")
        if self.Sigma_mean.shape != (p, p):
            raise ValueError(
                f"Sigma_mean shape {self.Sigma_mean.shape} != ({p}, {p})"
            )
        if self.sigma_sq_means.shape != (N,):
            raise ValueError(
                f"sigma_sq_means shape {self.sigma_sq_means.shape} != ({N},)"
            )
        if self.n_iter <= 0 or self.n_burn < 0 or self.n_burn >= self.n_iter:
            raise ValueError(
                f"Require 0 <= n_burn < n_iter; got n_burn={self.n_burn}, "
                f"n_iter={self.n_iter}"
            )

    @property
    def n_assets(self) -> int:
        return int(self.theta_means.shape[0])

    @property
    def n_coefs(self) -> int:
        return int(self.theta_means.shape[1])


@dataclass(frozen=True)
class DynamicFactorFit:
    """Fitted Stock-Watson dynamic factor model via EM.

    Cast as a state-space with state ``f_t``, observation matrix ``Lambda``,
    transition ``Phi``, state-innovation covariance ``Q``, and observation-
    noise diagonal ``Psi``.

    Attributes
    ----------
    Lambda:
        ``(N, r)`` factor loadings.
    Phi:
        ``(r, r)`` factor AR(1) transition.
    Q:
        ``(r, r)`` factor-innovation covariance (typically identity post-
        identification).
    Psi:
        Length-``N`` diagonal idiosyncratic variances.
    factors:
        ``(T, r)`` smoothed factor scores.
    tickers / factor_names:
        Row / column labels.
    log_likelihood:
        Final EM log-likelihood.
    n_iter:
        Number of EM iterations actually used.
    """

    Lambda: np.ndarray
    Phi: np.ndarray
    Q: np.ndarray
    Psi: np.ndarray
    factors: np.ndarray
    tickers: tuple[str, ...]
    factor_names: tuple[str, ...]
    log_likelihood: float
    n_iter: int
    index: pd.Index
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        N, r = self.Lambda.shape
        if len(self.tickers) != N:
            raise ValueError(
                f"Lambda has {N} rows but {len(self.tickers)} tickers given"
            )
        if r != len(self.factor_names):
            raise ValueError(
                f"Lambda has {r} cols but {len(self.factor_names)} factor names given"
            )
        if self.Phi.shape != (r, r):
            raise ValueError(f"Phi shape {self.Phi.shape} != ({r}, {r})")
        if self.Q.shape != (r, r):
            raise ValueError(f"Q shape {self.Q.shape} != ({r}, {r})")
        if self.Psi.shape != (N,):
            raise ValueError(f"Psi shape {self.Psi.shape} != ({N},)")
        if self.factors.shape[1] != r:
            raise ValueError(
                f"factors has {self.factors.shape[1]} cols but r={r}"
            )

    @property
    def n_assets(self) -> int:
        return int(self.Lambda.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.Lambda.shape[1])


@dataclass(frozen=True)
class BayesianKalmanInputs:
    """Everything `calibrate`, `predict`, and `validate` consume.

    Built by `BayesianKalman.fetch_data`. Which fields are populated depends
    on ``mode``:

    - ``"kalman_beta"``      : ``asset_returns`` and ``market_returns`` required.
    - ``"hierarchical"``      : ``panel_returns`` and ``panel_design`` (per-asset
      design matrices) required.
    - ``"dynamic_factor"``    : ``panel_returns`` required.
    """

    mode: ModelMode
    timestamp: datetime
    asset_returns: pd.Series | None = None
    market_returns: pd.Series | None = None
    panel_returns: pd.DataFrame | None = None
    panel_design: dict[str, np.ndarray] | None = None
    panel_targets: dict[str, np.ndarray] | None = None
    coef_names: tuple[str, ...] | None = None
    tickers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_MODES)}, got {self.mode!r}"
            )
        if self.mode == "kalman_beta":
            if self.asset_returns is None or self.market_returns is None:
                raise ValueError(
                    "kalman_beta mode requires asset_returns and market_returns"
                )
        elif self.mode == "hierarchical":
            if self.panel_design is None or self.panel_targets is None:
                raise ValueError(
                    "hierarchical mode requires panel_design and panel_targets"
                )
            if not self.tickers:
                raise ValueError("hierarchical mode requires tickers")
        elif self.mode == "dynamic_factor" and (
            self.panel_returns is None or self.panel_returns.empty
        ):
            raise ValueError("dynamic_factor mode requires a non-empty panel_returns")
