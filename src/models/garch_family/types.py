"""Dataclasses specific to the GARCH-family volatility model.

Shared `Signal` / `RiskMetric` / `Forecast` / `CalibrationResult` live in
`src.core.types`; this module holds the model-internal shapes that flow
between `fetch_data`, `calibrate`, `predict`, and `validate`.

Returns are stored in **percent** units (`100 * log diff`) — the convention
used by the `arch` package and the spec's yfinance example. This rescales the
likelihood for numerical stability; all variance / volatility numbers reported
out of this module are therefore in percent-return units unless explicitly
annualized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

GARCHSpec = Literal["GARCH", "GJR", "EGARCH"]
InnovationDist = Literal["Gaussian", "Student-t"]
MeanModel = Literal["Constant", "Zero"]

VALID_SPECS: frozenset[str] = frozenset({"GARCH", "GJR", "EGARCH"})
VALID_DISTS: frozenset[str] = frozenset({"Gaussian", "Student-t"})
VALID_MEAN_MODELS: frozenset[str] = frozenset({"Constant", "Zero"})


@dataclass(frozen=True)
class GARCHParams:
    """Estimated parameters of a GARCH-family fit.

    Attributes
    ----------
    omega:
        Constant term in the conditional-variance recursion. Always > 0 for
        GARCH / GJR; unconstrained for EGARCH (where it lives in log-variance
        space).
    alpha:
        ARCH coefficient. For EGARCH, the coefficient on the symmetric term
        `|z| - E|z|`.
    beta:
        GARCH coefficient (persistence). `|beta| < 1` for EGARCH stationarity;
        `alpha + beta < 1` for GARCH(1,1) stationarity.
    gamma:
        Asymmetry / leverage coefficient. Zero for plain GARCH. Positive in
        GJR (negative shocks amplify variance); negative in EGARCH (same
        economic interpretation, opposite sign convention).
    nu:
        Degrees of freedom for the standardized Student-t innovation
        distribution. `None` under Gaussian innovations.
    mu:
        Constant conditional mean of returns (in percent units). Zero when
        the mean model is "Zero".
    """

    omega: float
    alpha: float
    beta: float
    gamma: float = 0.0
    nu: float | None = None
    mu: float = 0.0

    @property
    def persistence(self) -> float:
        """`alpha + beta` for GARCH, `alpha + beta + gamma/2` for GJR, `beta`
        for EGARCH (rate at which log-variance shocks decay).
        """

        return self.alpha + self.beta

    def to_dict(self) -> dict[str, float | None]:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "nu": self.nu,
            "mu": self.mu,
        }


@dataclass(frozen=True)
class GARCHFit:
    """Output of MLE: parameters, in-sample paths, and convergence info.

    Attributes
    ----------
    params:
        Estimated `GARCHParams`.
    spec / distribution / mean_model:
        Echo of the configuration used to fit.
    sigma2_path:
        `(T,)` in-sample conditional-variance path. Aligned to `returns` index
        when the fit owner re-attaches it (the array itself is index-free).
    eps_path:
        `(T,)` mean-centered returns used as innovations.
    std_resid:
        `(T,)` standardized residuals `eps / sigma`. Spec validation 1.
    log_likelihood:
        Maximized log-likelihood value.
    converged:
        Whether the optimizer terminated normally.
    n_iter:
        Number of optimizer iterations consumed.
    n_obs:
        Length of the input return series.
    unconditional_variance:
        Long-run mean of `sigma_t^2` under the fitted parameters. `None`
        for EGARCH (no closed form unless innovation distribution is
        specified; we don't compute it).
    """

    params: GARCHParams
    spec: GARCHSpec
    distribution: InnovationDist
    mean_model: MeanModel
    sigma2_path: np.ndarray
    eps_path: np.ndarray
    std_resid: np.ndarray
    log_likelihood: float
    converged: bool
    n_iter: int
    n_obs: int
    unconditional_variance: float | None = None

    def __post_init__(self) -> None:
        if self.spec not in VALID_SPECS:
            raise ValueError(
                f"GARCHFit.spec must be one of {sorted(VALID_SPECS)}, got {self.spec!r}"
            )
        if self.distribution not in VALID_DISTS:
            raise ValueError(
                f"GARCHFit.distribution must be one of {sorted(VALID_DISTS)}, "
                f"got {self.distribution!r}"
            )
        n = self.sigma2_path.shape[0]
        if self.eps_path.shape[0] != n or self.std_resid.shape[0] != n:
            raise ValueError(
                "sigma2_path, eps_path, and std_resid must share the same length"
            )
        if self.n_obs != n:
            raise ValueError(
                f"n_obs={self.n_obs} does not match path length {n}"
            )


@dataclass(frozen=True)
class GARCHForecastResult:
    """Multi-horizon variance / volatility forecast from a fitted model.

    All values are in **percent-return** units (because returns are stored as
    percent in this module). `annualized_vol` multiplies the daily forecast by
    `sqrt(252)` and divides by 100 to land in the more familiar decimal
    annualized convention.
    """

    horizons: tuple[int, ...]
    variances: np.ndarray  # percent^2 per step
    vols: np.ndarray  # percent per step
    annualized_vols: np.ndarray  # decimal annualized
    timestamp: datetime

    def __post_init__(self) -> None:
        k = len(self.horizons)
        if self.variances.shape != (k,) or self.vols.shape != (k,) or self.annualized_vols.shape != (k,):
            raise ValueError(
                f"GARCHForecastResult arrays must have shape ({k},); got "
                f"variances={self.variances.shape}, vols={self.vols.shape}, "
                f"annualized={self.annualized_vols.shape}"
            )


@dataclass(frozen=True)
class GARCHInputs:
    """Everything `calibrate`, `predict`, and `validate` consume.

    Produced by `GARCHModel.fetch_data`.

    Attributes
    ----------
    ticker:
        Equity / ETF symbol.
    returns_pct:
        Daily log-returns in **percent** units (`100 * log diff`), indexed by
        date. Empty / NaN rows have been dropped upstream.
    timestamp:
        When the inputs were assembled.
    spec / distribution / mean_model:
        Pinned configuration for this fit pass.
    forecast_horizons:
        Variance / volatility forecast horizons in trading days.
    """

    ticker: str
    returns_pct: pd.Series
    timestamp: datetime
    spec: GARCHSpec
    distribution: InnovationDist
    mean_model: MeanModel
    forecast_horizons: tuple[int, ...] = (1, 5, 10, 22)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spec not in VALID_SPECS:
            raise ValueError(
                f"GARCHInputs.spec must be one of {sorted(VALID_SPECS)}, got {self.spec!r}"
            )
        if self.distribution not in VALID_DISTS:
            raise ValueError(
                f"GARCHInputs.distribution must be one of {sorted(VALID_DISTS)}, "
                f"got {self.distribution!r}"
            )
        if self.mean_model not in VALID_MEAN_MODELS:
            raise ValueError(
                f"GARCHInputs.mean_model must be one of {sorted(VALID_MEAN_MODELS)}, "
                f"got {self.mean_model!r}"
            )
        if len(self.returns_pct) < 30:
            raise ValueError(
                f"GARCHInputs.returns_pct must have >= 30 observations, "
                f"got {len(self.returns_pct)}"
            )
        if any(h < 1 for h in self.forecast_horizons):
            raise ValueError(
                f"forecast_horizons must all be >= 1, got {self.forecast_horizons}"
            )
