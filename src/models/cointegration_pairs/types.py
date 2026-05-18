"""Dataclasses specific to the cointegration / pair-trading model.

Shared `Signal` / `CalibrationResult` types live in `src.core.types`; this module
holds the model-internal shapes that flow between `fetch_data`, `calibrate`,
`predict`, and `validate`.

DataFrames (or `pd.Series`) are used only at the data-loading boundary
(`PairInputs.log_price_a`, `PairInputs.log_price_b`). Once a fit is locked in,
all arrays are dense `numpy.ndarray`s to keep the matrix recursions (Kalman,
AR(1)) simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

PairMethod = Literal["static", "kalman"]
VALID_METHODS: frozenset[str] = frozenset({"static", "kalman"})

PairStatus = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class PairInputs:
    """Everything `calibrate`, `predict`, and `validate` consume for one pair.

    Produced by `CointegrationPairs.fetch_data`. Both legs are stored as
    log-close `pd.Series` on a common date index (any non-overlapping dates
    are dropped during alignment).

    Attributes
    ----------
    ticker_a, ticker_b:
        Cash-leg tickers. ``ticker_a`` is the *dependent* variable in the
        Engle-Granger regression `p_A = alpha + beta * p_B + Z`.
    log_price_a, log_price_b:
        Aligned log-close series for the two legs.
    timestamp:
        When the data was fetched.
    metadata:
        Free-form fetch metadata (period, source provider, etc.).
    """

    ticker_a: str
    ticker_b: str
    log_price_a: pd.Series
    log_price_b: pd.Series
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.log_price_a) != len(self.log_price_b):
            raise ValueError(
                f"log_price_a (len={len(self.log_price_a)}) and log_price_b "
                f"(len={len(self.log_price_b)}) must share the same index"
            )
        if not self.log_price_a.index.equals(self.log_price_b.index):
            raise ValueError("log_price_a and log_price_b must be aligned on the same index")
        if self.ticker_a == self.ticker_b:
            raise ValueError(
                f"ticker_a and ticker_b must differ; both are {self.ticker_a!r}"
            )


@dataclass(frozen=True)
class ADFResult:
    """Augmented Dickey-Fuller test result.

    `t_stat` is the t-statistic on the `Z_{t-1}` coefficient in the DF
    regression `delta Z_t = mu + phi Z_{t-1} + sum psi_j delta Z_{t-j} + u_t`.

    `critical_values` is keyed by significance level (e.g. ``{"1%": -3.43,
    "5%": -2.86, "10%": -2.57}``). The test rejects the unit-root null when
    `t_stat` is *less than* (more negative than) the critical value.
    """

    t_stat: float
    n_obs: int
    n_lags: int
    critical_values: dict[str, float]
    regression: Literal["c", "nc"]
    test_type: Literal["adf", "engle_granger"]

    def rejects_unit_root(self, level: str = "5%") -> bool:
        """True iff the unit-root null is rejected at `level`."""
        if level not in self.critical_values:
            raise KeyError(
                f"level {level!r} not in critical_values {sorted(self.critical_values)}"
            )
        return bool(self.t_stat < self.critical_values[level])


@dataclass(frozen=True)
class EngleGrangerFit:
    """OLS regression `p_A = alpha + beta * p_B + Z` plus stationarity tests.

    Attributes
    ----------
    alpha, beta:
        OLS intercept and slope.
    residuals:
        `Z_t = p_A - alpha - beta * p_B`, indexed identically to inputs.
    adf_leg_a, adf_leg_b:
        Standalone ADF on each leg's log-price series. Both should *fail* to
        reject the unit-root null for the pair to be a valid cointegration
        candidate (i.e. each leg is `I(1)`).
    adf_residual:
        Engle-Granger ADF on `Z_t`. Uses the more conservative critical values
        appropriate for an OLS-residual series with one regressor (Phillips-
        Ouliaris / MacKinnon tables). Rejection of the null = cointegration.
    """

    alpha: float
    beta: float
    residuals: pd.Series
    adf_leg_a: ADFResult
    adf_leg_b: ADFResult
    adf_residual: ADFResult


@dataclass(frozen=True)
class OUFit:
    """AR(1) / Ornstein-Uhlenbeck parameters of a spread series.

    Discrete AR(1):     `Z_t = c + phi Z_{t-1} + eps_t`,  `eps_t ~ N(0, sigma_eps^2)`
    Continuous OU:      `dZ = kappa (mu - Z) dt + sigma dW`,  `dt = 1 trading day`

    Mapping:
        phi  = exp(-kappa * dt)
        c    = mu * (1 - phi)
        sigma_eps^2 = sigma^2 * (1 - exp(-2 kappa dt)) / (2 kappa)

    Attributes
    ----------
    phi, c, sigma_eps:
        AR(1) parameters.
    kappa, mu, sigma:
        Continuous-time OU parameters derived from the AR(1) fit.
    half_life:
        `log(2) / kappa` in units of `dt`. For daily bars, in trading days.
    sigma_eq:
        Equilibrium standard deviation `sqrt(sigma^2 / (2 kappa))`. Used as
        the static-beta z-score denominator under the OU long-run-moment
        convention.
    dt:
        Sample spacing (trading days for daily bars).
    """

    phi: float
    c: float
    sigma_eps: float
    kappa: float
    mu: float
    sigma: float
    half_life: float
    sigma_eq: float
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma_eps < 0:
            raise ValueError(f"sigma_eps must be non-negative, got {self.sigma_eps}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.sigma_eq < 0:
            raise ValueError(f"sigma_eq must be non-negative, got {self.sigma_eq}")


@dataclass(frozen=True)
class KalmanFit:
    """Kalman filter dynamic-beta path on a price pair.

    State-space:
        beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, Q)
        p_A_t  = beta_t * p_B_t + eps_t,  eps_t ~ N(0, R)

    Attributes
    ----------
    beta_path:
        `(T,)` filtered beta estimates `beta_hat_{t|t}`.
    variance_path:
        `(T,)` filtered state variance `P_{t|t}`.
    innovations:
        `(T,)` innovation series `y_t = p_A_t - beta_hat_{t|t-1} p_B_t`.
    innovation_variance:
        `(T,)` innovation variance `S_t = p_B_t^2 P_{t|t-1} + R`.
    standardized_innovations:
        `(T,)` `y_t / sqrt(S_t)`. The dynamic-beta analogue of the static
        z-score; the trading signal is built directly off this.
    Q, R, beta0, P0:
        Hyperparameters / initial state used to produce the path.
    """

    beta_path: np.ndarray
    variance_path: np.ndarray
    innovations: np.ndarray
    innovation_variance: np.ndarray
    standardized_innovations: np.ndarray
    Q: float
    R: float
    beta0: float
    P0: float

    def __post_init__(self) -> None:
        n = self.beta_path.shape[0]
        for name, arr in (
            ("variance_path", self.variance_path),
            ("innovations", self.innovations),
            ("innovation_variance", self.innovation_variance),
            ("standardized_innovations", self.standardized_innovations),
        ):
            if arr.shape != (n,):
                raise ValueError(
                    f"KalmanFit.{name} shape {arr.shape} does not match beta_path "
                    f"length {n}"
                )
        if self.Q < 0 or self.R <= 0 or self.P0 < 0:
            raise ValueError(
                f"KalmanFit requires Q >= 0, R > 0, P0 >= 0; got Q={self.Q}, "
                f"R={self.R}, P0={self.P0}"
            )


@dataclass(frozen=True)
class TradingRule:
    """Entry / exit / stop thresholds on the standardized spread."""

    s_in: float = 2.0
    s_out: float = 0.5
    s_stop: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.s_out < self.s_in < self.s_stop:
            raise ValueError(
                f"TradingRule requires 0 <= s_out < s_in < s_stop, got "
                f"s_out={self.s_out}, s_in={self.s_in}, s_stop={self.s_stop}"
            )


@dataclass(frozen=True)
class PairFit:
    """Aggregated calibration output for one pair.

    Always includes the Engle-Granger fit and OU fit. Includes a Kalman fit
    only when `method="kalman"`. `status="rejected"` means the pair failed
    one of the screening filters (non-`I(1)` legs, no residual stationarity,
    half-life out of band); downstream `predict` should emit a flat signal.
    """

    method: PairMethod
    eg_fit: EngleGrangerFit
    ou_fit: OUFit
    kalman_fit: KalmanFit | None
    status: PairStatus
    rejection_reason: str | None
    significance: str
    half_life_band: tuple[float, float]

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"PairFit.method must be one of {sorted(VALID_METHODS)}, got "
                f"{self.method!r}"
            )
        if self.method == "kalman" and self.kalman_fit is None and self.status == "accepted":
            raise ValueError("PairFit.method='kalman' with accepted status requires kalman_fit")
        if self.status == "rejected" and not self.rejection_reason:
            raise ValueError("rejected PairFit must include a rejection_reason")

    @property
    def is_cointegrated(self) -> bool:
        return self.status == "accepted"


__all__ = [
    "ADFResult",
    "EngleGrangerFit",
    "KalmanFit",
    "OUFit",
    "PairFit",
    "PairInputs",
    "PairMethod",
    "PairStatus",
    "TradingRule",
    "VALID_METHODS",
]
