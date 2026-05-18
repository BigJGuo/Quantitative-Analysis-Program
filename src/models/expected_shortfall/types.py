"""Dataclasses specific to the Expected Shortfall (FRTB) risk model.

Shared `RiskMetric` / `CalibrationResult` live in `src.core.types`; this module
holds the model-internal shapes that flow between `fetch_data`, `calibrate`,
`predict`, and `validate`.

Returns are stored as **decimal** simple returns (`pct_change`) — the spec's
convention — and dollar P&L is computed by `returns @ w_dollar` where
`w_dollar` is the per-ticker dollar notional vector. The output `ES_alpha` is
therefore in **dollars of one-day loss** at the given confidence level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

ESMethod = Literal["parametric_normal", "parametric_t", "historical", "monte_carlo"]

VALID_METHODS: frozenset[str] = frozenset(
    {"parametric_normal", "parametric_t", "historical", "monte_carlo"}
)

# FRTB Basel III default: 97.5% Expected Shortfall.
FRTB_ALPHA: float = 0.975


@dataclass(frozen=True)
class ESInputs:
    """Bundle passed from `fetch_data` to `calibrate` / `predict` / `validate`.

    Attributes
    ----------
    tickers:
        Tuple of portfolio symbols (uppercased, sorted to keep the bundle
        deterministically hashable).
    returns:
        `(T, N)` daily simple returns, indexed by date, columns aligned to
        `tickers`. NaN-rows already dropped upstream.
    dollar_positions:
        `(N,)` dollar notional per ticker, aligned to `tickers`. Sign carries
        long (+) / short (-).
    alpha:
        ES confidence level (Basel default `0.975`).
    method:
        One of {parametric_normal, parametric_t, historical, monte_carlo}.
    lookback_N:
        Number of trailing days used (>= 500 recommended at alpha=0.975).
    timestamp:
        When the bundle was assembled.
    portfolio_id:
        Free-form identifier carried into the `RiskMetric` output.
    liquidity_scalars:
        Optional `(N,)` per-ticker FRTB liquidity-horizon multipliers
        `sqrt(h_k / 10)`. Applied to returns before risk aggregation. `None`
        means no scaling (standard one-day ES).
    """

    tickers: tuple[str, ...]
    returns: pd.DataFrame
    dollar_positions: np.ndarray
    alpha: float
    method: ESMethod
    lookback_N: int
    timestamp: datetime
    portfolio_id: str = "PORTFOLIO"
    liquidity_scalars: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"ESInputs.method must be one of {sorted(VALID_METHODS)}, "
                f"got {self.method!r}"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"ESInputs.alpha must lie in (0, 1), got {self.alpha}")
        n = len(self.tickers)
        if self.returns.shape[1] != n:
            raise ValueError(
                f"returns has {self.returns.shape[1]} columns but {n} tickers given"
            )
        if self.dollar_positions.shape != (n,):
            raise ValueError(
                f"dollar_positions shape {self.dollar_positions.shape} "
                f"does not match N={n}"
            )
        if self.lookback_N < 30:
            raise ValueError(
                f"ESInputs.lookback_N must be >= 30, got {self.lookback_N}"
            )
        if len(self.returns) < self.lookback_N:
            raise ValueError(
                f"returns has {len(self.returns)} rows but lookback_N="
                f"{self.lookback_N} demanded"
            )
        if self.liquidity_scalars is not None and self.liquidity_scalars.shape != (n,):
            raise ValueError(
                f"liquidity_scalars shape {self.liquidity_scalars.shape} "
                f"does not match N={n}"
            )


@dataclass(frozen=True)
class ESFit:
    """Estimated risk parameters used to compute ES.

    Always-present fields cover the sample mean / covariance estimated from
    the lookback window. The Student-t fields populate only when the
    parametric-t method has been calibrated.

    Attributes
    ----------
    mu:
        `(N,)` per-ticker mean return.
    Sigma:
        `(N, N)` sample covariance.
    sigma_p:
        Portfolio dollar standard deviation, `sqrt(w^T Sigma w)`.
    method:
        Echo of the calibrating method.
    alpha:
        Confidence level the fit is keyed to.
    n_obs:
        Number of observations used in estimation.
    nu / loc / scale:
        Student-t parameters fitted to the portfolio P&L when
        `method == "parametric_t"`. `None` otherwise.
    """

    mu: np.ndarray
    Sigma: np.ndarray
    sigma_p: float
    method: ESMethod
    alpha: float
    n_obs: int
    nu: float | None = None
    loc: float | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        n = self.mu.shape[0]
        if self.Sigma.shape != (n, n):
            raise ValueError(
                f"Sigma shape {self.Sigma.shape} does not match mu length {n}"
            )
        if self.sigma_p < 0:
            raise ValueError(f"sigma_p must be non-negative, got {self.sigma_p}")
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"ESFit.method must be one of {sorted(VALID_METHODS)}, "
                f"got {self.method!r}"
            )
        if self.method == "parametric_t":
            if self.nu is None or self.loc is None or self.scale is None:
                raise ValueError(
                    "ESFit with method='parametric_t' requires nu, loc, scale"
                )
            if self.nu <= 2.0:
                raise ValueError(f"Student-t nu must be > 2 (for ES), got {self.nu}")
            if self.scale <= 0.0:
                raise ValueError(f"Student-t scale must be > 0, got {self.scale}")


@dataclass(frozen=True)
class ESResult:
    """Full output of one ES computation pass.

    `VaR` and `ES` are dollar losses (positive numbers under typical sign
    conventions). `tail_losses` is the empirical or simulated loss vector
    exceeding `VaR` — used by Z1/Z2 backtests downstream.
    """

    VaR: float
    ES: float
    method: ESMethod
    alpha: float
    tail_losses: np.ndarray
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"ESResult.alpha must lie in (0, 1), got {self.alpha}")
        if self.VaR - 1e-9 > self.ES:
            # ES must be at least as severe as VaR by construction (the
            # tail-mean of values exceeding the quantile is >= the quantile).
            raise ValueError(
                f"ES ({self.ES}) must be >= VaR ({self.VaR}); ES is the "
                "average tail loss above the VaR cutoff."
            )
        if self.tail_losses.ndim != 1:
            raise ValueError(
                f"tail_losses must be 1-D, got shape {self.tail_losses.shape}"
            )
