"""Dataclasses specific to the Value-at-Risk (VaR) model.

Shared `Signal` / `RiskMetric` / `Forecast` / `CalibrationResult` live in
`src.core.types`; this module holds the model-internal shapes that flow
between `fetch_data`, `calibrate`, `predict`, and `validate`.

Units convention: portfolio positions are denominated in **dollars** (a
signed scalar per ticker), and asset returns are **decimal simple returns**
(`P_t / P_{t-1} - 1`). Portfolio P&L is then the plain inner product
`w_dollar . r`, and the resulting VaR is itself in dollars. This matches
the spec's `pct_change()` recipe and avoids the percent-units convention
that the GARCH module inherits from the `arch` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

VaRMethod = Literal["parametric", "historical", "monte_carlo"]
TrafficLight = Literal["green", "yellow", "red"]

VALID_METHODS: frozenset[str] = frozenset({"parametric", "historical", "monte_carlo"})
VALID_TRAFFIC_LIGHTS: frozenset[str] = frozenset({"green", "yellow", "red"})

PORTFOLIO_TICKER: str = "PORTFOLIO"


@dataclass(frozen=True)
class VaRInputs:
    """Everything `calibrate`, `predict`, and `validate` consume.

    Produced by `VaRModel.fetch_data`.

    Attributes
    ----------
    positions:
        Mapping ``ticker -> dollar notional``. Signed (positive = long,
        negative = short). Order is preserved and must match
        ``returns.columns``.
    returns:
        Decimal simple returns indexed by date. Columns in the same order
        as ``positions``.
    timestamp:
        When the inputs were assembled.
    method:
        ``"parametric"``, ``"historical"``, or ``"monte_carlo"``.
    alpha:
        Confidence level (e.g. ``0.99`` -> 1% breach rate).
    horizon_days:
        Reporting horizon in trading days. The 1-day VaR is always reported
        in the primary metric; this controls the sqrt-scaled extra horizon
        (commonly 10 days for the Basel regulatory cut).
    mc_samples:
        Number of Monte Carlo draws (ignored for parametric / historical).
    mc_seed:
        Optional seed for the MC RNG.
    metadata:
        Free-form passthrough.
    """

    positions: dict[str, float]
    returns: pd.DataFrame
    timestamp: datetime
    method: VaRMethod
    alpha: float
    horizon_days: int
    mc_samples: int = 50_000
    mc_seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"VaRInputs.method must be one of {sorted(VALID_METHODS)}, "
                f"got {self.method!r}"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                f"VaRInputs.alpha must lie in (0, 1), got {self.alpha}"
            )
        if self.horizon_days < 1:
            raise ValueError(
                f"VaRInputs.horizon_days must be >= 1, got {self.horizon_days}"
            )
        if not self.positions:
            raise ValueError("VaRInputs.positions must be non-empty.")
        if self.returns.empty:
            raise ValueError("VaRInputs.returns must be non-empty.")
        if list(self.returns.columns) != list(self.positions.keys()):
            raise ValueError(
                f"VaRInputs.returns.columns ({list(self.returns.columns)}) "
                f"must match positions.keys() ({list(self.positions.keys())})"
            )
        if len(self.returns) < 30:
            raise ValueError(
                f"VaRInputs.returns must have >= 30 observations, "
                f"got {len(self.returns)}"
            )
        if self.mc_samples < 1:
            raise ValueError(
                f"VaRInputs.mc_samples must be >= 1, got {self.mc_samples}"
            )


@dataclass(frozen=True)
class VaRFit:
    """Output of `calibrate`: method-specific fit + computed VaR figures.

    For all three methods we always record the resulting 1-day VaR and the
    sqrt-scaled horizon VaR in dollars, plus the underlying P&L vector
    (empirical for HS, simulated for MC, analytic projection for parametric).

    Attributes
    ----------
    method:
        ``"parametric"`` / ``"historical"`` / ``"monte_carlo"``.
    alpha:
        Confidence level at which VaR was computed.
    horizon_days:
        Reporting horizon (1-day VaR is also stored).
    n_obs:
        Number of return rows used in the fit.
    n_assets:
        Number of risk factors (= portfolio names).
    portfolio_value:
        Signed sum of ``positions.values()`` (V_0 in the spec).
    gross_exposure:
        Sum of absolute position values.
    var_1d_dollar:
        Positive scalar: the 1-day VaR in dollars.
    var_horizon_dollar:
        Sqrt-h scaled VaR at ``horizon_days``.
    pnl_vector:
        For HS: shape ``(n_obs,)`` empirical P&L using *current* positions.
        For MC: shape ``(mc_samples,)`` simulated P&L vector.
        For parametric: shape ``(n_obs,)`` empirical P&L (the parametric
        VaR is computed analytically; the vector is recorded for plotting /
        cross-checks).
    mu_p_dollar:
        Mean portfolio P&L (parametric / MC); ``None`` for HS.
    sigma_p_dollar:
        Standard deviation of portfolio P&L (parametric / MC); ``None``
        for HS.
    sigma_matrix:
        Asset return covariance (n x n) for parametric / MC; ``None``
        for HS.
    mc_samples:
        Number of MC draws (``None`` for parametric / historical).
    metadata:
        Free-form.
    """

    method: VaRMethod
    alpha: float
    horizon_days: int
    n_obs: int
    n_assets: int
    portfolio_value: float
    gross_exposure: float
    var_1d_dollar: float
    var_horizon_dollar: float
    pnl_vector: np.ndarray
    mu_p_dollar: float | None = None
    sigma_p_dollar: float | None = None
    sigma_matrix: np.ndarray | None = None
    mc_samples: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"VaRFit.method must be one of {sorted(VALID_METHODS)}, "
                f"got {self.method!r}"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"VaRFit.alpha must lie in (0, 1), got {self.alpha}")
        if self.horizon_days < 1:
            raise ValueError(
                f"VaRFit.horizon_days must be >= 1, got {self.horizon_days}"
            )
        if self.n_obs < 1:
            raise ValueError(f"VaRFit.n_obs must be >= 1, got {self.n_obs}")
        if self.n_assets < 1:
            raise ValueError(f"VaRFit.n_assets must be >= 1, got {self.n_assets}")
        if self.var_1d_dollar < 0:
            raise ValueError(
                f"VaRFit.var_1d_dollar must be >= 0, got {self.var_1d_dollar}"
            )
        if self.var_horizon_dollar < 0:
            raise ValueError(
                f"VaRFit.var_horizon_dollar must be >= 0, got "
                f"{self.var_horizon_dollar}"
            )
        if self.pnl_vector.ndim != 1:
            raise ValueError(
                f"VaRFit.pnl_vector must be 1-D, got shape {self.pnl_vector.shape}"
            )
        if self.sigma_matrix is not None and self.sigma_matrix.shape != (
            self.n_assets,
            self.n_assets,
        ):
            raise ValueError(
                f"VaRFit.sigma_matrix shape {self.sigma_matrix.shape} "
                f"does not match n_assets ({self.n_assets})"
            )


@dataclass(frozen=True)
class VaRBacktestResult:
    """Diagnostic output of `validate`.

    A rolling-window VaR backtest plus the Kupiec POF and Christoffersen
    independence likelihood-ratio tests, ending in a Basel traffic-light
    bucket. Always computed on the historical-simulation projection of the
    fitted positions onto the past return panel (the parametric and MC
    variants project through their respective generators, but the *test*
    itself only needs realized losses vs. one-step VaR cutoffs).

    Attributes
    ----------
    method:
        Method used to compute the VaR cutoffs in the backtest.
    alpha:
        Confidence level.
    n:
        Number of out-of-rolling-window days backtested.
    breaches:
        Count of days where loss exceeded the VaR cutoff.
    breach_rate:
        ``breaches / n``. Should approximate ``1 - alpha`` under correct
        specification.
    expected_breaches:
        ``n * (1 - alpha)``.
    kupiec_lr:
        Kupiec proportion-of-failures likelihood ratio statistic.
    kupiec_p:
        Asymptotic p-value (chi-squared with 1 dof).
    christoffersen_lr:
        Christoffersen independence statistic.
    christoffersen_p:
        Asymptotic p-value (chi-squared with 1 dof).
    traffic_light:
        Basel traffic-light bucket — ``"green"``, ``"yellow"``, ``"red"``.
        Calibrated to ``T = 250`` at ``alpha = 0.99``.
    """

    method: VaRMethod
    alpha: float
    n: int
    breaches: int
    breach_rate: float
    expected_breaches: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    traffic_light: TrafficLight

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"VaRBacktestResult.method must be one of "
                f"{sorted(VALID_METHODS)}, got {self.method!r}"
            )
        if self.traffic_light not in VALID_TRAFFIC_LIGHTS:
            raise ValueError(
                f"VaRBacktestResult.traffic_light must be one of "
                f"{sorted(VALID_TRAFFIC_LIGHTS)}, got {self.traffic_light!r}"
            )
        if self.n < 0:
            raise ValueError(f"VaRBacktestResult.n must be >= 0, got {self.n}")
        if self.breaches < 0:
            raise ValueError(
                f"VaRBacktestResult.breaches must be >= 0, got {self.breaches}"
            )
