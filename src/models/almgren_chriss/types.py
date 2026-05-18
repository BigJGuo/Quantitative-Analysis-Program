"""Dataclasses specific to the Almgren-Chriss optimal-execution model.

Shared ``Signal`` / ``Forecast`` / ``RiskMetric`` / ``CalibrationResult`` live
in ``src.core.types``; this module holds the model-internal shapes that flow
between ``fetch_data``, ``calibrate``, ``predict``, ``validate``, and the
schedule-building helpers in ``signal.py``.

Units convention
----------------
- Inventory ``X``, ``x_k``, ``n_k``: **shares** (real-valued; the optimal
  schedule does not round to whole lots — round at the application layer).
- Horizon ``T``: trading-day fractions (1.0 = one trading session). Convert
  from clock seconds via ``T_seconds / (6.5 * 3600)`` for US equities.
- Volatility ``sigma``: **annualized**, decimal (0.20 = 20%/yr). The
  schedule math scales ``sigma`` to the horizon as ``sigma * sqrt(T / 252)``
  internally; users pass annualized.
- Temporary impact ``eta``: dollars per share per (share / trading-day).
  Multiplying by ``v_t`` (shares per trading-day) gives the per-share
  slippage in dollars.
- Permanent impact ``gamma``: dollars per share per share — every share
  pushes the mid down by ``gamma`` for all subsequent fills.
- Risk aversion ``lam``: 1 / dollar. ``lam ~ 1e-6`` is a typical sell-side
  default; see the spec for the calibration ladder.

Side convention
---------------
``side="sell"`` liquidates a long inventory ``X > 0``; ``side="buy"``
liquidates a short inventory (``X`` interpreted as the *magnitude* to
buy). The schedule is symmetric — child sizes and cost magnitudes are
identical; only the sign of the cash flow flips. The arithmetic in
``signal.py`` operates on positive ``X`` and the model rewraps for buys
at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

Side = Literal["sell", "buy"]
DiscretizationMode = Literal["continuous", "discrete"]

VALID_SIDES: frozenset[str] = frozenset({"sell", "buy"})
VALID_DISCRETIZATIONS: frozenset[str] = frozenset({"continuous", "discrete"})


@dataclass(frozen=True)
class ImpactParams:
    """Calibrated market-impact and volatility parameters.

    Attributes
    ----------
    sigma:
        Annualized return volatility (decimal, e.g. 0.25 for 25%/yr).
    eta:
        Temporary impact coefficient ($/share per (share/trading-day)).
    gamma:
        Permanent impact coefficient ($/share per share).
    last_price:
        Reference price ``S_0`` used to convert between basis points and
        dollars in cost reporting.
    daily_volume:
        Average daily volume (shares), used as a participation-rate
        denominator and as the basis for ``eta`` calibration.
    """

    sigma: float
    eta: float
    gamma: float
    last_price: float
    daily_volume: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError(f"ImpactParams.sigma must be > 0, got {self.sigma}")
        if not np.isfinite(self.eta) or self.eta <= 0:
            raise ValueError(f"ImpactParams.eta must be > 0, got {self.eta}")
        if not np.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError(f"ImpactParams.gamma must be >= 0, got {self.gamma}")
        if not np.isfinite(self.last_price) or self.last_price <= 0:
            raise ValueError(
                f"ImpactParams.last_price must be > 0, got {self.last_price}"
            )
        if not np.isfinite(self.daily_volume) or self.daily_volume <= 0:
            raise ValueError(
                f"ImpactParams.daily_volume must be > 0, got {self.daily_volume}"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "sigma": self.sigma,
            "eta": self.eta,
            "gamma": self.gamma,
            "last_price": self.last_price,
            "daily_volume": self.daily_volume,
        }


@dataclass(frozen=True)
class LiquidationProblem:
    """A single optimal-execution problem statement.

    Attributes
    ----------
    ticker:
        Underlying instrument.
    X:
        Shares to liquidate. Must be > 0 (the magnitude); ``side`` carries
        the buy/sell sign.
    T:
        Horizon in trading-day fractions (1.0 = one full session). For US
        equities, ``T = T_seconds / (6.5 * 3600)``.
    N:
        Number of equally-spaced slices. ``N >= 1``.
    side:
        ``"sell"`` (liquidate a long) or ``"buy"`` (cover a short).
    lam:
        Mean-variance risk aversion (1/$). Larger -> more front-loaded.
    """

    ticker: str
    X: float
    T: float
    N: int
    side: Side
    lam: float

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("LiquidationProblem.ticker must be non-empty")
        if not np.isfinite(self.X) or self.X <= 0:
            raise ValueError(f"LiquidationProblem.X must be > 0, got {self.X}")
        if not np.isfinite(self.T) or self.T <= 0:
            raise ValueError(f"LiquidationProblem.T must be > 0, got {self.T}")
        if self.N < 1:
            raise ValueError(f"LiquidationProblem.N must be >= 1, got {self.N}")
        if self.side not in VALID_SIDES:
            raise ValueError(
                f"LiquidationProblem.side must be one of {sorted(VALID_SIDES)}, "
                f"got {self.side!r}"
            )
        if not np.isfinite(self.lam) or self.lam <= 0:
            raise ValueError(
                f"LiquidationProblem.lam must be > 0, got {self.lam}"
            )


@dataclass(frozen=True)
class ExecutionSchedule:
    """Optimal slice schedule produced by ``optimal_schedule``.

    Arrays index across slice boundaries / slices:

    - ``times`` shape ``(N+1,)``: knot times ``t_k = k * tau`` for
      ``k = 0..N``.
    - ``inventory`` shape ``(N+1,)``: ``x_k`` — remaining shares at each
      knot. ``x_0 = X``, ``x_N = 0``.
    - ``child_orders`` shape ``(N,)``: ``n_k = x_{k-1} - x_k`` for
      ``k = 1..N``. Same sign as ``X`` (all positive in our convention).
    - ``trading_rate`` shape ``(N,)``: ``v_k = n_k / tau``.

    Cost summaries:

    - ``expected_cost`` in dollars: $\\mathbb{E}[C]$ — implementation
      shortfall vs. arrival price.
    - ``cost_variance`` in dollars-squared.
    - ``expected_cost_bps``: ``E[C] / (X * S_0) * 1e4``.
    - ``kappa``: dimensionless ``sqrt(lam * sigma_T^2 / eta)`` evaluated in
      the model's internal time units (trading-day fractions).
    - ``kappa_T``: ``kappa * T`` — the spec's single dimensionless control
      parameter. ``< 0.3`` ~ TWAP, ``> 3`` ~ front-loaded.
    """

    ticker: str
    side: Side
    discretization: DiscretizationMode
    X: float
    T: float
    N: int
    tau: float
    times: np.ndarray
    inventory: np.ndarray
    child_orders: np.ndarray
    trading_rate: np.ndarray
    expected_cost: float
    cost_variance: float
    expected_cost_bps: float
    kappa: float
    kappa_T: float
    params: ImpactParams
    lam: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in VALID_SIDES:
            raise ValueError(
                f"ExecutionSchedule.side must be one of {sorted(VALID_SIDES)}, "
                f"got {self.side!r}"
            )
        if self.discretization not in VALID_DISCRETIZATIONS:
            raise ValueError(
                f"ExecutionSchedule.discretization must be one of "
                f"{sorted(VALID_DISCRETIZATIONS)}, got {self.discretization!r}"
            )
        if self.times.shape != (self.N + 1,):
            raise ValueError(
                f"ExecutionSchedule.times must have shape ({self.N + 1},), "
                f"got {self.times.shape}"
            )
        if self.inventory.shape != (self.N + 1,):
            raise ValueError(
                f"ExecutionSchedule.inventory must have shape ({self.N + 1},), "
                f"got {self.inventory.shape}"
            )
        if self.child_orders.shape != (self.N,):
            raise ValueError(
                f"ExecutionSchedule.child_orders must have shape ({self.N},), "
                f"got {self.child_orders.shape}"
            )
        if self.trading_rate.shape != (self.N,):
            raise ValueError(
                f"ExecutionSchedule.trading_rate must have shape ({self.N},), "
                f"got {self.trading_rate.shape}"
            )

    @property
    def cost_std(self) -> float:
        """Standard deviation of cost in dollars."""

        return float(np.sqrt(max(self.cost_variance, 0.0)))

    def participation_rates(self) -> np.ndarray:
        """Per-slice trading rate as a fraction of daily volume.

        ``rho_k = n_k / (tau * V_daily)`` — the share of average daily
        volume traded during slice k.
        """

        if self.params.daily_volume <= 0:
            raise ValueError("daily_volume must be > 0 to compute participation")
        return np.asarray(self.child_orders, dtype=float) / (
            self.tau * self.params.daily_volume
        )


@dataclass(frozen=True)
class EfficientFrontierPoint:
    """One point on the (variance, expected-cost) efficient frontier."""

    lam: float
    kappa: float
    kappa_T: float
    expected_cost: float
    cost_variance: float
    expected_cost_bps: float

    @property
    def cost_std(self) -> float:
        return float(np.sqrt(max(self.cost_variance, 0.0)))


@dataclass(frozen=True)
class EfficientFrontier:
    """Sweep of the efficient frontier across a grid of risk aversions."""

    points: tuple[EfficientFrontierPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("EfficientFrontier requires >= 1 point")

    def expected_costs(self) -> np.ndarray:
        return np.asarray([p.expected_cost for p in self.points], dtype=float)

    def variances(self) -> np.ndarray:
        return np.asarray([p.cost_variance for p in self.points], dtype=float)

    def kappa_T_values(self) -> np.ndarray:
        return np.asarray([p.kappa_T for p in self.points], dtype=float)

    def lambdas(self) -> np.ndarray:
        return np.asarray([p.lam for p in self.points], dtype=float)


@dataclass(frozen=True)
class AlmgrenChrissInputs:
    """Bundle assembled by ``AlmgrenChriss.fetch_data``.

    Carries everything ``calibrate``, ``predict``, and ``validate`` need.

    Attributes
    ----------
    ticker:
        Equity / ETF symbol.
    close:
        Daily close prices, indexed by date.
    volume:
        Daily volume series, indexed by date.
    intraday_volume_profile:
        Per-slice mean volume fraction across the trading session; sums
        to 1.0 over the day. ``None`` if intraday data was unavailable
        (the model then falls back to a flat profile).
    last_price:
        Arrival reference price ``S_0``.
    problem:
        The single-asset liquidation problem to solve. The model can be
        re-pointed at a new problem post-calibration without refetching
        market data — see ``AlmgrenChriss.with_problem``.
    timestamp:
        Inputs-assembly time.
    """

    ticker: str
    close: pd.Series
    volume: pd.Series
    intraday_volume_profile: np.ndarray | None
    last_price: float
    problem: LiquidationProblem
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("AlmgrenChrissInputs.ticker must be non-empty")
        if self.close is None or self.close.empty:
            raise ValueError("AlmgrenChrissInputs.close must be non-empty")
        if self.volume is None or self.volume.empty:
            raise ValueError("AlmgrenChrissInputs.volume must be non-empty")
        if not np.isfinite(self.last_price) or self.last_price <= 0:
            raise ValueError(
                f"AlmgrenChrissInputs.last_price must be > 0, got {self.last_price}"
            )
        if self.problem.ticker != self.ticker:
            raise ValueError(
                f"AlmgrenChrissInputs.problem.ticker ({self.problem.ticker!r}) "
                f"must equal ticker ({self.ticker!r})"
            )
        if self.intraday_volume_profile is not None:
            prof = self.intraday_volume_profile
            if prof.ndim != 1 or prof.size == 0:
                raise ValueError(
                    "intraday_volume_profile must be a non-empty 1-D array"
                )
            if np.any(prof <= 0):
                raise ValueError(
                    "intraday_volume_profile entries must all be > 0"
                )


@dataclass(frozen=True)
class MultiAssetProblem:
    """Basket liquidation problem (Algorithm D).

    Attributes
    ----------
    tickers:
        Asset symbols, length M.
    X:
        Per-asset shares to liquidate, shape ``(M,)``. Magnitudes; sides
        come from ``sides``.
    T:
        Horizon in trading-day fractions (shared across the basket).
    N:
        Number of slices (shared).
    sides:
        Per-asset side, length M.
    lam:
        Shared risk aversion (1/$).
    """

    tickers: tuple[str, ...]
    X: np.ndarray
    T: float
    N: int
    sides: tuple[Side, ...]
    lam: float

    def __post_init__(self) -> None:
        M = len(self.tickers)
        if M < 1:
            raise ValueError("MultiAssetProblem.tickers must be non-empty")
        if self.X.shape != (M,):
            raise ValueError(
                f"MultiAssetProblem.X must have shape ({M},), got {self.X.shape}"
            )
        if np.any(self.X <= 0) or not np.all(np.isfinite(self.X)):
            raise ValueError("MultiAssetProblem.X entries must all be > 0")
        if len(self.sides) != M:
            raise ValueError(
                f"MultiAssetProblem.sides must have length {M}, got {len(self.sides)}"
            )
        if any(s not in VALID_SIDES for s in self.sides):
            raise ValueError(
                f"MultiAssetProblem.sides entries must be in {sorted(VALID_SIDES)}, "
                f"got {self.sides!r}"
            )
        if not np.isfinite(self.T) or self.T <= 0:
            raise ValueError(f"MultiAssetProblem.T must be > 0, got {self.T}")
        if self.N < 1:
            raise ValueError(f"MultiAssetProblem.N must be >= 1, got {self.N}")
        if not np.isfinite(self.lam) or self.lam <= 0:
            raise ValueError(f"MultiAssetProblem.lam must be > 0, got {self.lam}")


@dataclass(frozen=True)
class MultiAssetSchedule:
    """Optimal basket schedule from Algorithm D.

    Attributes
    ----------
    tickers:
        Asset symbols, length M.
    times:
        Knot grid, shape ``(N+1,)``.
    inventory:
        Per-asset inventory path, shape ``(N+1, M)``. Row 0 equals
        ``X``; row N equals zero.
    child_orders:
        Per-slice child orders, shape ``(N, M)``. Each entry is
        ``x[k-1, m] - x[k, m]``.
    expected_cost:
        Total expected basket cost in dollars.
    cost_variance:
        Total cost variance in dollars-squared.
    kappas:
        Per-mode decay rates, shape ``(M,)``. Each principal component of
        the eta-Sigma problem liquidates on its own timescale.
    """

    tickers: tuple[str, ...]
    sides: tuple[Side, ...]
    T: float
    N: int
    tau: float
    times: np.ndarray
    inventory: np.ndarray
    child_orders: np.ndarray
    expected_cost: float
    cost_variance: float
    kappas: np.ndarray
    lam: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        M = len(self.tickers)
        if self.inventory.shape != (self.N + 1, M):
            raise ValueError(
                f"MultiAssetSchedule.inventory must have shape "
                f"({self.N + 1}, {M}), got {self.inventory.shape}"
            )
        if self.child_orders.shape != (self.N, M):
            raise ValueError(
                f"MultiAssetSchedule.child_orders must have shape "
                f"({self.N}, {M}), got {self.child_orders.shape}"
            )
        if self.kappas.shape != (M,):
            raise ValueError(
                f"MultiAssetSchedule.kappas must have shape ({M},), "
                f"got {self.kappas.shape}"
            )


__all__ = [
    "DiscretizationMode",
    "VALID_DISCRETIZATIONS",
    "VALID_SIDES",
    "AlmgrenChrissInputs",
    "EfficientFrontier",
    "EfficientFrontierPoint",
    "ExecutionSchedule",
    "ImpactParams",
    "LiquidationProblem",
    "MultiAssetProblem",
    "MultiAssetSchedule",
    "Side",
]
