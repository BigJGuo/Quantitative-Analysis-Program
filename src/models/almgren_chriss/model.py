"""`AlmgrenChriss` — Layer 5 execution model.

Thin orchestration shell. ``fetch_data`` pulls daily close, daily volume,
and a 30-day intraday-volume profile through the shared ``DataProvider``.
``calibrate`` produces ``ImpactParams`` (σ, η, γ). ``predict`` returns a
``RiskMetric`` whose value is the expected implementation shortfall in
**basis points** (with the dollar number and the schedule in
``metadata``). ``validate`` runs the spec's diagnostic block.

The class exposes additional helpers — ``schedule``, ``frontier``,
``vwap_schedule``, ``with_problem`` — for callers who need the raw
schedule object rather than the single-number ``RiskMetric``.

Spec: ``models/layer5_execution/16_almgren_chriss.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.almgren_chriss.calibration import (
    MODEL_NAME,
    EtaRule,
)
from src.models.almgren_chriss.calibration import (
    calibrate as _calibrate_impl,
)
from src.models.almgren_chriss.signal import (
    annual_return_vol_to_dollar_per_day,
    efficient_frontier,
    intraday_volume_profile,
    optimal_schedule,
    schedule_diagnostics,
    stress_schedule,
    vwap_shaped_schedule,
)
from src.models.almgren_chriss.types import (
    VALID_DISCRETIZATIONS,
    VALID_SIDES,
    AlmgrenChrissInputs,
    DiscretizationMode,
    EfficientFrontier,
    ExecutionSchedule,
    ImpactParams,
    LiquidationProblem,
    Side,
)

_DEFAULT_HISTORY_PERIOD: str = "60d"
_DEFAULT_DAILY_INTERVAL: str = "1d"
_DEFAULT_INTRADAY_INTERVAL: str = "5m"
_DEFAULT_INTRADAY_LOOKBACK: int = 30
_DEFAULT_BARS_PER_DAY_5M: int = 78  # 6.5h × 60min / 5min for US equities
_DEFAULT_REFIT_FREQUENCY: Literal["daily"] = "daily"


@register_model
class AlmgrenChriss(BaseModel):
    """Almgren-Chriss optimal-execution model.

    Parameters
    ----------
    ticker:
        Symbol to schedule. One instance per ticker.
    X:
        Shares to liquidate (magnitude). ``side`` carries the sign.
    T:
        Horizon in trading-day fractions (1.0 = one full session).
    N:
        Number of equally-spaced slices.
    side:
        ``"sell"`` (default) or ``"buy"``.
    lam:
        Mean-variance risk aversion (1/$). Default ``1e-6`` (typical
        sell-side desk).
    discretization:
        ``"continuous"`` (default; Algorithm A) or ``"discrete"``
        (Algorithm B).
    history_period:
        yfinance ``period`` for the daily-close pull. Default ``"60d"``
        — needs to cover both the σ window (30 days) and the volume
        window (20 days) with margin.
    eta_rule:
        Which η-calibration rule to use. ``"almgren_2005"`` (default)
        or ``"half_spread"``.
    half_spread_bps:
        Half-spread in bp of price; used only under the ``"half_spread"``
        η rule.
    gamma_halflife_days:
        Permanent-impact half-life used to set γ.
    sigma_window:
        Trailing-bar window for σ estimation.
    volume_window:
        Trailing-bar window for daily-volume estimation.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 5
    refit_frequency: ClassVar[Literal["daily"]] = _DEFAULT_REFIT_FREQUENCY

    def __init__(
        self,
        ticker: str,
        *,
        X: float,
        T: float,
        N: int,
        side: Side = "sell",
        lam: float = 1.0e-6,
        discretization: DiscretizationMode = "continuous",
        history_period: str = _DEFAULT_HISTORY_PERIOD,
        eta_rule: EtaRule = "almgren_2005",
        half_spread_bps: float = 1.0,
        gamma_halflife_days: float = 1.0,
        sigma_window: int = 30,
        volume_window: int = 20,
        intraday_lookback_days: int = _DEFAULT_INTRADAY_LOOKBACK,
        intraday_interval: str = _DEFAULT_INTRADAY_INTERVAL,
        bars_per_day: int = _DEFAULT_BARS_PER_DAY_5M,
        now_func: Any = None,
    ) -> None:
        if not ticker:
            raise ValueError("AlmgrenChriss requires a non-empty ticker")
        if side not in VALID_SIDES:
            raise ValueError(
                f"side must be one of {sorted(VALID_SIDES)}, got {side!r}"
            )
        if discretization not in VALID_DISCRETIZATIONS:
            raise ValueError(
                f"discretization must be one of {sorted(VALID_DISCRETIZATIONS)}, "
                f"got {discretization!r}"
            )
        if X <= 0 or not np.isfinite(X):
            raise ValueError(f"X must be > 0, got {X}")
        if T <= 0 or not np.isfinite(T):
            raise ValueError(f"T must be > 0, got {T}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if lam <= 0 or not np.isfinite(lam):
            raise ValueError(f"lam must be > 0, got {lam}")

        self.ticker = ticker.upper()
        self.X = float(X)
        self.T = float(T)
        self.N = int(N)
        self.side: Side = side
        self.lam = float(lam)
        self.discretization: DiscretizationMode = discretization
        self.history_period = history_period
        self.eta_rule: EtaRule = eta_rule
        self.half_spread_bps = float(half_spread_bps)
        self.gamma_halflife_days = float(gamma_halflife_days)
        self.sigma_window = int(sigma_window)
        self.volume_window = int(volume_window)
        self.intraday_lookback_days = int(intraday_lookback_days)
        self.intraday_interval = str(intraday_interval)
        self.bars_per_day = int(bars_per_day)
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._impact: ImpactParams | None = None

    # ---- BaseModel interface -------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> AlmgrenChrissInputs:
        bars = provider.fetch_prices(
            self.ticker, self.history_period, _DEFAULT_DAILY_INTERVAL
        )
        if bars is None or bars.empty:
            raise RuntimeError(
                f"No daily history returned for {self.ticker!r}; "
                "cannot calibrate Almgren-Chriss."
            )
        if "Close" not in bars.columns or "Volume" not in bars.columns:
            raise RuntimeError(
                f"Daily bars for {self.ticker!r} missing Close/Volume columns: "
                f"{list(bars.columns)!r}"
            )
        close = bars["Close"].astype(float)
        volume = bars["Volume"].astype(float)
        idx: pd.DatetimeIndex
        if "Date" in bars.columns:
            idx = pd.DatetimeIndex(pd.to_datetime(bars["Date"]))
        elif "Datetime" in bars.columns:
            idx = pd.DatetimeIndex(pd.to_datetime(bars["Datetime"]))
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(bars.index))
        close.index = idx
        volume.index = idx
        close = close.sort_index().dropna()
        volume = volume.sort_index().dropna()
        if close.empty:
            raise RuntimeError(
                f"After cleaning, no close observations for {self.ticker!r}"
            )
        last_price = float(close.iloc[-1])

        # Intraday volume profile (best-effort; falls back to flat).
        profile: np.ndarray | None
        try:
            intraday = provider.fetch_intraday(
                self.ticker, self.intraday_interval, self.intraday_lookback_days
            )
        except Exception:  # noqa: BLE001
            intraday = pd.DataFrame()
        if intraday is None or intraday.empty:
            profile = None
        else:
            profile = intraday_volume_profile(
                intraday, bars_per_day=self.bars_per_day
            )

        problem = LiquidationProblem(
            ticker=self.ticker,
            X=self.X,
            T=self.T,
            N=self.N,
            side=self.side,
            lam=self.lam,
        )
        return AlmgrenChrissInputs(
            ticker=self.ticker,
            close=close,
            volume=volume,
            intraday_volume_profile=profile,
            last_price=last_price,
            problem=problem,
            timestamp=self._now_func(),
            metadata={
                "history_period": self.history_period,
                "n_close": len(close),
                "n_volume": len(volume),
                "intraday_available": profile is not None,
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, AlmgrenChrissInputs):
            raise TypeError(
                f"AlmgrenChriss.calibrate expects AlmgrenChrissInputs, "
                f"got {type(data).__name__}"
            )
        result = _calibrate_impl(
            data,
            sigma_window=self.sigma_window,
            volume_window=self.volume_window,
            eta_rule=self.eta_rule,
            half_spread_bps=self.half_spread_bps,
            gamma_halflife_days=self.gamma_halflife_days,
            timestamp=data.timestamp,
        )
        self._impact = cast(ImpactParams, result.parameters["impact"])
        return result

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, AlmgrenChrissInputs):
            raise TypeError(
                f"AlmgrenChriss.predict expects AlmgrenChrissInputs, "
                f"got {type(data).__name__}"
            )
        params = self._ensure_calibrated(data)
        sched = optimal_schedule(
            data.problem,
            params,
            discretization=self.discretization,
            timestamp=data.timestamp,
        )
        sigma_d = annual_return_vol_to_dollar_per_day(
            params.sigma, params.last_price
        )
        return RiskMetric(
            ticker=self.ticker,
            metric_name="implementation_shortfall_bps",
            value=float(sched.expected_cost_bps),
            timestamp=data.timestamp,
            confidence_level=None,
            horizon=self._horizon_string(),
            metadata={
                "side": self.side,
                "expected_cost_dollars": float(sched.expected_cost),
                "cost_variance_dollars2": float(sched.cost_variance),
                "cost_std_dollars": float(sched.cost_std),
                "kappa": float(sched.kappa),
                "kappa_T": float(sched.kappa_T),
                "tau": float(sched.tau),
                "X": float(sched.X),
                "T": float(sched.T),
                "N": int(sched.N),
                "lam": float(self.lam),
                "discretization": self.discretization,
                "sigma_annual": float(params.sigma),
                "sigma_dollar_per_day": float(sigma_d),
                "eta": float(params.eta),
                "gamma": float(params.gamma),
                "last_price": float(params.last_price),
                "daily_volume": float(params.daily_volume),
                "inventory": sched.inventory.tolist(),
                "child_orders": sched.child_orders.tolist(),
                "trading_rate": sched.trading_rate.tolist(),
                "times": sched.times.tolist(),
                "participation_rates": sched.participation_rates().tolist(),
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, AlmgrenChrissInputs):
            raise TypeError(
                f"AlmgrenChriss.validate expects AlmgrenChrissInputs, "
                f"got {type(data).__name__}"
            )
        params = self._ensure_calibrated(data)
        sched = optimal_schedule(
            data.problem,
            params,
            discretization=self.discretization,
            timestamp=data.timestamp,
        )
        diag = schedule_diagnostics(sched)
        # Spec validation 5: stress test (σ × 3).
        stressed = stress_schedule(
            data.problem, params, sigma_multiplier=3.0
        )
        accelerated = stressed.kappa_T > sched.kappa_T
        # Spec validation 4: efficient-frontier convexity.
        frontier = efficient_frontier(
            data.problem.X,
            data.problem.T,
            params,
            lambdas=np.logspace(-9, -3, 7),
        )
        e_costs = frontier.expected_costs()
        variances = frontier.variances()
        # The frontier should be sorted such that E decreases and V increases
        # as λ shrinks. Check monotonicity.
        order = np.argsort(variances)
        e_sorted = e_costs[order]
        v_sorted = variances[order]
        frontier_monotone = bool(
            np.all(np.diff(e_sorted) <= 1e-6 * max(abs(e_sorted).max(), 1.0))
        )
        # Convexity: second differences >= 0 (E vs V on the frontier).
        if len(v_sorted) >= 3:
            d2 = np.diff(e_sorted, n=2)
            # Numerical noise tolerance.
            frontier_convex = bool(np.all(d2 >= -1e-3 * max(abs(e_sorted).max(), 1.0)))
        else:
            frontier_convex = True

        out: dict[str, Any] = {
            "ticker": self.ticker,
            "n_close": len(data.close),
            "n_volume": len(data.volume),
            "intraday_available": data.intraday_volume_profile is not None,
            "sigma_annual": float(params.sigma),
            "eta": float(params.eta),
            "gamma": float(params.gamma),
            "kappa": float(sched.kappa),
            "kappa_T": float(sched.kappa_T),
            "regime": diag["regime"],
            "monotone_inventory": bool(diag["monotone_inventory"]),
            "positive_children": bool(diag["positive_children"]),
            "sum_n_minus_X_relative": float(diag["sum_n_minus_X_relative"]),
            "expected_cost_dollars": float(sched.expected_cost),
            "expected_cost_bps": float(sched.expected_cost_bps),
            "cost_std_dollars": float(sched.cost_std),
            "stressed_kappa_T": float(stressed.kappa_T),
            "stress_accelerated": bool(accelerated),
            "frontier_monotone": frontier_monotone,
            "frontier_convex": frontier_convex,
            "frontier_kappa_T_min": float(np.min(frontier.kappa_T_values())),
            "frontier_kappa_T_max": float(np.max(frontier.kappa_T_values())),
            "numerical_stability_warning": bool(
                diag["numerical_stability_warning"]
            ),
            "max_participation_rate": float(
                np.max(sched.participation_rates())
            ),
        }
        return out

    # ---- Public conveniences -------------------------------------------------

    def schedule(self, data: AlmgrenChrissInputs) -> ExecutionSchedule:
        """Return the full ``ExecutionSchedule`` for the configured problem."""

        params = self._ensure_calibrated(data)
        return optimal_schedule(
            data.problem,
            params,
            discretization=self.discretization,
            timestamp=data.timestamp,
        )

    def vwap_schedule(self, data: AlmgrenChrissInputs) -> ExecutionSchedule:
        """Algorithm C: volume-shaped Almgren-Chriss schedule.

        Falls back to a flat volume profile if no intraday data was
        fetched at ``fetch_data`` time.
        """

        params = self._ensure_calibrated(data)
        if data.intraday_volume_profile is None:
            profile = np.full(self.N, 1.0 / self.N)
        else:
            profile = data.intraday_volume_profile
        return vwap_shaped_schedule(
            data.problem,
            params,
            volume_profile=profile,
            timestamp=data.timestamp,
        )

    def frontier(
        self,
        data: AlmgrenChrissInputs,
        *,
        lambdas: Sequence[float] | None = None,
    ) -> EfficientFrontier:
        """Efficient frontier sweep across the supplied ``lambdas`` (default
        is a 7-point log sweep from 1e-9 to 1e-3, matching the spec's
        decade-range recommendation).
        """

        params = self._ensure_calibrated(data)
        lam_grid: np.ndarray
        if lambdas is None:
            lam_grid = np.logspace(-9, -3, 7)
        else:
            lam_grid = np.asarray(list(lambdas), dtype=float)
        return efficient_frontier(
            data.problem.X, data.problem.T, params, lambdas=lam_grid
        )

    def with_problem(
        self,
        data: AlmgrenChrissInputs,
        *,
        X: float | None = None,
        T: float | None = None,
        N: int | None = None,
        side: Side | None = None,
        lam: float | None = None,
    ) -> AlmgrenChrissInputs:
        """Return a copy of ``data`` with a re-pointed ``problem``.

        Lets callers re-solve for a different parent order without
        refetching market data.
        """

        new_problem = replace(
            data.problem,
            X=data.problem.X if X is None else float(X),
            T=data.problem.T if T is None else float(T),
            N=data.problem.N if N is None else int(N),
            side=data.problem.side if side is None else side,
            lam=data.problem.lam if lam is None else float(lam),
        )
        return replace(data, problem=new_problem)

    @property
    def impact(self) -> ImpactParams:
        """Calibrated ``ImpactParams``. Raises if ``calibrate`` not yet called."""

        if self._impact is None:
            raise RuntimeError(
                "AlmgrenChriss has not been calibrated. Call calibrate(data) "
                "first."
            )
        return self._impact

    # ---- Internals -----------------------------------------------------------

    def _ensure_calibrated(self, data: AlmgrenChrissInputs) -> ImpactParams:
        if self._impact is None:
            self.calibrate(data)
        assert self._impact is not None
        return self._impact

    def _horizon_string(self) -> str:
        """Convert ``self.T`` (trading-day fractions) to a horizon label."""

        if self.T >= 1.0:
            return f"{self.T:.2f}d"
        # Hours assuming a 6.5h trading session.
        hours = self.T * 6.5
        if hours >= 1.0:
            return f"{hours:.2f}h"
        minutes = hours * 60.0
        return f"{minutes:.0f}m"


__all__ = ["AlmgrenChriss"]
