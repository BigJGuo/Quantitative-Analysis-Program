"""`VaRModel` — Layer 6 portfolio Value-at-Risk model.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. Pulls a daily-close panel for every ticker in the
portfolio through the shared `DataProvider`, aligns the rows, runs the
chosen VaR engine (parametric / historical / Monte Carlo) and emits a
`RiskMetric` carrying the 1-day VaR in dollars together with the
sqrt-scaled horizon VaR and the empirical P&L vector in metadata.

Spec: `models/layer6_risk/17_var.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.var.calibration import MODEL_NAME
from src.models.var.calibration import calibrate as _calibrate_impl
from src.models.var.signal import (
    align_positions,
    basel_traffic_light,
    christoffersen_independence_test,
    horizon_scale,
    kupiec_pof_test,
    rolling_var_backtest,
    simple_returns,
)
from src.models.var.types import (
    PORTFOLIO_TICKER,
    VALID_METHODS,
    VaRBacktestResult,
    VaRFit,
    VaRInputs,
    VaRMethod,
)

_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_HISTORY_PERIOD: str = "3y"
_DEFAULT_REFIT_FREQUENCY: Literal["daily"] = "daily"
_DEFAULT_ALPHA: float = 0.99
_DEFAULT_LOOKBACK: int = 500
_DEFAULT_HORIZON: int = 10
_DEFAULT_MC_SAMPLES: int = 50_000
_DEFAULT_BACKTEST_WINDOW: int = 250


@register_model
class VaRModel(BaseModel):
    """Portfolio Value-at-Risk model — parametric / historical / Monte Carlo.

    Parameters
    ----------
    positions:
        Mapping ``ticker -> signed dollar notional``. Positive values are
        long, negative values are short.
    method:
        ``"parametric"`` (variance-covariance, Gaussian),
        ``"historical"`` (default — empirical quantile of trailing P&L),
        or ``"monte_carlo"`` (Gaussian MC with sample mean / covariance).
    alpha:
        VaR confidence level (default ``0.99`` — the Basel regulatory cut).
    lookback_days:
        Length of the trailing return window used for the fit. Spec
        recommends ``250`` as the regulatory floor and ``500`` or ``750``
        when feasible — default ``500``.
    horizon_days:
        Reporting horizon for the sqrt-scaled VaR (default ``10`` —
        Basel).
    history_period:
        ``yfinance`` ``period`` argument; should comfortably exceed
        ``lookback_days + backtest_window`` so the validate step has data.
        Default ``"3y"`` (roughly 750 trading days, enough for the
        defaults).
    mc_samples:
        Number of Monte Carlo draws (only used when ``method ==
        "monte_carlo"``).
    mc_seed:
        Seed for the MC RNG.
    backtest_window:
        Rolling-window length for the `validate` Kupiec / Christoffersen
        backtest. Default ``250`` (Basel) — the spec's traffic-light
        thresholds are calibrated to this length at ``alpha = 0.99``.
    include_mean:
        Whether to retain the sample drift term in parametric / MC. The
        spec recommends dropping it at daily frequency (``False``) but the
        default keeps it (``True``) for fidelity to the closed-form
        expression.
    now_func:
        Override the timestamping clock (for deterministic tests).
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 6
    refit_frequency: ClassVar[Literal["daily"]] = _DEFAULT_REFIT_FREQUENCY

    def __init__(
        self,
        positions: Mapping[str, float],
        *,
        method: VaRMethod = "historical",
        alpha: float = _DEFAULT_ALPHA,
        lookback_days: int = _DEFAULT_LOOKBACK,
        horizon_days: int = _DEFAULT_HORIZON,
        history_period: str = _DEFAULT_HISTORY_PERIOD,
        mc_samples: int = _DEFAULT_MC_SAMPLES,
        mc_seed: int | None = None,
        backtest_window: int = _DEFAULT_BACKTEST_WINDOW,
        include_mean: bool = True,
        now_func: Callable[[], datetime] | None = None,
    ) -> None:
        if not positions:
            raise ValueError("VaRModel.positions must be a non-empty mapping.")
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(VALID_METHODS)}, got {method!r}"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        if lookback_days < 30:
            raise ValueError(
                f"lookback_days must be >= 30, got {lookback_days}"
            )
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")
        if backtest_window < 30:
            raise ValueError(
                f"backtest_window must be >= 30, got {backtest_window}"
            )

        self.positions: dict[str, float] = {
            str(k).upper(): float(v) for k, v in positions.items()
        }
        if any(not math_finite(v) for v in self.positions.values()):
            raise ValueError(
                f"VaRModel.positions contains non-finite values: {self.positions}"
            )
        self.method: VaRMethod = method
        self.alpha: float = float(alpha)
        self.lookback_days: int = int(lookback_days)
        self.horizon_days: int = int(horizon_days)
        self.history_period: str = history_period
        self.mc_samples: int = int(mc_samples)
        self.mc_seed: int | None = mc_seed
        self.backtest_window: int = int(backtest_window)
        self.include_mean: bool = bool(include_mean)
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._fit: VaRFit | None = None
        self._returns: pd.DataFrame | None = None

    # ---- BaseModel hooks -----------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> VaRInputs:
        """Pull a daily-close panel for every portfolio ticker.

        Returns are decimal simple returns aligned across tickers via
        inner join, then tailed to ``lookback_days + backtest_window``
        rows so the backtest has data without re-fetching.
        """

        tickers = list(self.positions.keys())
        close_by_ticker: dict[str, pd.Series] = {}
        for tkr in tickers:
            bars = provider.fetch_prices(tkr, self.history_period, _DEFAULT_INTERVAL)
            series = _bars_to_close(bars)
            if series is None or series.empty:
                continue
            close_by_ticker[tkr] = series.astype(float)
        if not close_by_ticker:
            raise RuntimeError(
                f"No usable price history for positions {tickers!r}; every "
                "ticker returned empty bars."
            )
        missing = [t for t in tickers if t not in close_by_ticker]
        if missing:
            raise RuntimeError(
                f"Price history unavailable for tickers {missing!r}; "
                f"cannot construct a portfolio VaR."
            )

        close_panel = pd.DataFrame(
            {t: close_by_ticker[t] for t in tickers}
        ).sort_index()
        rets = simple_returns(close_panel)
        # Tail to lookback + backtest_window so validate has overlap.
        keep = self.lookback_days + self.backtest_window
        if len(rets) < self.lookback_days:
            raise RuntimeError(
                f"Aligned return panel has {len(rets)} rows, need >= "
                f"{self.lookback_days} for VaR fit (history_period="
                f"{self.history_period!r})."
            )
        rets = rets.tail(keep).copy()
        self._returns = rets

        return VaRInputs(
            positions=dict(self.positions),
            returns=rets,
            timestamp=self._now_func(),
            method=self.method,
            alpha=self.alpha,
            horizon_days=self.horizon_days,
            mc_samples=self.mc_samples,
            mc_seed=self.mc_seed,
            metadata={
                "period": self.history_period,
                "lookback_days": self.lookback_days,
                "backtest_window": self.backtest_window,
                "n_rows": len(rets),
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, VaRInputs):
            raise TypeError(
                f"VaRModel.calibrate expects VaRInputs, got {type(data).__name__}"
            )
        # Calibration uses only the *trailing lookback_days* — the extra
        # rows in `data.returns` are reserved for the backtest in
        # validate().
        rets_for_fit = data.returns.tail(self.lookback_days)
        result = _calibrate_impl(
            returns=rets_for_fit,
            positions=data.positions,
            method=data.method,
            alpha=data.alpha,
            horizon_days=data.horizon_days,
            mc_samples=data.mc_samples,
            mc_seed=data.mc_seed,
            timestamp=data.timestamp,
            include_mean=self.include_mean,
        )
        self._fit = cast(VaRFit, result.parameters["fit"])
        return result

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, VaRInputs):
            raise TypeError(
                f"VaRModel.predict expects VaRInputs, got {type(data).__name__}"
            )
        fit = self._ensure_fit(data)
        metadata: dict[str, Any] = {
            "method": fit.method,
            "alpha": fit.alpha,
            "horizon_days": fit.horizon_days,
            "var_1d_dollar": fit.var_1d_dollar,
            "var_horizon_dollar": fit.var_horizon_dollar,
            "portfolio_value": fit.portfolio_value,
            "gross_exposure": fit.gross_exposure,
            "n_obs": fit.n_obs,
            "n_assets": fit.n_assets,
            "pnl_vector": fit.pnl_vector.tolist(),
            "pnl_mean": float(fit.pnl_vector.mean()),
            "pnl_std": (
                float(fit.pnl_vector.std(ddof=1)) if fit.pnl_vector.size > 1 else 0.0
            ),
            "positions": dict(self.positions),
        }
        if fit.gross_exposure > 0:
            metadata["var_1d_pct_gross"] = fit.var_1d_dollar / fit.gross_exposure
            metadata["var_horizon_pct_gross"] = (
                fit.var_horizon_dollar / fit.gross_exposure
            )
        if fit.mu_p_dollar is not None:
            metadata["mu_p_dollar"] = fit.mu_p_dollar
        if fit.sigma_p_dollar is not None:
            metadata["sigma_p_dollar"] = fit.sigma_p_dollar
        if fit.mc_samples is not None:
            metadata["mc_samples"] = fit.mc_samples

        return RiskMetric(
            ticker=PORTFOLIO_TICKER,
            metric_name="value_at_risk",
            value=fit.var_1d_dollar,
            timestamp=data.timestamp,
            confidence_level=fit.alpha,
            horizon="1d",
            metadata=metadata,
        )

    def validate(self, data: Any) -> dict[str, Any]:
        """Spec validation — rolling-window Kupiec POF, Christoffersen
        independence, and Basel traffic-light bucket.

        Uses the same ``method`` and ``alpha`` configured on the model. The
        rolling window is ``backtest_window``; the backtest runs out of
        sample relative to that window, so a panel of
        ``lookback_days + backtest_window`` rows yields exactly
        ``lookback_days`` backtest observations.
        """

        if not isinstance(data, VaRInputs):
            raise TypeError(
                f"VaRModel.validate expects VaRInputs, got {type(data).__name__}"
            )
        fit = self._ensure_fit(data)
        rets_array = np.asarray(data.returns.values, dtype=float)
        w, _ = align_positions(data.positions, data.returns.columns)

        if rets_array.shape[0] <= self.backtest_window:
            # Not enough data for a meaningful backtest — return a degenerate
            # diagnostic block so the orchestration layer doesn't crash.
            return {
                "method": fit.method,
                "alpha": fit.alpha,
                "var_1d_dollar": fit.var_1d_dollar,
                "var_horizon_dollar": fit.var_horizon_dollar,
                "backtest_n": 0,
                "backtest_breaches": 0,
                "backtest_breach_rate": float("nan"),
                "backtest_expected_breaches": 0.0,
                "kupiec_lr": float("nan"),
                "kupiec_p": float("nan"),
                "christoffersen_lr": float("nan"),
                "christoffersen_p": float("nan"),
                "traffic_light": "green",
                "warning": "insufficient_history_for_backtest",
            }

        var_path, loss_path, breach = rolling_var_backtest(
            rets_array,
            w,
            self.alpha,
            method=self.method,
            window=self.backtest_window,
            mc_samples=self.mc_samples,
            mc_seed=self.mc_seed,
        )
        breaches = int(breach.sum())
        n = int(breach.size)
        breach_rate = breaches / n if n > 0 else float("nan")
        expected = n * (1.0 - self.alpha)
        kupiec_lr, kupiec_p = kupiec_pof_test(breaches, n, self.alpha)
        chr_lr, chr_p = christoffersen_independence_test(breach)
        light = basel_traffic_light(breaches)

        backtest = VaRBacktestResult(
            method=fit.method,
            alpha=fit.alpha,
            n=n,
            breaches=breaches,
            breach_rate=breach_rate,
            expected_breaches=expected,
            kupiec_lr=kupiec_lr,
            kupiec_p=kupiec_p,
            christoffersen_lr=chr_lr,
            christoffersen_p=chr_p,
            traffic_light=light,
        )

        return {
            "method": backtest.method,
            "alpha": backtest.alpha,
            "var_1d_dollar": fit.var_1d_dollar,
            "var_horizon_dollar": fit.var_horizon_dollar,
            "backtest_n": backtest.n,
            "backtest_breaches": backtest.breaches,
            "backtest_breach_rate": backtest.breach_rate,
            "backtest_expected_breaches": backtest.expected_breaches,
            "kupiec_lr": backtest.kupiec_lr,
            "kupiec_p": backtest.kupiec_p,
            "christoffersen_lr": backtest.christoffersen_lr,
            "christoffersen_p": backtest.christoffersen_p,
            "traffic_light": backtest.traffic_light,
            "var_path": var_path.tolist(),
            "loss_path": loss_path.tolist(),
        }

    # ---- public conveniences -------------------------------------------------

    def horizon_var(self, horizon_days: int) -> float:
        """Return sqrt-h scaled VaR at ``horizon_days``. Requires `calibrate`
        to have been called.
        """

        if self._fit is None:
            raise RuntimeError(
                "VaRModel.horizon_var requires a prior call to calibrate()."
            )
        return horizon_scale(self._fit.var_1d_dollar, horizon_days)

    @property
    def fit(self) -> VaRFit:
        if self._fit is None:
            raise RuntimeError(
                "VaRModel has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    # ---- internals -----------------------------------------------------------

    def _ensure_fit(self, data: VaRInputs) -> VaRFit:
        if self._fit is None:
            self.calibrate(data)
        assert self._fit is not None
        return self._fit


def math_finite(x: float) -> bool:
    """Local `math.isfinite` wrapper that accepts arbitrary numeric input."""

    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _bars_to_close(bars: pd.DataFrame | None) -> pd.Series | None:
    """Pull a single ``Close`` series out of a bars DataFrame.

    Handles the two layouts produced by `YFinanceProvider.fetch_prices`:
    a frame with a ``Date`` column post-`reset_index`, or one indexed by
    `DatetimeIndex` directly. Returns ``None`` when the panel is empty
    or has no ``Close`` column.
    """

    if bars is None or bars.empty or "Close" not in bars.columns:
        return None
    close = bars["Close"]
    if "Date" in bars.columns:
        close.index = pd.to_datetime(bars["Date"])
    elif "Datetime" in bars.columns:
        close.index = pd.to_datetime(bars["Datetime"])
    else:
        close.index = pd.to_datetime(bars.index)
    return close.sort_index().dropna()


__all__ = ["VaRModel"]
