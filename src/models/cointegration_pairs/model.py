"""`CointegrationPairs` — Layer 4 cointegration / pair-trading model.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. Pulls aligned log-price series for two tickers via
`DataProvider`, runs Engle-Granger screening (and optionally Kalman dynamic
beta) in `calibrate`, and emits a directional `Signal` from `predict`.

A *single* pair is the model's unit of work. The orchestration layer is
expected to instantiate one `CointegrationPairs` per candidate pair from the
universe screen; we deliberately do not wrap a basket here so that each pair's
fit, signal, and refit cadence stay independent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Signal, SignalDirection
from src.models.cointegration_pairs.calibration import MODEL_NAME
from src.models.cointegration_pairs.calibration import calibrate as _calibrate_impl
from src.models.cointegration_pairs.signal import (
    align_pair,
    compute_log_prices,
    kalman_dynamic_beta,
    ljung_box_pvalue,
    pair_position_path,
    realized_half_life,
    rolling_eg_residual_tstat,
    rolling_zscore_moments,
    static_zscore,
)
from src.models.cointegration_pairs.types import (
    VALID_METHODS,
    PairFit,
    PairInputs,
    PairMethod,
    TradingRule,
)

_DEFAULT_PERIOD: str = "2y"
_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_ROLLING_WINDOW: int = 60
_DEFAULT_HORIZON: str = "1d"


@register_model
class CointegrationPairs(BaseModel):
    """Engle-Granger / Kalman cointegration pair-trading model.

    Parameters
    ----------
    ticker_a, ticker_b:
        The two legs. ``ticker_a`` is the dependent variable in the
        cointegrating regression.
    method:
        ``"static"`` (default) trades the OLS-beta rolling z-score.
        ``"kalman"`` re-fits the hedge ratio each step and trades the
        standardized innovation.
    period:
        yfinance ``period`` string for the historical pull. Default ``"2y"``
        per the spec's 1-2 year lookback guidance.
    significance:
        ADF / EG significance level used during screening. ``"5%"`` by default.
    half_life_min, half_life_max:
        Acceptable OU half-life band (trading days). Pairs outside the band
        get `status="rejected"` and emit flat signals.
    rolling_window:
        Window for the static-beta rolling z-score moments. Spec default 60
        trading days.
    s_in, s_out, s_stop:
        Trading thresholds applied to the standardized spread.
    Q, R:
        Kalman process / observation noise variances. Used only when
        `method="kalman"`.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "weekly"

    def __init__(
        self,
        ticker_a: str,
        ticker_b: str,
        *,
        method: PairMethod = "static",
        period: str = _DEFAULT_PERIOD,
        significance: str = "5%",
        half_life_min: float = 0.5,
        half_life_max: float = 30.0,
        rolling_window: int = _DEFAULT_ROLLING_WINDOW,
        s_in: float = 2.0,
        s_out: float = 0.5,
        s_stop: float = 4.0,
        adf_lags: int = 1,
        Q: float = 1e-4,
        R: float = 1e-3,
    ) -> None:
        if ticker_a == ticker_b:
            raise ValueError(
                f"ticker_a and ticker_b must differ; both are {ticker_a!r}"
            )
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(VALID_METHODS)}, got {method!r}"
            )
        if rolling_window < 10:
            raise ValueError(
                f"rolling_window must be >= 10, got {rolling_window}"
            )
        self.ticker_a: str = ticker_a
        self.ticker_b: str = ticker_b
        self.method: PairMethod = method
        self.period: str = period
        self.significance: str = significance
        self.half_life_min: float = half_life_min
        self.half_life_max: float = half_life_max
        self.rolling_window: int = rolling_window
        self.rule: TradingRule = TradingRule(s_in=s_in, s_out=s_out, s_stop=s_stop)
        self.adf_lags: int = adf_lags
        self.Q: float = Q
        self.R: float = R
        self._fit: PairFit | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> PairInputs:
        bars_a = provider.fetch_prices(self.ticker_a, self.period, _DEFAULT_INTERVAL)
        bars_b = provider.fetch_prices(self.ticker_b, self.period, _DEFAULT_INTERVAL)
        log_a_full = _bars_to_log_close(bars_a)
        log_b_full = _bars_to_log_close(bars_b)
        if log_a_full.empty or log_b_full.empty:
            raise RuntimeError(
                f"Empty price history for {self.ticker_a!r} or {self.ticker_b!r}"
            )
        log_a, log_b = align_pair(log_a_full, log_b_full)
        if len(log_a) < 60:
            raise RuntimeError(
                f"Aligned overlap too short ({len(log_a)} bars) for cointegration"
                f" on {self.ticker_a!r} / {self.ticker_b!r}"
            )
        return PairInputs(
            ticker_a=self.ticker_a,
            ticker_b=self.ticker_b,
            log_price_a=log_a,
            log_price_b=log_b,
            timestamp=datetime.now(UTC),
            metadata={"period": self.period, "n_obs": len(log_a)},
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, PairInputs):
            raise TypeError(
                f"CointegrationPairs.calibrate expects PairInputs, got "
                f"{type(data).__name__}"
            )
        result = _calibrate_impl(
            data.log_price_a,
            data.log_price_b,
            method=self.method,
            significance=self.significance,
            half_life_min=self.half_life_min,
            half_life_max=self.half_life_max,
            adf_lags=self.adf_lags,
            Q=self.Q,
            R=self.R,
            timestamp=data.timestamp,
        )
        self._fit = cast(PairFit, result.parameters["pair_fit"])
        return result

    def predict(self, data: Any) -> Signal:
        if not isinstance(data, PairInputs):
            raise TypeError(
                f"CointegrationPairs.predict expects PairInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        timestamp = data.timestamp

        if fit.status == "rejected":
            return Signal(
                ticker=self._pair_ticker(),
                direction="flat",
                strength=0.0,
                timestamp=timestamp,
                horizon=_DEFAULT_HORIZON,
                metadata={
                    "method": fit.method,
                    "status": fit.status,
                    "rejection_reason": fit.rejection_reason,
                    "beta": fit.eg_fit.beta,
                    "ticker_a": self.ticker_a,
                    "ticker_b": self.ticker_b,
                },
            )

        s_series = self._standardized_spread(data, fit)
        positions = pair_position_path(s_series.to_numpy(), self.rule)
        last_position = int(positions[-1]) if positions.size > 0 else 0
        last_s = float(s_series.iloc[-1]) if not s_series.empty else float("nan")

        direction = _position_to_direction(last_position)
        strength = _strength_from_score(last_s, rule=self.rule)

        return Signal(
            ticker=self._pair_ticker(),
            direction=direction,
            strength=strength,
            timestamp=timestamp,
            horizon=_DEFAULT_HORIZON,
            metadata={
                "method": fit.method,
                "status": fit.status,
                "alpha": fit.eg_fit.alpha,
                "beta": fit.eg_fit.beta,
                "half_life": fit.ou_fit.half_life,
                "kappa": fit.ou_fit.kappa,
                "current_z_score": last_s,
                "s_in": self.rule.s_in,
                "s_out": self.rule.s_out,
                "s_stop": self.rule.s_stop,
                "ticker_a": self.ticker_a,
                "ticker_b": self.ticker_b,
                "hedge_ratio_a": 1.0,
                "hedge_ratio_b": -fit.eg_fit.beta,
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, PairInputs):
            raise TypeError(
                f"CointegrationPairs.validate expects PairInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        out: dict[str, Any] = {
            "status": fit.status,
            "rejection_reason": fit.rejection_reason,
            "method": fit.method,
            "alpha": fit.eg_fit.alpha,
            "beta": fit.eg_fit.beta,
            "adf_leg_a_tstat": fit.eg_fit.adf_leg_a.t_stat,
            "adf_leg_b_tstat": fit.eg_fit.adf_leg_b.t_stat,
            "eg_residual_tstat": fit.eg_fit.adf_residual.t_stat,
            "eg_residual_5pct_critical": fit.eg_fit.adf_residual.critical_values["5%"],
            "ou_phi": fit.ou_fit.phi,
            "ou_kappa": fit.ou_fit.kappa,
            "ou_mu": fit.ou_fit.mu,
            "ou_sigma": fit.ou_fit.sigma,
            "ou_half_life": fit.ou_fit.half_life,
            "realized_half_life": realized_half_life(fit.eg_fit.residuals),
        }

        # Rolling EG t-statistic (spec validation step 2). Only attempted when
        # we have enough history for at least one full rolling window.
        window = min(252, max(60, len(data.log_price_a) // 2))
        if len(data.log_price_a) >= window + 10:
            rolling = rolling_eg_residual_tstat(
                data.log_price_a, data.log_price_b, window=window, n_lags=self.adf_lags
            )
            if not rolling.empty:
                out["rolling_eg_tstat_min"] = float(rolling.min())
                out["rolling_eg_tstat_max"] = float(rolling.max())
                out["rolling_eg_tstat_last"] = float(rolling.iloc[-1])

        # Kalman-innovation Ljung-Box (spec validation step 4).
        if fit.kalman_fit is not None:
            out["kalman_innovation_ljung_box_p"] = ljung_box_pvalue(
                fit.kalman_fit.standardized_innovations, max_lag=10
            )
            out["kalman_beta_range"] = (
                float(np.min(fit.kalman_fit.beta_path)),
                float(np.max(fit.kalman_fit.beta_path)),
            )
            out["kalman_beta_drift"] = float(
                fit.kalman_fit.beta_path[-1] - fit.kalman_fit.beta_path[0]
            )

        return out

    # ----- Public conveniences ---------------------------------------------

    @property
    def fit(self) -> PairFit:
        return self._require_fit()

    def spread(self, data: PairInputs) -> pd.Series:
        """Residual `Z_t = p_A - alpha - beta * p_B` using the calibrated fit."""

        fit = self._require_fit()
        z = data.log_price_a - fit.eg_fit.alpha - fit.eg_fit.beta * data.log_price_b
        return z.rename("spread")

    def standardized_spread(self, data: PairInputs) -> pd.Series:
        """Standardized spread series used by `predict`."""

        return self._standardized_spread(data, self._require_fit())

    def position_path(self, data: PairInputs) -> pd.Series:
        """Per-bar `{-1, 0, +1}` position series for this pair on `data`."""

        s = self.standardized_spread(data)
        positions = pair_position_path(s.to_numpy(), self.rule)
        return pd.Series(positions, index=s.index, name="position")

    # ----- Internal helpers ------------------------------------------------

    def _require_fit(self) -> PairFit:
        if self._fit is None:
            raise RuntimeError(
                "CointegrationPairs has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    def _pair_ticker(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"

    def _standardized_spread(self, data: PairInputs, fit: PairFit) -> pd.Series:
        """Standardized spread per the active method.

        Static: residual `Z_t` minus rolling 60-day mean over rolling std.
                Falls back to the OU equilibrium moments when the window has
                not yet filled.
        Kalman: standardized innovation `y_t / sqrt(S_t)` recomputed against
                the live observations (re-running the recursion lets the
                signal stay current with the most-recent prices, not just the
                snapshot from `calibrate`).
        """

        if fit.method == "kalman":
            beta0 = fit.kalman_fit.beta0 if fit.kalman_fit is not None else fit.eg_fit.beta
            P0 = fit.kalman_fit.P0 if fit.kalman_fit is not None else max(
                1e-6, fit.ou_fit.sigma_eps ** 2
            )
            Q = fit.kalman_fit.Q if fit.kalman_fit is not None else self.Q
            R = fit.kalman_fit.R if fit.kalman_fit is not None else self.R
            live = kalman_dynamic_beta(
                data.log_price_a.to_numpy(),
                data.log_price_b.to_numpy(),
                Q=Q,
                R=R,
                beta0=beta0,
                P0=P0,
            )
            return pd.Series(
                live.standardized_innovations,
                index=data.log_price_a.index,
                name="z_kalman",
            )

        z = data.log_price_a - fit.eg_fit.alpha - fit.eg_fit.beta * data.log_price_b
        rolling_mu, rolling_sigma = rolling_zscore_moments(z, window=self.rolling_window)
        ou_mu = fit.ou_fit.mu
        ou_sigma = fit.ou_fit.sigma_eq if fit.ou_fit.sigma_eq > 0 else fit.ou_fit.sigma_eps
        if not np.isfinite(ou_sigma) or ou_sigma <= 0:
            ou_sigma = float(z.std(ddof=1)) or 1.0
        if not np.isfinite(ou_mu):
            ou_mu = float(z.mean())
        mu_used = rolling_mu.fillna(ou_mu)
        sigma_used = rolling_sigma.fillna(ou_sigma).replace(0.0, ou_sigma)
        s = (z - mu_used) / sigma_used
        # Final-bar fallback in case any remaining NaN slipped through.
        s_arr = static_zscore(z.to_numpy(), ou_mu, ou_sigma if ou_sigma > 0 else 1.0)
        s_filled = s.fillna(pd.Series(s_arr, index=z.index))
        return s_filled.rename("z_static")


# ---- module-level helpers ---------------------------------------------------


def _position_to_direction(position: int) -> SignalDirection:
    if position > 0:
        return "long"
    if position < 0:
        return "short"
    return "flat"


def _strength_from_score(s: float, *, rule: TradingRule) -> float:
    """Map a standardized spread value to a `Signal.strength` in `[0, 1]`.

    Linear ramp from `s_in` to `s_stop`. Below `s_in` -> 0 (no edge). Above
    `s_stop` -> 0 (stop-out, the trade is closed). The peak sits at the
    midpoint between entry and stop.
    """

    if not np.isfinite(s):
        return 0.0
    abs_s = abs(s)
    if abs_s <= rule.s_in or abs_s >= rule.s_stop:
        return 0.0
    span = rule.s_stop - rule.s_in
    if span <= 0:
        return 0.0
    midpoint = rule.s_in + span / 2.0
    distance = abs(abs_s - midpoint)
    half_span = span / 2.0
    strength = 1.0 - (distance / half_span)
    return float(max(0.0, min(1.0, strength)))


def _bars_to_log_close(bars: pd.DataFrame) -> pd.Series:
    """Pull a log-close series out of a yfinance OHLCV frame.

    Mirrors the helper used by the factor model — handles both
    `reset_index()`-shaped frames (from `YFinanceProvider`) and frames that
    still have their `DatetimeIndex`.
    """

    if bars is None or bars.empty or "Close" not in bars.columns:
        return pd.Series(dtype=float)
    close = bars["Close"].astype(float)
    if "Date" in bars.columns:
        new_index = pd.DatetimeIndex(pd.to_datetime(bars["Date"]))
    elif "Datetime" in bars.columns:
        new_index = pd.DatetimeIndex(pd.to_datetime(bars["Datetime"]))
    else:
        new_index = pd.DatetimeIndex(pd.to_datetime(bars.index))
    close = close.set_axis(new_index).sort_index()
    return compute_log_prices(close)


__all__ = [
    "CointegrationPairs",
    "compute_log_prices",  # re-export for downstream convenience
]
