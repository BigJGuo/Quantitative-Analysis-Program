"""`HARRVModel` — Layer 4 signals / ML — Corsi 2009 HAR realized-volatility.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. The class fetches intraday bars for one ticker, builds the
daily realized-variance series, fits the HAR-RV regression, and produces a
one-day-ahead (or `h`-day-ahead direct) `Forecast` of realized volatility.

When intraday history is unavailable (yfinance caps 5-minute history at
~60 days), the class falls back to a Garman-Klass RV proxy from the daily
OHLC feed so the regression has enough history to fit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Forecast
from src.models.har_rv.calibration import MODEL_NAME
from src.models.har_rv.calibration import calibrate as _calibrate_impl
from src.models.har_rv.signal import (
    DEFAULT_M_WINDOW,
    DEFAULT_W_WINDOW,
    aggregate_har_components,
    arch_lm_pvalue,
    build_har_design_matrix,
    compute_daily_rv,
    compute_garman_klass_rv,
    forecast_one_step,
    in_sample_forecasts,
    latest_har_state,
    ljung_box_pvalue,
    mincer_zarnowitz,
    qlike_loss,
)
from src.models.har_rv.types import (
    VALID_SPECS,
    HARFit,
    HARRVInputs,
    HARSpec,
    RVComponents,
    RVSource,
)

_DEFAULT_INTRADAY_INTERVAL: str = "5m"
_DEFAULT_INTRADAY_LOOKBACK_DAYS: int = 60  # yfinance cap for sub-hourly bars
_DEFAULT_DAILY_PERIOD: str = "5y"
_DEFAULT_DAILY_INTERVAL: str = "1d"
_TRADING_DAYS_PER_YEAR: int = 252


@register_model
class HARRVModel(BaseModel):
    """Heterogeneous Autoregressive realized-volatility model for one ticker.

    Parameters
    ----------
    ticker:
        Underlying instrument symbol (e.g. ``"SPY"``).
    intraday_interval:
        yfinance interval for the high-frequency RV step. Default ``"5m"``.
    intraday_lookback_days:
        How many calendar days of intraday bars to pull. Capped at 60 for any
        sub-hourly interval (yfinance constraint).
    spec:
        ``"log"`` (Corsi-preferred) or ``"level"``.
    horizon:
        Forecast horizon `h` in trading days. `h=1` is one-day ahead; higher
        values use the direct multi-step regression on the `h`-period mean.
    include_overnight:
        Add `(log P_open - log P_prev_close)^2` to each day's RV. Default
        ``True`` (Andersen-Bollerslev convention (a) in the spec).
    use_daily_fallback:
        When ``True``, splice a Garman-Klass RV proxy from the daily feed
        onto the front of the intraday series so the regression has enough
        history to fit. Default ``True``.
    daily_period:
        yfinance `period` string used for the daily-OHLC fallback.
    w_window / m_window:
        Aggregation windows for weekly and monthly RV. Defaults are 5 and 22.
    hac_lag:
        Newey-West truncation lag. ``None`` uses Newey's rule of thumb.
    annualize:
        Report the forecast as `sqrt(252 * RV_forecast)` (annualized vol)
        rather than the raw daily realized variance.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "monthly"

    def __init__(
        self,
        ticker: str,
        *,
        intraday_interval: str = _DEFAULT_INTRADAY_INTERVAL,
        intraday_lookback_days: int = _DEFAULT_INTRADAY_LOOKBACK_DAYS,
        spec: HARSpec = "log",
        horizon: int = 1,
        include_overnight: bool = True,
        use_daily_fallback: bool = True,
        daily_period: str = _DEFAULT_DAILY_PERIOD,
        w_window: int = DEFAULT_W_WINDOW,
        m_window: int = DEFAULT_M_WINDOW,
        hac_lag: int | None = None,
        annualize: bool = True,
    ) -> None:
        if not ticker:
            raise ValueError("HARRVModel: ticker must be a non-empty string")
        if spec not in VALID_SPECS:
            raise ValueError(
                f"HARRVModel: spec must be one of {sorted(VALID_SPECS)}, got {spec!r}"
            )
        if horizon < 1:
            raise ValueError(f"HARRVModel: horizon must be >= 1, got {horizon}")
        if intraday_lookback_days < 1:
            raise ValueError(
                f"HARRVModel: intraday_lookback_days must be positive, "
                f"got {intraday_lookback_days}"
            )
        if w_window < 1 or m_window <= w_window:
            raise ValueError(
                f"HARRVModel: require 1 <= w_window < m_window, "
                f"got w_window={w_window}, m_window={m_window}"
            )

        self.ticker = ticker
        self.intraday_interval = intraday_interval
        self.intraday_lookback_days = intraday_lookback_days
        self.spec: HARSpec = spec
        self.horizon = horizon
        self.include_overnight = include_overnight
        self.use_daily_fallback = use_daily_fallback
        self.daily_period = daily_period
        self.w_window = w_window
        self.m_window = m_window
        self.hac_lag = hac_lag
        self.annualize = annualize
        self._fit: HARFit | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> HARRVInputs:
        intraday_rv: pd.Series = pd.Series(dtype="float64")
        components: RVComponents | None = None
        intraday_metadata: dict[str, Any] = {}
        try:
            bars = provider.fetch_intraday(
                self.ticker,
                self.intraday_interval,
                self.intraday_lookback_days,
            )
            if bars is not None and not bars.empty:
                components = compute_daily_rv(
                    bars,
                    include_overnight=self.include_overnight,
                )
                intraday_rv = components.rv_d.copy()
                intraday_metadata = {
                    "intraday_bars": float(len(bars)),
                    "intraday_days": float(len(intraday_rv)),
                    "n_intraday_bars_per_day": components.n_intraday_bars,
                }
        except Exception as exc:  # noqa: BLE001
            # Provider raised — we'll fall back to daily-OHLC GK if enabled.
            intraday_metadata = {"intraday_error": repr(exc)}

        daily_rv = pd.Series(dtype="float64")
        if self.use_daily_fallback:
            try:
                daily_bars = provider.fetch_prices(
                    self.ticker, self.daily_period, _DEFAULT_DAILY_INTERVAL
                )
                if daily_bars is not None and not daily_bars.empty:
                    daily_rv = compute_garman_klass_rv(daily_bars)
            except Exception as exc:  # noqa: BLE001
                intraday_metadata["daily_fallback_error"] = repr(exc)

        rv_d = _splice_rv(daily_rv, intraday_rv)
        if rv_d.empty:
            raise RuntimeError(
                f"HARRVModel.fetch_data: no usable RV history for {self.ticker!r}"
            )
        if components is None or components.rv_d.empty:
            # Use the spliced series as `rv_d`; downstream consumers don't
            # need bipower / semivariances from the fallback path.
            source: RVSource = "garman_klass" if not intraday_rv.size else "intraday"
            components = RVComponents(rv_d=rv_d, source=source)
        else:
            # Replace the intraday-only series with the spliced one so the
            # regression has enough history; keep BV / semivariance series
            # over the intraday window (they are NaN/0 outside that range).
            components = RVComponents(
                rv_d=rv_d,
                bipower=components.bipower,
                rv_minus=components.rv_minus,
                rv_plus=components.rv_plus,
                source=components.source,
                n_intraday_bars=components.n_intraday_bars,
            )

        metadata: dict[str, Any] = {
            "n_obs_rv": float(len(rv_d)),
            "ticker": self.ticker,
            "intraday_interval": self.intraday_interval,
            **intraday_metadata,
        }
        return HARRVInputs(
            ticker=self.ticker,
            rv_components=components,
            spec=self.spec,
            horizon=self.horizon,
            timestamp=datetime.now(UTC),
            annualize=self.annualize,
            trading_days_per_year=_TRADING_DAYS_PER_YEAR,
            metadata=metadata,
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        inputs = self._require_inputs(data)
        result = _calibrate_impl(
            rv_d=inputs.rv_components.rv_d,
            spec=inputs.spec,
            horizon=inputs.horizon,
            w_window=self.w_window,
            m_window=self.m_window,
            hac_lag=self.hac_lag,
            timestamp=inputs.timestamp,
            metadata={
                "ticker": inputs.ticker,
                "rv_source": inputs.rv_components.source,
            },
        )
        self._fit = cast(HARFit, result.parameters["har_fit"])
        return result

    def predict(self, data: Any) -> Forecast:
        inputs = self._require_inputs(data)
        fit = self._require_fit()
        rv_d_t, rv_w_t, rv_m_t, latest_date = latest_har_state(
            inputs.rv_components.rv_d,
            w_window=self.w_window,
            m_window=self.m_window,
        )
        rv_forecast = forecast_one_step(fit, rv_d_t, rv_w_t, rv_m_t)
        rv_forecast = float(max(rv_forecast, 0.0))
        vol_daily = float(np.sqrt(rv_forecast))
        if inputs.annualize:
            value = float(np.sqrt(inputs.trading_days_per_year * rv_forecast))
            horizon_label = f"{inputs.horizon}d_annualized"
        else:
            value = vol_daily
            horizon_label = f"{inputs.horizon}d"

        metadata = {
            "rv_forecast_daily": rv_forecast,
            "rvol_forecast_daily": vol_daily,
            "rv_d_t": rv_d_t,
            "rv_w_t": rv_w_t,
            "rv_m_t": rv_m_t,
            "as_of_date": str(latest_date.date()),
            "spec": fit.spec,
            "horizon": fit.horizon,
            "ticker": inputs.ticker,
            "annualized": inputs.annualize,
            "trading_days_per_year": inputs.trading_days_per_year,
            "rv_source": inputs.rv_components.source,
            "persistence_sum": float(
                fit.coefficients[1] + fit.coefficients[2] + fit.coefficients[3]
            ),
        }
        return Forecast(
            ticker=inputs.ticker,
            horizon=horizon_label,
            value=value,
            timestamp=inputs.timestamp,
            metadata=metadata,
        )

    def validate(self, data: Any) -> dict[str, Any]:
        inputs = self._require_inputs(data)
        fit = self._require_fit()
        y, X, _idx = build_har_design_matrix(
            inputs.rv_components.rv_d,
            spec=inputs.spec,
            horizon=inputs.horizon,
            w_window=self.w_window,
            m_window=self.m_window,
        )
        spec_forecasts = in_sample_forecasts(fit, X)

        # Convert in-sample forecasts back to RV space for MZ / QLIKE.
        if fit.spec == "log":
            rv_realized = np.exp(y)
            rv_forecast = np.exp(spec_forecasts + 0.5 * fit.residual_variance)
        else:
            rv_realized = y
            rv_forecast = spec_forecasts

        mz = mincer_zarnowitz(rv_realized, rv_forecast)
        qlike = qlike_loss(rv_realized, rv_forecast)
        lb_resid = ljung_box_pvalue(fit.residuals, max_lag=10)
        arch_lm = arch_lm_pvalue(fit.residuals, lags=5)

        components = aggregate_har_components(
            inputs.rv_components.rv_d,
            w_window=self.w_window,
            m_window=self.m_window,
        )
        rv_d_clean = components["rv_d"].dropna()
        out: dict[str, Any] = {
            "n_obs": int(fit.n_obs),
            "r_squared": float(fit.r_squared),
            "residual_variance": float(fit.residual_variance),
            "coefficient_persistence": float(
                fit.coefficients[1] + fit.coefficients[2] + fit.coefficients[3]
            ),
            "coefficient_standard_errors": fit.standard_errors.tolist(),
            "coefficient_t_statistics": fit.t_statistics().tolist(),
            "hac_lag": int(fit.hac_lag),
            "mincer_zarnowitz_a": mz["a"],
            "mincer_zarnowitz_b": mz["b"],
            "mincer_zarnowitz_r_squared": mz["r_squared"],
            "mincer_zarnowitz_t_a": mz.get("t_a", float("nan")),
            "mincer_zarnowitz_t_b_minus_1": mz.get("t_b_minus_1", float("nan")),
            "qlike_loss_in_sample": qlike,
            "ljung_box_residuals_pvalue": lb_resid,
            "arch_lm_residuals_pvalue": arch_lm,
            "rv_min": float(rv_d_clean.min()) if not rv_d_clean.empty else float("nan"),
            "rv_max": float(rv_d_clean.max()) if not rv_d_clean.empty else float("nan"),
            "rv_mean": float(rv_d_clean.mean()) if not rv_d_clean.empty else float("nan"),
        }
        return out

    # ----- Public conveniences -----------------------------------------------

    @property
    def fit(self) -> HARFit:
        return self._require_fit()

    # ----- Internal helpers --------------------------------------------------

    def _require_fit(self) -> HARFit:
        if self._fit is None:
            raise RuntimeError(
                "HARRVModel has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    @staticmethod
    def _require_inputs(data: Any) -> HARRVInputs:
        if not isinstance(data, HARRVInputs):
            raise TypeError(
                f"HARRVModel expects HARRVInputs, got {type(data).__name__}"
            )
        return data


# ---- module-level helpers ---------------------------------------------------


def _splice_rv(daily_rv: pd.Series, intraday_rv: pd.Series) -> pd.Series:
    """Splice an intraday RV series onto the front of a daily-OHLC GK series.

    Intraday RV (when present) takes precedence on overlapping dates because
    it is asymptotically closer to integrated variance than the GK proxy.
    Both inputs are tolerated as empty. The output index is tz-naive and
    normalized to date boundaries — yfinance returns daily bars tz-naive
    and intraday bars tz-aware, so we strip tz to avoid mismatch errors on
    set operations.
    """

    if intraday_rv.empty and daily_rv.empty:
        return pd.Series(dtype="float64", name="rv_d")
    if intraday_rv.empty:
        return pd.Series(
            daily_rv.to_numpy(dtype=float),
            index=_naive_date_index(daily_rv.index),
            name="rv_d",
        ).sort_index()
    if daily_rv.empty:
        return pd.Series(
            intraday_rv.to_numpy(dtype=float),
            index=_naive_date_index(intraday_rv.index),
            name="rv_d",
        ).sort_index()

    daily_idx = _naive_date_index(daily_rv.index)
    intraday_idx = _naive_date_index(intraday_rv.index)
    daily_aligned = pd.Series(daily_rv.to_numpy(dtype=float), index=daily_idx)
    intraday_aligned = pd.Series(
        intraday_rv.to_numpy(dtype=float), index=intraday_idx
    )
    # Drop overlapping dates from the daily series; intraday wins.
    daily_only = daily_aligned.loc[~daily_aligned.index.isin(intraday_aligned.index)]
    combined = pd.concat([daily_only, intraday_aligned]).sort_index()
    combined.name = "rv_d"
    return combined


def _naive_date_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Tz-naive, normalized DatetimeIndex (date-only, no time-of-day)."""

    out = pd.DatetimeIndex(pd.to_datetime(idx))
    if out.tz is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out.normalize()


__all__ = ["HARRVModel"]
