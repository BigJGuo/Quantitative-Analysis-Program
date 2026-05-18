"""Calibration of Kyle's lambda on daily / intraday OHLCV bars.

Exposes one public entry point, ``calibrate``, plus three helpers
(``calibrate_daily``, ``calibrate_intraday``, ``estimate_intraday_pooled``).

``calibrate`` is the spec-compliant dispatcher returning a
``CalibrationResult`` whose ``parameters["fit"]`` carries a
``KyleLambdaFit``. The dispatcher decides ``daily`` vs ``intraday`` based
on the ``frequency`` kwarg.

The optimization itself is closed-form OLS (Section "Empirical Kyle
lambda" in the spec); HC1 is hand-rolled in ``signal.hc1_se_no_intercept``
so no scipy / statsmodels dependency is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.kyle_lambda_sqrt_impact.signal import (
    compute_log_returns,
    compute_signed_volume,
    hc1_se_no_intercept,
    ols_slope_no_intercept,
    r_squared_no_intercept,
    wald_ci_95,
)
from src.models.kyle_lambda_sqrt_impact.types import KyleFrequency, KyleLambdaFit

MODEL_NAME: str = "kyle_lambda_sqrt_impact"


def _align_return_and_signed_volume(
    bars: pd.DataFrame, *, dollar: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Build paired (signed_volume, log_return) vectors with first row dropped.

    Used by both the daily and intraday calibrators. ``Volume == 0`` rows
    contribute zero signed volume and remain in the panel; the OLS handles
    them naturally (no information, no leverage).
    """

    sgn_vol = compute_signed_volume(bars, dollar=dollar)
    ret = compute_log_returns(bars)
    df = pd.concat([sgn_vol, ret], axis=1).dropna()
    x = df["signed_volume"].to_numpy(dtype=float)
    y = df["log_return"].to_numpy(dtype=float)
    return x, y


def _fit_one(
    x: np.ndarray, y: np.ndarray, frequency: KyleFrequency
) -> KyleLambdaFit:
    """Run the OLS + HC1 pipeline on aligned (x, y) and pack the result."""

    if x.size < 2:
        raise ValueError(
            f"Kyle lambda OLS needs >= 2 observations, got {x.size}"
        )
    beta_hat = ols_slope_no_intercept(x, y)
    se = hc1_se_no_intercept(x, y, beta_hat)
    ci_low, ci_high = wald_ci_95(beta_hat, se)
    r2 = r_squared_no_intercept(x, y, beta_hat)
    return KyleLambdaFit(
        lambda_hat=beta_hat,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        r_squared=r2,
        n_obs=int(x.size),
        frequency=frequency,
        mean_signed_volume_abs=float(np.mean(np.abs(x))),
        mean_volume=float(np.mean(np.abs(x))),
    )


def calibrate_daily(
    daily_bars: pd.DataFrame,
    *,
    lookback: int | None = None,
    dollar: bool = False,
) -> KyleLambdaFit:
    """Daily Kyle lambda (spec Algorithm A).

    ``lookback`` (in trading days) tail-trims the bar frame; passing
    ``None`` uses the entire frame. Spec defaults: 60 or 120 days.
    """

    if lookback is not None:
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2 or None, got {lookback}")
        daily_bars = daily_bars.tail(lookback + 1)
    x, y = _align_return_and_signed_volume(daily_bars, dollar=dollar)
    fit = _fit_one(x, y, frequency="daily")
    # Overwrite mean_volume to refer to unsigned volume (more informative
    # than mean abs signed volume, though numerically equal for
    # non-zero-volume bars).
    vol = daily_bars["Volume"].astype(float).dropna()
    if dollar:
        close = daily_bars["Close"].astype(float).dropna()
        common_idx = vol.index.intersection(close.index)
        mean_vol = float((vol.loc[common_idx] * close.loc[common_idx]).mean())
    else:
        mean_vol = float(vol.mean())
    return KyleLambdaFit(
        lambda_hat=fit.lambda_hat,
        se=fit.se,
        ci_low=fit.ci_low,
        ci_high=fit.ci_high,
        r_squared=fit.r_squared,
        n_obs=fit.n_obs,
        frequency="daily",
        mean_signed_volume_abs=fit.mean_signed_volume_abs,
        mean_volume=mean_vol,
    )


def _bars_with_date(intraday_bars: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``date`` column so we can groupby trading day."""

    df = intraday_bars.copy()
    if "Datetime" in df.columns:
        ts = pd.to_datetime(df["Datetime"]).reset_index(drop=True)
    elif "Date" in df.columns:
        ts = pd.to_datetime(df["Date"]).reset_index(drop=True)
    else:
        ts = pd.Series(pd.to_datetime(df.index))
    df = df.reset_index(drop=True)
    df["date"] = ts.dt.date
    return df


def calibrate_intraday(
    intraday_bars: pd.DataFrame,
    *,
    dollar: bool = False,
) -> tuple[KyleLambdaFit, dict[pd.Timestamp, float]]:
    """Per-day intraday Kyle lambda plus a pooled estimate (spec Algorithm B).

    Returns ``(pooled_fit, per_day_lambdas)``. The per-day map preserves
    the median diagnostic the spec recommends.
    """

    df = _bars_with_date(intraday_bars)
    per_day_lambdas: dict[pd.Timestamp, float] = {}
    for date, day_df in df.groupby("date", sort=True):
        if len(day_df) < 5:
            # Need at least a handful of bars per day for a stable
            # within-day slope. Skip otherwise.
            continue
        x_d, y_d = _align_return_and_signed_volume(day_df, dollar=dollar)
        if x_d.size < 2 or float(np.dot(x_d, x_d)) <= 0.0:
            continue
        try:
            lam_d = ols_slope_no_intercept(x_d, y_d)
        except ValueError:
            continue
        per_day_lambdas[pd.Timestamp(str(date))] = lam_d
    if not per_day_lambdas:
        raise ValueError("No trading days had enough bars to fit an intraday lambda")
    x, y = _align_return_and_signed_volume(df, dollar=dollar)
    pooled = _fit_one(x, y, frequency="intraday")
    vol = df["Volume"].astype(float).dropna()
    if dollar:
        close = df["Close"].astype(float).dropna()
        common_idx = vol.index.intersection(close.index)
        mean_vol = float((vol.loc[common_idx] * close.loc[common_idx]).mean())
    else:
        mean_vol = float(vol.mean())
    pooled = KyleLambdaFit(
        lambda_hat=pooled.lambda_hat,
        se=pooled.se,
        ci_low=pooled.ci_low,
        ci_high=pooled.ci_high,
        r_squared=pooled.r_squared,
        n_obs=pooled.n_obs,
        frequency="intraday",
        mean_signed_volume_abs=pooled.mean_signed_volume_abs,
        mean_volume=mean_vol,
    )
    return pooled, per_day_lambdas


def estimate_intraday_pooled(intraday_bars: pd.DataFrame) -> KyleLambdaFit:
    """Convenience wrapper returning only the pooled intraday fit."""

    pooled, _ = calibrate_intraday(intraday_bars)
    return pooled


def calibrate(
    daily_bars: pd.DataFrame,
    *,
    frequency: KyleFrequency = "daily",
    lookback: int | None = 120,
    intraday_bars: pd.DataFrame | None = None,
    dollar: bool = False,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Spec-compliant entry point. Returns a ``CalibrationResult``.

    ``parameters["fit"]`` holds the ``KyleLambdaFit``. ``parameters`` also
    surfaces the bare scalar fields (``lambda_hat``, ``se``, ``ci_low``,
    ``ci_high``) for callers that want them without unpacking the dataclass.
    ``fit_metrics`` holds the diagnostic scalars (``n_obs``, ``r_squared``,
    ``mean_volume``, ``lambda_normalized``).
    """

    ts = timestamp or datetime.now(UTC)
    if frequency == "daily":
        fit = calibrate_daily(daily_bars, lookback=lookback, dollar=dollar)
        per_day: dict[pd.Timestamp, float] | None = None
    elif frequency == "intraday":
        if intraday_bars is None:
            raise ValueError(
                "calibrate(frequency='intraday') requires intraday_bars"
            )
        fit, per_day = calibrate_intraday(intraday_bars, dollar=dollar)
    else:
        raise ValueError(f"Unknown frequency {frequency!r}")

    metadata: dict[str, object] = {"frequency": fit.frequency, "dollar": dollar}
    if per_day is not None:
        # Median of per-day lambdas — the spec validation step.
        per_day_values = np.array(list(per_day.values()), dtype=float)
        metadata["per_day_median_lambda"] = float(np.median(per_day_values))
        metadata["per_day_count"] = int(per_day_values.size)
        metadata["per_day_positive_share"] = float(np.mean(per_day_values > 0))

    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={
            "fit": fit,
            "lambda_hat": fit.lambda_hat,
            "se": fit.se,
            "ci_low": fit.ci_low,
            "ci_high": fit.ci_high,
        },
        fit_metrics={
            "n_obs": float(fit.n_obs),
            "r_squared": float(fit.r_squared),
            "mean_volume": float(fit.mean_volume),
            "lambda_normalized": float(fit.lambda_normalized),
        },
        timestamp=ts,
        metadata=metadata,
    )
