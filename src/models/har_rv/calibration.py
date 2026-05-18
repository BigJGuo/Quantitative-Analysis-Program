"""Calibration entry point for the HAR-RV model.

`calibrate()` is the public, data-only path: hand it a daily realized-
variance series and it returns a `CalibrationResult` whose
`parameters["har_fit"]` is the `HARFit`. Data fetching is the model class's
job — keeping I/O out of this module makes the regression unit-testable with
synthetic inputs.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.core.types import CalibrationResult
from src.models.har_rv.signal import (
    DEFAULT_M_WINDOW,
    DEFAULT_MIN_RV,
    DEFAULT_W_WINDOW,
    fit_har,
)
from src.models.har_rv.types import VALID_SPECS, HARFit, HARSpec

MODEL_NAME: str = "har_rv"


def calibrate(
    *,
    rv_d: pd.Series,
    spec: HARSpec = "log",
    horizon: int = 1,
    w_window: int = DEFAULT_W_WINDOW,
    m_window: int = DEFAULT_M_WINDOW,
    hac_lag: int | None = None,
    min_rv: float = DEFAULT_MIN_RV,
    timestamp: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> CalibrationResult:
    """Fit HAR-RV on a daily realized-variance series.

    Parameters
    ----------
    rv_d:
        Daily realized variance, indexed by trading date.
    spec:
        ``"log"`` (default) or ``"level"``.
    horizon:
        Forecast horizon `h`. `h=1` is one-day-ahead; for `h>1` the dependent
        variable is the mean of `RV_{t+1..t+h}` (Corsi "direct" multi-step).
    w_window / m_window:
        Aggregation windows for the weekly and monthly RV regressors.
    hac_lag:
        Newey-West Bartlett truncation lag. ``None`` (default) uses Newey's
        rule of thumb `floor(4 * (T/100)^{2/9})`.
    min_rv:
        Lower bound applied before taking logs (log-spec only). Guards
        against zero-RV days that would crash the regression.
    """

    if spec not in VALID_SPECS:
        raise ValueError(
            f"calibrate: spec must be one of {sorted(VALID_SPECS)}, got {spec!r}"
        )
    if rv_d is None or rv_d.empty:
        raise ValueError("calibrate: rv_d must be a non-empty pandas Series")
    if horizon < 1:
        raise ValueError(f"calibrate: horizon must be >= 1, got {horizon}")

    ts = timestamp or datetime.now(UTC)
    fit = fit_har(
        rv_d,
        spec=spec,
        horizon=horizon,
        w_window=w_window,
        m_window=m_window,
        hac_lag=hac_lag,
        min_rv=min_rv,
    )

    return _wrap(fit, ts, metadata or {})


def _wrap(
    fit: HARFit,
    ts: datetime,
    extra_metadata: dict[str, object],
) -> CalibrationResult:
    coefs = fit.coefficients
    ses = fit.standard_errors
    parameters: dict[str, object] = {
        "har_fit": fit,
        "spec": fit.spec,
        "horizon": fit.horizon,
        "c": float(coefs[0]),
        "beta_d": float(coefs[1]),
        "beta_w": float(coefs[2]),
        "beta_m": float(coefs[3]),
        "se_c": float(ses[0]),
        "se_beta_d": float(ses[1]),
        "se_beta_w": float(ses[2]),
        "se_beta_m": float(ses[3]),
    }
    fit_metrics = {
        "r_squared": float(fit.r_squared),
        "residual_variance": float(fit.residual_variance),
        "n_obs": float(fit.n_obs),
        "hac_lag": float(fit.hac_lag),
        "persistence_sum": float(coefs[1] + coefs[2] + coefs[3]),
    }
    metadata = {"horizon": fit.horizon, "spec": fit.spec, **extra_metadata}
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata=metadata,
    )


__all__ = ["MODEL_NAME", "calibrate"]
