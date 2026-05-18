"""Estimate the risk parameters (`mu`, `Sigma`, optionally `nu, loc, scale`)
that feed the four ES computation paths.

Calibration here is intentionally light: there is no joint optimization
across all four methods. `calibrate` returns a single `ESFit` and the
`CalibrationResult` wraps it with diagnostic numbers (effective tail
observations, fitted Student-t parameters when applicable, condition number
of the covariance). The heavy ES math lives in `signal.py`; this module is
the glue between the raw lookback window and the inputs that math expects.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from src.core.types import CalibrationResult
from src.models.expected_shortfall.signal import (
    apply_liquidity_scaling,
    fit_student_t,
    portfolio_pnl,
)
from src.models.expected_shortfall.types import ESFit, ESInputs

MODEL_NAME: str = "expected_shortfall"


def calibrate(
    data: ESInputs,
    *,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Estimate the risk parameters keyed to `data`'s method / lookback.

    For `parametric_normal` and `monte_carlo`, the deliverable is the
    `(mu, Sigma)` pair on the trailing `lookback_N` returns. For
    `parametric_t`, a 1-D Student-t (`nu, loc, scale`) is additionally fitted
    to the **portfolio loss series** `-(returns @ w_dollar)`. For
    `historical`, the empirical distribution is the calibration itself —
    only sample moments are reported for diagnostic comparability.
    """

    if not isinstance(data, ESInputs):
        raise TypeError(f"calibrate expects ESInputs, got {type(data).__name__}")

    rets = data.returns.tail(data.lookback_N)
    rets = apply_liquidity_scaling(rets, data.liquidity_scalars)
    arr = np.asarray(rets.values, dtype=float)
    mu = arr.mean(axis=0)
    Sigma = np.cov(arr, rowvar=False, ddof=1)
    if Sigma.ndim == 0:
        # Single-ticker portfolio: np.cov returns a 0-D scalar.
        Sigma = np.array([[float(Sigma)]], dtype=float)
    w = data.dollar_positions
    sigma_p_sq = float(w @ Sigma @ w)
    sigma_p = float(np.sqrt(max(sigma_p_sq, 0.0)))

    nu: float | None = None
    loc: float | None = None
    scale: float | None = None
    if data.method == "parametric_t":
        port_pnl = portfolio_pnl(rets, w)
        loss_series = -port_pnl
        nu, loc, scale = fit_student_t(loss_series)

    fit = ESFit(
        mu=mu,
        Sigma=Sigma,
        sigma_p=sigma_p,
        method=data.method,
        alpha=data.alpha,
        n_obs=arr.shape[0],
        nu=nu,
        loc=loc,
        scale=scale,
    )

    fit_metrics: dict[str, float] = {
        "sigma_p": sigma_p,
        "n_obs": float(arr.shape[0]),
        "mean_dollar_pnl": float(mu @ w),
        "cov_condition_number": (
            float(np.linalg.cond(Sigma)) if Sigma.shape[0] > 0 else float("nan")
        ),
    }
    if nu is not None:
        assert loc is not None and scale is not None
        fit_metrics["t_nu"] = nu
        fit_metrics["t_loc"] = loc
        fit_metrics["t_scale"] = scale

    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={"fit": fit, "tickers": data.tickers},
        fit_metrics=fit_metrics,
        timestamp=timestamp or data.timestamp or datetime.now(UTC),
        metadata={
            "method": data.method,
            "alpha": data.alpha,
            "lookback_N": data.lookback_N,
            "portfolio_id": data.portfolio_id,
        },
    )
