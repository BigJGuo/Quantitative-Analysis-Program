"""Calibration entry point for the stress-tests risk model.

`calibrate()` builds the `StressFit` that downstream `predict()` needs:

- Per-asset OLS betas onto the canonical factor set (`DEFAULT_FACTORS`).
- Sample covariance of factor returns over the calibration window.

Data fetching is the model class's job — keeping I/O out of this module
makes calibration unit-testable with synthetic factor / return panels.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.stress_tests.signal import estimate_betas_ols
from src.models.stress_tests.types import StressFit

MODEL_NAME: str = "stress_tests"


def calibrate(
    *,
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    capital: float,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Fit per-asset betas and factor covariance.

    Parameters
    ----------
    asset_returns:
        `(T, N)` panel of asset (portfolio-position) daily returns.
    factor_returns:
        `(T, K)` panel of factor daily returns. Columns are factor names.
    capital:
        Notional capital denominator for the reverse-stress loss target.

    Returns
    -------
    CalibrationResult
        ``parameters["fit"]`` is the `StressFit`; ``parameters["betas"]`` is
        the same data exposed as a `DataFrame` for downstream convenience.
    """

    if capital <= 0:
        raise ValueError(f"capital must be > 0, got {capital}")
    if asset_returns.empty:
        raise ValueError("asset_returns must be non-empty")
    if factor_returns.empty:
        raise ValueError("factor_returns must be non-empty")

    ts = timestamp or datetime.now(UTC)
    betas_df = estimate_betas_ols(
        asset_returns=asset_returns, factor_returns=factor_returns
    )

    aligned_factors = factor_returns.dropna()
    factor_cov = np.cov(aligned_factors.to_numpy(dtype=float), rowvar=False, ddof=1)
    if factor_cov.ndim == 0:
        factor_cov = factor_cov.reshape(1, 1)

    tickers = tuple(str(c) for c in betas_df.index)
    factor_names = tuple(str(c) for c in betas_df.columns)
    fit = StressFit(
        tickers=tickers,
        factor_names=factor_names,
        betas=betas_df.to_numpy(dtype=float),
        factor_covariance=factor_cov,
        capital=float(capital),
    )

    fit_metrics: dict[str, float] = {
        "n_assets": float(fit.n_assets),
        "n_factors": float(fit.n_factors),
        "n_obs": float(len(aligned_factors)),
        "mean_abs_beta": float(np.mean(np.abs(fit.betas))),
        "max_abs_beta": float(np.max(np.abs(fit.betas))),
        "factor_cov_trace": float(np.trace(fit.factor_covariance)),
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={"fit": fit, "betas": betas_df},
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata={"factor_names": list(factor_names), "capital": float(capital)},
    )
