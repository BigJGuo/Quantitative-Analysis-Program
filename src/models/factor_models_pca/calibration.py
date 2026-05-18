"""Calibration entry point for the factor-model / PCA risk model.

`calibrate()` dispatches on `method` to one of three estimators:

- ``"PCA"``         — eigendecomposition of the sample covariance.
- ``"FamaFrench"``  — per-asset time-series OLS on exogenously supplied
                      factor returns.
- ``"Barra"``       — daily cross-sectional WLS on supplied exposures, then
                      pooled into a factor-return covariance.

All three return the same `CalibrationResult` shape so the orchestration
layer doesn't need to branch.

The function is deliberately data-only: it accepts a pre-built returns panel
(plus, for the fundamental paths, a factor-return panel or an exposures
DataFrame). Data fetching is the model class's job — keeping I/O out of this
module makes the math unit-testable with synthetic inputs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.factor_models_pca.signal import (
    DEFAULT_MIN_SPECIFIC_VARIANCE,
    cross_sectional_factor_regression,
    pca_decompose,
    time_series_factor_regression,
)
from src.models.factor_models_pca.types import (
    VALID_METHODS,
    FactorFit,
    FactorMethod,
)

MODEL_NAME: str = "factor_models_pca"


def calibrate(
    *,
    returns: pd.DataFrame,
    method: FactorMethod = "PCA",
    K: int | None = 5,
    use_mp_cutoff: bool = False,
    standardize: bool = False,
    min_specific_variance: float = DEFAULT_MIN_SPECIFIC_VARIANCE,
    factor_returns: pd.DataFrame | None = None,
    exposures_by_date: dict[pd.Timestamp, pd.DataFrame] | None = None,
    market_cap_by_date: dict[pd.Timestamp, pd.Series] | None = None,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Fit the factor model.

    Parameters
    ----------
    returns:
        `(T, N)` returns panel. Columns are tickers; rows are dates.
    method:
        Which estimator to dispatch to.
    K:
        Number of factors (PCA only — ignored for the fundamental paths,
        which infer K from the supplied factor / exposure shape).
    use_mp_cutoff:
        For PCA, when True, override `K` (or cap it) with the Marchenko-
        Pastur upper edge: keep only eigenvalues above
        `(1 + sqrt(N/T))^2 * sigma^2_noise`.
    standardize:
        Optionally divide each column by its sample std before PCA. Used to
        balance high- and low-vol stocks.
    factor_returns:
        Required for `method="FamaFrench"`. `(T, K)` panel of factor return
        time series, sharing dates with `returns`.
    exposures_by_date:
        Required for `method="Barra"`. Mapping `date -> (N, K)` exposures
        DataFrame aligned to `returns.columns`.
    market_cap_by_date:
        Optional for Barra. Mapping `date -> N-length market caps` used for
        the WLS weight `sqrt(market_cap)`. When omitted, all assets receive
        unit weight (OLS).
    """

    if method not in VALID_METHODS:
        raise ValueError(
            f"calibrate: method must be one of {sorted(VALID_METHODS)}, "
            f"got {method!r}"
        )
    if returns is None or returns.empty:
        raise ValueError("calibrate requires a non-empty returns panel")
    ts = timestamp or datetime.now(UTC)

    if method == "PCA":
        fit = pca_decompose(
            returns,
            K=K,
            use_mp_cutoff=use_mp_cutoff,
            min_specific_variance=min_specific_variance,
            standardize=standardize,
        )
        fit_metrics = _pca_metrics(fit, n_obs=returns.shape[0])
        return _wrap(fit, ts, fit_metrics, {"standardize": standardize})

    if method == "FamaFrench":
        if factor_returns is None or factor_returns.empty:
            raise ValueError("FamaFrench calibration requires `factor_returns`")
        fit = time_series_factor_regression(
            returns,
            factor_returns,
            min_specific_variance=min_specific_variance,
        )
        fit_metrics = _fundamental_metrics(fit)
        return _wrap(fit, ts, fit_metrics, {"factor_count": fit.n_factors})

    # Barra path.
    if exposures_by_date is None or not exposures_by_date:
        raise ValueError("Barra calibration requires `exposures_by_date`")
    fit = _barra_fit(
        returns=returns,
        exposures_by_date=exposures_by_date,
        market_cap_by_date=market_cap_by_date,
        min_specific_variance=min_specific_variance,
    )
    fit_metrics = _fundamental_metrics(fit)
    return _wrap(fit, ts, fit_metrics, {"factor_count": fit.n_factors})


def _barra_fit(
    *,
    returns: pd.DataFrame,
    exposures_by_date: dict[pd.Timestamp, pd.DataFrame],
    market_cap_by_date: dict[pd.Timestamp, pd.Series] | None,
    min_specific_variance: float,
) -> FactorFit:
    """Pool one cross-sectional regression per date into a `FactorFit`.

    Daily steps:

    1. For each date `t`, solve `r_t = B_t f_t + eps_t` via WLS with weights
       `sqrt(market_cap_i(t))` (uniform if no market caps supplied).
    2. Collect the time series of `f_hat_t` and per-asset residuals.
    3. Aggregate: `B` is the *latest* day's exposures (the model uses the
       current snapshot for downstream risk calcs), `F` is the sample
       covariance of `f_hat_t`, `D` is the per-asset residual variance.
    """

    tickers = tuple(str(c) for c in returns.columns)
    sample_exposures = next(iter(exposures_by_date.values()))
    factor_names = tuple(str(c) for c in sample_exposures.columns)
    f_hat_rows: list[np.ndarray] = []
    resid_rows: list[np.ndarray] = []
    f_hat_dates: list[pd.Timestamp] = []

    for raw_date, row in returns.iterrows():
        date = cast(pd.Timestamp, raw_date)
        exposures = exposures_by_date.get(date)
        if exposures is None:
            continue
        if list(exposures.columns) != list(factor_names):
            raise ValueError(
                f"Exposures for {date!s} have factor columns "
                f"{list(exposures.columns)}, expected {list(factor_names)}"
            )
        if list(exposures.index) != list(returns.columns):
            raise ValueError(
                f"Exposures for {date!s} have ticker index "
                f"{list(exposures.index)}, expected {list(returns.columns)}"
            )
        weights = None
        if market_cap_by_date is not None and date in market_cap_by_date:
            caps = market_cap_by_date[date].reindex(returns.columns).to_numpy(dtype=float)
            weights = np.clip(caps, 0.0, None)
        f_hat, resid = cross_sectional_factor_regression(
            row.to_numpy(dtype=float),
            exposures.to_numpy(dtype=float),
            weights=weights,
        )
        f_hat_rows.append(f_hat)
        resid_rows.append(resid)
        f_hat_dates.append(date)

    if not f_hat_rows:
        raise ValueError("Barra calibration produced no overlapping dates")

    f_hat_panel = np.vstack(f_hat_rows)
    resid_panel = np.vstack(resid_rows)
    F_cov = np.cov(f_hat_panel, rowvar=False, ddof=1)
    if F_cov.ndim == 0:
        F_cov = F_cov.reshape(1, 1)
    specific_var = resid_panel.var(axis=0, ddof=1)
    specific_var = np.maximum(specific_var, min_specific_variance)

    last_date = f_hat_dates[-1]
    B = exposures_by_date[last_date].to_numpy(dtype=float)

    total_var = returns.var(axis=0, ddof=1).to_numpy(dtype=float)
    total_var = np.where(total_var > 0, total_var, 1.0)
    r_squared_per_asset = 1.0 - (resid_panel.var(axis=0, ddof=1) / total_var)
    variance_explained = float(np.mean(np.clip(r_squared_per_asset, 0.0, 1.0)))

    return FactorFit(
        B=B,
        F=F_cov,
        specific_variances=specific_var,
        tickers=tickers,
        factor_names=factor_names,
        method="Barra",
        eigenvalues=None,
        marchenko_pastur_cutoff=None,
        variance_explained=variance_explained,
        min_specific_variance=min_specific_variance,
    )


def _pca_metrics(fit: FactorFit, *, n_obs: int) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_assets": float(fit.n_assets),
        "n_factors": float(fit.n_factors),
        "n_obs": float(n_obs),
        "variance_explained": fit.variance_explained,
        "mean_specific_variance": float(fit.specific_variances.mean()),
        "max_specific_variance": float(fit.specific_variances.max()),
    }
    if fit.eigenvalues is not None and fit.eigenvalues.size > 0:
        metrics["top_eigenvalue"] = float(fit.eigenvalues[0])
        metrics["eigenvalue_sum"] = float(fit.eigenvalues.sum())
    if fit.marchenko_pastur_cutoff is not None:
        metrics["mp_upper_edge"] = float(fit.marchenko_pastur_cutoff)
        if fit.eigenvalues is not None:
            metrics["n_factors_above_mp"] = float(
                int(np.sum(fit.eigenvalues > fit.marchenko_pastur_cutoff))
            )
    return metrics


def _fundamental_metrics(fit: FactorFit) -> dict[str, float]:
    return {
        "n_assets": float(fit.n_assets),
        "n_factors": float(fit.n_factors),
        "variance_explained": fit.variance_explained,
        "mean_specific_variance": float(fit.specific_variances.mean()),
        "max_specific_variance": float(fit.specific_variances.max()),
    }


def _wrap(
    fit: FactorFit,
    ts: datetime,
    fit_metrics: dict[str, float],
    metadata: dict[str, object],
) -> CalibrationResult:
    parameters: dict[str, object] = {
        "factor_fit": fit,
        "method": fit.method,
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata=metadata,
    )
