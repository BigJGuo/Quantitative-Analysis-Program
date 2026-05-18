"""Calibration entry point for the Avellaneda-Lee PCA-residual stat-arb model.

`calibrate(data, **params)` runs the full per-stock pipeline:

1. Build factor returns (PCA eigenportfolios or pre-supplied sector ETFs).
2. For each stock, regress returns on factors to get residuals.
3. Integrate residuals into a cumulative process `X`.
4. Fit AR(1) to `X` over the OU window, recover OU parameters.
5. Filter on stationarity (`0 < b < 1`), half-life band, and factor R^2.
6. Compute the s-score and drift-corrected s-score for each survivor.

The function is data-only — no DataProvider, no yfinance. The model class
handles I/O.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.avellaneda_lee_stat_arb.signal import (
    cumulative_residual,
    factor_regression,
    fit_ou_ar1,
    ou_parameters_from_ar1,
    pca_eigenportfolio_returns,
    s_score,
    s_score_modified,
)
from src.models.avellaneda_lee_stat_arb.types import (
    DropReason,
    FactorMode,
    OUFit,
    StatArbFit,
    StatArbInputs,
)

MODEL_NAME: str = "avellaneda_lee_stat_arb"

DEFAULT_HALF_LIFE_BAND: tuple[float, float] = (5.0, 30.0)
DEFAULT_MIN_R_SQUARED: float = 0.20


def calibrate(
    data: StatArbInputs,
    *,
    half_life_band: tuple[float, float] = DEFAULT_HALF_LIFE_BAND,
    min_r_squared: float = DEFAULT_MIN_R_SQUARED,
    use_drift_correction: bool = True,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Run the full Avellaneda-Lee calibration on a `StatArbInputs` bundle.

    Parameters
    ----------
    data:
        Returns panel, factor specification, and window sizes.
    half_life_band:
        `(lo, hi)` in trading days. Stocks whose OU half-life falls outside
        the band are dropped per spec section "Calibration approach" item 5.
    min_r_squared:
        Minimum factor-regression R^2 over the OU window. Stocks below this
        floor are dropped per spec item 6.
    use_drift_correction:
        When True the OU fit's `s_score_mod` subtracts the drift term per
        spec eq. 15; otherwise `s_score_mod == s_score`.
    timestamp:
        Calibration time. Defaults to ``datetime.now(UTC)``.
    """

    if half_life_band[0] <= 0 or half_life_band[1] <= half_life_band[0]:
        raise ValueError(
            f"half_life_band must be (lo, hi) with 0 < lo < hi; got "
            f"{half_life_band!r}"
        )
    if not 0.0 <= min_r_squared <= 1.0:
        raise ValueError(
            f"min_r_squared must be in [0, 1], got {min_r_squared}"
        )

    ts = timestamp or datetime.now(UTC)
    returns = data.returns
    if returns.shape[0] < data.pca_window:
        raise ValueError(
            f"calibrate received {returns.shape[0]} rows but pca_window="
            f"{data.pca_window} requires at least that many."
        )

    pca_panel = returns.iloc[-data.pca_window :]
    ou_panel = returns.iloc[-data.ou_window :]

    factor_returns, eigvecs, eigvals, mp_cutoff = _build_factors(
        pca_panel=pca_panel,
        ou_panel=ou_panel,
        factor_mode=data.factor_mode,
        K=data.K,
        supplied_factor_returns=data.factor_returns,
    )

    surviving: dict[str, OUFit] = {}
    dropped: dict[str, DropReason] = {}

    for i, ticker in enumerate(data.tickers):
        asset_returns = pca_panel.iloc[:, i].to_numpy(dtype=float)
        # Factor regression uses the *full* PCA window so loadings reflect
        # the longer-horizon factor structure; residuals over the OU window
        # are extracted for the AR(1) fit.
        try:
            reg = factor_regression(asset_returns, factor_returns.to_numpy(dtype=float))
        except ValueError:
            dropped[ticker] = "insufficient_history"
            continue
        if reg.r_squared < min_r_squared:
            dropped[ticker] = "low_r_squared"
            continue

        ou_residuals = reg.residuals[-data.ou_window :]
        if ou_residuals.size < 4 or not np.all(np.isfinite(ou_residuals)):
            dropped[ticker] = "insufficient_history"
            continue
        X = cumulative_residual(ou_residuals)
        ar1 = fit_ou_ar1(X)
        if not 0.0 < ar1.b < 1.0:
            dropped[ticker] = "non_stationary"
            continue
        try:
            ou = ou_parameters_from_ar1(ar1)
        except ValueError:
            dropped[ticker] = "non_stationary"
            continue
        if ou.sigma_eq <= 0 or not math.isfinite(ou.sigma_eq):
            dropped[ticker] = "degenerate_residual"
            continue
        if not (half_life_band[0] <= ou.half_life <= half_life_band[1]):
            dropped[ticker] = "half_life_out_of_band"
            continue

        x_last = float(X[-1])
        s_raw = s_score(x_last, ou.m, ou.sigma_eq)
        s_mod = (
            s_score_modified(s_raw, reg.alpha, ou.kappa, ou.sigma_eq)
            if use_drift_correction
            else s_raw
        )
        if not (math.isfinite(s_raw) and math.isfinite(s_mod)):
            dropped[ticker] = "degenerate_residual"
            continue

        surviving[ticker] = OUFit(
            ticker=ticker,
            alpha=reg.alpha,
            beta=reg.beta,
            r_squared=reg.r_squared,
            a=ar1.a,
            b=ar1.b,
            kappa=ou.kappa,
            m=ou.m,
            sigma=ou.sigma,
            sigma_zeta=ar1.sigma_zeta,
            sigma_eq=ou.sigma_eq,
            half_life=ou.half_life,
            X_last=x_last,
            s_score=s_raw,
            s_score_mod=s_mod,
        )

    fit = StatArbFit(
        factor_mode=data.factor_mode,
        factor_names=tuple(str(c) for c in factor_returns.columns),
        universe=data.tickers,
        ou_fits=surviving,
        dropped_tickers=dropped,
        factor_returns=factor_returns,
        eigenvalues=eigvals,
        marchenko_pastur_cutoff=mp_cutoff,
        half_life_band=half_life_band,
        min_r_squared=min_r_squared,
        timestamp=ts,
    )

    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={
            "stat_arb_fit": fit,
            "factor_mode": data.factor_mode,
        },
        fit_metrics={
            "n_universe": float(len(data.tickers)),
            "n_surviving": float(fit.n_surviving),
            "n_dropped": float(fit.n_dropped),
            "survival_rate": float(fit.n_surviving / max(len(data.tickers), 1)),
            "mean_half_life": _safe_mean([f.half_life for f in surviving.values()]),
            "mean_r_squared": _safe_mean([f.r_squared for f in surviving.values()]),
            "mean_kappa": _safe_mean([f.kappa for f in surviving.values()]),
            "s_score_mean": _safe_mean([f.s_score_mod for f in surviving.values()]),
            "s_score_std": _safe_std([f.s_score_mod for f in surviving.values()]),
        },
        timestamp=ts,
        metadata={
            "factor_mode": data.factor_mode,
            "K": float(data.K),
            "pca_window": float(data.pca_window),
            "ou_window": float(data.ou_window),
            "half_life_band": list(half_life_band),
            "min_r_squared": min_r_squared,
            "use_drift_correction": use_drift_correction,
        },
    )


def _build_factors(
    *,
    pca_panel: pd.DataFrame,
    ou_panel: pd.DataFrame,
    factor_mode: FactorMode,
    K: int,
    supplied_factor_returns: pd.DataFrame | None,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, float | None]:
    """Build the factor-return panel aligned to `pca_panel.index`.

    PCA path: compute eigenportfolios over the PCA window.
    ETF path: align the supplied factor return panel to the PCA window;
    raise if any rows are missing.

    Returns ``(factor_returns, eigvecs, eigvals, mp_cutoff)``; the trailing
    three are ``None`` for the ETF path.
    """

    if factor_mode == "PCA":
        factor_returns, eigvecs, eigvals = pca_eigenportfolio_returns(
            pca_panel, K=K
        )
        n_assets = pca_panel.shape[1]
        n_obs = pca_panel.shape[0]
        # Marchenko-Pastur upper edge on the *correlation* spectrum (sigma^2=1
        # because we standardized inside `pca_eigenportfolio_returns`).
        mp_cutoff = float((1.0 + math.sqrt(n_assets / n_obs)) ** 2)
        return factor_returns, eigvecs, eigvals, mp_cutoff

    # ETF path.
    if supplied_factor_returns is None:
        raise ValueError("factor_mode='ETF' requires supplied factor_returns")
    aligned = supplied_factor_returns.reindex(pca_panel.index)
    if aligned.isna().any().any():
        n_missing = int(aligned.isna().any(axis=1).sum())
        raise ValueError(
            f"Supplied factor_returns is missing values on {n_missing} of the "
            "PCA-window dates after reindexing — fill or trim before calibration."
        )
    # K is informational in ETF mode; we use all supplied columns.
    _ = ou_panel  # silence linter; OU window slicing happens in the caller.
    return aligned, None, None, None


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _safe_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1))
