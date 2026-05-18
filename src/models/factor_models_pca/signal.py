"""Pure math layer of the factor / PCA model.

Nothing in this module performs I/O. Every function is a deterministic
transformation of arrays (or DataFrames at the data-loading boundary), which
keeps the math unit-testable without a mock provider.

The functions roughly mirror the spec's Algorithm Outline:

- `compute_log_returns`, `align_returns_panel`               -> step 1
- `marchenko_pastur_upper_edge`, `pca_decompose`             -> step 2
- `time_series_factor_regression`                            -> step 3
- `cross_sectional_factor_regression`                        -> step 4
- `compute_full_covariance`, `portfolio_risk`                -> step 5-6
- `residualize`, `residual_returns`                          -> step 7
- `residual_autocorrelation`, `residual_max_cross_corr`,
  `variance_explained_curve`, `ljung_box_pvalue`             -> step 9 / validation
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from src.models.factor_models_pca.types import FactorFit, PortfolioRisk

DEFAULT_MIN_SPECIFIC_VARIANCE: float = 1e-8


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: `log(P_t) - log(P_{t-1})`.

    Accepts a (T, N) DataFrame of adjusted-close prices. Drops the leading row
    (the first diff is NaN) and any rows that ended up all-NaN.
    """

    if prices.empty:
        return prices.iloc[0:0].copy()
    log_arr = np.log(prices.to_numpy(dtype=float))
    log_p = pd.DataFrame(log_arr, index=prices.index, columns=prices.columns)
    return log_p.diff().dropna(how="all")


def align_returns_panel(
    returns_by_ticker: dict[str, pd.Series],
    *,
    min_coverage: float = 0.95,
    winsorize_quantile: float | None = 0.01,
) -> pd.DataFrame:
    """Align per-ticker return Series on a common date index.

    Drops tickers whose coverage on the union of dates is below `min_coverage`,
    forward-fills the rest, and optionally winsorizes each column at the
    `winsorize_quantile` / `1 - winsorize_quantile` levels. Returns a panel
    `R` of shape `(T, N_kept)`.
    """

    if not returns_by_ticker:
        return pd.DataFrame()

    panel = pd.DataFrame(returns_by_ticker)
    n_total = len(panel)
    if n_total == 0:
        return panel

    keep_cols = []
    for col in panel.columns:
        coverage = panel[col].notna().sum() / n_total
        if coverage >= min_coverage:
            keep_cols.append(col)
    panel = panel.loc[:, keep_cols]
    # Drop dates where every kept column is NaN, then forward-fill the holes.
    panel = panel.dropna(how="all").ffill().dropna(how="any")

    if winsorize_quantile is not None and not panel.empty:
        lo = panel.quantile(winsorize_quantile)
        hi = panel.quantile(1.0 - winsorize_quantile)
        panel = panel.clip(lower=lo, upper=hi, axis=1)

    return panel


def marchenko_pastur_upper_edge(n_assets: int, n_obs: int, sigma_squared: float) -> float:
    """RMT upper edge `(1 + sqrt(N/T))^2 * sigma^2`.

    Eigenvalues above this threshold are interpreted as real factor signal;
    eigenvalues below are statistical noise consistent with a Wishart sample
    covariance of i.i.d. variance-`sigma^2` returns.
    """

    if n_assets <= 0 or n_obs <= 0 or sigma_squared <= 0:
        raise ValueError(
            f"marchenko_pastur_upper_edge requires N, T, sigma^2 > 0; "
            f"got N={n_assets}, T={n_obs}, sigma^2={sigma_squared}"
        )
    ratio = n_assets / n_obs
    return float(sigma_squared * (1.0 + np.sqrt(ratio)) ** 2)


def pca_decompose(
    returns: pd.DataFrame,
    *,
    K: int | None = None,
    use_mp_cutoff: bool = False,
    min_specific_variance: float = DEFAULT_MIN_SPECIFIC_VARIANCE,
    standardize: bool = False,
) -> FactorFit:
    """PCA factor decomposition of a returns panel.

    Mirrors spec Algorithm step 2:

    1. Center (and optionally standardize) the returns.
    2. Eigendecompose the sample covariance.
    3. Pick K either explicitly or via the Marchenko-Pastur upper edge.
    4. Form `B = U_{:K} * sqrt(lambda_{:K})` and `D = diag(Sigma - B B^T)`.

    Exactly one of `K` or `use_mp_cutoff=True` must determine the rank: pass
    `K` to force a count, pass `use_mp_cutoff=True` with `K=None` to let RMT
    decide. (If both are passed, `K` is treated as a hard cap on the
    RMT-selected count.)
    """

    if returns.empty:
        raise ValueError("pca_decompose requires a non-empty returns panel")
    n_obs, n_assets = returns.shape
    if n_obs < 2:
        raise ValueError(
            f"pca_decompose requires at least 2 observations, got {n_obs}"
        )

    tickers = tuple(str(c) for c in returns.columns)
    R = returns.to_numpy(dtype=float)
    means = R.mean(axis=0)
    R_c = R - means

    if standardize:
        stds = R_c.std(axis=0, ddof=1)
        # Guard against zero-vol columns; they are degenerate and should not
        # blow up the divide. Replace zero with 1 so the column stays at 0.
        safe_stds = np.where(stds > 0, stds, 1.0)
        R_c = R_c / safe_stds

    sigma_hat = (R_c.T @ R_c) / (n_obs - 1)
    # Symmetrize defensively — accumulated round-off can knock `eigh` off.
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(sigma_hat)
    # eigh returns ascending order; flip so eigenvalues are descending.
    eigvals = eigvals_asc[::-1]
    eigvecs = eigvecs_asc[:, ::-1]
    eigvals = np.clip(eigvals, 0.0, None)

    sigma_squared_noise = float(np.median(eigvals))
    mp_cutoff = marchenko_pastur_upper_edge(n_assets, n_obs, max(sigma_squared_noise, 1e-12))

    if K is None and not use_mp_cutoff:
        raise ValueError(
            "pca_decompose requires either K or use_mp_cutoff=True"
        )

    if use_mp_cutoff:
        k_rmt = int(np.sum(eigvals > mp_cutoff))
        k_rmt = max(k_rmt, 1)
        K_keep = min(K, k_rmt) if K is not None else k_rmt
    else:
        assert K is not None
        K_keep = K

    K_keep = max(1, min(K_keep, n_assets))

    sqrt_lambda = np.sqrt(eigvals[:K_keep])
    B = eigvecs[:, :K_keep] * sqrt_lambda  # (N, K)
    # Fitted covariance from the top-K subspace.
    fitted = R_c @ eigvecs[:, :K_keep] @ eigvecs[:, :K_keep].T
    resid = R_c - fitted
    specific_var = resid.var(axis=0, ddof=1)
    specific_var = np.maximum(specific_var, min_specific_variance)

    F = np.eye(K_keep)
    total_var = float(eigvals.sum())
    variance_explained = float(eigvals[:K_keep].sum() / total_var) if total_var > 0 else 0.0

    factor_names = tuple(f"PC{i + 1}" for i in range(K_keep))
    return FactorFit(
        B=B,
        F=F,
        specific_variances=specific_var,
        tickers=tickers,
        factor_names=factor_names,
        method="PCA",
        eigenvalues=eigvals,
        marchenko_pastur_cutoff=mp_cutoff,
        variance_explained=variance_explained,
        min_specific_variance=min_specific_variance,
    )


def time_series_factor_regression(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    min_specific_variance: float = DEFAULT_MIN_SPECIFIC_VARIANCE,
) -> FactorFit:
    """OLS time-series regression of each asset's returns on factor returns.

    Used for the Fama-French / Carhart path. The two frames must share a
    common date index; the regression is per-asset and an intercept is
    included implicitly by centering both sides (we estimate `alpha_i`
    implicitly via the residual mean but only report the slope).
    """

    if returns.empty or factor_returns.empty:
        raise ValueError("time_series_factor_regression requires non-empty inputs")
    aligned = returns.join(factor_returns, how="inner")
    factor_cols = list(factor_returns.columns)
    asset_cols = list(returns.columns)
    aligned = aligned.dropna()
    if aligned.empty:
        raise ValueError("returns and factor_returns share no overlapping dates")

    F_mat = aligned[factor_cols].to_numpy(dtype=float)
    R_mat = aligned[asset_cols].to_numpy(dtype=float)
    F_mean = F_mat.mean(axis=0)
    R_mean = R_mat.mean(axis=0)
    F_c = F_mat - F_mean
    R_c = R_mat - R_mean

    # Solve (F_c^T F_c) beta = F_c^T R_c column-wise; lstsq is the robust route.
    betas, *_ = np.linalg.lstsq(F_c, R_c, rcond=None)  # (K, N)
    B = betas.T  # (N, K)

    fitted = F_c @ betas
    resid = R_c - fitted
    specific_var = resid.var(axis=0, ddof=1)
    specific_var = np.maximum(specific_var, min_specific_variance)

    factor_cov = np.cov(F_c, rowvar=False, ddof=1)
    if factor_cov.ndim == 0:
        factor_cov = factor_cov.reshape(1, 1)

    asset_var_total = R_mat.var(axis=0, ddof=1)
    asset_var_total = np.where(asset_var_total > 0, asset_var_total, 1.0)
    r_squared_per_asset = 1.0 - (resid.var(axis=0, ddof=1) / asset_var_total)
    variance_explained = float(np.mean(np.clip(r_squared_per_asset, 0.0, 1.0)))

    return FactorFit(
        B=B,
        F=factor_cov,
        specific_variances=specific_var,
        tickers=tuple(str(c) for c in asset_cols),
        factor_names=tuple(str(c) for c in factor_cols),
        method="FamaFrench",
        eigenvalues=None,
        marchenko_pastur_cutoff=None,
        variance_explained=variance_explained,
        min_specific_variance=min_specific_variance,
    )


def cross_sectional_factor_regression(
    asset_returns: np.ndarray,
    exposures: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-period WLS cross-sectional regression `r_t = B_t f_t + eps_t`.

    `asset_returns` is `(N,)`, `exposures` is `(N, K)`, optional `weights` is
    `(N,)` (e.g. `sqrt(market_cap)`). Returns `(f_hat, residuals)` with shapes
    `(K,)` and `(N,)`.

    Provided for the Barra path; the model's Barra calibration loops this over
    a window and aggregates the time series of `f_hat_t`.
    """

    n_assets = asset_returns.shape[0]
    if exposures.shape[0] != n_assets:
        raise ValueError(
            f"exposures has {exposures.shape[0]} rows but asset_returns has "
            f"{n_assets}"
        )
    if weights is None:
        weights = np.ones(n_assets)
    if weights.shape != (n_assets,):
        raise ValueError(
            f"weights shape {weights.shape} does not match N={n_assets}"
        )
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    Bw = exposures * sqrt_w[:, None]
    rw = asset_returns * sqrt_w
    f_hat, *_ = np.linalg.lstsq(Bw, rw, rcond=None)
    residuals = asset_returns - exposures @ f_hat
    return f_hat, residuals


def compute_full_covariance(fit: FactorFit) -> np.ndarray:
    """Reconstitute `Sigma = B F B^T + D` from a fitted model."""

    D = np.diag(fit.specific_variances)
    return cast(np.ndarray, fit.B @ fit.F @ fit.B.T + D)


def portfolio_risk(weights: np.ndarray, fit: FactorFit) -> PortfolioRisk:
    """Decompose portfolio variance into systematic + specific contributions.

    Returns a `PortfolioRisk` whose `factor_contributions[k]` is the variance
    that factor `k` adds to the portfolio (so the K-vector sums to
    `systematic_vol ** 2`).
    """

    if weights.shape != (fit.n_assets,):
        raise ValueError(
            f"weights shape {weights.shape} does not match N={fit.n_assets}"
        )
    factor_exposures = fit.B.T @ weights  # (K,)
    systematic_var = float(factor_exposures @ fit.F @ factor_exposures)
    specific_var = float(weights @ (fit.specific_variances * weights))
    total_var = systematic_var + specific_var

    # Per-factor contribution: x_k * (F x)_k. Sum equals x^T F x.
    Fx = fit.F @ factor_exposures
    factor_contributions = factor_exposures * Fx

    return PortfolioRisk(
        total_vol=float(np.sqrt(max(total_var, 0.0))),
        systematic_vol=float(np.sqrt(max(systematic_var, 0.0))),
        specific_vol=float(np.sqrt(max(specific_var, 0.0))),
        factor_exposures=factor_exposures,
        factor_contributions=factor_contributions,
    )


def residualize(signal_values: np.ndarray, fit: FactorFit) -> np.ndarray:
    """Project a cross-sectional signal onto the factor null-space.

    `s_neutral = s - B (B^T B)^{-1} B^T s`. Returns the orthogonalized signal,
    same shape as input. This is the projection step downstream stat-arb
    models use to strip out factor beta from raw signals.
    """

    if signal_values.shape != (fit.n_assets,):
        raise ValueError(
            f"signal_values shape {signal_values.shape} does not match "
            f"N={fit.n_assets}"
        )
    B = fit.B
    gamma, *_ = np.linalg.lstsq(B, signal_values, rcond=None)
    return cast(np.ndarray, signal_values - B @ gamma)


def residual_returns(returns: pd.DataFrame, fit: FactorFit) -> pd.DataFrame:
    """Residual returns `R - F B^T` for use by downstream stat-arb.

    For PCA, `F` (the factor scores time series) is recovered as
    `R_c U_{:K} Lambda^{-1/2}` per the spec; we recompute it on the fly from
    the saved `B` and the supplied returns panel.
    """

    if tuple(str(c) for c in returns.columns) != fit.tickers:
        raise ValueError(
            "returns columns must exactly match fit.tickers and ordering"
        )
    R = returns.to_numpy(dtype=float)
    R_c = R - R.mean(axis=0)
    # For an arbitrary B (PCA or fundamental), the projection that minimizes
    # ||R_c - F_hat B^T||^2 is F_hat = R_c B (B^T B)^{-1}.
    BtB = fit.B.T @ fit.B
    try:
        BtB_inv = np.linalg.inv(BtB)
    except np.linalg.LinAlgError:
        BtB_inv = np.linalg.pinv(BtB)
    F_hat = R_c @ fit.B @ BtB_inv
    fitted = F_hat @ fit.B.T
    resid = R_c - fitted
    return pd.DataFrame(resid, index=returns.index, columns=list(returns.columns))


def variance_explained_curve(eigenvalues: np.ndarray) -> np.ndarray:
    """Cumulative variance-explained vector `cumsum(lambda_k) / sum(lambda)`."""

    total = float(np.sum(eigenvalues))
    if total <= 0:
        return np.zeros_like(eigenvalues)
    return np.cumsum(eigenvalues) / total


def residual_autocorrelation(residuals: pd.DataFrame, lag: int = 1) -> np.ndarray:
    """Per-asset lag-`lag` autocorrelation of the residual time series."""

    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    R = residuals.to_numpy(dtype=float)
    n_obs, n_assets = R.shape
    if n_obs <= lag:
        return np.full(n_assets, np.nan)
    R_c = R - R.mean(axis=0)
    num = (R_c[lag:] * R_c[:-lag]).sum(axis=0)
    denom = (R_c * R_c).sum(axis=0)
    denom = np.where(denom > 0, denom, np.nan)
    return cast(np.ndarray, num / denom)


def ljung_box_pvalue(residual_series: np.ndarray, max_lag: int = 5) -> float:
    """Ljung-Box Q-statistic p-value against H0 = no autocorrelation.

    Implemented inline to avoid pulling scipy as a dependency. Uses a chi-square
    survival approximation via the Wilson-Hilferty transform — accurate to
    better than 0.01 in the tail region we care about (p < 0.1).
    """

    x = np.asarray(residual_series, dtype=float)
    n = x.size
    if n <= max_lag + 1:
        return float("nan")
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float("nan")
    q_stat = 0.0
    for lag in range(1, max_lag + 1):
        rho = float((x[lag:] * x[:-lag]).sum()) / denom
        q_stat += rho * rho / (n - lag)
    q_stat *= n * (n + 2)
    # Wilson-Hilferty chi^2 survival approx: if X ~ chi^2_k, then
    # ((X/k)^{1/3} - (1 - 2/(9k))) / sqrt(2/(9k)) is approx N(0, 1).
    k = max_lag
    if q_stat <= 0:
        return 1.0
    z = ((q_stat / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    # P(chi^2 > q) ≈ 1 - Phi(z) = 0.5 * erfc(z / sqrt(2)).
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def residual_max_cross_corr(residuals: pd.DataFrame) -> float:
    """Maximum off-diagonal absolute correlation among residual columns.

    Large pairwise correlations indicate a missing factor. The factor model
    should drive residuals close to cross-sectionally orthogonal.
    """

    if residuals.shape[1] < 2:
        return 0.0
    corr = np.array(residuals.corr().to_numpy(), copy=True)
    np.fill_diagonal(corr, 0.0)
    return float(np.nanmax(np.abs(corr)))


def diagonal_shrinkage(covariance: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink a sample covariance toward its diagonal at intensity `alpha`.

    `Sigma_shrunk = (1 - alpha) * Sigma + alpha * diag(Sigma)`. With
    `alpha = 1` only specific risk survives; with `alpha = 0` the sample
    covariance is returned unchanged. A pragmatic stand-in for the Ledoit-
    Wolf constant-correlation target when computing the optimal shrinkage
    intensity isn't worth the complexity (e.g., for diagnostics that just
    need a regularized covariance).
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"covariance must be square 2-D, got shape {covariance.shape}"
        )
    target = np.diag(np.diag(covariance))
    return (1.0 - alpha) * covariance + alpha * target
