"""Calibration entry point for the Futures-Implied NAV model.

The fit target is the vector of regression coefficients beta on the hedge
returns, regularized with ridge and constrained to a plausible region. The
spec calls for:

* overnight basket returns r_B[d] = NAV_open(d+1) / NAV_close(d) - 1
* hedge returns r_H[d, j] over the same US window
* ridge regression with lambda chosen by K-fold CV
* box constraints 0 <= beta_j <= 1.5 and a sum constraint sum(beta) <= 1.2,
  enforced via projected gradient descent when the unconstrained ridge
  solution violates them
* variance estimates for the three blend sources from realized residuals
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.core.types import CalibrationResult
from src.models.futures_implied_nav.types import (
    BetaVector,
    HedgePanel,
    SourceVariances,
)

MODEL_NAME: str = "futures_implied_nav"

_BETA_MAX: float = 1.5
_BETA_SUM_MAX: float = 1.2
_DEFAULT_LAMBDAS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
_DEFAULT_CV_FOLDS: int = 5
_MIN_OBS: int = 8
_PROJ_MAX_ITERS: int = 500
_PROJ_TOL: float = 1e-9
_EPS: float = 1e-12


FloatArray = NDArray[np.float64]


def _overnight_returns(panel: HedgePanel) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Build the design matrix R_H and target r_B from a `HedgePanel`.

    Returns `(r_B, R_H, tickers)`. Ticker order is sorted for determinism.
    """

    tickers = tuple(sorted(panel.hedge_close.keys()))
    nav_close = np.asarray(panel.nav_close, dtype=np.float64)
    nav_open = np.asarray(panel.nav_open, dtype=np.float64)
    if np.any(nav_close <= 0):
        raise ValueError("HedgePanel.nav_close contains non-positive entries")

    r_b: FloatArray = nav_open / nav_close - 1.0

    hedge_returns: list[FloatArray] = []
    for ticker in tickers:
        p_close = np.asarray(panel.hedge_close[ticker], dtype=np.float64)
        p_now = np.asarray(panel.hedge_now[ticker], dtype=np.float64)
        if np.any(p_close <= 0):
            raise ValueError(f"hedge_close[{ticker!r}] has non-positive entries")
        hedge_returns.append(p_now / p_close - 1.0)
    r_h: FloatArray = (
        np.column_stack(hedge_returns) if hedge_returns else np.zeros((len(r_b), 0))
    )
    return r_b, r_h, tickers


def _ridge_closed_form(r_h: FloatArray, r_b: FloatArray, lam: float) -> FloatArray:
    """Closed-form ridge solution `(X'X + lam I)^-1 X'y` (no intercept)."""

    j = r_h.shape[1]
    gram = r_h.T @ r_h + lam * np.eye(j)
    return cast(FloatArray, np.linalg.solve(gram, r_h.T @ r_b))


def _project_box_and_sum(
    beta: FloatArray, beta_max: float, sum_max: float
) -> FloatArray:
    """Project `beta` onto {0 <= b_j <= beta_max, sum(b) <= sum_max}.

    Box projection is a simple clip. The sum constraint, when binding, is
    handled by solving for a single threshold mu such that
    `sum(clip(b - mu, 0, beta_max)) = sum_max`. Bisection on mu is
    monotone in [0, beta.max()] and converges in ~50 iters to 1e-12.
    """

    b: FloatArray = np.clip(beta, 0.0, beta_max)
    s = float(b.sum())
    if s <= sum_max + _PROJ_TOL:
        return b

    lo, hi = 0.0, float(b.max())
    for _ in range(60):
        mu = 0.5 * (lo + hi)
        s_mu = float(np.clip(b - mu, 0.0, beta_max).sum())
        if s_mu > sum_max:
            lo = mu
        else:
            hi = mu
        if hi - lo < _PROJ_TOL:
            break
    return cast(FloatArray, np.clip(b - 0.5 * (lo + hi), 0.0, beta_max))


def _constrained_ridge(
    r_h: FloatArray,
    r_b: FloatArray,
    lam: float,
    beta_max: float = _BETA_MAX,
    sum_max: float = _BETA_SUM_MAX,
) -> FloatArray:
    """Ridge regression with box + sum constraints via projected gradient descent.

    Starts from the unconstrained ridge solution. If it already satisfies the
    constraints we return immediately. Otherwise projected gradient steps with
    a Lipschitz-safe stepsize converge quickly because the objective is
    strongly convex (lam > 0).
    """

    j = r_h.shape[1]
    if j == 0:
        return np.zeros(0, dtype=np.float64)

    beta_unc = _ridge_closed_form(r_h, r_b, lam)
    if (
        np.all(beta_unc >= -_PROJ_TOL)
        and np.all(beta_unc <= beta_max + _PROJ_TOL)
        and float(beta_unc.sum()) <= sum_max + _PROJ_TOL
    ):
        return cast(FloatArray, np.clip(beta_unc, 0.0, beta_max))

    gram = r_h.T @ r_h + lam * np.eye(j)
    rhs = r_h.T @ r_b
    eigvals = np.linalg.eigvalsh(gram)
    lipschitz = float(eigvals.max()) + _EPS
    step = 1.0 / lipschitz

    beta = _project_box_and_sum(beta_unc, beta_max, sum_max)
    for _ in range(_PROJ_MAX_ITERS):
        grad = gram @ beta - rhs
        new = _project_box_and_sum(beta - step * grad, beta_max, sum_max)
        if float(np.linalg.norm(new - beta)) < _PROJ_TOL:
            beta = new
            break
        beta = new
    return beta


def _r_squared(r_b: FloatArray, r_h: FloatArray, beta: FloatArray) -> float:
    pred = r_h @ beta
    resid = r_b - pred
    ss_res = float(resid @ resid)
    centered = r_b - r_b.mean()
    ss_tot = float(centered @ centered)
    if ss_tot <= _EPS:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _cv_lambda(
    r_h: FloatArray,
    r_b: FloatArray,
    lambdas: Sequence[float],
    n_folds: int,
) -> float:
    """Pick lambda by K-fold MSE on the held-out fold.

    Falls back to the smallest lambda when the sample is too small to fold.
    """

    n = r_b.shape[0]
    n_folds = max(2, min(n_folds, n))
    if n < n_folds * 2:
        return float(lambdas[0])

    fold_assign = np.arange(n) % n_folds
    best_lam = float(lambdas[0])
    best_mse = float("inf")
    for lam in lambdas:
        mse_acc = 0.0
        for k in range(n_folds):
            test_mask = fold_assign == k
            train_mask = ~test_mask
            beta = _constrained_ridge(r_h[train_mask], r_b[train_mask], lam)
            pred = r_h[test_mask] @ beta
            resid = r_b[test_mask] - pred
            mse_acc += float(resid @ resid) / max(int(test_mask.sum()), 1)
        mse_acc /= n_folds
        if mse_acc < best_mse:
            best_mse = mse_acc
            best_lam = float(lam)
    return best_lam


def _source_variances(
    residuals: FloatArray,
    *,
    minutes_since_strike: float,
    intraday_variance_per_minute: float,
    etf_realized_variance: float,
) -> SourceVariances:
    """Estimate the three blend variances from realized residuals.

    `sigma2_hedge` is the realized residual variance from the regression.
    `sigma2_nav` grows linearly with staleness: time-since-strike (in minutes)
    times the intraday basket variance per minute. `sigma2_etf` is supplied
    by the caller (online realized variance of the ETF mid around fair value).
    """

    sigma2_hedge = float(np.var(residuals, ddof=1)) if residuals.size > 1 else 1e-6
    sigma2_hedge = max(sigma2_hedge, _EPS)
    sigma2_nav = max(
        minutes_since_strike * intraday_variance_per_minute, _EPS
    )
    sigma2_etf = max(etf_realized_variance, _EPS)
    return SourceVariances(
        sigma2_etf=sigma2_etf, sigma2_nav=sigma2_nav, sigma2_hedge=sigma2_hedge
    )


def calibrate(
    panel: HedgePanel,
    *,
    lambdas: Sequence[float] = _DEFAULT_LAMBDAS,
    n_folds: int = _DEFAULT_CV_FOLDS,
    beta_max: float = _BETA_MAX,
    sum_max: float = _BETA_SUM_MAX,
    minutes_since_strike: float = 240.0,
    intraday_variance_per_minute: float = 1e-8,
    etf_realized_variance: float = 1e-6,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Fit beta + variances from a `HedgePanel` of overnight returns.

    Returns a `CalibrationResult` whose `parameters` dict contains a `BetaVector`
    under "beta" and a `SourceVariances` under "variances".
    """

    if not isinstance(panel, HedgePanel):
        raise TypeError(f"calibrate() expected HedgePanel, got {type(panel).__name__}")

    r_b, r_h, tickers = _overnight_returns(panel)
    n_obs, n_hedges = r_h.shape
    if n_obs < _MIN_OBS:
        raise ValueError(
            f"calibrate() needs at least {_MIN_OBS} observations, got {n_obs}"
        )
    if n_hedges == 0:
        raise ValueError("calibrate() requires at least one hedge in the panel")

    lam = _cv_lambda(r_h, r_b, lambdas, n_folds)
    beta_arr = _constrained_ridge(r_h, r_b, lam, beta_max=beta_max, sum_max=sum_max)
    residuals = r_b - r_h @ beta_arr
    r2 = _r_squared(r_b, r_h, beta_arr)
    sigma2_hedge_fit = float(np.var(residuals, ddof=1)) if residuals.size > 1 else 0.0

    beta_vec = BetaVector(
        tickers=tickers,
        betas=tuple(float(b) for b in beta_arr),
        ridge_lambda=lam,
        r_squared=r2,
        residual_variance=sigma2_hedge_fit,
    )

    variances = _source_variances(
        residuals,
        minutes_since_strike=minutes_since_strike,
        intraday_variance_per_minute=intraday_variance_per_minute,
        etf_realized_variance=etf_realized_variance,
    )

    fit_metrics: dict[str, float] = {
        "n_obs": float(n_obs),
        "n_hedges": float(n_hedges),
        "ridge_lambda": lam,
        "r_squared": r2,
        "residual_variance": sigma2_hedge_fit,
        "beta_sum": float(beta_arr.sum()),
        "beta_max": float(beta_arr.max()) if beta_arr.size else 0.0,
    }
    parameters: dict[str, object] = {
        "beta": beta_vec,
        "variances": variances,
    }

    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=timestamp or datetime.now(UTC),
        metadata={
            "lambda_grid": tuple(float(x) for x in lambdas),
            "n_folds": float(n_folds),
            "beta_max_bound": float(beta_max),
            "sum_max_bound": float(sum_max),
        },
    )
