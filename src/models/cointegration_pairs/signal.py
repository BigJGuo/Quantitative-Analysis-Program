"""Pure math layer of the cointegration / pair-trading model.

Nothing in this module performs I/O. Every function is a deterministic
transformation of arrays (or `pd.Series` at the data-loading boundary), which
keeps the math unit-testable without a mock provider.

The functions roughly mirror the spec's Algorithm outline:

- `compute_log_prices`, `align_pair`                 -> step 1
- `ols_intercept_slope`, `adf_test`,
  `engle_granger_test`                               -> steps 2-5
- `fit_ar1`, `fit_ou`, `half_life_from_phi`          -> steps 6-7
- `static_zscore`, `rolling_zscore_moments`,
  `pair_position_path`                               -> stage 2a (steps 9a-d)
- `kalman_dynamic_beta`                              -> stage 2b (steps 10-12)
- `ljung_box_pvalue`, `rolling_eg_pvalue_path`,
  `realized_half_life`                               -> validation diagnostics
"""

from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
import pandas as pd

from src.models.cointegration_pairs.types import (
    ADFResult,
    EngleGrangerFit,
    KalmanFit,
    OUFit,
    TradingRule,
)

# ----- ADF / Engle-Granger critical values -----------------------------------
#
# Asymptotic critical values from MacKinnon (1996).
#
# ADF with constant only ("c"): one-sided rejection of unit-root null when the
# t-statistic on the lagged level is less than the critical value.
_ADF_CRITICAL_VALUES_C: dict[str, float] = {
    "1%": -3.43,
    "5%": -2.86,
    "10%": -2.57,
}

# Engle-Granger residual ADF, one regressor with constant, N=2.
# More conservative than the standalone ADF because the residual is an
# estimated quantity (Phillips-Ouliaris (1990), MacKinnon (1996) Table 4).
_EG_CRITICAL_VALUES_1R: dict[str, float] = {
    "1%": -3.90,
    "5%": -3.34,
    "10%": -3.04,
}


# ----- data-prep primitives --------------------------------------------------


def compute_log_prices(prices: pd.Series) -> pd.Series:
    """Element-wise `log(P_t)` of a positive price series.

    Drops non-positive observations rather than letting `log` emit -inf / nan,
    so downstream alignment doesn't have to special-case them.
    """

    if prices.empty:
        return prices.iloc[0:0].copy()
    positive = prices[prices > 0].astype(float)
    return cast(pd.Series, np.log(positive))


def align_pair(p_a: pd.Series, p_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two price series on date index and drop any remaining NaNs."""

    combined = pd.concat([p_a.rename("a"), p_b.rename("b")], axis=1, join="inner").dropna()
    return cast(pd.Series, combined["a"].astype(float)), cast(
        pd.Series, combined["b"].astype(float)
    )


# ----- ordinary least squares ------------------------------------------------


def ols_intercept_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS regression `y = alpha + beta * x + residual`.

    Returns `(alpha, beta, residuals)`. Used both for the Engle-Granger
    cointegrating regression and (with a unit `x`) for inner ADF helpers.
    """

    if y.shape != x.shape:
        raise ValueError(f"y shape {y.shape} != x shape {x.shape}")
    if y.size < 2:
        raise ValueError(f"OLS requires at least 2 observations, got {y.size}")
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev = x - x_mean
    denom = float((x_dev * x_dev).sum())
    if denom <= 0.0:
        raise ValueError("OLS regressor has zero variance; cannot estimate slope")
    beta = float(((y - y_mean) * x_dev).sum() / denom)
    alpha = float(y_mean - beta * x_mean)
    residuals = y - alpha - beta * x
    return alpha, beta, residuals


# ----- Augmented Dickey-Fuller test ------------------------------------------


def _adf_design_matrix(
    z: np.ndarray, n_lags: int, regression: Literal["c", "nc"]
) -> tuple[np.ndarray, np.ndarray]:
    """Build the design matrix for the ADF regression.

    `delta_z_t = mu? + phi * z_{t-1} + sum_{j=1..p} psi_j delta_z_{t-j} + u_t`

    Returns `(X, dy)` where the first design-matrix column is the lagged level
    `z_{t-1}` (so its OLS coefficient and t-statistic are the ADF target).
    """

    dz = np.diff(z)
    n = dz.size - n_lags
    if n <= 0:
        raise ValueError(
            f"ADF requires more observations: have {z.size}, need > {n_lags + 2}"
        )
    # Lagged level column.
    lagged_level = z[n_lags : n_lags + n]
    cols: list[np.ndarray] = [lagged_level]
    # Lagged differences.
    for j in range(1, n_lags + 1):
        cols.append(dz[n_lags - j : n_lags - j + n])
    if regression == "c":
        cols.append(np.ones(n))
    X = np.column_stack(cols)
    dy = dz[n_lags : n_lags + n]
    return X, dy


def adf_test(
    series: np.ndarray | pd.Series,
    *,
    n_lags: int = 1,
    regression: Literal["c", "nc"] = "c",
    test_type: Literal["adf", "engle_granger"] = "adf",
) -> ADFResult:
    """Augmented Dickey-Fuller unit-root test.

    Implements the regression
        `delta Z_t = mu + phi Z_{t-1} + sum_{j=1..p} psi_j delta Z_{t-j} + u_t`
    by OLS and reports the t-statistic on `phi`.

    `test_type="adf"` uses the standalone ADF critical values (MacKinnon 1996).
    `test_type="engle_granger"` switches to the more conservative Phillips-
    Ouliaris / MacKinnon EG-residual critical values appropriate for the
    cointegration residual of a single-regressor OLS.

    The standalone ADF requires `regression="c"` for the constant-only model
    (the only model exercised here — the spec assumes a non-trending spread).
    """

    if regression not in ("c", "nc"):
        raise ValueError(f"regression must be 'c' or 'nc', got {regression!r}")
    if n_lags < 0:
        raise ValueError(f"n_lags must be non-negative, got {n_lags}")
    if test_type not in ("adf", "engle_granger"):
        raise ValueError(f"test_type must be 'adf' or 'engle_granger', got {test_type!r}")
    z = np.asarray(series, dtype=float).flatten()
    if z.size < n_lags + 3:
        raise ValueError(
            f"ADF requires at least {n_lags + 3} observations, got {z.size}"
        )

    X, dy = _adf_design_matrix(z, n_lags=n_lags, regression=regression)
    n, k = X.shape
    # Normal equations: beta = (X^T X)^-1 X^T y
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ dy
    residual = dy - X @ beta
    dof = n - k
    if dof <= 0:
        raise ValueError(
            f"ADF regression has non-positive degrees of freedom: n={n}, k={k}"
        )
    sigma_sq = float((residual @ residual) / dof)
    var_beta_0 = sigma_sq * float(XtX_inv[0, 0])
    t_stat = (
        float("nan") if var_beta_0 <= 0 else float(beta[0] / math.sqrt(var_beta_0))
    )

    critical = (
        dict(_ADF_CRITICAL_VALUES_C)
        if test_type == "adf"
        else dict(_EG_CRITICAL_VALUES_1R)
    )

    return ADFResult(
        t_stat=t_stat,
        n_obs=int(n),
        n_lags=int(n_lags),
        critical_values=critical,
        regression=regression,
        test_type=test_type,
    )


# ----- Engle-Granger two-step --------------------------------------------------


def engle_granger_test(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    n_lags: int = 1,
) -> EngleGrangerFit:
    """Run the full Engle-Granger procedure on a price pair.

    1. Standalone ADF on each leg (each should be I(1) for the test to be
       meaningful).
    2. OLS `p_A = alpha + beta * p_B + Z`.
    3. ADF on the residual `Z`, using EG critical values.
    """

    if not p_a.index.equals(p_b.index):
        raise ValueError("engle_granger_test requires aligned series")
    y = p_a.to_numpy(dtype=float)
    x = p_b.to_numpy(dtype=float)
    alpha, beta, resid = ols_intercept_slope(y, x)
    residuals = pd.Series(resid, index=p_a.index, name="spread")

    adf_a = adf_test(y, n_lags=n_lags, regression="c", test_type="adf")
    adf_b = adf_test(x, n_lags=n_lags, regression="c", test_type="adf")
    # Engle-Granger residual ADF: drop the intercept because the residual is
    # already mean-zero by construction, and use EG critical values.
    adf_resid = adf_test(resid, n_lags=n_lags, regression="nc", test_type="engle_granger")

    return EngleGrangerFit(
        alpha=alpha,
        beta=beta,
        residuals=residuals,
        adf_leg_a=adf_a,
        adf_leg_b=adf_b,
        adf_residual=adf_resid,
    )


def select_dependent_orientation(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    n_lags: int = 1,
) -> Literal["AonB", "BonA"]:
    """Pick which leg should be the dependent variable.

    Per the spec, run OLS in both directions and prefer the orientation whose
    residual ADF rejects more strongly (smallest, i.e. most negative, t-stat).
    """

    fit_ab = engle_granger_test(p_a, p_b, n_lags=n_lags)
    fit_ba = engle_granger_test(p_b, p_a, n_lags=n_lags)
    if fit_ab.adf_residual.t_stat <= fit_ba.adf_residual.t_stat:
        return "AonB"
    return "BonA"


# ----- OU / AR(1) fitting -----------------------------------------------------


def fit_ar1(z: np.ndarray | pd.Series) -> tuple[float, float, float]:
    """OLS AR(1) fit: `Z_t = c + phi Z_{t-1} + eps_t`.

    Returns `(phi, c, sigma_eps)` where `sigma_eps` is the residual standard
    error from the AR(1) regression.
    """

    arr = np.asarray(z, dtype=float).flatten()
    if arr.size < 3:
        raise ValueError(f"fit_ar1 requires >= 3 observations, got {arr.size}")
    y = arr[1:]
    x = arr[:-1]
    c, phi, resid = ols_intercept_slope(y, x)
    dof = max(y.size - 2, 1)
    sigma_eps = float(math.sqrt((resid @ resid) / dof))
    return float(phi), float(c), sigma_eps


def half_life_from_phi(phi: float, dt: float = 1.0) -> float:
    """Half-life of an AR(1) `phi`: `tau_{1/2} = -log(2) / log(phi) * dt`.

    Returns `+inf` for `phi >= 1` (non-mean-reverting) or `phi <= 0` (oscillatory
    sign flip — the OU representation breaks down).
    """

    if phi >= 1.0 or phi <= 0.0:
        return float("inf")
    return float(-math.log(2.0) / math.log(phi) * dt)


def fit_ou(z: np.ndarray | pd.Series, *, dt: float = 1.0) -> OUFit:
    """Fit continuous-time OU parameters from an AR(1) on `z`."""

    phi, c, sigma_eps = fit_ar1(z)
    half_life = half_life_from_phi(phi, dt=dt)
    if 0.0 < phi < 1.0:
        kappa = float(-math.log(phi) / dt)
        mu = float(c / (1.0 - phi)) if (1.0 - phi) > 1e-12 else float("nan")
        if kappa > 0 and sigma_eps > 0:
            # sigma_eps^2 = sigma^2 * (1 - exp(-2 kappa dt)) / (2 kappa)
            scale = (1.0 - math.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
            sigma = float(math.sqrt(sigma_eps * sigma_eps / scale)) if scale > 0 else 0.0
            sigma_eq = float(math.sqrt(sigma * sigma / (2.0 * kappa))) if kappa > 0 else 0.0
        else:
            sigma = 0.0
            sigma_eq = 0.0
    else:
        kappa = 0.0
        mu = float("nan")
        sigma = 0.0
        sigma_eq = 0.0

    return OUFit(
        phi=phi,
        c=c,
        sigma_eps=sigma_eps,
        kappa=kappa,
        mu=mu,
        sigma=sigma,
        half_life=half_life,
        sigma_eq=sigma_eq,
        dt=dt,
    )


# ----- z-score signals --------------------------------------------------------


def static_zscore(z: np.ndarray | pd.Series, mu: float, sigma: float) -> np.ndarray:
    """Standardize `z` against a fixed `(mu, sigma)`."""

    if sigma <= 0:
        raise ValueError(f"static_zscore requires sigma > 0, got {sigma}")
    arr = np.asarray(z, dtype=float)
    return (arr - mu) / sigma


def rolling_zscore_moments(
    z: pd.Series, *, window: int
) -> tuple[pd.Series, pd.Series]:
    """Trailing mean and std of `z` over the last `window` observations.

    The output series are aligned to `z.index`. Leading entries (until the
    window fills) are NaN.
    """

    if window < 2:
        raise ValueError(f"rolling_zscore_moments requires window >= 2, got {window}")
    return z.rolling(window=window, min_periods=window).mean(), z.rolling(
        window=window, min_periods=window
    ).std(ddof=1)


# ----- Kalman filter ---------------------------------------------------------


def kalman_dynamic_beta(
    p_a: np.ndarray | pd.Series,
    p_b: np.ndarray | pd.Series,
    *,
    Q: float,
    R: float,
    beta0: float,
    P0: float,
) -> KalmanFit:
    """Linear Gaussian Kalman filter for the random-walk hedge ratio.

    State `beta_t = beta_{t-1} + eta_t`, observation `p_A_t = beta_t p_B_t + eps_t`,
    where `eta_t ~ N(0, Q)` and `eps_t ~ N(0, R)`. The recursion at each step:

        predict:   beta_pred = beta_{t-1};   P_pred = P_{t-1} + Q
        innov:     y_t       = p_A_t - p_B_t * beta_pred
        var:       S_t       = p_B_t^2 P_pred + R
        gain:      K_t       = P_pred * p_B_t / S_t
        update:    beta_t    = beta_pred + K_t y_t
                   P_t       = (1 - K_t p_B_t) P_pred
    """

    if R <= 0:
        raise ValueError(f"kalman_dynamic_beta requires R > 0, got {R}")
    if Q < 0 or P0 < 0:
        raise ValueError(
            f"kalman_dynamic_beta requires Q >= 0 and P0 >= 0; got Q={Q}, P0={P0}"
        )
    a = np.asarray(p_a, dtype=float).flatten()
    b = np.asarray(p_b, dtype=float).flatten()
    if a.shape != b.shape:
        raise ValueError(f"p_a shape {a.shape} != p_b shape {b.shape}")
    n = a.size
    beta_path = np.empty(n)
    var_path = np.empty(n)
    innovations = np.empty(n)
    innovation_variance = np.empty(n)
    standardized = np.empty(n)

    beta_prev = beta0
    P_prev = P0
    for t in range(n):
        # Predict.
        beta_pred = beta_prev
        P_pred = P_prev + Q
        # Innovation.
        y_t = a[t] - b[t] * beta_pred
        S_t = b[t] * b[t] * P_pred + R
        K_t = P_pred * b[t] / S_t if S_t > 0 else 0.0
        beta_now = beta_pred + K_t * y_t
        P_now = (1.0 - K_t * b[t]) * P_pred

        beta_path[t] = beta_now
        var_path[t] = P_now
        innovations[t] = y_t
        innovation_variance[t] = S_t
        standardized[t] = y_t / math.sqrt(S_t) if S_t > 0 else 0.0

        beta_prev = beta_now
        P_prev = P_now

    return KalmanFit(
        beta_path=beta_path,
        variance_path=var_path,
        innovations=innovations,
        innovation_variance=innovation_variance,
        standardized_innovations=standardized,
        Q=float(Q),
        R=float(R),
        beta0=float(beta0),
        P0=float(P0),
    )


# ----- trading rules ----------------------------------------------------------


def pair_position_path(
    standardized_spread: np.ndarray | pd.Series,
    rule: TradingRule,
) -> np.ndarray:
    """Apply the canonical entry / exit / stop rules to a standardized series.

    Outputs `(T,)` integer positions in `{-1, 0, +1}` aligned to the input:

        position = +1   ->  long  the spread = long  A, short beta * B
        position = -1   ->  short the spread = short A, long  beta * B
        position =  0   ->  flat

    Rules (spec stage 2a, mirrored for the Kalman variant):

        if  s > +s_in   and flat:        go short the spread (-1)
        if  s < -s_in   and flat:        go long  the spread (+1)
        if |s| < s_out  and in position: close (0)
        if |s| > s_stop:                  hard stop (0)

    NaN inputs are treated as "no decision" — position carries forward.
    """

    s = np.asarray(standardized_spread, dtype=float).flatten()
    n = s.size
    pos = np.zeros(n, dtype=int)
    current = 0
    for t in range(n):
        s_t = s[t]
        if math.isnan(s_t):
            pos[t] = current
            continue
        abs_s = abs(s_t)
        if abs_s > rule.s_stop:
            current = 0
        elif current == 0:
            if s_t > rule.s_in:
                current = -1
            elif s_t < -rule.s_in:
                current = +1
        else:
            if abs_s < rule.s_out:
                current = 0
        pos[t] = current
    return pos


# ----- diagnostics ------------------------------------------------------------


def ljung_box_pvalue(residual_series: np.ndarray | pd.Series, max_lag: int = 10) -> float:
    """Ljung-Box Q-statistic p-value against H0 = no autocorrelation.

    Implemented inline to avoid pulling scipy as a dependency. Uses a chi-square
    survival approximation via the Wilson-Hilferty transform — accurate to
    better than 0.01 in the tail region we care about (p < 0.1).
    """

    x = np.asarray(residual_series, dtype=float).flatten()
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
    k = max_lag
    if q_stat <= 0:
        return 1.0
    z = ((q_stat / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def rolling_eg_residual_tstat(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    window: int,
    step: int = 1,
    n_lags: int = 1,
) -> pd.Series:
    """Rolling Engle-Granger residual ADF t-statistic.

    For each window of size `window` (advancing by `step`), re-runs OLS +
    residual ADF and records the t-statistic. The output series is indexed by
    the *last* date in each window, so it lines up with the live trading clock.

    Used by the validation step to track stationarity stability over time
    (spec validation step 2).
    """

    if window < 30:
        raise ValueError(f"rolling_eg_residual_tstat requires window >= 30, got {window}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if not p_a.index.equals(p_b.index):
        raise ValueError("rolling_eg_residual_tstat requires aligned series")
    out_idx: list[pd.Timestamp] = []
    out_vals: list[float] = []
    n = len(p_a)
    for end in range(window, n + 1, step):
        start = end - window
        try:
            fit = engle_granger_test(
                p_a.iloc[start:end],
                p_b.iloc[start:end],
                n_lags=n_lags,
            )
        except ValueError:
            continue
        out_idx.append(p_a.index[end - 1])
        out_vals.append(fit.adf_residual.t_stat)
    return pd.Series(out_vals, index=pd.Index(out_idx), name="eg_tstat")


def realized_half_life(spread: pd.Series, *, max_lag: int = 200) -> float:
    """Empirical half-life: lag at which the autocorrelation drops below 0.5.

    Compared with the AR(1) half-life from `fit_ou`, this provides a sanity
    check that the AR(1) representation matches what the spread actually does
    in sample (spec validation step 6).

    Returns `+inf` if the autocorrelation never drops below 0.5 within
    `max_lag`.
    """

    x = spread.to_numpy(dtype=float)
    n = x.size
    if n < 4:
        return float("inf")
    x_c = x - x.mean()
    denom = float((x_c * x_c).sum())
    if denom <= 0:
        return float("inf")
    upper = min(max_lag, n - 1)
    for lag in range(1, upper + 1):
        rho = float((x_c[lag:] * x_c[:-lag]).sum()) / denom
        if rho < 0.5:
            return float(lag)
    return float("inf")


__all__ = [
    "adf_test",
    "align_pair",
    "compute_log_prices",
    "engle_granger_test",
    "fit_ar1",
    "fit_ou",
    "half_life_from_phi",
    "kalman_dynamic_beta",
    "ljung_box_pvalue",
    "ols_intercept_slope",
    "pair_position_path",
    "realized_half_life",
    "rolling_eg_residual_tstat",
    "rolling_zscore_moments",
    "select_dependent_orientation",
    "static_zscore",
]
