"""Pure-math layer for the GARCH-family model.

Everything here is `numpy`/`pandas`-only — no I/O, no class state. The
`GARCHModel` orchestration shell composes these functions to produce
`Signal` / `Forecast` / `RiskMetric` outputs.

All variance / volatility numbers in this module are in **percent-return**
units (returns are multiplied by 100 before fitting; this rescales the
likelihood for numerical stability per the `arch` package convention).
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from src.models.garch_family.types import (
    GARCHFit,
    GARCHParams,
    GARCHSpec,
    InnovationDist,
)

# Returns floor: variances are clipped above this to avoid log(0) / div-by-0
# blow-ups inside the recursion when the optimizer probes an extreme region.
_MIN_SIGMA2: float = 1e-12
_LOG_SIGMA2_FLOOR: float = math.log(_MIN_SIGMA2)
_LOG_SIGMA2_CEIL: float = math.log(1e8)
_TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Return preprocessing
# ---------------------------------------------------------------------------


def log_returns_pct(close: pd.Series) -> pd.Series:
    """Convert a Close-price series to log returns in percent units.

    `100 * log(P_t / P_{t-1})`. The arch package's convention; rescales the
    likelihood so omega does not collapse to ~1e-8 and break the optimizer.
    """

    if close is None or close.empty:
        return pd.Series(dtype="float64")
    s = close.astype(float).sort_index()
    log_s = pd.Series(np.log(s.to_numpy()), index=s.index)
    r = 100.0 * log_s.diff().dropna()
    return cast(pd.Series, r)


# ---------------------------------------------------------------------------
# Innovation distributions
# ---------------------------------------------------------------------------


def normal_cdf(x: float | np.ndarray) -> np.ndarray:
    """Standard-normal CDF using `math.erf` (no scipy dependency)."""

    arr = np.asarray(x, dtype=float)
    # vectorized erf via numpy: numpy >= 2.0 exposes np.special.erf only in some
    # builds; we fall back to math.erf on a python loop for safety.
    out = np.empty_like(arr, dtype=float)
    flat = arr.ravel()
    out_flat = out.ravel()
    for i, v in enumerate(flat):
        out_flat[i] = 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
    return out


def normal_ppf(q: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 over the open interval (0, 1). scipy-free.
    """

    if not 0.0 < q < 1.0:
        raise ValueError(f"normal_ppf requires q in (0, 1), got {q}")
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if q < plow:
        u = math.sqrt(-2.0 * math.log(q))
        return (
            (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5])
            / ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0)
        )
    if q > phigh:
        u = math.sqrt(-2.0 * math.log(1.0 - q))
        return -(
            (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5])
            / ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0)
        )
    u = q - 0.5
    r = u * u
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * u
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def expected_abs_z_gaussian() -> float:
    """`E|z|` under standard normal: `sqrt(2/pi)`."""

    return math.sqrt(2.0 / math.pi)


def expected_abs_z_student_t(nu: float) -> float:
    """`E|z|` under the standardized Student-t with `nu > 2` degrees of freedom.

    Closed form from spec:
        E|z| = 2 * sqrt(nu - 2) * Gamma((nu+1)/2)
             / ( sqrt(pi) * (nu - 1) * Gamma(nu/2) )

    Evaluated as `2 * sqrt(nu-2) * exp(lgamma((nu+1)/2) - lgamma(nu/2))
    / (sqrt(pi) * (nu-1))` so that `math.gamma` does not overflow for
    large nu (math.gamma(170+) is already past the float64 ceiling).
    """

    if nu <= 2.0:
        raise ValueError(f"expected_abs_z_student_t requires nu > 2, got {nu}")
    log_ratio = _lgamma_half_step(nu / 2.0)
    return 2.0 * math.sqrt(nu - 2.0) * math.exp(log_ratio) / (
        math.sqrt(math.pi) * (nu - 1.0)
    )


def student_t_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    """Log-pdf of the standardized Student-t (unit variance, mean zero).

    Uses a stable form of `lgamma((nu+1)/2) - lgamma(nu/2)`: subtracting two
    huge lgammas directly loses all precision when `nu >> 100`. We switch to
    the Stirling-series expansion `0.5*log(x) - 1/(8x)` (x = nu/2) above the
    crossover threshold; below it the direct evaluation is fine.
    """

    if nu <= 2.0:
        raise ValueError(f"student_t_logpdf requires nu > 2, got {nu}")
    log_gamma_diff = _lgamma_half_step(nu / 2.0)  # lgamma((nu+1)/2) - lgamma(nu/2)
    c = log_gamma_diff - 0.5 * math.log(math.pi * (nu - 2.0))
    out = c - 0.5 * (nu + 1.0) * np.log1p(z * z / (nu - 2.0))
    return np.asarray(out, dtype=float)


def _lgamma_half_step(x: float) -> float:
    """`lgamma(x + 0.5) - lgamma(x)` with full precision for any `x > 0`.

    For small `x` the direct subtraction is accurate; for large `x` the
    Stirling expansion `0.5*log(x) - 1/(8x) + 1/(192*x^3) - ...` is far
    more stable than subtracting two near-equal lgammas.
    """

    if x < 1e6:
        return math.lgamma(x + 0.5) - math.lgamma(x)
    inv = 1.0 / x
    return 0.5 * math.log(x) - 0.125 * inv + inv * inv * inv / 192.0


def student_t_ppf(q: float, nu: float) -> float:
    """Inverse CDF of the standardized Student-t. Bisection on the CDF.

    Slower than scipy but quite adequate for VaR cutoffs (one call per
    horizon). Standardized variant: divide a raw-t quantile by `sqrt(nu/(nu-2))`.
    """

    if not 0.0 < q < 1.0:
        raise ValueError(f"student_t_ppf requires q in (0, 1), got {q}")
    if nu <= 2.0:
        raise ValueError(f"student_t_ppf requires nu > 2, got {nu}")
    # Raw-t quantile via bisection on the regularized incomplete beta.
    # F_t(x) = 1 - 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2)        for x >= 0
    # Symmetric: F_t(-x) = 1 - F_t(x).
    target = q
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = _student_t_cdf_raw(mid, nu)
        if cdf < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    x_raw = 0.5 * (lo + hi)
    scale = math.sqrt(nu / (nu - 2.0))
    return x_raw / scale


def _student_t_cdf_raw(x: float, nu: float) -> float:
    """CDF of the raw (unit-scale) Student-t. Uses a series for the
    regularized incomplete beta; accurate enough for VaR cutoffs.
    """

    if x == 0.0:
        return 0.5
    sign = 1.0 if x > 0 else -1.0
    z = nu / (nu + x * x)
    ibeta = _regularized_incomplete_beta(z, nu / 2.0, 0.5)
    cdf_pos = 1.0 - 0.5 * ibeta
    return cdf_pos if sign > 0 else 1.0 - cdf_pos


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """`I_x(a, b)` via continued fraction (Lentz). Same approach as Numerical
    Recipes 6.4 — accurate to ~1e-12 in the regimes we use it.
    """

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            return h
    return h


# ---------------------------------------------------------------------------
# Conditional-variance recursions
# ---------------------------------------------------------------------------


def compute_sigma2_path(
    returns: np.ndarray,
    params: GARCHParams,
    spec: GARCHSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-pass the variance recursion.

    Returns
    -------
    sigma2 : `(T,)` array of conditional variances.
    eps    : `(T,)` array of mean-subtracted returns.
    """

    r = np.asarray(returns, dtype=float)
    t = r.shape[0]
    if t < 2:
        raise ValueError(f"compute_sigma2_path requires T >= 2, got {t}")
    eps = r - params.mu
    sigma2 = np.empty(t, dtype=float)

    # Initial variance: sample variance of the first 30 obs (or all if shorter).
    init_window = min(30, t)
    init_var = float(np.var(eps[:init_window], ddof=0))
    if not np.isfinite(init_var) or init_var <= 0:
        init_var = max(float(np.var(eps, ddof=0)), _MIN_SIGMA2)
    sigma2[0] = max(init_var, _MIN_SIGMA2)

    if spec == "GARCH":
        omega, alpha, beta = params.omega, params.alpha, params.beta
        for i in range(1, t):
            sigma2[i] = omega + alpha * eps[i - 1] ** 2 + beta * sigma2[i - 1]
            if not np.isfinite(sigma2[i]) or sigma2[i] < _MIN_SIGMA2:
                sigma2[i] = _MIN_SIGMA2
    elif spec == "GJR":
        omega, alpha, gamma, beta = params.omega, params.alpha, params.gamma, params.beta
        for i in range(1, t):
            e2 = eps[i - 1] ** 2
            ind = 1.0 if eps[i - 1] < 0.0 else 0.0
            sigma2[i] = omega + alpha * e2 + gamma * ind * e2 + beta * sigma2[i - 1]
            if not np.isfinite(sigma2[i]) or sigma2[i] < _MIN_SIGMA2:
                sigma2[i] = _MIN_SIGMA2
    elif spec == "EGARCH":
        omega, alpha, gamma, beta = params.omega, params.alpha, params.gamma, params.beta
        # E|z| under the assumed innovation distribution: passed implicitly
        # through `params.nu` (None -> Gaussian, otherwise Student-t).
        e_abs_z = (
            expected_abs_z_gaussian()
            if params.nu is None
            else expected_abs_z_student_t(params.nu)
        )
        log_sigma2 = np.empty(t, dtype=float)
        log_sigma2[0] = math.log(sigma2[0])
        for i in range(1, t):
            sigma_prev = math.sqrt(sigma2[i - 1])
            z = eps[i - 1] / sigma_prev if sigma_prev > 0 else 0.0
            log_s2 = (
                omega
                + alpha * (abs(z) - e_abs_z)
                + gamma * z
                + beta * log_sigma2[i - 1]
            )
            # Clip in log-space for numerical sanity.
            if log_s2 < _LOG_SIGMA2_FLOOR:
                log_s2 = _LOG_SIGMA2_FLOOR
            elif log_s2 > _LOG_SIGMA2_CEIL:
                log_s2 = _LOG_SIGMA2_CEIL
            log_sigma2[i] = log_s2
            sigma2[i] = math.exp(log_s2)
    else:
        raise ValueError(f"Unknown GARCH spec: {spec!r}")

    return sigma2, eps


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------


def negloglik(
    returns: np.ndarray,
    params: GARCHParams,
    spec: GARCHSpec,
    distribution: InnovationDist,
) -> float:
    """Negative log-likelihood at `params`. Used directly by the optimizer.

    Returns `+inf` when the recursion produces a non-finite path (e.g. the
    optimizer wandered into an unstable region).
    """

    try:
        sigma2, eps = compute_sigma2_path(returns, params, spec)
    except Exception:  # noqa: BLE001
        return math.inf
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return math.inf

    log_sigma2 = np.log(sigma2)
    sigma = np.sqrt(sigma2)
    z = eps / sigma

    if distribution == "Gaussian":
        # ll_t = -0.5 (log 2pi + log sigma2 + eps^2 / sigma2)
        ll = -0.5 * np.sum(np.log(2.0 * math.pi) + log_sigma2 + (eps * eps) / sigma2)
    elif distribution == "Student-t":
        if params.nu is None or params.nu <= 2.0:
            return math.inf
        ll = float(np.sum(student_t_logpdf(z, params.nu) - 0.5 * log_sigma2))
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    if not np.isfinite(ll):
        return math.inf
    return float(-ll)


# ---------------------------------------------------------------------------
# Multi-step forecasts
# ---------------------------------------------------------------------------


def forecast_variance(
    fit: GARCHFit,
    horizons: tuple[int, ...],
) -> np.ndarray:
    """Variance forecasts at each horizon `h` in `horizons` (>= 1).

    Implementation uses the GARCH(1,1) closed form when applicable and
    falls back to iterating the recursion for GJR and EGARCH (no clean
    closed form once asymmetry enters).
    """

    if any(h < 1 for h in horizons):
        raise ValueError(f"forecast horizons must be >= 1, got {horizons}")
    h_max = max(horizons)
    params = fit.params
    eps_last = float(fit.eps_path[-1])
    sigma2_last = float(fit.sigma2_path[-1])

    if fit.spec == "GARCH":
        omega, alpha, beta = params.omega, params.alpha, params.beta
        # One-step forecast uses the *current* innovation, not the recursion-only
        # form (which would give sigma2_{T+1} from sigma2_T alone).
        sigma2_next = omega + alpha * eps_last * eps_last + beta * sigma2_last
        persistence = alpha + beta
        unc = omega / (1.0 - persistence) if persistence < 1.0 else None
        path = np.empty(h_max, dtype=float)
        path[0] = sigma2_next
        for i in range(1, h_max):
            if unc is not None:
                path[i] = unc + persistence ** i * (sigma2_next - unc)
            else:
                # IGARCH limit: forecasts are flat at sigma2_next.
                path[i] = sigma2_next
    elif fit.spec == "GJR":
        # Iterate analytically using E[I_{eps<0} eps^2 / sigma^2] = 1/2 under
        # symmetric standardized innovations. Then E[sigma2_{t+1} | F_{t-1}]
        # uses (alpha + gamma/2 + beta) as effective persistence.
        omega, alpha, gamma, beta = params.omega, params.alpha, params.gamma, params.beta
        ind = 1.0 if eps_last < 0.0 else 0.0
        sigma2_next = (
            omega
            + alpha * eps_last * eps_last
            + gamma * ind * eps_last * eps_last
            + beta * sigma2_last
        )
        persistence = alpha + 0.5 * gamma + beta
        unc = omega / (1.0 - persistence) if persistence < 1.0 else None
        path = np.empty(h_max, dtype=float)
        path[0] = sigma2_next
        for i in range(1, h_max):
            if unc is not None:
                path[i] = unc + persistence ** i * (sigma2_next - unc)
            else:
                path[i] = sigma2_next
    elif fit.spec == "EGARCH":
        # No closed form once asymmetry enters; iterate the recursion taking
        # E[|z|] under the assumed innovation distribution and E[z] = 0.
        omega, alpha, gamma, beta = params.omega, params.alpha, params.gamma, params.beta
        e_abs_z = (
            expected_abs_z_gaussian()
            if params.nu is None
            else expected_abs_z_student_t(params.nu)
        )
        sigma_last = math.sqrt(sigma2_last)
        z_last = eps_last / sigma_last if sigma_last > 0 else 0.0
        log_sigma2_curr = math.log(sigma2_last)
        # one-step:
        log_sigma2_next = (
            omega
            + alpha * (abs(z_last) - e_abs_z)
            + gamma * z_last
            + beta * log_sigma2_curr
        )
        path = np.empty(h_max, dtype=float)
        path[0] = math.exp(log_sigma2_next)
        # For h >= 2: E[alpha (|z| - E|z|)] = 0 and E[gamma z] = 0, so
        # log_sigma2_{T+h} ~ omega + beta * log_sigma2_{T+h-1} in expectation
        # (Jensen gap ignored). Iterate.
        log_curr = log_sigma2_next
        for i in range(1, h_max):
            log_curr = omega + beta * log_curr
            if log_curr < _LOG_SIGMA2_FLOOR:
                log_curr = _LOG_SIGMA2_FLOOR
            elif log_curr > _LOG_SIGMA2_CEIL:
                log_curr = _LOG_SIGMA2_CEIL
            path[i] = math.exp(log_curr)
    else:
        raise ValueError(f"Unknown GARCH spec: {fit.spec!r}")

    return np.asarray([path[h - 1] for h in horizons], dtype=float)


def annualize_vol_pct(sigma_daily_pct: float) -> float:
    """Daily-pct volatility -> annualized decimal volatility.

    `sigma_pct` is in percent-return units. Multiply by `sqrt(252)` and divide
    by 100 to land in decimal annualized form (e.g. 0.18 = 18% per year).
    """

    return float(sigma_daily_pct * math.sqrt(_TRADING_DAYS_PER_YEAR) / 100.0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def standardized_residuals(fit: GARCHFit) -> np.ndarray:
    """`z_t = eps_t / sigma_t`. Spec validation step 7."""

    return fit.std_resid


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic and approximate p-value.

    Q ~ chi-squared(`lags`) under the null of no autocorrelation. The
    p-value is computed by a series for the regularized lower incomplete
    gamma function so we avoid a scipy dependency.
    """

    arr = np.asarray(x, dtype=float)
    arr = arr - arr.mean()
    n = arr.size
    if n <= lags + 1:
        return (math.nan, math.nan)
    denom = float(np.sum(arr * arr))
    if denom <= 0:
        return (math.nan, math.nan)
    q = 0.0
    for k in range(1, lags + 1):
        num = float(np.sum(arr[:-k] * arr[k:]))
        rho_k = num / denom
        q += rho_k * rho_k / (n - k)
    q *= n * (n + 2)
    # p-value: 1 - CDF of chi-squared(lags) at q.
    p = 1.0 - _chi2_cdf(q, lags)
    return (float(q), float(p))


def arch_lm_test(z: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Engle's ARCH-LM test on the squared standardized residuals.

    Regress `z_t^2` on `[1, z_{t-1}^2, ..., z_{t-p}^2]`; the test stat is
    `n * R^2` and is chi-squared(`lags`) under the null of no ARCH effect.
    """

    arr = np.asarray(z, dtype=float)
    sq = arr * arr
    n = sq.size
    if n <= lags + 1:
        return (math.nan, math.nan)
    y = sq[lags:]
    x = np.empty((y.size, lags + 1), dtype=float)
    x[:, 0] = 1.0
    for k in range(1, lags + 1):
        x[:, k] = sq[lags - k : n - k]
    # OLS via lstsq, R^2 from residuals.
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return (math.nan, math.nan)
    r2 = 1.0 - ss_res / ss_tot
    lm = y.size * max(r2, 0.0)
    p = 1.0 - _chi2_cdf(lm, lags)
    return (float(lm), float(p))


def sign_bias_test(
    std_resid: np.ndarray, eps: np.ndarray
) -> dict[str, float]:
    """Engle-Ng (1993) joint sign-bias test.

    Regress `z_t^2` on `[1, I(eps_{t-1}<0), I(eps_{t-1}<0) * eps_{t-1},
    I(eps_{t-1}>=0) * eps_{t-1}]`. The joint F-statistic on the three slopes
    flags missing asymmetry; significant individual slopes pinpoint which
    asymmetry (size vs sign) is mis-specified.
    """

    z = np.asarray(std_resid, dtype=float)
    e = np.asarray(eps, dtype=float)
    n = z.size
    if n < 20 or e.size != n:
        return {
            "joint_F": math.nan,
            "joint_p": math.nan,
            "sign_bias": math.nan,
            "neg_size_bias": math.nan,
            "pos_size_bias": math.nan,
        }
    y = z[1:] ** 2
    eprev = e[:-1]
    ind_neg = (eprev < 0.0).astype(float)
    ind_pos = 1.0 - ind_neg
    x = np.column_stack(
        [
            np.ones_like(y),
            ind_neg,
            ind_neg * eprev,
            ind_pos * eprev,
        ]
    )
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # Restricted model: intercept only.
    ss_res_r = ss_tot
    df_full = y.size - 4
    df_diff = 3
    if df_full <= 0 or ss_res <= 0:
        return {
            "joint_F": math.nan,
            "joint_p": math.nan,
            "sign_bias": float(coef[1]) if coef.size > 1 else math.nan,
            "neg_size_bias": float(coef[2]) if coef.size > 2 else math.nan,
            "pos_size_bias": float(coef[3]) if coef.size > 3 else math.nan,
        }
    f_stat = ((ss_res_r - ss_res) / df_diff) / (ss_res / df_full)
    p_val = 1.0 - _f_cdf(f_stat, df_diff, df_full)
    return {
        "joint_F": float(f_stat),
        "joint_p": float(p_val),
        "sign_bias": float(coef[1]),
        "neg_size_bias": float(coef[2]),
        "pos_size_bias": float(coef[3]),
    }


def mincer_zarnowitz_regression(
    realized_var: np.ndarray, predicted_var: np.ndarray
) -> dict[str, float]:
    """OLS of `RV_t` on `[1, sigma_hat_t^2]`. Well-calibrated forecasts have
    intercept zero and slope one.
    """

    y = np.asarray(realized_var, dtype=float)
    x_full = np.column_stack([np.ones_like(y), np.asarray(predicted_var, dtype=float)])
    coef, *_ = np.linalg.lstsq(x_full, y, rcond=None)
    y_hat = x_full @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "r_squared": float(r2),
    }


def value_at_risk(
    sigma_pct: float,
    mu_pct: float,
    alpha: float,
    distribution: InnovationDist,
    nu: float | None,
) -> float:
    """Conditional one-step VaR at level `alpha` (e.g. 0.05).

    Returns the *loss* threshold in **percent** units (sign convention: a
    positive value means "you stand to lose at least this much, with
    probability `alpha`").
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"VaR alpha must lie in (0, 1), got {alpha}")
    if distribution == "Gaussian":
        q = normal_ppf(alpha)
    elif distribution == "Student-t":
        if nu is None or nu <= 2.0:
            raise ValueError("Student-t VaR requires nu > 2")
        q = student_t_ppf(alpha, nu)
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")
    return -(mu_pct + sigma_pct * q)


def kupiec_pof_test(failures: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures unconditional-coverage test.

    `LR = -2 log( ((1-a)^(n-x) a^x) / ((1-p)^(n-x) p^x) )` is asymptotically
    chi-squared(1). Returns (LR, p-value).
    """

    if n <= 0:
        return (math.nan, math.nan)
    x = failures
    p = x / n
    if x == 0:
        ll_unrestricted = (n - x) * math.log(1.0 - p) if (1.0 - p) > 0 else 0.0
    elif x == n:
        ll_unrestricted = x * math.log(p)
    else:
        ll_unrestricted = x * math.log(p) + (n - x) * math.log(1.0 - p)
    ll_null = x * math.log(alpha) + (n - x) * math.log(1.0 - alpha)
    lr = -2.0 * (ll_null - ll_unrestricted)
    return (float(lr), float(1.0 - _chi2_cdf(lr, 1)))


def qlike_loss(realized_var: np.ndarray, predicted_var: np.ndarray) -> float:
    """`QLIKE = mean(RV/sigma_hat^2 - log(RV/sigma_hat^2) - 1)`.

    Robust to RV proxy noise (Patton 2011). Spec validation step 7 (oos).
    """

    rv = np.asarray(realized_var, dtype=float)
    sv = np.asarray(predicted_var, dtype=float)
    if rv.shape != sv.shape:
        raise ValueError(
            f"qlike_loss: shape mismatch ({rv.shape} vs {sv.shape})"
        )
    mask = (rv > 0) & (sv > 0) & np.isfinite(rv) & np.isfinite(sv)
    if not np.any(mask):
        return math.nan
    ratio = rv[mask] / sv[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


# ---------------------------------------------------------------------------
# Distribution helpers (chi-squared, F) used by the diagnostic p-values
# ---------------------------------------------------------------------------


def _chi2_cdf(x: float, k: int) -> float:
    """CDF of chi-squared with `k` degrees of freedom via the regularized
    lower incomplete gamma function. Accurate enough for diagnostic p-values.
    """

    if x <= 0:
        return 0.0
    return _regularized_lower_incomplete_gamma(k / 2.0, x / 2.0)


def _regularized_lower_incomplete_gamma(a: float, x: float) -> float:
    """`P(a, x) = gamma(a, x) / Gamma(a)`. Series for small x, continued
    fraction for large x, per Numerical Recipes.
    """

    if x < 0 or a <= 0:
        return math.nan
    if x == 0:
        return 0.0
    if x < a + 1.0:
        # series expansion
        ap = a
        sum_ = 1.0 / a
        delta = sum_
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            sum_ += delta
            if abs(delta) < abs(sum_) * 3e-12:
                break
        return sum_ * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def _f_cdf(x: float, d1: int, d2: int) -> float:
    """CDF of the F-distribution with (`d1`, `d2`) dof at `x >= 0`.

    `F(x; d1, d2) = I_{(d1 x) / (d1 x + d2)}(d1/2, d2/2)`.
    """

    if x <= 0:
        return 0.0
    z = (d1 * x) / (d1 * x + d2)
    return _regularized_incomplete_beta(z, d1 / 2.0, d2 / 2.0)


__all__ = [
    "annualize_vol_pct",
    "arch_lm_test",
    "compute_sigma2_path",
    "expected_abs_z_gaussian",
    "expected_abs_z_student_t",
    "forecast_variance",
    "kupiec_pof_test",
    "ljung_box",
    "log_returns_pct",
    "mincer_zarnowitz_regression",
    "negloglik",
    "normal_cdf",
    "normal_ppf",
    "qlike_loss",
    "sign_bias_test",
    "standardized_residuals",
    "student_t_logpdf",
    "student_t_ppf",
    "value_at_risk",
]
