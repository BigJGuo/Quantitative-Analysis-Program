"""Pure-math layer for the Expected Shortfall (FRTB) risk model.

Everything here is `numpy`/`pandas`-only — no I/O, no class state. Closed-form
ES under Gaussian and Student-t marginals, the empirical historical-simulation
estimator, and the Monte-Carlo multivariate-t simulator all live here, along
with the Acerbi-Szekely Z1 / Z2 backtests.

Numerical helpers (`normal_ppf`, `normal_pdf`, `student_t_ppf`,
`student_t_pdf`) are reimplemented locally to keep the ES module
self-contained — they are pure functions and the alternative of importing
from `garch_family` would create a structural dependency on an unrelated
model's load order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Standard-normal helpers (scipy-free)
# ---------------------------------------------------------------------------


def normal_pdf(x: float) -> float:
    """`phi(x)` — standard-normal density at `x`."""

    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    """`Phi(x)` — standard-normal CDF via `math.erf`."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(q: float) -> float:
    """`Phi^{-1}(q)` — Acklam's rational approximation, accurate to ~1e-9."""

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


# ---------------------------------------------------------------------------
# Raw (non-standardized) Student-t pdf / cdf / ppf
# ---------------------------------------------------------------------------


def student_t_pdf_raw(x: float, nu: float) -> float:
    """Density of the unit-scale Student-t with `nu` degrees of freedom."""

    if nu <= 0.0:
        raise ValueError(f"student_t_pdf_raw requires nu > 0, got {nu}")
    log_c = (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
    )
    log_kernel = -0.5 * (nu + 1.0) * math.log1p(x * x / nu)
    return math.exp(log_c + log_kernel)


def student_t_cdf_raw(x: float, nu: float) -> float:
    """CDF of the unit-scale Student-t. Regularized incomplete beta (Lentz)."""

    if nu <= 0.0:
        raise ValueError(f"student_t_cdf_raw requires nu > 0, got {nu}")
    if x == 0.0:
        return 0.5
    z = nu / (nu + x * x)
    ibeta = _regularized_incomplete_beta(z, 0.5 * nu, 0.5)
    cdf_pos = 1.0 - 0.5 * ibeta
    return cdf_pos if x > 0 else 1.0 - cdf_pos


def student_t_ppf_raw(q: float, nu: float) -> float:
    """Inverse CDF of the unit-scale Student-t via bisection."""

    if not 0.0 < q < 1.0:
        raise ValueError(f"student_t_ppf_raw requires q in (0, 1), got {q}")
    if nu <= 0.0:
        raise ValueError(f"student_t_ppf_raw requires nu > 0, got {nu}")
    lo, hi = -1.0e3, 1.0e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf_raw(mid, nu) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """`I_x(a, b)` via continued fraction (Numerical Recipes 6.4)."""

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
# Student-t MLE on a 1-D sample (scipy-free)
# ---------------------------------------------------------------------------


def fit_student_t(
    x: np.ndarray, nu_grid: Sequence[float] | None = None, max_em_iter: int = 200
) -> tuple[float, float, float]:
    """Fit a location-scale Student-t to `x` via profile MLE.

    Returns ``(nu, loc, scale)`` such that `x ~ loc + scale * T_nu` (i.e.,
    `scale` is the standard deviation of the underlying unit Student-t, **not**
    the standard deviation of `x`; `Var(x) = scale^2 * nu/(nu-2)`).

    Strategy: for each candidate `nu`, run a small EM loop to convergence to
    refit `(loc, scale)`, score the joint log-likelihood, and pick the
    `(nu, loc, scale)` with the highest likelihood. Robust to outliers
    (heavy tails are the whole reason we use the t in the first place).
    """

    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 20:
        raise ValueError(
            f"fit_student_t needs >= 20 samples, got {arr.size}"
        )
    grid = (
        tuple(nu_grid)
        if nu_grid is not None
        else (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
    )
    best = (-math.inf, 5.0, float(arr.mean()), float(arr.std(ddof=1)))
    for nu in grid:
        loc, scale = _em_t_fixed_nu(arr, nu, max_iter=max_em_iter)
        ll = _student_t_loglik(arr, nu, loc, scale)
        if ll > best[0]:
            best = (ll, nu, loc, scale)
    _, nu_hat, loc_hat, scale_hat = best
    return float(nu_hat), float(loc_hat), float(scale_hat)


def _em_t_fixed_nu(
    x: np.ndarray, nu: float, max_iter: int = 200, tol: float = 1e-8
) -> tuple[float, float]:
    """EM for (loc, scale) under fixed `nu` via scale-mixture of normals."""

    mu = float(x.mean())
    sigma = float(x.std(ddof=1)) or 1.0
    for _ in range(max_iter):
        z2 = ((x - mu) / sigma) ** 2
        w = (nu + 1.0) / (nu + z2)
        wx = w * x
        new_mu = wx.sum() / w.sum()
        new_var = (w * (x - new_mu) ** 2).sum() / x.size
        new_sigma = math.sqrt(max(new_var, 1e-30))
        if abs(new_mu - mu) < tol and abs(new_sigma - sigma) < tol:
            mu, sigma = new_mu, new_sigma
            break
        mu, sigma = new_mu, new_sigma
    return mu, sigma


def _student_t_loglik(x: np.ndarray, nu: float, loc: float, scale: float) -> float:
    z = (x - loc) / scale
    log_c = (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
        - math.log(scale)
    )
    log_kernel = -0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    return float(log_c * x.size + log_kernel.sum())


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------


def portfolio_pnl(returns: pd.DataFrame, dollar_positions: np.ndarray) -> np.ndarray:
    """Dollar P&L vector = `returns @ dollar_positions`. NaNs dropped upstream."""

    if returns.shape[1] != dollar_positions.shape[0]:
        raise ValueError(
            f"returns columns ({returns.shape[1]}) must match dollar_positions "
            f"length ({dollar_positions.shape[0]})"
        )
    return np.asarray(returns.values @ dollar_positions, dtype=float)


def apply_liquidity_scaling(
    returns: pd.DataFrame, scalars: np.ndarray | None
) -> pd.DataFrame:
    """Multiply per-ticker returns by `sqrt(h_k / 10)` for FRTB liquidity
    horizons. `scalars=None` is the identity (standard one-day ES).
    """

    if scalars is None:
        return returns
    if scalars.shape != (returns.shape[1],):
        raise ValueError(
            f"scalars shape {scalars.shape} must match returns column count "
            f"{returns.shape[1]}"
        )
    return returns * scalars[np.newaxis, :]


# ---------------------------------------------------------------------------
# Closed-form parametric ES
# ---------------------------------------------------------------------------


def es_gaussian(mu_p: float, sigma_p: float, alpha: float) -> tuple[float, float]:
    """Parametric Gaussian ``(VaR, ES)`` for a single P&L distribution.

    `mu_p, sigma_p` are the **loss** distribution's mean and standard
    deviation in dollar units. Spec boxed formula::

        VaR = mu_p + sigma_p * Phi^{-1}(alpha)
        ES  = mu_p + sigma_p * phi(z_alpha) / (1 - alpha)
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if sigma_p < 0.0:
        raise ValueError(f"sigma_p must be non-negative, got {sigma_p}")
    z = normal_ppf(alpha)
    var = mu_p + sigma_p * z
    es = mu_p + sigma_p * normal_pdf(z) / (1.0 - alpha)
    return var, es


def es_student_t(
    loc: float, scale: float, nu: float, alpha: float
) -> tuple[float, float]:
    """Parametric Student-t ``(VaR, ES)`` for a single loss distribution.

    Convention: ``loss ~ loc + scale * T_nu`` (raw, unit-scale Student-t).
    Closed form::

        VaR = loc + scale * T_nu^{-1}(alpha)
        ES  = loc + scale * f_nu(t_alpha)/(1-alpha) * (nu + t_alpha^2)/(nu - 1)
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if nu <= 1.0:
        raise ValueError(f"Student-t ES requires nu > 1, got {nu}")
    t_a = student_t_ppf_raw(alpha, nu)
    f_a = student_t_pdf_raw(t_a, nu)
    var = loc + scale * t_a
    es = loc + scale * (f_a / (1.0 - alpha)) * (nu + t_a * t_a) / (nu - 1.0)
    return var, es


# ---------------------------------------------------------------------------
# Empirical ES (historical simulation)
# ---------------------------------------------------------------------------


def es_historical(losses: np.ndarray, alpha: float) -> tuple[float, float, np.ndarray]:
    """Historical-simulation ``(VaR, ES, tail)`` from a loss vector.

    `losses` is positive-for-loss. Spec definition::

        VaR_hat = quantile(losses, alpha)
        ES_hat  = mean(losses[losses > VaR_hat])

    If no observation exceeds the empirical quantile (can happen at small
    `N` with ties at the cutoff), fall back to the `>= VaR` mean so the
    estimator is always defined.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    arr = np.asarray(losses, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("es_historical requires a non-empty loss vector")
    var = float(np.quantile(arr, alpha))
    tail = arr[arr > var]
    if tail.size == 0:
        tail = arr[arr >= var]
    es = float(tail.mean())
    return var, es, tail


# ---------------------------------------------------------------------------
# Monte Carlo ES (multivariate-t via Gaussian / chi-squared mixture)
# ---------------------------------------------------------------------------


def es_monte_carlo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    dollar_positions: np.ndarray,
    alpha: float,
    *,
    K: int = 100_000,
    nu: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray]:
    """Multivariate-t Monte-Carlo ``(VaR, ES, tail_losses)``.

    Spec algorithm: simulate `K` draws of asset returns from a multivariate-t
    with the given empirical mean/covariance and degrees of freedom, then
    apply `dollar_positions` to land on a P&L draw and take the empirical tail
    mean. The Student-t is built via the standard Gaussian/chi-squared scale
    mixture: ``Z ~ N(0, Sigma)``, ``g ~ chi2(nu)/nu``, ``X = mu + Z / sqrt(g)``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if nu <= 2.0:
        raise ValueError(f"Monte-Carlo nu must be > 2 (else variance is infinite), got {nu}")
    if K < 1000:
        raise ValueError(f"Monte-Carlo K must be >= 1000 for stable tails, got {K}")
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma shape {Sigma.shape} does not match mu length {n}")
    if dollar_positions.shape != (n,):
        raise ValueError(
            f"dollar_positions shape {dollar_positions.shape} does not match N={n}"
        )
    if rng is None:
        rng = np.random.default_rng()
    # Cholesky; nudge the diagonal if the empirical covariance is
    # numerically not-quite-PSD.
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eps = 1e-10 * float(np.trace(Sigma) / max(n, 1))
        L = np.linalg.cholesky(Sigma + eps * np.eye(n))
    g = rng.chisquare(nu, size=K) / nu
    Z = rng.standard_normal(size=(K, n))
    sim_rets = mu[np.newaxis, :] + (Z @ L.T) / np.sqrt(g)[:, np.newaxis]
    pnl = sim_rets @ dollar_positions
    losses = -pnl
    var = float(np.quantile(losses, alpha))
    tail = losses[losses > var]
    if tail.size == 0:
        tail = losses[losses >= var]
    es = float(tail.mean())
    return var, es, tail


# ---------------------------------------------------------------------------
# Acerbi-Szekely backtests
# ---------------------------------------------------------------------------


def acerbi_szekely_z1(
    losses: np.ndarray, var_path: np.ndarray, es_path: np.ndarray
) -> float:
    """`Z1 = mean_{t : L_t > VaR_t} (L_t / ES_t) - 1`.

    Under the null (correctly-specified ES on breach days), `E[Z1] = 0`. A
    positive value flags ES under-prediction.
    """

    losses = np.asarray(losses, dtype=float)
    var_path = np.asarray(var_path, dtype=float)
    es_path = np.asarray(es_path, dtype=float)
    if losses.shape != var_path.shape or losses.shape != es_path.shape:
        raise ValueError(
            f"acerbi_szekely_z1 input shapes must match; got "
            f"{losses.shape}, {var_path.shape}, {es_path.shape}"
        )
    breaches = losses > var_path
    if not breaches.any():
        return float("nan")
    return float((losses[breaches] / es_path[breaches]).mean() - 1.0)


def acerbi_szekely_z2(
    losses: np.ndarray, var_path: np.ndarray, es_path: np.ndarray, alpha: float
) -> float:
    """`Z2 = sum_t (L_t * I_t) / (T (1-alpha) ES_t) - 1`.

    Integrates over the full sample (not just breach days) for more power.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    losses = np.asarray(losses, dtype=float)
    var_path = np.asarray(var_path, dtype=float)
    es_path = np.asarray(es_path, dtype=float)
    T = losses.size
    if T == 0:
        return float("nan")
    if var_path.shape != losses.shape or es_path.shape != losses.shape:
        raise ValueError(
            f"acerbi_szekely_z2 input shapes must match; got "
            f"{losses.shape}, {var_path.shape}, {es_path.shape}"
        )
    breaches = (losses > var_path).astype(float)
    contrib = (losses * breaches) / es_path
    return float(contrib.sum() / (T * (1.0 - alpha)) - 1.0)


def traffic_light_band(es_var_ratio: float) -> str:
    """Map an `ES / VaR` ratio to a Basel-style traffic-light band.

    For a Gaussian portfolio at alpha=0.975, ES/VaR ~ 1.21. Ratios that drift
    upward signal fattening tails (Student-t with low `nu` pushes ES/VaR
    materially higher); ratios below ~1.10 suggest model under-statement
    relative to a Gaussian assumption.
    """

    if not math.isfinite(es_var_ratio):
        return "undefined"
    if es_var_ratio < 1.10:
        return "amber-light"
    if es_var_ratio < 1.35:
        return "green"
    if es_var_ratio < 1.75:
        return "amber"
    return "red"


# ---------------------------------------------------------------------------
# Convenience: rolling realized tail-mean for diagnostic plots
# ---------------------------------------------------------------------------


def rolling_realized_tail_mean(
    losses: np.ndarray, window: int, alpha: float
) -> np.ndarray:
    """Rolling realized tail-mean of `losses` over windows of size `window`.

    For each rolling block ending at index `t`, computes the empirical tail
    mean above the in-window `alpha`-quantile. NaN-padded at the start.
    """

    if window < 30:
        raise ValueError(f"window must be >= 30 for a stable tail-mean, got {window}")
    arr = np.asarray(losses, dtype=float).ravel()
    out = np.full(arr.shape, float("nan"))
    for t in range(window - 1, arr.size):
        block = arr[t - window + 1 : t + 1]
        q = np.quantile(block, alpha)
        tail = block[block > q]
        if tail.size == 0:
            tail = block[block >= q]
        out[t] = tail.mean()
    return out
