"""Pure-math layer for Kyle-lambda and the square-root impact law.

Functions in this module are deliberately stateless and operate on
``numpy`` / ``pandas`` inputs only. Every routine here corresponds to a
formula spelled out in ``models/layer5_execution/15_kyle_lambda_sqrt_impact.md``
so the tests can pin them to worked-out values.

The module exposes three layers:

1. **Kyle linear-impact equilibrium**: closed-form ``lambda``, ``beta``,
   and aggregate-flow variance from ``sigma_v`` and ``sigma_u`` (the textbook
   single-period model used for unit-test ground truth).
2. **Empirical Kyle-lambda regression**: signed-volume construction, OLS
   slope-only regression, HC1 standard error, R^2, Wald 95% CI.
3. **Square-root impact law**: relative impact, bps and dollar cost,
   Q*-crossover with the linear-Kyle estimate.

All functions raise ``ValueError`` on malformed inputs and return finite
floats wherever the spec guarantees finiteness.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Z critical value for 95% two-sided Wald CI on a Gaussian-asymptotic
# regression coefficient. The spec specifies HC1 + 95% CI for the lambda fit.
_Z_95: float = 1.959963984540054

# Cap on participation ratio Q/V; above this the square-root law is no
# longer empirically calibrated (spec limitation #2).
_QV_EXTRAPOLATION_THRESHOLD: float = 0.10


# ---------------------------------------------------------------------------
# Kyle (1985) closed-form equilibrium
# ---------------------------------------------------------------------------


def kyle_equilibrium_lambda(sigma_v: float, sigma_u: float) -> float:
    """Closed-form Kyle lambda from the single-period model.

    ``lambda = sigma_v / (2 * sigma_u)`` (see spec derivation sketch).
    """

    if sigma_v <= 0:
        raise ValueError(f"sigma_v must be > 0, got {sigma_v}")
    if sigma_u <= 0:
        raise ValueError(f"sigma_u must be > 0, got {sigma_u}")
    return sigma_v / (2.0 * sigma_u)


def kyle_equilibrium_beta(sigma_v: float, sigma_u: float) -> float:
    """Closed-form informed-trader aggressiveness ``beta = sigma_u / sigma_v``."""

    if sigma_v <= 0:
        raise ValueError(f"sigma_v must be > 0, got {sigma_v}")
    if sigma_u <= 0:
        raise ValueError(f"sigma_u must be > 0, got {sigma_u}")
    return sigma_u / sigma_v


# ---------------------------------------------------------------------------
# Daily Lee-Ready proxy
# ---------------------------------------------------------------------------


def compute_signed_volume(
    bars: pd.DataFrame,
    *,
    dollar: bool = False,
) -> pd.Series:
    """Daily / intraday Lee-Ready proxy ``sign(C - O) * V``.

    The Lee-Ready tick rule classifies trades as buys / sells based on the
    price relative to the previous trade. With OHLCV bars (no tick data),
    the spec uses ``sign(Close - Open)`` as the bar-level proxy.

    Bars where ``Close == Open`` contribute zero signed volume (sign is
    treated as zero). The resulting Series shares the index of ``bars``.

    Parameters
    ----------
    bars:
        DataFrame with at least ``Open``, ``Close``, ``Volume`` columns.
    dollar:
        If ``True`` returns dollar-signed volume ``sign(C - O) * C * V``
        instead of share-signed volume.
    """

    for col in ("Open", "Close", "Volume"):
        if col not in bars.columns:
            raise ValueError(f"bars missing required column {col!r}")
    o = bars["Open"].astype(float)
    c = bars["Close"].astype(float)
    v = bars["Volume"].astype(float)
    sgn = pd.Series(np.sign(c.to_numpy() - o.to_numpy()), index=bars.index)
    signed_qty = sgn * v
    if dollar:
        signed_qty = signed_qty * c
    signed_qty.name = "signed_volume"
    return signed_qty


def compute_log_returns(bars: pd.DataFrame) -> pd.Series:
    """``log(C_t / C_{t-1})`` with the first row dropped."""

    if "Close" not in bars.columns:
        raise ValueError("bars missing required column 'Close'")
    c = bars["Close"].astype(float)
    ratio = c / c.shift(1)
    out = pd.Series(np.log(ratio.to_numpy()), index=bars.index, name="log_return")
    return out


# ---------------------------------------------------------------------------
# OLS slope + HC1 standard error
# ---------------------------------------------------------------------------


def ols_slope_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    """``lambda_hat = (x.T y) / (x.T x)`` for the no-intercept regression.

    Equivalent to ``numpy.linalg.lstsq(x[:, None], y)[0][0]`` but explicit
    so unit tests can read it.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x shape {x.shape} does not match y shape {y.shape}")
    if x.size < 2:
        raise ValueError(f"need >= 2 observations, got {x.size}")
    xtx = float(np.dot(x, x))
    if xtx <= 0.0:
        raise ValueError("regressor has zero variance (xtx <= 0)")
    return float(np.dot(x, y) / xtx)


def hc1_se_no_intercept(x: np.ndarray, y: np.ndarray, beta_hat: float) -> float:
    """HC1 (White) heteroskedasticity-robust SE for the slope-only OLS.

    For the 1-regressor model ``y = beta * x + eps`` the sandwich estimator
    collapses to::

        Var_HC1(beta) = (n / (n - 1)) * (sum(x_i^2 * e_i^2)) / (sum(x_i^2))^2

    where ``e_i = y_i - beta_hat * x_i``. The leading ``n/(n-1)`` is the
    HC1 small-sample correction relative to HC0.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError(f"need >= 2 observations, got {n}")
    resid = y - beta_hat * x
    xtx = float(np.dot(x, x))
    if xtx <= 0.0:
        raise ValueError("regressor has zero variance (xtx <= 0)")
    meat = float(np.sum((x * resid) ** 2))
    var_beta = (n / (n - 1.0)) * meat / (xtx * xtx)
    return math.sqrt(max(var_beta, 0.0))


def r_squared_no_intercept(x: np.ndarray, y: np.ndarray, beta_hat: float) -> float:
    """Uncentered R^2 for the no-intercept regression.

    Without an intercept, the conventional centered R^2 is not bounded in
    ``[0, 1]``; the uncentered ``1 - SSR / sum(y^2)`` is. This is the value
    most software (statsmodels, R's ``lm(y ~ 0 + x)``) reports for
    no-intercept fits and the one referenced in the spec validation
    section.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ss_tot = float(np.dot(y, y))
    if ss_tot <= 0.0:
        return float("nan")
    resid = y - beta_hat * x
    ss_res = float(np.dot(resid, resid))
    return 1.0 - ss_res / ss_tot


def wald_ci_95(beta_hat: float, se: float) -> tuple[float, float]:
    """Two-sided Wald 95% CI ``beta_hat +/- 1.96 * se``."""

    if se < 0.0:
        raise ValueError(f"se must be >= 0, got {se}")
    half = _Z_95 * se
    return beta_hat - half, beta_hat + half


# ---------------------------------------------------------------------------
# Square-root impact law
# ---------------------------------------------------------------------------


def sqrt_law_impact(
    sigma_daily: float,
    Q: float,
    V: float,
    *,
    Y: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Empirical square-root law: ``Delta P / P = Y * sigma * (Q / V) ** alpha``.

    Spec gives ``alpha = 0.5`` and a literature prior of ``Y ~ 1`` for US
    large-cap equities. ``alpha`` is exposed only so the calibrated-power
    variant from the spec ("many papers also fit a power") can be plugged
    in if an execution dataset is available.

    Returns the unsigned relative impact ``Delta P / P``.
    """

    if sigma_daily < 0:
        raise ValueError(f"sigma_daily must be >= 0, got {sigma_daily}")
    if Q <= 0:
        raise ValueError(f"Q must be > 0, got {Q}")
    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")
    if Y <= 0:
        raise ValueError(f"Y must be > 0, got {Y}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    return float(Y * sigma_daily * (Q / V) ** alpha)


def participation_ratio_exceeds_threshold(Q: float, V: float) -> bool:
    """``Q / V > 0.10`` — the spec extrapolation threshold."""

    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")
    return Q / V > _QV_EXTRAPOLATION_THRESHOLD


def crossover_size(lambda_hat: float, Y: float, sigma_daily: float, V: float) -> float:
    """Order size where linear-Kyle and sqrt-law impacts agree.

    Solves ``lambda * Q = Y * sigma * sqrt(Q / V)`` (i.e. the spec
    Algorithm D crossover): ``Q* = Y^2 * sigma^2 / (lambda^2 * V)``.

    Returns ``+inf`` when ``lambda_hat = 0`` (linear regime is always
    cheaper, sqrt never kicks in).
    """

    if Y <= 0:
        raise ValueError(f"Y must be > 0, got {Y}")
    if sigma_daily < 0:
        raise ValueError(f"sigma_daily must be >= 0, got {sigma_daily}")
    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")
    if lambda_hat == 0.0:
        return math.inf
    return (Y * Y * sigma_daily * sigma_daily) / (lambda_hat * lambda_hat * V)


def linear_impact(lambda_hat: float, Q: float) -> float:
    """Linear-Kyle absolute impact ``|lambda * Q|`` (unsigned)."""

    return abs(lambda_hat * Q)


# ---------------------------------------------------------------------------
# Volatility and ADV helpers
# ---------------------------------------------------------------------------


def realized_daily_vol(closes: pd.Series) -> float:
    """Sample stdev of close-to-close simple returns. Spec default for sigma."""

    if len(closes) < 2:
        raise ValueError(f"need >= 2 closes for realized vol, got {len(closes)}")
    pct = closes.astype(float).pct_change().dropna()
    if pct.empty:
        raise ValueError("close-to-close returns are all NaN")
    return float(pct.std(ddof=1))


def adv(volumes: pd.Series, *, window: int = 20) -> float:
    """20-day average daily volume by default (spec)."""

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(volumes) < 1:
        raise ValueError("volumes is empty")
    v = volumes.astype(float).dropna()
    if len(v) == 0:
        raise ValueError("volumes are all NaN")
    return float(v.tail(window).to_numpy().mean())
