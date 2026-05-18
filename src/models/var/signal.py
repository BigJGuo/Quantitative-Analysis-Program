"""Pure-math layer for the Value-at-Risk model.

Everything here is ``numpy``/``pandas``-only — no I/O, no class state. The
`VaRModel` orchestration shell composes these functions to produce the
public `RiskMetric` output and the validation surface.

Units convention:

- Asset returns are **decimal simple returns** (``P_t / P_{t-1} - 1``),
  matching the spec's ``pct_change()`` recipe.
- Positions are **dollar notionals**: a signed scalar per ticker.
- Resulting VaR figures are in **dollars** (always reported as a positive
  number; loss = ``-P&L``).

We re-derive Acklam's rational normal-PPF approximation and the
chi-squared CDF here so this module has no scipy dependency and no
intra-`src/models` coupling.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from src.models.var.types import TrafficLight, VaRMethod

# Floor on portfolio variance to avoid div-by-zero when positions cancel out.
_MIN_PORTFOLIO_VAR: float = 1e-18


# ---------------------------------------------------------------------------
# Return preprocessing
# ---------------------------------------------------------------------------


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a panel of close prices to decimal simple returns.

    Equivalent to ``prices.pct_change().dropna()`` but stable when columns
    have leading NaNs from non-overlapping listing dates: each column is
    differenced over its own non-null history before the panel is realigned.
    """

    if prices is None or prices.empty:
        return pd.DataFrame(dtype=float)
    sorted_prices = prices.sort_index().astype(float)
    rets = sorted_prices.pct_change()
    return rets.dropna(how="any")


def align_positions(
    positions: Mapping[str, float], columns: Iterable[str]
) -> tuple[np.ndarray, list[str]]:
    """Order ``positions`` to match the return-panel column order.

    Returns the dollar-position vector ``w`` aligned with ``columns`` (a
    1-D ``np.ndarray``) and the column list it was aligned against.
    Tickers in ``columns`` missing from ``positions`` raise ``KeyError``.
    """

    cols = list(columns)
    w = np.empty(len(cols), dtype=float)
    for i, c in enumerate(cols):
        if c not in positions:
            raise KeyError(
                f"align_positions: ticker {c!r} appears in returns but not "
                f"in positions."
            )
        w[i] = float(positions[c])
    return w, cols


# ---------------------------------------------------------------------------
# Distribution primitives
# ---------------------------------------------------------------------------


def normal_ppf(q: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 over the open interval ``(0, 1)``. scipy-free.
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


def chi2_cdf(x: float, k: int) -> float:
    """CDF of chi-squared with ``k`` degrees of freedom at ``x``.

    Uses the regularized lower incomplete gamma function (series for small
    ``x``, continued fraction for large ``x``). Accurate to ~1e-10 in the
    range we care about for diagnostic p-values.
    """

    if x <= 0.0:
        return 0.0
    return _reg_lower_incomplete_gamma(k / 2.0, x / 2.0)


def _reg_lower_incomplete_gamma(a: float, x: float) -> float:
    """``P(a, x) = gamma(a, x) / Gamma(a)``.

    Numerical Recipes 6.2 — series for ``x < a + 1``, continued fraction
    otherwise.
    """

    if x < 0 or a <= 0:
        return math.nan
    if x == 0:
        return 0.0
    if x < a + 1.0:
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


# ---------------------------------------------------------------------------
# Portfolio P&L
# ---------------------------------------------------------------------------


def portfolio_pnl_vector(
    returns: np.ndarray, w_dollar: np.ndarray
) -> np.ndarray:
    """Empirical P&L vector ``rets @ w_dollar`` in **dollars**.

    Parameters
    ----------
    returns:
        ``(N, n)`` matrix of decimal simple returns.
    w_dollar:
        ``(n,)`` dollar position vector aligned with ``returns`` columns.

    Returns
    -------
    pnl:
        ``(N,)`` array. Positive = gain, negative = loss.
    """

    r = np.asarray(returns, dtype=float)
    w = np.asarray(w_dollar, dtype=float)
    if r.ndim != 2:
        raise ValueError(
            f"portfolio_pnl_vector: returns must be 2-D, got shape {r.shape}"
        )
    if w.ndim != 1 or w.size != r.shape[1]:
        raise ValueError(
            f"portfolio_pnl_vector: w shape {w.shape} incompatible with "
            f"returns shape {r.shape}"
        )
    return np.asarray(r @ w, dtype=float)


# ---------------------------------------------------------------------------
# Three VaR methods
# ---------------------------------------------------------------------------


def parametric_var(
    returns: np.ndarray,
    w_dollar: np.ndarray,
    alpha: float,
    *,
    include_mean: bool = True,
) -> tuple[float, float, float, np.ndarray]:
    """Variance-covariance VaR under joint-Gaussian returns.

    Closed-form: ``VaR = -mu_p + sigma_p * z_alpha`` with ``z_alpha =
    Phi^{-1}(alpha)``. Returns are mapped to portfolio dollar space via
    ``mu_p = w_dollar^T mean(rets)`` and ``sigma_p = sqrt(w_dollar^T Sigma
    w_dollar)``.

    Returns
    -------
    var_1d_dollar:
        Positive scalar.
    mu_p_dollar:
        Portfolio mean P&L.
    sigma_p_dollar:
        Portfolio P&L standard deviation.
    sigma_matrix:
        ``(n, n)`` sample asset-return covariance.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"parametric_var: alpha must lie in (0, 1), got {alpha}")
    r = np.asarray(returns, dtype=float)
    w = np.asarray(w_dollar, dtype=float)
    if r.shape[0] < 2:
        raise ValueError(
            f"parametric_var requires >= 2 return rows, got {r.shape[0]}"
        )

    sigma_matrix = np.cov(r, rowvar=False, ddof=1)
    # np.cov returns a 0-d array when n_assets == 1; force 2-D.
    sigma_matrix = np.atleast_2d(sigma_matrix)

    mu_assets = r.mean(axis=0)
    mu_p = float(w @ mu_assets) if include_mean else 0.0
    portfolio_var = float(w @ sigma_matrix @ w)
    if portfolio_var < _MIN_PORTFOLIO_VAR:
        portfolio_var = _MIN_PORTFOLIO_VAR
    sigma_p = math.sqrt(portfolio_var)
    z = normal_ppf(alpha)
    var_1d = -mu_p + sigma_p * z
    # Numerical guard: tiny negative results when mu_p dominates sigma_p (e.g.
    # very low alpha + strong drift). Clip at zero — VaR is non-negative.
    if var_1d < 0.0:
        var_1d = 0.0
    return float(var_1d), float(mu_p), float(sigma_p), sigma_matrix


def historical_var(
    returns: np.ndarray, w_dollar: np.ndarray, alpha: float
) -> tuple[float, np.ndarray]:
    """Historical-simulation VaR.

    Empirical ``alpha``-quantile of the loss series ``-P&L`` where
    ``P&L = rets @ w_dollar``. Uses ``numpy.quantile`` with the default
    linear interpolation, which matches the spec's recipe.

    Returns
    -------
    var_1d_dollar:
        Empirical VaR in dollars (non-negative; clipped at 0 if the loss
        distribution sits entirely below zero, which can happen for
        deeply long-biased books with strong positive drift).
    pnl_vector:
        The underlying P&L vector, ``(N,)``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"historical_var: alpha must lie in (0, 1), got {alpha}")
    pnl = portfolio_pnl_vector(returns, w_dollar)
    loss = -pnl
    q = float(np.quantile(loss, alpha))
    if q < 0.0:
        q = 0.0
    return q, pnl


def monte_carlo_var(
    returns: np.ndarray,
    w_dollar: np.ndarray,
    alpha: float,
    *,
    n_samples: int = 50_000,
    seed: int | None = None,
    include_mean: bool = True,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Monte Carlo VaR under a multivariate Gaussian generator.

    Draws ``n_samples`` jointly-Gaussian asset return vectors with the
    sample mean and covariance, revalues the portfolio, and reports the
    empirical ``alpha``-quantile of simulated losses.

    Returns
    -------
    var_1d_dollar:
        MC VaR in dollars.
    mu_p_dollar:
        Sample portfolio mean P&L (in dollars).
    sigma_p_dollar:
        Sample portfolio P&L standard deviation (analytic; same as
        ``sqrt(w^T Sigma w)``).
    sigma_matrix:
        ``(n, n)`` sample asset-return covariance.
    pnl_sim:
        ``(n_samples,)`` simulated portfolio P&L.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"monte_carlo_var: alpha must lie in (0, 1), got {alpha}")
    if n_samples < 1:
        raise ValueError(
            f"monte_carlo_var: n_samples must be >= 1, got {n_samples}"
        )
    r = np.asarray(returns, dtype=float)
    w = np.asarray(w_dollar, dtype=float)
    if r.shape[0] < 2:
        raise ValueError(
            f"monte_carlo_var requires >= 2 return rows, got {r.shape[0]}"
        )

    mu_assets = r.mean(axis=0) if include_mean else np.zeros(r.shape[1])
    sigma_matrix = np.atleast_2d(np.cov(r, rowvar=False, ddof=1))
    chol = _safe_cholesky(sigma_matrix)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, r.shape[1]))
    sim_rets = mu_assets + z @ chol.T  # (K, n)
    pnl_sim = sim_rets @ w  # (K,)
    loss_sim = -pnl_sim
    var_1d = float(np.quantile(loss_sim, alpha))
    if var_1d < 0.0:
        var_1d = 0.0
    mu_p = float(w @ mu_assets) if include_mean else 0.0
    portfolio_var = float(w @ sigma_matrix @ w)
    sigma_p = math.sqrt(max(portfolio_var, _MIN_PORTFOLIO_VAR))
    return var_1d, mu_p, sigma_p, sigma_matrix, pnl_sim


def _safe_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Cholesky factor with a small ridge fallback for near-singular ``Sigma``.

    Adds ``1e-12 * I`` and retries if the first factorization fails. Used
    by the MC sampler so almost-singular covariance (e.g. duplicated
    columns from highly-correlated tickers) doesn't crash the run.
    """

    a = np.asarray(sigma, dtype=float)
    a = 0.5 * (a + a.T)
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        n = a.shape[0]
        ridge = 1e-12 * float(np.trace(a)) / max(n, 1)
        if ridge <= 0:
            ridge = 1e-12
        return np.linalg.cholesky(a + ridge * np.eye(n))


# ---------------------------------------------------------------------------
# Horizon scaling
# ---------------------------------------------------------------------------


def horizon_scale(var_1d: float, horizon_days: int) -> float:
    """Apply iid Gaussian sqrt-h scaling: ``VaR_h = sqrt(h) * VaR_1``.

    The spec calls this out as an approximation: it is exact under iid
    Gaussian increments and wrong under volatility clustering. Filtered
    historical simulation (Barone-Adesi et al. 1999) replaces this with
    a GARCH-implied cumulative-variance forecast; the README documents
    that as a known limitation.
    """

    if horizon_days < 1:
        raise ValueError(
            f"horizon_scale: horizon_days must be >= 1, got {horizon_days}"
        )
    return float(var_1d) * math.sqrt(horizon_days)


# ---------------------------------------------------------------------------
# Validation: Kupiec, Christoffersen, Basel traffic-light
# ---------------------------------------------------------------------------


def kupiec_pof_test(failures: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures unconditional-coverage test.

    Likelihood-ratio statistic against the null that the breach rate
    equals ``1 - alpha``; asymptotically chi-squared with 1 dof.

    Returns ``(LR, p_value)``. Edge cases:
    - ``n <= 0`` -> ``(nan, nan)``.
    - ``failures == 0`` -> ``LR = -2 n log(alpha)`` (degenerate but well-defined).
    """

    if n <= 0:
        return (math.nan, math.nan)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"kupiec_pof_test: alpha must lie in (0, 1), got {alpha}")
    if failures < 0 or failures > n:
        raise ValueError(
            f"kupiec_pof_test: failures must lie in [0, n], got {failures}/{n}"
        )
    p_null = 1.0 - alpha  # expected breach rate
    p_hat = failures / n
    if failures == 0:
        ll_unrestricted = n * math.log(1.0 - p_hat) if (1.0 - p_hat) > 0 else 0.0
    elif failures == n:
        ll_unrestricted = n * math.log(p_hat)
    else:
        ll_unrestricted = (
            failures * math.log(p_hat) + (n - failures) * math.log(1.0 - p_hat)
        )
    ll_null = failures * math.log(p_null) + (n - failures) * math.log(1.0 - p_null)
    lr = -2.0 * (ll_null - ll_unrestricted)
    p_value = 1.0 - chi2_cdf(lr, 1)
    return float(lr), float(p_value)


def christoffersen_independence_test(
    breach_indicators: np.ndarray,
) -> tuple[float, float]:
    """Christoffersen (1998) independence likelihood-ratio test.

    Tests whether consecutive VaR breaches are serially dependent —
    breaches clustering in time is a classic GARCH-effect / regime-change
    symptom even when the unconditional rate looks correct.

    Define indicators ``I_t = 1{loss_t > VaR_t}`` and transition counts
    ``n_ij`` for ``(I_{t-1}, I_t)``. Estimate ``pi_ij = n_ij / (n_i0 +
    n_i1)``, the null being ``pi_01 = pi_11 = pi``. The likelihood-ratio
    statistic is asymptotically chi-squared with 1 dof.

    Returns ``(LR, p_value)``. Edge cases:
    - too few breaches to estimate transition probabilities -> ``(0.0, 1.0)``.
    """

    ind = np.asarray(breach_indicators, dtype=int).ravel()
    if ind.size < 2:
        return (math.nan, math.nan)
    if not set(np.unique(ind).tolist()).issubset({0, 1}):
        raise ValueError(
            "christoffersen_independence_test: breach_indicators must be "
            "0/1 only."
        )
    prev = ind[:-1]
    curr = ind[1:]
    n_00 = int(np.sum((prev == 0) & (curr == 0)))
    n_01 = int(np.sum((prev == 0) & (curr == 1)))
    n_10 = int(np.sum((prev == 1) & (curr == 0)))
    n_11 = int(np.sum((prev == 1) & (curr == 1)))
    n_0 = n_00 + n_01
    n_1 = n_10 + n_11
    if n_0 == 0 or n_1 == 0:
        # Not enough variation in the indicator series; the test is
        # degenerate. Return ``(0, 1)`` — "no evidence of dependence".
        return (0.0, 1.0)
    pi_01 = n_01 / n_0
    pi_11 = n_11 / n_1
    pi = (n_01 + n_11) / (n_0 + n_1)
    # log-likelihoods using safe-log (skip terms where the probability is 0).
    ll_full = _safe_xlogp(n_00, 1.0 - pi_01) + _safe_xlogp(n_01, pi_01)
    ll_full += _safe_xlogp(n_10, 1.0 - pi_11) + _safe_xlogp(n_11, pi_11)
    ll_null = _safe_xlogp(n_00 + n_10, 1.0 - pi) + _safe_xlogp(n_01 + n_11, pi)
    lr = -2.0 * (ll_null - ll_full)
    if lr < 0.0:
        # Numerical noise when the unrestricted MLE is on the boundary.
        lr = 0.0
    p_value = 1.0 - chi2_cdf(lr, 1)
    return float(lr), float(p_value)


def _safe_xlogp(count: int, p: float) -> float:
    if count == 0:
        return 0.0
    if p <= 0.0:
        return -math.inf
    return count * math.log(p)


def basel_traffic_light(failures: int) -> TrafficLight:
    """Basel traffic-light bucket at ``T = 250`` days, ``alpha = 0.99``.

    Per BCBS (1996, 2009) Annex 10a:
    - **Green**: 0-4 breaches (no capital penalty).
    - **Yellow**: 5-9 breaches (penalty scalar 0.40-0.85).
    - **Red**: >= 10 breaches (additive penalty + mandatory review).

    The thresholds are calibrated for ``T = 250`` and a 1% nominal breach
    rate; the bucket is informative outside that window but no longer the
    regulator's signal.
    """

    if failures < 0:
        raise ValueError(
            f"basel_traffic_light: failures must be >= 0, got {failures}"
        )
    if failures <= 4:
        return "green"
    if failures <= 9:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------


def rolling_var_backtest(
    returns: np.ndarray,
    w_dollar: np.ndarray,
    alpha: float,
    *,
    method: VaRMethod = "historical",
    window: int = 250,
    mc_samples: int = 10_000,
    mc_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-step VaR backtest under a fixed-size rolling window.

    For each ``t in [window, N)`` recompute VaR from the trailing
    ``window`` return rows, then compare to the realized loss
    ``-(returns[t] @ w_dollar)``.

    Parameters
    ----------
    returns:
        ``(N, n)`` decimal-return matrix.
    w_dollar:
        ``(n,)`` dollar position vector.
    alpha:
        Confidence level.
    method:
        VaR engine. The default is ``"historical"`` because parametric
        backtests are highly sensitive to the Gaussian assumption and MC
        backtests are slow.
    window:
        Length of the rolling estimation window in days.
    mc_samples / mc_seed:
        Forwarded to ``monte_carlo_var`` when ``method == "monte_carlo"``.

    Returns
    -------
    var_path:
        ``(N - window,)`` series of VaR cutoffs.
    loss_path:
        ``(N - window,)`` series of realized losses at the same indices.
    breach_indicators:
        ``(N - window,)`` ``0/1`` array.
    """

    r = np.asarray(returns, dtype=float)
    w = np.asarray(w_dollar, dtype=float)
    n_rows = r.shape[0]
    if window < 30:
        raise ValueError(f"rolling_var_backtest: window must be >= 30, got {window}")
    if n_rows <= window:
        raise ValueError(
            f"rolling_var_backtest: need n_rows > window; got "
            f"{n_rows} <= {window}"
        )
    n_out = n_rows - window
    var_path = np.empty(n_out, dtype=float)
    loss_path = np.empty(n_out, dtype=float)

    for i in range(n_out):
        train = r[i : i + window]
        if method == "parametric":
            v, *_ = parametric_var(train, w, alpha)
        elif method == "historical":
            v, _ = historical_var(train, w, alpha)
        elif method == "monte_carlo":
            v, *_ = monte_carlo_var(
                train, w, alpha, n_samples=mc_samples, seed=mc_seed
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")
        var_path[i] = v
        loss_path[i] = -float(r[i + window] @ w)
    breach = (loss_path > var_path).astype(int)
    return var_path, loss_path, breach


__all__ = [
    "align_positions",
    "basel_traffic_light",
    "chi2_cdf",
    "christoffersen_independence_test",
    "historical_var",
    "horizon_scale",
    "kupiec_pof_test",
    "monte_carlo_var",
    "normal_ppf",
    "parametric_var",
    "portfolio_pnl_vector",
    "rolling_var_backtest",
    "simple_returns",
]
