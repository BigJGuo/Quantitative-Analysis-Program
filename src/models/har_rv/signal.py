"""Pure math layer of the HAR-RV model.

Nothing in this module performs I/O. The functions follow the spec's
Algorithm Outline:

- `compute_intraday_log_returns`, `compute_daily_rv`,
  `compute_realized_semivariance`, `compute_bipower_variation`        -> step 1
- `compute_garman_klass_rv`, `compute_yang_zhang_rv`                  -> Strategy C
- `aggregate_har_components`                                          -> step 2
- `build_har_design_matrix`, `ols_fit`, `newey_west_covariance`,
  `nw_optimal_lag`                                                    -> step 3
- `forecast_one_step`, `forecast_multi_step_direct`                   -> step 4, 5
- `mincer_zarnowitz`, `qlike_loss`, `ljung_box_pvalue`,
  `arch_lm_pvalue`, `diebold_mariano_pvalue`                          -> step 6
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from src.models.har_rv.types import HAR_COEF_NAMES, HARFit, HARSpec, RVComponents

DEFAULT_W_WINDOW: int = 5
DEFAULT_M_WINDOW: int = 22
DEFAULT_MIN_RV: float = 1e-12  # floor so log(RV) doesn't blow up


# ---------------------------------------------------------------------------
# Intraday-return helpers
# ---------------------------------------------------------------------------


def compute_intraday_log_returns(
    bars: pd.DataFrame, *, price_col: str = "Close"
) -> pd.DataFrame:
    """Append a per-bar log return + trading-date column to an intraday frame.

    The returned DataFrame has columns ``log_return`` and ``date`` in addition
    to whatever was in ``bars``. The first bar of every trading day gets a
    NaN ``log_return`` (no prior intraday observation that day) — caller is
    responsible for either replacing this with an overnight return or
    dropping it before summing into RV.
    """

    if bars is None or bars.empty:
        return pd.DataFrame(columns=list(bars.columns) + ["log_return", "date"])
    if price_col not in bars.columns:
        raise KeyError(
            f"compute_intraday_log_returns: price column {price_col!r} not "
            f"in bars (available: {list(bars.columns)})"
        )

    idx = _datetime_index(bars)
    out = bars.copy()
    out.index = idx
    out["date"] = idx.date
    out = out.sort_index()
    log_price = pd.Series(
        np.log(out[price_col].astype(float).to_numpy()),
        index=out.index,
        name="log_price",
    )
    out["log_return"] = log_price.groupby(out["date"]).diff()
    return out


def compute_daily_rv(
    bars: pd.DataFrame,
    *,
    price_col: str = "Close",
    include_overnight: bool = True,
) -> RVComponents:
    """Build the four daily variance series from intraday OHLCV bars.

    Returns an `RVComponents` with `rv_d`, `bipower`, `rv_minus`, `rv_plus`
    indexed by trading date. The optional ``include_overnight`` flag follows
    the most-common Andersen-Bollerslev convention (a): add a single
    overnight return (`log(C_t_open) - log(C_{t-1}_close)`) into the day's
    sum-of-squares.
    """

    enriched = compute_intraday_log_returns(bars, price_col=price_col)
    if enriched.empty:
        return RVComponents(
            rv_d=pd.Series(dtype="float64", name="rv_d"),
            bipower=pd.Series(dtype="float64", name="bipower"),
            rv_minus=pd.Series(dtype="float64", name="rv_minus"),
            rv_plus=pd.Series(dtype="float64", name="rv_plus"),
            source="intraday",
            n_intraday_bars=None,
        )

    intraday = enriched.dropna(subset=["log_return"]).copy()
    grouped_dates = intraday["date"]

    daily_intraday_rv: pd.Series = (
        (intraday["log_return"] ** 2).groupby(grouped_dates).sum()
    )

    if include_overnight:
        # First bar of each day: log(C_first / C_lastClose_yday)
        log_price_series = pd.Series(
            np.log(enriched[price_col].astype(float).to_numpy()),
            index=enriched.index,
            name="log_price",
        )
        first_open = log_price_series.groupby(enriched["date"]).first()
        last_close = log_price_series.groupby(enriched["date"]).last()
        overnight = first_open - last_close.shift(1)
        overnight_sq = (overnight.fillna(0.0)) ** 2
        rv_d = daily_intraday_rv.add(overnight_sq, fill_value=0.0)
    else:
        rv_d = daily_intraday_rv

    rv_d = pd.Series(
        rv_d.to_numpy(dtype=float),
        index=pd.to_datetime(list(rv_d.index)),
        name="rv_d",
    ).sort_index()

    # Semivariances (intraday-only — overnight return is signed and would
    # need to be added with its sign; we keep semivariance intraday-only to
    # stay aligned with Barndorff-Nielsen-Kinnebrock-Shephard).
    pos = intraday.loc[intraday["log_return"] > 0]
    neg = intraday.loc[intraday["log_return"] < 0]
    rv_plus = (pos["log_return"] ** 2).groupby(pos["date"]).sum()
    rv_minus = (neg["log_return"] ** 2).groupby(neg["date"]).sum()
    rv_plus = pd.Series(
        rv_plus.to_numpy(dtype=float),
        index=pd.to_datetime(list(rv_plus.index)),
        name="rv_plus",
    ).reindex(rv_d.index, fill_value=0.0)
    rv_minus = pd.Series(
        rv_minus.to_numpy(dtype=float),
        index=pd.to_datetime(list(rv_minus.index)),
        name="rv_minus",
    ).reindex(rv_d.index, fill_value=0.0)

    # Bipower variation per day. mu_1 = sqrt(2/pi), so 1/mu_1^2 = pi/2.
    bipower_values: dict[pd.Timestamp, float] = {}
    for date, group in intraday.groupby("date"):
        r = group["log_return"].to_numpy(dtype=float)
        bipower_values[pd.Timestamp(str(date))] = _bipower_one_day(r)
    bipower = pd.Series(bipower_values, name="bipower").sort_index()
    bipower = bipower.reindex(rv_d.index, fill_value=0.0)

    n_bars_per_day = (
        intraday.groupby("date").size().mean() if len(intraday) else None
    )
    return RVComponents(
        rv_d=rv_d,
        bipower=bipower,
        rv_minus=rv_minus,
        rv_plus=rv_plus,
        source="intraday",
        n_intraday_bars=float(n_bars_per_day) if n_bars_per_day else None,
    )


def _bipower_one_day(returns: np.ndarray) -> float:
    """Scalar bipower variation for one trading day's intraday returns."""

    m = returns.size
    if m < 2:
        return 0.0
    abs_r = np.abs(returns)
    raw = float(np.sum(abs_r[1:] * abs_r[:-1]))
    return (math.pi / 2.0) * (m / (m - 1.0)) * raw


# ---------------------------------------------------------------------------
# OHLC range-based fallback estimators (Strategy C in the spec)
# ---------------------------------------------------------------------------


def compute_garman_klass_rv(ohlc: pd.DataFrame) -> pd.Series:
    """Garman-Klass (1980) daily realized variance from OHLC.

    Formula: 0.5 * (log H/L)^2 - (2 log 2 - 1) * (log C/O)^2.

    Expects columns ``Open``, ``High``, ``Low``, ``Close``. Returns a Series
    indexed by date, named ``rv_d``.
    """

    for col in ("Open", "High", "Low", "Close"):
        if col not in ohlc.columns:
            raise KeyError(
                f"compute_garman_klass_rv: column {col!r} not in OHLC "
                f"(available: {list(ohlc.columns)})"
            )
    o = ohlc["Open"].astype(float).to_numpy()
    h = ohlc["High"].astype(float).to_numpy()
    low = ohlc["Low"].astype(float).to_numpy()
    c = ohlc["Close"].astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        hi_lo = np.log(h / low)
        co = np.log(c / o)
        rv = 0.5 * hi_lo**2 - (2.0 * np.log(2.0) - 1.0) * co**2
    rv = np.where(np.isfinite(rv), rv, np.nan)
    rv = np.where(rv > 0, rv, np.nan)
    idx = _datetime_index(ohlc, prefer_date=True).normalize()
    return pd.Series(rv, index=idx, name="rv_d").dropna()


def compute_yang_zhang_rv(ohlc: pd.DataFrame, window: int = 20) -> pd.Series:
    """Yang-Zhang (2000) minimum-variance OHLC RV estimator.

    Returns a rolling-window estimate (constant within each window) suitable
    as a long-history RV proxy when intraday data is unavailable. The
    estimator combines overnight, open-to-close, and Rogers-Satchell parts:

        sigma_yz^2 = sigma_o^2 + k * sigma_c^2 + (1 - k) * sigma_rs^2
        k = 0.34 / (1.34 + (n+1)/(n-1))
    """

    if window < 2:
        raise ValueError(f"compute_yang_zhang_rv: window must be >= 2, got {window}")
    for col in ("Open", "High", "Low", "Close"):
        if col not in ohlc.columns:
            raise KeyError(
                f"compute_yang_zhang_rv: column {col!r} not in OHLC "
                f"(available: {list(ohlc.columns)})"
            )
    idx = _datetime_index(ohlc, prefer_date=True).normalize()
    df = ohlc.copy()
    df.index = idx
    o = pd.Series(np.log(df["Open"].astype(float).to_numpy()), index=idx, name="o")
    h = pd.Series(np.log(df["High"].astype(float).to_numpy()), index=idx, name="h")
    low_log = pd.Series(
        np.log(df["Low"].astype(float).to_numpy()), index=idx, name="l"
    )
    c = pd.Series(np.log(df["Close"].astype(float).to_numpy()), index=idx, name="c")
    prev_c = c.shift(1)

    # Overnight log return (open vs prior close)
    overnight = o - prev_c
    open_close = c - o
    rs_term = (h - c) * (h - o) + (low_log - c) * (low_log - o)

    sigma_o2 = overnight.rolling(window).var(ddof=1)
    sigma_c2 = open_close.rolling(window).var(ddof=1)
    sigma_rs2 = rs_term.rolling(window).mean()

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
    return pd.Series(yz.to_numpy(dtype=float), index=idx, name="rv_d").dropna()


# ---------------------------------------------------------------------------
# HAR design matrix + OLS
# ---------------------------------------------------------------------------


def aggregate_har_components(
    rv_d: pd.Series,
    *,
    w_window: int = DEFAULT_W_WINDOW,
    m_window: int = DEFAULT_M_WINDOW,
) -> pd.DataFrame:
    """Compute the daily/weekly/monthly aggregations on a daily RV series.

    Returns a DataFrame with columns ``rv_d``, ``rv_w``, ``rv_m`` aligned to
    the input dates. The first ``m_window - 1`` rows are NaN by construction
    (insufficient history).
    """

    if rv_d.empty:
        return pd.DataFrame(columns=["rv_d", "rv_w", "rv_m"])
    s = pd.Series(rv_d.to_numpy(dtype=float), index=rv_d.index, name="rv_d").sort_index()
    rv_w = s.rolling(window=w_window, min_periods=w_window).mean()
    rv_m = s.rolling(window=m_window, min_periods=m_window).mean()
    return pd.DataFrame({"rv_d": s, "rv_w": rv_w, "rv_m": rv_m})


def build_har_design_matrix(
    rv_d: pd.Series,
    *,
    spec: HARSpec = "log",
    horizon: int = 1,
    w_window: int = DEFAULT_W_WINDOW,
    m_window: int = DEFAULT_M_WINDOW,
    min_rv: float = DEFAULT_MIN_RV,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build `(y, X, fit_index)` for the HAR regression.

    The dependent variable is `RV_{t+1:t+h}` averaged across the horizon
    (Corsi-style direct multi-step). Regressors are the lagged daily, weekly,
    monthly aggregations *as of date t*. In log spec, both sides have the
    minimum-floor `min_rv` applied before the log.

    Returns
    -------
    y : ndarray of shape (T_eff,)
    X : ndarray of shape (T_eff, 4)  -- columns [1, RV_d, RV_w, RV_m] (or logs)
    fit_index : DatetimeIndex of length T_eff -- the date `t` of each row
    """

    if horizon < 1:
        raise ValueError(f"build_har_design_matrix: horizon must be >= 1, got {horizon}")
    components = aggregate_har_components(
        rv_d, w_window=w_window, m_window=m_window
    )
    aligned = components.dropna()
    if len(aligned) <= horizon + 1:
        raise ValueError(
            f"build_har_design_matrix: only {len(aligned)} usable observations "
            f"after the {m_window}-day burn-in (need > {horizon + 1})"
        )

    # Multi-horizon dependent variable: mean of next `h` daily RVs.
    daily = components["rv_d"]
    if horizon == 1:
        y_full = daily.shift(-1)
    else:
        # rolling(window=h).mean() at index t gives mean(t-h+1..t), so shift
        # so index t corresponds to mean(t+1..t+h).
        y_full = daily.rolling(window=horizon, min_periods=horizon).mean().shift(-horizon)

    df = pd.concat(
        [
            aligned.rename(columns={"rv_d": "rv_d_t", "rv_w": "rv_w_t", "rv_m": "rv_m_t"}),
            y_full.rename("y"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(df) < 4:
        raise ValueError(
            "build_har_design_matrix: not enough joint observations to fit HAR "
            f"(got {len(df)} rows, need at least 4 free parameters)"
        )

    rv_d_t = df["rv_d_t"].to_numpy(dtype=float)
    rv_w_t = df["rv_w_t"].to_numpy(dtype=float)
    rv_m_t = df["rv_m_t"].to_numpy(dtype=float)
    y_arr = df["y"].to_numpy(dtype=float)

    if spec == "log":
        floor = max(min_rv, 1e-300)
        y_out = np.log(np.maximum(y_arr, floor))
        x_d = np.log(np.maximum(rv_d_t, floor))
        x_w = np.log(np.maximum(rv_w_t, floor))
        x_m = np.log(np.maximum(rv_m_t, floor))
    elif spec == "level":
        y_out = y_arr
        x_d = rv_d_t
        x_w = rv_w_t
        x_m = rv_m_t
    else:
        raise ValueError(f"build_har_design_matrix: unknown spec {spec!r}")

    intercept = np.ones_like(x_d)
    X = np.column_stack([intercept, x_d, x_w, x_m])
    return y_out, X, pd.DatetimeIndex(df.index)


def ols_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Plain OLS. Returns `(theta, residuals, sigma2_residual, r_squared)`.

    `sigma2_residual = RSS / (T - K)`.
    """

    n_obs, n_par = X.shape
    if n_obs <= n_par:
        raise ValueError(f"ols_fit: need T > K, got T={n_obs}, K={n_par}")
    XtX = X.T @ X
    theta = np.linalg.solve(XtX, X.T @ y)
    residuals = y - X @ theta
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / (n_obs - n_par)
    r_squared = 0.0 if tss <= 0.0 else 1.0 - rss / tss
    return theta, residuals, sigma2, r_squared


def nw_optimal_lag(n_obs: int) -> int:
    """Bartlett-window truncation lag `floor(4 * (T/100)^(2/9))` (Newey 1994)."""

    if n_obs < 4:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def newey_west_covariance(
    X: np.ndarray, residuals: np.ndarray, lag: int
) -> np.ndarray:
    """HAC sandwich covariance with a Bartlett kernel.

    Returns `(X^T X)^{-1} S (X^T X)^{-1}` where
    `S = sum_{l=-L..L} (1 - |l|/(L+1)) * sum_t e_t e_{t-l} x_t x_{t-l}^T`.
    """

    n_obs, n_par = X.shape
    if residuals.shape != (n_obs,):
        raise ValueError(
            f"newey_west_covariance: residuals shape {residuals.shape} != T={n_obs}"
        )
    if lag < 0:
        raise ValueError(f"newey_west_covariance: lag must be >= 0, got {lag}")

    eX = residuals[:, None] * X  # (T, K)
    s0 = eX.T @ eX
    s_total = s0.copy()
    for l_val in range(1, lag + 1):
        weight = 1.0 - l_val / (lag + 1.0)
        gamma_l = eX[l_val:].T @ eX[:-l_val]
        s_total += weight * (gamma_l + gamma_l.T)

    XtX_inv = np.linalg.inv(X.T @ X)
    return cast(np.ndarray, XtX_inv @ s_total @ XtX_inv)


def fit_har(
    rv_d: pd.Series,
    *,
    spec: HARSpec = "log",
    horizon: int = 1,
    w_window: int = DEFAULT_W_WINDOW,
    m_window: int = DEFAULT_M_WINDOW,
    hac_lag: int | None = None,
    min_rv: float = DEFAULT_MIN_RV,
) -> HARFit:
    """End-to-end HAR fit: design matrix -> OLS -> HAC covariance.

    `hac_lag=None` uses Newey's rule of thumb. All inference is in the spec
    space (log or level).
    """

    y, X, fit_index = build_har_design_matrix(
        rv_d,
        spec=spec,
        horizon=horizon,
        w_window=w_window,
        m_window=m_window,
        min_rv=min_rv,
    )
    theta, residuals, sigma2, r_squared = ols_fit(y, X)
    lag = hac_lag if hac_lag is not None else nw_optimal_lag(len(y))
    hac_cov = newey_west_covariance(X, residuals, lag)
    return HARFit(
        coefficients=theta,
        hac_covariance=hac_cov,
        residuals=residuals,
        residual_variance=sigma2,
        r_squared=r_squared,
        n_obs=len(y),
        hac_lag=lag,
        spec=spec,
        horizon=horizon,
        fit_index=fit_index,
        metadata={"coef_names": list(HAR_COEF_NAMES)},
    )


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def forecast_one_step(
    fit: HARFit,
    rv_d_t: float,
    rv_w_t: float,
    rv_m_t: float,
    *,
    min_rv: float = DEFAULT_MIN_RV,
) -> float:
    """Conditional one-step (or h-step direct) forecast of `RV_{t+1}^{(d)}`.

    In log space, returns `exp(linear_combination + 0.5 * sigma_eps^2)`
    (the lognormal correction). In level space, returns the linear
    combination directly.
    """

    c_, b_d, b_w, b_m = fit.coefficients
    if fit.spec == "log":
        floor = max(min_rv, 1e-300)
        log_pred = (
            c_
            + b_d * np.log(max(rv_d_t, floor))
            + b_w * np.log(max(rv_w_t, floor))
            + b_m * np.log(max(rv_m_t, floor))
        )
        return float(np.exp(log_pred + 0.5 * fit.residual_variance))
    if fit.spec == "level":
        return float(c_ + b_d * rv_d_t + b_w * rv_w_t + b_m * rv_m_t)
    raise ValueError(f"forecast_one_step: unknown spec {fit.spec!r}")


def latest_har_state(
    rv_d: pd.Series,
    *,
    w_window: int = DEFAULT_W_WINDOW,
    m_window: int = DEFAULT_M_WINDOW,
) -> tuple[float, float, float, pd.Timestamp]:
    """Return `(RV_d_T, RV_w_T, RV_m_T, T)` from the latest available date.

    Raises if there isn't enough history to compute the monthly average.
    """

    components = aggregate_har_components(
        rv_d, w_window=w_window, m_window=m_window
    ).dropna()
    if components.empty:
        raise ValueError(
            f"latest_har_state: not enough RV history "
            f"(need >= {m_window} valid daily RVs)"
        )
    last_row = components.iloc[-1]
    return (
        float(last_row["rv_d"]),
        float(last_row["rv_w"]),
        float(last_row["rv_m"]),
        cast(pd.Timestamp, components.index[-1]),
    )


def in_sample_forecasts(fit: HARFit, X: np.ndarray) -> np.ndarray:
    """Apply the fit's coefficients to a design matrix in spec space.

    For log-spec fits the returned vector is in *log space*; apply the
    lognormal correction `exp(. + 0.5 sigma^2)` externally if you want
    forecasts of `RV` itself.
    """

    return cast(np.ndarray, X @ fit.coefficients)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def mincer_zarnowitz(
    realized: np.ndarray, forecast: np.ndarray
) -> dict[str, float]:
    """Run `realized = a + b * forecast + u`. Returns `a`, `b`, R^2.

    Well-calibrated forecasts have `a ≈ 0` and `b ≈ 1`. We also return the
    OLS t-statistics of those coefficients (against H0: a=0, b=1) so callers
    can apply a Wald-style test outside.
    """

    if realized.shape != forecast.shape:
        raise ValueError(
            f"mincer_zarnowitz: shape mismatch realized {realized.shape} vs "
            f"forecast {forecast.shape}"
        )
    n = realized.size
    if n < 3:
        return {"a": float("nan"), "b": float("nan"), "r_squared": float("nan")}
    X = np.column_stack([np.ones(n), forecast])
    theta, residuals, sigma2, r_squared = ols_fit(realized, X)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    a, b = float(theta[0]), float(theta[1])
    return {
        "a": a,
        "b": b,
        "r_squared": float(r_squared),
        "se_a": float(se[0]),
        "se_b": float(se[1]),
        "t_a": float(a / se[0]) if se[0] > 0 else float("nan"),
        "t_b_minus_1": float((b - 1.0) / se[1]) if se[1] > 0 else float("nan"),
    }


def qlike_loss(realized: np.ndarray, forecast: np.ndarray) -> float:
    """Patton (2011) robust QLIKE loss: mean of `R/F - log(R/F) - 1`.

    Both inputs are assumed to be in `RV` units (not log). Strictly positive
    entries only — non-positive values are dropped before averaging.
    """

    if realized.shape != forecast.shape:
        raise ValueError(
            f"qlike_loss: shape mismatch realized {realized.shape} vs "
            f"forecast {forecast.shape}"
        )
    mask = (realized > 0) & (forecast > 0)
    if not mask.any():
        return float("nan")
    r = realized[mask]
    f = forecast[mask]
    ratio = r / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def ljung_box_pvalue(series: np.ndarray | pd.Series, max_lag: int = 10) -> float:
    """Ljung-Box Q-statistic p-value (H0: no autocorrelation).

    Implemented locally with a Wilson-Hilferty chi-square survival
    approximation so the model has no scipy dependency.
    """

    x = np.asarray(series, dtype=float)
    n = x.size
    if n <= max_lag + 1 or max_lag < 1:
        return float("nan")
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float("nan")
    q = 0.0
    for lag in range(1, max_lag + 1):
        rho = float((x[lag:] * x[:-lag]).sum()) / denom
        q += rho * rho / (n - lag)
    q *= n * (n + 2)
    return _chi2_sf(q, max_lag)


def arch_lm_pvalue(series: np.ndarray | pd.Series, lags: int = 5) -> float:
    """Engle (1982) ARCH-LM test p-value.

    Regresses squared (de-meaned) residuals on `lags` of their own past, and
    forms `T * R^2 ~ chi^2_{lags}` under H0 of no remaining ARCH effects.
    """

    x = np.asarray(series, dtype=float)
    if lags < 1:
        raise ValueError(f"arch_lm_pvalue: lags must be >= 1, got {lags}")
    e2 = (x - x.mean()) ** 2
    n = e2.size
    if n <= lags + 1:
        return float("nan")
    y = e2[lags:]
    rows = [e2[lags - k - 1 : n - k - 1] for k in range(lags)]
    X = np.column_stack([np.ones_like(y)] + rows)
    try:
        theta, residuals, _, r_squared = ols_fit(y, X)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    _ = theta, residuals  # unused — only R^2 enters the LM statistic
    lm_stat = max(0.0, r_squared) * (n - lags)
    return _chi2_sf(lm_stat, lags)


def diebold_mariano_pvalue(
    loss_a: np.ndarray, loss_b: np.ndarray, *, hac_lag: int | None = None
) -> float:
    """Two-sided Diebold-Mariano test p-value for `mean(loss_a - loss_b) = 0`.

    HAC-corrected with a Bartlett kernel. Returns NaN for sub-30 samples or
    zero-variance differentials.
    """

    if loss_a.shape != loss_b.shape:
        raise ValueError(
            f"diebold_mariano_pvalue: shape mismatch {loss_a.shape} vs {loss_b.shape}"
        )
    d = loss_a - loss_b
    n = d.size
    if n < 30:
        return float("nan")
    d_bar = float(d.mean())
    d_c = d - d_bar
    lag = hac_lag if hac_lag is not None else nw_optimal_lag(n)
    gamma0 = float((d_c * d_c).sum()) / n
    var = gamma0
    for l_val in range(1, lag + 1):
        weight = 1.0 - l_val / (lag + 1.0)
        gamma = float((d_c[l_val:] * d_c[:-l_val]).sum()) / n
        var += 2.0 * weight * gamma
    if var <= 0:
        return float("nan")
    dm = d_bar / math.sqrt(var / n)
    return float(math.erfc(abs(dm) / math.sqrt(2.0)))


def _chi2_sf(stat: float, dof: int) -> float:
    """Wilson-Hilferty chi-square survival approximation.

    Same form used in `factor_models_pca.signal.ljung_box_pvalue`. Accurate
    in the tail region (p < 0.1) we care about for hypothesis testing.
    """

    if dof < 1:
        return float("nan")
    if stat <= 0:
        return 1.0
    k = float(dof)
    z = ((stat / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _datetime_index(frame: pd.DataFrame, *, prefer_date: bool = False) -> pd.DatetimeIndex:
    """Best-effort DatetimeIndex extraction for yfinance OHLCV frames.

    yfinance returns the time stamp as the index by default; after a
    `reset_index()` (which our DataProvider performs), the timestamp lives
    in either a ``Datetime`` (intraday) or ``Date`` (daily) column.
    """

    if "Datetime" in frame.columns and not prefer_date:
        return pd.DatetimeIndex(pd.to_datetime(frame["Datetime"]))
    if "Date" in frame.columns:
        return pd.DatetimeIndex(pd.to_datetime(frame["Date"]))
    if "Datetime" in frame.columns:
        return pd.DatetimeIndex(pd.to_datetime(frame["Datetime"]))
    return pd.DatetimeIndex(pd.to_datetime(frame.index))


__all__ = [
    "DEFAULT_MIN_RV",
    "DEFAULT_M_WINDOW",
    "DEFAULT_W_WINDOW",
    "aggregate_har_components",
    "arch_lm_pvalue",
    "build_har_design_matrix",
    "compute_daily_rv",
    "compute_garman_klass_rv",
    "compute_intraday_log_returns",
    "compute_yang_zhang_rv",
    "diebold_mariano_pvalue",
    "fit_har",
    "forecast_one_step",
    "in_sample_forecasts",
    "latest_har_state",
    "ljung_box_pvalue",
    "mincer_zarnowitz",
    "newey_west_covariance",
    "nw_optimal_lag",
    "ols_fit",
    "qlike_loss",
]
