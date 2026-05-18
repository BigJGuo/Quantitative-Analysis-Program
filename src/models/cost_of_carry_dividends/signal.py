"""Pure math for the Cost-of-Carry / Discrete-Dividend model.

All transformations operate on already-fetched dataclasses (no I/O). The math
mirrors the spec's "Algorithm outline" and the formulas in
`models/layer2_structural/05_cost_of_carry_dividends.md`:

* time_to_maturity
* interpolate_rate (log-linear on the rate curve)
* pv_dividend_stream / fv_dividend_stream
* theoretical_futures_discrete / theoretical_futures_continuous
* equivalent_continuous_yield
* theoretical_futures_price (auto-switching)
* implied_repo_rate
* basis / classify_basis_action / signal_strength
* compute_result
* ETF parallel path: accrued_dividends / etf_premium / classify_etf_action
* forward_dividend_strip (term-structure consistency)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from src.core.types import SignalDirection
from src.models.cost_of_carry_dividends.types import (
    BasisDecomposition,
    BasketHolding,
    CostOfCarryInputs,
    CostOfCarryResult,
    CostParameters,
    Dividend,
    DividendStream,
    ETFBasketResult,
    IndexConstituent,
    RateCurve,
)

_YEAR_DAYS: float = 365.25
_EPS: float = 1e-12
_DAY_SECONDS: float = 86_400.0


# --------------------------------------------------------------------------- #
# Time and rate plumbing                                                      #
# --------------------------------------------------------------------------- #

def time_to_maturity(today: datetime, expiry: datetime) -> float:
    """Compute (expiry - today) in years using ACT/365.25.

    Returns 0 when expiry is at or before today — the discrete formula
    degenerates to F == S in that case. Robust to mixed tz-naive / tz-aware
    inputs (both legs are coerced to wall-clock-preserving naive Timestamps).
    """

    seconds = _seconds_between(today, expiry)
    if seconds <= 0:
        return 0.0
    return seconds / (_DAY_SECONDS * _YEAR_DAYS)


def interpolate_rate(curve: RateCurve, tau: float) -> float:
    """Log-linear interpolation of the rate curve at tenor `tau` (years).

    Outside the pillar support we flat-extrapolate (use the nearest endpoint).
    With a single pillar the curve is treated as flat.
    """

    if tau <= 0:
        raise ValueError(f"interpolate_rate: tau must be positive, got {tau}")

    tenors = curve.tenors
    if len(tenors) == 1:
        return float(curve.rates[tenors[0]])

    if tau <= tenors[0]:
        return float(curve.rates[tenors[0]])
    if tau >= tenors[-1]:
        return float(curve.rates[tenors[-1]])

    # Find bracketing pillars.
    lo = tenors[0]
    hi = tenors[-1]
    for i in range(len(tenors) - 1):
        if tenors[i] <= tau <= tenors[i + 1]:
            lo, hi = tenors[i], tenors[i + 1]
            break

    r_lo = curve.rates[lo]
    r_hi = curve.rates[hi]
    # Log-linear in tenor (interpolate the log of the discount factor).
    log_df_lo = -r_lo * lo
    log_df_hi = -r_hi * hi
    weight = (tau - lo) / (hi - lo)
    log_df = log_df_lo + weight * (log_df_hi - log_df_lo)
    return float(-log_df / tau)


# --------------------------------------------------------------------------- #
# Dividend stream PV / FV                                                     #
# --------------------------------------------------------------------------- #

def _seconds_between(start: datetime, end: datetime) -> float:
    """Seconds from `start` to `end`, wall-clock-preserving across tz mixes."""

    return (_to_naive_ts(end) - _to_naive_ts(start)).total_seconds()


def _dividend_ttm(today: datetime, ex_date: datetime) -> float:
    """Years from `today` to `ex_date` (can be negative if already past)."""

    return _seconds_between(today, ex_date) / (_DAY_SECONDS * _YEAR_DAYS)


def pv_dividend_stream(
    stream: DividendStream,
    rate: float,
    today: datetime,
    expiry: datetime,
) -> float:
    """PV at `today` of dividends with ex-dates in (today, expiry]."""

    total = 0.0
    today_ts = _to_naive_ts(today)
    expiry_ts = _to_naive_ts(expiry)
    for d in stream.dividends:
        ex_ts = _to_naive_ts(d.ex_date)
        if ex_ts <= today_ts or ex_ts > expiry_ts:
            continue
        t_i = _dividend_ttm(today, d.ex_date)
        total += d.amount_index_points * math.exp(-rate * t_i)
    return total


def fv_dividend_stream(
    stream: DividendStream,
    rate: float,
    today: datetime,
    expiry: datetime,
) -> float:
    """Future value at `expiry` of dividends with ex-dates in (today, expiry].

    Each dividend earns the financing rate from its ex-date to expiry, so the
    aggregate is `Σ D_i · exp(r · (T − t_i))`.
    """

    total = 0.0
    tau = time_to_maturity(today, expiry)
    today_ts = _to_naive_ts(today)
    expiry_ts = _to_naive_ts(expiry)
    for d in stream.dividends:
        ex_ts = _to_naive_ts(d.ex_date)
        if ex_ts <= today_ts or ex_ts > expiry_ts:
            continue
        t_i = _dividend_ttm(today, d.ex_date)
        total += d.amount_index_points * math.exp(rate * (tau - t_i))
    return total


# --------------------------------------------------------------------------- #
# Theoretical futures price                                                   #
# --------------------------------------------------------------------------- #

def theoretical_futures_discrete(
    spot: float, rate: float, tau: float, fv_dividends: float
) -> float:
    """Discrete-dividend cost-of-carry formula.

    `F = S · exp(r τ) − Σ D_i · exp(r (T − t_i))`. Pass `fv_dividend_stream(...)`
    as `fv_dividends`.
    """

    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    return spot * math.exp(rate * tau) - fv_dividends


def theoretical_futures_continuous(
    spot: float, rate: float, dividend_yield: float, tau: float
) -> float:
    """Continuous-yield cost-of-carry formula `F = S · exp((r − q) τ)`."""

    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    return spot * math.exp((rate - dividend_yield) * tau)


def equivalent_continuous_yield(
    spot: float,
    rate: float,
    tau: float,
    stream: DividendStream,
    today: datetime,
    expiry: datetime,
) -> float:
    """Solve for `q` such that the continuous formula matches the discrete one.

    `q = (1/τ) · ln(1 + Σ D_i · exp(−r · (t_i − t)) / S)`. Returns 0 on the
    degenerate τ ≤ 0 or empty-stream cases.
    """

    if tau <= _EPS:
        return 0.0
    pv = pv_dividend_stream(stream, rate, today, expiry)
    if pv <= 0:
        return 0.0
    return math.log(1.0 + pv / spot) / tau


def theoretical_futures_price(
    *,
    spot: float,
    rate: float,
    today: datetime,
    expiry: datetime,
    stream: DividendStream,
    switch_threshold_years: float = 0.25,
) -> tuple[float, Literal["discrete", "continuous"], float, float]:
    """Auto-switching theoretical futures price.

    Returns `(F_theo, method, pv_divs, fv_divs)`.

    For tau < `switch_threshold_years` we use the discrete formula (accurate
    around clustered ex-dates); for longer maturities we use the continuous
    approximation parametrized by `equivalent_continuous_yield`.
    """

    tau = time_to_maturity(today, expiry)
    pv = pv_dividend_stream(stream, rate, today, expiry)
    fv = fv_dividend_stream(stream, rate, today, expiry)
    if tau <= 0:
        return spot, "discrete", pv, fv

    if tau < switch_threshold_years:
        return theoretical_futures_discrete(spot, rate, tau, fv), "discrete", pv, fv
    q = equivalent_continuous_yield(spot, rate, tau, stream, today, expiry)
    return theoretical_futures_continuous(spot, rate, q, tau), "continuous", pv, fv


# --------------------------------------------------------------------------- #
# Implied repo and basis                                                      #
# --------------------------------------------------------------------------- #

def implied_repo_rate(
    *,
    futures: float,
    spot: float,
    rate: float,
    today: datetime,
    expiry: datetime,
    stream: DividendStream,
) -> float:
    """Invert the cost-of-carry formula to extract the futures-implied repo rate.

    `r_impl = (1/τ) · ln((F + Σ D_i · exp(r (T − t_i))) / S)`.
    """

    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    tau = time_to_maturity(today, expiry)
    if tau <= _EPS:
        return rate
    fv = fv_dividend_stream(stream, rate, today, expiry)
    numerator = futures + fv
    if numerator <= 0:
        raise ValueError(
            f"implied_repo_rate: F + Σ FV(D) must be positive, got {numerator}"
        )
    return math.log(numerator / spot) / tau


def basis(futures: float, theoretical: float) -> float:
    """Market-vs-model basis in index points."""

    return futures - theoretical


def classify_basis_action(
    futures: float, theoretical: float, cost: CostParameters
) -> str:
    """Map the signed basis to a trade action.

    Positive basis beyond the band: future trades rich -> SELL_FUTURE.
    Negative basis beyond the band: future trades cheap -> BUY_FUTURE.
    """

    band = cost.futures_band_points
    b = basis(futures, theoretical)
    if b > band:
        return "SELL_FUTURE"
    if b < -band:
        return "BUY_FUTURE"
    return "FLAT"


_ACTION_TO_DIRECTION: dict[str, SignalDirection] = {
    "SELL_FUTURE": "short",
    "BUY_FUTURE": "long",
    "FLAT": "flat",
}


def action_to_signal_direction(action: str) -> SignalDirection:
    """Convert SELL_FUTURE / BUY_FUTURE / FLAT to a shared-`Signal` direction."""

    try:
        return _ACTION_TO_DIRECTION[action]
    except KeyError as exc:
        raise ValueError(f"Unknown action: {action!r}") from exc


def signal_strength(
    basis_points: float,
    cost: CostParameters,
    scale_points: float = 2.0,
) -> float:
    """Bound the absolute basis beyond the cost band to [0, 1].

    `scale_points` is the size of "basis beyond band" that maps to strength=1.
    For SPX-class futures a 2 index-point deviation past the cost band is a
    clear-edge opportunity.
    """

    excess = max(abs(basis_points) - cost.futures_band_points, 0.0)
    if scale_points <= 0:
        return 1.0 if excess > 0 else 0.0
    scaled = excess / scale_points
    return max(0.0, min(1.0, scaled))


# --------------------------------------------------------------------------- #
# ETF basket valuation (parallel path)                                        #
# --------------------------------------------------------------------------- #

def _to_naive_ts(value: datetime) -> pd.Timestamp:
    """Coerce `value` to a tz-naive `pd.Timestamp` (wall-clock-preserving)."""

    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts
    return ts.tz_localize(None)


def _tz_naive_index(index: pd.Index) -> pd.Index:
    if not isinstance(index, pd.DatetimeIndex):
        return index
    if index.tz is not None:
        return index.tz_localize(None)
    return index


def accrued_dividends(
    holdings: Iterable[BasketHolding],
    dividends_per_share: Mapping[str, pd.Series],
    today: datetime,
    pay_lag_days: int = 30,
) -> float:
    """Sum dividends declared (ex-date <= today) but not yet paid (ex-date + lag > today).

    For each holding `n_i`, accrues `n_i · d_{i,j}` for each cash dividend whose
    ex-date is within the [today - lag, today] window. The default lag (30
    calendar days) matches the spec's T+30 approximation.
    """

    total = 0.0
    today_ts = _to_naive_ts(today)
    lower = today_ts - pd.Timedelta(days=pay_lag_days)
    for holding in holdings:
        series = dividends_per_share.get(holding.ticker)
        if series is None or len(series) == 0:
            continue
        index = _tz_naive_index(series.index)
        if not isinstance(index, pd.DatetimeIndex):
            continue
        mask = np.asarray((index >= lower) & (index <= today_ts))
        accruing = series.to_numpy()[mask]
        total += float(holding.shares) * float(accruing.sum())
    return total


def basket_nav(
    holdings: Iterable[BasketHolding],
    prices: Mapping[str, float],
    accrued: float,
    liabilities: float,
) -> float:
    """NAV_t = Σ n_i · P_i + A_t − L_t."""

    total = 0.0
    for holding in holdings:
        price = prices.get(holding.ticker)
        if price is None:
            raise KeyError(
                f"basket_nav: missing price for holding {holding.ticker!r}"
            )
        total += holding.shares * float(price)
    return total + accrued - liabilities


def etf_premium(market_mid: float, nav_per_share: float) -> float:
    """Relative deviation `(M_t − NAV_pu) / NAV_pu`."""

    if nav_per_share <= 0:
        raise ValueError(
            f"etf_premium: nav_per_share must be positive, got {nav_per_share}"
        )
    return (market_mid - nav_per_share) / nav_per_share


def classify_etf_action(premium: float, cost: CostParameters) -> str:
    """ETF arbitrage trigger: CREATE when premium > band, REDEEM when < -band."""

    band = cost.etf_band_fraction
    if premium > band:
        return "CREATE"
    if premium < -band:
        return "REDEEM"
    return "FLAT"


def compute_etf_basket_result(
    *,
    etf_ticker: str,
    timestamp: datetime,
    holdings: Iterable[BasketHolding],
    prices: Mapping[str, float],
    dividends_per_share: Mapping[str, pd.Series],
    liabilities: float,
    market_mid: float,
    shares_per_creation_unit: float,
    cost_params: CostParameters,
    pay_lag_days: int = 30,
) -> ETFBasketResult:
    """Wrap the parallel ETF-NAV calculation into a single result object."""

    if shares_per_creation_unit <= 0:
        raise ValueError(
            f"shares_per_creation_unit must be positive, got {shares_per_creation_unit}"
        )
    holdings_tuple = tuple(holdings)
    accrued = accrued_dividends(holdings_tuple, dividends_per_share, timestamp, pay_lag_days)
    nav = basket_nav(holdings_tuple, prices, accrued, liabilities)
    nav_per_share = nav / shares_per_creation_unit
    premium = etf_premium(market_mid, nav_per_share)
    action = classify_etf_action(premium, cost_params)
    return ETFBasketResult(
        etf_ticker=etf_ticker,
        timestamp=timestamp,
        nav=nav,
        nav_per_share=nav_per_share,
        market_mid=market_mid,
        premium=premium,
        accrued_dividends=accrued,
        liabilities=liabilities,
        action=action,
        cost_band_bps=cost_params.etf_cost_band_bps,
    )


# --------------------------------------------------------------------------- #
# Term-structure consistency                                                  #
# --------------------------------------------------------------------------- #

def forward_dividend_strip(
    *,
    f_near: float,
    f_far: float,
    rate: float,
    tau_near: float,
    tau_far: float,
) -> float:
    """Implied FV of dividends between two consecutive futures expiries.

    `Σ D_i · exp(r (T_far − t_i)) = F_near · exp(r (T_far − T_near)) − F_far`.
    Negative outputs indicate either model mis-specification or a quarter-end
    distortion that warrants investigation.
    """

    if tau_far < tau_near:
        raise ValueError(
            f"forward_dividend_strip: tau_far ({tau_far}) must be >= tau_near ({tau_near})"
        )
    return f_near * math.exp(rate * (tau_far - tau_near)) - f_far


# --------------------------------------------------------------------------- #
# Constituent-level dividend aggregation                                      #
# --------------------------------------------------------------------------- #

def aggregate_constituent_dividends(
    *,
    constituents: Iterable[IndexConstituent],
    dividends_per_share: Mapping[str, pd.Series],
    today: datetime,
    expiry: datetime,
) -> DividendStream:
    """Aggregate per-share dividend series into index-point dividends.

    For each constituent with weight `w_i` and shares factor `c_i`, each
    per-share dividend `d_{i,t}` contributes `w_i · c_i · d_{i,t}` index
    points on its ex-date.
    """

    dividends: list[Dividend] = []
    today_ts = _to_naive_ts(today)
    expiry_ts = _to_naive_ts(expiry)
    for constituent in constituents:
        series = dividends_per_share.get(constituent.ticker)
        if series is None or len(series) == 0:
            continue
        index = _tz_naive_index(series.index)
        if not isinstance(index, pd.DatetimeIndex):
            continue
        mask = np.asarray((index > today_ts) & (index <= expiry_ts))
        in_window_index = index[mask]
        in_window_values = series.to_numpy()[mask]
        for ex_date_pd, amount in zip(in_window_index, in_window_values, strict=True):
            ex_date = ex_date_pd.to_pydatetime()
            idx_points = float(amount) * constituent.weight * constituent.shares_factor
            dividends.append(
                Dividend(
                    ex_date=ex_date,
                    amount_index_points=idx_points,
                    source="announced",
                    ticker=constituent.ticker,
                )
            )
    return DividendStream(tuple(dividends))


# --------------------------------------------------------------------------- #
# End-to-end                                                                  #
# --------------------------------------------------------------------------- #

def compute_decomposition(inputs: CostOfCarryInputs) -> BasisDecomposition:
    """Compute the full basis breakdown without classification."""

    tau = time_to_maturity(inputs.timestamp, inputs.expiry)
    if tau <= 0:
        rate = interpolate_rate(inputs.rate_curve, max(tau, 1.0 / _YEAR_DAYS))
    else:
        rate = interpolate_rate(inputs.rate_curve, tau)
    rate += inputs.stock_loan_spread

    f_theo, method, pv, fv = theoretical_futures_price(
        spot=inputs.spot,
        rate=rate,
        today=inputs.timestamp,
        expiry=inputs.expiry,
        stream=inputs.dividends,
        switch_threshold_years=inputs.switch_threshold_years,
    )
    b = basis(inputs.futures, f_theo)
    r_impl = implied_repo_rate(
        futures=inputs.futures,
        spot=inputs.spot,
        rate=rate,
        today=inputs.timestamp,
        expiry=inputs.expiry,
        stream=inputs.dividends,
    )
    q = equivalent_continuous_yield(
        inputs.spot, rate, max(tau, _EPS), inputs.dividends, inputs.timestamp, inputs.expiry
    )
    return BasisDecomposition(
        spot=inputs.spot,
        futures=inputs.futures,
        theoretical=f_theo,
        basis=b,
        tau_years=tau,
        rate=rate,
        pv_dividends=pv,
        fv_dividends=fv,
        implied_repo=r_impl,
        method=method,
        equivalent_yield=q,
    )


def compute_result(inputs: CostOfCarryInputs) -> CostOfCarryResult:
    """Top-level entry: produce the basis snapshot output."""

    decomposition = compute_decomposition(inputs)
    action = classify_basis_action(
        inputs.futures, decomposition.theoretical, inputs.cost_params
    )
    return CostOfCarryResult(
        futures_ticker=inputs.futures_ticker,
        timestamp=inputs.timestamp,
        spot=inputs.spot,
        futures=inputs.futures,
        theoretical=decomposition.theoretical,
        basis=decomposition.basis,
        implied_repo=decomposition.implied_repo,
        action=action,
        cost_band_points=inputs.cost_params.futures_band_points,
        decomposition=decomposition,
        metadata={
            "tau_years": decomposition.tau_years,
            "rate": decomposition.rate,
            "pv_dividends": decomposition.pv_dividends,
            "fv_dividends": decomposition.fv_dividends,
            "implied_repo": decomposition.implied_repo,
            "equivalent_yield": decomposition.equivalent_yield,
            "method_discrete": 1.0 if decomposition.method == "discrete" else 0.0,
        },
    )
