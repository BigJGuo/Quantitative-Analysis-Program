"""Merton-KMV math: option pricing, distance-to-default, PD, credit spread.

Every function in this module is a pure transformation of numeric inputs. The
iterative calibration loop lives in `calibration.py`; the orchestration shell
lives in `model.py`. Keeping the math here makes each formula directly
unit-testable against the spec's worked-out expressions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.merton_kmv.types import (
    ArbStance,
    BalanceSheetSnapshot,
    CreditQuality,
    KMVSolution,
    MertonInputs,
    MertonResult,
)

_TRADING_DAYS_PER_YEAR: int = 252
_BPS_PER_UNIT: float = 1e4

# DD bucket boundaries used for the `credit_quality` label. Numbers from the
# spec's "DD level reasonableness" diagnostic: IG sits at 3-6, HY at 1-3,
# distressed below 1, default-imminent below 0.
_DD_THRESHOLD_INVESTMENT_GRADE: float = 3.0
_DD_THRESHOLD_HIGH_YIELD: float = 1.0
_DD_THRESHOLD_DISTRESSED: float = 0.0


def norm_cdf(x: float) -> float:
    """Standard normal CDF computed via the error function.

    Used instead of `scipy.stats.norm.cdf` so the model has no scipy dependency.
    The relative error of `math.erf` is < 1e-15 over the range that matters here.
    """

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def default_point(
    bs: BalanceSheetSnapshot, weight_lt_debt: float = 0.5
) -> float:
    """KMV default point: ST debt + w * LT debt.

    The empirical w = 0.5 reflects that not all long-term debt is callable
    at the moment of distress (spec, "KMV default point").
    """

    if not 0.0 <= weight_lt_debt <= 1.0:
        raise ValueError(
            f"weight_lt_debt must lie in [0, 1], got {weight_lt_debt}"
        )
    return bs.short_term_debt + weight_lt_debt * bs.long_term_debt


def merton_d1_d2(
    asset_value: float,
    default_pt: float,
    risk_free_rate: float,
    horizon_years: float,
    asset_volatility: float,
) -> tuple[float, float]:
    """`d1` and `d2` from the Black-Scholes formula applied to firm value.

    Defined per the spec's equation under "Merton's structural framework".
    """

    if asset_value <= 0:
        raise ValueError(f"asset_value must be positive, got {asset_value}")
    if default_pt <= 0:
        raise ValueError(f"default_pt must be positive, got {default_pt}")
    if asset_volatility <= 0:
        raise ValueError(
            f"asset_volatility must be positive, got {asset_volatility}"
        )
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be positive, got {horizon_years}")

    sqrt_t = math.sqrt(horizon_years)
    d1 = (
        math.log(asset_value / default_pt)
        + (risk_free_rate + 0.5 * asset_volatility * asset_volatility) * horizon_years
    ) / (asset_volatility * sqrt_t)
    d2 = d1 - asset_volatility * sqrt_t
    return d1, d2


def merton_call_price(
    asset_value: float,
    default_pt: float,
    risk_free_rate: float,
    horizon_years: float,
    asset_volatility: float,
) -> float:
    """Black-Scholes call value: E_t given V_t and asset vol.

    `E_t = V N(d1) - D exp(-r T) N(d2)`.
    """

    d1, d2 = merton_d1_d2(
        asset_value, default_pt, risk_free_rate, horizon_years, asset_volatility
    )
    discount = math.exp(-risk_free_rate * horizon_years)
    return asset_value * norm_cdf(d1) - default_pt * discount * norm_cdf(d2)


def equity_vol_from_asset_vol(
    asset_value: float,
    equity_value: float,
    default_pt: float,
    risk_free_rate: float,
    horizon_years: float,
    asset_volatility: float,
) -> float:
    """`sigma_E = (V/E) * N(d1) * sigma_V` (Ito's-lemma identity)."""

    d1, _ = merton_d1_d2(
        asset_value, default_pt, risk_free_rate, horizon_years, asset_volatility
    )
    if equity_value <= 0:
        raise ValueError(f"equity_value must be positive, got {equity_value}")
    return (asset_value / equity_value) * norm_cdf(d1) * asset_volatility


def solve_for_v_given_e(
    equity_value: float,
    default_pt: float,
    risk_free_rate: float,
    horizon_years: float,
    asset_volatility: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> float:
    """Newton inversion of the Black-Scholes formula for `V` given `E`.

    The subroutine in the spec's "Newton subroutine `solve_for_V_given_E`"
    section. Uses `dE/dV = N(d1)` as the derivative, with `V_0 = E + D e^{-rT}`
    as the starting point.
    """

    if equity_value <= 0:
        raise ValueError(f"equity_value must be positive, got {equity_value}")

    v = equity_value + default_pt * math.exp(-risk_free_rate * horizon_years)
    for _ in range(max_iter):
        d1, d2 = merton_d1_d2(
            v, default_pt, risk_free_rate, horizon_years, asset_volatility
        )
        discount = math.exp(-risk_free_rate * horizon_years)
        f = v * norm_cdf(d1) - default_pt * discount * norm_cdf(d2) - equity_value
        fp = norm_cdf(d1)
        # If N(d1) collapses to zero (very deep OTM), Newton blows up. Bump V
        # by 1% instead so the iteration keeps moving toward the real root.
        v_new = v * 1.01 if fp <= 0 else v - f / fp
        if v_new <= 0:
            # Keep V strictly positive throughout the iteration.
            v_new = max(v * 0.5, 1e-6 * equity_value)
        if abs(v_new - v) < tol * max(abs(v), 1.0):
            return v_new
        v = v_new
    return v


def realized_equity_volatility(prices: pd.Series, window: int | None = None) -> float:
    """Annualized standard deviation of daily log returns.

    `window` defaults to the full price series; if provided, only the trailing
    `window` observations are used (spec: 252-day rolling).
    """

    if len(prices) < 2:
        raise ValueError(
            f"realized_equity_volatility needs >= 2 observations, got {len(prices)}"
        )
    log_prices = np.log(prices.to_numpy(dtype=float))
    log_returns = np.diff(log_prices)
    if window is not None and window < len(log_returns):
        log_returns = log_returns[-window:]
    if len(log_returns) < 2:
        raise ValueError(
            "realized_equity_volatility needs >= 2 returns after windowing"
        )
    return float(np.std(log_returns, ddof=1) * math.sqrt(_TRADING_DAYS_PER_YEAR))


def distance_to_default(
    asset_value: float,
    default_pt: float,
    asset_volatility: float,
    drift: float,
    horizon_years: float,
) -> float:
    """DD under the supplied drift.

    For physical DD pass `drift = mu`; for risk-neutral DD pass `drift = r`.
    Spec formula: `(log(V/D) + (drift - 0.5 sigma^2) T) / (sigma sqrt(T))`.
    """

    if asset_value <= 0:
        raise ValueError(f"asset_value must be positive, got {asset_value}")
    if default_pt <= 0:
        raise ValueError(f"default_pt must be positive, got {default_pt}")
    if asset_volatility <= 0:
        raise ValueError(
            f"asset_volatility must be positive, got {asset_volatility}"
        )
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be positive, got {horizon_years}")
    numerator = math.log(asset_value / default_pt) + (
        drift - 0.5 * asset_volatility * asset_volatility
    ) * horizon_years
    denominator = asset_volatility * math.sqrt(horizon_years)
    return numerator / denominator


def prob_default(distance: float) -> float:
    """`PD = N(-DD)` under the lognormal-asset-return assumption."""

    return norm_cdf(-distance)


def credit_spread(
    risk_neutral_pd: float, horizon_years: float, loss_given_default: float = 0.6
) -> float:
    """Theoretical credit spread implied by the risk-neutral PD.

    `s = -log(1 - PD_Q * LGD) / T` (spec, "Risk-neutral version"). Returned in
    decimal units (e.g. 0.025 for 250 bps). The caller scales to bps if needed.
    """

    if not 0.0 <= risk_neutral_pd <= 1.0:
        raise ValueError(
            f"risk_neutral_pd must lie in [0, 1], got {risk_neutral_pd}"
        )
    if not 0.0 <= loss_given_default <= 1.0:
        raise ValueError(
            f"loss_given_default must lie in [0, 1], got {loss_given_default}"
        )
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be positive, got {horizon_years}")
    inside = 1.0 - risk_neutral_pd * loss_given_default
    if inside <= 0:
        # Near-certain default — return a very wide spread rather than -inf.
        return 10.0
    return -math.log(inside) / horizon_years


def classify_credit_quality(distance: float) -> CreditQuality:
    """Bucket the DD into a coarse credit-quality label (spec diagnostic 7)."""

    if distance < _DD_THRESHOLD_DISTRESSED:
        return "default_imminent"
    if distance < _DD_THRESHOLD_HIGH_YIELD:
        return "distressed"
    if distance < _DD_THRESHOLD_INVESTMENT_GRADE:
        return "high_yield"
    return "investment_grade"


def cap_struct_arb_stance(
    model_spread_bps: float,
    market_cds_spread_bps: float | None,
    threshold_bps: float = 25.0,
) -> ArbStance:
    """Capital-structure arbitrage signal from model vs. market CDS divergence.

    If the CDS market is materially wider than the structural-model spread the
    bond/CDS leg is "rich" relative to equity — the AP-style trade is short
    CDS protection (or long bonds) versus a corresponding equity hedge.
    """

    if market_cds_spread_bps is None:
        return "no_market_data"
    diff = market_cds_spread_bps - model_spread_bps
    if diff > threshold_bps:
        return "cds_rich"
    if diff < -threshold_bps:
        return "cds_cheap"
    return "fair"


def estimate_drift_from_assets(asset_value_series: pd.Series) -> float:
    """Annualized mean log-return on assets — the trailing-mean drift `mu`.

    Spec ("Drift estimation", Option B): `mean(diff(logV)) * 252`. Returns 0 if
    the series is too short to take a difference.
    """

    if len(asset_value_series) < 2:
        return 0.0
    log_v = np.log(asset_value_series.to_numpy(dtype=float))
    return float(np.mean(np.diff(log_v)) * _TRADING_DAYS_PER_YEAR)


def equity_price_to_market_cap(
    close_prices: pd.Series, shares_outstanding: float
) -> pd.Series:
    """Reconstruct historical E_s = Close_s * shares_out.

    Shares outstanding is treated as constant over the calibration window —
    this is the spec's stated approximation (small bias from buybacks/issuance).
    """

    if shares_outstanding <= 0:
        raise ValueError(
            f"shares_outstanding must be positive, got {shares_outstanding}"
        )
    return close_prices.astype(float) * shares_outstanding


def compute_merton_result(
    inputs: MertonInputs,
    solution: KMVSolution,
    drift: float,
    *,
    arb_threshold_bps: float = 25.0,
) -> MertonResult:
    """Assemble the public `MertonResult` from inputs and the calibrated solve."""

    dd_physical = distance_to_default(
        solution.asset_value,
        solution.default_point,
        solution.asset_volatility,
        drift,
        inputs.horizon_years,
    )
    dd_risk_neutral = distance_to_default(
        solution.asset_value,
        solution.default_point,
        solution.asset_volatility,
        inputs.risk_free_rate,
        inputs.horizon_years,
    )
    pd_physical = prob_default(dd_physical)
    pd_risk_neutral = prob_default(dd_risk_neutral)
    spread_decimal = credit_spread(
        pd_risk_neutral, inputs.horizon_years, inputs.lgd
    )
    spread_bps = spread_decimal * _BPS_PER_UNIT

    quality = classify_credit_quality(dd_physical)
    arb = cap_struct_arb_stance(
        spread_bps, inputs.market_cds_spread_bps, arb_threshold_bps
    )

    leverage = solution.default_point / solution.asset_value
    return MertonResult(
        ticker=inputs.ticker,
        timestamp=inputs.timestamp,
        asset_value=solution.asset_value,
        asset_volatility=solution.asset_volatility,
        equity_volatility=solution.equity_volatility,
        default_point=solution.default_point,
        drift_physical=drift,
        risk_free_rate=inputs.risk_free_rate,
        horizon_years=inputs.horizon_years,
        dd_physical=dd_physical,
        dd_risk_neutral=dd_risk_neutral,
        pd_physical=pd_physical,
        pd_risk_neutral=pd_risk_neutral,
        credit_spread_bps=spread_bps,
        n_iterations=solution.n_iterations,
        converged=solution.converged,
        g1_residual=solution.g1_residual,
        g2_residual=solution.g2_residual,
        credit_quality=quality,
        arb_stance=arb,
        metadata={
            "leverage": leverage,
            "equity_market_value": inputs.equity_market_value,
            "shares_outstanding": inputs.shares_outstanding,
            "lgd": inputs.lgd,
        },
    )


def signal_direction_from_quality(quality: CreditQuality, arb: ArbStance) -> str:
    """Map (credit quality, arb stance) to a shared-`Signal` direction.

    When CDS market data is available the cap-structure arb stance dominates;
    otherwise we fall back to the credit-quality bucket as a directional view
    on the underlying equity (distressed -> short, healthy -> long).
    """

    if arb == "cds_rich":
        return "short"
    if arb == "cds_cheap":
        return "long"
    if arb == "fair":
        return "flat"
    if quality in ("distressed", "default_imminent"):
        return "short"
    if quality == "investment_grade":
        return "long"
    return "flat"


def signal_strength_from_dd(distance: float, scale: float = 4.0) -> float:
    """Bound DD-derived conviction to [0, 1] for the shared `Signal` field.

    The further DD is from the IG/distressed midpoint (DD ~= 2), the higher the
    conviction. `scale` controls how quickly strength saturates — by default a
    DD shift of 4 units (e.g. from 2 to 6) saturates the signal.
    """

    pivot = 0.5 * (_DD_THRESHOLD_INVESTMENT_GRADE + _DD_THRESHOLD_HIGH_YIELD)
    excess = abs(distance - pivot) / scale
    return max(0.0, min(1.0, excess))


def diagnostic_residuals(
    asset_value: float,
    equity_value: float,
    default_pt: float,
    risk_free_rate: float,
    horizon_years: float,
    asset_volatility: float,
    equity_volatility: float,
) -> tuple[float, float]:
    """Return `(|g1|, |g2|)` for the converged solution.

    `g1 = V N(d1) - D e^{-rT} N(d2) - E`
    `g2 = (V/E) N(d1) sigma_V - sigma_E`

    Spec validation 2 expects both relative residuals below 1e-4.
    """

    model_e = merton_call_price(
        asset_value, default_pt, risk_free_rate, horizon_years, asset_volatility
    )
    g1 = abs(model_e - equity_value) / max(abs(equity_value), 1.0)
    model_sigma_e = equity_vol_from_asset_vol(
        asset_value,
        equity_value,
        default_pt,
        risk_free_rate,
        horizon_years,
        asset_volatility,
    )
    g2 = abs(model_sigma_e - equity_volatility) / max(abs(equity_volatility), 1e-6)
    return g1, g2


def select_recent_prices(
    prices: pd.Series, history_days: int
) -> pd.Series:
    """Trailing `history_days` observations from `prices`, in price-series form."""

    if len(prices) <= history_days:
        return prices
    return prices.iloc[-history_days:]


def kmv_summary_table(
    asset_value_series: pd.Series, asset_volatility: float
) -> dict[str, float]:
    """Lightweight summary of the asset-value path for diagnostics."""

    values = asset_value_series.to_numpy(dtype=float)
    log_returns = np.diff(np.log(values)) if len(values) > 1 else np.array([])
    annualization = math.sqrt(_TRADING_DAYS_PER_YEAR)
    return {
        "v_first": float(values[0]),
        "v_last": float(values[-1]),
        "v_min": float(values.min()),
        "v_max": float(values.max()),
        "asset_vol_realized": (
            float(np.std(log_returns, ddof=1) * annualization)
            if len(log_returns) > 1
            else 0.0
        ),
        "asset_vol_solved": asset_volatility,
    }
