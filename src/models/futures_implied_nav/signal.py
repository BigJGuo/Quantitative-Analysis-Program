"""Intraday fair-value, premium, and signal logic.

Pure transformations of already-fetched `FuturesImpliedNAVInputs` — no I/O.
The math mirrors the spec's "Algorithm outline / Intraday hot loop" steps:

1. Hedge returns since NAV strike.
2. Implied basket return r_B_hat = beta @ r_H.
3. FX adjustment ratio.
4. Single-factor FV_hedge = NAV_close * (1 + r_B_hat) * fx_ratio.
5. NAV_official = NAV_close * fx_ratio  (no intraday update).
6. Inverse-variance blend weights.
7. Multi-source FV_t.
8. Premium, action.
"""

from __future__ import annotations

from src.core.types import SignalDirection
from src.models.futures_implied_nav.types import (
    BetaVector,
    BlendWeights,
    CostParameters,
    FairValueDecomposition,
    FuturesImpliedNAVInputs,
    FuturesImpliedNAVResult,
    NAVAnchor,
    SourceVariances,
)

_EPS: float = 1e-12


def hedge_return(price_now: float, price_close: float) -> float:
    """Simple return P_now / P_close - 1 with a positivity guard on the denominator."""

    if price_close <= 0:
        raise ValueError(f"hedge_return: price_close must be positive, got {price_close}")
    return price_now / price_close - 1.0


def implied_basket_return(beta: BetaVector, hedge_returns: dict[str, float]) -> float:
    """Compute r_B_hat = sum_j beta_j * r_H_j.

    Missing hedges contribute zero — calibration enforces a non-empty hedge set,
    so a missing intraday print just degrades the projection rather than
    breaking it. The caller can detect missing hedges via `validate()`.
    """

    total = 0.0
    for ticker, b in zip(beta.tickers, beta.betas, strict=True):
        r = hedge_returns.get(ticker)
        if r is None:
            continue
        total += b * r
    return total


def fx_adjustment(fx_now: dict[str, float], fx_close: dict[str, float]) -> float:
    """Geometric mean of per-currency FX ratios, defaulting to 1.0 with no FX.

    A multi-currency basket has multiple FX legs; we apply their geometric mean
    as a first-order approximation. For a single-currency ETF (typical case)
    this reduces to FX_t / FX_close exactly.
    """

    if not fx_close:
        return 1.0
    log_sum = 0.0
    count = 0
    for currency, p_close in fx_close.items():
        p_now = fx_now.get(currency)
        if p_now is None or p_close <= 0 or p_now <= 0:
            continue
        ratio = p_now / p_close
        log_sum += _safe_log(ratio)
        count += 1
    if count == 0:
        return 1.0
    return _safe_exp(log_sum / count)


def _safe_log(x: float) -> float:
    if x <= 0:
        raise ValueError(f"_safe_log requires positive input, got {x}")
    import math

    return math.log(x)


def _safe_exp(x: float) -> float:
    import math

    return math.exp(x)


def single_factor_fair_value(
    nav_close: float,
    implied_return: float,
    fx_ratio: float,
) -> float:
    """FV_hedge = NAV_close * (1 + r_B_hat) * fx_ratio."""

    if nav_close <= 0:
        raise ValueError(f"nav_close must be positive, got {nav_close}")
    return nav_close * (1.0 + implied_return) * fx_ratio


def inverse_variance_weights(variances: SourceVariances) -> BlendWeights:
    """w_k = (1/sigma_k^2) / sum_j (1/sigma_j^2)."""

    inv_etf = 1.0 / variances.sigma2_etf
    inv_nav = 1.0 / variances.sigma2_nav
    inv_hedge = 1.0 / variances.sigma2_hedge
    total = inv_etf + inv_nav + inv_hedge
    if total <= _EPS:
        raise ValueError("inverse_variance_weights: total precision degenerate to zero")
    return BlendWeights(
        w_etf=inv_etf / total,
        w_nav=inv_nav / total,
        w_hedge=inv_hedge / total,
    )


def multi_source_fair_value(
    weights: BlendWeights,
    etf_mid: float,
    nav_official: float,
    fv_hedge: float,
) -> float:
    """FV_t = w_ETF * M_t + w_NAV * NAV_off + w_Hedge * FV_hedge."""

    return weights.w_etf * etf_mid + weights.w_nav * nav_official + weights.w_hedge * fv_hedge


def compute_premium(market_mid: float, fair_value: float) -> float:
    """Relative deviation of the ETF's secondary mid from the fair value."""

    if fair_value <= 0:
        raise ValueError(f"fair_value must be positive to compute premium, got {fair_value}")
    return (market_mid - fair_value) / fair_value


def classify_action(premium: float, cost: CostParameters) -> str:
    """Map the signed premium to a market-making action: SELL_ETF / BUY_ETF / FLAT.

    When the ETF mid exceeds FV by more than the cost band, the model says
    sell the ETF (it's rich). When it falls below, buy the ETF (it's cheap).
    """

    band = cost.total_band
    if premium > band:
        return "SELL_ETF"
    if premium < -band:
        return "BUY_ETF"
    return "FLAT"


_ACTION_TO_DIRECTION: dict[str, SignalDirection] = {
    "SELL_ETF": "short",
    "BUY_ETF": "long",
    "FLAT": "flat",
}


def action_to_signal_direction(action: str) -> SignalDirection:
    """Convert SELL_ETF / BUY_ETF / FLAT to a shared-`Signal` direction."""

    try:
        return _ACTION_TO_DIRECTION[action]
    except KeyError as exc:
        raise ValueError(f"Unknown action: {action!r}") from exc


def signal_strength(premium: float, cost: CostParameters, scale_bps: float = 50.0) -> float:
    """Bound the absolute premium beyond the cost band to [0, 1].

    `scale_bps` is the size of "premium beyond band" that maps to strength=1.
    For EEM-class international ETFs a 50 bp deviation past the cost band is
    a clear-edge opportunity.
    """

    excess = max(abs(premium) - cost.total_band, 0.0)
    scaled = excess / (scale_bps * 1e-4)
    return max(0.0, min(1.0, scaled))


def nav_official_usd(anchor: NAVAnchor, fx_ratio: float) -> float:
    """NAV_official = NAV_close * fx_ratio. No intraday refresh during foreign close."""

    return anchor.nav_close * fx_ratio


def compute_fair_value(inputs: FuturesImpliedNAVInputs) -> FairValueDecomposition:
    """Full multi-source FV computation from inputs."""

    hedge_returns: dict[str, float] = {}
    for ticker, p_close in inputs.anchor.hedge_close.items():
        p_now = inputs.hedge_now.get(ticker)
        if p_now is None or p_close <= 0:
            continue
        hedge_returns[ticker] = hedge_return(p_now, p_close)

    r_b_hat = implied_basket_return(inputs.beta, hedge_returns)
    fx_ratio = fx_adjustment(inputs.fx_now, inputs.anchor.fx_close)
    fv_hedge = single_factor_fair_value(inputs.anchor.nav_close, r_b_hat, fx_ratio)
    nav_off = nav_official_usd(inputs.anchor, fx_ratio)
    weights = inverse_variance_weights(inputs.variances)
    fv_total = multi_source_fair_value(weights, inputs.etf_mid, nav_off, fv_hedge)

    return FairValueDecomposition(
        fv_hedge=fv_hedge,
        nav_official=nav_off,
        etf_mid=inputs.etf_mid,
        fv_total=fv_total,
        weights=weights,
        implied_basket_return=r_b_hat,
        fx_ratio=fx_ratio,
    )


def compute_result(inputs: FuturesImpliedNAVInputs) -> FuturesImpliedNAVResult:
    """Top-level entry: produce the snapshot output."""

    decomposition = compute_fair_value(inputs)
    premium = compute_premium(inputs.etf_mid, decomposition.fv_total)
    action = classify_action(premium, inputs.cost_params)
    band_bps = inputs.cost_params.half_spread_bps + inputs.cost_params.cost_band_bps

    return FuturesImpliedNAVResult(
        etf_ticker=inputs.etf_ticker,
        timestamp=inputs.timestamp,
        fair_value=decomposition.fv_total,
        market_mid=inputs.etf_mid,
        premium=premium,
        decomposition=decomposition,
        action=action,
        cost_band_bps=band_bps,
        metadata={
            "fv_hedge": decomposition.fv_hedge,
            "nav_official": decomposition.nav_official,
            "implied_basket_return": decomposition.implied_basket_return,
            "fx_ratio": decomposition.fx_ratio,
            "w_etf": decomposition.weights.w_etf,
            "w_nav": decomposition.weights.w_nav,
            "w_hedge": decomposition.weights.w_hedge,
            "r_squared": inputs.beta.r_squared,
        },
    )
