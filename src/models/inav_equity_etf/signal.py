"""Fair-value decomposition, premium calculation, and CREATE/REDEEM signal logic.

The math here mirrors the spec's "Algorithm outline" steps 4-8: deterministic
accrual of cash and liabilities, partition of constituents into LIVE / STALE,
aggregation into iNAV per share, and finally a sign-based mapping from the
premium to an AP action.

Nothing here touches I/O — every function is a pure transformation of the
already-fetched `INAVInputs` (or a subset of it). That keeps the math
unit-testable without any mock provider.
"""

from __future__ import annotations

from datetime import datetime

from src.models.inav_equity_etf.types import (
    BasketHolding,
    ConstituentQuote,
    CostParameters,
    INAVDecomposition,
    INAVInputs,
    INAVResult,
)

_SECONDS_PER_DAY: float = 86_400.0
_TRADING_DAYS_PER_YEAR: int = 252
_OVERNIGHT_DAYS_PER_YEAR: int = 360


def is_quote_stale(
    quote: ConstituentQuote, now: datetime, threshold_seconds: float
) -> bool:
    """True when the quote is older than `threshold_seconds` or already flagged."""

    if quote.is_stale:
        return True
    age_seconds = (now - quote.timestamp).total_seconds()
    return age_seconds > threshold_seconds


def accrue_cash(
    *,
    prior_cash: float,
    short_rate_annual: float,
    elapsed_seconds: float,
    dividends_received: float,
) -> float:
    """Roll forward C_t from the prior-day seed.

    `short_rate_annual` is the annualized overnight rate (e.g. 0.0525). It is
    accrued on an ACT/360 basis to match money-market convention; the spec
    formula `r * Δt / 86400` is in days, so we divide by 360 here.
    """

    elapsed_day_fraction = max(elapsed_seconds, 0.0) / _SECONDS_PER_DAY
    accrual_factor = 1.0 + short_rate_annual * (elapsed_day_fraction / _OVERNIGHT_DAYS_PER_YEAR)
    return prior_cash * accrual_factor + dividends_received


def accrue_liabilities(
    *,
    prior_liabilities: float,
    expense_ratio_annual: float,
    eod_nav: float,
    eod_shares_out: float,
    elapsed_seconds: float,
) -> float:
    """Roll forward L_t.

    Daily expense is `f / 252 * NAV * shares_out`; we pro-rate within a trading
    day by the elapsed fraction so the model produces a smooth intraday accrual
    rather than a step at NAV strike.
    """

    elapsed_day_fraction = max(elapsed_seconds, 0.0) / _SECONDS_PER_DAY
    daily_expense = (expense_ratio_annual / _TRADING_DAYS_PER_YEAR) * eod_nav * eod_shares_out
    return prior_liabilities + daily_expense * elapsed_day_fraction


def basket_value_usd(
    holdings: tuple[BasketHolding, ...],
    quotes: dict[str, ConstituentQuote],
    fx_rates: dict[str, float],
    *,
    now: datetime,
    staleness_threshold_seconds: float,
    live: bool,
) -> tuple[float, int]:
    """Sum n_i * P_i * FX_i over the LIVE (or STALE) subset.

    Returns `(usd_value, count)`. Constituents with missing quotes are skipped
    silently — the caller can detect them via the count drop in `validate()`.
    """

    total = 0.0
    count = 0
    for holding in holdings:
        quote = quotes.get(holding.ticker)
        if quote is None:
            continue
        stale = is_quote_stale(quote, now, staleness_threshold_seconds)
        if stale == live:
            continue
        fx = 1.0 if holding.currency.upper() == "USD" else fx_rates.get(holding.currency, 1.0)
        total += holding.shares * quote.price * fx
        count += 1
    return total, count


def compute_decomposition(inputs: INAVInputs) -> INAVDecomposition:
    """Build the full iNAV decomposition from already-fetched inputs."""

    now = inputs.timestamp
    elapsed_seconds = max((now - inputs.seed.eod_release_time).total_seconds(), 0.0)
    dividends_total = sum(
        h.shares * inputs.dividends_since_eod.get(h.ticker, 0.0) for h in inputs.basket
    )

    cash = accrue_cash(
        prior_cash=inputs.seed.prior_cash,
        short_rate_annual=inputs.short_rate_annual,
        elapsed_seconds=elapsed_seconds,
        dividends_received=dividends_total,
    )
    liabilities = accrue_liabilities(
        prior_liabilities=inputs.seed.prior_liabilities,
        expense_ratio_annual=inputs.expense_ratio,
        eod_nav=inputs.seed.eod_nav,
        eod_shares_out=inputs.seed.eod_shares_out,
        elapsed_seconds=elapsed_seconds,
    )

    basket_live_value, _ = basket_value_usd(
        inputs.basket,
        inputs.quotes,
        inputs.fx_rates,
        now=now,
        staleness_threshold_seconds=inputs.staleness_threshold_seconds,
        live=True,
    )
    basket_stale_value, _ = basket_value_usd(
        inputs.basket,
        inputs.quotes,
        inputs.fx_rates,
        now=now,
        staleness_threshold_seconds=inputs.staleness_threshold_seconds,
        live=False,
    )

    shares_out = inputs.seed.eod_shares_out
    inav = (basket_live_value + basket_stale_value + cash - liabilities) / shares_out

    return INAVDecomposition(
        basket_live=basket_live_value,
        basket_stale=basket_stale_value,
        cash=cash,
        liabilities=liabilities,
        shares_outstanding=shares_out,
        inav=inav,
    )


def compute_premium(market_mid: float, inav: float) -> float:
    """Relative deviation of the secondary-market price from iNAV."""

    if inav <= 0:
        raise ValueError(f"iNAV must be positive to compute premium, got {inav}")
    return (market_mid - inav) / inav


def classify_action(premium: float, cost: CostParameters) -> str:
    """Map the signed premium to an AP action: CREATE / REDEEM / FLAT."""

    band = cost.total_band
    if premium > band:
        return "CREATE"
    if premium < -band:
        return "REDEEM"
    return "FLAT"


_ACTION_TO_DIRECTION: dict[str, str] = {
    # ETF is rich -> AP sells the ETF (short) and delivers basket -> "short" the ETF.
    "CREATE": "short",
    # ETF is cheap -> AP buys the ETF (long) and redeems for basket -> "long" the ETF.
    "REDEEM": "long",
    "FLAT": "flat",
}


def action_to_signal_direction(action: str) -> str:
    """Convert CREATE/REDEEM/FLAT to a shared-`Signal` direction."""

    try:
        return _ACTION_TO_DIRECTION[action]
    except KeyError as exc:
        raise ValueError(f"Unknown AP action: {action!r}") from exc


def signal_strength(premium: float, cost: CostParameters, scale_bps: float = 25.0) -> float:
    """Bound the absolute premium to [0, 1] for the shared `Signal` strength field.

    The premium beyond the no-arb band is divided by `scale_bps` (default 25 bps,
    a healthy creation/redemption opportunity) and clipped. The exact scale is
    cosmetic — the signed action and raw premium are the load-bearing outputs.
    """

    excess = max(abs(premium) - cost.total_band, 0.0)
    scaled = excess / (scale_bps * 1e-4)
    return max(0.0, min(1.0, scaled))


def compute_inav_result(inputs: INAVInputs) -> INAVResult:
    """Top-level entry point: produce the full `INAVResult` from inputs."""

    decomposition = compute_decomposition(inputs)
    _, n_live = basket_value_usd(
        inputs.basket,
        inputs.quotes,
        inputs.fx_rates,
        now=inputs.timestamp,
        staleness_threshold_seconds=inputs.staleness_threshold_seconds,
        live=True,
    )
    _, n_stale = basket_value_usd(
        inputs.basket,
        inputs.quotes,
        inputs.fx_rates,
        now=inputs.timestamp,
        staleness_threshold_seconds=inputs.staleness_threshold_seconds,
        live=False,
    )
    basket_total = decomposition.basket_live + decomposition.basket_stale
    stale_fraction = (
        decomposition.basket_stale / basket_total if basket_total > 0 else 0.0
    )

    premium = compute_premium(inputs.etf_mid, decomposition.inav)
    action = classify_action(premium, inputs.cost_params)

    return INAVResult(
        etf_ticker=inputs.etf_ticker,
        timestamp=inputs.timestamp,
        inav=decomposition.inav,
        market_mid=inputs.etf_mid,
        premium=premium,
        decomposition=decomposition,
        n_live=n_live,
        n_stale=n_stale,
        stale_fraction=stale_fraction,
        action=action,
        metadata={
            "basket_live_usd": decomposition.basket_live,
            "basket_stale_usd": decomposition.basket_stale,
            "cash_usd": decomposition.cash,
            "liabilities_usd": decomposition.liabilities,
            "band_bps": (inputs.cost_params.kappa_bps + inputs.cost_params.tau_bps),
        },
    )
