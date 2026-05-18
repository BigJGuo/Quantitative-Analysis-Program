"""Calibration entry point for the Liquidity-Adjusted VaR model.

For a risk model the "calibration" step is the assembly of per-ticker
liquidity statistics — ADV, relative spread, FRTB bucket, internal tier,
and the Almgren-Chriss impact coefficient. The fitted parameters are
those statistics; no optimization is involved (the spec's STEP 7 prefers
TCA-fitted η in production, which is out of scope for a yfinance-only
pipeline).

`calibrate` returns a `CalibrationResult` whose ``parameters["ticker_stats"]``
holds the tuple of `TickerLiquidityStats` consumed downstream by
`compute_l_var`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.liquidity_adjusted_var.signal import (
    almgren_chriss_eta,
    assign_frtb_bucket,
    fallback_spread_from_high_low,
    liquidation_horizon_days,
    relative_spread_from_bid_ask,
    tier_for_adv_ratio,
)
from src.models.liquidity_adjusted_var.types import (
    Position,
    TickerLiquidityStats,
)

MODEL_NAME: str = "liquidity_adjusted_var"


def _last_finite(series: pd.Series) -> float:
    """Last non-NaN value of ``series``. Raises if none exists."""

    s = series.dropna()
    if s.empty:
        raise ValueError("Series has no finite values")
    return float(s.iloc[-1])


def _resolve_spread(
    *,
    fundamentals: dict[str, Any],
    highs: pd.Series | None,
    lows: pd.Series | None,
    spread_fallback_lookback: int,
) -> tuple[float, str]:
    """Pick a relative spread for one ticker, recording the source.

    Tries live ``info["bid"]`` / ``info["ask"]`` first per spec STEP 2;
    falls back to the high-low range proxy when quotes are absent / zero.
    Returns ``(spread_relative, source)`` where source is either
    ``"bid_ask"`` or ``"hl_proxy"``.
    """

    bid_raw = fundamentals.get("bid")
    ask_raw = fundamentals.get("ask")
    if bid_raw is not None and ask_raw is not None:
        try:
            spread = relative_spread_from_bid_ask(float(bid_raw), float(ask_raw))
        except (TypeError, ValueError):
            spread = None
        if spread is not None and spread > 0.0:
            return spread, "bid_ask"
    if highs is None or lows is None:
        return 0.0, "hl_proxy"
    return (
        fallback_spread_from_high_low(highs, lows, lookback=spread_fallback_lookback),
        "hl_proxy",
    )


def calibrate(
    *,
    positions: tuple[Position, ...],
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    highs: pd.DataFrame | None,
    lows: pd.DataFrame | None,
    fundamentals: dict[str, dict[str, Any]],
    max_participation: float,
    adv_lookback: int = 60,
    spread_fallback_lookback: int = 20,
    ac_impact_coef: float = 0.1,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Build per-ticker `TickerLiquidityStats` from the panels supplied.

    Parameters
    ----------
    positions:
        Ordered positions. Order is preserved through the output stats.
    prices, volumes:
        Wide ``pd.DataFrame``s with one column per ticker. Indexed by
        trading day; missing rows are tolerated and dropped during
        last-price / ADV computation.
    highs, lows:
        Wide high/low frames used only for the spread fallback. May be
        ``None`` when the caller is confident every name has a live
        bid/ask.
    fundamentals:
        Dict ``{ticker -> info_dict}`` carrying at minimum ``bid``,
        ``ask``, ``marketCap``. Missing keys are tolerated.
    max_participation:
        Maximum fraction of ADV the desk is willing to trade per day.
    adv_lookback:
        Window over which average daily volume is computed (default
        60 trading days per spec STEP 2).
    spread_fallback_lookback:
        Window for the high/low spread proxy when bid/ask is missing.
    ac_impact_coef:
        Scaling constant ``c`` in
        ``eta = c * sigma_dollar / ADV_dollar`` (default 0.1 per spec).
    timestamp:
        Calibration timestamp (defaults to ``datetime.now(UTC)``).
    """

    if not positions:
        raise ValueError("calibrate requires at least one position")
    if not 0.0 < max_participation <= 1.0:
        raise ValueError(
            f"max_participation must lie in (0, 1], got {max_participation}"
        )
    if adv_lookback < 1:
        raise ValueError(f"adv_lookback must be >= 1, got {adv_lookback}")
    if spread_fallback_lookback < 1:
        raise ValueError(
            f"spread_fallback_lookback must be >= 1, got {spread_fallback_lookback}"
        )
    if ac_impact_coef < 0.0:
        raise ValueError(f"ac_impact_coef must be >= 0, got {ac_impact_coef}")

    ts = timestamp or datetime.now(UTC)

    tickers = [p.ticker for p in positions]
    missing_price = [t for t in tickers if t not in prices.columns]
    missing_vol = [t for t in tickers if t not in volumes.columns]
    if missing_price:
        raise ValueError(f"prices DataFrame missing tickers: {missing_price}")
    if missing_vol:
        raise ValueError(f"volumes DataFrame missing tickers: {missing_vol}")

    stats_list: list[TickerLiquidityStats] = []
    for pos in positions:
        price_series = prices[pos.ticker].astype(float)
        vol_series = volumes[pos.ticker].astype(float)
        last_price = _last_finite(price_series)
        adv_shares = float(vol_series.tail(adv_lookback).dropna().mean())
        if not np.isfinite(adv_shares) or adv_shares <= 0.0:
            raise ValueError(
                f"Ticker {pos.ticker!r}: ADV computed as {adv_shares}; "
                "cannot size liquidation horizon."
            )
        adv_dollar = adv_shares * last_price
        shares_held = pos.dollar_value / last_price
        position_horizon = liquidation_horizon_days(
            shares_held=shares_held,
            adv_shares=adv_shares,
            max_participation=max_participation,
        )
        returns = price_series.pct_change().dropna()
        daily_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0

        info = fundamentals.get(pos.ticker, {})
        ticker_highs = (
            highs[pos.ticker] if (highs is not None and pos.ticker in highs.columns) else None
        )
        ticker_lows = (
            lows[pos.ticker] if (lows is not None and pos.ticker in lows.columns) else None
        )
        spread_rel, source = _resolve_spread(
            fundamentals=info,
            highs=ticker_highs,
            lows=ticker_lows,
            spread_fallback_lookback=spread_fallback_lookback,
        )

        market_cap_raw = info.get("marketCap")
        market_cap: float | None
        if market_cap_raw is None:
            market_cap = None
        else:
            try:
                cap_val = float(market_cap_raw)
            except (TypeError, ValueError):
                cap_val = float("nan")
            market_cap = cap_val if np.isfinite(cap_val) and cap_val > 0.0 else None

        bucket = assign_frtb_bucket(
            market_cap=market_cap,
            position_horizon_days=position_horizon,
        )
        tier = tier_for_adv_ratio(pos.dollar_value, adv_dollar)
        eta = almgren_chriss_eta(
            daily_return_std=daily_std,
            last_price=last_price,
            adv_dollar=adv_dollar,
            impact_coef=ac_impact_coef,
        )

        stats_list.append(
            TickerLiquidityStats(
                ticker=pos.ticker,
                last_price=last_price,
                shares_held=shares_held,
                adv_shares=adv_shares,
                adv_dollar=adv_dollar,
                daily_return_std=daily_std,
                spread_relative=spread_rel,
                spread_source=source,
                market_cap=market_cap,
                position_horizon_days=position_horizon,
                frtb_bucket_days=bucket,
                tier=tier,
                ac_eta=eta,
            )
        )

    ticker_stats = tuple(stats_list)
    gross = sum(abs(p.dollar_value) for p in positions)
    net = sum(p.dollar_value for p in positions)
    fit_metrics: dict[str, float] = {
        "n_positions": float(len(positions)),
        "gross_exposure": float(gross),
        "net_exposure": float(net),
        "mean_spread_relative": float(
            np.mean([s.spread_relative for s in ticker_stats])
        ),
        "mean_position_horizon_days": float(
            np.mean([s.position_horizon_days for s in ticker_stats])
        ),
        "max_position_horizon_days": float(
            max(s.position_horizon_days for s in ticker_stats)
        ),
        "max_participation": float(max_participation),
        "adv_lookback": float(adv_lookback),
    }

    parameters: dict[str, Any] = {
        "ticker_stats": ticker_stats,
        "max_participation": max_participation,
        "adv_lookback": adv_lookback,
        "spread_fallback_lookback": spread_fallback_lookback,
        "ac_impact_coef": ac_impact_coef,
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata={
            "tickers": tickers,
            "n_tier_4": sum(1 for s in ticker_stats if s.tier == 4),
            "n_hl_proxy_spreads": sum(
                1 for s in ticker_stats if s.spread_source == "hl_proxy"
            ),
        },
    )


__all__ = ["MODEL_NAME", "calibrate"]
