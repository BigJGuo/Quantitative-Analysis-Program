"""Dataclasses for the Liquidity-Adjusted VaR model.

L-VaR is a portfolio-level risk metric: one model instance owns a dict of
``{ticker -> dollar position}`` and produces a single L-VaR number plus a
per-name diagnostics block. The shared `RiskMetric` / `CalibrationResult`
types live in `src.core.types`; this module holds only the model-internal
shapes that flow between `fetch_data`, `calibrate`, `predict`, and `validate`.

All dollar quantities are reported in the same currency as the positions
(USD by default given the yfinance data source).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

# FRTB IMA cumulative-endpoint table from BCBS d457 / d518. The implicit
# "preceding endpoint" before LH_1 is zero; bucket scaling uses the
# increment between consecutive endpoints. T_BASE is the regulator's
# unit horizon (10 trading days).
FRTB_HORIZON_ENDPOINTS: tuple[int, ...] = (10, 20, 40, 60, 120)
FRTB_T_BASE: int = 10

VALID_TIERS: frozenset[int] = frozenset({1, 2, 3, 4})


@dataclass(frozen=True)
class Position:
    """Signed dollar position in a single instrument.

    ``dollar_value`` is positive for longs, negative for shorts. The
    liquidity component of L-VaR uses the absolute value (since exiting
    either side pays the spread); the price-risk component uses the signed
    value.
    """

    ticker: str
    dollar_value: float

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("Position.ticker must be a non-empty string")
        if not isinstance(self.dollar_value, int | float):
            raise TypeError(
                f"Position.dollar_value must be numeric, got {type(self.dollar_value).__name__}"
            )


@dataclass(frozen=True)
class TickerLiquidityStats:
    """Calibrated per-ticker liquidity statistics.

    All fields are point-in-time as of the calibration timestamp.

    Attributes
    ----------
    ticker:
        Symbol the stats describe.
    last_price:
        Most recent close used to translate dollar positions to shares.
    shares_held:
        Signed share count = ``dollar_value / last_price``.
    adv_shares:
        Average daily share volume over the configured lookback (default 60d).
    adv_dollar:
        ``adv_shares * last_price``.
    daily_return_std:
        Sample standard deviation of arithmetic daily returns over the
        full price history (decimal, not percent).
    spread_relative:
        Relative bid-ask spread, ``(ask - bid) / mid``. Falls back to a
        high-low range proxy when live quotes are missing — see
        ``signal.fallback_spread_from_high_low``.
    spread_source:
        ``"bid_ask"`` when computed from live quotes, ``"hl_proxy"`` when
        the fallback was used. Surfaces in diagnostics so stale quotes
        are visible.
    market_cap:
        ``info["marketCap"]`` if available, else ``None``.
    position_horizon_days:
        Raw ADV-implied liquidation horizon
        ``|shares_held| / (max_participation * adv_shares)``. Floored at 1.0.
    frtb_bucket_days:
        FRTB endpoint assigned to this ticker — one of
        ``FRTB_HORIZON_ENDPOINTS = (10, 20, 40, 60, 120)``. Computed as the
        smallest endpoint that is both (a) >= the market-cap bucket and
        (b) >= ``ceil(position_horizon_days)``.
    tier:
        Internal liquidity tier in ``{1, 2, 3, 4}``. Driven by the
        position-to-ADV ratio per the spec's tier table.
    ac_eta:
        Almgren-Chriss temporary-impact coefficient, calibrated as
        ``ac_impact_coef * (daily_return_std * last_price) / adv_dollar``
        per the spec's STEP 7 heuristic. Units are loose — this is a
        first-cut estimate; TCA-fitted η should replace it in production.
    """

    ticker: str
    last_price: float
    shares_held: float
    adv_shares: float
    adv_dollar: float
    daily_return_std: float
    spread_relative: float
    spread_source: str
    market_cap: float | None
    position_horizon_days: float
    frtb_bucket_days: int
    tier: int
    ac_eta: float

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("TickerLiquidityStats.ticker must be a non-empty string")
        if self.last_price <= 0.0:
            raise ValueError(
                f"TickerLiquidityStats.last_price must be > 0, got {self.last_price}"
            )
        if self.adv_shares <= 0.0:
            raise ValueError(
                f"TickerLiquidityStats.adv_shares must be > 0, got {self.adv_shares}"
            )
        if self.spread_relative < 0.0:
            raise ValueError(
                f"TickerLiquidityStats.spread_relative must be >= 0, "
                f"got {self.spread_relative}"
            )
        if self.frtb_bucket_days not in FRTB_HORIZON_ENDPOINTS:
            raise ValueError(
                f"frtb_bucket_days must be one of {FRTB_HORIZON_ENDPOINTS}, "
                f"got {self.frtb_bucket_days}"
            )
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"tier must be one of {sorted(VALID_TIERS)}, got {self.tier}"
            )
        if self.position_horizon_days < 1.0:
            raise ValueError(
                f"position_horizon_days is floored at 1.0; got {self.position_horizon_days}"
            )


@dataclass(frozen=True)
class LiquidityVaRInputs:
    """Everything `calibrate`, `predict`, and `validate` consume.

    Built by `LiquidityAdjustedVaR.fetch_data` after pulling the prices /
    volumes / fundamentals panels for the position universe through the
    `DataProvider`.

    Attributes
    ----------
    positions:
        Ordered tuple of signed dollar positions. Order is preserved in all
        derived matrices.
    alpha:
        Confidence level for VaR / ES. The spec defaults to 0.975 (the FRTB
        ES level); 0.99 is also common.
    max_participation:
        Maximum fraction of ADV the desk is willing to trade per day when
        sizing the liquidation horizon. Typical 0.10–0.20.
    returns_panel:
        Wide ``pd.DataFrame`` of arithmetic daily returns. Columns are tickers
        in the same order as ``positions``; index is the trading-day calendar.
        NaNs have been dropped row-wise upstream.
    ticker_stats:
        Per-ticker liquidity stats in the same order as ``positions``.
    timestamp:
        Calibration timestamp.
    metadata:
        Free-form (period strings, n_obs counts, etc.).
    """

    positions: tuple[Position, ...]
    alpha: float
    max_participation: float
    returns_panel: pd.DataFrame
    ticker_stats: tuple[TickerLiquidityStats, ...]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError("LiquidityVaRInputs.positions must be non-empty")
        if not 0.5 < self.alpha < 1.0:
            raise ValueError(
                f"alpha must lie in (0.5, 1.0), got {self.alpha}"
            )
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError(
                f"max_participation must lie in (0, 1], got {self.max_participation}"
            )
        if len(self.positions) != len(self.ticker_stats):
            raise ValueError(
                f"positions ({len(self.positions)}) and ticker_stats "
                f"({len(self.ticker_stats)}) must have the same length"
            )
        pos_tickers = [p.ticker for p in self.positions]
        stat_tickers = [s.ticker for s in self.ticker_stats]
        if pos_tickers != stat_tickers:
            raise ValueError(
                f"positions and ticker_stats must share order; "
                f"positions={pos_tickers}, stats={stat_tickers}"
            )
        panel_cols = list(self.returns_panel.columns)
        if panel_cols != pos_tickers:
            raise ValueError(
                f"returns_panel columns must equal position order; "
                f"got panel={panel_cols}, positions={pos_tickers}"
            )
        if len(self.returns_panel) < 30:
            raise ValueError(
                f"returns_panel must have >= 30 rows, got {len(self.returns_panel)}"
            )


@dataclass(frozen=True)
class LVaRResult:
    """Output of `compute_l_var` — the spec's STEP 8 return shape.

    Attributes
    ----------
    price_var_1d:
        Historical-simulation 1-day portfolio VaR at the chosen alpha,
        in dollars.
    es_frtb:
        Liquidity-horizon-scaled ES per the FRTB combination formula,
        in dollars.
    liquidity_cost_spread:
        Bangia-Diebold round-trip spread cost
        ``sum_t 0.5 * spread_t * |position_t|``, in dollars.
    liquidity_cost_ac:
        Sum of per-name Almgren-Chriss expected execution costs in the
        units returned by ``almgren_chriss_simple_cost`` (see signal.py for
        the unit caveat).
    l_var:
        Headline L-VaR = ``es_frtb + liquidity_cost_spread``. The spec
        offers ``liquidity_cost_ac`` as an alternative.
    per_name:
        Diagnostic dict keyed by ticker: position_value, shares_held,
        adv_dollar, spread_relative, frtb_bucket_days, position_horizon_days,
        tier, ac_cost.
    alpha:
        Echo of the input alpha.
    timestamp:
        Echo of the input timestamp.
    """

    price_var_1d: float
    es_frtb: float
    liquidity_cost_spread: float
    liquidity_cost_ac: float
    l_var: float
    per_name: dict[str, dict[str, Any]]
    alpha: float
    timestamp: datetime
