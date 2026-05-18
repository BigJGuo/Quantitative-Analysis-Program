"""Dataclasses specific to the iNAV / equity-ETF fair-value model.

The shared `Signal` / `Forecast` / `RiskMetric` / `CalibrationResult` types live in
`src.core.types`; this module holds the model-internal shapes that flow between
`fetch_data`, `calibrate`, and `predict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class BasketHolding:
    """One row of the creation-unit basket.

    `shares` is the contractual number of shares of the constituent held per
    creation unit; it stays fixed between issuer rebalances.
    """

    ticker: str
    shares: float
    currency: str = "USD"
    weight: float | None = None

    def __post_init__(self) -> None:
        if self.shares < 0:
            raise ValueError(f"BasketHolding.shares must be non-negative, got {self.shares}")
        if not self.currency:
            raise ValueError("BasketHolding.currency must be a non-empty ISO code")


@dataclass(frozen=True)
class ConstituentQuote:
    """Last-print quote for a single basket constituent in its local currency."""

    ticker: str
    price: float
    timestamp: datetime
    is_stale: bool = False

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError(f"ConstituentQuote.price must be non-negative, got {self.price}")


@dataclass(frozen=True)
class INAVDecomposition:
    """Per-share fair-value decomposition.

    All monetary fields are denominated in the ETF's reporting currency
    (USD for US-listed ETFs). `basket_live` and `basket_stale` are the *total*
    USD value of the LIVE and STALE constituent legs (not per-share).
    """

    basket_live: float
    basket_stale: float
    cash: float
    liabilities: float
    shares_outstanding: float
    inav: float

    def __post_init__(self) -> None:
        if self.shares_outstanding <= 0:
            raise ValueError(
                f"INAVDecomposition.shares_outstanding must be positive, "
                f"got {self.shares_outstanding}"
            )


@dataclass(frozen=True)
class INAVResult:
    """End-to-end snapshot returned by the model on each pulse."""

    etf_ticker: str
    timestamp: datetime
    inav: float
    market_mid: float
    premium: float
    decomposition: INAVDecomposition
    n_live: int
    n_stale: int
    stale_fraction: float
    action: str
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CostParameters:
    """Half-spread + transaction-cost band that defines the no-arb boundary.

    Both fields are in basis points of iNAV (1 bp = 1e-4). `kappa` is the
    issuer creation/redemption fee per share; `tau` is the AP's basket
    transaction cost (slippage, commissions). Together they bound the
    arbitrage-free premium.
    """

    kappa_bps: float
    tau_bps: float

    def __post_init__(self) -> None:
        if self.kappa_bps < 0 or self.tau_bps < 0:
            raise ValueError("CostParameters.kappa_bps and tau_bps must be non-negative")

    @property
    def total_band(self) -> float:
        """Total half-band on the premium, expressed as a fraction (e.g. 0.0003)."""

        return (self.kappa_bps + self.tau_bps) * 1e-4


@dataclass(frozen=True)
class CashLedgerSeed:
    """Prior-day end-of-day seeds used to roll cash and liabilities forward."""

    eod_nav: float
    eod_shares_out: float
    eod_release_time: datetime
    prior_cash: float
    prior_liabilities: float


@dataclass(frozen=True)
class INAVInputs:
    """Bundle of everything `predict` and `calibrate` consume.

    Produced by `INAVEquityETF.fetch_data`. Holding this as a single dataclass
    keeps the contract with the orchestration layer narrow (one `data` blob).
    """

    etf_ticker: str
    timestamp: datetime
    basket: tuple[BasketHolding, ...]
    quotes: dict[str, ConstituentQuote]
    fx_rates: dict[str, float]
    dividends_since_eod: dict[str, float]
    short_rate_annual: float
    expense_ratio: float
    seed: CashLedgerSeed
    etf_mid: float
    staleness_threshold_seconds: float
    cost_params: CostParameters
