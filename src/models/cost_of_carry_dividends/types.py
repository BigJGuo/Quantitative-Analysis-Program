"""Dataclasses specific to the Cost-of-Carry / Discrete-Dividend model.

Shared shapes (`Signal`, `Forecast`, `RiskMetric`, `CalibrationResult`) live in
`src.core.types`; this module holds the model-internal shapes that flow between
`fetch_data`, `calibrate`, and `predict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

DividendSource = Literal["announced", "projected"]


@dataclass(frozen=True)
class Dividend:
    """A single discrete dividend payment at the index level.

    `amount_index_points` is the per-share cash payment already converted to
    index points (i.e. weighted by constituent index weight and the index
    divisor). `source` distinguishes announced ex-dates (high confidence) from
    model-projected ones (lower confidence).
    """

    ex_date: datetime
    amount_index_points: float
    source: DividendSource = "announced"
    ticker: str = ""

    def __post_init__(self) -> None:
        if self.amount_index_points < 0:
            raise ValueError(
                f"Dividend.amount_index_points must be non-negative, "
                f"got {self.amount_index_points}"
            )
        if self.source not in {"announced", "projected"}:
            raise ValueError(
                f"Dividend.source must be 'announced' or 'projected', got {self.source!r}"
            )


@dataclass(frozen=True)
class DividendStream:
    """Ordered collection of dividends between trade date and futures expiry."""

    dividends: tuple[Dividend, ...]

    def __post_init__(self) -> None:
        # Sort defensively at construction so downstream math can rely on it.
        sorted_divs = tuple(sorted(self.dividends, key=lambda d: d.ex_date))
        object.__setattr__(self, "dividends", sorted_divs)

    @property
    def total_index_points(self) -> float:
        return sum(d.amount_index_points for d in self.dividends)

    def filter_window(self, start: datetime, end: datetime) -> DividendStream:
        """Return a new stream containing only dividends with `start < ex_date <= end`."""
        return DividendStream(
            tuple(d for d in self.dividends if start < d.ex_date <= end)
        )

    def __len__(self) -> int:
        return len(self.dividends)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.dividends)


@dataclass(frozen=True)
class RateCurve:
    """Sparse zero-rate curve keyed by tenor in years.

    The mapping must contain at least one tenor. `interpolate_rate` (in
    signal.py) does log-linear interpolation between the two pillars that
    straddle the requested tenor, and flat-extrapolates outside the support.
    """

    rates: dict[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rates:
            raise ValueError("RateCurve must contain at least one tenor")
        for tenor, rate in self.rates.items():
            if tenor <= 0:
                raise ValueError(f"RateCurve tenor must be positive, got {tenor}")
            if rate < -0.05 or rate > 0.30:
                raise ValueError(
                    f"RateCurve rate {rate} at tenor {tenor} outside sanity bounds"
                )

    @property
    def tenors(self) -> tuple[float, ...]:
        return tuple(sorted(self.rates.keys()))


@dataclass(frozen=True)
class IndexConstituent:
    """One constituent of the underlying index for dividend aggregation.

    `shares_factor` is `(index_weight * index_divisor) / index_level`, i.e. the
    multiplier that turns a per-share cash dividend into the corresponding
    move in index points. For ETF basket valuation use `BasketHolding` instead.
    """

    ticker: str
    weight: float
    shares_factor: float

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("IndexConstituent.ticker must be non-empty")
        if self.weight < 0:
            raise ValueError(
                f"IndexConstituent.weight must be non-negative, got {self.weight}"
            )
        if self.shares_factor <= 0:
            raise ValueError(
                f"IndexConstituent.shares_factor must be positive, got {self.shares_factor}"
            )


@dataclass(frozen=True)
class BasketHolding:
    """One holding in an ETF creation basket.

    `shares` is the integer share count per creation unit. The basket-NAV
    formula `NAV = Σ n_i * P_i + A_t - L_t` consumes these directly.
    """

    ticker: str
    shares: float

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("BasketHolding.ticker must be non-empty")
        if self.shares <= 0:
            raise ValueError(f"BasketHolding.shares must be positive, got {self.shares}")


@dataclass(frozen=True)
class CostParameters:
    """Per-side transaction-cost band defining the no-arb trigger.

    Expressed in **index points** because the underlying basis is an index-point
    quantity. For SPX-class futures, the round-trip cost is typically O(0.5)
    points; for thinner index futures it can be a few points.

    `etf_cost_band_bps` is the equivalent band on the ETF premium leg, in
    basis points of NAV.
    """

    futures_band_points: float
    etf_cost_band_bps: float

    def __post_init__(self) -> None:
        if self.futures_band_points < 0:
            raise ValueError(
                f"futures_band_points must be non-negative, got {self.futures_band_points}"
            )
        if self.etf_cost_band_bps < 0:
            raise ValueError(
                f"etf_cost_band_bps must be non-negative, got {self.etf_cost_band_bps}"
            )

    @property
    def etf_band_fraction(self) -> float:
        return self.etf_cost_band_bps * 1e-4


@dataclass(frozen=True)
class BasisDecomposition:
    """Breakdown of the basis calculation per the spec's algorithm outline."""

    spot: float
    futures: float
    theoretical: float
    basis: float
    tau_years: float
    rate: float
    pv_dividends: float
    fv_dividends: float
    implied_repo: float
    method: Literal["discrete", "continuous"]
    equivalent_yield: float


@dataclass(frozen=True)
class CostOfCarryResult:
    """Top-level snapshot returned by `CostOfCarry.predict()`."""

    futures_ticker: str
    timestamp: datetime
    spot: float
    futures: float
    theoretical: float
    basis: float
    implied_repo: float
    action: str
    cost_band_points: float
    decomposition: BasisDecomposition
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ETFBasketResult:
    """Parallel ETF basket valuation result (spec algorithm step 8/9)."""

    etf_ticker: str
    timestamp: datetime
    nav: float
    nav_per_share: float
    market_mid: float
    premium: float
    accrued_dividends: float
    liabilities: float
    action: str
    cost_band_bps: float


@dataclass(frozen=True)
class CostOfCarryInputs:
    """Bundle consumed by `predict` and `validate`.

    Produced by `CostOfCarry.fetch_data`. Holding everything in a single
    dataclass keeps the orchestration contract narrow.
    """

    futures_ticker: str
    spot_ticker: str
    timestamp: datetime
    expiry: datetime
    spot: float
    futures: float
    rate_curve: RateCurve
    stock_loan_spread: float
    dividends: DividendStream
    cost_params: CostParameters
    switch_threshold_years: float = 0.25
