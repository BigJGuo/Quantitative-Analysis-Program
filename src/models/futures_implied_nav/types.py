"""Dataclasses specific to the Futures-Implied NAV model.

The shared `Signal` / `Forecast` / `RiskMetric` / `CalibrationResult` types live in
`src.core.types`; this module holds the model-internal shapes that flow between
`fetch_data`, `calibrate`, and `predict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class HedgeSpec:
    """Specification for one hedge instrument used to project the basket return.

    `ticker` follows yfinance conventions (e.g. `^N225`, `EWZ`, `JPYUSD=X`).
    `kind` distinguishes futures / equity-proxy / FX so that downstream logic
    (FX is applied multiplicatively, not regressed) can route correctly.
    """

    ticker: str
    kind: str = "equity"
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("HedgeSpec.ticker must be non-empty")
        if self.kind not in {"futures", "equity", "fx", "adr"}:
            raise ValueError(
                f"HedgeSpec.kind must be one of futures/equity/fx/adr, got {self.kind!r}"
            )


@dataclass(frozen=True)
class HedgePanel:
    """Aligned price panel over (overnight) calibration days.

    `dates` enumerates the foreign-close-to-foreign-open windows. `nav_close`
    and `nav_open` are the per-day NAV anchors. `hedge_close` and `hedge_now`
    are dicts keyed by hedge ticker, each a sequence of length len(dates).
    """

    dates: tuple[datetime, ...]
    nav_close: tuple[float, ...]
    nav_open: tuple[float, ...]
    hedge_close: dict[str, tuple[float, ...]]
    hedge_now: dict[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        n = len(self.dates)
        if n == 0:
            raise ValueError("HedgePanel must contain at least one window")
        if len(self.nav_close) != n or len(self.nav_open) != n:
            raise ValueError("HedgePanel NAV series must align with dates")
        for ticker, series in self.hedge_close.items():
            if len(series) != n:
                raise ValueError(
                    f"hedge_close[{ticker!r}] length {len(series)} != {n}"
                )
        for ticker, series in self.hedge_now.items():
            if len(series) != n:
                raise ValueError(
                    f"hedge_now[{ticker!r}] length {len(series)} != {n}"
                )
        if set(self.hedge_close) != set(self.hedge_now):
            raise ValueError(
                "hedge_close and hedge_now must cover the same hedge tickers"
            )


@dataclass(frozen=True)
class BetaVector:
    """Calibrated hedge loadings (in the order given by `tickers`)."""

    tickers: tuple[str, ...]
    betas: tuple[float, ...]
    ridge_lambda: float
    r_squared: float
    residual_variance: float

    def __post_init__(self) -> None:
        if len(self.tickers) != len(self.betas):
            raise ValueError("BetaVector.tickers and .betas must have the same length")
        if self.ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be non-negative, got {self.ridge_lambda}")
        if self.residual_variance < 0:
            raise ValueError(
                f"residual_variance must be non-negative, got {self.residual_variance}"
            )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.tickers, self.betas, strict=True))


@dataclass(frozen=True)
class SourceVariances:
    """Conditional variances of each fair-value source, used for inverse-variance blending."""

    sigma2_etf: float
    sigma2_nav: float
    sigma2_hedge: float

    def __post_init__(self) -> None:
        for name, val in (
            ("sigma2_etf", self.sigma2_etf),
            ("sigma2_nav", self.sigma2_nav),
            ("sigma2_hedge", self.sigma2_hedge),
        ):
            if val <= 0:
                raise ValueError(f"SourceVariances.{name} must be positive, got {val}")


@dataclass(frozen=True)
class BlendWeights:
    """Inverse-variance weights for the multi-source fair value.

    Normalized to sum to 1.0 by construction.
    """

    w_etf: float
    w_nav: float
    w_hedge: float

    def __post_init__(self) -> None:
        for name, val in (
            ("w_etf", self.w_etf),
            ("w_nav", self.w_nav),
            ("w_hedge", self.w_hedge),
        ):
            if val < 0 or val > 1:
                raise ValueError(f"BlendWeights.{name} must be in [0, 1], got {val}")
        total = self.w_etf + self.w_nav + self.w_hedge
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"BlendWeights must sum to 1, got {total}")


@dataclass(frozen=True)
class FairValueDecomposition:
    """Breakdown of FV_t into its three sources.

    `fv_hedge` is the single-factor futures-implied FV (NAV_close · (1 + β·r_H) · fx_ratio).
    `nav_official` is the most recently published NAV adjusted for FX (no intraday refresh).
    `etf_mid` is the secondary-market mid. `fv_total` is the inverse-variance blend.
    """

    fv_hedge: float
    nav_official: float
    etf_mid: float
    fv_total: float
    weights: BlendWeights
    implied_basket_return: float
    fx_ratio: float


@dataclass(frozen=True)
class FuturesImpliedNAVResult:
    """End-to-end snapshot returned by the model on each pulse."""

    etf_ticker: str
    timestamp: datetime
    fair_value: float
    market_mid: float
    premium: float
    decomposition: FairValueDecomposition
    action: str
    cost_band_bps: float
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CostParameters:
    """Half-spread / transaction-cost band that defines the no-arb boundary.

    Both fields are in basis points of fair value (1 bp = 1e-4).
    """

    half_spread_bps: float
    cost_band_bps: float

    def __post_init__(self) -> None:
        if self.half_spread_bps < 0 or self.cost_band_bps < 0:
            raise ValueError(
                "CostParameters fields must be non-negative"
            )

    @property
    def total_band(self) -> float:
        """Total trade-trigger threshold as a fraction (e.g. 0.0010 for 10 bps)."""

        return (self.half_spread_bps + self.cost_band_bps) * 1e-4


@dataclass(frozen=True)
class NAVAnchor:
    """Most recent issuer-published NAV with its strike-time hedge prices.

    `nav_close` is the NAV in USD struck at `strike_time`. `hedge_close` /
    `fx_close` are the contemporaneous quotes of the hedge / FX legs that the
    intraday update uses as the "since-NAV" baseline.
    """

    nav_close: float
    strike_time: datetime
    hedge_close: dict[str, float]
    fx_close: dict[str, float]

    def __post_init__(self) -> None:
        if self.nav_close <= 0:
            raise ValueError(f"NAVAnchor.nav_close must be positive, got {self.nav_close}")


@dataclass(frozen=True)
class FuturesImpliedNAVInputs:
    """Bundle consumed by `predict` and `validate`.

    Produced by `FuturesImpliedNAV.fetch_data`. Holding everything in a single
    dataclass keeps the orchestration contract narrow (one `data` blob).
    """

    etf_ticker: str
    timestamp: datetime
    hedges: tuple[HedgeSpec, ...]
    hedge_now: dict[str, float]
    fx_now: dict[str, float]
    anchor: NAVAnchor
    etf_mid: float
    beta: BetaVector
    variances: SourceVariances
    cost_params: CostParameters
    panel: HedgePanel | None = None
