"""Dataclasses specific to the Merton-KMV structural credit model.

The shared `Signal` / `Forecast` / `RiskMetric` / `CalibrationResult` types live in
`src.core.types`; this module holds the model-internal shapes that flow between
`fetch_data`, `calibrate`, and `predict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

CreditQuality = Literal["investment_grade", "high_yield", "distressed", "default_imminent"]
ArbStance = Literal["cds_rich", "cds_cheap", "fair", "no_market_data"]


@dataclass(frozen=True)
class BalanceSheetSnapshot:
    """Subset of balance-sheet fields needed for the KMV default point.

    Values are in the firm's reporting currency (USD for US listings). When the
    balance-sheet feed reports only `total_debt`, the caller is expected to
    split it via `assumed_short_term_fraction` before constructing the snapshot.
    """

    short_term_debt: float
    long_term_debt: float
    cash_and_equivalents: float = 0.0
    total_debt: float | None = None

    def __post_init__(self) -> None:
        if self.short_term_debt < 0:
            raise ValueError(
                f"BalanceSheetSnapshot.short_term_debt must be non-negative, "
                f"got {self.short_term_debt}"
            )
        if self.long_term_debt < 0:
            raise ValueError(
                f"BalanceSheetSnapshot.long_term_debt must be non-negative, "
                f"got {self.long_term_debt}"
            )


@dataclass(frozen=True)
class MertonInputs:
    """Bundle of everything `calibrate` and `predict` consume.

    Produced by `MertonKMV.fetch_data`. Keeping it as a single dataclass keeps
    the contract with the orchestration layer narrow (one `data` blob).
    """

    ticker: str
    timestamp: datetime
    equity_market_value: float
    equity_prices: pd.Series
    shares_outstanding: float
    balance_sheet: BalanceSheetSnapshot
    risk_free_rate: float
    horizon_years: float
    weight_lt_debt: float
    industry: str
    market_cds_spread_bps: float | None = None
    lgd: float = 0.6

    def __post_init__(self) -> None:
        if self.equity_market_value <= 0:
            raise ValueError(
                f"MertonInputs.equity_market_value must be positive, "
                f"got {self.equity_market_value}"
            )
        if self.shares_outstanding <= 0:
            raise ValueError(
                f"MertonInputs.shares_outstanding must be positive, "
                f"got {self.shares_outstanding}"
            )
        if self.horizon_years <= 0:
            raise ValueError(
                f"MertonInputs.horizon_years must be positive, got {self.horizon_years}"
            )
        if not 0.0 <= self.weight_lt_debt <= 1.0:
            raise ValueError(
                f"MertonInputs.weight_lt_debt must lie in [0, 1], got {self.weight_lt_debt}"
            )
        if not 0.0 <= self.lgd <= 1.0:
            raise ValueError(f"MertonInputs.lgd must lie in [0, 1], got {self.lgd}")
        if len(self.equity_prices) < 30:
            raise ValueError(
                f"MertonInputs.equity_prices needs >= 30 daily observations, "
                f"got {len(self.equity_prices)}"
            )


@dataclass(frozen=True)
class KMVSolution:
    """Output of the iterative KMV solve.

    `asset_value_series` is the calibration-window-long path of implied
    asset values; the final element is the spot V_t. `n_iterations` and
    `converged` come from the outer KMV loop.
    """

    asset_value: float
    asset_volatility: float
    asset_value_series: pd.Series
    equity_volatility: float
    default_point: float
    n_iterations: int
    converged: bool
    g1_residual: float
    g2_residual: float


@dataclass(frozen=True)
class MertonResult:
    """End-to-end snapshot returned by the model on each refit."""

    ticker: str
    timestamp: datetime
    asset_value: float
    asset_volatility: float
    equity_volatility: float
    default_point: float
    drift_physical: float
    risk_free_rate: float
    horizon_years: float
    dd_physical: float
    dd_risk_neutral: float
    pd_physical: float
    pd_risk_neutral: float
    credit_spread_bps: float
    n_iterations: int
    converged: bool
    g1_residual: float
    g2_residual: float
    credit_quality: CreditQuality
    arb_stance: ArbStance
    metadata: dict[str, float] = field(default_factory=dict)
