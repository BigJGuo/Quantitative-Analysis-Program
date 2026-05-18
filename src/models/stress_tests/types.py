"""Dataclasses specific to the stress-tests risk model.

Shared `Signal` / `RiskMetric` / `CalibrationResult` live in `src.core.types`;
this module holds the model-internal shapes that flow between `fetch_data`,
`calibrate`, `predict`, and `validate`.

Three scenario types are supported (`ScenarioType`):

- ``"historical"`` — replay of a named crisis window.
- ``"hypothetical"`` — hand-specified factor shocks (rates, equity, vol, FX).
- ``"reverse"``     — solve for the smallest factor move that produces a
                      catastrophic loss target.

The fitted state (`StressFit`) carries the per-ticker beta matrix and the
factor-return covariance used by the reverse-stress optimizer; both are
refreshed by `calibrate()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

ScenarioType = Literal["historical", "hypothetical", "reverse"]
VALID_SCENARIO_TYPES: frozenset[str] = frozenset({"historical", "hypothetical", "reverse"})

# Canonical macro factor universe used for hypothetical and reverse scenarios.
# Keys are stable factor names; values are yfinance tickers from which factor
# returns are derived.
DEFAULT_FACTORS: dict[str, str] = {
    "equity": "SPY",
    "rates": "^TNX",
    "vol": "^VIX",
    "dxy": "DX-Y.NYB",
    "ig": "LQD",
    "hy": "HYG",
}


@dataclass(frozen=True)
class CrisisWindow:
    """A named historical-replay scenario window.

    `t1` and `t2` are ISO-format date strings; the implementation slices the
    daily price panel to `[t1, t2]` and uses the first and last available
    closes inside the window.
    """

    name: str
    t1: str
    t2: str

    def __post_init__(self) -> None:
        ts1 = pd.Timestamp(self.t1)
        ts2 = pd.Timestamp(self.t2)
        if ts1 > ts2:
            raise ValueError(
                f"CrisisWindow {self.name!r}: t1 ({self.t1}) > t2 ({self.t2})"
            )


@dataclass(frozen=True)
class HypotheticalScenario:
    """A board-level hypothetical scenario.

    `shocks` keys map to entries in `DEFAULT_FACTORS` (e.g. ``"equity": -0.20``).
    Unrecognized keys are tolerated and ignored — that keeps experimentation
    cheap, but the scenario-coverage diagnostic will flag them.
    """

    name: str
    shocks: dict[str, float]


@dataclass(frozen=True)
class ScenarioPnL:
    """Result of evaluating one scenario against the current positions.

    `contributions` maps each ticker to its dollar P&L under the scenario, so
    the top-3 contributor diagnostic is a direct lookup. `factor_changes` is
    the realized factor shock vector applied to the book (percentage moves for
    historical replay, raw shocks for hypothetical).
    """

    scenario_name: str
    scenario_type: ScenarioType
    total_pnl: float
    contributions: dict[str, float]
    factor_changes: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scenario_type not in VALID_SCENARIO_TYPES:
            raise ValueError(
                f"ScenarioPnL.scenario_type must be one of "
                f"{sorted(VALID_SCENARIO_TYPES)}, got {self.scenario_type!r}"
            )


@dataclass(frozen=True)
class ReverseStressResult:
    """Output of the reverse-stress optimization.

    `dF_star` is the closed-form min-Mahalanobis factor shock that produces
    `loss_target`. `std_devs` reports the same shock in factor-standard-deviation
    units, and `mahalanobis_distance` is the Mahalanobis norm in that metric —
    this is the "plausibility" measure the spec flags at the 4-sigma threshold.
    """

    dF_star: np.ndarray
    std_devs: np.ndarray
    mahalanobis_distance: float
    loss_target: float
    factor_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dF_star.shape != self.std_devs.shape:
            raise ValueError(
                f"dF_star {self.dF_star.shape} and std_devs {self.std_devs.shape} "
                "must share a shape"
            )
        if self.dF_star.shape != (len(self.factor_names),):
            raise ValueError(
                f"dF_star shape {self.dF_star.shape} does not match "
                f"len(factor_names)={len(self.factor_names)}"
            )
        if self.loss_target <= 0:
            raise ValueError(
                f"ReverseStressResult.loss_target must be > 0, got {self.loss_target}"
            )


@dataclass(frozen=True)
class StressFit:
    """Calibrated state used by predict/validate.

    Attributes
    ----------
    tickers:
        Order-preserving tuple of position tickers the fit covers.
    factor_names:
        Order-preserving tuple of factor names (subset of `DEFAULT_FACTORS`
        keys).
    betas:
        `(N, K)` per-asset OLS regression coefficients of asset returns onto
        factor returns. Row `i` corresponds to `tickers[i]`, column `j` to
        `factor_names[j]`.
    factor_covariance:
        `(K, K)` sample covariance of factor returns over the calibration
        window. Used by the reverse-stress optimizer.
    capital:
        Notional capital denominator for the loss-target computation. Same
        currency units as the positions.
    """

    tickers: tuple[str, ...]
    factor_names: tuple[str, ...]
    betas: np.ndarray
    factor_covariance: np.ndarray
    capital: float

    def __post_init__(self) -> None:
        n, k = self.betas.shape
        if n != len(self.tickers):
            raise ValueError(
                f"betas has {n} rows but {len(self.tickers)} tickers were supplied"
            )
        if k != len(self.factor_names):
            raise ValueError(
                f"betas has {k} columns but {len(self.factor_names)} factor "
                "names were supplied"
            )
        if self.factor_covariance.shape != (k, k):
            raise ValueError(
                f"factor_covariance shape {self.factor_covariance.shape} does "
                f"not match K={k}"
            )
        if self.capital <= 0:
            raise ValueError(f"StressFit.capital must be > 0, got {self.capital}")

    @property
    def n_assets(self) -> int:
        return int(self.betas.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.betas.shape[1])


@dataclass(frozen=True)
class StressInputs:
    """Everything `calibrate`, `predict`, and `validate` consume.

    Produced by `StressTests.fetch_data`. `crisis_prices` is keyed by
    `(scenario_name, ticker)` and holds the daily-close series sliced to that
    window — including any sector-proxy or `^GSPC` fallback price panels. The
    fallback metadata is captured in `proxy_used` so diagnostics can surface
    which scenarios relied on a proxy.
    """

    positions: dict[str, float]
    capital: float
    crisis_windows: tuple[CrisisWindow, ...]
    hypotheticals: tuple[HypotheticalScenario, ...]
    loss_target_fraction: float
    timestamp: datetime
    crisis_prices: dict[tuple[str, str], pd.Series]
    factor_returns: pd.DataFrame
    sector_map: dict[str, str]
    proxy_used: dict[tuple[str, str], str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError("StressInputs.positions must be non-empty")
        if self.capital <= 0:
            raise ValueError(f"StressInputs.capital must be > 0, got {self.capital}")
        if not 0.0 < self.loss_target_fraction < 1.0:
            raise ValueError(
                "StressInputs.loss_target_fraction must lie in (0, 1), got "
                f"{self.loss_target_fraction}"
            )


@dataclass(frozen=True)
class StressReport:
    """Full output of one stress run.

    `worst_case` is the most negative scenario P&L; `worst_case_name` is the
    label of the offending scenario. `historical` and `hypothetical` are
    per-scenario `ScenarioPnL` records, indexed by scenario name. `reverse`
    holds the reverse-stress optimization output, when computed.
    """

    timestamp: datetime
    capital: float
    historical: dict[str, ScenarioPnL]
    hypothetical: dict[str, ScenarioPnL]
    reverse: ReverseStressResult | None
    worst_case: float
    worst_case_name: str
    worst_case_type: ScenarioType

    def __post_init__(self) -> None:
        if self.worst_case_type not in VALID_SCENARIO_TYPES:
            raise ValueError(
                f"StressReport.worst_case_type must be one of "
                f"{sorted(VALID_SCENARIO_TYPES)}, got {self.worst_case_type!r}"
            )
