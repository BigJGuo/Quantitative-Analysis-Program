"""Dataclasses specific to the Avellaneda-Lee PCA-residual stat-arb model.

Shared `Signal` / `CalibrationResult` live in `src.core.types`; this module
holds the model-internal shapes that flow between `fetch_data`, `calibrate`,
and `predict`.

Per-stock OU fits travel as a tuple of `OUFit` records (one per surviving
ticker). Non-survivors — stocks that fail the stationarity, half-life, or
R-squared filters — are dropped from `StatArbFit.ou_fits` but still listed
in `dropped_tickers` together with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

FactorMode = Literal["PCA", "ETF"]

VALID_FACTOR_MODES: frozenset[str] = frozenset({"PCA", "ETF"})

DropReason = Literal[
    "insufficient_history",
    "non_stationary",
    "half_life_out_of_band",
    "low_r_squared",
    "degenerate_residual",
]


@dataclass(frozen=True)
class SignalThresholds:
    """Avellaneda-Lee s-score trading rules (spec table 1 defaults).

    Symmetric long/short open and close levels. Defaults match the original
    paper: open at |s| = 1.25, close at |s| = 0.50.
    """

    open_long: float = -1.25
    open_short: float = 1.25
    close_long: float = -0.50
    close_short: float = 0.50

    def __post_init__(self) -> None:
        if self.open_long >= 0 or self.open_short <= 0:
            raise ValueError(
                "open_long must be negative and open_short positive; got "
                f"open_long={self.open_long}, open_short={self.open_short}"
            )
        if self.close_long >= 0 or self.close_short <= 0:
            raise ValueError(
                "close_long must be negative and close_short positive; got "
                f"close_long={self.close_long}, close_short={self.close_short}"
            )
        if self.close_long < self.open_long:
            raise ValueError(
                "close_long must be greater (closer to zero) than open_long"
            )
        if self.close_short > self.open_short:
            raise ValueError(
                "close_short must be less (closer to zero) than open_short"
            )


@dataclass(frozen=True)
class OUFit:
    """AR(1) / Ornstein-Uhlenbeck fit for one stock's cumulative residual.

    All parameters are in trading-day units (`dt = 1`). `s_score` is the
    raw standardized residual; `s_score_mod` subtracts the drift term
    `alpha / (kappa * sigma_eq)` per spec eq. 15.

    Attributes
    ----------
    ticker:
        Stock identifier.
    alpha, beta:
        Per-stock factor-regression intercept and `(K,)` loadings.
    r_squared:
        Factor-regression R^2 over the OU window.
    a, b:
        AR(1) intercept and slope of the cumulative-residual process X.
    kappa, m, sigma:
        OU mean-reversion speed, long-run mean, diffusion (continuous-time).
    sigma_zeta:
        Residual SD of the AR(1) innovations `zeta`.
    sigma_eq:
        Equilibrium dispersion of X around m (`sigma_zeta / sqrt(1 - b^2)`).
    half_life:
        Mean-reversion half-life in trading days: `ln(2) / kappa`.
    X_last:
        Most recent value of the cumulative residual.
    s_score, s_score_mod:
        Raw and drift-corrected Avellaneda-Lee s-scores.
    """

    ticker: str
    alpha: float
    beta: np.ndarray
    r_squared: float
    a: float
    b: float
    kappa: float
    m: float
    sigma: float
    sigma_zeta: float
    sigma_eq: float
    half_life: float
    X_last: float
    s_score: float
    s_score_mod: float

    def __post_init__(self) -> None:
        if self.beta.ndim != 1:
            raise ValueError(
                f"OUFit.beta must be 1-D, got shape {self.beta.shape}"
            )
        if not np.isfinite(self.s_score) or not np.isfinite(self.s_score_mod):
            raise ValueError(
                f"OUFit.s_score(_mod) must be finite, got "
                f"s={self.s_score}, s_mod={self.s_score_mod}"
            )


@dataclass(frozen=True)
class StatArbInputs:
    """Inputs for the Avellaneda-Lee pipeline.

    `returns` is the (T, N) asset returns panel; `factor_returns` is the
    (T, K) factor-return panel — required when `factor_mode == "ETF"` and
    ignored (recomputed via PCA) when `factor_mode == "PCA"`.
    """

    returns: pd.DataFrame
    tickers: tuple[str, ...]
    factor_mode: FactorMode
    K: int
    pca_window: int
    ou_window: int
    timestamp: datetime
    factor_returns: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.factor_mode not in VALID_FACTOR_MODES:
            raise ValueError(
                f"factor_mode must be one of {sorted(VALID_FACTOR_MODES)}, "
                f"got {self.factor_mode!r}"
            )
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.pca_window < 30:
            raise ValueError(
                f"pca_window must be >= 30 trading days, got {self.pca_window}"
            )
        if self.ou_window < 10:
            raise ValueError(
                f"ou_window must be >= 10 trading days, got {self.ou_window}"
            )
        if self.ou_window > self.pca_window:
            raise ValueError(
                f"ou_window ({self.ou_window}) cannot exceed pca_window "
                f"({self.pca_window})"
            )
        if self.returns.shape[1] != len(self.tickers):
            raise ValueError(
                f"returns has {self.returns.shape[1]} columns but "
                f"{len(self.tickers)} tickers were supplied"
            )
        if self.factor_mode == "ETF" and (
            self.factor_returns is None or self.factor_returns.empty
        ):
            raise ValueError(
                "factor_mode='ETF' requires non-empty factor_returns"
            )


@dataclass(frozen=True)
class StatArbFit:
    """Calibration output of `AvellanedaLeeStatArb`.

    `ou_fits` is keyed by ticker for surviving stocks only. `dropped_tickers`
    maps each rejected ticker to the reason — e.g. ``"non_stationary"`` or
    ``"half_life_out_of_band"`` — so downstream diagnostics can attribute
    universe attrition.
    """

    factor_mode: FactorMode
    factor_names: tuple[str, ...]
    universe: tuple[str, ...]
    ou_fits: dict[str, OUFit]
    dropped_tickers: dict[str, DropReason]
    factor_returns: pd.DataFrame
    eigenvalues: np.ndarray | None
    marchenko_pastur_cutoff: float | None
    half_life_band: tuple[float, float]
    min_r_squared: float
    timestamp: datetime

    @property
    def n_surviving(self) -> int:
        return len(self.ou_fits)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_tickers)

    def s_scores(self, *, modified: bool = True) -> dict[str, float]:
        """Cross-sectional dict of s-scores for surviving tickers."""
        return {
            t: (fit.s_score_mod if modified else fit.s_score)
            for t, fit in self.ou_fits.items()
        }
