"""Dataclasses specific to the factor-model / PCA risk model.

Shared `Signal` / `RiskMetric` / `CalibrationResult` live in `src.core.types`;
this module holds the model-internal shapes that flow between `fetch_data`,
`calibrate`, and `predict`.

Arrays are `numpy.ndarray`s rather than DataFrames once a fit is locked in —
DataFrames are reserved for the data-loading boundary (`FactorInputs.returns`)
because that is where the ticker/date alignment work happens. Downstream
matrix math is happier with raw `ndarray`s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

FactorMethod = Literal["PCA", "FamaFrench", "Barra"]

VALID_METHODS: frozenset[str] = frozenset({"PCA", "FamaFrench", "Barra"})


@dataclass(frozen=True)
class FactorFit:
    """Fitted factor model: exposures, factor covariance, specific risk.

    All arrays are dense `numpy.ndarray`s with rows aligned to `tickers` and
    factor columns aligned to `factor_names`.

    Attributes
    ----------
    B:
        `(N, K)` factor exposures (loadings). For PCA, columns are scaled
        eigenvectors `U_{1:K} * sqrt(lambda_{1:K})`.
    F:
        `(K, K)` factor-return covariance. For PCA this is the identity (PCA
        factors are orthonormal by construction); for Fama-French / Barra it
        is the sample covariance of the factor-return time series.
    specific_variances:
        `(N,)` diagonal of the specific-risk matrix `D`. Always floored at
        `min_specific_variance` to keep downstream inversions well-conditioned.
    tickers / factor_names:
        Row / column labels for `B`. Tuples so the fit is hashable + frozen.
    method:
        Which calibration path produced this fit (PCA / FamaFrench / Barra).
    eigenvalues:
        Full descending eigenvalue spectrum from the sample covariance (PCA
        only). `None` for fundamental factor methods.
    marchenko_pastur_cutoff:
        RMT noise floor `(1 + sqrt(N/T))^2 * sigma^2`. Eigenvalues above this
        are deemed real signal. `None` for non-PCA fits.
    variance_explained:
        Fraction of total variance captured by the kept factors. For PCA this
        is `sum(eigenvalues[:K]) / sum(eigenvalues)`; for fundamental factors
        it is the time-series-regression R^2 averaged across assets.
    min_specific_variance:
        Floor applied to `specific_variances` to guard against zero diagonals.
    """

    B: np.ndarray
    F: np.ndarray
    specific_variances: np.ndarray
    tickers: tuple[str, ...]
    factor_names: tuple[str, ...]
    method: FactorMethod
    eigenvalues: np.ndarray | None = None
    marchenko_pastur_cutoff: float | None = None
    variance_explained: float = 0.0
    min_specific_variance: float = 1e-8

    def __post_init__(self) -> None:
        n, k = self.B.shape
        if n != len(self.tickers):
            raise ValueError(
                f"B has {n} rows but {len(self.tickers)} tickers were supplied"
            )
        if k != len(self.factor_names):
            raise ValueError(
                f"B has {k} columns but {len(self.factor_names)} factor names "
                "were supplied"
            )
        if self.F.shape != (k, k):
            raise ValueError(
                f"F shape {self.F.shape} does not match K={k}"
            )
        if self.specific_variances.shape != (n,):
            raise ValueError(
                f"specific_variances shape {self.specific_variances.shape} "
                f"does not match N={n}"
            )
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"FactorFit.method must be one of {sorted(VALID_METHODS)}, "
                f"got {self.method!r}"
            )

    @property
    def n_assets(self) -> int:
        return int(self.B.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.B.shape[1])

    def asset_index(self, ticker: str) -> int:
        """Return the row index of `ticker` in `B` / `specific_variances`."""
        try:
            return self.tickers.index(ticker)
        except ValueError as exc:
            raise KeyError(
                f"Ticker {ticker!r} not in fit universe {self.tickers!r}"
            ) from exc


@dataclass(frozen=True)
class FactorInputs:
    """Everything `calibrate`, `predict`, and `validate` consume.

    Produced by `FactorModelPCA.fetch_data`. The returns panel is the only
    universally-required field; `factor_returns` is required for the
    Fama-French path and `exposures` for Barra.
    """

    returns: pd.DataFrame
    tickers: tuple[str, ...]
    method: FactorMethod
    K: int | None
    timestamp: datetime
    factor_returns: pd.DataFrame | None = None
    exposures: pd.DataFrame | None = None
    weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"FactorInputs.method must be one of {sorted(VALID_METHODS)}, "
                f"got {self.method!r}"
            )
        n_cols = self.returns.shape[1]
        if n_cols != len(self.tickers):
            raise ValueError(
                f"returns has {n_cols} columns but {len(self.tickers)} "
                "tickers were supplied"
            )
        if self.weights is not None and self.weights.shape != (n_cols,):
            raise ValueError(
                f"weights shape {self.weights.shape} does not match N={n_cols}"
            )


@dataclass(frozen=True)
class PortfolioRisk:
    """Decomposition of portfolio variance into systematic + specific parts.

    All `*_vol` fields are standard deviations (square root of the variance).
    `factor_exposures` is `B^T w` and `factor_contributions[k]` is the variance
    that factor `k` contributes to the portfolio (so the K-vector sums to
    `systematic_vol ** 2`).
    """

    total_vol: float
    systematic_vol: float
    specific_vol: float
    factor_exposures: np.ndarray
    factor_contributions: np.ndarray

    def __post_init__(self) -> None:
        if self.total_vol < 0 or self.systematic_vol < 0 or self.specific_vol < 0:
            raise ValueError("PortfolioRisk volatilities must be non-negative")
