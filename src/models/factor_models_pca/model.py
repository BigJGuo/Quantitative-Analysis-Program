"""`FactorModelPCA` — Layer 2 structural / cross-sectional risk model.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. The class fetches a returns panel for `universe`, calls
`calibrate` to fit a `FactorFit`, and exposes risk decomposition + signal
residualization to downstream consumers (Avellaneda-Lee stat-arb, the
portfolio layer, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.factor_models_pca.calibration import MODEL_NAME
from src.models.factor_models_pca.calibration import calibrate as _calibrate_impl
from src.models.factor_models_pca.signal import (
    align_returns_panel,
    compute_full_covariance,
    compute_log_returns,
    portfolio_risk,
    residual_autocorrelation,
    residual_max_cross_corr,
    residual_returns,
    residualize,
    variance_explained_curve,
)
from src.models.factor_models_pca.types import (
    FactorFit,
    FactorInputs,
    FactorMethod,
    PortfolioRisk,
)

_DEFAULT_PERIOD: str = "2y"
_DEFAULT_INTERVAL: str = "1d"
_PORTFOLIO_TICKER: str = "PORTFOLIO"
_DEFAULT_HORIZON: str = "1d"
_TRADING_DAYS_PER_YEAR: int = 252


@register_model
class FactorModelPCA(BaseModel):
    """Cross-sectional factor risk model for an equity universe.

    Parameters
    ----------
    universe:
        Tickers to model. Order is preserved across all output matrices.
    K:
        Number of factors to keep (PCA). For Fama-French / Barra the factor
        count is inferred from the supplied panels.
    method:
        ``"PCA"`` (default), ``"FamaFrench"``, or ``"Barra"``.
    lookback_days:
        Approximate calendar window of historical returns to pull. yfinance
        is queried with `period="{lookback_days // 21}mo"` capped at 24mo.
    use_mp_cutoff:
        When True (PCA only), cap the kept factor count at the Marchenko-
        Pastur upper-edge count.
    standardize:
        Optionally divide each return column by its sample std before PCA.
    winsorize_quantile:
        Two-sided winsorization quantile applied to each return column
        during panel construction. `None` to disable.
    weights:
        Optional portfolio weight vector (length ``len(universe)``).
        ``None`` -> equal-weighted long.
    annualize:
        Multiply per-asset / portfolio vol by `sqrt(252)` when reporting.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 2
    refit_frequency: ClassVar[RefitFrequency] = "weekly"

    def __init__(
        self,
        universe: Sequence[str],
        *,
        K: int | None = 5,
        method: FactorMethod = "PCA",
        lookback_days: int = 252,
        use_mp_cutoff: bool = False,
        standardize: bool = False,
        winsorize_quantile: float | None = 0.01,
        min_coverage: float = 0.95,
        weights: np.ndarray | None = None,
        annualize: bool = True,
    ) -> None:
        if len(universe) < 2:
            raise ValueError(
                f"FactorModelPCA requires at least 2 tickers, got {len(universe)}"
            )
        if K is not None and K < 1:
            raise ValueError(f"K must be a positive integer or None, got {K}")
        if lookback_days < 30:
            raise ValueError(
                f"lookback_days must be >= 30 to fit a covariance, got {lookback_days}"
            )
        self.universe: tuple[str, ...] = tuple(universe)
        self.K: int | None = K
        self.method: FactorMethod = method
        self.lookback_days: int = lookback_days
        self.use_mp_cutoff: bool = use_mp_cutoff
        self.standardize: bool = standardize
        self.winsorize_quantile: float | None = winsorize_quantile
        self.min_coverage: float = min_coverage
        self.weights: np.ndarray | None = weights.copy() if weights is not None else None
        self.annualize: bool = annualize
        self._fit: FactorFit | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> FactorInputs:
        period = _period_from_lookback(self.lookback_days)
        returns_by_ticker: dict[str, pd.Series] = {}
        for ticker in self.universe:
            bars = provider.fetch_prices(ticker, period, _DEFAULT_INTERVAL)
            series = _bars_to_log_returns(bars)
            if series is not None:
                returns_by_ticker[ticker] = series

        panel = align_returns_panel(
            returns_by_ticker,
            min_coverage=self.min_coverage,
            winsorize_quantile=self.winsorize_quantile,
        )
        if panel.empty:
            raise RuntimeError(
                f"No usable returns for universe {self.universe!r} — every "
                "ticker fell below the coverage threshold or returned empty bars."
            )

        # `align_returns_panel` may drop columns whose coverage is too low;
        # the kept tickers (in original order) become the model's effective
        # universe for this fit.
        kept_tickers = tuple(str(c) for c in panel.columns)
        return FactorInputs(
            returns=panel,
            tickers=kept_tickers,
            method=self.method,
            K=self.K,
            timestamp=datetime.now(UTC),
            factor_returns=None,
            exposures=None,
            weights=self._weights_for(kept_tickers),
            metadata={"period": period, "n_dropped": len(self.universe) - len(kept_tickers)},
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, FactorInputs):
            raise TypeError(
                f"FactorModelPCA.calibrate expects FactorInputs, got "
                f"{type(data).__name__}"
            )
        result = _calibrate_impl(
            returns=data.returns,
            method=data.method,
            K=data.K,
            use_mp_cutoff=self.use_mp_cutoff,
            standardize=self.standardize,
            factor_returns=data.factor_returns,
            exposures_by_date=None,  # Barra exposures aren't wired through fetch_data yet
            timestamp=data.timestamp,
        )
        self._fit = cast(FactorFit, result.parameters["factor_fit"])
        return result

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, FactorInputs):
            raise TypeError(
                f"FactorModelPCA.predict expects FactorInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        weights = data.weights if data.weights is not None else _equal_weights(fit.n_assets)
        risk = portfolio_risk(weights, fit)
        scale = float(np.sqrt(_TRADING_DAYS_PER_YEAR)) if self.annualize else 1.0
        horizon = "1y" if self.annualize else _DEFAULT_HORIZON
        return RiskMetric(
            ticker=_PORTFOLIO_TICKER,
            metric_name="portfolio_volatility",
            value=float(risk.total_vol * scale),
            timestamp=data.timestamp,
            confidence_level=None,
            horizon=horizon,
            metadata={
                "systematic_vol": float(risk.systematic_vol * scale),
                "specific_vol": float(risk.specific_vol * scale),
                "n_assets": float(fit.n_assets),
                "n_factors": float(fit.n_factors),
                "method": fit.method,
                "annualized": self.annualize,
                "factor_names": list(fit.factor_names),
                "factor_exposures": risk.factor_exposures.tolist(),
                "factor_contributions": risk.factor_contributions.tolist(),
                "universe": list(fit.tickers),
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, FactorInputs):
            raise TypeError(
                f"FactorModelPCA.validate expects FactorInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        resid = residual_returns(data.returns, fit)
        autocorr = residual_autocorrelation(resid, lag=1)
        max_cross_corr = residual_max_cross_corr(resid)
        out: dict[str, Any] = {
            "n_assets": fit.n_assets,
            "n_factors": fit.n_factors,
            "method": fit.method,
            "variance_explained": fit.variance_explained,
            "mean_specific_vol": float(np.sqrt(fit.specific_variances).mean()),
            "max_residual_abs_autocorrelation_lag1": float(np.nanmax(np.abs(autocorr))),
            "max_residual_cross_correlation": max_cross_corr,
            "min_specific_variance": float(fit.specific_variances.min()),
        }
        if fit.eigenvalues is not None:
            ve_curve = variance_explained_curve(fit.eigenvalues)
            out["top_eigenvalue"] = float(fit.eigenvalues[0])
            out["cumulative_variance_explained"] = ve_curve.tolist()
        if fit.marchenko_pastur_cutoff is not None:
            out["mp_upper_edge"] = float(fit.marchenko_pastur_cutoff)
            if fit.eigenvalues is not None:
                out["n_factors_above_mp"] = int(
                    np.sum(fit.eigenvalues > fit.marchenko_pastur_cutoff)
                )
        return out

    # ----- Public conveniences for downstream models -------------------------

    @property
    def fit(self) -> FactorFit:
        return self._require_fit()

    def covariance(self) -> np.ndarray:
        """Full asset-return covariance `B F B^T + D`."""

        return compute_full_covariance(self._require_fit())

    def portfolio_risk(self, weights: np.ndarray) -> PortfolioRisk:
        """Decompose portfolio variance into systematic + specific."""

        return portfolio_risk(weights, self._require_fit())

    def residualize_signal(self, signal_values: np.ndarray) -> np.ndarray:
        """Project an N-vector signal onto the factor null-space."""

        return residualize(signal_values, self._require_fit())

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Per-asset residual returns given a returns panel."""

        return residual_returns(returns, self._require_fit())

    # ----- Internal helpers --------------------------------------------------

    def _require_fit(self) -> FactorFit:
        if self._fit is None:
            raise RuntimeError(
                "FactorModelPCA has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    def _weights_for(self, tickers: tuple[str, ...]) -> np.ndarray | None:
        """Align the user-supplied weights to the kept-ticker subset."""

        if self.weights is None:
            return None
        if self.weights.shape != (len(self.universe),):
            raise ValueError(
                f"weights shape {self.weights.shape} does not match universe "
                f"size {len(self.universe)}"
            )
        index_map = {t: i for i, t in enumerate(self.universe)}
        return np.array([self.weights[index_map[t]] for t in tickers], dtype=float)


# ---- module-level helpers ---------------------------------------------------


def _equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=float)


def _period_from_lookback(lookback_days: int) -> str:
    """Map a calendar-day window to a yfinance ``period`` string.

    yfinance accepts a small set of preset periods; we pick the smallest one
    that comfortably covers `lookback_days` plus a buffer. Anything past two
    years is clipped to ``2y`` (the default) because longer panels make the
    rolling-window assumption less defensible anyway.
    """

    days = lookback_days + 30
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    return "2y"


def _bars_to_log_returns(bars: pd.DataFrame) -> pd.Series | None:
    """Convert a yfinance OHLCV frame to a log-return Series indexed by date."""

    if bars is None or bars.empty or "Close" not in bars.columns:
        return None
    close_arr = bars["Close"].astype(float).to_numpy()
    if "Date" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Date"]))
    elif "Datetime" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Datetime"]))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(bars.index))
    close = pd.Series(close_arr, index=index).sort_index()
    log_r = pd.Series(np.log(close.to_numpy()), index=close.index).diff().dropna()
    return log_r


__all__ = [
    "FactorModelPCA",
    "compute_log_returns",  # re-export for downstream convenience
]
