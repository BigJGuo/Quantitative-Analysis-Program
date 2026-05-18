"""`AvellanedaLeeStatArb` — Layer 4 signal model.

Thin orchestration shell. `fetch_data` builds a returns panel for the user
universe (and, in ETF mode, a sector-ETF factor panel) through the shared
``DataProvider``; `calibrate` runs the spec's per-stock pipeline; `predict`
emits a `Signal` for a focal ticker; `predict_all` returns the full cross
section. Pure math is delegated to `signal.py` and `calibration.py`.
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
from src.core.types import CalibrationResult, Signal, SignalDirection
from src.models.avellaneda_lee_stat_arb.calibration import (
    DEFAULT_HALF_LIFE_BAND,
    DEFAULT_MIN_R_SQUARED,
    MODEL_NAME,
)
from src.models.avellaneda_lee_stat_arb.calibration import calibrate as _calibrate_impl
from src.models.avellaneda_lee_stat_arb.signal import (
    POSITION_FLAT,
    POSITION_LONG,
    POSITION_SHORT,
    adf_test_pvalue,
    cumulative_residual,
    factor_neutral_hedge,
    factor_regression,
    position_decision,
)
from src.models.avellaneda_lee_stat_arb.types import (
    FactorMode,
    OUFit,
    SignalThresholds,
    StatArbFit,
    StatArbInputs,
)

# US sector-SPDR ETFs used as the canonical ETF-mode factor panel.
DEFAULT_SECTOR_ETFS: tuple[str, ...] = (
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "XLI",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
)

_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_HORIZON: str = "5d"  # Avellaneda-Lee median holding period


@register_model
class AvellanedaLeeStatArb(BaseModel):
    """PCA-residual statistical-arbitrage signal model.

    Parameters
    ----------
    universe:
        Tickers to trade. Order is preserved.
    K:
        Number of factors (PCA mode) or number of supplied sector ETFs to
        retain (ETF mode).
    factor_mode:
        ``"PCA"`` (default) computes eigenportfolios from the universe;
        ``"ETF"`` uses pre-fetched sector-ETF returns.
    sector_etfs:
        ETF tickers used in ``factor_mode="ETF"``. Defaults to the 11 SPDR
        sector ETFs.
    pca_window:
        Rolling window length for the factor regression / PCA. Spec default
        is 252 trading days.
    ou_window:
        Rolling window length for the OU / AR(1) fit on cumulative residuals.
        Spec default is 60 trading days.
    half_life_band:
        `(lo, hi)` in trading days; stocks outside the band are dropped.
    min_r_squared:
        Minimum factor-regression R^2 over the PCA window.
    use_drift_correction:
        Whether to apply the spec's drift correction to the s-score.
    thresholds:
        Open/close thresholds for the position state machine. Defaults to
        the spec's `+/-1.25` open and `+/-0.50` close.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "daily"

    def __init__(
        self,
        universe: Sequence[str],
        *,
        K: int = 5,
        factor_mode: FactorMode = "PCA",
        sector_etfs: Sequence[str] | None = None,
        pca_window: int = 252,
        ou_window: int = 60,
        half_life_band: tuple[float, float] = DEFAULT_HALF_LIFE_BAND,
        min_r_squared: float = DEFAULT_MIN_R_SQUARED,
        use_drift_correction: bool = True,
        thresholds: SignalThresholds | None = None,
    ) -> None:
        if len(universe) < 2:
            raise ValueError(
                f"AvellanedaLeeStatArb requires at least 2 tickers, got {len(universe)}"
            )
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if pca_window < 30:
            raise ValueError(f"pca_window must be >= 30, got {pca_window}")
        if ou_window < 10:
            raise ValueError(f"ou_window must be >= 10, got {ou_window}")
        if ou_window > pca_window:
            raise ValueError(
                f"ou_window ({ou_window}) cannot exceed pca_window ({pca_window})"
            )
        self.universe: tuple[str, ...] = tuple(universe)
        self.K: int = K
        self.factor_mode: FactorMode = factor_mode
        self.sector_etfs: tuple[str, ...] = (
            tuple(sector_etfs) if sector_etfs is not None else DEFAULT_SECTOR_ETFS
        )
        self.pca_window: int = pca_window
        self.ou_window: int = ou_window
        self.half_life_band: tuple[float, float] = half_life_band
        self.min_r_squared: float = min_r_squared
        self.use_drift_correction: bool = use_drift_correction
        self.thresholds: SignalThresholds = thresholds or SignalThresholds()
        self._fit: StatArbFit | None = None
        self._prev_positions: dict[str, int] = {}

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> StatArbInputs:
        period = _period_from_window(self.pca_window)
        returns_by_ticker: dict[str, pd.Series] = {}
        for ticker in self.universe:
            bars = provider.fetch_prices(ticker, period, _DEFAULT_INTERVAL)
            series = _bars_to_log_returns(bars)
            if series is not None and not series.empty:
                returns_by_ticker[ticker] = series

        if not returns_by_ticker:
            raise RuntimeError(
                f"No usable returns for universe {self.universe!r}; every "
                "ticker returned empty bars."
            )

        panel = pd.DataFrame(returns_by_ticker)
        panel = panel.dropna(how="all")
        # Keep tickers with at least pca_window valid rows after alignment.
        min_required = self.pca_window
        keep_cols = [
            c for c in panel.columns if panel[c].notna().sum() >= min_required
        ]
        panel = panel.loc[:, keep_cols].dropna(how="any")
        if panel.shape[0] < self.pca_window:
            raise RuntimeError(
                f"Aligned panel has only {panel.shape[0]} rows; need "
                f"pca_window={self.pca_window}."
            )

        kept_tickers = tuple(str(c) for c in panel.columns)

        factor_returns: pd.DataFrame | None = None
        if self.factor_mode == "ETF":
            etf_series: dict[str, pd.Series] = {}
            for etf in self.sector_etfs:
                bars = provider.fetch_prices(etf, period, _DEFAULT_INTERVAL)
                series = _bars_to_log_returns(bars)
                if series is not None and not series.empty:
                    etf_series[etf] = series
            if not etf_series:
                raise RuntimeError(
                    "ETF factor mode requires at least one sector ETF return "
                    "series; provider returned none."
                )
            factor_returns = (
                pd.DataFrame(etf_series).reindex(panel.index).dropna(how="any")
            )
            # Drop returns rows that don't have matching ETF data.
            panel = panel.loc[factor_returns.index]

        return StatArbInputs(
            returns=panel,
            tickers=kept_tickers,
            factor_mode=self.factor_mode,
            K=self.K,
            pca_window=self.pca_window,
            ou_window=self.ou_window,
            timestamp=datetime.now(UTC),
            factor_returns=factor_returns,
            metadata={
                "period": period,
                "n_dropped_universe": len(self.universe) - len(kept_tickers),
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, StatArbInputs):
            raise TypeError(
                f"AvellanedaLeeStatArb.calibrate expects StatArbInputs, got "
                f"{type(data).__name__}"
            )
        result = _calibrate_impl(
            data,
            half_life_band=self.half_life_band,
            min_r_squared=self.min_r_squared,
            use_drift_correction=self.use_drift_correction,
        )
        self._fit = cast(StatArbFit, result.parameters["stat_arb_fit"])
        return result

    def predict(self, data: Any) -> Signal:
        """Emit the signal for the *focal* ticker (universe[0] by default).

        For the full cross-section, call `predict_all(data)`. The choice to
        return a single `Signal` here is to fit the `BaseModel.predict`
        contract (returns one `Signal | Forecast | RiskMetric`).
        """

        if not isinstance(data, StatArbInputs):
            raise TypeError(
                f"AvellanedaLeeStatArb.predict expects StatArbInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        focal = self._focal_ticker(fit)
        return self._signal_for(focal, fit, data.timestamp)

    def validate(self, data: Any) -> dict[str, Any]:
        """Spec validation: residual stationarity, half-life distribution,
        s-score distribution, factor coverage."""

        if not isinstance(data, StatArbInputs):
            raise TypeError(
                f"AvellanedaLeeStatArb.validate expects StatArbInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._require_fit()
        if not fit.ou_fits:
            return {
                "n_surviving": 0,
                "n_dropped": fit.n_dropped,
                "drop_reasons": _count_drop_reasons(fit),
            }

        s_values = np.array(
            [f.s_score_mod for f in fit.ou_fits.values()], dtype=float
        )
        half_lives = np.array(
            [f.half_life for f in fit.ou_fits.values()], dtype=float
        )
        r_squareds = np.array(
            [f.r_squared for f in fit.ou_fits.values()], dtype=float
        )

        # ADF p-values for each surviving stock's cumulative residual.
        pca_panel = data.returns.iloc[-data.pca_window :]
        factor_returns = fit.factor_returns
        adf_pvalues: dict[str, float] = {}
        for ticker, ou_fit in fit.ou_fits.items():
            try:
                idx = data.tickers.index(ticker)
            except ValueError:
                continue
            asset_returns = pca_panel.iloc[:, idx].to_numpy(dtype=float)
            reg = factor_regression(
                asset_returns, factor_returns.to_numpy(dtype=float)
            )
            ou_resid = reg.residuals[-data.ou_window :]
            X = cumulative_residual(ou_resid)
            adf_pvalues[ticker] = adf_test_pvalue(X)
            # Tie back to the stored OU fit so we don't drift between fits.
            _ = ou_fit

        out: dict[str, Any] = {
            "n_surviving": fit.n_surviving,
            "n_dropped": fit.n_dropped,
            "drop_reasons": _count_drop_reasons(fit),
            "s_score_mean": float(s_values.mean()),
            "s_score_std": float(s_values.std(ddof=1)) if s_values.size > 1 else 0.0,
            "s_score_skew": _skewness(s_values),
            "s_score_kurtosis": _kurtosis(s_values),
            "half_life_mean": float(half_lives.mean()),
            "half_life_median": float(np.median(half_lives)),
            "r_squared_mean": float(r_squareds.mean()),
            "n_long_candidates": int(np.sum(s_values < self.thresholds.open_long)),
            "n_short_candidates": int(np.sum(s_values > self.thresholds.open_short)),
            "adf_pvalues": adf_pvalues,
            "adf_pvalue_median": float(np.median(list(adf_pvalues.values())))
            if adf_pvalues
            else float("nan"),
        }
        if fit.eigenvalues is not None and fit.eigenvalues.size > 0:
            ev = fit.eigenvalues
            total = float(ev.sum())
            if total > 0:
                out["variance_explained_top_K"] = float(
                    ev[: self.K].sum() / total
                )
            out["top_eigenvalue"] = float(ev[0])
        if fit.marchenko_pastur_cutoff is not None:
            out["mp_upper_edge"] = float(fit.marchenko_pastur_cutoff)
        return out

    # ----- Public conveniences ------------------------------------------------

    @property
    def fit(self) -> StatArbFit:
        return self._require_fit()

    def predict_all(self, data: StatArbInputs) -> list[Signal]:
        """Emit one `Signal` per surviving ticker (cross-section)."""

        fit = self._require_fit()
        return [
            self._signal_for(t, fit, data.timestamp) for t in fit.ou_fits
        ]

    def aggregate_factor_exposure(self) -> np.ndarray:
        """Spec step 7: factor exposure of the current positions.

        Uses the model's cached `_prev_positions` (updated by `predict` /
        `predict_all`). Returns a `(K,)` exposure vector; the caller takes
        `-exposure` units of each factor portfolio to neutralize.
        """

        fit = self._require_fit()
        betas = {t: f.beta for t, f in fit.ou_fits.items()}
        positions = {
            t: self._prev_positions.get(t, POSITION_FLAT) for t in fit.ou_fits
        }
        return factor_neutral_hedge(positions, betas)

    def reset_positions(self) -> None:
        """Clear cached prior positions so the state machine restarts flat."""

        self._prev_positions = {}

    # ----- Internal helpers ---------------------------------------------------

    def _require_fit(self) -> StatArbFit:
        if self._fit is None:
            raise RuntimeError(
                "AvellanedaLeeStatArb has not been calibrated. Call "
                "calibrate(data) first."
            )
        return self._fit

    def _focal_ticker(self, fit: StatArbFit) -> str:
        """First surviving ticker in the original universe order, else first
        in the universe (with a flat signal)."""

        for t in self.universe:
            if t in fit.ou_fits:
                return t
        return self.universe[0]

    def _signal_for(
        self, ticker: str, fit: StatArbFit, timestamp: datetime
    ) -> Signal:
        ou_fit: OUFit | None = fit.ou_fits.get(ticker)
        prev = self._prev_positions.get(ticker, POSITION_FLAT)
        if ou_fit is None:
            # Surviving universe didn't include this ticker — emit flat.
            direction: SignalDirection = "flat"
            strength = 0.0
            new_position = POSITION_FLAT
            metadata: dict[str, Any] = {
                "reason": fit.dropped_tickers.get(ticker, "not_in_universe"),
                "prev_position": prev,
            }
        else:
            new_position = position_decision(
                ou_fit.s_score_mod,
                prev,
                open_long=self.thresholds.open_long,
                open_short=self.thresholds.open_short,
                close_long=self.thresholds.close_long,
                close_short=self.thresholds.close_short,
            )
            direction = _direction_from_position(new_position)
            # Strength scales |s-score| into [0, 1] via a saturating ramp at
            # |s| = 3 (3 sigma is effectively saturated under stationary OU).
            strength = float(min(abs(ou_fit.s_score_mod) / 3.0, 1.0))
            metadata = {
                "s_score": ou_fit.s_score,
                "s_score_mod": ou_fit.s_score_mod,
                "kappa": ou_fit.kappa,
                "half_life": ou_fit.half_life,
                "sigma_eq": ou_fit.sigma_eq,
                "alpha": ou_fit.alpha,
                "r_squared": ou_fit.r_squared,
                "beta": ou_fit.beta.tolist(),
                "factor_names": list(fit.factor_names),
                "prev_position": prev,
                "new_position": new_position,
            }
        self._prev_positions[ticker] = new_position
        return Signal(
            ticker=ticker,
            direction=direction,
            strength=strength,
            timestamp=timestamp,
            horizon=_DEFAULT_HORIZON,
            metadata=metadata,
        )


# ---- module-level helpers ---------------------------------------------------


def _direction_from_position(position: int) -> SignalDirection:
    if position == POSITION_LONG:
        return "long"
    if position == POSITION_SHORT:
        return "short"
    return "flat"


def _period_from_window(window_days: int) -> str:
    """Map a trading-day window to a yfinance ``period`` string.

    The caller wants at least `window_days` trading rows; we pad generously
    (1.5x calendar days) and pick the smallest preset that covers it.
    """

    calendar_days = int(window_days * 1.6) + 30
    if calendar_days <= 90:
        return "3mo"
    if calendar_days <= 180:
        return "6mo"
    if calendar_days <= 365:
        return "1y"
    if calendar_days <= 730:
        return "2y"
    return "5y"


def _bars_to_log_returns(bars: pd.DataFrame) -> pd.Series | None:
    """Convert a yfinance OHLCV frame to a log-return Series indexed by date."""

    if bars is None or bars.empty or "Close" not in bars.columns:
        return None
    close = bars["Close"].astype(float)
    if "Date" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Date"]))
    elif "Datetime" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Datetime"]))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(bars.index))
    close.index = index
    close = close.sort_index()
    log_close = pd.Series(np.log(close.to_numpy(dtype=float)), index=close.index)
    log_r = log_close.diff().dropna()
    return cast(pd.Series, log_r)


def _count_drop_reasons(fit: StatArbFit) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in fit.dropped_tickers.values():
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s <= 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s**3))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher convention: normal -> 0)."""

    if x.size < 4:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s <= 0:
        return 0.0
    return float(((x - m) ** 4).mean() / (s**4) - 3.0)


__all__ = [
    "DEFAULT_SECTOR_ETFS",
    "AvellanedaLeeStatArb",
]
