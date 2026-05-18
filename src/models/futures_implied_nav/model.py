"""`FuturesImpliedNAV` — the BaseModel subclass that ties everything together.

`fetch_data` pulls daily history + latest intraday prints from the `DataProvider`,
`calibrate` builds the overnight-returns panel and fits beta, `predict` runs
the intraday hot loop and produces a `Signal`, `validate` reports calibration
diagnostics.

The model is stateful between calibrate and predict: a fresh instance has no
beta, so `calibrate` must be called before `predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Signal
from src.models.futures_implied_nav.calibration import calibrate as _calibrate_fn
from src.models.futures_implied_nav.signal import (
    action_to_signal_direction,
    compute_result,
    signal_strength,
)
from src.models.futures_implied_nav.types import (
    BetaVector,
    CostParameters,
    FuturesImpliedNAVInputs,
    HedgePanel,
    HedgeSpec,
    NAVAnchor,
    SourceVariances,
)

_DEFAULT_HEDGES: tuple[HedgeSpec, ...] = (
    HedgeSpec(ticker="^N225", kind="futures", currency="JPY"),
    HedgeSpec(ticker="EWJ", kind="equity", currency="USD"),
)
_DEFAULT_FX: tuple[str, ...] = ("JPYUSD=X",)


@dataclass
class FetchedData:
    """Bundle returned by `fetch_data` and consumed by calibrate / predict.

    `daily_history` maps ticker -> daily OHLC DataFrame (indexed/sorted by date).
    `latest_prints` maps ticker -> (timestamp, last close) for the intraday hot loop.
    `nav_anchor` is the NAV strike-time anchor used by `predict`.
    `cost_params` defines the no-arb cost band.
    `etf_mid` is the most recent ETF print.
    """

    etf_ticker: str
    timestamp: datetime
    hedges: tuple[HedgeSpec, ...]
    fx_tickers: tuple[str, ...]
    daily_history: dict[str, pd.DataFrame]
    latest_prints: dict[str, tuple[datetime, float]]
    etf_mid: float
    nav_anchor: NAVAnchor
    cost_params: CostParameters


@register_model
class FuturesImpliedNAV(BaseModel):
    """Multi-source futures-implied fair-value model for international ETFs.

    See `models/layer2_structural/02_futures_implied_nav.md` for the math.
    """

    name: ClassVar[str] = "futures_implied_nav"
    layer: ClassVar[int] = 2
    refit_frequency: ClassVar[RefitFrequency] = "daily"

    def __init__(
        self,
        etf_ticker: str = "EWJ",
        hedges: tuple[HedgeSpec, ...] = _DEFAULT_HEDGES,
        fx_tickers: tuple[str, ...] = _DEFAULT_FX,
        history_days: int = 90,
        cost_params: CostParameters | None = None,
        nav_anchor_override: NAVAnchor | None = None,
    ) -> None:
        if not etf_ticker:
            raise ValueError("etf_ticker must be non-empty")
        if not hedges:
            raise ValueError("hedges must be a non-empty tuple")
        if history_days < 20:
            raise ValueError(f"history_days too small ({history_days}); need >= 20")
        self.etf_ticker = etf_ticker
        self.hedges = hedges
        self.fx_tickers = fx_tickers
        self.history_days = history_days
        self.cost_params = cost_params or CostParameters(
            half_spread_bps=2.0, cost_band_bps=5.0
        )
        self.nav_anchor_override = nav_anchor_override
        self._beta: BetaVector | None = None
        self._variances: SourceVariances | None = None

    # ----- BaseModel interface -----

    def fetch_data(self, provider: DataProvider) -> FetchedData:
        period = f"{self.history_days}d"
        daily: dict[str, pd.DataFrame] = {}
        latest: dict[str, tuple[datetime, float]] = {}

        tickers = (self.etf_ticker, *(h.ticker for h in self.hedges), *self.fx_tickers)
        for ticker in tickers:
            df = provider.fetch_prices(ticker, period=period, interval="1d")
            daily[ticker] = _normalize_history(df)
            latest[ticker] = _latest_price(daily[ticker])

        etf_ts, etf_mid = latest[self.etf_ticker]
        anchor = self.nav_anchor_override or _build_anchor(
            etf_ticker=self.etf_ticker,
            daily=daily,
            hedge_tickers=tuple(h.ticker for h in self.hedges),
            fx_tickers=self.fx_tickers,
        )
        return FetchedData(
            etf_ticker=self.etf_ticker,
            timestamp=etf_ts,
            hedges=self.hedges,
            fx_tickers=self.fx_tickers,
            daily_history=daily,
            latest_prints=latest,
            etf_mid=etf_mid,
            nav_anchor=anchor,
            cost_params=self.cost_params,
        )

    def calibrate(self, data: FetchedData) -> CalibrationResult:
        panel = build_panel_from_daily(
            etf_ticker=data.etf_ticker,
            hedges=data.hedges,
            daily=data.daily_history,
        )
        result = _calibrate_fn(panel)
        beta = result.parameters["beta"]
        variances = result.parameters["variances"]
        assert isinstance(beta, BetaVector)
        assert isinstance(variances, SourceVariances)
        self._beta = beta
        self._variances = variances
        return result

    def predict(self, data: FetchedData) -> Signal:
        if self._beta is None or self._variances is None:
            raise RuntimeError(
                "FuturesImpliedNAV.predict requires calibrate() to have been called first"
            )

        hedge_now = {h.ticker: data.latest_prints[h.ticker][1] for h in data.hedges}
        fx_now = {
            ticker: data.latest_prints[ticker][1] for ticker in data.fx_tickers
        }

        inputs = FuturesImpliedNAVInputs(
            etf_ticker=data.etf_ticker,
            timestamp=data.timestamp,
            hedges=data.hedges,
            hedge_now=hedge_now,
            fx_now=fx_now,
            anchor=data.nav_anchor,
            etf_mid=data.etf_mid,
            beta=self._beta,
            variances=self._variances,
            cost_params=data.cost_params,
        )
        result = compute_result(inputs)
        direction = action_to_signal_direction(result.action)
        strength = signal_strength(result.premium, data.cost_params)
        return Signal(
            ticker=result.etf_ticker,
            direction=direction,
            strength=strength,
            timestamp=result.timestamp,
            horizon="intraday",
            metadata={
                "fair_value": result.fair_value,
                "premium": result.premium,
                "action": result.action,
                "w_hedge": result.decomposition.weights.w_hedge,
                "w_nav": result.decomposition.weights.w_nav,
                "w_etf": result.decomposition.weights.w_etf,
                "r_squared": self._beta.r_squared,
                "cost_band_bps": result.cost_band_bps,
            },
        )

    def validate(self, data: FetchedData) -> dict[str, Any]:
        """Diagnostic snapshot per the spec's "Validation and diagnostics" section."""

        if self._beta is None:
            return {"status": "uncalibrated"}

        panel = build_panel_from_daily(
            etf_ticker=data.etf_ticker,
            hedges=data.hedges,
            daily=data.daily_history,
        )
        nav_close = np.asarray(panel.nav_close, dtype=float)
        nav_open = np.asarray(panel.nav_open, dtype=float)
        r_b = nav_open / nav_close - 1.0
        tickers = self._beta.tickers
        r_h_cols = []
        for ticker in tickers:
            p_close = np.asarray(panel.hedge_close[ticker], dtype=float)
            p_now = np.asarray(panel.hedge_now[ticker], dtype=float)
            r_h_cols.append(p_now / p_close - 1.0)
        r_h = np.column_stack(r_h_cols) if r_h_cols else np.zeros((len(r_b), 0))
        betas = np.asarray(self._beta.betas, dtype=float)
        residuals = r_b - r_h @ betas

        diagnostics: dict[str, Any] = {
            "status": "ok",
            "n_obs": int(len(r_b)),
            "r_squared": self._beta.r_squared,
            "residual_variance": float(np.var(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
            "residual_mean": float(residuals.mean()),
            "residual_max_abs": float(np.max(np.abs(residuals))),
            "beta_sum": float(betas.sum()),
            "beta_within_bounds": bool(np.all((betas >= 0) & (betas <= 1.5))),
            "r_squared_acceptable": self._beta.r_squared >= 0.85,
            "ridge_lambda": self._beta.ridge_lambda,
        }
        # Cross-hedge consistency: rerun fit on disjoint halves of the hedge set.
        if r_h.shape[1] >= 2:
            half = r_h.shape[1] // 2
            beta_a = _quick_fit(r_h[:, :half], r_b)
            beta_b = _quick_fit(r_h[:, half:], r_b)
            pred_a = r_h[:, :half] @ beta_a
            pred_b = r_h[:, half:] @ beta_b
            diagnostics["cross_hedge_rmse_bps"] = float(
                1e4 * np.sqrt(np.mean((pred_a - pred_b) ** 2))
            )
        return diagnostics


def build_panel_from_daily(
    *,
    etf_ticker: str,
    hedges: tuple[HedgeSpec, ...],
    daily: dict[str, pd.DataFrame],
) -> HedgePanel:
    """Construct overnight-return windows from daily bars.

    With daily bars we use the close-of-day d as the NAV strike anchor and
    the close-of-day d+1 as the "next-day open" anchor. The hedge returns
    over the same calendar day are similarly close(d) -> close(d+1). This
    is the close-to-close approximation noted in the README; tests can
    swap in higher-resolution panels by constructing `HedgePanel` directly.
    """

    etf_df = daily[etf_ticker]
    if len(etf_df) < 2:
        raise ValueError(
            f"build_panel_from_daily: need at least 2 days for {etf_ticker}, "
            f"got {len(etf_df)}"
        )
    etf_close = etf_df["Close"].to_numpy(dtype=float)
    etf_dates = etf_df["Date"].to_list()
    n = len(etf_close) - 1
    if n < 1:
        raise ValueError("build_panel_from_daily: degenerate ETF series")

    nav_close_arr = etf_close[:-1]
    nav_open_arr = etf_close[1:]
    dates = tuple(_as_datetime(d) for d in etf_dates[:-1])

    hedge_close_map: dict[str, tuple[float, ...]] = {}
    hedge_now_map: dict[str, tuple[float, ...]] = {}
    for hedge in hedges:
        df = daily[hedge.ticker]
        # Align on Date so missing days drop cleanly.
        aligned = _align_to_dates(df, etf_dates)
        h_close = aligned[:-1]
        h_next = aligned[1:]
        hedge_close_map[hedge.ticker] = tuple(float(x) for x in h_close)
        hedge_now_map[hedge.ticker] = tuple(float(x) for x in h_next)

    return HedgePanel(
        dates=dates,
        nav_close=tuple(float(x) for x in nav_close_arr),
        nav_open=tuple(float(x) for x in nav_open_arr),
        hedge_close=hedge_close_map,
        hedge_now=hedge_now_map,
    )


def _quick_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(0)
    gram = x.T @ x + lam * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a daily-bar DataFrame has Date/Close columns sorted ascending."""

    if df is None or len(df) == 0:
        raise ValueError("provider returned an empty DataFrame")
    out = df.copy()
    if "Date" not in out.columns:
        if "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "Date"})
        else:
            out = out.reset_index()
            if "Date" not in out.columns and "Datetime" in out.columns:
                out = out.rename(columns={"Datetime": "Date"})
            elif "Date" not in out.columns and "index" in out.columns:
                out = out.rename(columns={"index": "Date"})
    if "Close" not in out.columns:
        raise ValueError(f"history DataFrame missing 'Close' column: {list(out.columns)}")
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out = out.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    out = out.reset_index(drop=True)
    return out


def _latest_price(df: pd.DataFrame) -> tuple[datetime, float]:
    if len(df) == 0:
        raise ValueError("_latest_price: empty DataFrame")
    last = df.iloc[-1]
    ts = pd.to_datetime(last["Date"]).to_pydatetime()
    return ts, float(last["Close"])


def _align_to_dates(df: pd.DataFrame, etf_dates: list[Any]) -> np.ndarray:
    """Forward-fill hedge prices onto the ETF date grid; backfill any leading NaN."""

    target = pd.to_datetime(pd.Series(etf_dates)).dt.tz_localize(None)
    aligned = (
        df.set_index("Date")["Close"].reindex(target).ffill().bfill().to_numpy(dtype=float)
    )
    if np.any(~np.isfinite(aligned)):
        raise ValueError("_align_to_dates: still have NaNs after ffill/bfill")
    return aligned


def _build_anchor(
    *,
    etf_ticker: str,
    daily: dict[str, pd.DataFrame],
    hedge_tickers: tuple[str, ...],
    fx_tickers: tuple[str, ...],
) -> NAVAnchor:
    """Use the prior-day ETF close as a NAV proxy when no issuer feed is wired in.

    Hedge / FX strike-time prices are the corresponding prior-day closes.
    """

    etf_df = daily[etf_ticker]
    if len(etf_df) < 2:
        raise ValueError(
            f"_build_anchor needs >= 2 days of ETF history, got {len(etf_df)}"
        )
    prior = etf_df.iloc[-2]
    nav_close = float(prior["Close"])
    strike_time = pd.to_datetime(prior["Date"]).to_pydatetime()
    hedge_close = {t: float(daily[t].iloc[-2]["Close"]) for t in hedge_tickers}
    fx_close = {t: float(daily[t].iloc[-2]["Close"]) for t in fx_tickers}
    return NAVAnchor(
        nav_close=nav_close,
        strike_time=strike_time,
        hedge_close=hedge_close,
        fx_close=fx_close,
    )


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    converted = pd.to_datetime(value).to_pydatetime()
    if not isinstance(converted, datetime):
        raise TypeError(
            f"_as_datetime: pd.to_datetime returned {type(converted).__name__}, not datetime"
        )
    return converted
