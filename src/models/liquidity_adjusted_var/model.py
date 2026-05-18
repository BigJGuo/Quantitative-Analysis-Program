"""`LiquidityAdjustedVaR` — Layer 6 risk model.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. The class is portfolio-level: one instance owns a dict
of ``{ticker -> dollar position}`` and produces a single `RiskMetric`
whose ``.value`` is the L-VaR (in dollars), with per-name diagnostics in
``metadata``.

Spec: ``models/layer6_risk/20_liquidity_adjusted_var.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.liquidity_adjusted_var.calibration import MODEL_NAME
from src.models.liquidity_adjusted_var.calibration import calibrate as _calibrate_impl
from src.models.liquidity_adjusted_var.signal import (
    compute_arithmetic_returns,
    compute_l_var,
    concentration_ratio,
    horizon_coverage_violations,
    stress_spread_l_var,
)
from src.models.liquidity_adjusted_var.types import (
    LiquidityVaRInputs,
    LVaRResult,
    Position,
    TickerLiquidityStats,
)

_DEFAULT_PERIOD: str = "2y"
_DEFAULT_INTERVAL: str = "1d"
_PORTFOLIO_TICKER: str = "PORTFOLIO"
_DEFAULT_ALPHA: float = 0.975
_DEFAULT_MAX_PARTICIPATION: float = 0.15
_DEFAULT_ADV_LOOKBACK: int = 60
_DEFAULT_SPREAD_FALLBACK_LOOKBACK: int = 20
_DEFAULT_AC_IMPACT_COEF: float = 0.1
_DEFAULT_STRESS_MULTIPLIER: float = 5.0


@register_model
class LiquidityAdjustedVaR(BaseModel):
    """Portfolio-level Liquidity-Adjusted VaR.

    Parameters
    ----------
    positions:
        ``{ticker -> signed dollar position}``. Order is preserved on
        Python 3.7+; the model materializes ``tuple[Position, ...]`` from
        it internally.
    alpha:
        Confidence level for VaR / ES. Defaults to the FRTB level 0.975;
        0.99 is also common.
    max_participation:
        Maximum fraction of ADV the desk is willing to trade per day when
        sizing the per-name liquidation horizon. Typical 0.10–0.20.
    history_period:
        yfinance ``period`` argument for the price / volume history pull.
        Defaults to ``"2y"`` per spec STEP 1.
    adv_lookback:
        Window over which average daily volume is computed (default 60d).
    spread_fallback_lookback:
        Window for the high/low range proxy when bid/ask is missing.
    ac_impact_coef:
        Scaling constant in the Almgren-Chriss eta heuristic
        (``eta = c * sigma_dollar / ADV_dollar``).
    stress_multiplier:
        Multiplier applied to all spreads in the stress diagnostic
        (default 5× per spec validation 5).
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 6
    refit_frequency: ClassVar[RefitFrequency] = "daily"

    def __init__(
        self,
        positions: Mapping[str, float],
        *,
        alpha: float = _DEFAULT_ALPHA,
        max_participation: float = _DEFAULT_MAX_PARTICIPATION,
        history_period: str = _DEFAULT_PERIOD,
        adv_lookback: int = _DEFAULT_ADV_LOOKBACK,
        spread_fallback_lookback: int = _DEFAULT_SPREAD_FALLBACK_LOOKBACK,
        ac_impact_coef: float = _DEFAULT_AC_IMPACT_COEF,
        stress_multiplier: float = _DEFAULT_STRESS_MULTIPLIER,
        now_func: Any = None,
    ) -> None:
        if not positions:
            raise ValueError("LiquidityAdjustedVaR requires at least one position")
        if not 0.5 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0.5, 1.0), got {alpha}")
        if not 0.0 < max_participation <= 1.0:
            raise ValueError(
                f"max_participation must lie in (0, 1], got {max_participation}"
            )
        if adv_lookback < 1:
            raise ValueError(f"adv_lookback must be >= 1, got {adv_lookback}")
        if spread_fallback_lookback < 1:
            raise ValueError(
                f"spread_fallback_lookback must be >= 1, "
                f"got {spread_fallback_lookback}"
            )
        if ac_impact_coef < 0.0:
            raise ValueError(f"ac_impact_coef must be >= 0, got {ac_impact_coef}")
        if stress_multiplier <= 0.0:
            raise ValueError(
                f"stress_multiplier must be > 0, got {stress_multiplier}"
            )

        self.positions: tuple[Position, ...] = tuple(
            Position(ticker=t, dollar_value=float(v)) for t, v in positions.items()
        )
        self.alpha: float = float(alpha)
        self.max_participation: float = float(max_participation)
        self.history_period: str = history_period
        self.adv_lookback: int = int(adv_lookback)
        self.spread_fallback_lookback: int = int(spread_fallback_lookback)
        self.ac_impact_coef: float = float(ac_impact_coef)
        self.stress_multiplier: float = float(stress_multiplier)
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._stats: tuple[TickerLiquidityStats, ...] | None = None

    # ---- BaseModel interface -------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> LiquidityVaRInputs:
        """Pull price / volume bars + fundamentals for every position.

        Each ticker is fetched independently via the DataProvider; results
        are stitched into wide ``pd.DataFrame``s aligned on the trading-day
        calendar and a fundamentals dict keyed by ticker.
        """

        tickers = [p.ticker for p in self.positions]
        close_by_ticker: dict[str, pd.Series] = {}
        volume_by_ticker: dict[str, pd.Series] = {}
        high_by_ticker: dict[str, pd.Series] = {}
        low_by_ticker: dict[str, pd.Series] = {}
        for ticker in tickers:
            bars = provider.fetch_prices(ticker, self.history_period, _DEFAULT_INTERVAL)
            if bars is None or bars.empty or "Close" not in bars.columns:
                raise RuntimeError(
                    f"No daily price history returned for {ticker!r}; "
                    "cannot compute liquidity-adjusted VaR."
                )
            index = _index_from_bars(bars)
            close_by_ticker[ticker] = pd.Series(
                bars["Close"].astype(float).to_numpy(), index=index
            )
            if "Volume" in bars.columns:
                volume_by_ticker[ticker] = pd.Series(
                    bars["Volume"].astype(float).to_numpy(), index=index
                )
            else:
                raise RuntimeError(
                    f"Volume column missing for {ticker!r}; cannot compute ADV."
                )
            if "High" in bars.columns and "Low" in bars.columns:
                high_by_ticker[ticker] = pd.Series(
                    bars["High"].astype(float).to_numpy(), index=index
                )
                low_by_ticker[ticker] = pd.Series(
                    bars["Low"].astype(float).to_numpy(), index=index
                )

        prices = pd.DataFrame(close_by_ticker).sort_index()
        volumes = pd.DataFrame(volume_by_ticker).reindex(prices.index)
        highs = (
            pd.DataFrame(high_by_ticker).reindex(prices.index)
            if high_by_ticker
            else None
        )
        lows = (
            pd.DataFrame(low_by_ticker).reindex(prices.index)
            if low_by_ticker
            else None
        )

        fundamentals: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            try:
                fundamentals[ticker] = dict(provider.fetch_fundamentals(ticker) or {})
            except Exception:  # noqa: BLE001
                fundamentals[ticker] = {}

        timestamp = self._now_func()
        cal = _calibrate_impl(
            positions=self.positions,
            prices=prices[tickers],
            volumes=volumes[tickers],
            highs=highs[tickers] if highs is not None else None,
            lows=lows[tickers] if lows is not None else None,
            fundamentals=fundamentals,
            max_participation=self.max_participation,
            adv_lookback=self.adv_lookback,
            spread_fallback_lookback=self.spread_fallback_lookback,
            ac_impact_coef=self.ac_impact_coef,
            timestamp=timestamp,
        )
        ticker_stats = cast(
            tuple[TickerLiquidityStats, ...], cal.parameters["ticker_stats"]
        )
        returns_panel = compute_arithmetic_returns(prices[tickers])
        if returns_panel.empty:
            raise RuntimeError(
                "No usable return rows after pct_change; cannot run L-VaR."
            )

        return LiquidityVaRInputs(
            positions=self.positions,
            alpha=self.alpha,
            max_participation=self.max_participation,
            returns_panel=returns_panel,
            ticker_stats=ticker_stats,
            timestamp=timestamp,
            metadata={
                "period": self.history_period,
                "n_obs": int(len(returns_panel)),
                "n_tickers": len(tickers),
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        """Re-run the per-ticker liquidity-stats calibration from the inputs.

        ``fetch_data`` already triggered calibration to materialize the
        stats — this method exists to satisfy the `BaseModel` contract
        and to support call-paths that fetch once but want a fresh
        `CalibrationResult` for orchestration logging.
        """

        if not isinstance(data, LiquidityVaRInputs):
            raise TypeError(
                f"LiquidityAdjustedVaR.calibrate expects LiquidityVaRInputs, "
                f"got {type(data).__name__}"
            )
        self._stats = data.ticker_stats
        gross = sum(abs(p.dollar_value) for p in data.positions)
        net = sum(p.dollar_value for p in data.positions)
        fit_metrics: dict[str, float] = {
            "n_positions": float(len(data.positions)),
            "gross_exposure": float(gross),
            "net_exposure": float(net),
            "mean_spread_relative": float(
                np.mean([s.spread_relative for s in data.ticker_stats])
            ),
            "mean_position_horizon_days": float(
                np.mean([s.position_horizon_days for s in data.ticker_stats])
            ),
            "max_position_horizon_days": float(
                max(s.position_horizon_days for s in data.ticker_stats)
            ),
            "max_participation": float(data.max_participation),
        }
        return CalibrationResult(
            model_name=MODEL_NAME,
            parameters={
                "ticker_stats": data.ticker_stats,
                "max_participation": data.max_participation,
                "alpha": data.alpha,
            },
            fit_metrics=fit_metrics,
            timestamp=data.timestamp,
            metadata={
                "tickers": [p.ticker for p in data.positions],
                "n_obs": int(len(data.returns_panel)),
            },
        )

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, LiquidityVaRInputs):
            raise TypeError(
                f"LiquidityAdjustedVaR.predict expects LiquidityVaRInputs, "
                f"got {type(data).__name__}"
            )
        result = compute_l_var(data)
        self._stats = data.ticker_stats
        return RiskMetric(
            ticker=_PORTFOLIO_TICKER,
            metric_name="liquidity_adjusted_var",
            value=float(result.l_var),
            timestamp=data.timestamp,
            confidence_level=float(data.alpha),
            horizon="1d",
            metadata={
                "alpha": float(data.alpha),
                "max_participation": float(data.max_participation),
                "price_var_1d": float(result.price_var_1d),
                "es_frtb": float(result.es_frtb),
                "liquidity_cost_spread": float(result.liquidity_cost_spread),
                "liquidity_cost_ac": float(result.liquidity_cost_ac),
                "l_var": float(result.l_var),
                "gross_exposure": float(
                    sum(abs(p.dollar_value) for p in data.positions)
                ),
                "net_exposure": float(sum(p.dollar_value for p in data.positions)),
                "tickers": [p.ticker for p in data.positions],
                "per_name": result.per_name,
                "n_obs": int(len(data.returns_panel)),
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        """Spec validation block (concentration, horizon coverage, stress).

        The spec's blocks 1-2 (implied vs realized slippage and stressed
        spread backtest) require historical TCA / VIX-bucketed data that
        is not in yfinance; they are out of scope for the operational
        implementation. The remaining diagnostics (3-5) are computed here.
        """

        if not isinstance(data, LiquidityVaRInputs):
            raise TypeError(
                f"LiquidityAdjustedVaR.validate expects LiquidityVaRInputs, "
                f"got {type(data).__name__}"
            )
        result = compute_l_var(data)
        conc = concentration_ratio(data.positions, data.ticker_stats)
        violations = horizon_coverage_violations(data.positions, data.ticker_stats)
        stressed = stress_spread_l_var(data, shock_multiplier=self.stress_multiplier)
        gross = sum(abs(p.dollar_value) for p in data.positions)
        out: dict[str, Any] = {
            "n_positions": len(data.positions),
            "n_obs": int(len(data.returns_panel)),
            "alpha": float(data.alpha),
            "max_participation": float(data.max_participation),
            "gross_exposure": float(gross),
            "net_exposure": float(sum(p.dollar_value for p in data.positions)),
            "price_var_1d": float(result.price_var_1d),
            "es_frtb": float(result.es_frtb),
            "liquidity_cost_spread": float(result.liquidity_cost_spread),
            "liquidity_cost_ac": float(result.liquidity_cost_ac),
            "l_var": float(result.l_var),
            "l_var_to_price_var_ratio": (
                float(result.l_var / result.price_var_1d)
                if result.price_var_1d > 0.0
                else float("nan")
            ),
            "stressed_l_var": float(stressed),
            "stress_multiplier": float(self.stress_multiplier),
            "stress_uplift": float(stressed - result.l_var),
            "concentration_ratio": int(conc),
            "horizon_coverage_violations": list(violations),
            "n_horizon_violations": len(violations),
            "n_tier_4": sum(1 for s in data.ticker_stats if s.tier == 4),
            "n_hl_proxy_spreads": sum(
                1 for s in data.ticker_stats if s.spread_source == "hl_proxy"
            ),
            "mean_spread_relative": float(
                np.mean([s.spread_relative for s in data.ticker_stats])
            ),
            "max_position_horizon_days": float(
                max(s.position_horizon_days for s in data.ticker_stats)
            ),
        }
        return out

    # ---- public conveniences -------------------------------------------------

    def compute(self, data: LiquidityVaRInputs) -> LVaRResult:
        """Return the structured `LVaRResult` (without the orchestration wrap)."""

        return compute_l_var(data)


def _index_from_bars(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Pick a DatetimeIndex out of an OHLCV frame.

    yfinance returns either a `"Date"` column (daily) or `"Datetime"`
    (intraday); when neither is present, we fall back to the existing
    index. The result is sorted ascending.
    """

    if "Date" in bars.columns:
        raw: Any = pd.to_datetime(bars["Date"])
    elif "Datetime" in bars.columns:
        raw = pd.to_datetime(bars["Datetime"])
    else:
        raw = pd.to_datetime(bars.index)
    return pd.DatetimeIndex(raw).sort_values()


__all__ = ["LiquidityAdjustedVaR"]
