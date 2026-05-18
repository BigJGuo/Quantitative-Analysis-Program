"""`INAVEquityETF` — Layer 2 structural model.

The class is a thin orchestration shell around the pure functions in
`signal.py` and `calibration.py`. The interesting math lives there; this file
glues fetched data, calibration state, and the shared `BaseModel` contract
together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Signal
from src.models.inav_equity_etf.calibration import calibrate as _calibrate_params
from src.models.inav_equity_etf.signal import (
    action_to_signal_direction,
    compute_inav_result,
    is_quote_stale,
    signal_strength,
)
from src.models.inav_equity_etf.types import (
    BasketHolding,
    CashLedgerSeed,
    ConstituentQuote,
    CostParameters,
    INAVInputs,
    INAVResult,
)

_INTRADAY_INTERVAL: str = "1m"
_INTRADAY_LOOKBACK_DAYS: int = 1
_SHORT_RATE_TICKER: str = "^IRX"
_SHORT_RATE_PERIOD: str = "5d"
_SHORT_RATE_INTERVAL: str = "1d"
_DEFAULT_HORIZON: str = "intraday"


@register_model
class INAVEquityETF(BaseModel):
    """Indicative-NAV / fair-value model for an equity ETF.

    Parameters
    ----------
    etf_ticker:
        Symbol of the ETF being modelled (e.g. ``"SPY"``).
    holdings:
        Creation-unit basket. Required because `DataProvider.fetch_holdings`
        currently exposes only `(symbol, weight)` and the model needs full
        share-count and currency information.
    seed:
        Prior-day end-of-day cash/liabilities/NAV/shares snapshot.
    staleness_threshold_seconds:
        Constituents whose last print is older than this are routed to the
        STALE leg of the decomposition. 300 s matches the spec's US default.
    kappa_bps, tau_bps:
        Half-spread parameters defining the no-arb band. Refreshed by
        `calibrate()` when realized premia are supplied.
    """

    name: ClassVar[str] = "inav_equity_etf"
    layer: ClassVar[int] = 2
    refit_frequency: ClassVar[RefitFrequency] = "intraday"

    def __init__(
        self,
        etf_ticker: str,
        holdings: Sequence[BasketHolding],
        seed: CashLedgerSeed,
        *,
        staleness_threshold_seconds: float = 300.0,
        kappa_bps: float = 1.0,
        tau_bps: float = 2.0,
        now_func: Any = None,
    ) -> None:
        if not holdings:
            raise ValueError("INAVEquityETF requires a non-empty holdings basket")
        self.etf_ticker = etf_ticker.upper()
        self.holdings: tuple[BasketHolding, ...] = tuple(holdings)
        self.seed = seed
        self.staleness_threshold_seconds = staleness_threshold_seconds
        self.cost_params = CostParameters(kappa_bps=kappa_bps, tau_bps=tau_bps)
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self.recent_premia: Sequence[float] | None = None

    def fetch_data(self, provider: DataProvider) -> INAVInputs:
        now = self._now_func()
        quotes = self._fetch_constituent_quotes(provider)
        fx_rates = self._fetch_fx_rates(provider, now)
        dividends = self._fetch_dividends_since_eod(provider)
        short_rate = self._fetch_short_rate(provider)
        expense_ratio = self._fetch_expense_ratio(provider)
        etf_mid = self._fetch_etf_mid(provider)

        return INAVInputs(
            etf_ticker=self.etf_ticker,
            timestamp=now,
            basket=self.holdings,
            quotes=quotes,
            fx_rates=fx_rates,
            dividends_since_eod=dividends,
            short_rate_annual=short_rate,
            expense_ratio=expense_ratio,
            seed=self.seed,
            etf_mid=etf_mid,
            staleness_threshold_seconds=self.staleness_threshold_seconds,
            cost_params=self.cost_params,
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        """Refresh basket / seeds / cost band.

        Accepts either a fully-populated `INAVInputs` (the standard case, after
        `fetch_data`) or `None` to recalibrate against the model's current
        configuration. Optional `realized_premia` history may be passed by
        setting `self.recent_premia` before calling.
        """

        inputs = data if isinstance(data, INAVInputs) else None
        basket = inputs.basket if inputs is not None else self.holdings
        seed = inputs.seed if inputs is not None else self.seed
        realized: Sequence[float] | None = self.recent_premia

        result = _calibrate_params(
            basket=basket,
            seed=seed,
            realized_premia=realized,
            default_kappa_bps=self.cost_params.kappa_bps,
            default_tau_bps=self.cost_params.tau_bps,
            timestamp=inputs.timestamp if inputs is not None else self._now_func(),
        )
        new_cost = cast(CostParameters, result.parameters["cost_params"])
        self.cost_params = new_cost
        self.holdings = cast(tuple[BasketHolding, ...], result.parameters["basket"])
        self.seed = cast(CashLedgerSeed, result.parameters["seed"])
        return result

    def predict(self, data: Any) -> Signal:
        if not isinstance(data, INAVInputs):
            raise TypeError(
                f"INAVEquityETF.predict expects INAVInputs, got {type(data).__name__}"
            )
        # Ensure the latest cost band (possibly refreshed since fetch) is used.
        data = replace(data, cost_params=self.cost_params)
        result = compute_inav_result(data)
        return self._result_to_signal(result)

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, INAVInputs):
            raise TypeError(
                f"INAVEquityETF.validate expects INAVInputs, got {type(data).__name__}"
            )
        data = replace(data, cost_params=self.cost_params)
        result = compute_inav_result(data)

        # Per-share basket coverage check: every constituent listed in the
        # holdings file should produce a quote. A drop below 99% suggests a
        # stale holdings file or a corp action.
        basket_size = len(self.holdings)
        quotes_present = sum(1 for h in self.holdings if h.ticker in data.quotes)
        coverage = quotes_present / basket_size if basket_size else 0.0

        # Premium magnitude in bps for easy log scanning.
        premium_bps = result.premium * 1e4

        # EOD reconciliation residual: when called within a tight window of the
        # next strike, |iNAV - eod_nav| should be < 5 bps for US-only ETFs.
        eod_residual_bps = (result.inav - self.seed.eod_nav) / self.seed.eod_nav * 1e4

        return {
            "inav": result.inav,
            "market_mid": result.market_mid,
            "premium_bps": premium_bps,
            "action": result.action,
            "n_live": result.n_live,
            "n_stale": result.n_stale,
            "stale_fraction": result.stale_fraction,
            "coverage": coverage,
            "coverage_ok": coverage >= 0.99,
            "eod_residual_bps": eod_residual_bps,
            "band_bps": self.cost_params.kappa_bps + self.cost_params.tau_bps,
        }

    def snapshot(self, data: INAVInputs) -> INAVResult:
        """Convenience accessor returning the full structural decomposition.

        `predict()` collapses everything into a shared `Signal`; this method
        preserves the per-leg breakdown for diagnostics and downstream models
        (e.g. Model 2 will consume `INAVResult.decomposition.basket_stale`).
        """

        data = replace(data, cost_params=self.cost_params)
        return compute_inav_result(data)

    def _result_to_signal(self, result: INAVResult) -> Signal:
        direction = action_to_signal_direction(result.action)
        strength = signal_strength(result.premium, self.cost_params)
        signal_dir = cast(Any, direction)
        return Signal(
            ticker=result.etf_ticker,
            direction=signal_dir,
            strength=strength,
            timestamp=result.timestamp,
            horizon=_DEFAULT_HORIZON,
            metadata={
                "action": result.action,
                "inav": result.inav,
                "market_mid": result.market_mid,
                "premium": result.premium,
                "n_live": result.n_live,
                "n_stale": result.n_stale,
                "stale_fraction": result.stale_fraction,
                "basket_live_usd": result.decomposition.basket_live,
                "basket_stale_usd": result.decomposition.basket_stale,
                "cash_usd": result.decomposition.cash,
                "liabilities_usd": result.decomposition.liabilities,
            },
        )

    # ---- data-fetch helpers (provider-only; no yfinance import) -------------

    def _fetch_constituent_quotes(
        self, provider: DataProvider
    ) -> dict[str, ConstituentQuote]:
        out: dict[str, ConstituentQuote] = {}
        for holding in self.holdings:
            bars = provider.fetch_intraday(
                holding.ticker, _INTRADAY_INTERVAL, _INTRADAY_LOOKBACK_DAYS
            )
            quote = self._last_bar_to_quote(holding.ticker, bars)
            if quote is not None:
                out[holding.ticker] = quote
        return out

    def _fetch_fx_rates(
        self, provider: DataProvider, now: datetime
    ) -> dict[str, float]:
        currencies = {h.currency.upper() for h in self.holdings if h.currency.upper() != "USD"}
        rates: dict[str, float] = {}
        for currency in currencies:
            symbol = f"{currency}USD=X"
            bars = provider.fetch_intraday(symbol, _INTRADAY_INTERVAL, _INTRADAY_LOOKBACK_DAYS)
            quote = self._last_bar_to_quote(symbol, bars)
            if quote is None:
                continue
            # FX feeds can themselves be stale outside their session — flag it
            # in the quote, but still use the last available rate.
            if is_quote_stale(quote, now, self.staleness_threshold_seconds):
                quote = ConstituentQuote(
                    ticker=symbol, price=quote.price, timestamp=quote.timestamp, is_stale=True
                )
            rates[currency] = quote.price
        return rates

    def _fetch_dividends_since_eod(
        self, provider: DataProvider
    ) -> dict[str, float]:
        cutoff = self.seed.eod_release_time
        out: dict[str, float] = {}
        for holding in self.holdings:
            series = provider.fetch_dividends(holding.ticker)
            if series is None or len(series) == 0:
                continue
            mask = series.index > pd.Timestamp(cutoff)
            recent = series[mask]
            if len(recent) > 0:
                out[holding.ticker] = float(recent.sum())
        return out

    def _fetch_short_rate(self, provider: DataProvider) -> float:
        bars = provider.fetch_prices(
            _SHORT_RATE_TICKER, _SHORT_RATE_PERIOD, _SHORT_RATE_INTERVAL
        )
        if bars is None or bars.empty or "Close" not in bars.columns:
            return 0.0
        last_pct = float(bars["Close"].iloc[-1])
        # `^IRX` quotes the 13-week T-bill yield as a percent (e.g. 5.25 = 5.25%).
        return last_pct / 100.0

    def _fetch_expense_ratio(self, provider: DataProvider) -> float:
        info = provider.fetch_fundamentals(self.etf_ticker)
        value = info.get("annualReportExpenseRatio") if info else None
        if value is None:
            return 0.0
        return float(value)

    def _fetch_etf_mid(self, provider: DataProvider) -> float:
        bars = provider.fetch_intraday(
            self.etf_ticker, _INTRADAY_INTERVAL, _INTRADAY_LOOKBACK_DAYS
        )
        if bars is None or bars.empty:
            raise RuntimeError(
                f"No intraday bars returned for ETF {self.etf_ticker!r}; "
                "cannot compute market mid."
            )
        last = bars.iloc[-1]
        if "High" in bars.columns and "Low" in bars.columns:
            return float((last["High"] + last["Low"]) / 2.0)
        return float(last["Close"])

    @staticmethod
    def _last_bar_to_quote(
        ticker: str, bars: pd.DataFrame
    ) -> ConstituentQuote | None:
        if bars is None or bars.empty:
            return None
        last = bars.iloc[-1]
        if "Close" not in bars.columns:
            return None
        price = float(last["Close"])
        ts = _extract_timestamp(bars, last)
        return ConstituentQuote(ticker=ticker, price=price, timestamp=ts)


def _extract_timestamp(bars: pd.DataFrame, last_row: pd.Series[Any]) -> datetime:
    # YFinanceProvider calls `.reset_index()`, so the timestamp lives in a
    # column ("Datetime" or "Date") rather than the index.
    for col in ("Datetime", "Date", "index"):
        if col in bars.columns:
            ts_value = last_row[col]
            return pd.Timestamp(ts_value).to_pydatetime()
    idx = bars.index[-1]
    return pd.Timestamp(idx).to_pydatetime()
