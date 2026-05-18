"""`CostOfCarry` — BaseModel subclass for futures-cash basis trading.

The class is a thin orchestration shell around the pure functions in
`signal.py` and `calibration.py`. `fetch_data` pulls spot/futures/rate-curve
prints from the provider and aggregates the constituent dividend series into
an index-point dividend stream; `calibrate` refreshes the cost band and the
projected dividend stream; `predict` runs the theoretical-price / basis /
implied-repo formulas and emits a directional `Signal`; `validate` reports
the diagnostics enumerated in the spec.

The model is stateful between calibrate and predict: a freshly constructed
instance starts with default cost parameters and an empty projected dividend
set, so `calibrate()` should be called before `predict()` whenever historical
basis data and constituent dividends are available.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Signal
from src.models.cost_of_carry_dividends.calibration import (
    calibrate as _calibrate_fn,
)
from src.models.cost_of_carry_dividends.signal import (
    action_to_signal_direction,
    aggregate_constituent_dividends,
    compute_decomposition,
    compute_result,
    signal_strength,
)
from src.models.cost_of_carry_dividends.types import (
    CostOfCarryInputs,
    CostParameters,
    DividendStream,
    IndexConstituent,
    RateCurve,
)

_DEFAULT_SHORT_RATE_TICKER: str = "^IRX"
_DEFAULT_MID_RATE_TICKER: str = "^TNX"
_DEFAULT_HORIZON: str = "intraday"
_DEFAULT_HISTORY_DAYS: int = 5
_INTRADAY_INTERVAL: str = "1m"
_INTRADAY_LOOKBACK_DAYS: int = 1
_DEFAULT_STOCK_LOAN_BPS: float = 10.0  # ~10 bps general-collateral spread


@dataclass
class FetchedData:
    """Bundle returned by `fetch_data` and consumed by calibrate / predict.

    Holds the latest spot/futures prints, the term-structure of rates assembled
    from short and mid-curve proxies, and the dividends series fetched per
    constituent (raw, in per-share cash). The aggregated index-point stream is
    rebuilt on each calibration and stored on the model.
    """

    futures_ticker: str
    spot_ticker: str
    timestamp: datetime
    expiry: datetime
    spot: float
    futures: float
    rate_curve: RateCurve
    stock_loan_spread: float
    dividends_per_share: dict[str, pd.Series]
    constituents: tuple[IndexConstituent, ...]
    cost_params: CostParameters
    switch_threshold_years: float


@register_model
class CostOfCarry(BaseModel):
    """Cost-of-carry / discrete-dividend model for index futures.

    See `models/layer2_structural/05_cost_of_carry_dividends.md` for the math.

    Parameters
    ----------
    futures_ticker:
        Symbol of the index future being modelled (e.g. ``"ES=F"``).
    spot_ticker:
        Symbol of the underlying spot index (e.g. ``"^GSPC"``).
    expiry:
        Contract expiry datetime. yfinance often does not expose this
        reliably so it must be supplied externally (CME calendar).
    constituents:
        Sequence of `IndexConstituent` tuples used to aggregate per-share
        dividends into index-point dividends. May be empty when the caller
        passes a pre-built dividend stream via `dividend_stream_override`.
    cost_params:
        Initial cost band; refreshed by `calibrate()` from realized basis
        history (95% quantile by default).
    short_rate_ticker / mid_rate_ticker:
        Yield-curve pillars used to interpolate the term-matched financing
        rate. Defaults follow the spec: 13-week T-bill and 10-year Treasury.
    stock_loan_spread_bps:
        Add-on to the SOFR-equivalent rate to account for general-collateral
        stock-loan financing (~10 bps for SPX).
    switch_threshold_years:
        Below this tau, the discrete-dividend formula is used. Above, the
        equivalent continuous-yield approximation. Default 0.25 (3 months).
    dividend_stream_override:
        Skip the constituent aggregation entirely and use this stream
        instead. Used by tests and by callers that maintain their own
        dividend calendar.
    """

    name: ClassVar[str] = "cost_of_carry_dividends"
    layer: ClassVar[int] = 2
    refit_frequency: ClassVar[RefitFrequency] = "daily"

    def __init__(
        self,
        *,
        futures_ticker: str = "ES=F",
        spot_ticker: str = "^GSPC",
        expiry: datetime,
        constituents: Sequence[IndexConstituent] = (),
        cost_params: CostParameters | None = None,
        short_rate_ticker: str = _DEFAULT_SHORT_RATE_TICKER,
        mid_rate_ticker: str = _DEFAULT_MID_RATE_TICKER,
        stock_loan_spread_bps: float = _DEFAULT_STOCK_LOAN_BPS,
        switch_threshold_years: float = 0.25,
        history_days: int = _DEFAULT_HISTORY_DAYS,
        dividend_stream_override: DividendStream | None = None,
        now_func: Any = None,
    ) -> None:
        if not futures_ticker:
            raise ValueError("futures_ticker must be non-empty")
        if not spot_ticker:
            raise ValueError("spot_ticker must be non-empty")
        self.futures_ticker = futures_ticker
        self.spot_ticker = spot_ticker
        self.expiry = _ensure_aware(expiry)
        self.constituents = tuple(constituents)
        self.cost_params = cost_params or CostParameters(
            futures_band_points=0.5, etf_cost_band_bps=2.0
        )
        self.short_rate_ticker = short_rate_ticker
        self.mid_rate_ticker = mid_rate_ticker
        self.stock_loan_spread = stock_loan_spread_bps * 1e-4
        self.switch_threshold_years = switch_threshold_years
        self.history_days = history_days
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._dividend_stream: DividendStream = (
            dividend_stream_override if dividend_stream_override is not None else DividendStream(())
        )
        self._basis_history: list[float] = []

    # ----- BaseModel interface ----- #

    def fetch_data(self, provider: DataProvider) -> FetchedData:
        now = self._normalize_now(self._now_func())
        spot = self._fetch_last_price(provider, self.spot_ticker)
        futures = self._fetch_last_price(provider, self.futures_ticker)
        rate_curve = self._fetch_rate_curve(provider)
        divs_per_share: dict[str, pd.Series] = {}
        for constituent in self.constituents:
            divs_per_share[constituent.ticker] = provider.fetch_dividends(constituent.ticker)
        return FetchedData(
            futures_ticker=self.futures_ticker,
            spot_ticker=self.spot_ticker,
            timestamp=now,
            expiry=self.expiry,
            spot=spot,
            futures=futures,
            rate_curve=rate_curve,
            stock_loan_spread=self.stock_loan_spread,
            dividends_per_share=divs_per_share,
            constituents=self.constituents,
            cost_params=self.cost_params,
            switch_threshold_years=self.switch_threshold_years,
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        """Refresh the cost band from `self._basis_history` and project dividends."""

        timestamp = data.timestamp if isinstance(data, FetchedData) else self._normalize_now(self._now_func())
        constituents = data.constituents if isinstance(data, FetchedData) else self.constituents
        divs_per_share = (
            data.dividends_per_share if isinstance(data, FetchedData) else {}
        )
        result = _calibrate_fn(
            basis_history=tuple(self._basis_history),
            constituents=constituents,
            dividends_per_share=divs_per_share,
            today=timestamp,
            expiry=self.expiry,
            fallback_band_points=self.cost_params.futures_band_points,
            fallback_etf_band_bps=self.cost_params.etf_cost_band_bps,
            timestamp=timestamp,
        )
        new_cost = result.parameters["cost_params"]
        new_stream = result.parameters["dividend_stream"]
        assert isinstance(new_cost, CostParameters)
        assert isinstance(new_stream, DividendStream)
        self.cost_params = new_cost
        # When no constituents were supplied but a stream was injected via
        # override, leave the override untouched.
        if constituents or new_stream.dividends:
            self._dividend_stream = new_stream
        return result

    def predict(self, data: Any) -> Signal:
        if not isinstance(data, FetchedData):
            raise TypeError(
                f"CostOfCarry.predict expects FetchedData, got {type(data).__name__}"
            )
        # If the constituent dividend series were fetched but calibrate has not
        # been run yet, build the announced stream on the fly so predict still
        # produces a usable signal.
        stream = self._dividend_stream
        if (
            not stream.dividends
            and data.constituents
            and data.dividends_per_share
        ):
            stream = aggregate_constituent_dividends(
                constituents=data.constituents,
                dividends_per_share=data.dividends_per_share,
                today=data.timestamp,
                expiry=data.expiry,
            )

        inputs = CostOfCarryInputs(
            futures_ticker=data.futures_ticker,
            spot_ticker=data.spot_ticker,
            timestamp=data.timestamp,
            expiry=data.expiry,
            spot=data.spot,
            futures=data.futures,
            rate_curve=data.rate_curve,
            stock_loan_spread=data.stock_loan_spread,
            dividends=stream,
            cost_params=self.cost_params,
            switch_threshold_years=data.switch_threshold_years,
        )
        result = compute_result(inputs)
        # Track the realized basis for the next calibration window.
        self._basis_history.append(result.basis)
        if len(self._basis_history) > 250:
            self._basis_history = self._basis_history[-250:]

        direction = action_to_signal_direction(result.action)
        strength = signal_strength(result.basis, self.cost_params)
        return Signal(
            ticker=result.futures_ticker,
            direction=direction,
            strength=strength,
            timestamp=result.timestamp,
            horizon=_DEFAULT_HORIZON,
            metadata={
                "action": result.action,
                "spot": result.spot,
                "futures": result.futures,
                "theoretical": result.theoretical,
                "basis": result.basis,
                "implied_repo": result.implied_repo,
                "tau_years": result.decomposition.tau_years,
                "rate": result.decomposition.rate,
                "equivalent_yield": result.decomposition.equivalent_yield,
                "cost_band_points": result.cost_band_points,
                "n_dividends": float(len(stream)),
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        """Diagnostic snapshot per the spec's "Validation and diagnostics" section."""

        if not isinstance(data, FetchedData):
            raise TypeError(
                f"CostOfCarry.validate expects FetchedData, got {type(data).__name__}"
            )

        stream = self._dividend_stream
        if not stream.dividends and data.constituents and data.dividends_per_share:
            stream = aggregate_constituent_dividends(
                constituents=data.constituents,
                dividends_per_share=data.dividends_per_share,
                today=data.timestamp,
                expiry=data.expiry,
            )

        inputs = CostOfCarryInputs(
            futures_ticker=data.futures_ticker,
            spot_ticker=data.spot_ticker,
            timestamp=data.timestamp,
            expiry=data.expiry,
            spot=data.spot,
            futures=data.futures,
            rate_curve=data.rate_curve,
            stock_loan_spread=data.stock_loan_spread,
            dividends=stream,
            cost_params=self.cost_params,
            switch_threshold_years=data.switch_threshold_years,
        )
        decomposition = compute_decomposition(inputs)

        # Spec validation criteria:
        # 1. Basis stationary near zero — flag if |basis| > 5 * cost band.
        # 2. Implied repo within SOFR +- 50 bps (we approximate SOFR by `rate`).
        # 3. Continuous vs discrete agreement for long tau (use both).
        rate = decomposition.rate
        repo_ok = abs(decomposition.implied_repo - rate) < 0.005  # 50 bps

        # Cross-check continuous vs discrete: rebuild both and compare in bps.
        cross_check_bps = _cross_check_methods(decomposition, inputs)

        # Coverage: count constituents that returned at least one dividend.
        covered = sum(
            1
            for c in data.constituents
            if len(data.dividends_per_share.get(c.ticker, pd.Series(dtype="float64"))) > 0
        )
        coverage = covered / len(data.constituents) if data.constituents else 1.0

        return {
            "status": "ok",
            "tau_years": decomposition.tau_years,
            "rate": rate,
            "implied_repo": decomposition.implied_repo,
            "repo_ok": repo_ok,
            "repo_spread_bps": (decomposition.implied_repo - rate) * 1e4,
            "basis": decomposition.basis,
            "basis_within_band": abs(decomposition.basis)
            <= 5.0 * self.cost_params.futures_band_points,
            "method": decomposition.method,
            "method_cross_check_bps": cross_check_bps,
            "n_dividends": len(stream),
            "coverage": coverage,
            "coverage_ok": coverage >= 0.95 if data.constituents else True,
            "cost_band_points": self.cost_params.futures_band_points,
            "n_basis_obs": len(self._basis_history),
        }

    # ---- helpers -------------------------------------------------------- #

    def _normalize_now(self, now: datetime) -> datetime:
        """Ensure timestamps are timezone-aware to match `self.expiry`."""

        if now.tzinfo is None:
            return now.replace(tzinfo=self.expiry.tzinfo or UTC)
        return now

    def _fetch_last_price(self, provider: DataProvider, ticker: str) -> float:
        bars = provider.fetch_intraday(
            ticker, _INTRADAY_INTERVAL, _INTRADAY_LOOKBACK_DAYS
        )
        price = _extract_last_close(bars)
        if price is None:
            # Fallback to daily close.
            daily = provider.fetch_prices(ticker, f"{self.history_days}d", "1d")
            price = _extract_last_close(daily)
        if price is None:
            raise RuntimeError(f"CostOfCarry: no price available for {ticker!r}")
        return float(price)

    def _fetch_rate_curve(self, provider: DataProvider) -> RateCurve:
        short_pct = self._fetch_last_index_yield(provider, self.short_rate_ticker)
        mid_pct = self._fetch_last_index_yield(provider, self.mid_rate_ticker)
        rates: dict[float, float] = {}
        if short_pct is not None:
            rates[0.25] = short_pct / 100.0  # ^IRX is a 13-week yield in pct
        if mid_pct is not None:
            rates[10.0] = mid_pct / 100.0
        if not rates:
            # Last-resort defaults so the model still runs offline-ish.
            rates[0.25] = 0.045
            rates[10.0] = 0.045
        return RateCurve(rates=rates)

    @staticmethod
    def _fetch_last_index_yield(
        provider: DataProvider, ticker: str
    ) -> float | None:
        df = provider.fetch_prices(ticker, "5d", "1d")
        return _extract_last_close(df)


# ----- module-level helpers --------------------------------------------- #


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _extract_last_close(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None
    if "Close" not in df.columns:
        return None
    return float(df["Close"].iloc[-1])


def _cross_check_methods(
    decomposition: Any, inputs: CostOfCarryInputs
) -> float:
    """Recompute the other method and report the divergence in basis points of spot."""

    from src.models.cost_of_carry_dividends.signal import (
        theoretical_futures_continuous,
        theoretical_futures_discrete,
    )

    if decomposition.tau_years <= 0 or inputs.spot <= 0:
        return 0.0
    rate = decomposition.rate
    fv = decomposition.fv_dividends
    f_disc = theoretical_futures_discrete(inputs.spot, rate, decomposition.tau_years, fv)
    f_cont = theoretical_futures_continuous(
        inputs.spot, rate, decomposition.equivalent_yield, decomposition.tau_years
    )
    diff = f_cont - f_disc
    return float(diff / inputs.spot * 1e4)


def build_default_expiry(today: datetime, months_out: int = 3) -> datetime:
    """Approximate the next quarterly expiry (3rd Friday) `months_out` months ahead.

    yfinance does not always expose `expireDate` reliably, so this helper lets
    callers spin up the model with a sensible default when they do not maintain
    their own CME contract calendar. Production callers should pass an
    authoritative expiry instead.
    """

    target_month = today.month + months_out
    target_year = today.year + (target_month - 1) // 12
    target_month = ((target_month - 1) % 12) + 1
    first = datetime(target_year, target_month, 1, 16, 0, tzinfo=today.tzinfo or UTC)
    # weekday(): Monday=0, ..., Friday=4. Days to first Friday:
    offset_to_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=offset_to_friday)
    return first_friday + timedelta(days=14)
