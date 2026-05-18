"""`StressTests` — Layer 6 risk / capital model.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. The class fetches long-history daily prices for every
position plus a canonical set of macro / sector proxies, runs historical
replay against the spec's canonical crisis windows, evaluates the standard
hypothetical scenarios, and (optionally) solves the reverse-stress problem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.stress_tests.calibration import MODEL_NAME
from src.models.stress_tests.calibration import calibrate as _calibrate_impl
from src.models.stress_tests.signal import (
    apply_proxy_fill,
    historical_replay_pnl,
    hypothetical_pnl,
    is_plausible,
    portfolio_factor_sensitivities,
    reverse_stress_shock,
    scenario_coverage_matrix,
    slice_window,
    top_contributors,
    window_endpoint_sensitivity,
    window_factor_change,
)
from src.models.stress_tests.types import (
    DEFAULT_FACTORS,
    CrisisWindow,
    HypotheticalScenario,
    ReverseStressResult,
    ScenarioPnL,
    ScenarioType,
    StressFit,
    StressInputs,
    StressReport,
)

_PORTFOLIO_TICKER: str = "PORTFOLIO"
_DEFAULT_HORIZON: str = "scenario"
_BENCHMARK_TICKER: str = "^GSPC"
_PRICE_PERIOD: str = "max"
_PRICE_INTERVAL: str = "1d"

# Default crisis windows from the spec's Routine A.
DEFAULT_CRISIS_WINDOWS: tuple[CrisisWindow, ...] = (
    CrisisWindow(name="Black Monday 1987", t1="1987-10-15", t2="1987-10-20"),
    CrisisWindow(name="LTCM 1998", t1="1998-08-17", t2="1998-10-08"),
    CrisisWindow(name="Lehman 2008", t1="2008-09-12", t2="2008-10-10"),
    CrisisWindow(name="Eurozone 2011", t1="2011-08-01", t2="2011-08-19"),
    CrisisWindow(name="China devaluation", t1="2015-08-17", t2="2015-08-26"),
    CrisisWindow(name="Volmageddon 2018", t1="2018-02-02", t2="2018-02-09"),
    CrisisWindow(name="Q4 2018 selloff", t1="2018-10-03", t2="2018-12-24"),
    CrisisWindow(name="COVID crash", t1="2020-02-19", t2="2020-03-23"),
    CrisisWindow(name="GameStop unwind", t1="2021-01-25", t2="2021-02-05"),
    CrisisWindow(name="2022 bond rout", t1="2022-01-03", t2="2022-10-14"),
)

# Default hypothetical scenarios from the spec's Routine B table.
DEFAULT_HYPOTHETICALS: tuple[HypotheticalScenario, ...] = (
    HypotheticalScenario(name="Equity -20%", shocks={"equity": -0.20}),
    HypotheticalScenario(name="Equity -30%", shocks={"equity": -0.30}),
    HypotheticalScenario(name="Rates +100bp", shocks={"rates": 0.01}),
    HypotheticalScenario(name="USD +10%", shocks={"dxy": 0.10}),
    HypotheticalScenario(
        name="Combined 1-in-100",
        shocks={"equity": -0.25, "vol": 0.10, "ig": 0.015, "hy": 0.04, "dxy": 0.08},
    ),
)

# Default sector-ETF proxy map. Falls back to ^GSPC for unmapped tickers.
DEFAULT_SECTOR_MAP: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOG": "XLC", "GOOGL": "XLC",
    "META": "XLC", "AMZN": "XLY", "TSLA": "XLY", "JPM": "XLF", "BAC": "XLF",
    "WFC": "XLF", "GS": "XLF", "MS": "XLF", "XOM": "XLE", "CVX": "XLE",
    "COP": "XLE", "JNJ": "XLV", "PFE": "XLV", "UNH": "XLV", "PG": "XLP",
    "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "BA": "XLI", "CAT": "XLI",
    "GE": "XLI", "NEE": "XLU", "DUK": "XLU", "LIN": "XLB", "FCX": "XLB",
    "AMT": "XLRE", "PLD": "XLRE",
}


@register_model
class StressTests(BaseModel):
    """Portfolio stress-test engine (historical replay + hypothetical + reverse).

    Parameters
    ----------
    positions:
        Mapping of ticker -> signed dollar position. Long positions are
        positive, short positions negative.
    capital:
        Notional capital used to set the reverse-stress loss target.
    crisis_windows:
        Overrides the spec's default historical-replay set.
    hypotheticals:
        Overrides the spec's default hypothetical scenario set.
    sector_map:
        Per-ticker sector-ETF proxy for crises where the ticker pre-dates its
        IPO. Defaults to a small mapping of common large-caps.
    loss_target_fraction:
        Reverse-stress loss target as a fraction of capital. The spec's
        default is 30%.
    calibration_period:
        yfinance ``period`` string for the beta / factor-cov calibration
        window. The spec calls for ~5y of daily history.
    benchmark_ticker:
        Long-history fallback when a position-ticker has no data in a crisis
        window and no sector proxy is available. Defaults to `^GSPC` (1928-).
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 6
    refit_frequency: ClassVar[RefitFrequency] = "daily"

    def __init__(
        self,
        positions: Mapping[str, float],
        capital: float,
        *,
        crisis_windows: Sequence[CrisisWindow] | None = None,
        hypotheticals: Sequence[HypotheticalScenario] | None = None,
        sector_map: Mapping[str, str] | None = None,
        loss_target_fraction: float = 0.30,
        calibration_period: str = "5y",
        benchmark_ticker: str = _BENCHMARK_TICKER,
    ) -> None:
        if not positions:
            raise ValueError("StressTests requires at least one position")
        if capital <= 0:
            raise ValueError(f"capital must be > 0, got {capital}")
        if not 0.0 < loss_target_fraction < 1.0:
            raise ValueError(
                "loss_target_fraction must lie in (0, 1), got "
                f"{loss_target_fraction}"
            )
        self.positions: dict[str, float] = {str(k): float(v) for k, v in positions.items()}
        self.capital: float = float(capital)
        self.crisis_windows: tuple[CrisisWindow, ...] = (
            tuple(crisis_windows) if crisis_windows is not None else DEFAULT_CRISIS_WINDOWS
        )
        self.hypotheticals: tuple[HypotheticalScenario, ...] = (
            tuple(hypotheticals) if hypotheticals is not None else DEFAULT_HYPOTHETICALS
        )
        self.sector_map: dict[str, str] = (
            dict(sector_map) if sector_map is not None else dict(DEFAULT_SECTOR_MAP)
        )
        self.loss_target_fraction: float = float(loss_target_fraction)
        self.calibration_period: str = calibration_period
        self.benchmark_ticker: str = benchmark_ticker
        self._fit: StressFit | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> StressInputs:
        now = datetime.now(UTC)
        tickers = list(self.positions.keys())

        # Distinct universe of price series we need to pull. We fetch full
        # history once per ticker and slice locally — the DataProvider has no
        # start/end signature today.
        proxy_tickers = sorted({self.sector_map[t] for t in tickers if t in self.sector_map})
        factor_tickers = sorted(set(DEFAULT_FACTORS.values()))
        long_history_universe = sorted(
            set(tickers) | set(proxy_tickers) | {self.benchmark_ticker} | set(factor_tickers)
        )

        long_history: dict[str, pd.Series] = {}
        for ticker in long_history_universe:
            bars = provider.fetch_prices(ticker, _PRICE_PERIOD, _PRICE_INTERVAL)
            series = _bars_to_close_series(bars)
            if series is not None and not series.empty:
                long_history[ticker] = series

        price_panel = (
            pd.concat(long_history, axis=1, sort=True) if long_history else pd.DataFrame()
        )

        # Per-window per-ticker slices, including proxy and benchmark fills.
        crisis_prices: dict[tuple[str, str], pd.Series] = {}
        proxy_used: dict[tuple[str, str], str] = {}
        for window in self.crisis_windows:
            relevant = set(tickers) | set(proxy_tickers) | {self.benchmark_ticker}
            for ticker in relevant:
                series = long_history.get(ticker)
                if series is None:
                    continue
                sliced = slice_window(series, window.t1, window.t2)
                if not sliced.empty:
                    crisis_prices[(window.name, ticker)] = sliced

        # Factor-returns panel for calibration / reverse-stress. Use the recent
        # `calibration_period` slice; if a factor is missing the column is
        # dropped so the calibration still proceeds with the available subset.
        factor_returns = self._build_factor_returns_panel(long_history=long_history)

        return StressInputs(
            positions=dict(self.positions),
            capital=self.capital,
            crisis_windows=self.crisis_windows,
            hypotheticals=self.hypotheticals,
            loss_target_fraction=self.loss_target_fraction,
            timestamp=now,
            crisis_prices=crisis_prices,
            factor_returns=factor_returns,
            sector_map=self.sector_map,
            proxy_used=proxy_used,
            metadata={
                "long_history_tickers": list(long_history.keys()),
                "price_panel": price_panel,
                "calibration_period": self.calibration_period,
                "benchmark_ticker": self.benchmark_ticker,
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, StressInputs):
            raise TypeError(
                f"StressTests.calibrate expects StressInputs, got {type(data).__name__}"
            )
        asset_returns, factor_returns = self._returns_for_calibration(data)
        result = _calibrate_impl(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            capital=data.capital,
            timestamp=data.timestamp,
        )
        self._fit = cast(StressFit, result.parameters["fit"])
        return result

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, StressInputs):
            raise TypeError(
                f"StressTests.predict expects StressInputs, got {type(data).__name__}"
            )
        report = self.run_full_report(data)
        return RiskMetric(
            ticker=_PORTFOLIO_TICKER,
            metric_name="stress_worst_case_pnl",
            value=report.worst_case,
            timestamp=report.timestamp,
            confidence_level=None,
            horizon=_DEFAULT_HORIZON,
            metadata={
                "worst_case_name": report.worst_case_name,
                "worst_case_type": report.worst_case_type,
                "capital": report.capital,
                "loss_pct_of_capital": (
                    -report.worst_case / report.capital if report.capital else 0.0
                ),
                "historical_pnls": {
                    name: scenario.total_pnl for name, scenario in report.historical.items()
                },
                "hypothetical_pnls": {
                    name: scenario.total_pnl for name, scenario in report.hypothetical.items()
                },
                "reverse_mahalanobis": (
                    report.reverse.mahalanobis_distance if report.reverse else None
                ),
                "reverse_plausible": (
                    is_plausible(report.reverse) if report.reverse else None
                ),
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, StressInputs):
            raise TypeError(
                f"StressTests.validate expects StressInputs, got {type(data).__name__}"
            )
        report = self.run_full_report(data)
        coverage = scenario_coverage_matrix(
            scenarios=list(self.hypotheticals),
            factor_names=list(DEFAULT_FACTORS.keys()),
        )
        empty_factors = [c for c in coverage.columns if not coverage[c].any()]
        empty_scenarios = [r for r in coverage.index if not coverage.loc[r].any()]

        # Per-window endpoint sensitivity over the most recent crisis. Provides
        # the regime-stability check from the spec validation list.
        latest_window = self.crisis_windows[-1]
        endpoint_prices = {
            ticker: data.crisis_prices.get((latest_window.name, ticker))
            for ticker in self.positions
        }
        endpoint_pnls = window_endpoint_sensitivity(
            window=latest_window,
            prices_by_ticker={
                t: s for t, s in endpoint_prices.items() if s is not None
            },
            positions=self.positions,
            shift_days=2,
        )
        endpoint_values = list(endpoint_pnls.values())
        endpoint_range = (
            float(max(endpoint_values) - min(endpoint_values)) if endpoint_values else 0.0
        )

        worst_hist = (
            min(report.historical.values(), key=lambda s: s.total_pnl)
            if report.historical
            else None
        )
        worst_hyp = (
            min(report.hypothetical.values(), key=lambda s: s.total_pnl)
            if report.hypothetical
            else None
        )

        out: dict[str, Any] = {
            "worst_case": report.worst_case,
            "worst_case_name": report.worst_case_name,
            "worst_case_type": report.worst_case_type,
            "n_historical": len(report.historical),
            "n_hypothetical": len(report.hypothetical),
            "coverage_empty_factors": empty_factors,
            "coverage_empty_scenarios": empty_scenarios,
            "endpoint_window": latest_window.name,
            "endpoint_pnls": endpoint_pnls,
            "endpoint_range": endpoint_range,
        }
        if worst_hist is not None:
            out["worst_historical_top3"] = top_contributors(worst_hist, k=3)
        if worst_hyp is not None:
            out["worst_hypothetical_top3"] = top_contributors(worst_hyp, k=3)
        if report.reverse is not None:
            out["reverse_mahalanobis"] = report.reverse.mahalanobis_distance
            out["reverse_plausible"] = is_plausible(report.reverse)
            out["reverse_loss_target"] = report.reverse.loss_target
            out["reverse_std_devs"] = dict(
                zip(report.reverse.factor_names, report.reverse.std_devs.tolist(), strict=True)
            )
        return out

    # ----- Public conveniences for downstream models -------------------------

    @property
    def fit(self) -> StressFit:
        return self._require_fit()

    def run_full_report(self, data: StressInputs) -> StressReport:
        """Run all three flavors and assemble the worst-case envelope."""

        historical = self._run_historical(data)
        hypothetical = self._run_hypothetical(data)
        reverse = self._run_reverse(data)

        worst_case: float = 0.0
        worst_name: str = "none"
        worst_type: ScenarioType = "historical"
        all_scenarios: list[tuple[ScenarioType, ScenarioPnL]] = (
            [("historical", s) for s in historical.values()]
            + [("hypothetical", s) for s in hypothetical.values()]
        )
        if all_scenarios:
            kind, worst_scenario = min(all_scenarios, key=lambda pair: pair[1].total_pnl)
            worst_case = worst_scenario.total_pnl
            worst_name = worst_scenario.scenario_name
            worst_type = kind

        return StressReport(
            timestamp=data.timestamp,
            capital=data.capital,
            historical=historical,
            hypothetical=hypothetical,
            reverse=reverse,
            worst_case=worst_case,
            worst_case_name=worst_name,
            worst_case_type=worst_type,
        )

    # ----- Internals ---------------------------------------------------------

    def _run_historical(self, data: StressInputs) -> dict[str, ScenarioPnL]:
        out: dict[str, ScenarioPnL] = {}
        tickers = list(self.positions.keys())
        betas_to_benchmark = self._beta_to_benchmark(data)
        for window in data.crisis_windows:
            direct_changes: dict[str, float] = {}
            for ticker in tickers:
                series = data.crisis_prices.get((window.name, ticker))
                if series is None:
                    continue
                direct_changes[ticker] = window_factor_change(series)
            proxy_changes: dict[str, float] = {}
            for proxy_ticker in set(data.sector_map.values()):
                series = data.crisis_prices.get((window.name, proxy_ticker))
                if series is None:
                    continue
                proxy_changes[proxy_ticker] = window_factor_change(series)
            benchmark_series = data.crisis_prices.get(
                (window.name, self.benchmark_ticker)
            )
            fallback_change = (
                window_factor_change(benchmark_series)
                if benchmark_series is not None
                else float("nan")
            )
            filled, source = apply_proxy_fill(
                tickers=tickers,
                direct_changes=direct_changes,
                proxy_changes=proxy_changes,
                fallback_change=fallback_change,
                sector_map=data.sector_map,
                betas_to_fallback=betas_to_benchmark,
            )
            out[window.name] = historical_replay_pnl(
                window=window,
                positions=self.positions,
                factor_changes=filled,
                source=source,
            )
        return out

    def _run_hypothetical(self, data: StressInputs) -> dict[str, ScenarioPnL]:
        if self._fit is None:
            return {}
        fit = self._fit
        betas_df = pd.DataFrame(
            fit.betas, index=list(fit.tickers), columns=list(fit.factor_names)
        )
        out: dict[str, ScenarioPnL] = {}
        for scenario in data.hypotheticals:
            out[scenario.name] = hypothetical_pnl(
                scenario=scenario,
                positions=self.positions,
                betas=betas_df,
            )
        return out

    def _run_reverse(self, data: StressInputs) -> ReverseStressResult | None:
        if self._fit is None:
            return None
        fit = self._fit
        betas_df = pd.DataFrame(
            fit.betas, index=list(fit.tickers), columns=list(fit.factor_names)
        )
        g = portfolio_factor_sensitivities(positions=self.positions, betas=betas_df)
        if float(g @ fit.factor_covariance @ g) <= 0:
            return None
        loss_target = data.capital * data.loss_target_fraction
        return reverse_stress_shock(
            g=g,
            factor_covariance=fit.factor_covariance,
            loss_target=loss_target,
            factor_names=fit.factor_names,
        )

    def _beta_to_benchmark(self, data: StressInputs) -> dict[str, float]:
        """Per-ticker beta to the equity factor (proxy for `^GSPC`).

        Used by `apply_proxy_fill` to scale the `^GSPC` move when neither
        direct nor sector-proxy data is available in a crisis window. Falls
        back to a uniform 1.0 when the fit is unavailable or the equity
        factor column is missing.
        """

        del data  # only used to satisfy the call signature for future per-factor pulls
        out: dict[str, float] = dict.fromkeys(self.positions.keys(), 1.0)
        if self._fit is None or "equity" not in self._fit.factor_names:
            return out
        fit = self._fit
        col_idx = fit.factor_names.index("equity")
        for i, ticker in enumerate(fit.tickers):
            out[ticker] = float(fit.betas[i, col_idx])
        return out

    def _build_factor_returns_panel(
        self,
        *,
        long_history: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Build the factor-return panel from the long-history series.

        Equity, IG, HY, FX, sector ETFs are reported as price series — pct
        change gives a return. `^TNX` and `^VIX` are level series (yield in %
        and vol points); for the stress engine we treat their *changes* as the
        factor shocks (`d_yield`, `d_vol`) — i.e., a +100 bp rate shock means
        `factor_returns["rates"]` should reflect 1.0 (since `^TNX` is in pct).
        """

        cols: dict[str, pd.Series] = {}
        n_lookback_days = _period_to_days(self.calibration_period)
        for factor_name, ticker in DEFAULT_FACTORS.items():
            series = long_history.get(ticker)
            if series is None or series.empty:
                continue
            tail = series.tail(n_lookback_days) if n_lookback_days > 0 else series
            if factor_name in ("rates", "vol"):
                # Level series: factor "return" is a first difference (in the
                # series' native units: pct points for ^TNX, vol points for ^VIX).
                # We divide ^TNX by 100 to get rate in decimal form; ^VIX we
                # leave as vol points so a "vol +5" shock maps to 5.0.
                if ticker == "^TNX":
                    tail = tail / 100.0
                cols[factor_name] = tail.diff()
            else:
                cols[factor_name] = tail.pct_change()
        if not cols:
            return pd.DataFrame()
        panel = pd.concat(cols, axis=1, sort=True)
        return panel.dropna(how="all")

    def _returns_for_calibration(
        self, data: StressInputs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Slice asset returns from the long-history panel, aligning with factors."""

        if data.factor_returns.empty:
            raise RuntimeError(
                "Factor-returns panel is empty — calibration cannot proceed. "
                "Check provider data for the macro tickers."
            )
        price_panel = data.metadata.get("price_panel")
        if not isinstance(price_panel, pd.DataFrame) or price_panel.empty:
            raise RuntimeError(
                "Calibration requires the full price panel in StressInputs.metadata."
                " Make sure fetch_data populated it."
            )
        asset_tickers = [t for t in self.positions if t in price_panel.columns]
        if not asset_tickers:
            raise RuntimeError(
                "None of the position tickers have price history available — "
                "calibration cannot proceed."
            )
        n_lookback_days = _period_to_days(self.calibration_period)
        prices = price_panel[asset_tickers]
        if n_lookback_days > 0:
            prices = prices.tail(n_lookback_days)
        asset_returns = prices.pct_change().dropna(how="all")
        common = asset_returns.index.intersection(data.factor_returns.index)
        if common.empty:
            raise RuntimeError(
                "Asset and factor return panels share no overlapping dates."
            )
        return asset_returns.loc[common], data.factor_returns.loc[common].dropna()

    def _require_fit(self) -> StressFit:
        if self._fit is None:
            raise RuntimeError(
                "StressTests has not been calibrated. Call calibrate(data) first."
            )
        return self._fit


# ---- module-level helpers ---------------------------------------------------


def _bars_to_close_series(bars: pd.DataFrame) -> pd.Series | None:
    """Convert a yfinance OHLCV frame to a close-price Series indexed by date.

    yfinance returns tz-aware timestamps for some tickers and tz-naive for
    others; daily bars sometimes carry an intraday hh:mm component depending
    on the underlying exchange. We normalize to a tz-naive midnight timestamp
    here so the rest of the pipeline (window slicing, panel alignment) can
    treat every series as if it were keyed by calendar date.
    """

    if bars is None or bars.empty or "Close" not in bars.columns:
        return None
    if "Date" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Date"], utc=True))
    elif "Datetime" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Datetime"], utc=True))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    index = index.tz_convert(None).normalize()
    close = pd.Series(bars["Close"].astype(float).to_numpy(), index=index).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    return close.dropna()


def _period_to_days(period: str) -> int:
    """Approximate yfinance ``period`` -> trading-day count.

    Returns 0 if the period is ``"max"`` (use the full series).
    """

    mapping = {
        "1d": 1, "5d": 5, "1mo": 21, "3mo": 63, "6mo": 126,
        "1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "ytd": 252,
    }
    if period == "max":
        return 0
    return mapping.get(period, 1260)


__all__ = [
    "DEFAULT_CRISIS_WINDOWS",
    "DEFAULT_HYPOTHETICALS",
    "DEFAULT_SECTOR_MAP",
    "StressTests",
]
