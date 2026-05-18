"""`MertonKMV` — Layer 2 structural credit model.

The class is a thin orchestration shell around the pure functions in
`signal.py` and `calibration.py`. It pulls everything it needs through the
shared `DataProvider`, persists the most recent calibration on the instance,
and projects the result into the shared `Signal` type for the orchestration
layer.

Spec: `models/layer2_structural/03_merton_kmv.md`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, cast

import pandas as pd

from src.core.base_model import BaseModel
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Signal
from src.models.merton_kmv.calibration import calibrate as _calibrate
from src.models.merton_kmv.signal import (
    compute_merton_result,
    signal_direction_from_quality,
    signal_strength_from_dd,
)
from src.models.merton_kmv.types import (
    BalanceSheetSnapshot,
    KMVSolution,
    MertonInputs,
    MertonResult,
)

_DEFAULT_HORIZON_YEARS: float = 1.0
_DEFAULT_HISTORY_DAYS: int = 252
_DEFAULT_WEIGHT_LT_DEBT: float = 0.5
_DEFAULT_LGD: float = 0.6
_SHORT_RATE_TICKER: str = "^IRX"
_SHORT_RATE_PERIOD: str = "5d"
_SHORT_RATE_INTERVAL: str = "1d"
_DEFAULT_PRICE_PERIOD: str = "1y"
_DEFAULT_PRICE_INTERVAL: str = "1d"
_DEFAULT_REFIT_FREQUENCY: Literal["daily"] = "daily"


@register_model
class MertonKMV(BaseModel):
    """Merton-KMV structural credit model.

    Parameters
    ----------
    ticker:
        Equity ticker (e.g. ``"GS"``).
    horizon_years:
        Default-horizon `T - t`. Spec convention is 1.0; longer horizons can
        be chosen by passing a different value (and using `^TNX` upstream).
    history_days:
        Calibration window for `sigma_E` and the iterative KMV loop.
    weight_lt_debt:
        Fraction of long-term debt rolled into the default point `D`. The
        empirical KMV value is 0.5.
    lgd:
        Loss-given-default used for the theoretical credit spread.
    market_cds_spread_bps:
        Optional externally supplied CDS quote for the cap-structure arb
        stance. The `DataProvider` interface does not (yet) carry CDS data;
        pass it in directly here when running paper-trade reconciliation.
    arb_threshold_bps:
        Two-sided threshold on `(market - model)` spread before tagging the
        name "CDS rich" or "CDS cheap".
    drift_mode:
        ``"risk_free"`` (use `r` as the physical drift — spec Option C, the
        default in practice when `mu` is uncertain), ``"trailing"`` (use the
        annualized mean log-return on assets — spec Option B), or ``"capm"``
        (currently unsupported without a market-beta input).
    """

    name: ClassVar[str] = "merton_kmv"
    layer: ClassVar[int] = 2
    refit_frequency: ClassVar[Literal["daily"]] = _DEFAULT_REFIT_FREQUENCY

    def __init__(
        self,
        ticker: str,
        *,
        horizon_years: float = _DEFAULT_HORIZON_YEARS,
        history_days: int = _DEFAULT_HISTORY_DAYS,
        weight_lt_debt: float = _DEFAULT_WEIGHT_LT_DEBT,
        lgd: float = _DEFAULT_LGD,
        market_cds_spread_bps: float | None = None,
        arb_threshold_bps: float = 25.0,
        drift_mode: Literal["risk_free", "trailing"] = "risk_free",
        now_func: Any = None,
    ) -> None:
        if not ticker:
            raise ValueError("MertonKMV requires a non-empty ticker")
        if horizon_years <= 0:
            raise ValueError(
                f"horizon_years must be positive, got {horizon_years}"
            )
        if history_days < 30:
            raise ValueError(
                f"history_days must be >= 30, got {history_days}"
            )
        if not 0.0 <= weight_lt_debt <= 1.0:
            raise ValueError(
                f"weight_lt_debt must lie in [0, 1], got {weight_lt_debt}"
            )
        if not 0.0 <= lgd <= 1.0:
            raise ValueError(f"lgd must lie in [0, 1], got {lgd}")

        self.ticker = ticker.upper()
        self.horizon_years = horizon_years
        self.history_days = history_days
        self.weight_lt_debt = weight_lt_debt
        self.lgd = lgd
        self.market_cds_spread_bps = market_cds_spread_bps
        self.arb_threshold_bps = arb_threshold_bps
        self.drift_mode = drift_mode
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._last_solution: KMVSolution | None = None
        self._last_drift: float | None = None

    # ---- BaseModel interface -------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> MertonInputs:
        now = self._now_func()
        info = provider.fetch_fundamentals(self.ticker)
        if not info:
            raise RuntimeError(
                f"No fundamentals returned for {self.ticker!r}; "
                "cannot run Merton-KMV."
            )

        equity_market_value = _resolve_market_cap(info, self.ticker)
        shares_out = _resolve_shares_out(info, equity_market_value, self.ticker)
        bs = _resolve_balance_sheet(info, self.ticker)

        prices = self._fetch_price_history(provider)
        risk_free = self._fetch_risk_free_rate(provider)
        industry = str(info.get("industry") or info.get("sector") or "unknown")

        return MertonInputs(
            ticker=self.ticker,
            timestamp=now,
            equity_market_value=equity_market_value,
            equity_prices=prices,
            shares_outstanding=shares_out,
            balance_sheet=bs,
            risk_free_rate=risk_free,
            horizon_years=self.horizon_years,
            weight_lt_debt=self.weight_lt_debt,
            industry=industry,
            market_cds_spread_bps=self.market_cds_spread_bps,
            lgd=self.lgd,
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, MertonInputs):
            raise TypeError(
                f"MertonKMV.calibrate expects MertonInputs, got {type(data).__name__}"
            )
        result = _calibrate(
            data,
            history_days=self.history_days,
            timestamp=data.timestamp,
        )
        self._last_solution = cast(KMVSolution, result.parameters["solution"])
        self._last_drift = cast(float, result.parameters["drift_trailing"])
        return result

    def predict(self, data: Any) -> Signal:
        if not isinstance(data, MertonInputs):
            raise TypeError(
                f"MertonKMV.predict expects MertonInputs, got {type(data).__name__}"
            )
        result = self._compute_result(data)
        return self._result_to_signal(result)

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, MertonInputs):
            raise TypeError(
                f"MertonKMV.validate expects MertonInputs, got {type(data).__name__}"
            )
        result = self._compute_result(data)

        # Spec validation 2. g1 is exactly enforced by the BS inversion; g2 is
        # only approximately enforced by the iterative KMV procedure (sigma_V
        # is empirical, not chosen to satisfy the Ito identity), so g2 carries
        # a looser threshold. Joint-solve callers would tighten both.
        residual_ok = result.g1_residual < 1e-4 and result.g2_residual < 0.15
        # Spec validation 6: sigma_V should be lower (more stable) than sigma_E.
        vol_ratio = (
            result.asset_volatility / result.equity_volatility
            if result.equity_volatility > 0
            else float("nan")
        )
        leverage = float(result.metadata.get("leverage", 0.0))

        return {
            "ticker": self.ticker,
            "asset_value": result.asset_value,
            "asset_volatility": result.asset_volatility,
            "equity_volatility": result.equity_volatility,
            "vol_ratio_asset_to_equity": vol_ratio,
            "default_point": result.default_point,
            "leverage": leverage,
            "dd_physical": result.dd_physical,
            "dd_risk_neutral": result.dd_risk_neutral,
            "pd_physical": result.pd_physical,
            "pd_risk_neutral": result.pd_risk_neutral,
            "credit_spread_bps": result.credit_spread_bps,
            "credit_quality": result.credit_quality,
            "arb_stance": result.arb_stance,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
            "g1_residual": result.g1_residual,
            "g2_residual": result.g2_residual,
            "residual_ok": residual_ok,
        }

    # ---- public diagnostic accessor -----------------------------------------

    def snapshot(self, data: MertonInputs) -> MertonResult:
        """Return the full structural snapshot (preserves DD / PD / spread)."""

        return self._compute_result(data)

    # ---- internals -----------------------------------------------------------

    def _compute_result(self, data: MertonInputs) -> MertonResult:
        if self._last_solution is None or self._last_drift is None:
            # Lazy calibration so callers that go straight to predict() still work.
            self.calibrate(data)
        solution = cast(KMVSolution, self._last_solution)
        drift_trailing = cast(float, self._last_drift)
        drift = (
            data.risk_free_rate if self.drift_mode == "risk_free" else drift_trailing
        )
        return compute_merton_result(
            data,
            solution,
            drift,
            arb_threshold_bps=self.arb_threshold_bps,
        )

    def _result_to_signal(self, result: MertonResult) -> Signal:
        direction = signal_direction_from_quality(
            result.credit_quality, result.arb_stance
        )
        strength = signal_strength_from_dd(result.dd_physical)
        return Signal(
            ticker=result.ticker,
            direction=cast(Any, direction),
            strength=strength,
            timestamp=result.timestamp,
            horizon=f"{self.horizon_years:g}y",
            metadata={
                "asset_value": result.asset_value,
                "asset_volatility": result.asset_volatility,
                "equity_volatility": result.equity_volatility,
                "default_point": result.default_point,
                "dd_physical": result.dd_physical,
                "dd_risk_neutral": result.dd_risk_neutral,
                "pd_physical": result.pd_physical,
                "pd_risk_neutral": result.pd_risk_neutral,
                "credit_spread_bps": result.credit_spread_bps,
                "credit_quality": result.credit_quality,
                "arb_stance": result.arb_stance,
                "converged": result.converged,
                "leverage": result.metadata.get("leverage", 0.0),
            },
        )

    def _fetch_price_history(self, provider: DataProvider) -> pd.Series:
        bars = provider.fetch_prices(
            self.ticker, _DEFAULT_PRICE_PERIOD, _DEFAULT_PRICE_INTERVAL
        )
        if bars is None or bars.empty:
            raise RuntimeError(
                f"No daily price history returned for {self.ticker!r}; "
                "cannot calibrate Merton-KMV."
            )
        if "Close" not in bars.columns:
            raise RuntimeError(
                f"Price feed for {self.ticker!r} missing 'Close' column"
            )
        closes = bars["Close"].astype(float).dropna()
        if len(closes) < 30:
            raise RuntimeError(
                f"Need >= 30 daily closes for {self.ticker!r}, got {len(closes)}"
            )
        return closes

    def _fetch_risk_free_rate(self, provider: DataProvider) -> float:
        bars = provider.fetch_prices(
            _SHORT_RATE_TICKER, _SHORT_RATE_PERIOD, _SHORT_RATE_INTERVAL
        )
        if bars is None or bars.empty or "Close" not in bars.columns:
            return 0.0
        last_pct = float(bars["Close"].iloc[-1])
        # `^IRX` quotes the 13-week T-bill yield as a percent.
        return last_pct / 100.0


# ----- balance-sheet resolution from the `info` dict ------------------------
#
# `DataProvider.fetch_fundamentals` returns yfinance's `info` blob. We extract
# the debt fields here rather than in the model body so the fallback logic
# stays out of the main flow. See the open SHARED-CHANGE-REQUEST in the README
# for a first-class balance-sheet method.


_MARKET_CAP_KEYS: tuple[str, ...] = ("marketCap", "market_cap")
_SHARES_KEYS: tuple[str, ...] = ("sharesOutstanding", "shares_outstanding", "impliedSharesOutstanding")
_TOTAL_DEBT_KEYS: tuple[str, ...] = ("totalDebt", "total_debt")
_LT_DEBT_KEYS: tuple[str, ...] = (
    "longTermDebt",
    "long_term_debt",
    "longTermDebtTotal",
)
_ST_DEBT_KEYS: tuple[str, ...] = (
    "currentDebt",
    "shortTermDebt",
    "short_term_debt",
    "shortLongTermDebt",
)
_CASH_KEYS: tuple[str, ...] = (
    "cash",
    "totalCash",
    "cashAndCashEquivalents",
    "cash_and_cash_equivalents",
)


def _first_present(info: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = info.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not _is_nan(float(v)):
            return float(v)
    return None


def _is_nan(x: float) -> bool:
    return x != x  # NaN != NaN


def _resolve_market_cap(info: dict[str, Any], ticker: str) -> float:
    value = _first_present(info, _MARKET_CAP_KEYS)
    if value is None or value <= 0:
        raise RuntimeError(
            f"No usable market cap for {ticker!r} in fundamentals; "
            "Merton-KMV cannot run."
        )
    return value


def _resolve_shares_out(
    info: dict[str, Any], market_cap: float, ticker: str
) -> float:
    value = _first_present(info, _SHARES_KEYS)
    if value is not None and value > 0:
        return value
    # Fall back: market_cap / latest price would need the price series; instead
    # we leave it to the caller — `MertonInputs.__post_init__` will reject a
    # zero or negative shares count.
    raise RuntimeError(
        f"No usable shares-outstanding for {ticker!r}; Merton-KMV cannot run."
    )


def _resolve_balance_sheet(
    info: dict[str, Any], ticker: str
) -> BalanceSheetSnapshot:
    st_raw = _first_present(info, _ST_DEBT_KEYS)
    lt_raw = _first_present(info, _LT_DEBT_KEYS)
    total_debt = _first_present(info, _TOTAL_DEBT_KEYS)
    cash = _first_present(info, _CASH_KEYS) or 0.0

    # Prefer explicit ST/LT; fall back to a 30/70 ST/LT split of total debt
    # when only the aggregate is reported. The split is heuristic; the user
    # can override via `MertonKMV(..., weight_lt_debt=...)` for a fully-known
    # capital structure.
    st_debt: float
    lt_debt: float
    if st_raw is not None and lt_raw is not None:
        st_debt, lt_debt = st_raw, lt_raw
    else:
        if total_debt is None or total_debt <= 0:
            raise RuntimeError(
                f"No usable debt fields for {ticker!r} (need totalDebt or "
                "currentDebt + longTermDebt); Merton-KMV cannot run. "
                "See SHARED-CHANGE-REQUEST in README for a structured "
                "balance-sheet feed."
            )
        if st_raw is None and lt_raw is None:
            st_debt = 0.3 * total_debt
            lt_debt = 0.7 * total_debt
        elif st_raw is None:
            assert lt_raw is not None
            lt_debt = lt_raw
            st_debt = max(total_debt - lt_raw, 0.0)
        else:
            st_debt = st_raw
            lt_debt = max(total_debt - st_raw, 0.0)

    return BalanceSheetSnapshot(
        short_term_debt=st_debt,
        long_term_debt=lt_debt,
        cash_and_equivalents=float(cash),
        total_debt=total_debt,
    )
