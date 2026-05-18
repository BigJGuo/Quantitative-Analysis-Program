"""``KyleSqrtImpact`` — Layer-5 execution / impact model.

Thin orchestration shell that wires the pure functions in ``signal.py`` and
``calibration.py`` into the ``BaseModel`` contract. Steps:

1. ``fetch_data`` pulls daily (and optionally 5-minute) OHLCV bars from the
   shared ``DataProvider`` and packs them into a ``KyleInputs`` bundle.
2. ``calibrate`` runs the OLS + HC1 lambda regression at the requested
   frequency and stores the resulting ``KyleLambdaFit`` on the instance.
3. ``predict`` returns a ``RiskMetric`` carrying the spec's pre-trade
   impact estimate (in bps) for a default parent order, using Algorithm D
   to switch between linear-Kyle and the sqrt law.
4. ``validate`` reports the spec's diagnostic block (R^2, lambda sign,
   cross-sectional sanity).

Spec: ``models/layer5_execution/15_kyle_lambda_sqrt_impact.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, cast

import pandas as pd

from src.core.base_model import BaseModel
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.kyle_lambda_sqrt_impact.calibration import (
    MODEL_NAME,
)
from src.models.kyle_lambda_sqrt_impact.calibration import (
    calibrate as _calibrate_impl,
)
from src.models.kyle_lambda_sqrt_impact.signal import (
    adv,
    crossover_size,
    linear_impact,
    participation_ratio_exceeds_threshold,
    realized_daily_vol,
    sqrt_law_impact,
)
from src.models.kyle_lambda_sqrt_impact.types import (
    VALID_FREQUENCIES,
    CombinedImpactPrediction,
    KyleFrequency,
    KyleInputs,
    KyleLambdaFit,
    SqrtImpactPrediction,
)

_DEFAULT_DAILY_PERIOD: str = "120d"
_DEFAULT_DAILY_INTERVAL: str = "1d"
_DEFAULT_INTRADAY_INTERVAL: str = "5m"
_DEFAULT_INTRADAY_LOOKBACK: int = 60
_DEFAULT_ADV_WINDOW: int = 20
_DEFAULT_Y: float = 1.0
_DEFAULT_PARENT_SHARES: float = 10_000.0
_DEFAULT_SIDE: int = 1
_BPS_PER_UNIT: float = 1.0e4


@register_model
class KyleSqrtImpact(BaseModel):
    """Kyle-lambda + square-root impact pre-trade cost estimator.

    Parameters
    ----------
    ticker:
        Equity / ETF symbol.
    frequency:
        ``"daily"`` (default) fits ``lambda`` on daily bars; ``"intraday"``
        fits on 5-minute bars over ``intraday_lookback_days`` days.
    daily_period:
        yfinance ``period`` string for the daily history pull. Default
        ``"120d"`` matches the spec calibration window.
    daily_lookback:
        Number of trailing daily bars used in the regression. Spec default
        is 60 or 120; we keep 120 for stability.
    intraday_interval / intraday_lookback_days:
        yfinance arguments for the 5-minute bar pull. yfinance caps 5m
        history at ~60 days.
    Y_prefactor:
        Square-root law prefactor. Spec literature prior for US large-cap
        equities is ``Y = 1.0``.
    parent_order_shares / side:
        Default parent order used by ``predict``. ``side`` is ``+1`` for
        buy, ``-1`` for sell.
    dollar_signed:
        If ``True``, run the lambda regression on dollar-signed volume
        instead of share-signed volume.
    adv_window:
        Rolling window for the ``V`` input to the sqrt law. Spec default
        ``20`` trading days.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 5
    refit_frequency: ClassVar[Literal["weekly"]] = "weekly"

    def __init__(
        self,
        ticker: str,
        *,
        frequency: KyleFrequency = "daily",
        daily_period: str = _DEFAULT_DAILY_PERIOD,
        daily_lookback: int = 120,
        intraday_interval: str = _DEFAULT_INTRADAY_INTERVAL,
        intraday_lookback_days: int = _DEFAULT_INTRADAY_LOOKBACK,
        Y_prefactor: float = _DEFAULT_Y,
        parent_order_shares: float = _DEFAULT_PARENT_SHARES,
        side: int = _DEFAULT_SIDE,
        dollar_signed: bool = False,
        adv_window: int = _DEFAULT_ADV_WINDOW,
        now_func: Any = None,
    ) -> None:
        if not ticker:
            raise ValueError("KyleSqrtImpact requires a non-empty ticker")
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(VALID_FREQUENCIES)}, "
                f"got {frequency!r}"
            )
        if daily_lookback < 30:
            raise ValueError(
                f"daily_lookback must be >= 30 for a stable fit, got {daily_lookback}"
            )
        if intraday_lookback_days < 1:
            raise ValueError(
                f"intraday_lookback_days must be >= 1, got {intraday_lookback_days}"
            )
        if Y_prefactor <= 0:
            raise ValueError(f"Y_prefactor must be > 0, got {Y_prefactor}")
        if parent_order_shares <= 0:
            raise ValueError(
                f"parent_order_shares must be > 0, got {parent_order_shares}"
            )
        if side not in (-1, 1):
            raise ValueError(f"side must be +/-1, got {side}")
        if adv_window < 1:
            raise ValueError(f"adv_window must be >= 1, got {adv_window}")

        self.ticker = ticker.upper()
        self.frequency: KyleFrequency = frequency
        self.daily_period = daily_period
        self.daily_lookback = int(daily_lookback)
        self.intraday_interval = intraday_interval
        self.intraday_lookback_days = int(intraday_lookback_days)
        self.Y_prefactor = float(Y_prefactor)
        self.parent_order_shares = float(parent_order_shares)
        self.side = int(side)
        self.dollar_signed = bool(dollar_signed)
        self.adv_window = int(adv_window)
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._fit: KyleLambdaFit | None = None

    # ---- BaseModel interface -------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> KyleInputs:
        daily = provider.fetch_prices(
            self.ticker, self.daily_period, _DEFAULT_DAILY_INTERVAL
        )
        if daily is None or daily.empty:
            raise RuntimeError(
                f"No daily OHLCV history returned for {self.ticker!r}; "
                "cannot fit Kyle lambda."
            )
        daily = self._normalize_bars(daily)
        intraday: pd.DataFrame | None = None
        if self.frequency == "intraday":
            intra = provider.fetch_intraday(
                self.ticker,
                self.intraday_interval,
                self.intraday_lookback_days,
            )
            if intra is None or intra.empty:
                raise RuntimeError(
                    f"No {self.intraday_interval} intraday bars returned for "
                    f"{self.ticker!r}; cannot fit intraday Kyle lambda."
                )
            intraday = self._normalize_bars(intra)
        return KyleInputs(
            ticker=self.ticker,
            daily_bars=daily,
            intraday_bars=intraday,
            frequency=self.frequency,
            parent_order_shares=self.parent_order_shares,
            side=self.side,
            Y_prefactor=self.Y_prefactor,
            timestamp=self._now_func(),
            metadata={
                "daily_period": self.daily_period,
                "intraday_interval": self.intraday_interval,
                "intraday_lookback_days": self.intraday_lookback_days,
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, KyleInputs):
            raise TypeError(
                f"KyleSqrtImpact.calibrate expects KyleInputs, "
                f"got {type(data).__name__}"
            )
        result = _calibrate_impl(
            daily_bars=data.daily_bars,
            frequency=data.frequency,
            lookback=self.daily_lookback,
            intraday_bars=data.intraday_bars,
            dollar=self.dollar_signed,
            timestamp=data.timestamp,
        )
        self._fit = cast(KyleLambdaFit, result.parameters["fit"])
        return result

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, KyleInputs):
            raise TypeError(
                f"KyleSqrtImpact.predict expects KyleInputs, "
                f"got {type(data).__name__}"
            )
        fit = self._ensure_fit(data)
        combined = self.forecast_combined(data)
        return RiskMetric(
            ticker=self.ticker,
            metric_name="expected_impact_bps",
            value=combined.impact_bps,
            timestamp=data.timestamp,
            confidence_level=None,
            horizon="parent_order",
            metadata={
                "regime": combined.regime,
                "lambda_hat": fit.lambda_hat,
                "lambda_se": fit.se,
                "lambda_ci_low": fit.ci_low,
                "lambda_ci_high": fit.ci_high,
                "Y_prefactor": combined.Y_prefactor,
                "sigma_daily": combined.sigma_daily,
                "volume_daily": combined.volume_daily,
                "Q": combined.Q,
                "Q_over_V": combined.Q_over_V,
                "Q_crossover": combined.Q_crossover,
                "impact_relative": combined.impact_relative,
                "impact_dollars": combined.impact_dollars,
                "extrapolation_warning": combined.extrapolation_warning,
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, KyleInputs):
            raise TypeError(
                f"KyleSqrtImpact.validate expects KyleInputs, "
                f"got {type(data).__name__}"
            )
        fit = self._ensure_fit(data)
        closes = data.daily_bars["Close"].astype(float).dropna()
        volumes = data.daily_bars["Volume"].astype(float).dropna()
        sigma = realized_daily_vol(closes)
        v_daily = adv(volumes, window=self.adv_window)

        # Spec validation step 3: Kyle's theory predicts lambda increasing in
        # sigma and decreasing in V. We surface the ratio sigma/V as the
        # cross-sectional sanity input; downstream cross-asset reports plot
        # log(lambda) against log(sigma/V).
        sigma_over_v = sigma / v_daily if v_daily > 0 else float("nan")

        return {
            "ticker": self.ticker,
            "frequency": fit.frequency,
            "n_obs": fit.n_obs,
            "lambda_hat": fit.lambda_hat,
            "lambda_se": fit.se,
            "lambda_ci_low": fit.ci_low,
            "lambda_ci_high": fit.ci_high,
            "lambda_normalized": fit.lambda_normalized,
            "lambda_positive": fit.lambda_hat > 0.0,
            "r_squared": fit.r_squared,
            "r_squared_in_spec_band": 0.0 < fit.r_squared < 0.5,
            "sigma_daily": sigma,
            "adv_window_volume": v_daily,
            "sigma_over_volume": sigma_over_v,
            "mean_volume": fit.mean_volume,
        }

    # ---- public conveniences -------------------------------------------------

    def forecast_sqrt(self, data: KyleInputs) -> SqrtImpactPrediction:
        """Pre-trade impact from the square-root law alone (Algorithm C)."""

        closes = data.daily_bars["Close"].astype(float).dropna()
        volumes = data.daily_bars["Volume"].astype(float).dropna()
        sigma = realized_daily_vol(closes)
        v_daily = adv(volumes, window=self.adv_window)
        Q = data.parent_order_shares
        rel = sqrt_law_impact(sigma, Q, v_daily, Y=data.Y_prefactor)
        bps = _BPS_PER_UNIT * rel
        price_now = float(closes.iloc[-1])
        dollars = data.side * rel * price_now * Q
        return SqrtImpactPrediction(
            impact_relative=data.side * rel,
            impact_bps=bps,
            impact_dollars=dollars,
            sigma_daily=sigma,
            volume_daily=v_daily,
            Q=Q,
            Q_over_V=Q / v_daily if v_daily > 0 else float("inf"),
            Y_prefactor=data.Y_prefactor,
            extrapolation_warning=participation_ratio_exceeds_threshold(Q, v_daily),
        )

    def forecast_combined(self, data: KyleInputs) -> CombinedImpactPrediction:
        """Combined Kyle-for-small / sqrt-for-large estimator (Algorithm D)."""

        fit = self._ensure_fit(data)
        closes = data.daily_bars["Close"].astype(float).dropna()
        volumes = data.daily_bars["Volume"].astype(float).dropna()
        sigma = realized_daily_vol(closes)
        v_daily = adv(volumes, window=self.adv_window)
        Q = data.parent_order_shares
        q_star = crossover_size(fit.lambda_hat, data.Y_prefactor, sigma, v_daily)

        # Pick regime by comparing Q to Q* (and protect against the
        # lambda <= 0 case — if we got a degenerate fit, default to sqrt).
        if fit.lambda_hat > 0.0 and q_star >= Q:
            rel = linear_impact(fit.lambda_hat, Q)
            regime: Literal["linear", "sqrt"] = "linear"
        else:
            rel = sqrt_law_impact(sigma, Q, v_daily, Y=data.Y_prefactor)
            regime = "sqrt"
        bps = _BPS_PER_UNIT * rel
        price_now = float(closes.iloc[-1])
        dollars = data.side * rel * price_now * Q
        return CombinedImpactPrediction(
            impact_relative=data.side * rel,
            impact_bps=bps,
            impact_dollars=dollars,
            Q=Q,
            Q_over_V=Q / v_daily if v_daily > 0 else float("inf"),
            Q_crossover=q_star,
            regime=regime,
            lambda_hat=fit.lambda_hat,
            Y_prefactor=data.Y_prefactor,
            sigma_daily=sigma,
            volume_daily=v_daily,
            extrapolation_warning=participation_ratio_exceeds_threshold(Q, v_daily),
        )

    @property
    def fit(self) -> KyleLambdaFit:
        if self._fit is None:
            raise RuntimeError(
                "KyleSqrtImpact has not been calibrated. "
                "Call calibrate(data) first."
            )
        return self._fit

    # ---- internals -----------------------------------------------------------

    @staticmethod
    def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
        """Standardize a yfinance frame: drop NaN OHLCV rows, sort by time."""

        df = bars.copy()
        if "Date" in df.columns:
            df["__ts"] = pd.to_datetime(df["Date"])
        elif "Datetime" in df.columns:
            df["__ts"] = pd.to_datetime(df["Datetime"])
        else:
            df["__ts"] = pd.to_datetime(df.index)
        df = df.sort_values("__ts").reset_index(drop=True)
        keep_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        out = df[keep_cols].copy()
        out.index = df["__ts"]
        out = out.dropna(subset=[c for c in ("Open", "Close", "Volume") if c in out.columns])
        return out

    def _ensure_fit(self, data: KyleInputs) -> KyleLambdaFit:
        if self._fit is None:
            self.calibrate(data)
        assert self._fit is not None
        return self._fit


__all__ = ["KyleSqrtImpact"]
