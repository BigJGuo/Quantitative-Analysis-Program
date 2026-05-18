"""`OnlineLearning` — Layer 4 signals / ML — Hedge / FTRL meta-learner.

The default configuration follows the spec's first deployment pattern:
**Hedge across vol-forecast experts**. The model pulls daily prices for one
ticker, builds an in-process panel of rolling-σ and EWMA-σ "experts", and
replays them through Hedge to produce a meta volatility forecast for the
next trading day.

This keeps the model self-contained — no cross-model imports — while still
exercising the full streaming Hedge update path from the spec. To run Hedge
over actual upstream model outputs (HAR-RV, GARCH, ML ensembles), pass a
pre-built `ExpertStream` to `calibrate()` directly and skip `fetch_data`.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Forecast
from src.models.online_learning.calibration import MODEL_NAME
from src.models.online_learning.calibration import calibrate as _calibrate_impl
from src.models.online_learning.signal import (
    build_vol_expert_panel,
    cusum_break_detect,
    hedge_predict,
    regret_bound_hedge,
    weight_entropy,
)
from src.models.online_learning.types import (
    VALID_KINDS,
    ExpertStream,
    HedgeConfig,
    HedgeFit,
    LearnerKind,
    OnlineLearningInputs,
)

_DEFAULT_DAILY_PERIOD: str = "5y"
_DEFAULT_DAILY_INTERVAL: str = "1d"
_TRADING_DAYS_PER_YEAR: int = 252


@register_model
class OnlineLearning(BaseModel):
    """Hedge / FTRL online learner for one ticker.

    Default configuration: Hedge across five vol-forecast experts (rolling-σ
    over 5/20/60 days plus EWMA-σ for λ=0.94 and λ=0.97). The class behaves
    as a daily-cadence meta volatility forecaster.

    Parameters
    ----------
    ticker:
        Underlying symbol (e.g. ``"SPY"``).
    kind:
        ``"hedge"`` (default), ``"ftrl"``, or ``"ftpl"``. The default
        `fetch_data` path only supports ``"hedge"`` and ``"ftpl"`` (which
        shares Hedge's interface); FTRL workflows should call
        `calibrate_ftrl()` directly with a sparse feature stream.
    rolling_windows:
        Rolling-σ expert windows (in trading days).
    ewma_lambdas:
        EWMA-σ decay constants.
    eta:
        Hedge constant learning rate (used when ``eta_schedule="constant"``).
    eta_schedule:
        ``"constant"`` or ``"adaptive"`` (``sqrt(log N / t)``).
    fixed_share:
        Herbster-Warmuth mixing (`beta`). `0.01` is typical for
        non-stationary streams.
    l_max:
        Loss clip ceiling. Defaults to a value appropriate for log-squared
        vol losses (rarely exceeded on liquid US equities).
    loss_kind:
        ``"log_squared"`` (default for vols), ``"squared"``, or ``"log"``.
    annualize:
        If ``True``, the predicted next-day daily vol is reported as
        `sqrt(252) * sigma_daily`.
    daily_period:
        yfinance ``period`` string for the daily history pull.
    warm_up_passes:
        Number of times the historical stream is replayed during calibration.
    cusum_threshold:
        If positive, `validate()` flags a regime break whenever the
        prequential-loss CUSUM exceeds it (spec: "Reset on regime break").
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "daily"

    def __init__(
        self,
        ticker: str,
        *,
        kind: LearnerKind = "hedge",
        rolling_windows: tuple[int, ...] = (5, 20, 60),
        ewma_lambdas: tuple[float, ...] = (0.94, 0.97),
        eta: float = 0.5,
        eta_schedule: str = "adaptive",
        fixed_share: float = 0.01,
        l_max: float = 4.0,
        loss_kind: str = "log_squared",
        annualize: bool = True,
        daily_period: str = _DEFAULT_DAILY_PERIOD,
        warm_up_passes: int = 1,
        cusum_threshold: float = 0.0,
    ) -> None:
        if not ticker:
            raise ValueError("OnlineLearning: ticker must be a non-empty string")
        if kind not in VALID_KINDS:
            raise ValueError(
                f"OnlineLearning: kind must be one of {sorted(VALID_KINDS)}, "
                f"got {kind!r}"
            )
        if kind == "ftrl":
            raise ValueError(
                "OnlineLearning.fetch_data does not support FTRL — use "
                "calibrate_ftrl() with a sparse feature stream instead."
            )
        if len(rolling_windows) + len(ewma_lambdas) < 2:
            raise ValueError(
                "OnlineLearning: need at least 2 experts in total "
                f"(got {len(rolling_windows)} rolling + {len(ewma_lambdas)} ewma)"
            )

        self.ticker = ticker
        self.kind: LearnerKind = kind
        self.rolling_windows = tuple(int(w) for w in rolling_windows)
        self.ewma_lambdas = tuple(float(lam) for lam in ewma_lambdas)
        self.eta = float(eta)
        self.eta_schedule = eta_schedule
        self.fixed_share = float(fixed_share)
        self.l_max = float(l_max)
        self.loss_kind = loss_kind
        self.annualize = annualize
        self.daily_period = daily_period
        self.warm_up_passes = int(warm_up_passes)
        self.cusum_threshold = float(cusum_threshold)
        self._fit: HedgeFit | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> OnlineLearningInputs:
        bars = provider.fetch_prices(
            self.ticker, self.daily_period, _DEFAULT_DAILY_INTERVAL
        )
        if bars is None or bars.empty:
            raise RuntimeError(
                f"OnlineLearning.fetch_data: no daily prices for {self.ticker!r}"
            )

        prices = _close_series(bars)
        stream = build_vol_expert_panel(
            prices,
            rolling_windows=self.rolling_windows,
            ewma_lambdas=self.ewma_lambdas,
        )

        if stream.predictions.shape[0] < max(self.rolling_windows) + 5:
            raise RuntimeError(
                f"OnlineLearning.fetch_data: too few rounds after alignment "
                f"({stream.predictions.shape[0]})"
            )

        metadata: dict[str, Any] = {
            "n_rounds": float(stream.predictions.shape[0]),
            "n_experts": float(stream.predictions.shape[1]),
            "ticker": self.ticker,
            "rolling_windows": list(self.rolling_windows),
            "ewma_lambdas": list(self.ewma_lambdas),
        }
        return OnlineLearningInputs(
            ticker=self.ticker,
            stream=stream,
            kind=self.kind,
            timestamp=datetime.now(UTC),
            horizon="1d",
            annualize=self.annualize,
            trading_days_per_year=_TRADING_DAYS_PER_YEAR,
            metadata=metadata,
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        inputs = self._require_inputs(data)
        config = self._build_hedge_config(inputs.stream)
        result = _calibrate_impl(
            stream=inputs.stream,
            config=config,
            n_passes=self.warm_up_passes,
            timestamp=inputs.timestamp,
            metadata={"ticker": inputs.ticker, "kind": inputs.kind},
        )
        self._fit = cast(HedgeFit, result.parameters["hedge_fit"])
        return result

    def predict(self, data: Any) -> Forecast:
        inputs = self._require_inputs(data)
        fit = self._require_fit()
        last_predictions = inputs.stream.predictions.iloc[-1].to_numpy(dtype=float)
        meta_pred_daily = hedge_predict(fit.weights, last_predictions)
        if inputs.annualize:
            value = float(meta_pred_daily * math.sqrt(inputs.trading_days_per_year))
            horizon_label = f"{inputs.horizon}_annualized"
        else:
            value = float(meta_pred_daily)
            horizon_label = inputs.horizon

        per_expert_predictions = {
            name: float(p)
            for name, p in zip(fit.expert_names, last_predictions, strict=True)
        }
        metadata = {
            "kind": inputs.kind,
            "ticker": inputs.ticker,
            "meta_prediction_daily": float(meta_pred_daily),
            "annualized": inputs.annualize,
            "trading_days_per_year": inputs.trading_days_per_year,
            "weights": fit.weights.tolist(),
            "expert_names": list(fit.expert_names),
            "expert_predictions_daily": per_expert_predictions,
            "weight_entropy": weight_entropy(fit.weights),
            "as_of_date": str(inputs.stream.predictions.index[-1]),
        }
        return Forecast(
            ticker=inputs.ticker,
            horizon=horizon_label,
            value=value,
            timestamp=inputs.timestamp,
            metadata=metadata,
        )

    def validate(self, data: Any) -> dict[str, Any]:
        self._require_inputs(data)
        fit = self._require_fit()
        t = int(fit.weight_history.shape[0])
        n = int(fit.weights.shape[0])

        mean_loss = (
            float(np.nanmean(fit.prequential_losses.to_numpy(dtype=float)))
            if fit.prequential_losses.notna().any()
            else float("nan")
        )

        # The argmin of the cumulative-loss vector is the "best fixed expert."
        best_idx = int(np.argmin(fit.cumulative_expert_losses))
        best_name = fit.expert_names[best_idx]
        best_loss = float(fit.cumulative_expert_losses[best_idx])

        regret_theory = regret_bound_hedge(t, n) if t >= 1 and n >= 2 else float("nan")

        regime_break = (
            cusum_break_detect(
                fit.prequential_losses,
                h=self.cusum_threshold,
            )
            if self.cusum_threshold > 0
            else False
        )

        return {
            "n_rounds": t,
            "n_experts": n,
            "cumulative_meta_loss": float(fit.cumulative_meta_loss),
            "mean_prequential_loss": mean_loss,
            "best_expert_index": best_idx,
            "best_expert_name": best_name,
            "best_expert_cum_loss": best_loss,
            "empirical_regret": float(fit.empirical_regret),
            "theoretical_regret_bound": regret_theory,
            "regret_ratio_vs_bound": (
                float(fit.empirical_regret) / regret_theory
                if regret_theory > 0 and math.isfinite(regret_theory)
                else float("nan")
            ),
            "final_weight_entropy": weight_entropy(fit.weights),
            "uniform_entropy": math.log(n),
            "max_weight": float(fit.weights.max()),
            "min_weight": float(fit.weights.min()),
            "regime_break_flag": regime_break,
            "weights": fit.weights.tolist(),
            "expert_names": list(fit.expert_names),
        }

    # ----- Public conveniences -----------------------------------------------

    @property
    def fit(self) -> HedgeFit:
        return self._require_fit()

    # ----- Internal helpers --------------------------------------------------

    def _build_hedge_config(self, stream: ExpertStream) -> HedgeConfig:
        return HedgeConfig(
            n_experts=int(stream.predictions.shape[1]),
            eta=self.eta,
            eta_schedule=cast(Any, self.eta_schedule),
            fixed_share=self.fixed_share,
            l_max=self.l_max,
            loss_kind=cast(Any, self.loss_kind),
        )

    def _require_fit(self) -> HedgeFit:
        if self._fit is None:
            raise RuntimeError(
                "OnlineLearning has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    @staticmethod
    def _require_inputs(data: Any) -> OnlineLearningInputs:
        if not isinstance(data, OnlineLearningInputs):
            raise TypeError(
                f"OnlineLearning expects OnlineLearningInputs, got "
                f"{type(data).__name__}"
            )
        return data


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _close_series(bars: pd.DataFrame) -> pd.Series:
    """Extract a clean `Close` series from a yfinance-style daily OHLC frame."""

    if "Close" not in bars.columns:
        raise ValueError(
            "OnlineLearning: daily bars frame must include a 'Close' column"
        )
    # yfinance's reset_index moves the trading date into a 'Date' or
    # 'Datetime' column; the fixture provider may keep it on the index.
    if "Date" in bars.columns:
        raw_dates: Any = bars["Date"]
    elif "Datetime" in bars.columns:
        raw_dates = bars["Datetime"]
    else:
        raw_dates = bars.index
    idx = pd.DatetimeIndex(pd.to_datetime(raw_dates)).normalize()
    series = pd.Series(
        bars["Close"].to_numpy(dtype=float),
        index=idx,
        name="Close",
    )
    return series.dropna()


__all__ = ["OnlineLearning"]
