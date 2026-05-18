"""`GARCHModel` — Layer 4 ML/signals daily-volatility model.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. The class fetches `~10 years` of daily closes through the
shared `DataProvider`, calls `calibrate` to fit one of {GARCH, GJR-GARCH,
EGARCH}, and projects the result into a `RiskMetric` (one-step daily vol,
annualized) for the orchestration layer.

The full forecast / diagnostics surface (multi-horizon variance forecast,
Ljung-Box, ARCH-LM, sign-bias, VaR, QLIKE) is exposed via the class-level
helpers `forecast()` and `validate()`.

Spec: `models/layer4_signals_ml/08_garch_family.md`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.garch_family.calibration import MODEL_NAME
from src.models.garch_family.calibration import calibrate as _calibrate_impl
from src.models.garch_family.signal import (
    annualize_vol_pct,
    arch_lm_test,
    forecast_variance,
    kupiec_pof_test,
    ljung_box,
    log_returns_pct,
    mincer_zarnowitz_regression,
    qlike_loss,
    sign_bias_test,
    value_at_risk,
)
from src.models.garch_family.types import (
    VALID_DISTS,
    VALID_MEAN_MODELS,
    VALID_SPECS,
    GARCHFit,
    GARCHForecastResult,
    GARCHInputs,
    GARCHSpec,
    InnovationDist,
    MeanModel,
)

_DEFAULT_PERIOD: str = "10y"
_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 22)
_DEFAULT_REFIT_FREQUENCY: Literal["monthly"] = "monthly"
_VAR_ALPHA: float = 0.05


@register_model
class GARCHModel(BaseModel):
    """GARCH-family conditional-volatility model.

    Parameters
    ----------
    ticker:
        Symbol to fit. The spec is single-asset; one model instance per
        ticker.
    spec:
        ``"GARCH"`` (default), ``"GJR"``, or ``"EGARCH"``.
    distribution:
        ``"Student-t"`` (default) or ``"Gaussian"`` innovations.
    mean_model:
        ``"Constant"`` (default) or ``"Zero"``.
    history_period:
        yfinance ``period`` argument for the history pull. Spec recommends
        5–10 years; default is ``"10y"``.
    forecast_horizons:
        Trading-day horizons reported by `forecast()` / `validate()`. Spec
        defaults: 1, 5, 10, 22 days.
    var_alpha:
        Confidence level for the conditional VaR estimate (default 5%).
    max_iter:
        Cap on Nelder-Mead iterations during MLE.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[Literal["monthly"]] = _DEFAULT_REFIT_FREQUENCY

    def __init__(
        self,
        ticker: str,
        *,
        spec: GARCHSpec = "GARCH",
        distribution: InnovationDist = "Student-t",
        mean_model: MeanModel = "Constant",
        history_period: str = _DEFAULT_PERIOD,
        forecast_horizons: tuple[int, ...] = _DEFAULT_HORIZONS,
        var_alpha: float = _VAR_ALPHA,
        max_iter: int = 2000,
        now_func: Any = None,
    ) -> None:
        if not ticker:
            raise ValueError("GARCHModel requires a non-empty ticker")
        if spec not in VALID_SPECS:
            raise ValueError(
                f"spec must be one of {sorted(VALID_SPECS)}, got {spec!r}"
            )
        if distribution not in VALID_DISTS:
            raise ValueError(
                f"distribution must be one of {sorted(VALID_DISTS)}, got {distribution!r}"
            )
        if mean_model not in VALID_MEAN_MODELS:
            raise ValueError(
                f"mean_model must be one of {sorted(VALID_MEAN_MODELS)}, "
                f"got {mean_model!r}"
            )
        if not forecast_horizons or any(h < 1 for h in forecast_horizons):
            raise ValueError(
                f"forecast_horizons must be a non-empty tuple of positive ints, "
                f"got {forecast_horizons}"
            )
        if not 0.0 < var_alpha < 1.0:
            raise ValueError(f"var_alpha must lie in (0, 1), got {var_alpha}")

        self.ticker = ticker.upper()
        self.spec: GARCHSpec = spec
        self.distribution: InnovationDist = distribution
        self.mean_model: MeanModel = mean_model
        self.history_period = history_period
        self.forecast_horizons = tuple(int(h) for h in forecast_horizons)
        self.var_alpha = float(var_alpha)
        self.max_iter = int(max_iter)
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._fit: GARCHFit | None = None
        self._last_returns: pd.Series | None = None

    # ---- BaseModel interface -------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> GARCHInputs:
        bars = provider.fetch_prices(
            self.ticker, self.history_period, _DEFAULT_INTERVAL
        )
        if bars is None or bars.empty or "Close" not in bars.columns:
            raise RuntimeError(
                f"No daily price history returned for {self.ticker!r}; "
                "cannot fit GARCH."
            )
        close = bars["Close"].astype(float)
        if "Date" in bars.columns:
            close.index = pd.to_datetime(bars["Date"])
        elif "Datetime" in bars.columns:
            close.index = pd.to_datetime(bars["Datetime"])
        else:
            close.index = pd.to_datetime(bars.index)
        close = close.sort_index().dropna()
        returns = log_returns_pct(close)
        if len(returns) < 250:
            raise RuntimeError(
                f"GARCH needs >= 250 daily returns for a stable fit, "
                f"got {len(returns)} for {self.ticker!r}"
            )
        self._last_returns = returns
        return GARCHInputs(
            ticker=self.ticker,
            returns_pct=returns,
            timestamp=self._now_func(),
            spec=self.spec,
            distribution=self.distribution,
            mean_model=self.mean_model,
            forecast_horizons=self.forecast_horizons,
            metadata={"period": self.history_period, "n_obs": len(returns)},
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, GARCHInputs):
            raise TypeError(
                f"GARCHModel.calibrate expects GARCHInputs, got {type(data).__name__}"
            )
        result = _calibrate_impl(
            returns=data.returns_pct,
            spec=data.spec,
            distribution=data.distribution,
            mean_model=data.mean_model,
            timestamp=data.timestamp,
            max_iter=self.max_iter,
        )
        self._fit = cast(GARCHFit, result.parameters["fit"])
        self._last_returns = data.returns_pct
        return result

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, GARCHInputs):
            raise TypeError(
                f"GARCHModel.predict expects GARCHInputs, got {type(data).__name__}"
            )
        fit = self._ensure_fit(data)
        fc = self.forecast(data)
        # Primary deliverable: one-step daily vol, annualized.
        sigma_daily_pct = float(fc.vols[0])
        sigma_ann = annualize_vol_pct(sigma_daily_pct)
        var_5pct = value_at_risk(
            sigma_daily_pct,
            mu_pct=fit.params.mu,
            alpha=self.var_alpha,
            distribution=fit.distribution,
            nu=fit.params.nu,
        )
        return RiskMetric(
            ticker=self.ticker,
            metric_name="conditional_volatility",
            value=sigma_ann,
            timestamp=data.timestamp,
            confidence_level=None,
            horizon="1d",
            metadata={
                "spec": fit.spec,
                "distribution": fit.distribution,
                "sigma_daily_pct": sigma_daily_pct,
                "sigma_annualized": sigma_ann,
                "forecast_horizons": list(self.forecast_horizons),
                "forecast_variances_pct2": fc.variances.tolist(),
                "forecast_vols_pct": fc.vols.tolist(),
                "forecast_annualized_vols": fc.annualized_vols.tolist(),
                "var_alpha": self.var_alpha,
                "var_pct": var_5pct,
                "params": fit.params.to_dict(),
                "persistence": float(fit.params.alpha + fit.params.beta),
                "unconditional_variance": fit.unconditional_variance,
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, GARCHInputs):
            raise TypeError(
                f"GARCHModel.validate expects GARCHInputs, got {type(data).__name__}"
            )
        fit = self._ensure_fit(data)
        z = fit.std_resid
        eps = fit.eps_path

        lb_z_10 = ljung_box(z, lags=10)
        lb_z_20 = ljung_box(z, lags=20)
        lb_z2_10 = ljung_box(z * z, lags=10)
        lb_z2_20 = ljung_box(z * z, lags=20)
        archlm = arch_lm_test(z, lags=10)
        sign_bias = sign_bias_test(z, eps)

        # Mincer-Zarnowitz on in-sample squared returns vs predicted variance.
        rv_proxy = eps * eps
        mz = mincer_zarnowitz_regression(rv_proxy, fit.sigma2_path)

        # In-sample VaR backtest using fitted innovation distribution.
        sigma_path = np.sqrt(fit.sigma2_path)
        var_cutoffs = np.empty_like(sigma_path)
        for i, s_t in enumerate(sigma_path):
            var_cutoffs[i] = value_at_risk(
                float(s_t),
                mu_pct=fit.params.mu,
                alpha=self.var_alpha,
                distribution=fit.distribution,
                nu=fit.params.nu,
            )
        losses = -(eps)  # eps is mean-centered return in percent; loss = -return.
        # Skip the first 30 obs (variance warm-up).
        warmup = 30
        breaches = int(np.sum(losses[warmup:] > var_cutoffs[warmup:]))
        n_eff = int(losses.size - warmup)
        kupiec = kupiec_pof_test(breaches, n_eff, self.var_alpha)

        ql = qlike_loss(rv_proxy, fit.sigma2_path)

        persistence: float
        if fit.spec == "GARCH":
            persistence = fit.params.alpha + fit.params.beta
        elif fit.spec == "GJR":
            persistence = fit.params.alpha + 0.5 * fit.params.gamma + fit.params.beta
        else:
            persistence = fit.params.beta

        out: dict[str, Any] = {
            "ticker": self.ticker,
            "spec": fit.spec,
            "distribution": fit.distribution,
            "n_obs": fit.n_obs,
            "converged": fit.converged,
            "log_likelihood": fit.log_likelihood,
            "persistence": float(persistence),
            "unconditional_variance": fit.unconditional_variance,
            "mean_sigma_pct": float(sigma_path.mean()),
            "final_sigma_pct": float(sigma_path[-1]),
            "ljung_box_z_lag10_q": lb_z_10[0],
            "ljung_box_z_lag10_p": lb_z_10[1],
            "ljung_box_z_lag20_q": lb_z_20[0],
            "ljung_box_z_lag20_p": lb_z_20[1],
            "ljung_box_z2_lag10_q": lb_z2_10[0],
            "ljung_box_z2_lag10_p": lb_z2_10[1],
            "ljung_box_z2_lag20_q": lb_z2_20[0],
            "ljung_box_z2_lag20_p": lb_z2_20[1],
            "arch_lm_lag10_stat": archlm[0],
            "arch_lm_lag10_p": archlm[1],
            "sign_bias_joint_F": sign_bias["joint_F"],
            "sign_bias_joint_p": sign_bias["joint_p"],
            "mincer_zarnowitz_intercept": mz["intercept"],
            "mincer_zarnowitz_slope": mz["slope"],
            "mincer_zarnowitz_r2": mz["r_squared"],
            "var_backtest_breaches": breaches,
            "var_backtest_n": n_eff,
            "var_backtest_breach_rate": breaches / n_eff if n_eff > 0 else float("nan"),
            "kupiec_pof_LR": kupiec[0],
            "kupiec_pof_p": kupiec[1],
            "qlike_loss": ql,
        }
        return out

    # ---- public conveniences -------------------------------------------------

    def forecast(self, data: GARCHInputs | None = None) -> GARCHForecastResult:
        """Multi-horizon variance / vol forecast. Defaults to the horizons
        configured on this instance; uses the data's horizons if supplied.
        """

        fit = self._fit
        if fit is None:
            if data is None:
                raise RuntimeError(
                    "GARCHModel.forecast called before calibrate() and without "
                    "an inputs bundle to lazy-fit on."
                )
            self.calibrate(data)
            fit = self._fit
        assert fit is not None
        horizons = (
            tuple(int(h) for h in data.forecast_horizons)
            if data is not None
            else self.forecast_horizons
        )
        variances = forecast_variance(fit, horizons)
        vols = np.sqrt(variances)
        annualized = np.asarray([annualize_vol_pct(float(v)) for v in vols], dtype=float)
        return GARCHForecastResult(
            horizons=horizons,
            variances=variances,
            vols=vols,
            annualized_vols=annualized,
            timestamp=self._now_func(),
        )

    @property
    def fit(self) -> GARCHFit:
        if self._fit is None:
            raise RuntimeError(
                "GARCHModel has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    # ---- internals -----------------------------------------------------------

    def _ensure_fit(self, data: GARCHInputs) -> GARCHFit:
        if self._fit is None:
            self.calibrate(data)
        assert self._fit is not None
        return self._fit


__all__ = ["GARCHModel"]
