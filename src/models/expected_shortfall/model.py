"""`ExpectedShortfall` — Layer-6 FRTB 97.5% ES risk model.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. Fetches daily adjusted close prices for every portfolio
ticker through the shared `DataProvider`, builds an aligned return panel,
calibrates `(mu, Sigma)` (and `nu, loc, scale` for the Student-t path), then
dispatches to one of the four ES estimators specified in
`models/layer6_risk/18_expected_shortfall.md`.

`predict()` returns a `RiskMetric` whose `value` is the dollar ES; the
`metadata` dict carries VaR, the breach tail, ES/VaR ratio, traffic-light
band, and the realized fit parameters.

Spec: `models/layer6_risk/18_expected_shortfall.md`.
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
from src.models.expected_shortfall.calibration import MODEL_NAME
from src.models.expected_shortfall.calibration import calibrate as _calibrate_impl
from src.models.expected_shortfall.signal import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    apply_liquidity_scaling,
    es_gaussian,
    es_historical,
    es_monte_carlo,
    es_student_t,
    portfolio_pnl,
    rolling_realized_tail_mean,
    traffic_light_band,
)
from src.models.expected_shortfall.types import (
    FRTB_ALPHA,
    VALID_METHODS,
    ESFit,
    ESInputs,
    ESMethod,
    ESResult,
)

_DEFAULT_PERIOD: str = "3y"
_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_LOOKBACK_N: int = 500
_DEFAULT_MC_DRAWS: int = 100_000
_DEFAULT_MC_NU: float = 5.0


@register_model
class ExpectedShortfall(BaseModel):
    """FRTB 97.5% Expected Shortfall for a dollar-weighted equity portfolio.

    Parameters
    ----------
    positions:
        ``{ticker -> dollar_notional}``. Sign carries long/short.
    alpha:
        Confidence level. Spec / Basel default: 0.975.
    method:
        Which ES estimator to use. One of
        ``"parametric_normal"`` / ``"parametric_t"`` / ``"historical"``
        / ``"monte_carlo"``.
    lookback_N:
        Trailing days included in the calibration / empirical tail.
        Recommended: 500 at alpha=0.975 (yields ~12 breach observations).
    history_period:
        yfinance ``period`` argument. ``"3y"`` covers `lookback_N=500` with
        ample buffer.
    mc_draws / mc_nu:
        Monte-Carlo simulation size and degrees of freedom for the
        multivariate-t draws used by `method="monte_carlo"`.
    portfolio_id:
        Human-readable label carried through to `RiskMetric.ticker`.
    liquidity_scalars:
        Optional per-ticker `sqrt(h_k / 10)` multipliers for FRTB IMA. Order
        must match the sorted ticker list (use `tickers` after `fetch_data`
        to inspect the canonical ordering).
    mc_seed:
        Optional seed for the Monte-Carlo RNG (reproducibility in tests).
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 6
    refit_frequency: ClassVar[Literal["daily"]] = "daily"

    def __init__(
        self,
        positions: dict[str, float],
        *,
        alpha: float = FRTB_ALPHA,
        method: ESMethod = "historical",
        lookback_N: int = _DEFAULT_LOOKBACK_N,
        history_period: str = _DEFAULT_PERIOD,
        mc_draws: int = _DEFAULT_MC_DRAWS,
        mc_nu: float = _DEFAULT_MC_NU,
        portfolio_id: str = "PORTFOLIO",
        liquidity_scalars: np.ndarray | None = None,
        mc_seed: int | None = None,
        now_func: Any = None,
    ) -> None:
        if not positions:
            raise ValueError("ExpectedShortfall requires a non-empty positions dict")
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(VALID_METHODS)}, got {method!r}"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        if lookback_N < 30:
            raise ValueError(f"lookback_N must be >= 30, got {lookback_N}")
        if mc_draws < 1000:
            raise ValueError(f"mc_draws must be >= 1000, got {mc_draws}")
        if mc_nu <= 2.0:
            raise ValueError(f"mc_nu must be > 2, got {mc_nu}")

        self.positions = {k.upper(): float(v) for k, v in positions.items()}
        self.tickers: tuple[str, ...] = tuple(sorted(self.positions.keys()))
        self.alpha = float(alpha)
        self.method: ESMethod = method
        self.lookback_N = int(lookback_N)
        self.history_period = history_period
        self.mc_draws = int(mc_draws)
        self.mc_nu = float(mc_nu)
        self.portfolio_id = portfolio_id
        self.liquidity_scalars = (
            np.asarray(liquidity_scalars, dtype=float)
            if liquidity_scalars is not None
            else None
        )
        self.mc_seed = mc_seed
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._fit: ESFit | None = None

    # ---- BaseModel interface -------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> ESInputs:
        frames: dict[str, pd.Series] = {}
        for ticker in self.tickers:
            bars = provider.fetch_prices(ticker, self.history_period, _DEFAULT_INTERVAL)
            if bars is None or bars.empty or "Close" not in bars.columns:
                raise RuntimeError(
                    f"No daily price history returned for {ticker!r}; cannot "
                    "compute ES."
                )
            close = bars["Close"].astype(float)
            if "Date" in bars.columns:
                close.index = pd.to_datetime(bars["Date"])
            elif "Datetime" in bars.columns:
                close.index = pd.to_datetime(bars["Datetime"])
            else:
                close.index = pd.to_datetime(bars.index)
            close = close.sort_index().dropna()
            idx = close.index
            tz = getattr(idx, "tz", None)
            if tz is not None and hasattr(idx, "tz_localize"):
                close.index = idx.tz_localize(None)
            frames[ticker] = close

        prices = pd.DataFrame(frames).sort_index().dropna(how="any")
        if len(prices) < self.lookback_N + 1:
            raise RuntimeError(
                f"Aligned price panel has only {len(prices)} rows; need "
                f"{self.lookback_N + 1} to compute returns over lookback="
                f"{self.lookback_N}."
            )
        returns = prices.pct_change().dropna()
        dollar_positions = np.array(
            [self.positions[t] for t in self.tickers], dtype=float
        )

        return ESInputs(
            tickers=self.tickers,
            returns=returns,
            dollar_positions=dollar_positions,
            alpha=self.alpha,
            method=self.method,
            lookback_N=self.lookback_N,
            timestamp=self._now_func(),
            portfolio_id=self.portfolio_id,
            liquidity_scalars=self.liquidity_scalars,
            metadata={
                "period": self.history_period,
                "n_obs_total": int(returns.shape[0]),
            },
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        if not isinstance(data, ESInputs):
            raise TypeError(
                f"ExpectedShortfall.calibrate expects ESInputs, got "
                f"{type(data).__name__}"
            )
        result = _calibrate_impl(data, timestamp=data.timestamp)
        self._fit = cast(ESFit, result.parameters["fit"])
        return result

    def predict(self, data: Any) -> RiskMetric:
        if not isinstance(data, ESInputs):
            raise TypeError(
                f"ExpectedShortfall.predict expects ESInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._ensure_fit(data)
        es_result = self.compute_es(data, fit=fit)
        ratio = es_result.ES / es_result.VaR if es_result.VaR > 0 else float("nan")
        return RiskMetric(
            ticker=self.portfolio_id,
            metric_name="expected_shortfall",
            value=float(es_result.ES),
            timestamp=data.timestamp,
            confidence_level=self.alpha,
            horizon="1d",
            metadata={
                "method": self.method,
                "VaR": float(es_result.VaR),
                "ES": float(es_result.ES),
                "es_var_ratio": float(ratio),
                "traffic_light": traffic_light_band(ratio),
                "lookback_N": self.lookback_N,
                "n_tail_obs": int(es_result.tail_losses.size),
                "tickers": list(self.tickers),
                "dollar_positions": data.dollar_positions.tolist(),
                "portfolio_notional": float(np.abs(data.dollar_positions).sum()),
                "sigma_p": fit.sigma_p,
                "nu": fit.nu,
                "loc": fit.loc,
                "scale": fit.scale,
                **es_result.metadata,
            },
        )

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, ESInputs):
            raise TypeError(
                f"ExpectedShortfall.validate expects ESInputs, got "
                f"{type(data).__name__}"
            )
        fit = self._ensure_fit(data)

        # Build rolling per-day VaR / ES paths over the lookback window using
        # the *same* method, then run Acerbi-Szekely Z1 / Z2 against the
        # realized portfolio losses.
        rets = apply_liquidity_scaling(
            data.returns.tail(data.lookback_N), data.liquidity_scalars
        )
        pnl_path = portfolio_pnl(rets, data.dollar_positions)
        losses = -pnl_path

        # Single-shot VaR/ES applied across the window — the standard
        # approximation for the AS backtest when no path-dependent re-fit is
        # available (a rolling refit is much more expensive and is the focus
        # of the rolling-tail-mean diagnostic below).
        es_result = self.compute_es(data, fit=fit)
        var_path = np.full(losses.shape, es_result.VaR, dtype=float)
        es_path = np.full(losses.shape, es_result.ES, dtype=float)

        z1 = acerbi_szekely_z1(losses, var_path, es_path)
        z2 = acerbi_szekely_z2(losses, var_path, es_path, self.alpha)

        # Rolling realized tail mean for the ES-vs-realized diagnostic plot.
        window = min(250, losses.size)
        rolling_tail = rolling_realized_tail_mean(losses, window=window, alpha=self.alpha)
        finite_tail = rolling_tail[np.isfinite(rolling_tail)]

        breaches = int(np.sum(losses > es_result.VaR))
        breach_rate = breaches / losses.size if losses.size else float("nan")

        cov_eigs = np.linalg.eigvalsh(fit.Sigma)
        is_psd = bool(np.all(cov_eigs >= -1e-10))

        out: dict[str, Any] = {
            "portfolio_id": self.portfolio_id,
            "method": self.method,
            "alpha": self.alpha,
            "n_obs": fit.n_obs,
            "VaR": float(es_result.VaR),
            "ES": float(es_result.ES),
            "es_var_ratio": float(es_result.ES / es_result.VaR)
            if es_result.VaR > 0
            else float("nan"),
            "breaches_in_sample": breaches,
            "breach_rate_in_sample": float(breach_rate),
            "expected_breach_rate": float(1.0 - self.alpha),
            "acerbi_szekely_z1": z1,
            "acerbi_szekely_z2": z2,
            "rolling_tail_mean_avg": float(finite_tail.mean())
            if finite_tail.size
            else float("nan"),
            "rolling_tail_mean_max": float(finite_tail.max())
            if finite_tail.size
            else float("nan"),
            "tail_obs_count": int(es_result.tail_losses.size),
            "covariance_is_psd": is_psd,
            "covariance_min_eig": float(cov_eigs.min()) if cov_eigs.size else float("nan"),
            "sigma_p": fit.sigma_p,
        }
        if fit.nu is not None:
            out["t_nu"] = fit.nu
            out["t_loc"] = fit.loc
            out["t_scale"] = fit.scale
        return out

    # ---- public conveniences -------------------------------------------------

    def compute_es(
        self, data: ESInputs, *, fit: ESFit | None = None
    ) -> ESResult:
        """Run the configured ES estimator on `data`. Lazy-fits if needed."""

        if fit is None:
            fit = self._ensure_fit(data)
        rets = apply_liquidity_scaling(
            data.returns.tail(data.lookback_N), data.liquidity_scalars
        )
        w = data.dollar_positions

        if self.method == "parametric_normal":
            mu_p = float(fit.mu @ w)
            var, es = es_gaussian(-mu_p, fit.sigma_p, self.alpha)
            return ESResult(
                VaR=var,
                ES=es,
                method=self.method,
                alpha=self.alpha,
                tail_losses=np.array([], dtype=float),
                timestamp=data.timestamp,
                metadata={"mu_p": mu_p},
            )

        if self.method == "parametric_t":
            assert fit.nu is not None and fit.loc is not None and fit.scale is not None
            var, es = es_student_t(fit.loc, fit.scale, fit.nu, self.alpha)
            return ESResult(
                VaR=var,
                ES=es,
                method=self.method,
                alpha=self.alpha,
                tail_losses=np.array([], dtype=float),
                timestamp=data.timestamp,
                metadata={"nu": fit.nu, "loc": fit.loc, "scale": fit.scale},
            )

        if self.method == "historical":
            losses = -portfolio_pnl(rets, w)
            var, es, tail = es_historical(losses, self.alpha)
            return ESResult(
                VaR=var,
                ES=es,
                method=self.method,
                alpha=self.alpha,
                tail_losses=tail,
                timestamp=data.timestamp,
                metadata={"n_tail": int(tail.size)},
            )

        if self.method == "monte_carlo":
            rng = (
                np.random.default_rng(self.mc_seed)
                if self.mc_seed is not None
                else np.random.default_rng()
            )
            var, es, tail = es_monte_carlo(
                fit.mu,
                fit.Sigma,
                w,
                self.alpha,
                K=self.mc_draws,
                nu=self.mc_nu,
                rng=rng,
            )
            return ESResult(
                VaR=var,
                ES=es,
                method=self.method,
                alpha=self.alpha,
                tail_losses=tail,
                timestamp=data.timestamp,
                metadata={"K": self.mc_draws, "nu": self.mc_nu},
            )

        raise ValueError(f"Unknown method {self.method!r}")

    @property
    def fit(self) -> ESFit:
        if self._fit is None:
            raise RuntimeError(
                "ExpectedShortfall has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    # ---- internals -----------------------------------------------------------

    def _ensure_fit(self, data: ESInputs) -> ESFit:
        if self._fit is None:
            self.calibrate(data)
        assert self._fit is not None
        return self._fit


__all__ = ["ExpectedShortfall"]
