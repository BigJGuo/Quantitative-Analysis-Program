"""`BayesianKalman` — Layer 4 signals / ML.

Thin orchestration shell around the pure functions in `signal.py` and
`calibration.py`. The class dispatches on a ``mode`` parameter:

- ``"kalman_beta"``     — fetches asset + market daily returns and runs the
  time-varying-beta Kalman filter. `predict` returns a `Forecast` whose
  ``value`` is the latest beta estimate, with `(lower, upper)` set to a
  68% posterior interval derived from `P_{T|T}`.
- ``"hierarchical"``    — fetches a panel of asset returns + a market
  proxy and fits the Gaussian hierarchical regression of asset returns on
  the market (one coefficient per asset, partial-pooled). `predict` returns
  a `Forecast` for a chosen target asset.
- ``"dynamic_factor"``  — fetches a multi-asset panel and runs the
  Stock-Watson dynamic factor model via EM. `predict` returns a
  `RiskMetric` summarizing the latent-factor risk decomposition.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from src.core.base_model import BaseModel, RefitFrequency
from src.core.data_provider import DataProvider
from src.core.registry import register_model
from src.core.types import CalibrationResult, Forecast, RiskMetric, Signal
from src.models.bayesian_kalman.calibration import MODEL_NAME
from src.models.bayesian_kalman.calibration import calibrate as _calibrate_impl
from src.models.bayesian_kalman.signal import (
    _KalmanRun as KalmanRun,
)
from src.models.bayesian_kalman.signal import (
    innovation_normality_kurtosis,
    ljung_box_pvalue,
    standardized_innovations,
)
from src.models.bayesian_kalman.types import (
    VALID_MODES,
    BayesianKalmanInputs,
    DynamicFactorFit,
    HierarchicalFit,
    KalmanFit,
    ModelMode,
)

_DEFAULT_PERIOD: str = "2y"
_DEFAULT_INTERVAL: str = "1d"
_DEFAULT_MARKET: str = "SPY"


@register_model
class BayesianKalman(BaseModel):
    """Bayesian-hierarchical / Kalman signals model.

    Parameters
    ----------
    ticker:
        Primary asset. Required for ``"kalman_beta"`` and ``"hierarchical"``
        modes (where it doubles as the prediction target).
    market_ticker:
        Market proxy used in the observation equation. Default ``"SPY"``.
    universe:
        Required for ``"hierarchical"`` (the panel of assets to share
        information across) and ``"dynamic_factor"`` (the panel to extract
        latent factors from). ``ticker`` is always included implicitly when
        present.
    mode:
        Which estimator to dispatch to (see module docstring).
    period:
        yfinance period string (e.g. ``"2y"``).
    interval:
        yfinance bar interval (e.g. ``"1d"``).
    fix_hyperparameters:
        Optional ``(q_alpha, q_beta, R)`` triple for ``"kalman_beta"``.
        Skips MLE and uses these values directly — useful for fast
        incremental refits.
    n_iter / n_burn:
        Gibbs iterations and burn-in (``"hierarchical"`` only).
    n_em_iter:
        EM iterations (``"dynamic_factor"`` only).
    n_factors:
        Number of latent factors (``"dynamic_factor"`` only).
    seed:
        Seed for the Gibbs sampler.
    """

    name: ClassVar[str] = MODEL_NAME
    layer: ClassVar[int] = 4
    refit_frequency: ClassVar[RefitFrequency] = "weekly"

    def __init__(
        self,
        ticker: str | None = None,
        *,
        market_ticker: str = _DEFAULT_MARKET,
        universe: Sequence[str] | None = None,
        mode: ModelMode = "kalman_beta",
        period: str = _DEFAULT_PERIOD,
        interval: str = _DEFAULT_INTERVAL,
        fix_hyperparameters: tuple[float, float, float] | None = None,
        n_iter: int = 1000,
        n_burn: int = 500,
        n_em_iter: int = 30,
        n_factors: int = 3,
        seed: int = 20260518,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
            )
        if mode in ("kalman_beta", "hierarchical") and not ticker:
            raise ValueError(f"mode={mode!r} requires a primary ticker")
        if mode in ("hierarchical", "dynamic_factor") and (
            universe is None or len(universe) < 2
        ):
            raise ValueError(
                f"mode={mode!r} requires a universe of at least 2 tickers"
            )
        if n_burn < 0 or n_burn >= n_iter:
            raise ValueError("Require 0 <= n_burn < n_iter")
        if n_em_iter < 1:
            raise ValueError("n_em_iter must be >= 1")
        if n_factors < 1:
            raise ValueError("n_factors must be >= 1")

        self.ticker = ticker
        self.market_ticker = market_ticker
        self.universe: tuple[str, ...] = tuple(universe) if universe else ()
        self.mode: ModelMode = mode
        self.period = period
        self.interval = interval
        self.fix_hyperparameters = fix_hyperparameters
        self.n_iter = n_iter
        self.n_burn = n_burn
        self.n_em_iter = n_em_iter
        self.n_factors = n_factors
        self.seed = seed
        self._fit: KalmanFit | HierarchicalFit | DynamicFactorFit | None = None

    # ----- BaseModel hooks ---------------------------------------------------

    def fetch_data(self, provider: DataProvider) -> BayesianKalmanInputs:
        ts = datetime.now(UTC)
        if self.mode == "kalman_beta":
            assert self.ticker is not None
            asset = _fetch_log_returns(provider, self.ticker, self.period, self.interval)
            market = _fetch_log_returns(
                provider, self.market_ticker, self.period, self.interval
            )
            return BayesianKalmanInputs(
                mode="kalman_beta",
                timestamp=ts,
                asset_returns=asset,
                market_returns=market,
                tickers=(self.ticker, self.market_ticker),
                metadata={"period": self.period, "interval": self.interval},
            )

        if self.mode == "hierarchical":
            assert self.ticker is not None
            tickers = tuple(dict.fromkeys((self.ticker, *self.universe)))
            market = _fetch_log_returns(
                provider, self.market_ticker, self.period, self.interval
            )
            panel_design: dict[str, np.ndarray] = {}
            panel_targets: dict[str, np.ndarray] = {}
            kept: list[str] = []
            for t in tickers:
                asset = _fetch_log_returns(provider, t, self.period, self.interval)
                joined = pd.concat([asset, market], axis=1, join="inner").dropna()
                if joined.shape[0] < 30:
                    continue
                y = joined.iloc[:, 0].to_numpy(dtype=float)
                X = np.column_stack(
                    [np.ones(joined.shape[0]), joined.iloc[:, 1].to_numpy(dtype=float)]
                )
                panel_design[t] = X
                panel_targets[t] = y
                kept.append(t)
            if not kept:
                raise RuntimeError(
                    "hierarchical fetch_data: no asset had enough aligned data"
                )
            return BayesianKalmanInputs(
                mode="hierarchical",
                timestamp=ts,
                panel_design=panel_design,
                panel_targets=panel_targets,
                coef_names=("intercept", "market"),
                tickers=tuple(kept),
                metadata={"period": self.period, "interval": self.interval},
            )

        # dynamic_factor
        tickers = tuple(self.universe)
        series_by_ticker: dict[str, pd.Series] = {}
        for t in tickers:
            series = _fetch_log_returns(provider, t, self.period, self.interval)
            if not series.empty:
                series_by_ticker[t] = series
        if not series_by_ticker:
            raise RuntimeError("dynamic_factor fetch_data: no usable tickers")
        panel = pd.DataFrame(series_by_ticker).dropna(how="any")
        if panel.empty:
            raise RuntimeError(
                "dynamic_factor fetch_data: no overlapping dates across the universe"
            )
        return BayesianKalmanInputs(
            mode="dynamic_factor",
            timestamp=ts,
            panel_returns=panel,
            tickers=tuple(str(c) for c in panel.columns),
            metadata={"period": self.period, "interval": self.interval},
        )

    def calibrate(self, data: Any) -> CalibrationResult:
        inputs = self._require_inputs(data)
        result = _calibrate_impl(
            inputs=inputs,
            n_iter=self.n_iter,
            n_burn=self.n_burn,
            n_em_iter=self.n_em_iter,
            n_factors=self.n_factors,
            fix_hyperparameters=self.fix_hyperparameters,
            seed=self.seed,
            timestamp=inputs.timestamp,
        )
        if self.mode == "kalman_beta":
            self._fit = cast(KalmanFit, result.parameters["kalman_fit"])
        elif self.mode == "hierarchical":
            self._fit = cast(HierarchicalFit, result.parameters["hierarchical_fit"])
        else:
            self._fit = cast(DynamicFactorFit, result.parameters["dynamic_factor_fit"])
        return result

    def predict(self, data: Any) -> Signal | Forecast | RiskMetric:
        inputs = self._require_inputs(data)
        if self.mode == "kalman_beta":
            return self._predict_kalman(inputs)
        if self.mode == "hierarchical":
            return self._predict_hierarchical(inputs)
        return self._predict_dynamic_factor(inputs)

    def validate(self, data: Any) -> dict[str, Any]:
        inputs = self._require_inputs(data)
        if self.mode == "kalman_beta":
            return self._validate_kalman(inputs)
        if self.mode == "hierarchical":
            return self._validate_hierarchical()
        return self._validate_dynamic_factor(inputs)

    # ----- Public conveniences -----------------------------------------------

    @property
    def fit(self) -> KalmanFit | HierarchicalFit | DynamicFactorFit:
        return self._require_fit()

    # ----- Internal helpers --------------------------------------------------

    def _require_fit(self) -> KalmanFit | HierarchicalFit | DynamicFactorFit:
        if self._fit is None:
            raise RuntimeError(
                "BayesianKalman has not been calibrated. Call calibrate(data) first."
            )
        return self._fit

    @staticmethod
    def _require_inputs(data: Any) -> BayesianKalmanInputs:
        if not isinstance(data, BayesianKalmanInputs):
            raise TypeError(
                f"BayesianKalman expects BayesianKalmanInputs, got "
                f"{type(data).__name__}"
            )
        return data

    def _predict_kalman(self, inputs: BayesianKalmanInputs) -> Forecast:
        fit = cast(KalmanFit, self._require_fit())
        x_T = fit.latest_state()
        P_T = fit.latest_covariance()
        beta = float(x_T[1])
        sd_beta = float(np.sqrt(max(P_T[1, 1], 0.0)))
        return Forecast(
            ticker=self.ticker or "",
            horizon="1d",
            value=beta,
            timestamp=inputs.timestamp,
            lower=beta - sd_beta,
            upper=beta + sd_beta,
            metadata={
                "alpha": float(x_T[0]),
                "alpha_sd": float(np.sqrt(max(P_T[0, 0], 0.0))),
                "beta_sd": sd_beta,
                "log_likelihood": float(fit.log_likelihood),
                "q_alpha": float(fit.Q[0, 0]),
                "q_beta": float(fit.Q[1, 1]),
                "R": float(fit.R),
                "n_obs": float(fit.n_obs),
                "market_ticker": self.market_ticker,
            },
        )

    def _predict_hierarchical(self, inputs: BayesianKalmanInputs) -> Forecast:
        fit = cast(HierarchicalFit, self._require_fit())
        target = self.ticker
        assert target is not None
        if target not in fit.tickers:
            raise ValueError(
                f"ticker {target!r} not in hierarchical fit universe {fit.tickers!r}"
            )
        i = fit.tickers.index(target)
        # Predicted next-period return given the *last* design row (most
        # recent market return), using the posterior mean theta_i.
        assert inputs.panel_design is not None
        X = inputs.panel_design[target]
        last_row = X[-1]
        theta = fit.theta_means[i]
        sigma_sq = float(fit.sigma_sq_means[i])
        mean = float(last_row @ theta)
        # Predictive variance under posterior mean + observation noise.
        V = fit.theta_covs[i]
        pred_var = float(last_row @ V @ last_row + sigma_sq)
        sd = float(np.sqrt(max(pred_var, 0.0)))
        return Forecast(
            ticker=target,
            horizon="1d",
            value=mean,
            timestamp=inputs.timestamp,
            lower=mean - sd,
            upper=mean + sd,
            metadata={
                "posterior_theta": theta.tolist(),
                "posterior_sigma_sq": sigma_sq,
                "population_mu": fit.mu_mean.tolist(),
                "shrinkage_to_pop_mean": float(np.linalg.norm(theta - fit.mu_mean)),
                "n_assets_in_fit": float(fit.n_assets),
            },
        )

    def _predict_dynamic_factor(self, inputs: BayesianKalmanInputs) -> RiskMetric:
        fit = cast(DynamicFactorFit, self._require_fit())
        # Risk metric: average idiosyncratic + systematic vol contribution.
        # Total variance per asset from the DFM: Lambda_i Sigma_f Lambda_i^T + Psi_i
        # We sit Sigma_f at the stationary AR(1) covariance solving
        # Sigma_f = Phi Sigma_f Phi^T + Q (vec-form).
        Sigma_f = _stationary_state_cov(fit.Phi, fit.Q)
        systematic = np.einsum("ik,kl,il->i", fit.Lambda, Sigma_f, fit.Lambda)
        total = systematic + fit.Psi
        mean_total_vol = float(np.sqrt(np.mean(total)))
        return RiskMetric(
            ticker="PORTFOLIO",
            metric_name="dynamic_factor_vol",
            value=mean_total_vol,
            timestamp=inputs.timestamp,
            confidence_level=None,
            horizon="1d",
            metadata={
                "n_factors": float(fit.n_factors),
                "n_assets": float(fit.n_assets),
                "log_likelihood": float(fit.log_likelihood),
                "iterations": float(fit.n_iter),
                "mean_systematic_var": float(np.mean(systematic)),
                "mean_specific_var": float(np.mean(fit.Psi)),
                "tickers": list(fit.tickers),
            },
        )

    def _validate_kalman(self, inputs: BayesianKalmanInputs) -> dict[str, Any]:
        fit = cast(KalmanFit, self._require_fit())
        run = KalmanRun(
            states=fit.states,
            covariances=fit.covariances,
            innovations=fit.innovations,
            innovation_variances=fit.innovation_variances,
            log_likelihood=fit.log_likelihood,
        )
        std_innov = standardized_innovations(run)
        lb_pvalue = ljung_box_pvalue(std_innov, max_lag=10)
        kurt = innovation_normality_kurtosis(std_innov)
        beta_path = fit.states[:, 1]
        out: dict[str, Any] = {
            "n_obs": int(fit.n_obs),
            "log_likelihood": float(fit.log_likelihood),
            "mean_standardized_innovation": float(std_innov.mean()),
            "std_standardized_innovation": float(std_innov.std(ddof=1)),
            "ljung_box_pvalue": lb_pvalue,
            "innovation_excess_kurtosis": kurt,
            "beta_min": float(beta_path.min()),
            "beta_max": float(beta_path.max()),
            "beta_latest": float(beta_path[-1]),
            "beta_path_std": float(beta_path.std(ddof=1)),
            "q_alpha": float(fit.Q[0, 0]),
            "q_beta": float(fit.Q[1, 1]),
            "R": float(fit.R),
        }
        return out

    def _validate_hierarchical(self) -> dict[str, Any]:
        fit = cast(HierarchicalFit, self._require_fit())
        # Shrinkage diagnostics: distance of each theta_i from the pop mean
        # vs the spread implied by Sigma_mean.
        diffs = fit.theta_means - fit.mu_mean
        avg_shrinkage = float(np.mean(np.linalg.norm(diffs, axis=1)))
        # Per-asset uncertainty: trace of theta_covs.
        avg_theta_var = float(np.mean([float(np.trace(c)) for c in fit.theta_covs]))
        return {
            "n_assets": int(fit.n_assets),
            "n_coefs": int(fit.n_coefs),
            "n_iter": int(fit.n_iter),
            "n_burn": int(fit.n_burn),
            "population_mu_norm": float(np.linalg.norm(fit.mu_mean)),
            "average_theta_shrinkage_distance": avg_shrinkage,
            "average_theta_posterior_variance": avg_theta_var,
            "mean_sigma_sq": float(fit.sigma_sq_means.mean()),
            "max_sigma_sq": float(fit.sigma_sq_means.max()),
            "Sigma_trace": float(np.trace(fit.Sigma_mean)),
        }

    def _validate_dynamic_factor(self, inputs: BayesianKalmanInputs) -> dict[str, Any]:
        fit = cast(DynamicFactorFit, self._require_fit())
        panel = inputs.panel_returns
        assert panel is not None
        Y = panel.to_numpy(dtype=float)
        Y_c = Y - Y.mean(axis=0)
        fitted = fit.factors @ fit.Lambda.T
        resid = Y_c - fitted
        total_var = Y_c.var(axis=0, ddof=1)
        total_var = np.where(total_var > 0, total_var, 1.0)
        r2 = 1.0 - resid.var(axis=0, ddof=1) / total_var
        r2 = np.clip(r2, 0.0, 1.0)
        return {
            "n_assets": int(fit.n_assets),
            "n_factors": int(fit.n_factors),
            "n_iter": int(fit.n_iter),
            "log_likelihood": float(fit.log_likelihood),
            "phi_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(fit.Phi)))),
            "mean_r_squared": float(np.mean(r2)),
            "min_r_squared": float(np.min(r2)),
            "max_r_squared": float(np.max(r2)),
            "mean_psi": float(np.mean(fit.Psi)),
            "max_psi": float(np.max(fit.Psi)),
        }


# ---- module-level helpers ---------------------------------------------------


def _stationary_state_cov(Phi: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Discrete-time Lyapunov stationary covariance solving `S = Phi S Phi^T + Q`.

    Solved via the vec identity: ``(I - Phi (x) Phi) vec(S) = vec(Q)``. If
    ``Phi`` has a unit eigenvalue the equation is singular; in that fallback
    we return ``Q`` (a conservative under-estimate).
    """

    r = Phi.shape[0]
    A = np.eye(r * r) - np.kron(Phi, Phi)
    try:
        vec_S = np.linalg.solve(A, Q.reshape(-1))
        S = vec_S.reshape(r, r)
        S = 0.5 * (S + S.T)
        # Ensure PSD.
        w, V = np.linalg.eigh(S)
        if np.any(w < 0):
            w = np.clip(w, 1e-12, None)
            S = (V * w) @ V.T
        return cast(np.ndarray, S)
    except np.linalg.LinAlgError:
        return Q.copy()


def _fetch_log_returns(
    provider: DataProvider, ticker: str, period: str, interval: str
) -> pd.Series:
    """Pull bars from ``provider`` and convert to a log-return Series."""

    bars = provider.fetch_prices(ticker, period, interval)
    if bars is None or bars.empty or "Close" not in bars.columns:
        return pd.Series(dtype="float64", name=ticker)
    close_arr = bars["Close"].astype(float).to_numpy()
    if "Date" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Date"]))
    elif "Datetime" in bars.columns:
        index = pd.DatetimeIndex(pd.to_datetime(bars["Datetime"]))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(bars.index))
    close = pd.Series(close_arr, index=index).sort_index()
    log_r = pd.Series(np.log(close.to_numpy()), index=close.index).diff().dropna()
    log_r.name = ticker
    return log_r


__all__ = ["BayesianKalman"]
