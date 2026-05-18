"""End-to-end tests for the `FactorModelPCA` orchestration class.

Uses an `InMemoryProvider` populated with synthetic price series so the
fetch -> calibrate -> predict -> validate flow runs without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.factor_models_pca import FactorModelPCA
from src.models.factor_models_pca.types import FactorInputs


def _make_synthetic_prices(
    *,
    tickers: list[str],
    n_obs: int = 300,
    seed: int = 17,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Build a per-ticker yfinance-shaped DataFrame from a 1-factor process."""

    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.012, size=n_obs)
    betas = rng.uniform(0.7, 1.3, size=len(tickers))
    spec = rng.normal(0.0, 0.003, size=(n_obs, len(tickers)))
    log_rets = factor[:, None] * betas + spec
    # Build price paths from log returns starting at $100.
    log_prices = np.log(100.0) + log_rets.cumsum(axis=0)
    prices = np.exp(log_prices)
    dates = pd.date_range("2024-01-02", periods=n_obs, freq="B")

    out: dict[tuple[str, str, str], pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices[:, i],
                "High": prices[:, i] * 1.002,
                "Low": prices[:, i] * 0.998,
                "Close": prices[:, i],
                "Volume": np.full(n_obs, 1_000_000, dtype=int),
            }
        )
        # The model uses _period_from_lookback to pick a yfinance period
        # string. With lookback=252 days (252+30 <= 365) this resolves to "1y".
        out[(ticker, "1y", "1d")] = df
    return out


@pytest.fixture
def universe() -> list[str]:
    return ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


@pytest.fixture
def provider(universe: list[str]) -> InMemoryProvider:
    prices = _make_synthetic_prices(tickers=universe, n_obs=300)
    return InMemoryProvider(prices=prices)


class TestFactorModelPCAClass:
    def test_registered_in_global_registry(self) -> None:
        assert get_model("factor_models_pca") is FactorModelPCA

    def test_base_model_class_attrs(self) -> None:
        assert FactorModelPCA.name == "factor_models_pca"
        assert FactorModelPCA.layer == 2
        assert FactorModelPCA.refit_frequency == "weekly"

    def test_constructor_rejects_small_universe(self) -> None:
        with pytest.raises(ValueError, match="at least 2 tickers"):
            FactorModelPCA(universe=["AAA"])

    def test_constructor_rejects_bad_K(self) -> None:
        with pytest.raises(ValueError, match="K must be a positive"):
            FactorModelPCA(universe=["A", "B"], K=0)


class TestFetchData:
    def test_returns_factor_inputs(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        assert isinstance(data, FactorInputs)
        assert set(data.tickers) <= set(universe)
        assert data.returns.shape[1] == len(data.tickers)
        assert data.returns.shape[0] > 100

    def test_kept_tickers_preserve_universe_order(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        # All synthetic series come back clean, so every ticker should survive.
        assert list(data.tickers) == universe


class TestCalibrate:
    def test_round_trip_calibrate_predict(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        result = model.calibrate(data)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "factor_models_pca"
        assert model.fit.n_factors == 2

    def test_predict_returns_portfolio_risk(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        model.calibrate(data)
        risk = model.predict(data)
        assert isinstance(risk, RiskMetric)
        assert risk.ticker == "PORTFOLIO"
        assert risk.metric_name == "portfolio_volatility"
        assert risk.value > 0.0
        for key in ("systematic_vol", "specific_vol", "factor_exposures", "n_factors"):
            assert key in risk.metadata

    def test_predict_without_calibrate_raises(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            model.predict(data)


class TestValidate:
    def test_validate_emits_diagnostics(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        model.calibrate(data)
        diag = model.validate(data)
        for key in (
            "n_assets",
            "n_factors",
            "method",
            "variance_explained",
            "mean_specific_vol",
            "max_residual_abs_autocorrelation_lag1",
            "max_residual_cross_correlation",
            "top_eigenvalue",
            "cumulative_variance_explained",
            "mp_upper_edge",
        ):
            assert key in diag

    def test_variance_explained_above_threshold_for_synthetic_one_factor(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=1, lookback_days=252)
        data = model.fetch_data(provider)
        model.calibrate(data)
        diag = model.validate(data)
        # Synthetic process is single-factor; PC1 should dominate.
        assert diag["variance_explained"] > 0.8


class TestPublicConveniences:
    def test_covariance_is_psd(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        model.calibrate(data)
        sigma = model.covariance()
        # Symmetric.
        assert np.allclose(sigma, sigma.T)
        # PSD: smallest eigenvalue >= 0 (allow small numerical drift).
        eigvals = np.linalg.eigvalsh(sigma)
        assert eigvals.min() > -1e-10

    def test_residualize_signal_strips_factor_loading(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        model.calibrate(data)
        raw = np.linspace(-1.0, 1.0, len(data.tickers))
        neutral = model.residualize_signal(raw)
        proj = model.fit.B.T @ neutral
        assert np.allclose(proj, 0.0, atol=1e-9)

    def test_portfolio_risk_alias(
        self, universe: list[str], provider: InMemoryProvider
    ) -> None:
        model = FactorModelPCA(universe=universe, K=2, lookback_days=252)
        data = model.fetch_data(provider)
        model.calibrate(data)
        w = np.ones(model.fit.n_assets) / model.fit.n_assets
        pr = model.portfolio_risk(w)
        assert pr.total_vol > 0
