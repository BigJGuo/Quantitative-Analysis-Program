"""End-to-end tests for the `CointegrationPairs` orchestration class.

Uses an `InMemoryProvider` populated with synthetic price series so the
`fetch -> calibrate -> predict -> validate` flow runs without network access.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, Signal
from src.models.cointegration_pairs import CointegrationPairs
from src.models.cointegration_pairs.types import PairInputs


def _series_to_bars(close: pd.Series) -> pd.DataFrame:
    """yfinance-shaped OHLCV frame around an arbitrary `close` series."""

    values = np.asarray(close.values, dtype=float)
    return pd.DataFrame(
        {
            "Date": close.index,
            "Open": values,
            "High": values * 1.001,
            "Low": values * 0.999,
            "Close": values,
            "Volume": np.full(len(close), 1_000_000, dtype=int),
        }
    )


def _synthetic_pair_prices(
    seed: int = 20260518,
    n_obs: int = 800,
    alpha: float = 0.20,
    beta: float = 1.10,
    phi: float = 0.85,
) -> tuple[pd.Series, pd.Series]:
    """Generate a cointegrated pair as raw (not log) prices.

    Same recipe as the unit-test conftest fixture (random-walk leg B + AR(1)
    spread); kept inline here so the integration-style provider fixtures don't
    leak conftest state across files.
    """

    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0.0, 0.012, size=n_obs)) + math.log(50.0)
    z = np.empty(n_obs)
    eps = rng.normal(0.0, 0.02, size=n_obs)
    for t in range(1, n_obs):
        z[t] = phi * z[t - 1] + eps[t]
    log_a = alpha + beta * log_b + z
    idx = pd.date_range("2023-01-02", periods=n_obs, freq="B")
    return pd.Series(np.exp(log_a), index=idx), pd.Series(np.exp(log_b), index=idx)


@pytest.fixture
def cointegrated_provider() -> InMemoryProvider:
    a, b = _synthetic_pair_prices()
    return InMemoryProvider(
        prices={
            ("AAA", "2y", "1d"): _series_to_bars(a),
            ("BBB", "2y", "1d"): _series_to_bars(b),
        }
    )


@pytest.fixture
def independent_provider() -> InMemoryProvider:
    rng = np.random.default_rng(11)
    n_obs = 500
    a = pd.Series(
        np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n_obs)) + math.log(60.0)),
        index=pd.date_range("2023-01-02", periods=n_obs, freq="B"),
    )
    b = pd.Series(
        np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n_obs)) + math.log(120.0)),
        index=pd.date_range("2023-01-02", periods=n_obs, freq="B"),
    )
    return InMemoryProvider(
        prices={
            ("XXX", "2y", "1d"): _series_to_bars(a),
            ("YYY", "2y", "1d"): _series_to_bars(b),
        }
    )


class TestRegistration:
    def test_registered_in_global_registry(self) -> None:
        assert get_model("cointegration_pairs") is CointegrationPairs

    def test_base_model_class_attrs(self) -> None:
        assert CointegrationPairs.name == "cointegration_pairs"
        assert CointegrationPairs.layer == 4
        assert CointegrationPairs.refit_frequency == "weekly"


class TestConstructor:
    def test_rejects_same_ticker(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            CointegrationPairs("KO", "KO")

    def test_rejects_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            CointegrationPairs("A", "B", method="quadratic")  # type: ignore[arg-type]

    def test_rejects_tiny_window(self) -> None:
        with pytest.raises(ValueError, match="rolling_window"):
            CointegrationPairs("A", "B", rolling_window=5)


class TestFetchData:
    def test_returns_pair_inputs(self, cointegrated_provider: InMemoryProvider) -> None:
        model = CointegrationPairs("AAA", "BBB")
        data = model.fetch_data(cointegrated_provider)
        assert isinstance(data, PairInputs)
        assert data.ticker_a == "AAA"
        assert data.ticker_b == "BBB"
        assert data.log_price_a.index.equals(data.log_price_b.index)
        assert len(data.log_price_a) > 100


class TestCalibratePredict:
    def test_calibrate_accepts_cointegrated_pair(
        self, cointegrated_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("AAA", "BBB")
        data = model.fetch_data(cointegrated_provider)
        result = model.calibrate(data)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "cointegration_pairs"
        assert model.fit.is_cointegrated
        assert model.fit.method == "static"

    def test_predict_returns_signal(
        self, cointegrated_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("AAA", "BBB")
        data = model.fetch_data(cointegrated_provider)
        model.calibrate(data)
        signal = model.predict(data)
        assert isinstance(signal, Signal)
        assert signal.ticker == "AAA/BBB"
        assert signal.direction in ("long", "short", "flat")
        assert 0.0 <= signal.strength <= 1.0
        assert "beta" in signal.metadata
        assert "current_z_score" in signal.metadata

    def test_predict_flat_when_rejected(
        self, independent_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("XXX", "YYY", significance="5%")
        data = model.fetch_data(independent_provider)
        model.calibrate(data)
        signal = model.predict(data)
        assert signal.direction == "flat"
        assert signal.strength == 0.0
        assert signal.metadata["status"] == "rejected"

    def test_predict_without_calibrate_raises(
        self, cointegrated_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("AAA", "BBB")
        data = model.fetch_data(cointegrated_provider)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            model.predict(data)

    def test_kalman_method_predicts(
        self, cointegrated_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("AAA", "BBB", method="kalman")
        data = model.fetch_data(cointegrated_provider)
        model.calibrate(data)
        signal = model.predict(data)
        assert isinstance(signal, Signal)
        assert signal.metadata["method"] == "kalman"


class TestValidate:
    def test_emits_diagnostics(
        self, cointegrated_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("AAA", "BBB", method="kalman")
        data = model.fetch_data(cointegrated_provider)
        model.calibrate(data)
        diag = model.validate(data)
        for key in (
            "status",
            "method",
            "alpha",
            "beta",
            "adf_leg_a_tstat",
            "adf_leg_b_tstat",
            "eg_residual_tstat",
            "ou_phi",
            "ou_half_life",
            "realized_half_life",
        ):
            assert key in diag
        # Kalman diagnostics only emitted when the pair was accepted (and
        # thus a KalmanFit was actually built during calibrate).
        if diag["status"] == "accepted":
            assert "kalman_innovation_ljung_box_p" in diag
            assert "kalman_beta_range" in diag


class TestConveniences:
    def test_spread_aligned_to_input(
        self, cointegrated_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("AAA", "BBB")
        data = model.fetch_data(cointegrated_provider)
        model.calibrate(data)
        spread = model.spread(data)
        assert spread.index.equals(data.log_price_a.index)

    def test_position_path_in_neg1_0_pos1(
        self, cointegrated_provider: InMemoryProvider
    ) -> None:
        model = CointegrationPairs("AAA", "BBB")
        data = model.fetch_data(cointegrated_provider)
        model.calibrate(data)
        positions = model.position_path(data)
        assert set(np.unique(positions.to_numpy())) <= {-1, 0, 1}
