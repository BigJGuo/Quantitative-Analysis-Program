"""End-to-end tests for the `SupervisedAEMLP` orchestration class.

Uses an `InMemoryProvider` populated with synthetic bars so the fetch ->
calibrate -> predict -> validate flow runs without network access. Skips
when torch is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.core.data_provider import InMemoryProvider  # noqa: E402
from src.core.registry import get_model  # noqa: E402
from src.core.types import CalibrationResult, Signal  # noqa: E402
from src.models.supervised_autoencoder_mlp import (  # noqa: E402
    SAEConfig,
    SupervisedAEMLP,
    TrainConfig,
)
from src.models.supervised_autoencoder_mlp.types import SAEInputs  # noqa: E402


def _make_synthetic_bars(
    ticker: str, n: int = 250, seed: int = 11
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.012, size=n)
    log_prices = np.log(100.0) + log_rets.cumsum()
    prices = np.exp(log_prices)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": prices,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Adj Close": prices,
            "Volume": rng.integers(500_000, 2_000_000, size=n),
        }
    )


@pytest.fixture
def universe() -> list[str]:
    return ["AAA", "BBB", "CCC"]


@pytest.fixture
def provider(universe: list[str]) -> InMemoryProvider:
    prices = {(t, "2y", "1d"): _make_synthetic_bars(t, seed=i + 1) for i, t in enumerate(universe)}
    return InMemoryProvider(prices=prices)


@pytest.fixture
def model(universe: list[str]) -> SupervisedAEMLP:
    return SupervisedAEMLP(
        universe=universe,
        config=SAEConfig(bottleneck_dim=4, encoder_hidden=(8,), decoder_hidden=(8,), mlp_hidden=(16,)),
        train_cfg=TrainConfig(epochs=2, batch_size=64, n_folds=2, seeds=(0,)),
    )


class TestRegistration:
    def test_registered(self) -> None:
        assert get_model("supervised_autoencoder_mlp") is SupervisedAEMLP

    def test_class_attrs(self) -> None:
        assert SupervisedAEMLP.name == "supervised_autoencoder_mlp"
        assert SupervisedAEMLP.layer == 4
        assert SupervisedAEMLP.refit_frequency == "weekly"


class TestConstructor:
    def test_rejects_empty_universe(self) -> None:
        with pytest.raises(ValueError, match="at least 1 ticker"):
            SupervisedAEMLP(universe=[])

    def test_rejects_bad_horizons(self) -> None:
        with pytest.raises(ValueError, match="horizons"):
            SupervisedAEMLP(universe=["A"], horizons=(0,))


class TestFetchData:
    def test_returns_sae_inputs(
        self, model: SupervisedAEMLP, provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(provider)
        assert isinstance(data, SAEInputs)
        assert data.features.shape[0] > 50
        assert data.targets.shape == (data.features.shape[0], 3)


class TestEndToEnd:
    def test_calibrate_predict_validate(
        self, model: SupervisedAEMLP, provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(provider)
        result = model.calibrate(data)
        assert isinstance(result, CalibrationResult)
        sig = model.predict(data)
        assert isinstance(sig, Signal)
        assert sig.direction in {"long", "flat"}
        assert 0.0 <= sig.strength <= 1.0
        diag = model.validate(data)
        for key in (
            "n_rows",
            "n_members",
            "recon_error_mean",
            "brier_per_target",
            "auc_per_target",
            "mean_ensemble_seed_correlation",
        ):
            assert key in diag

    def test_predict_without_calibrate_raises(
        self, model: SupervisedAEMLP, provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(provider)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            model.predict(data)
