"""End-to-end integration test for the supervised autoencoder + MLP.

Hits yfinance via `YFinanceProvider` for a small universe and runs the full
`fetch_data -> calibrate -> predict -> validate` pipeline. Skipped if either
torch or yfinance is missing, or the network fetch fails — keeps offline
CI green.

We intentionally use a 3-ticker universe and a 1-epoch / 2-fold / 1-seed
training run so the test completes in under a minute on CPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yfinance")

from src.core.data_provider import YFinanceProvider  # noqa: E402
from src.core.types import CalibrationResult, Signal  # noqa: E402
from src.models.supervised_autoencoder_mlp import (  # noqa: E402
    SAEConfig,
    SAEInputs,
    SupervisedAEMLP,
    TrainConfig,
)

_UNIVERSE = ["SPY", "QQQ", "IWM"]


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model() -> SupervisedAEMLP:
    return SupervisedAEMLP(
        universe=_UNIVERSE,
        period="1y",
        config=SAEConfig(
            bottleneck_dim=4,
            encoder_hidden=(8,),
            decoder_hidden=(8,),
            mlp_hidden=(16,),
            dropout_encoder=0.0,
            dropout_mlp=0.0,
            noise_sigma=0.02,
        ),
        train_cfg=TrainConfig(epochs=2, batch_size=128, n_folds=2, seeds=(0,)),
    )


def _fetch_or_skip(model: SupervisedAEMLP, provider: YFinanceProvider) -> SAEInputs:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_pipeline_against_yfinance(
    provider: YFinanceProvider, model: SupervisedAEMLP
) -> None:
    data = _fetch_or_skip(model, provider)
    assert isinstance(data, SAEInputs)
    assert data.features.shape[0] > 50
    assert data.targets.shape[1] == 3

    result = model.calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "supervised_autoencoder_mlp"
    # 2 folds * 1 seed = 2 members.
    assert model.fit.n_members == 2

    sig = model.predict(data)
    assert isinstance(sig, Signal)
    assert sig.direction in {"long", "flat"}
    assert 0.0 <= sig.strength <= 1.0
    assert "probabilities" in sig.metadata
    assert len(sig.metadata["probabilities"]) == 3


@pytest.mark.integration
def test_validation_diagnostics(
    provider: YFinanceProvider, model: SupervisedAEMLP
) -> None:
    """Spec validation section: every diagnostic must be present and finite."""

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    diag = model.validate(data)
    assert diag["n_rows"] > 0
    assert diag["recon_error_mean"] >= 0.0
    for target_name in ("pos_1d", "pos_med_5d", "pos_sum_5d"):
        assert 0.0 <= diag["brier_per_target"][target_name] <= 1.0
        assert 0.0 <= diag["auc_per_target"][target_name] <= 1.0
