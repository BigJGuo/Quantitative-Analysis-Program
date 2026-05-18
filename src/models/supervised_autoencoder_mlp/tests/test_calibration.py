"""Calibration tests. Skipped automatically when torch isn't installed."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.core.types import CalibrationResult  # noqa: E402
from src.models.supervised_autoencoder_mlp.calibration import calibrate  # noqa: E402
from src.models.supervised_autoencoder_mlp.types import (  # noqa: E402
    DEFAULT_TARGETS,
    EnsembleFit,
    SAEConfig,
    SAEInputs,
    TrainConfig,
)


def _inputs_from_arrays(
    features: pd.DataFrame, targets: pd.DataFrame, dates: np.ndarray
) -> SAEInputs:
    return SAEInputs(
        features=features,
        targets=targets,
        sample_weights=None,
        dates=dates,
        ticker_ids=np.zeros(len(features), dtype=int),
        feature_names=tuple(str(c) for c in features.columns),
        target_names=DEFAULT_TARGETS,
        timestamp=datetime.now(UTC),
    )


class TestCalibrate:
    def test_round_trip_returns_calibration_result(
        self, synthetic_feature_panel: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    ) -> None:
        features, targets, dates = synthetic_feature_panel
        inputs = _inputs_from_arrays(features, targets, dates)
        result = calibrate(
            inputs,
            config=SAEConfig(n_features=features.shape[1], bottleneck_dim=4, encoder_hidden=(8,)),
            train_cfg=TrainConfig(epochs=2, batch_size=32, n_folds=3, seeds=(0,)),
        )
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "supervised_autoencoder_mlp"
        fit = result.parameters["ensemble_fit"]
        assert isinstance(fit, EnsembleFit)
        assert fit.n_members == 3  # 3 folds * 1 seed

    def test_auc_above_random_on_learnable_signal(
        self, synthetic_feature_panel: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    ) -> None:
        """Target 0 is a near-linear function of features 0+1 — the network
        should easily clear 0.55 AUC on it after a handful of epochs."""

        features, targets, dates = synthetic_feature_panel
        inputs = _inputs_from_arrays(features, targets, dates)
        result = calibrate(
            inputs,
            config=SAEConfig(n_features=features.shape[1], bottleneck_dim=4, encoder_hidden=(8,)),
            train_cfg=TrainConfig(epochs=20, batch_size=32, n_folds=3, seeds=(0,)),
        )
        # Average AUC on the learnable target across folds.
        aucs = [m.val_auc[0] for m in result.parameters["ensemble_fit"].members]
        assert np.mean(aucs) > 0.55

    def test_rejects_mismatched_features(
        self, synthetic_feature_panel: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    ) -> None:
        features, targets, dates = synthetic_feature_panel
        inputs = _inputs_from_arrays(features, targets, dates)
        bad_config = SAEConfig(n_features=999)
        with pytest.raises(ValueError, match="n_features"):
            calibrate(inputs, config=bad_config, train_cfg=TrainConfig(epochs=1))

    def test_multi_seed_ensemble(
        self, synthetic_feature_panel: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    ) -> None:
        features, targets, dates = synthetic_feature_panel
        inputs = _inputs_from_arrays(features, targets, dates)
        result = calibrate(
            inputs,
            config=SAEConfig(n_features=features.shape[1], bottleneck_dim=4, encoder_hidden=(8,)),
            train_cfg=TrainConfig(epochs=2, batch_size=32, n_folds=2, seeds=(0, 1)),
        )
        fit = result.parameters["ensemble_fit"]
        assert fit.n_members == 4  # 2 folds * 2 seeds
        # Members from different seeds should produce different state-dicts.
        sd0 = list(fit.members[0].state_dict.values())[0].numpy()
        sd1 = list(fit.members[1].state_dict.values())[0].numpy()
        assert not np.allclose(sd0, sd1)
