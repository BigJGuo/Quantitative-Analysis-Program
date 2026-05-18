"""Unit tests for the model-internal dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.supervised_autoencoder_mlp.types import (
    DEFAULT_TARGETS,
    EnsembleFit,
    SAEConfig,
    SAEFit,
    SAEInputs,
    TrainConfig,
)


class TestSAEConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = SAEConfig(n_features=10)
        assert cfg.bottleneck_dim == 64
        assert cfg.n_targets == 3
        assert cfg.noise_sigma == 0.03

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"n_features": -1}, "n_features"),
            ({"n_targets": 0}, "n_targets"),
            ({"bottleneck_dim": 0}, "bottleneck_dim"),
            ({"dropout_encoder": 1.0}, "dropout_encoder"),
            ({"dropout_mlp": -0.1}, "dropout_mlp"),
            ({"noise_sigma": -0.01}, "noise_sigma"),
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SAEConfig(**kwargs)


class TestTrainConfig:
    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.epochs >= 1
        assert cfg.n_folds >= 2
        assert 0.0 < cfg.decision_threshold < 1.0

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"epochs": 0}, "epochs"),
            ({"batch_size": 0}, "batch_size"),
            ({"lr": 0}, "lr"),
            ({"lr_min": -1.0}, "lr_min"),
            ({"weight_decay": -1.0}, "weight_decay"),
            ({"alpha_aux": -0.1}, "alpha_aux"),
            ({"seeds": ()}, "seeds"),
            ({"n_folds": 1}, "n_folds"),
            ({"embargo": -1}, "embargo"),
            ({"decision_threshold": 1.5}, "decision_threshold"),
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            TrainConfig(**kwargs)


class TestSAEInputs:
    def _build(self) -> SAEInputs:
        features = pd.DataFrame(np.zeros((5, 3)), columns=["a", "b", "c"])
        targets = pd.DataFrame(np.zeros((5, 3)), columns=list(DEFAULT_TARGETS))
        dates = np.arange(5)
        ticker_ids = np.zeros(5, dtype=int)
        return SAEInputs(
            features=features,
            targets=targets,
            sample_weights=None,
            dates=dates,
            ticker_ids=ticker_ids,
            feature_names=("a", "b", "c"),
        )

    def test_well_formed(self) -> None:
        inputs = self._build()
        assert inputs.features.shape == (5, 3)
        assert inputs.target_names == DEFAULT_TARGETS

    def test_empty_features_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SAEInputs(
                features=pd.DataFrame(),
                targets=pd.DataFrame(),
                sample_weights=None,
                dates=np.array([]),
                ticker_ids=np.array([], dtype=int),
                feature_names=(),
            )

    def test_target_row_mismatch_rejected(self) -> None:
        features = pd.DataFrame(np.zeros((5, 2)), columns=["a", "b"])
        targets = pd.DataFrame(np.zeros((4, 3)), columns=list(DEFAULT_TARGETS))
        with pytest.raises(ValueError, match="targets has"):
            SAEInputs(
                features=features,
                targets=targets,
                sample_weights=None,
                dates=np.arange(5),
                ticker_ids=np.zeros(5, dtype=int),
                feature_names=("a", "b"),
            )


class TestEnsembleFit:
    def test_requires_members(self) -> None:
        with pytest.raises(ValueError, match="members"):
            EnsembleFit(
                members=(),
                config=SAEConfig(n_features=3),
                train=TrainConfig(),
                feature_names=("a", "b", "c"),
                target_names=DEFAULT_TARGETS,
                timestamp=datetime.now(UTC),
            )

    def test_n_members(self) -> None:
        members = (
            SAEFit(
                fold=0,
                seed=0,
                feature_median=np.zeros(3),
                feature_mad=np.ones(3),
                state_dict={"w": "stub"},
                train_loss=0.1,
                val_loss=0.2,
                val_auc=(0.6, 0.6, 0.6),
                n_train=10,
                n_val=5,
            ),
        )
        fit = EnsembleFit(
            members=members,
            config=SAEConfig(n_features=3),
            train=TrainConfig(),
            feature_names=("a", "b", "c"),
            target_names=DEFAULT_TARGETS,
            timestamp=datetime.now(UTC),
        )
        assert fit.n_members == 1
