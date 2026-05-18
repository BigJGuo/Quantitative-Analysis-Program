"""Dataclasses specific to the Supervised Autoencoder + MLP model.

Shared `Signal` / `CalibrationResult` live in `src.core.types`; this module
holds the model-internal shapes that flow between `fetch_data`, `calibrate`,
and `predict`.

A fit is two-layered:

- `SAEFit` holds a single trained model's standardization statistics and the
  in-memory state-dict (kept as an opaque ``Any`` so this module does not need
  to import torch).
- `EnsembleFit` collects K folds * S seeds worth of `SAEFit` records together
  with the median-blend metadata used at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

TargetSpec = tuple[str, ...]
DEFAULT_TARGETS: TargetSpec = ("pos_1d", "pos_med_5d", "pos_sum_5d")


@dataclass(frozen=True)
class SAEConfig:
    """Static architecture hyperparameters for the SAE + MLP.

    Defaults follow the spec's canonical configuration:

    - encoder width 96, two hidden layers, bottleneck 64
    - decoder symmetric to the encoder (untied weights)
    - aux head: linear `bottleneck -> n_targets`
    - main head: 2 hidden layers of width 256 over `[x; h]`
    - SiLU activations + BatchNorm; dropout 0.2 on the encoder, 0.3 on the MLP
    """

    n_features: int = 0
    n_targets: int = 3
    encoder_hidden: tuple[int, ...] = (96, 96)
    bottleneck_dim: int = 64
    decoder_hidden: tuple[int, ...] = (96,)
    mlp_hidden: tuple[int, ...] = (256, 256)
    dropout_encoder: float = 0.2
    dropout_mlp: float = 0.3
    noise_sigma: float = 0.03

    def __post_init__(self) -> None:
        if self.n_features < 0:
            raise ValueError(f"n_features must be >= 0, got {self.n_features}")
        if self.n_targets < 1:
            raise ValueError(f"n_targets must be >= 1, got {self.n_targets}")
        if self.bottleneck_dim < 1:
            raise ValueError(f"bottleneck_dim must be >= 1, got {self.bottleneck_dim}")
        if not 0.0 <= self.dropout_encoder < 1.0:
            raise ValueError(
                f"dropout_encoder must lie in [0, 1), got {self.dropout_encoder}"
            )
        if not 0.0 <= self.dropout_mlp < 1.0:
            raise ValueError(
                f"dropout_mlp must lie in [0, 1), got {self.dropout_mlp}"
            )
        if self.noise_sigma < 0.0:
            raise ValueError(f"noise_sigma must be >= 0, got {self.noise_sigma}")


@dataclass(frozen=True)
class TrainConfig:
    """Optimization / training hyperparameters.

    Defaults follow the spec's "Calibration / training approach" section but
    are intentionally scaled down (fewer epochs, smaller batches) so unit
    tests can run without GPUs. Production callers override `epochs`,
    `batch_size`, `seeds`, and `n_folds`.
    """

    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-5
    alpha_recon: float = 1.0
    alpha_aux: float = 1.0
    alpha_main: float = 1.0
    seeds: tuple[int, ...] = (0,)
    n_folds: int = 3
    embargo: int = 1
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.lr_min < 0 or self.lr_min > self.lr:
            raise ValueError(
                f"lr_min must satisfy 0 <= lr_min <= lr; got {self.lr_min} vs {self.lr}"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        for name, val in (
            ("alpha_recon", self.alpha_recon),
            ("alpha_aux", self.alpha_aux),
            ("alpha_main", self.alpha_main),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if not self.seeds:
            raise ValueError("seeds must contain at least one integer")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {self.embargo}")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError(
                f"decision_threshold must lie in (0, 1), got {self.decision_threshold}"
            )


@dataclass(frozen=True)
class SAEInputs:
    """Inputs consumed by `calibrate`, `predict`, and `validate`.

    `features` is the (N, p) feature panel; `targets` is the (N, K) binary
    target panel; `sample_weights` is a length-N vector of per-row weights
    (uniform = ``None``). `dates` and `ticker_ids` are used by the purged
    K-fold split.
    """

    features: pd.DataFrame
    targets: pd.DataFrame
    sample_weights: np.ndarray | None
    dates: np.ndarray
    ticker_ids: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...] = DEFAULT_TARGETS
    timestamp: datetime = field(default_factory=lambda: datetime.fromtimestamp(0))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_rows = len(self.features)
        if n_rows == 0:
            raise ValueError("SAEInputs.features must be non-empty")
        if self.targets.shape[0] != n_rows:
            raise ValueError(
                f"targets has {self.targets.shape[0]} rows but features has {n_rows}"
            )
        if self.dates.shape != (n_rows,):
            raise ValueError(
                f"dates shape {self.dates.shape} does not match N={n_rows}"
            )
        if self.ticker_ids.shape != (n_rows,):
            raise ValueError(
                f"ticker_ids shape {self.ticker_ids.shape} does not match N={n_rows}"
            )
        if self.sample_weights is not None and self.sample_weights.shape != (n_rows,):
            raise ValueError(
                f"sample_weights shape {self.sample_weights.shape} does not match N={n_rows}"
            )
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features has {self.features.shape[1]} columns but "
                f"{len(self.feature_names)} feature_names were supplied"
            )
        if self.targets.shape[1] != len(self.target_names):
            raise ValueError(
                f"targets has {self.targets.shape[1]} columns but "
                f"{len(self.target_names)} target_names were supplied"
            )


@dataclass(frozen=True)
class SAEFit:
    """One trained SAE + MLP — standardization stats plus opaque state.

    `state_dict` is stored as ``Any`` so this module doesn't import torch.
    Callers that need the raw tensors load them back into a fresh
    `nn.Module` via `network.load_state_dict`.
    """

    fold: int
    seed: int
    feature_median: np.ndarray
    feature_mad: np.ndarray
    state_dict: Any
    train_loss: float
    val_loss: float
    val_auc: tuple[float, ...]
    n_train: int
    n_val: int

    def __post_init__(self) -> None:
        if self.feature_median.ndim != 1:
            raise ValueError(
                f"feature_median must be 1-D, got shape {self.feature_median.shape}"
            )
        if self.feature_mad.shape != self.feature_median.shape:
            raise ValueError(
                f"feature_mad shape {self.feature_mad.shape} != feature_median "
                f"shape {self.feature_median.shape}"
            )
        if self.fold < 0 or self.n_train < 0 or self.n_val < 0:
            raise ValueError("fold, n_train, n_val must be non-negative")


@dataclass(frozen=True)
class EnsembleFit:
    """Collection of `SAEFit`s plus inference metadata.

    `members` is the flat list of (fold, seed) trained models; predictions are
    median-blended across members per the spec's step 5.
    """

    members: tuple[SAEFit, ...]
    config: SAEConfig
    train: TrainConfig
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("EnsembleFit.members must be non-empty")

    @property
    def n_members(self) -> int:
        return len(self.members)
