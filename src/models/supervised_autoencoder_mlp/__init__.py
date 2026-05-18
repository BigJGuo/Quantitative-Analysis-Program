"""Public interface for the Supervised Autoencoder + MLP model.

See `models/layer4_signals_ml/10_supervised_autoencoder_mlp.md` for the full
specification.

Heavy training / inference (PyTorch) is loaded lazily — importing this package
does *not* import torch, so the rest of the ecosystem can introspect the model
metadata without pulling in a large optional dependency.
"""

from __future__ import annotations

from src.models.supervised_autoencoder_mlp.calibration import (
    MODEL_NAME,
    calibrate,
)
from src.models.supervised_autoencoder_mlp.model import SupervisedAEMLP
from src.models.supervised_autoencoder_mlp.signal import (
    action_rule,
    binarize_targets,
    blend_predictions,
    build_feature_panel,
    impute_with_median,
    median_mad_standardize,
    purged_kfold_indices,
    rolling_log_returns,
    rolling_std,
    winsorize,
)
from src.models.supervised_autoencoder_mlp.types import (
    EnsembleFit,
    SAEConfig,
    SAEFit,
    SAEInputs,
    TrainConfig,
)

__all__ = [
    "MODEL_NAME",
    "EnsembleFit",
    "SAEConfig",
    "SAEFit",
    "SAEInputs",
    "SupervisedAEMLP",
    "TrainConfig",
    "action_rule",
    "binarize_targets",
    "blend_predictions",
    "build_feature_panel",
    "calibrate",
    "impute_with_median",
    "median_mad_standardize",
    "purged_kfold_indices",
    "rolling_log_returns",
    "rolling_std",
    "winsorize",
]
