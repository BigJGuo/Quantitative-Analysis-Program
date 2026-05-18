"""Calibration entry point for the Supervised Autoencoder + MLP.

`calibrate(data, **params)` runs the full purged K-fold * seed-ensemble
training loop described in the spec's Algorithm outline (steps 3-4):

1. Slice the SAEInputs into purged-K-fold train/val splits.
2. For each (fold, seed):
   a. Robust-standardize features on the train indices.
   b. Apply the same statistics to the val indices.
   c. Train one network; record the held-out loss and per-target AUCs.
   d. Store the state-dict in a `SAEFit`.
3. Wrap all `SAEFit`s in an `EnsembleFit` and return it inside a
   `CalibrationResult`.

The function is pure data-in / parameters-out; the model class does the I/O.
Torch is imported lazily — calibrate() will raise RuntimeError if torch is
unavailable, which the model class can catch.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from src.core.types import CalibrationResult
from src.models.supervised_autoencoder_mlp.signal import (
    impute_with_median,
    median_mad_standardize,
    purged_kfold_indices,
)
from src.models.supervised_autoencoder_mlp.types import (
    EnsembleFit,
    SAEConfig,
    SAEFit,
    SAEInputs,
    TrainConfig,
)

MODEL_NAME: str = "supervised_autoencoder_mlp"


def calibrate(
    data: SAEInputs,
    *,
    config: SAEConfig | None = None,
    train_cfg: TrainConfig | None = None,
    device: str = "cpu",
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Train the ensemble. See module docstring for the algorithm.

    Parameters
    ----------
    data:
        Feature / target / weight bundle produced by the model's `fetch_data`.
    config:
        Architecture hyperparameters. If `n_features` is left at its default
        of 0, it is overwritten with `data.features.shape[1]`.
    train_cfg:
        Optimization hyperparameters. Defaults to `TrainConfig()`.
    device:
        Torch device string (``"cpu"`` for tests; ``"cuda"`` in production
        when a GPU is present).
    timestamp:
        Calibration timestamp; defaults to ``datetime.now(UTC)``.
    """

    ts = timestamp or datetime.now(UTC)
    train_cfg = train_cfg or TrainConfig()
    p = int(data.features.shape[1])
    if config is None:
        config = SAEConfig(n_features=p, n_targets=int(data.targets.shape[1]))
    elif config.n_features == 0:
        config = _override_features(config, p, int(data.targets.shape[1]))
    if config.n_features != p:
        raise ValueError(
            f"SAEConfig.n_features ({config.n_features}) does not match "
            f"data.features columns ({p})"
        )
    if config.n_targets != data.targets.shape[1]:
        raise ValueError(
            f"SAEConfig.n_targets ({config.n_targets}) does not match "
            f"data.targets columns ({data.targets.shape[1]})"
        )

    # Lazy torch import — surfaces a clear RuntimeError if torch is missing.
    from src.models.supervised_autoencoder_mlp.network import train_one

    X = data.features.to_numpy(dtype=float)
    Y = data.targets.to_numpy(dtype=float)
    W = (
        data.sample_weights.astype(float)
        if data.sample_weights is not None
        else None
    )

    members: list[SAEFit] = []
    for split in purged_kfold_indices(
        data.dates, n_folds=train_cfg.n_folds, embargo=train_cfg.embargo
    ):
        # Drop any train/val rows whose target row is NaN (e.g. tail rows that
        # have no forward return yet).
        train_idx = _filter_finite(split.train_idx, Y)
        val_idx = _filter_finite(split.val_idx, Y)
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        x_tr_raw = X[train_idx]
        x_va_raw = X[val_idx]
        # Impute then standardize using train-only statistics.
        x_tr_imp, train_median, _ = impute_with_median(x_tr_raw)
        x_va_imp, _, _ = impute_with_median(x_va_raw, median=train_median)
        x_tr_std, fit_median, fit_scale = median_mad_standardize(x_tr_imp)
        x_va_std, _, _ = median_mad_standardize(
            x_va_imp, median=fit_median, mad=fit_scale / 1.4826
        )
        y_tr = Y[train_idx]
        y_va = Y[val_idx]
        w_tr = W[train_idx] if W is not None else None

        for seed in train_cfg.seeds:
            state_dict, train_loss, val_loss, val_auc = train_one(
                x_train=x_tr_std,
                y_train=y_tr,
                x_val=x_va_std,
                y_val=y_va,
                sample_weights=w_tr,
                config=config,
                train_cfg=train_cfg,
                seed=seed,
                device=device,
            )
            members.append(
                SAEFit(
                    fold=split.fold,
                    seed=seed,
                    feature_median=fit_median,
                    feature_mad=fit_scale,
                    state_dict=state_dict,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    val_auc=tuple(float(a) for a in val_auc),
                    n_train=int(train_idx.size),
                    n_val=int(val_idx.size),
                )
            )

    if not members:
        raise RuntimeError(
            "calibrate produced no ensemble members — every fold had empty "
            "train or validation sets after NaN filtering."
        )

    fit = EnsembleFit(
        members=tuple(members),
        config=config,
        train=train_cfg,
        feature_names=data.feature_names,
        target_names=data.target_names,
        timestamp=ts,
    )

    fit_metrics = _ensemble_metrics(fit)
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={
            "ensemble_fit": fit,
            "config": config,
            "train_cfg": train_cfg,
        },
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata={
            "n_members": float(fit.n_members),
            "n_features": float(config.n_features),
            "n_targets": float(config.n_targets),
            "n_folds": float(train_cfg.n_folds),
            "n_seeds": float(len(train_cfg.seeds)),
            "epochs": float(train_cfg.epochs),
            "device": device,
        },
    )


def _filter_finite(idx: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Drop rows whose target vector contains NaN (no forward return available)."""

    if idx.size == 0:
        return idx
    sub = Y[idx]
    mask = np.all(np.isfinite(sub), axis=1)
    return idx[mask]


def _override_features(config: SAEConfig, n_features: int, n_targets: int) -> SAEConfig:
    """Return a copy of `config` with `n_features` / `n_targets` overridden."""

    return SAEConfig(
        n_features=n_features,
        n_targets=n_targets,
        encoder_hidden=config.encoder_hidden,
        bottleneck_dim=config.bottleneck_dim,
        decoder_hidden=config.decoder_hidden,
        mlp_hidden=config.mlp_hidden,
        dropout_encoder=config.dropout_encoder,
        dropout_mlp=config.dropout_mlp,
        noise_sigma=config.noise_sigma,
    )


def _ensemble_metrics(fit: EnsembleFit) -> dict[str, float]:
    train_losses = np.array([m.train_loss for m in fit.members], dtype=float)
    val_losses = np.array([m.val_loss for m in fit.members], dtype=float)
    aucs = np.array(
        [m.val_auc for m in fit.members], dtype=float
    )  # (n_members, n_targets)
    metrics: dict[str, float] = {
        "mean_train_loss": float(train_losses.mean()),
        "mean_val_loss": float(val_losses.mean()),
        "std_val_loss": float(val_losses.std(ddof=0)) if val_losses.size > 1 else 0.0,
        "min_val_loss": float(val_losses.min()),
        "max_val_loss": float(val_losses.max()),
        "n_members": float(fit.n_members),
    }
    for k, name in enumerate(fit.target_names):
        metrics[f"mean_auc_{name}"] = float(aucs[:, k].mean())
    return metrics
