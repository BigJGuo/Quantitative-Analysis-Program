"""Calibration entry point for the GBDT + Transformer ensemble.

`calibrate()` runs the full CV-train-stack pipeline:

  1. Build purged K-fold splits on the panel's date axis.
  2. For each fold, train each base learner on the train slice and
     predict the validation slice (out-of-fold).
  3. Stack the OOF predictions with a ridge meta-learner.
  4. Refit each base learner on the full panel with the same params.
  5. Compute diagnostics on the OOF predictions and return everything in
     an `EnsembleFit`.

This module is data-only — no I/O, no DataProvider. The model class
handles fetching prices and assembling the panel, then calls this.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.gbdt_transformer_ensembles.cv import (
    iter_row_masks,
    purged_group_kfold,
)
from src.models.gbdt_transformer_ensembles.features import (
    fill_nans_for_linear,
    fill_nans_for_trees,
)
from src.models.gbdt_transformer_ensembles.gbdt import (
    GBDTParams,
    fit_gbdt,
    predict_gbdt,
)
from src.models.gbdt_transformer_ensembles.sequence_model import (
    LinearSeqParams,
    fit_linear_seq,
    predict_linear_seq,
)
from src.models.gbdt_transformer_ensembles.signal import summarize_oof_metrics
from src.models.gbdt_transformer_ensembles.stacking import (
    fit_stacker,
    predict_stacker,
)
from src.models.gbdt_transformer_ensembles.types import (
    BaseLearnerName,
    EnsembleFit,
    FeatureSpec,
    GBDTFit,
    LinearSeqFit,
    PanelFrame,
    coerce_base_learners,
)

MODEL_NAME: str = "gbdt_transformer_ensembles"


def _train_base_learner(
    name: BaseLearnerName,
    panel: PanelFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    feature_spec: FeatureSpec,
    gbdt_params: GBDTParams,
    linseq_params: LinearSeqParams,
) -> tuple[Any, np.ndarray]:
    """Fit one base learner on the train rows; return (fit, val_predictions)."""

    if name == "gbdt":
        X_tree = fill_nans_for_trees(panel.X, feature_spec.nan_sentinel)
        gbdt_fit = fit_gbdt(
            X_tree[train_mask],
            panel.y.to_numpy()[train_mask],
            panel.w.to_numpy()[train_mask],
            feature_names=panel.feature_names,
            params=gbdt_params,
            X_val=X_tree[val_mask],
            y_val=panel.y.to_numpy()[val_mask],
            w_val=panel.w.to_numpy()[val_mask],
        )
        val_pred = predict_gbdt(gbdt_fit, X_tree[val_mask])
        return gbdt_fit, val_pred

    if name == "linear_seq":
        X_lin, _ = fill_nans_for_linear(panel.X)
        X_lin_df = pd.DataFrame(X_lin, columns=list(panel.feature_names))
        tk = panel.meta["ticker"].reset_index(drop=True)
        linseq_fit = fit_linear_seq(
            X_lin_df.loc[train_mask].reset_index(drop=True),
            panel.y.to_numpy()[train_mask],
            panel.w.to_numpy()[train_mask],
            feature_names=panel.feature_names,
            tickers=tk.loc[train_mask].reset_index(drop=True),
            params=linseq_params,
        )
        val_pred = predict_linear_seq(
            linseq_fit,
            X_lin_df.loc[val_mask].reset_index(drop=True),
            tk.loc[val_mask].reset_index(drop=True),
            windows=linseq_params.windows,
            ewma_alpha=linseq_params.ewma_alpha,
        )
        return linseq_fit, val_pred

    raise ValueError(f"Unknown base learner {name!r}")


def _refit_full(
    name: BaseLearnerName,
    panel: PanelFrame,
    *,
    feature_spec: FeatureSpec,
    gbdt_params: GBDTParams,
    linseq_params: LinearSeqParams,
) -> Any:
    if name == "gbdt":
        X_tree = fill_nans_for_trees(panel.X, feature_spec.nan_sentinel)
        # Disable early stopping on the full refit.
        full_params = GBDTParams(
            n_estimators=gbdt_params.n_estimators,
            learning_rate=gbdt_params.learning_rate,
            max_depth=gbdt_params.max_depth,
            min_samples_leaf=gbdt_params.min_samples_leaf,
            subsample=gbdt_params.subsample,
            colsample=gbdt_params.colsample,
            l2_leaf_reg=gbdt_params.l2_leaf_reg,
            loss=gbdt_params.loss,
            huber_delta=gbdt_params.huber_delta,
            early_stopping_rounds=None,
            seed=gbdt_params.seed,
        )
        return fit_gbdt(
            X_tree,
            panel.y.to_numpy(),
            panel.w.to_numpy(),
            feature_names=panel.feature_names,
            params=full_params,
        )
    if name == "linear_seq":
        X_lin, _ = fill_nans_for_linear(panel.X)
        X_lin_df = pd.DataFrame(X_lin, columns=list(panel.feature_names))
        return fit_linear_seq(
            X_lin_df,
            panel.y.to_numpy(),
            panel.w.to_numpy(),
            feature_names=panel.feature_names,
            tickers=panel.meta["ticker"].reset_index(drop=True),
            params=linseq_params,
        )
    raise ValueError(f"Unknown base learner {name!r}")


def calibrate(
    panel: PanelFrame,
    *,
    feature_spec: FeatureSpec,
    base_learners: Sequence[str] | None = None,
    n_splits: int = 5,
    purge: int = 5,
    embargo: int = 1,
    gbdt_params: GBDTParams | None = None,
    linseq_params: LinearSeqParams | None = None,
    stacker_alpha: float = 1.0,
    non_negative_stacker: bool = False,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Run the full pipeline: CV training, stacking, full-data refit.

    Parameters
    ----------
    panel:
        Long-format feature panel from `features.build_feature_panel()`.
    feature_spec:
        Feature config — used for the GBDT NaN sentinel and `clip_target`.
    base_learners:
        Subset of `{"gbdt", "linear_seq"}`. Defaults to both.
    n_splits, purge, embargo:
        Purged K-fold hyperparameters (date-axis).
    gbdt_params, linseq_params:
        Per-base-learner hyperparameters. `None` uses the dataclass defaults.
    stacker_alpha:
        Ridge penalty for the meta-learner. `non_negative_stacker=True`
        projects coefficients to the non-negative orthant and L1-normalizes
        them (true blend, no sign flips).
    """

    names = coerce_base_learners(base_learners)
    gbdt_p = gbdt_params or GBDTParams()
    lin_p = linseq_params or LinearSeqParams()
    ts = timestamp or datetime.now(UTC)

    if panel.n_rows < 2 * n_splits:
        raise ValueError(
            f"Panel has {panel.n_rows} rows but needs >= {2 * n_splits} "
            "for {n_splits}-fold CV"
        )

    folds = purged_group_kfold(
        panel.meta["date"],
        n_splits=n_splits,
        purge=purge,
        embargo=embargo,
    )

    oof_pred_cols: dict[str, np.ndarray] = {n: np.full(panel.n_rows, np.nan) for n in names}
    seen_mask = np.zeros(panel.n_rows, dtype=bool)

    for fold, train_mask, val_mask in iter_row_masks(folds, panel.meta["date"]):
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        for name in names:
            _, val_pred = _train_base_learner(
                name,
                panel,
                train_mask,
                val_mask,
                feature_spec=feature_spec,
                gbdt_params=gbdt_p,
                linseq_params=lin_p,
            )
            oof_pred_cols[name][val_mask] = val_pred
        seen_mask = seen_mask | val_mask
        _ = fold  # fold id retained in case we want per-fold reporting

    # Restrict OOF reporting to rows that actually got a validation prediction.
    keep = seen_mask & np.all(
        np.stack([np.isfinite(oof_pred_cols[n]) for n in names], axis=0), axis=0
    )
    if keep.sum() == 0:
        raise RuntimeError("No rows received a complete set of OOF predictions")

    oof_df = pd.DataFrame(
        {n: oof_pred_cols[n] for n in names},
        index=np.arange(panel.n_rows),
    )
    oof_df["y"] = panel.y.to_numpy()
    oof_df["w"] = panel.w.to_numpy()
    oof_df["date"] = panel.meta["date"].to_numpy()
    oof_df["ticker"] = panel.meta["ticker"].to_numpy()
    oof_df = oof_df.loc[keep].reset_index(drop=True)

    stacker = None
    if len(names) >= 2:
        Z = oof_df[list(names)].to_numpy(dtype=float)
        stacker = fit_stacker(
            Z,
            oof_df["y"].to_numpy(),
            oof_df["w"].to_numpy(),
            base_learner_names=names,
            alpha=stacker_alpha,
            non_negative=non_negative_stacker,
        )
        oof_df["yhat_ensemble"] = predict_stacker(stacker, Z)
    else:
        oof_df["yhat_ensemble"] = oof_df[names[0]].to_numpy()

    cv_metrics = summarize_oof_metrics(oof_df)

    base_fits: dict[str, Any] = {}
    for name in names:
        base_fits[name] = _refit_full(
            name,
            panel,
            feature_spec=feature_spec,
            gbdt_params=gbdt_p,
            linseq_params=lin_p,
        )

    fit = EnsembleFit(
        base_fits=base_fits,
        stacker=stacker,
        feature_names=panel.feature_names,
        base_learner_names=names,
        target_horizon=feature_spec.target_horizon,
        clip_target=feature_spec.clip_target,
        oof_predictions=oof_df,
        cv_metrics=cv_metrics,
        timestamp=ts,
        metadata={
            "n_splits": n_splits,
            "purge": purge,
            "embargo": embargo,
            "stacker_alpha": stacker_alpha,
        },
    )

    parameters: dict[str, Any] = {
        "ensemble_fit": fit,
        "base_learners": list(names),
        "target_horizon": feature_spec.target_horizon,
    }
    fit_metrics: dict[str, float] = {
        k: float(v) for k, v in cv_metrics.items() if isinstance(v, (int, float))
    }
    if fit.stacker is not None:
        for stk_name, c in zip(
            fit.stacker.base_learner_names, fit.stacker.coefficients, strict=True
        ):
            fit_metrics[f"stacker_w_{stk_name}"] = float(c)
        fit_metrics["stacker_intercept"] = float(fit.stacker.intercept)
    for bfit_name, b_fit in base_fits.items():
        if isinstance(b_fit, GBDTFit):
            fit_metrics[f"gbdt_n_trees_{bfit_name}"] = float(b_fit.n_trees)
        elif isinstance(b_fit, LinearSeqFit):
            fit_metrics[f"linseq_n_coef_{bfit_name}"] = float(b_fit.coefficients.shape[0])

    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata={
            "n_features": panel.X.shape[1],
            "n_samples": panel.n_rows,
            "n_dates": int(panel.meta["date"].nunique()),
            "n_tickers": int(panel.meta["ticker"].nunique()),
        },
    )


__all__ = ["MODEL_NAME", "calibrate"]
