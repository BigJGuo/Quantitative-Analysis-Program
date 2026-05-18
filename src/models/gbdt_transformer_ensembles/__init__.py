"""Public interface for Model 11: GBDT + Transformer Ensembles.

See `models/layer4_signals_ml/11_gbdt_transformer_ensembles.md` for the
full specification. This package implements:

- A pure-numpy gradient-boosted regression tree (Friedman 2001) with
  Huber/MSE losses, shrinkage, row/feature subsampling, leaf-L2 reg.
- A linear sequence-aware base learner with per-ticker rolling-window
  aggregates + ridge regression (a stand-in for the Transformer encoder
  when `torch` is not available).
- Purged group K-fold CV by date (Lopez de Prado 2018).
- A ridge stacking meta-learner over base-learner OOF predictions.
- A full feature engineering pipeline over yfinance OHLCV bars.
- Cross-sectional and time-series diagnostics (weighted R^2, rank IC,
  decile spread, PSI for concept drift, GBDT feature importance).
"""

from __future__ import annotations

from src.models.gbdt_transformer_ensembles.calibration import (
    MODEL_NAME,
    calibrate,
)
from src.models.gbdt_transformer_ensembles.cv import (
    iter_row_masks,
    purged_group_kfold,
)
from src.models.gbdt_transformer_ensembles.features import (
    build_feature_panel,
    compute_log_returns,
    fill_nans_for_linear,
    fill_nans_for_trees,
)
from src.models.gbdt_transformer_ensembles.gbdt import (
    GBDTParams,
    fit_gbdt,
    initial_prediction,
    loss_value,
    negative_gradient,
    predict_gbdt,
    predict_tree,
)
from src.models.gbdt_transformer_ensembles.model import GBDTTransformerEnsemble
from src.models.gbdt_transformer_ensembles.sequence_model import (
    LinearSeqParams,
    expand_features,
    fit_linear_seq,
    predict_linear_seq,
)
from src.models.gbdt_transformer_ensembles.signal import (
    calibration_bins,
    cross_sectional_rank_ic,
    decile_spread,
    population_stability_index,
    spearman_rank_ic,
    summarize_oof_metrics,
    time_series_ic,
    weighted_r2,
)
from src.models.gbdt_transformer_ensembles.stacking import (
    fit_stacker,
    predict_stacker,
)
from src.models.gbdt_transformer_ensembles.types import (
    VALID_BASE_LEARNERS,
    VALID_LOSSES,
    EnsembleFit,
    EnsemblePrediction,
    FeatureSpec,
    FoldSpec,
    GBDTFit,
    LinearSeqFit,
    PanelFrame,
    RegressionTree,
    StackingFit,
    TreeNode,
    coerce_base_learners,
)

__all__ = [
    "GBDTParams",
    "GBDTTransformerEnsemble",
    "MODEL_NAME",
    "VALID_BASE_LEARNERS",
    "VALID_LOSSES",
    "EnsembleFit",
    "EnsemblePrediction",
    "FeatureSpec",
    "FoldSpec",
    "GBDTFit",
    "LinearSeqFit",
    "LinearSeqParams",
    "PanelFrame",
    "RegressionTree",
    "StackingFit",
    "TreeNode",
    "build_feature_panel",
    "calibrate",
    "calibration_bins",
    "coerce_base_learners",
    "compute_log_returns",
    "cross_sectional_rank_ic",
    "decile_spread",
    "expand_features",
    "fill_nans_for_linear",
    "fill_nans_for_trees",
    "fit_gbdt",
    "fit_linear_seq",
    "fit_stacker",
    "initial_prediction",
    "iter_row_masks",
    "loss_value",
    "negative_gradient",
    "population_stability_index",
    "predict_gbdt",
    "predict_linear_seq",
    "predict_stacker",
    "predict_tree",
    "purged_group_kfold",
    "spearman_rank_ic",
    "summarize_oof_metrics",
    "time_series_ic",
    "weighted_r2",
]
