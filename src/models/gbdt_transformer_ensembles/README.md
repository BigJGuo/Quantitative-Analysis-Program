# Model 11 — GBDT + Transformer Ensembles

Stacked ensemble of heterogeneous base learners over a daily OHLCV panel.
Layer 4 — Signals & ML.

**Spec:** [`models/layer4_signals_ml/11_gbdt_transformer_ensembles.md`](../../../models/layer4_signals_ml/11_gbdt_transformer_ensembles.md)

## What's in here

| Module | Purpose |
|---|---|
| `types.py` | Frozen dataclasses (`FeatureSpec`, `PanelFrame`, `GBDTFit`, `LinearSeqFit`, `StackingFit`, `EnsembleFit`, `EnsemblePrediction`, `FoldSpec`, `RegressionTree`, `TreeNode`). |
| `features.py` | yfinance feature engineering: multi-horizon log returns, lagged returns, rolling vol/mean, range, log-volume, Amihud, cross-sectional ranks, calendar encodings, benchmark spread, forward-return target with clipping. |
| `gbdt.py` | Pure-numpy gradient-boosted regression trees (Friedman 2001). Huber + MSE losses; shrinkage, row/feature subsampling, leaf-L2, early stopping. |
| `sequence_model.py` | Linear "sequence-aware" base learner: per-ticker rolling-window aggregates + EWMA → ridge regression. Stands in for the spec's Transformer/GRU until `torch` is available (see SHARED-CHANGE-REQUEST below). |
| `cv.py` | Purged group K-fold by date with `purge` + `embargo` (Lopez de Prado 2018). |
| `stacking.py` | Ridge meta-learner over base-learner OOF predictions, with optional non-negative convex-blend projection. |
| `signal.py` | Diagnostics: weighted R², Spearman cross-sectional rank IC, time-series IC, decile spread, calibration bins, PSI. |
| `calibration.py` | `calibrate()` — runs CV, fits per-fold base learners, builds OOF predictions, fits the stacker, refits each base learner on the full panel. |
| `model.py` | `GBDTTransformerEnsemble(BaseModel)` — fetches OHLCV via the `DataProvider`, builds the panel, drives the pipeline, exposes `predict()` (universe-mean) and `predict_cross_section()` (per-ticker). |

## Public API

```python
from src.models.gbdt_transformer_ensembles import (
    GBDTTransformerEnsemble,
    FeatureSpec,
    GBDTParams,
    LinearSeqParams,
)
```

Top-level types: `EnsembleFit`, `EnsemblePrediction`, `GBDTFit`,
`LinearSeqFit`, `StackingFit`, `PanelFrame`, `FoldSpec`,
`RegressionTree`, `TreeNode`, `VALID_BASE_LEARNERS`, `VALID_LOSSES`.

Top-level functions: `build_feature_panel`, `compute_log_returns`,
`fill_nans_for_trees`, `fill_nans_for_linear`, `fit_gbdt`, `predict_gbdt`,
`predict_tree`, `loss_value`, `negative_gradient`, `initial_prediction`,
`fit_linear_seq`, `predict_linear_seq`, `expand_features`, `fit_stacker`,
`predict_stacker`, `purged_group_kfold`, `iter_row_masks`,
`weighted_r2`, `spearman_rank_ic`, `cross_sectional_rank_ic`,
`time_series_ic`, `decile_spread`, `calibration_bins`,
`population_stability_index`, `summarize_oof_metrics`, `calibrate`,
`coerce_base_learners`.

## 10-line usage example

```python
from src.core.data_provider import YFinanceProvider
from src.models.gbdt_transformer_ensembles import (
    GBDTTransformerEnsemble, FeatureSpec, GBDTParams, LinearSeqParams,
)

provider = YFinanceProvider(cache=None)
model = GBDTTransformerEnsemble(
    universe=["XLK", "XLF", "XLE", "XLV", "XLY", "XLP"],
    feature_spec=FeatureSpec(target_horizon=5, benchmark_ticker="SPY"),
    gbdt_params=GBDTParams(n_estimators=200, learning_rate=0.05, max_depth=5),
    linseq_params=LinearSeqParams(windows=(5, 21), ewma_alpha=0.1),
    n_splits=5, purge=5, embargo=1, period="2y",
)
panel = model.fetch_data(provider)
result = model.calibrate(panel)
cross_section = model.predict_cross_section(panel)
```

## Refit frequency

- GBDT base learners: weekly clean retrain on a 2-year rolling window.
- Linear sequence base learner: weekly retrain with the GBDTs.
- Stacking meta-learner: weekly with the latest OOF window.
- Trigger-based refit: rolling 20-day OOS R² below the 5th in-sample
  percentile.

## Known limitations

1. **No Transformer / GRU yet.** The spec calls for a real Transformer
   encoder and a GRU + wide-and-deep alternative. Both require `torch`,
   which is not declared in the project's core dependencies. The
   current `linear_seq` base learner — per-ticker rolling aggregates +
   EWMA + ridge — captures the *role* of a sequence-aware learner
   (time-aware features, low GBDT error correlation) but is materially
   weaker than a true Transformer on large panels. See the
   SHARED-CHANGE-REQUEST below.
2. **No `lightgbm`/`xgboost`/`catboost`.** The pure-numpy GBDT
   implements the spec's math (Friedman 2001 + Newton-style leaf-L2 reg)
   but is much slower than the production libraries. It's acceptable for
   the daily / weekly cadence of this layer, not for nightly retrains on
   tens of millions of rows.
3. **Tree extrapolation.** Per the spec, GBDTs cannot extrapolate beyond
   observed feature ranges; new regime values (negative oil, etc.) get
   mapped to the nearest leaf boundary.
4. **Target clipping artifacts.** Clipping creates a flat region in the
   loss landscape. The default loss is Huber to mitigate this.
5. **Inference latency.** The python tree-walk in `predict_tree` is
   intentionally simple. For latency-sensitive use, JIT-compile or swap
   in a vectorized backend.
6. **Concept drift.** PSI is computed against a 50/50 chronological
   split of OOF predictions in `validate()`. For production, swap that
   for a streaming reference window.
7. **Universe coverage drops.** Tickers with < `min_history_days` of
   contiguous OHLCV history are silently dropped (per the spec). Check
   `len(panel.meta["ticker"].unique())` after `fetch_data()` to see what
   stayed in.

## SHARED-CHANGE-REQUEST

```
SHARED-CHANGE-REQUEST
File: pyproject.toml
Reason: Model 11's spec calls for LightGBM/XGBoost/CatBoost as the GBDT
        backend and a Transformer encoder + GRU as the second base
        learner family. None of these are declared dependencies, so the
        model ships a pure-numpy GBDT and a linear sequence-model
        stand-in. Adding optional dependency groups would let
        production deployments swap in the canonical implementations
        without code changes.
Proposed change: Add the following optional dependency groups:

    [project.optional-dependencies]
    gbdt-libs = ["lightgbm>=4.0", "xgboost>=2.0", "catboost>=1.2"]
    deep      = ["torch>=2.0"]

Workaround until merged: This module uses pure-numpy GBDT and a linear
sequence model. The interfaces (`GBDTParams`, `LinearSeqParams`) leave
room for a future backend dispatch, so swapping in lightgbm or torch
is a localized change in `gbdt.py` / `sequence_model.py` once those
deps are installable.
```

## Tests

- Unit: `src/models/gbdt_transformer_ensembles/tests/` — 65 tests
  covering every public math function (loss/gradient identities,
  weighted-median initialization, tree-prediction edge cases, GBDT
  convergence on a step function, sequence-model recovery, purged-CV
  invariants, ridge-stacker behavior, diagnostic-metric properties).
- Integration: `tests/integration/test_gbdt_transformer_ensembles.py` —
  hits real yfinance for a sector-ETF universe and exercises the full
  pipeline including the spec's "Validation and diagnostics" section.
