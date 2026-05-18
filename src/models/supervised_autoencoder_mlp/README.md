# Model 10 — Supervised Autoencoder + MLP

Jane Street 2021 Kaggle winner: a tabular deep-learning architecture that
jointly trains a denoising autoencoder bottleneck `h = E_phi(x)` with two
classification heads (an auxiliary head on `h` and a main MLP head on
`[x; h]`) under a triple loss

```
L = alpha_rec * MSE(x_hat, x)
  + alpha_aux * BCE(y_aux, y)
  + alpha_main * BCE(y_hat, y).
```

Inference is a median-blended ensemble across `K` purged-K-folds and `S`
random seeds, and the spec's action rule fires when any horizon's blended
probability exceeds `tau = 0.5`.

**Layer:** 4 — Signals & ML.
**Spec:** [`models/layer4_signals_ml/10_supervised_autoencoder_mlp.md`](../../../models/layer4_signals_ml/10_supervised_autoencoder_mlp.md).
**Class:** `SupervisedAEMLP` (`src.models.supervised_autoencoder_mlp.model`).
**Registered as:** `"supervised_autoencoder_mlp"` in the global model registry.

## Public API

From `src.models.supervised_autoencoder_mlp`:

- `SupervisedAEMLP` — `BaseModel` subclass; the orchestration entry point.
- `calibrate(data, ...)` — pure training entry point; returns a
  `CalibrationResult` whose `parameters["ensemble_fit"]` is an `EnsembleFit`.
- `build_feature_panel(bars, ...)` — per-ticker OHLCV → feature DataFrame
  (returns at multiple horizons, rolling vol, realized vol, HL range,
  volume ratio, Amihud illiquidity).
- `binarize_targets(forward_returns, horizons)` — spec step 2: three binary
  targets from a forward-return panel.
- `purged_kfold_indices(dates, n_folds, embargo)` — purged time-series
  K-fold per de Prado AFML ch. 7.
- `median_mad_standardize`, `winsorize`, `impute_with_median` — robust
  per-feature preprocessing.
- `blend_predictions(list_of_probs)`, `action_rule(blended, tau)` — the
  spec's median blend + take-trade rule.
- Diagnostics: `roc_auc`, `brier_score`, `reconstruction_error`.
- Dataclasses: `SAEConfig`, `TrainConfig`, `SAEInputs`, `SAEFit`,
  `EnsembleFit`.

## Usage (10 lines)

```python
from src.core.data_provider import YFinanceProvider
from src.models.supervised_autoencoder_mlp import SAEConfig, SupervisedAEMLP, TrainConfig

universe = ["SPY", "QQQ", "IWM"]
model = SupervisedAEMLP(
    universe=universe,
    config=SAEConfig(bottleneck_dim=64),
    train_cfg=TrainConfig(epochs=50, n_folds=5, seeds=(0, 1, 2, 3, 4)),
)
data = model.fetch_data(YFinanceProvider(cache=None))
model.calibrate(data)              # CalibrationResult w/ EnsembleFit
sig  = model.predict(data)          # Signal: latest action + max probability
diag = model.validate(data)         # AUC, Brier, recon error, seed correlation
```

## Refit frequency

| Cadence (spec) | What runs                                          |
|----------------|----------------------------------------------------|
| Weekly         | Full retrain on a sliding ~5-year window           |
| Daily          | 1–3 epoch fine-tune on most recent 60 days         |
| Weekly         | Ensemble refresh: rotate one seed in / one out     |
| Trigger        | Full retrain if 20-day rolling Sharpe < 5th pctile |

The class is declared with `refit_frequency = "weekly"` to match the full
retrain cadence. The orchestration layer wires daily fine-tuning separately.

## Known limitations (carried from spec)

1. **Regime change.** Latent `h` drifts when the train window misses a new
   regime; reconstruction-error spikes are the early warning.
2. **Label leakage via target construction.** Forward-return horizons must be
   embargoed in the K-fold split — the model uses `embargo >= 1` trading day
   by default; bump it when targets span longer horizons.
3. **BatchNorm at inference.** Live single-sample inference requires
   `model.eval()` — `predict_proba` / `validate` always set this; calling
   the network directly does not.
4. **Seed collapse.** Two seeds drifting to the same local optimum kills
   ensemble benefit. `validate()` reports `mean_ensemble_seed_correlation`.
5. **Class imbalance.** Universes biased toward up days yield high `y_hat`;
   monitor per-fold base rates.
6. **yfinance survivorship bias.** Only currently-listed names are returned;
   train on a point-in-time universe to avoid look-ahead.
7. **Noise sigma sensitivity.** Too large destroys signal; too small offers
   no regularization. Tune `SAEConfig.noise_sigma` by validation utility.
8. **Bottleneck collapse.** If `alpha_main` dominates, `h` degenerates to a
   classifier embedding and the reconstruction loss stops constraining it.

## Open SHARED-CHANGE-REQUESTs

```
SHARED-CHANGE-REQUEST
File: pyproject.toml
Reason: This model needs PyTorch for the SAE+MLP network. Torch is loaded
        lazily so the package can be imported without it, but the calibrate
        / predict / validate paths require it at runtime; tests use
        `pytest.importorskip("torch")` and skip cleanly when absent.
Proposed change: Add `torch>=2.0` to `[project.optional-dependencies]` under
                 a new `ml` extra (so users who only want the structural
                 layer don't pay the torch install cost).
Workaround until merged: callers `pip install torch` themselves; tests skip
                         the torch-dependent suite when it isn't present.
```

## Notes on the test suite

- Unit tests in `src/models/supervised_autoencoder_mlp/tests/`:
  - `test_types.py`, `test_signal.py` — pure numpy / pandas, run without torch.
  - `test_calibration.py`, `test_model.py` — require torch
    (`pytest.importorskip`).
- Integration test in `tests/integration/test_supervised_autoencoder_mlp.py`
  exercises the full pipeline against yfinance for `SPY / QQQ / IWM` with a
  miniature 2-epoch / 2-fold / 1-seed configuration so the run stays under
  ~1 minute on CPU.
