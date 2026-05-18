# Model 13 — Online Learning (Hedge / FTRL / FTPL)

Adversarial-regret online optimization for streaming signal aggregation and
streaming linear/logistic regression. Implements the Freund-Schapire
exponential-weights algorithm (Hedge), the McMahan FTRL-Proximal update for
L1/L2-regularized online linear and logistic regression, and the Gumbel-form
Follow-the-Perturbed-Leader sampler.

**Layer:** 4 — Signals / ML (online meta-learner / streaming regression).
**Spec:** [`models/layer4_signals_ml/13_online_learning.md`](../../../models/layer4_signals_ml/13_online_learning.md).
**Class:** `OnlineLearning` (`src.models.online_learning.model`).
**Registered as:** `"online_learning"` in the global model registry.

## Public API

From `src.models.online_learning`:

- `OnlineLearning` — `BaseModel` subclass. Default workflow: Hedge across
  rolling-σ + EWMA-σ vol experts for one ticker.
- `calibrate(stream, config, ...)` — warm-up Hedge on an `ExpertStream`,
  returns a `CalibrationResult` whose `parameters["hedge_fit"]` is a
  `HedgeFit`.
- `calibrate_ftrl(features, targets, config)` — warm-up FTRL-Proximal on a
  sparse feature stream; `parameters["ftrl_fit"]` is an `FTRLFit`.
- Hedge math: `initial_hedge_state`, `hedge_step`, `hedge_predict`,
  `expert_losses`, `clip_normalize_losses`, `adaptive_eta`,
  `run_hedge_stream`.
- FTRL math: `initial_ftrl_state`, `ftrl_theta`, `ftrl_predict`,
  `ftrl_update`, `ftrl_step`, `ftrl_dense_theta`.
- FTPL math: `ftpl_sample_action` (Gumbel-perturbation form).
- Diagnostics: `weight_entropy`, `empirical_regret`, `regret_bound_hedge`,
  `sparsity_fraction`, `cusum_break_detect`.
- Vol-panel construction: `realized_vol_target`, `rolling_realized_vol`,
  `ewma_vol`, `build_vol_expert_panel`.
- Numerics: `sigmoid`, `log_loss`, `safe_log`.
- Dataclasses: `HedgeConfig`, `HedgeState`, `HedgeFit`, `FTRLConfig`,
  `FTRLState`, `FTRLFit`, `ExpertStream`, `OnlineLearningInputs`.

## Usage (10 lines)

```python
from src.core.data_provider import YFinanceProvider
from src.models.online_learning import OnlineLearning

model = OnlineLearning(ticker="SPY", kind="hedge", annualize=True)
inputs = model.fetch_data(YFinanceProvider(cache=None))
model.calibrate(inputs)
forecast = model.predict(inputs)              # Forecast: annualized vol for tomorrow
diagnostics = model.validate(inputs)          # regret, weight entropy, CUSUM
print(forecast.value, diagnostics["empirical_regret"], diagnostics["weights"])
```

For Hedge over upstream model outputs (HAR-RV / GARCH / ML ensembles) build
the `ExpertStream` yourself and call `calibrate(...)` directly — skip
`fetch_data`. For FTRL on a sparse feature stream call `calibrate_ftrl(...)`
with `(indices, values)` tuples per example.

## Refit frequency

| Parameter                                          | Cadence (spec)     |
|----------------------------------------------------|--------------------|
| Hedge weights `w` / FTRL state `(z, n)`            | **every round**    |
| Hyperparameter retune (`α, β, λ₁` / `η`)           | monthly grid       |
| State checkpoint to disk                           | daily after close  |
| Fixed-share / reset on regime break                | event-driven       |

The class is declared with `refit_frequency = "daily"` — the orchestration
layer re-runs the streaming update each trading day, but the actual state
mutation happens *every round* inside that update. Hyperparameters
(`α, β, λ₁, η, fixed_share`) should be re-tuned by grid search on the
trailing 60-day prequential loss roughly once a month.

## Known limitations (carried from spec)

1. **Adversarial bound is loose in practice.** Worst-case regret
   `O(sqrt(T log N))` is conservative on benign markets and looser on truly
   adversarial sequences (flash crashes).
2. **Loss scale matters.** Hedge requires losses in `[0, 1]`; uncontrolled
   MSE on tail returns blows up the multiplicative update. The
   implementation clips and rescales via `clip_normalize_losses` using
   `l_max`. Tune this if the per-expert loss histogram changes.
3. **Slow adaptation under abrupt regime shift.** Vanilla Hedge takes
   `O(log N / eta)` rounds to forget a previously-best expert. Use
   `fixed_share > 0` (default `0.01`) or trigger a reset via the CUSUM
   detector.
4. **Tied experts.** Highly correlated experts inflate effective `N` without
   diversification benefit. The default vol-expert panel mixes rolling-σ
   (slow-moving) and EWMA-σ (fast-moving) on purpose, but you should cluster
   experts if you add many more.
5. **L1 sparsity instability (FTRL).** A feature near the L1 boundary can
   flip between `0` and nonzero across consecutive rounds. Spec advice:
   consider Group-Lasso / proximal averaging — neither is implemented here.
6. **Latency.** Hedge is `O(N)` per round; FTRL is `O(nnz(x_t))` per
   example. State logging and checkpointing add constant overhead.
7. **No uncertainty estimates.** Outputs are point predictions; pair with
   online conformal prediction for intervals (not implemented here).
8. **yfinance data freshness.** Live online learning prefers low-latency
   data; yfinance is best for daily / 15-minute refresh.
9. **Default expert panel is intentionally simple.** Rolling-σ and EWMA-σ
   are self-contained so this model has no cross-model dependencies. For
   the spec's headline use case (Hedge across HAR-RV, GARCH, ML ensembles)
   pass a pre-built `ExpertStream` to `calibrate()`.

## Notes on the test suite

- Unit tests (`src/models/online_learning/tests/`) verify each math function
  against the spec's worked formulas on synthetic data: the multiplicative
  update collapses to the closed-form `exp(-eta L_i)` ratio; the FTRL prox
  recovers the spec's `theta_j` closed form; Hedge applied to a single
  dominant expert collapses to that expert; the regret bound holds against
  i.i.d. synthetic streams.
- The integration test (`tests/integration/test_online_learning.py`) runs
  the full `fetch_data → calibrate → predict → validate` pipeline against
  live yfinance for SPY, checking that the meta annualized-vol forecast
  lands in the historical SPY band, that empirical regret stays below the
  theoretical bound, and that the weight vector remains a valid probability
  distribution.

## Open SHARED-CHANGE-REQUESTs

None.
