# Model 07 — Avellaneda-Lee PCA-Residual Statistical Arbitrage

Cross-sectional mean-reversion signal model. For each stock in the universe,
project returns onto a low-rank factor panel (PCA eigenportfolios or sector
ETFs), integrate the residuals into a cumulative process `X(t)`, fit an
AR(1) / Ornstein-Uhlenbeck process to that process, and standardize the
deviation from the OU mean as the Avellaneda-Lee s-score. Positions follow
a symmetric open-on-extreme / close-near-mean state machine.

**Layer:** 4 — Signals & ML.
**Spec:** [`models/layer4_signals_ml/07_avellaneda_lee_stat_arb.md`](../../../models/layer4_signals_ml/07_avellaneda_lee_stat_arb.md).
**Class:** `AvellanedaLeeStatArb` (`src.models.avellaneda_lee_stat_arb.model`).
**Registered as:** `"avellaneda_lee_stat_arb"` in the global model registry.

## Public API

From `src.models.avellaneda_lee_stat_arb`:

- `AvellanedaLeeStatArb` — `BaseModel` subclass; the orchestration entry point.
  - `fetch_data`, `calibrate`, `predict`, `validate` — the `BaseModel` contract.
  - `predict_all(data)` — emit one `Signal` per surviving ticker (cross section).
  - `aggregate_factor_exposure()` — `(K,)` net factor exposure of the current
    positions; the caller takes `-exposure` units of each factor portfolio to
    enforce factor neutrality (spec step 7).
  - `reset_positions()` — clear the cached prior-position state machine.
- `calibrate(...)` — pure dispatch from `StatArbInputs` to `CalibrationResult`.
  Implements steps 2-6 of the spec's Algorithm outline.
- Pure math primitives (also exported):
  - `pca_eigenportfolio_returns` — Avellaneda-Lee eigenportfolios (step 2).
  - `factor_regression` — per-stock OLS on factors (step 3).
  - `cumulative_residual` — integrate residuals into the OU price (step 3).
  - `fit_ou_ar1`, `ou_parameters_from_ar1` — AR(1) → OU mapping (step 4).
  - `s_score`, `s_score_modified` — raw and drift-corrected signal (step 5).
  - `position_decision` — open/close/hold state machine (step 6).
  - `factor_neutral_hedge` — aggregate factor exposure of a position book.
  - `adf_test_pvalue` — ADF p-value used in validation step 1.
- Dataclasses: `StatArbInputs`, `StatArbFit`, `OUFit`, `SignalThresholds`,
  `AR1Fit`, `OUParameters`, `FactorRegressionResult`.

## Usage (10 lines)

```python
from src.core.data_provider import YFinanceProvider
from src.models.avellaneda_lee_stat_arb import AvellanedaLeeStatArb

universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ"]
model = AvellanedaLeeStatArb(universe=universe, K=3, factor_mode="PCA",
                             pca_window=252, ou_window=60)
data = model.fetch_data(YFinanceProvider(cache=None))
model.calibrate(data)
signals = model.predict_all(data)               # one Signal per survivor
diag = model.validate(data)                     # ADF, half-life, s-score stats
exposure = model.aggregate_factor_exposure()    # hedge with -exposure of F_k
```

## Refit frequency

| Parameter                            | Cadence (spec)   |
|--------------------------------------|------------------|
| PCA / factor regression (`B`, `α`)   | daily            |
| OU parameters (`κ`, `m`, `σ_eq`)     | daily            |
| s-score thresholds                   | quarterly        |
| Universe membership                  | monthly          |
| `K` (number of factors)              | quarterly        |

The class declares `refit_frequency = "daily"` to match the load-bearing
factor / OU cadence.

## Known limitations (carried from spec)

1. **Survivorship bias.** yfinance returns only currently-listed tickers;
   backtesting with a current-membership universe overstates Sharpe by
   0.3–0.7. Production needs a point-in-time membership history.
2. **Regime collapse.** During stress (`ρ → 1`) the model under-allocates
   exactly when liquidity-provision premia are largest.
3. **Factor mis-specification.** Missing factors (crypto, ESG, dollar
   sensitivity) leak into residuals as spurious s-scores. The
   `max_residual_cross_correlation` diagnostic catches this — fed in from
   the factor model's `validate()` output when integrated upstream.
4. **Crowding.** PCA-residual is widely known; co-movement of "idiosyncratic"
   returns and synchronized stop-outs occur when many participants target
   the same residuals.
5. **Transaction costs and borrow.** With `τ_½ ≈ 10` days and modest
   `σ_eq`, a 20 bps round-trip absorbs the entire signal. Hard-to-borrow
   names need extra borrow-fee modeling.
6. **Microcap noise.** Stale prices look like mean reversion; strict ADV
   thresholds are required even with the half-life filter.
7. **Non-stationary covariance.** The top eigenvectors are stable across
   rolling refits, but lower eigenvectors are unstable. The MP-cutoff
   diagnostic surfaces an `n_factors_above_mp` count that callers can use
   to cap `K`.
8. **Estimation noise in `κ`.** With 60 observations, `σ̂(κ̂) ≈ κ̂`; the
   half-life and `σ_eq` inherit large uncertainty.

## Cross-model dependencies

The model is **logically downstream** of `factor_models_pca` (Model 04) —
when ETF mode is used, Model 04's eigenportfolios can be substituted in;
when PCA mode is used, this model computes its own eigenportfolios with
the spec's standardization-by-σ_i convention (different from Model 04's
covariance PCA). The math primitives are deliberately re-implemented here
to match the Avellaneda-Lee paper's exact formulation; Model 04's
`pca_decompose` is *not* a drop-in replacement (different weighting).

## Notes on the test suite

- Unit tests (`src/models/avellaneda_lee_stat_arb/tests/`) cover each math
  primitive against worked-out spec formulas:
  - `test_signal.py` — factor regression, AR(1) → OU mapping (long-path OU
    simulation), s-score / drift correction, position state machine,
    eigenportfolio recovery on a one-factor synthetic, factor-neutral hedge,
    ADF p-value.
  - `test_calibration.py` — end-to-end calibration on a synthetic OU-residual
    panel; verifies filter behaviour and drift-correction toggle.
  - `test_model.py` — orchestration via `InMemoryProvider`; verifies
    `BaseModel` contract, the position state machine across predictions,
    and that `validate()` returns the spec's diagnostics.
- The integration test (`tests/integration/test_avellaneda_lee_stat_arb.py`)
  exercises the full `fetch_data → calibrate → predict → validate` flow
  against live yfinance for an 8-name large-cap universe.
