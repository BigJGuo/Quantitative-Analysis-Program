# Model 06 — Cointegration and Pair Trading

Two-leg statistical-arbitrage model. Tests for cointegration via Engle-Granger,
fits an Ornstein-Uhlenbeck process to the residual, and trades the standardized
spread. A Kalman-filter variant allows the hedge ratio to drift as a random
walk and trades the standardized innovation in place of the static z-score.

**Layer:** 4 — Signals / ML.
**Spec:** [`models/layer4_signals_ml/06_cointegration_pairs.md`](../../../models/layer4_signals_ml/06_cointegration_pairs.md).
**Class:** `CointegrationPairs` (`src.models.cointegration_pairs.model`).
**Registered as:** `"cointegration_pairs"` in the global model registry.

## Public API

From `src.models.cointegration_pairs`:

- `CointegrationPairs` — `BaseModel` subclass; the orchestration entry point.
- `calibrate(...)` — pure-function pipeline: Engle-Granger + OU + (optional)
  Kalman init; returns a `CalibrationResult` whose `parameters["pair_fit"]`
  is a `PairFit`.
- `engle_granger_test`, `adf_test`, `ols_intercept_slope` — Stage-1 building
  blocks as pure functions.
- `fit_ar1`, `fit_ou`, `half_life_from_phi` — AR(1) → OU mapping.
- `kalman_dynamic_beta` — random-walk hedge-ratio filter; returns a
  `KalmanFit` with the full beta / variance / innovation path.
- `static_zscore`, `rolling_zscore_moments`, `pair_position_path` — z-score
  and trading-rule logic (entry / exit / stop on `s_in`, `s_out`, `s_stop`).
- `select_dependent_orientation` — picks `A on B` vs `B on A` by EG t-stat.
- `ljung_box_pvalue`, `rolling_eg_residual_tstat`, `realized_half_life` —
  validation diagnostics.
- Dataclasses: `PairInputs`, `PairFit`, `EngleGrangerFit`, `OUFit`,
  `KalmanFit`, `ADFResult`, `TradingRule`, `PairMethod`, `PairStatus`.

## Usage (10 lines)

```python
from src.core.data_provider import YFinanceProvider
from src.models.cointegration_pairs import CointegrationPairs

model = CointegrationPairs("KO", "PEP", method="static", period="2y")
data = model.fetch_data(YFinanceProvider(cache=None))
model.calibrate(data)             # Stage 1: Engle-Granger + OU screening
signal = model.predict(data)      # Signal: direction + strength on KO/PEP
diag = model.validate(data)       # rolling EG t-stat, half-life check
spread = model.spread(data)       # residual Z_t time series
positions = model.position_path(data)  # {-1, 0, +1} per bar
```

For the Kalman variant: `CointegrationPairs("KO", "PEP", method="kalman", Q=1e-4, R=1e-3)`.

## Refit frequency

| Parameter                          | Cadence (spec)      |
|------------------------------------|---------------------|
| Engle-Granger β + α                | weekly              |
| OU κ, μ, σ                         | weekly              |
| Rolling z-score moments            | daily               |
| Kalman β_t                         | continuous (per bar)|
| Kalman Q, R                        | quarterly (MLE)     |
| Universe / pair screen             | monthly             |

The class is declared with `refit_frequency = "weekly"` to match the
load-bearing Engle-Granger cadence. The Kalman filter itself updates each bar
inside `predict`, so daily orchestration is enough.

## Algorithm summary (mapping to spec Algorithm outline)

| Spec step | Implemented in                                            |
|-----------|-----------------------------------------------------------|
| 1         | `model.fetch_data`, `signal.compute_log_prices`, `align_pair` |
| 2         | `signal.adf_test` (per leg)                               |
| 3         | `signal.ols_intercept_slope`                              |
| 4         | `signal.engle_granger_test` (residuals)                   |
| 5         | `signal.adf_test(test_type="engle_granger")`              |
| 6         | `signal.fit_ar1`                                          |
| 7         | `signal.fit_ou`, `half_life_from_phi`                     |
| 8         | `calibration._assess_pair` (half-life + φ filters)        |
| 9         | Stage 2a — `static_zscore`, `rolling_zscore_moments`, `pair_position_path` |
| 10-12     | Stage 2b — `kalman_dynamic_beta`                          |
| 13-14     | Out of scope at the per-pair level (orchestration layer)  |

ADF / Engle-Granger critical values are MacKinnon (1996) asymptotics; we do
not depend on statsmodels. Critical values are conservative for the
single-regressor EG residual case.

## Known limitations (carried from spec)

1. **Spurious cointegration.** This model tests one pair at a time; it does
   not Bonferroni-correct across an N-pair universe. The orchestration layer
   is responsible for multiple-testing discipline.
2. **Structural breaks.** Even the Kalman variant assumes Gaussian
   incremental drift; discrete events (mergers, spin-offs) violate the
   random-walk-β prior. Filter on corporate-action calendars upstream.
3. **Asymmetric mean reversion.** Symmetric `s_in` / `s_out` thresholds
   assume the spread reverts at the same rate from both sides. Real pairs
   often do not (asymmetric borrow / short-squeeze risk).
4. **Carry costs.** Borrow fees, dividends on the short, and financing on
   the long are ignored. Spreads with low σ and long half-lives can have
   carry > alpha.
5. **Regime dependence.** Cointegration weakens during stress (correlations
   → 1 kills idiosyncratic dispersion) — exactly when leverage is most
   dangerous. The rolling EG t-statistic diagnostic helps catch this.
6. **Survivorship and look-ahead.** yfinance returns only currently-listed
   tickers. Backtests on pairs that include delisted firms must source
   point-in-time data externally.
7. **Critical-value approximation.** We use asymptotic MacKinnon (1996)
   critical values rather than the response-surface p-values from
   `statsmodels.tsa.stattools.coint`. The rejection thresholds are exact
   asymptotically and conservative for `T = 250-500`.

## Notes on the test suite

- Unit tests (`src/models/cointegration_pairs/tests/`) use synthetic
  cointegrated pairs and AR(1) paths with *known* parameters, so the
  estimators can be checked against ground truth (β, φ, half-life) within
  noise.
- The integration test (`tests/integration/test_cointegration_pairs.py`)
  runs the full `fetch_data → calibrate → predict → validate` pipeline
  against live yfinance for the textbook KO / PEP pair. The test skips
  automatically when yfinance is unavailable.
