# Model 04 — Factor Models and PCA

Cross-sectional risk model for an equity universe. Decomposes the asset-return
covariance into a low-rank systematic part (`B F B^T`) plus diagonal specific
risk (`D`). Supports three calibration paths: statistical (PCA),
fundamental time-series (Fama-French), and fundamental cross-sectional (Barra).

The PCA path is the default and the one wired through to `fetch_data`. The
Fama-French and Barra paths are exposed as `calibrate()` modes and can be
called directly with externally-supplied factor returns / exposures.

**Layer:** 2 — Structural / cross-sectional.
**Spec:** [`models/layer2_structural/04_factor_models_pca.md`](../../../models/layer2_structural/04_factor_models_pca.md).
**Class:** `FactorModelPCA` (`src.models.factor_models_pca.model`).
**Registered as:** `"factor_models_pca"` in the global model registry.

## Public API

From `src.models.factor_models_pca`:

- `FactorModelPCA` — `BaseModel` subclass; the orchestration entry point.
- `calibrate(...)` — pure dispatch into PCA / Fama-French / Barra; returns a
  `CalibrationResult` whose `parameters["factor_fit"]` is a `FactorFit`.
- `pca_decompose`, `time_series_factor_regression`,
  `cross_sectional_factor_regression` — the three estimators as pure functions.
- `compute_full_covariance`, `portfolio_risk`, `residualize`,
  `residual_returns` — math used by downstream models (Avellaneda-Lee
  consumes `residual_returns`; the portfolio layer consumes `portfolio_risk`).
- `marchenko_pastur_upper_edge`, `variance_explained_curve`,
  `residual_autocorrelation`, `residual_max_cross_corr`, `ljung_box_pvalue` —
  diagnostics used in `validate()`.
- `diagonal_shrinkage` — Ledoit-Wolf-style shrinkage toward the diagonal
  with a user-supplied intensity.
- Dataclasses: `FactorFit`, `FactorInputs`, `PortfolioRisk`, `FactorMethod`.

## Usage (10 lines)

```python
from src.core.data_provider import YFinanceProvider
from src.models.factor_models_pca import FactorModelPCA

universe = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE"]
model = FactorModelPCA(universe=universe, K=3, method="PCA", lookback_days=252)
data = model.fetch_data(YFinanceProvider(cache=None))
model.calibrate(data)
risk = model.predict(data)            # RiskMetric: annualized portfolio vol
diag = model.validate(data)           # variance explained, MP edge, residual diagnostics
sigma_full = model.covariance()       # (N, N) covariance: B F B^T + D
neutral_alpha = model.residualize_signal(some_signal_vector)
```

## Refit frequency

| Parameter                          | Cadence (spec)      |
|------------------------------------|---------------------|
| PCA covariance (B, D, eigvalues)   | weekly              |
| Fama-French betas                  | monthly             |
| Barra exposures B_t                | daily               |
| Barra factor returns f_hat_t       | daily               |
| Specific-risk diagonal D           | weekly              |
| Shrinkage intensity α              | monthly             |

The class is declared with `refit_frequency = "weekly"` to match the
load-bearing PCA cadence. For Barra deployments, the orchestration layer
will need to invoke `calibrate()` daily.

## Known limitations (carried from spec)

1. **Regime drift.** Rolling estimation lags structural breaks by ~½ the
   window. Crisis-period loadings can flip sign (e.g., 2008 financials).
2. **PCA sign / rotation indeterminacy.** Adjacent recalibrations can flip
   the sign of any eigenvector. Downstream consumers must anchor against the
   prior week's `B` if sign stability matters.
3. **Missing factors load into residuals.** If the true model has K+1 factors
   and we fit K, the omitted factor's variance contaminates `D`. The
   `max_residual_cross_correlation` diagnostic catches this.
4. **Linear exposure assumption.** True size / momentum effects are concave;
   the Barra exposures here are linear in fundamentals.
5. **Diagonal specific-risk.** Sector-specific residual clustering breaks the
   diagonal-`D` assumption. Block-diagonal `D` is out of scope here.
6. **Survivorship bias.** yfinance only returns currently-listed tickers;
   universes built from current S&P 500 lists overstate factor premia and
   understate specific risk. Maintain a point-in-time universe externally.
7. **Fama-French factor construction.** Building SMB / HML from yfinance
   requires the full US cross-section, which yfinance isn't optimized for.
   Production deployments load Kenneth French's published factor series.
8. **Barra fundamentals.** The `_barra_fit` path expects pre-built daily
   exposures and (optionally) market caps. Fetching those at point-in-time
   from yfinance is outside the model's `fetch_data` scope today.

## Notes on the test suite

- Unit tests (`src/models/factor_models_pca/tests/`) use synthetic
  factor-structure fixtures so PCA loadings, Fama-French betas, and
  cross-sectional factor returns can be checked against ground truth (up to
  sign for PCA).
- The integration test (`tests/integration/test_factor_models_pca.py`)
  exercises the full `fetch_data -> calibrate -> predict -> validate`
  pipeline against live yfinance for the 10-sector ETF universe and asserts
  PC1 lands in the spec's expected variance-explained band.
