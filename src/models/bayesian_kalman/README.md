# Bayesian Hierarchical and Kalman (Model 12)

Layer 4 — Signals & ML. See the full spec at
[`models/layer4_signals_ml/12_bayesian_kalman.md`](../../../models/layer4_signals_ml/12_bayesian_kalman.md).

This package implements three closely-related estimators from the spec under
one `BayesianKalman` model class, dispatched on a `mode` parameter:

| `mode`              | Estimator                                          | `predict()` returns                  |
| ------------------- | -------------------------------------------------- | ------------------------------------ |
| `"kalman_beta"`     | Linear-Gaussian state-space, MLE-tuned `(Q, R)`    | `Forecast` of latest beta + 1-σ band |
| `"hierarchical"`    | Conjugate Gibbs (Normal-Inverse-Wishart)           | `Forecast` for the chosen ticker     |
| `"dynamic_factor"`  | Stock-Watson DFM via EM (PCA-initialized)          | `RiskMetric` summarizing DFM vol     |

## Public surface

```python
from src.models.bayesian_kalman import (
    BayesianKalman,
    BayesianKalmanInputs,
    KalmanFit,
    HierarchicalFit,
    DynamicFactorFit,
    # Pure-math entry points (re-exported from signal.py):
    time_varying_beta_filter,
    kalman_filter,
    kalman_smoother,
    kalman_log_likelihood,
    mle_time_varying_beta,
    hierarchical_gibbs,
    hierarchical_posterior_mean,
    dynamic_factor_em,
    ljung_box_pvalue,
    standardized_innovations,
    innovation_normality_kurtosis,
    calibrate,
)
```

## 10-line usage example

```python
from src.core.data_provider import YFinanceProvider
from src.models.bayesian_kalman import BayesianKalman

provider = YFinanceProvider()
model = BayesianKalman(ticker="AAPL", market_ticker="SPY", mode="kalman_beta", period="2y")
data = model.fetch_data(provider)
result = model.calibrate(data)
forecast = model.predict(data)
print(f"AAPL beta: {forecast.value:.3f} [{forecast.lower:.3f}, {forecast.upper:.3f}]")
diagnostics = model.validate(data)
print(f"Ljung-Box innovation p-value: {diagnostics['ljung_box_pvalue']:.3f}")
```

## Refit frequency

`refit_frequency = "weekly"`. The spec sets:

- Kalman hyperparameters `(Q, R)`: re-tune *monthly* by MLE on the last ~2 years.
- Hierarchical posterior: re-sample *weekly* (or monthly).
- Dynamic factor EM: re-fit *monthly*; filter daily.

`"weekly"` is the most conservative across the three modes; the orchestration
layer can call `calibrate()` more often if desired.

## Known limitations (from the spec)

1. **Linear-Gaussian assumption.** Innovations show fat tails for real-world
   returns — switch to Student-t observation noise for trading-grade use.
2. **Q / R identifiability.** Only the *ratio* is identified on short series;
   MLE on a 2-year window may be noisy. Consider pinning one variance by
   domain knowledge for production fits.
3. **Filter divergence.** Joseph-form covariance update is used to guard
   against loss of positive-definiteness.
4. **Sampler pathologies.** The hierarchical path uses a conjugate Gibbs
   sampler (not NUTS) to avoid PyMC as a dependency. Funnel pathologies in
   the scale parameter are mitigated by the conjugate Inverse-Wishart prior.
5. **Stationarity.** The Kalman beta path uses a random-walk transition
   (`F = I`). An AR(1) with `|rho| < 1` is safer for long horizons but
   adds an extra hyperparameter to identify.
6. **yfinance survivorship bias** affects all modes that fit hierarchical
   means or factor models on the cross-section.
7. **Computation.** The hand-rolled Gibbs sampler is single-threaded Python
   — fine for ~10 assets × few hundred observations, slow for ~1000 assets.
   For larger panels use ADVI / NUTS via PyMC (not bundled here).

## Coordination

No shared-core changes required as of this implementation. The
`DataProvider` interface in `src/core/data_provider.py` is sufficient for
the model's `fetch_data` path.
