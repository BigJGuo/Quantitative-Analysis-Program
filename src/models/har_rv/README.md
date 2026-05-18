# Model 09 — HAR-RV Realized Volatility

Corsi (2009) Heterogeneous Autoregressive model of realized volatility. A
three-component linear regression of realized variance on its daily, weekly,
and monthly lags. Captures the long-memory structure of realized variance
with four free parameters, no fractional integration, no MCMC.

**Layer:** 4 — Signals / ML (realized-vol forecasts).
**Spec:** [`models/layer4_signals_ml/09_har_rv.md`](../../../models/layer4_signals_ml/09_har_rv.md).
**Class:** `HARRVModel` (`src.models.har_rv.model`).
**Registered as:** `"har_rv"` in the global model registry.

## Public API

From `src.models.har_rv`:

- `HARRVModel` — `BaseModel` subclass; the orchestration entry point.
- `calibrate(...)` — pure HAR fit on a daily realized-variance Series;
  returns a `CalibrationResult` whose `parameters["har_fit"]` is a `HARFit`.
- `compute_daily_rv`, `compute_intraday_log_returns` — build daily RV
  (plus bipower variation and realized semivariances) from intraday OHLCV.
- `compute_garman_klass_rv`, `compute_yang_zhang_rv` — OHLC range-based
  RV proxies for the long-history fallback path (Strategy C in the spec).
- `aggregate_har_components`, `build_har_design_matrix` — daily/weekly/
  monthly aggregations and the regression design matrix.
- `ols_fit`, `newey_west_covariance`, `nw_optimal_lag`, `fit_har` —
  the regression itself with HAC standard errors.
- `forecast_one_step`, `latest_har_state`, `in_sample_forecasts` —
  forecasting helpers (lognormal correction in the log spec).
- `mincer_zarnowitz`, `qlike_loss`, `ljung_box_pvalue`, `arch_lm_pvalue`,
  `diebold_mariano_pvalue` — diagnostics used in `validate()`.
- Dataclasses: `HARFit`, `HARRVInputs`, `RVComponents`, `HARSpec`,
  `RVSource`.

## Usage (10 lines)

```python
from src.core.data_provider import YFinanceProvider
from src.models.har_rv import HARRVModel

model = HARRVModel(ticker="SPY", spec="log", horizon=1, annualize=True)
inputs = model.fetch_data(YFinanceProvider(cache=None))
model.calibrate(inputs)
forecast = model.predict(inputs)           # Forecast: annualized vol for tomorrow
diagnostics = model.validate(inputs)       # R^2, Mincer-Zarnowitz, Ljung-Box, ARCH-LM
print(forecast.value, forecast.metadata["rv_forecast_daily"])
```

## Refit frequency

| Parameter                                  | Cadence (spec)      |
|--------------------------------------------|---------------------|
| OLS coefficients `(c, β_d, β_w, β_m)`      | monthly             |
| Intraday RV update (daily 5-minute pull)   | daily after close   |
| Rolling sample window                      | 2-5 years of daily RV |
| Sampling-frequency review (`Δ` choice)     | annually            |
| HAR-CJ / semivariance-HAR extension review | quarterly           |

The class is declared with `refit_frequency = "monthly"`. The orchestration
layer pulls fresh daily RV every trading day (the data-fetch step is cheap)
but only re-estimates `(c, β_d, β_w, β_m)` monthly. Daily refits do not
improve out-of-sample QLIKE — HAR is parsimonious enough that monthly is
the empirical sweet spot.

## Known limitations (carried from spec)

1. **Intraday data dependence.** Full-power HAR requires intraday returns;
   yfinance caps 5-minute history at ~60 days. The model falls back to a
   Garman-Klass RV proxy on the long daily-OHLC history, which is biased
   downward 10-20% but preserves the persistence structure.
2. **Microstructure noise.** Below ~5-minute sampling, bid-ask bounce
   inflates RV. The model defaults to `5m`; check the volatility signature
   plot before lowering.
3. **Jumps treated as continuous variation.** Plain HAR-RV does not
   distinguish; jump days inflate forecasts for several days afterward.
   HAR-CJ would help but is not currently fitted by this model (the
   bipower-variation series is computed for diagnostic purposes only).
4. **Overnight gap.** Default convention adds the overnight squared return
   to the day's RV (Andersen-Bollerslev convention (a)). Switching the
   convention changes the forecast level materially.
5. **Half-trading days.** Shortened sessions produce structurally lower RV.
   They are not currently filtered out — the regression absorbs the lower
   level into the daily lag.
6. **Regime change.** HAR is linear and stationary; sudden volatility
   regime shifts (March 2020) produce persistent forecast errors that decay
   over weeks as the monthly average updates.
7. **Single-asset focus.** Univariate by design. Vector-HAR for realized
   covariance is out of scope.
8. **Coefficient identification.** The daily/weekly/monthly regressors are
   highly collinear by construction. Individual β's are unstable across
   recalibrations; only the persistence sum `β_d + β_w + β_m` and the
   overall fit are robust. Do not give economic weight to individual β's.
9. **Forecast horizon decay.** One-day-ahead forecasts are excellent;
   monthly-horizon forecasts collapse to the sample mean. The `horizon`
   parameter uses Corsi's direct multi-step regression to limit error
   compounding but cannot fix the underlying horizon decay.
10. **Liquidity dependence.** Thinly-traded names have many zero-return
    5-minute intervals; switch to longer intervals (15m / 30m) or use the
    OHLC range estimator.

## Notes on the test suite

- Unit tests (`src/models/har_rv/tests/`) use synthetic AR(1)-in-log-RV
  fixtures so the persistence sum, R^2 magnitude, and design-matrix shapes
  can be checked against known DGPs. Worked-out checks against closed-form
  Garman-Klass values and bipower-variation identities cover the per-day
  estimators.
- The integration test (`tests/integration/test_har_rv.py`) exercises the
  full `fetch_data -> calibrate -> predict -> validate` pipeline against
  live yfinance for SPY. The R^2, persistence-sum, and Mincer-Zarnowitz
  bounds are deliberately loose to absorb year-over-year market-regime
  variation.

## Open SHARED-CHANGE-REQUESTs

None.
