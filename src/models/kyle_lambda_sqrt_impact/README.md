# Kyle Lambda and Square-Root Impact (Model 15)

Layer 5 — Execution / Transaction-cost.

Full specification:
[../../../models/layer5_execution/15_kyle_lambda_sqrt_impact.md](../../../models/layer5_execution/15_kyle_lambda_sqrt_impact.md)

This package combines two complementary impact models:

1. **Kyle (1985) linear-impact equilibrium** — closed-form
   `lambda = sigma_v / (2 sigma_u)` plus an empirical OLS estimator
   `r_t = lambda * sign(C - O) * V_t + eps_t` with HC1 robust standard
   errors. Captures information-driven price impact and is the right
   small-order regime.
2. **Bouchaud-style square-root impact law** — `Delta P / P =
   Y sigma sqrt(Q / V)`, the empirically dominant pre-trade-cost law for
   metaorders. Captures mechanical, latent-liquidity impact and is the
   right large-order regime.

`KyleSqrtImpact.predict` returns the combined Algorithm-D estimate:
linear Kyle below the crossover `Q* = Y^2 sigma^2 / (lambda^2 V)`, the
sqrt law above it. Sign of impact follows the parent-order side.

## Public surface

- `KyleSqrtImpact` — `BaseModel` subclass, registered as
  `"kyle_lambda_sqrt_impact"`.
- `calibrate(daily_bars, frequency="daily", ...)` — entry point returning a
  `CalibrationResult`. Wraps `calibrate_daily` and `calibrate_intraday`.
- `KyleLambdaFit`, `KyleInputs`, `SqrtImpactPrediction`,
  `CombinedImpactPrediction` — boundary dataclasses.
- Pure-math helpers: `kyle_equilibrium_lambda`, `kyle_equilibrium_beta`,
  `compute_signed_volume`, `compute_log_returns`, `ols_slope_no_intercept`,
  `hc1_se_no_intercept`, `r_squared_no_intercept`, `wald_ci_95`,
  `sqrt_law_impact`, `crossover_size`, `linear_impact`,
  `participation_ratio_exceeds_threshold`, `realized_daily_vol`, `adv`.

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.kyle_lambda_sqrt_impact import KyleSqrtImpact

provider = YFinanceProvider(cache=None)
model = KyleSqrtImpact(
    ticker="SPY",
    daily_period="120d",
    daily_lookback=120,
    parent_order_shares=10_000.0,
    side=1,
)

inputs = model.fetch_data(provider)
model.calibrate(inputs)
risk = model.predict(inputs)
print(risk.value, "bps expected impact")
print(risk.metadata["regime"])  # "linear" or "sqrt"
```

## Refit frequency

- Daily Kyle `lambda`: weekly on a 60- or 120-day rolling window.
- Intraday Kyle `lambda` (5-minute bars): weekly on 30-60 days of bars.
- `Y` prefactor: held at the literature prior (1.0) unless execution data
  becomes available.
- `V` (20-day ADV) and `sigma` (20-30 day realized): refit daily inside
  `predict`.

## Known limitations (from spec)

1. **Trade-sign proxy noise.** Without TAQ tick data, `sign(C - O)` is
   only ~55% accurate at daily frequency, biasing `lambda_hat` downward.
2. **Extrapolation cap.** The square-root law is calibrated on
   `Q / V <= 0.10`. `predict` flags `extrapolation_warning = True` above
   that threshold.
3. **Regime dependence.** `lambda` doubles or more during stress days;
   static estimates underprice crisis-day liquidity. Validate
   `r_squared > 0` and `lambda > 0` before trusting the estimate.
4. **Propagator kernel out of scope.** The Bouchaud-Gefen-Potters-Wyart
   propagator model is documented in the spec but cannot be calibrated
   from yfinance bar data.
