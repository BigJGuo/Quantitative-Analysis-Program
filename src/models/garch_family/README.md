# GARCH Family — Conditional-Volatility Model

Implementation of Model 08 in the multi-agent quant ecosystem. Fits one of
{GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1)} with Gaussian or standardized
Student-t innovations by maximum likelihood. Produces one-step and
multi-horizon conditional-variance forecasts plus the full spec validation
block (Ljung-Box, ARCH-LM, sign-bias, Mincer-Zarnowitz, VaR / Kupiec, QLIKE).

**Spec:** [`models/layer4_signals_ml/08_garch_family.md`](../../../models/layer4_signals_ml/08_garch_family.md)

## Public API

Imported from `src.models.garch_family`:

| Symbol | Kind | Purpose |
|---|---|---|
| `GARCHModel` | class | `BaseModel` implementation; registered as `"garch_family"`. |
| `GARCHInputs` | dataclass | Bundle of fetched inputs (`fetch_data` output). |
| `GARCHFit` | dataclass | MLE output: parameters, sigma² path, standardized residuals, convergence info. |
| `GARCHParams` | dataclass | Estimated `(omega, alpha, beta, gamma, nu, mu)`. |
| `GARCHForecastResult` | dataclass | Multi-horizon variance / vol forecast. |
| `calibrate` | function | `pd.Series -> CalibrationResult` MLE entry. |
| `compute_sigma2_path` | function | Forward-pass the variance recursion for a given spec / params. |
| `negloglik` | function | Negative log-likelihood (Gaussian or standardized Student-t). |
| `forecast_variance` | function | One- and multi-step variance forecasts. |
| `value_at_risk` | function | Conditional one-step VaR under fitted innovation distribution. |
| `annualize_vol_pct` | function | Daily percent vol → decimal annualized vol. |
| `ljung_box`, `arch_lm_test`, `sign_bias_test` | function | Residual diagnostics. |
| `mincer_zarnowitz_regression` | function | Realized-vs-predicted variance regression. |
| `kupiec_pof_test`, `qlike_loss` | function | VaR backtest and forecast-loss metric. |
| `normal_cdf`, `normal_ppf` | function | Standard-normal CDF / inverse (scipy-free). |
| `student_t_logpdf`, `student_t_ppf` | function | Standardized Student-t pdf / inverse CDF. |
| `expected_abs_z_gaussian`, `expected_abs_z_student_t` | function | `E\|z\|` under each innovation distribution. |

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.garch_family import GARCHModel

provider = YFinanceProvider()
model = GARCHModel(
    ticker="SPY",
    spec="GARCH",           # or "GJR", "EGARCH"
    distribution="Student-t",  # or "Gaussian"
    mean_model="Constant",     # or "Zero"
    history_period="10y",
)

inputs = model.fetch_data(provider)
calibration = model.calibrate(inputs)
risk = model.predict(inputs)        # RiskMetric with annualized vol + VaR
diagnostics = model.validate(inputs) # full spec validation block

print(risk.value, risk.metadata["sigma_daily_pct"], risk.metadata["forecast_vols_pct"])
```

`predict` returns a single `RiskMetric` whose `.value` is the **annualized
one-step daily volatility** (decimal). The `metadata` block carries the
spec's full output surface: daily-pct vol, multi-horizon variance / vol
forecasts, VaR at the configured confidence level, fitted parameters, and
persistence.

## Refit frequency

- **Parameters:** monthly (`refit_frequency="monthly"` on the class).
  Weekly is acceptable in fast-moving markets; daily refits over-fit and
  destabilize parameter time series.
- **Spec choice (GARCH vs GJR vs EGARCH):** review annually via
  likelihood-ratio (GARCH vs GJR, nested) and AIC / BIC (GJR vs EGARCH,
  non-nested). The fit's `fit_metrics["aic"]` and `["bic"]` are populated.
- **Distribution `nu`:** re-estimated within the same MLE call at each
  refit; in stress periods, `nu` typically drops (tails fatten).

## Known limitations (from spec)

1. **Slow regime-shift response.** Persistence `alpha + beta ~ 0.98`
   implies effective memory of ~50 days; realized vol drops within hours
   while GARCH lags weeks.
2. **IGARCH degeneracy.** When `alpha + beta -> 1`, the unconditional
   variance is undefined and long-horizon forecasts collapse to the
   one-step value. The optimizer caps the constrained persistence at
   `0.999`; reported persistence near that ceiling should be reviewed.
3. **Single-day jumps.** Earnings, FOMC days, etc., are treated as
   persistent vol — they are not. A robust GARCH variant
   (Boudt-Croux-Laurent) is not implemented here.
4. **Asymmetry mis-specification.** Plain GARCH biases down-market vol
   downward; switch to GJR or EGARCH when the sign-bias diagnostic
   `sign_bias_joint_p` rejects.
5. **Distribution mis-specification.** Gaussian VaR is systematically too
   narrow in the tails. Student-t helps but tail estimates at α < 1%
   remain unreliable.
6. **Aggregation invariance.** GARCH(1,1) on daily data does not
   aggregate cleanly to weekly / monthly horizons.
7. **Daily-vol focus.** Intraday equivalents (HEAVY, HAR-RV) are not
   implemented here — see Model 9 for the realized-vol family.
8. **No structured forecast intervals.** The model returns point
   variance forecasts; analytic-confidence-interval estimation for
   sigma² (e.g. via the Hessian / sandwich estimator) is left for a
   future cut — Nelder-Mead does not produce a Hessian.

## MLE implementation notes

- **Optimizer:** Nelder-Mead simplex in pure `numpy` (no scipy). Typical
  convergence on 2500 obs of daily data: 200-500 iterations.
- **Reparameterization:** softmax-style for the positivity + stationarity
  constraints in GARCH / GJR (so `alpha + beta`, or `alpha + gamma/2 + beta`,
  is automatically in `(0, 1)`); `tanh` for `beta` in EGARCH; `omega = exp(*)`
  for GARCH / GJR; `nu = 2.5 + exp(*)` for Student-t (the floor at 2.5 keeps
  the second moment well-defined and avoids the `E|z|` singularity at
  `nu = 2`).
- **Returns scaling:** the spec follows the `arch` package convention of
  expressing returns in percent units (`100 * log(P_t / P_{t-1})`). This
  rescales `omega` from ~1e-8 to ~1e-2 and keeps the optimizer numerically
  well-behaved. All variance / volatility numbers reported by this module
  are therefore in percent-return units unless explicitly annualized
  (`annualize_vol_pct`).
- **Initial values:** `omega = 0.01 * sample_var`, `alpha = 0.05`,
  `gamma = 0.05`, `beta = 0.9`, `nu = 8` — the spec recommendation.

## Open SHARED-CHANGE-REQUEST

None for `src/core/*`. The spec's calibration recipe references the
`arch` Python package and `scipy.optimize`; the current build env has
neither, so we ship a from-scratch numpy MLE. If `scipy` and `arch` are
later added to the project's `pyproject.toml` (a packaging concern, not a
core-API concern), `calibration.py` can be migrated to a BFGS gradient
solve plus the `arch` package's reference distributions with no
public-API change.
