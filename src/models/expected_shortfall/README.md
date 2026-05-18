# Expected Shortfall (FRTB) — Tail Risk & Regulatory Capital

Implementation of Model 18 in the multi-agent quant ecosystem. Computes
Basel III FRTB 97.5% Expected Shortfall (ES) for a dollar-weighted equity
portfolio under four estimators: parametric Gaussian, parametric Student-t,
historical simulation, and multivariate-t Monte Carlo. Ships with the
Acerbi-Szekely Z1 / Z2 backtests and the coherence-axiom test suite from
Artzner-Delbaen-Eber-Heath (1999).

**Spec:** [`models/layer6_risk/18_expected_shortfall.md`](../../../models/layer6_risk/18_expected_shortfall.md)

## Public API

Imported from `src.models.expected_shortfall`:

| Symbol | Kind | Purpose |
|---|---|---|
| `ExpectedShortfall` | class | `BaseModel` implementation; registered as `"expected_shortfall"`. |
| `ESInputs` | dataclass | Bundle of fetched inputs (`fetch_data` output). |
| `ESFit` | dataclass | Estimated `(mu, Sigma, sigma_p)` + Student-t `(nu, loc, scale)`. |
| `ESResult` | dataclass | `(VaR, ES, tail_losses)` per spec output. |
| `ESMethod` | type | `Literal["parametric_normal", "parametric_t", "historical", "monte_carlo"]`. |
| `FRTB_ALPHA` | constant | `0.975` — Basel III FRTB default. |
| `calibrate` | function | `ESInputs -> CalibrationResult` entry. |
| `es_gaussian` | function | Closed-form Gaussian `(VaR, ES)`. |
| `es_student_t` | function | Closed-form Student-t `(VaR, ES)`. |
| `es_historical` | function | Empirical tail-mean ES. |
| `es_monte_carlo` | function | Multivariate-t simulator ES. |
| `acerbi_szekely_z1`, `acerbi_szekely_z2` | function | Tail backtests. |
| `fit_student_t` | function | 1-D Student-t MLE (scipy-free EM + grid). |
| `rolling_realized_tail_mean` | function | Diagnostic plot input. |
| `traffic_light_band` | function | Basel-style classification of ES/VaR ratio. |
| `apply_liquidity_scaling` | function | FRTB `sqrt(h_k/10)` per-ticker shock scaling. |
| `normal_pdf`, `normal_cdf`, `normal_ppf` | function | Standard-normal helpers (scipy-free). |
| `student_t_pdf_raw`, `student_t_cdf_raw`, `student_t_ppf_raw` | function | Raw (unit-scale) Student-t helpers. |

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.expected_shortfall import ExpectedShortfall

provider = YFinanceProvider()
model = ExpectedShortfall(
    positions={"SPY": 600_000.0, "AGG": 400_000.0},
    method="historical",         # or "parametric_normal" / "parametric_t" / "monte_carlo"
    alpha=0.975,                  # FRTB default
    lookback_N=500,               # >= 500 recommended at alpha=0.975
)

inputs = model.fetch_data(provider)
model.calibrate(inputs)
risk = model.predict(inputs)        # RiskMetric: .value is the dollar ES
diagnostics = model.validate(inputs) # Acerbi-Szekely Z1/Z2, breach rate, etc.

print(risk.value, risk.metadata["VaR"], risk.metadata["traffic_light"])
```

`predict` returns a single `RiskMetric` whose `.value` is the **one-day
97.5% Expected Shortfall in dollars**. `metadata` carries VaR, the ES/VaR
ratio, the realized fit parameters (`sigma_p`, optional `nu`, `loc`,
`scale`), the lookback window size, the breach count, and the
traffic-light band.

## Refit frequency

- **ES recomputation:** daily, with a rolling lookback (`refit_frequency =
  "daily"` on the class).
- **Model parameters (`nu` for the Student-t marginal, Monte-Carlo
  configuration, liquidity horizons):** quarterly.
- **PLA test:** annually for FRTB IMA eligibility.
- **Stressed-window reference:** updated whenever a period of greater
  severity emerges (handled at the orchestration layer; this model accepts
  any input window and does not select the stress window itself).

## Known limitations (from spec)

1. **Backtesting weakness.** At alpha=0.975 with `N=250`, only ~6 tail
   observations populate the empirical ES; statistical power to reject a
   mis-specified model is low. Acerbi-Szekely Z1/Z2 typically need 500+
   days; the default `lookback_N=500` is chosen accordingly.
2. **Sample-mean instability.** Empirical ES is the mean of a small number
   of order statistics; a single extreme observation can move it materially.
   Bootstrap intervals (not implemented here) are wide.
3. **Liquidity blindness.** Standard ES marks at posted prices. FRTB
   liquidity horizons can be injected via `liquidity_scalars` on the
   constructor, but this model does not estimate the horizons itself
   (see Model 20).
4. **Model risk on `nu`.** Parametric Student-t ES is more sensitive to
   the tail index than parametric VaR; small mis-specifications in `nu`
   produce material ES errors even when VaR is approximately right.
5. **Not elicitable.** ES is not elicitable as a single statistic
   (Gneiting 2011). The "jointly elicitable with VaR" Fissler-Ziegel
   (2016) scoring rule is not used here; the diagnostics block reports
   Acerbi-Szekely (the practitioner standard) instead.
6. **Procyclicality.** ES expands in vol-spike regimes, contributing to
   the forced-deleveraging dynamic. Inherited from VaR; not addressed.
7. **No copula support.** Marginals are assumed jointly Gaussian (or
   jointly multivariate-t with one shared `nu`) at the portfolio level.
   Asset-specific tail copulas (e.g. Gumbel for downside dependence) are
   not modeled here — see Layer 3 copula models.
8. **No stressed-ES blending.** The FRTB IMA capital charge combines
   non-stressed and stressed ES with a 0.5 weighting; this model produces
   the single-window ES that an orchestration layer would blend.

## Implementation notes

- **scipy-free.** The full Student-t pdf / cdf / ppf and the 1-D
  Student-t MLE are implemented in `signal.py` using `math.lgamma`, the
  regularized incomplete beta (Numerical Recipes 6.4 continued fraction),
  and an EM iteration on the scale-mixture-of-normals representation.
  This matches the rest of the codebase's scipy-free numerics policy.
- **Return convention.** Returns are **decimal simple returns**
  (`pct_change`), and dollar P&L = `returns @ dollar_positions`. ES and
  VaR are reported in **dollars of one-day loss**.
- **Sign convention.** `dollar_positions` carries sign — long positions
  positive, shorts negative. Losses are `-pnl`; under standard sign
  conventions, both VaR and ES are **positive numbers** representing
  losses.
- **Monte-Carlo sampler.** Multivariate-t via the Gaussian / chi-squared
  scale-mixture (`X = mu + Z/sqrt(g)` with `Z ~ N(0, Sigma)` and
  `g ~ chi2(nu)/nu`). Default `K = 100_000`, `nu = 5` per the spec; both
  configurable on the constructor.
- **Cholesky robustness.** If the empirical covariance is numerically not
  PSD, the diagonal is nudged by `1e-10 * trace / N` before Cholesky.

## Open SHARED-CHANGE-REQUEST

None. The spec references `scipy.stats` for Student-t fitting and PPF
work; the current build env has no `scipy`, so we ship from-scratch
numerics. If `scipy` is later added to `pyproject.toml`, the helpers in
`signal.py` can be migrated to `scipy.stats.t` / `scipy.optimize.minimize`
with no public-API change.
