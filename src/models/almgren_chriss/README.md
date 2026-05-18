# Almgren-Chriss Optimal Execution

Layer-5 execution model. Closed-form mean-variance optimal liquidation
schedule for a parent order subject to linear permanent and temporary
market impact.

Full mathematical specification:
[models/layer5_execution/16_almgren_chriss.md](../../../models/layer5_execution/16_almgren_chriss.md).

## Public API

### Class

- `AlmgrenChriss(BaseModel)` — registered as `"almgren_chriss"`. Inherits
  the standard `fetch_data` → `calibrate` → `predict` → `validate`
  contract from `src.core.BaseModel`.

### Schedule builders

- `optimal_schedule(problem, params, *, discretization)` — Algorithm A
  (continuous closed form) or Algorithm B (discrete difference equation
  with Newton-solved `κ̃`).
- `vwap_shaped_schedule(problem, params, *, volume_profile)` —
  Algorithm C. Volume-shaped per-slice impact `η_k = η_0 / u_k`; solved
  as a tridiagonal linear system.
- `multi_asset_schedule(problem, *, sigma_cov_daily, eta_diag,
  gamma_diag, ...)` — Algorithm D. Basket liquidation via
  eigendecomposition of `η^{-1/2} Σ η^{-1/2}`.
- `efficient_frontier(X, T, params, *, lambdas)` — sweep `λ` to trace
  the `(V[C], E[C])` curve.

### Calibration

- `calibrate(inputs, *, eta_rule, sigma_window, ...)` — fits `σ`, `η`,
  `γ` and returns a `CalibrationResult` with an `ImpactParams` payload.
- Lower-level: `realized_log_return_vol`, `average_daily_volume`,
  `eta_almgren_2005`, `eta_half_spread`, `gamma_from_eta`.

### Pure-math primitives

- `compute_kappa(sigma_d, eta, lam)` — `κ = sqrt(λ σ² / η)`.
- `inventory_path_continuous(X, T, N, kappa)` — `x_k`.
- `inventory_path_discrete(X, T, N, kappa_continuous)` — Newton-fitted
  `κ̃` plus `x_k`.
- `expected_temporary_cost`, `cost_variance_closed_form`,
  `permanent_cost` — textbook closed-form formulas.
- `sinh_ratio(a, b)` — overflow-safe `sinh(a)/sinh(b)`.

### Diagnostics

- `schedule_diagnostics(schedule)` — spec validation 1, 2, 7.
- `stress_schedule(problem, params, sigma_multiplier=3)` — spec
  validation 5.

### Dataclasses

`AlmgrenChrissInputs`, `ImpactParams`, `LiquidationProblem`,
`ExecutionSchedule`, `EfficientFrontier`, `EfficientFrontierPoint`,
`MultiAssetProblem`, `MultiAssetSchedule`.

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.almgren_chriss import AlmgrenChriss

model = AlmgrenChriss(
    ticker="SPY", X=5.0e5, T=1.0, N=50,
    side="sell", lam=1.0e-6,
)
provider = YFinanceProvider(cache=None)
inputs = model.fetch_data(provider)
model.calibrate(inputs)
risk = model.predict(inputs)            # RiskMetric: cost in bps
sched = model.schedule(inputs)          # full ExecutionSchedule
frontier = model.frontier(inputs)       # 7-point efficient frontier sweep
```

## Refit frequency

Per `BaseModel.refit_frequency = "daily"`:

- `σ` — daily, from a 30-day realized-vol window.
- `η` — weekly per ticker; the model defaults to the Almgren-2005 rule.
- `γ` — quarterly; defaults to `η / (10 · τ_halflife)` with
  `τ_halflife = 1 day`.
- `λ` — hand-tuned, reviewed monthly per the spec.
- Intraday volume profile (Algorithm C only) — weekly.

## Known limitations (per spec §10)

1. Linear impact only — concave (square-root) impact is more accurate
   for small orders; see Almgren (2003) extensions.
2. Constant `σ` over the horizon — ignores GARCH-style clustering.
3. No drift / alpha — purely defensive scheduling. For alpha-aware
   execution see Garleanu-Pedersen.
4. Constant `η` (single-asset Algos A/B); intraday variation handled
   via Algorithm C; regime variation not addressed.
5. No price-limit constraints; participation rate cap must be applied
   externally if needed (the schedule may otherwise prescribe rates
   above 25% ADV).
6. Mean-variance, not log-utility — disagreement is small in the
   calibrated `λ` range; large `λ` should be sanity-checked.
7. Permanent impact drops from optimization; the textbook model misses
   schedule-dependent information leakage.

For `κT > 30` the closed forms switch to asymptotic
(`x_t ≈ X exp(-κt)`, `E_temp ≈ η X² κ / 2`, `V ≈ σ² X² / (2κ)`) — see
`signal._SINH_OVERFLOW_THRESHOLD`.
