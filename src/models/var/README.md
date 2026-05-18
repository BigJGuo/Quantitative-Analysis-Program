# Value-at-Risk — Portfolio Market-Risk Model

Implementation of Model 17 in the multi-agent quant ecosystem. Computes
portfolio-level **Value-at-Risk** under three engines — **parametric**
(variance-covariance, Gaussian), **historical simulation**, and **Monte
Carlo** (Gaussian generator). Reports 1-day and sqrt-h-scaled
horizon VaR plus a Kupiec / Christoffersen backtest with the Basel
traffic-light bucket.

**Spec:** [`models/layer6_risk/17_var.md`](../../../models/layer6_risk/17_var.md)

## Public API

Imported from `src.models.var`:

| Symbol | Kind | Purpose |
|---|---|---|
| `VaRModel` | class | `BaseModel` implementation; registered as `"var"`. |
| `VaRInputs` | dataclass | Bundle of fetched inputs (`fetch_data` output). |
| `VaRFit` | dataclass | Output of `calibrate`: VaR figures, P&L vector, sample moments. |
| `VaRBacktestResult` | dataclass | Output of `validate`: Kupiec / Christoffersen LR stats + Basel bucket. |
| `calibrate` | function | `(returns, positions, method, alpha, ...) -> CalibrationResult`. |
| `parametric_var` | function | Closed-form variance-covariance VaR. |
| `historical_var` | function | Empirical-quantile VaR with the P&L vector. |
| `monte_carlo_var` | function | Gaussian MC VaR; deterministic with `seed`. |
| `horizon_scale` | function | `sqrt(h)` rescaling of 1-day VaR. |
| `rolling_var_backtest` | function | Rolling-window VaR vs. realized-loss backtest. |
| `kupiec_pof_test` | function | Unconditional-coverage LR statistic + p-value. |
| `christoffersen_independence_test` | function | Serial-dependence LR statistic + p-value. |
| `basel_traffic_light` | function | `green` / `yellow` / `red` Basel bucket (`T = 250`, `alpha = 0.99`). |
| `normal_ppf`, `chi2_cdf` | function | scipy-free distribution primitives. |
| `simple_returns` | function | `pct_change().dropna()` wrapper. |
| `portfolio_pnl_vector` | function | `rets @ w_dollar` (P&L vector in dollars). |
| `align_positions` | function | Reorder a `positions` dict to match a column list. |

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.var import VaRModel

provider = YFinanceProvider()
model = VaRModel(
    positions={"SPY": 1_000_000, "QQQ": 500_000, "TLT": -250_000},
    method="historical",     # or "parametric" / "monte_carlo"
    alpha=0.99,
    lookback_days=500,
    horizon_days=10,
    history_period="3y",
)

inputs = model.fetch_data(provider)
calibration = model.calibrate(inputs)
risk = model.predict(inputs)        # RiskMetric: 1-day VaR in dollars
diagnostics = model.validate(inputs) # Kupiec / Christoffersen / traffic-light
print(risk.value, risk.metadata["var_horizon_dollar"], diagnostics["traffic_light"])
```

`predict` returns a single `RiskMetric` whose `.value` is the **1-day VaR
in dollars** (always non-negative). The `metadata` block carries:

- `var_horizon_dollar` — sqrt-h scaled VaR at `horizon_days`.
- `pnl_vector` — empirical (HS / parametric) or simulated (MC) portfolio P&L.
- `var_1d_pct_gross`, `var_horizon_pct_gross` — VaR as a fraction of gross
  exposure.
- `mu_p_dollar`, `sigma_p_dollar` — portfolio mean and standard deviation
  (parametric / MC; HS reports empirical equivalents in `pnl_mean` / `pnl_std`).
- `method`, `alpha`, `horizon_days`, `n_obs`, `n_assets`, `gross_exposure`,
  `portfolio_value`, `positions`.

`validate` runs a rolling-window backtest of length `backtest_window`
(default 250 days) over the available history beyond the fit window and
reports:

- `backtest_n` / `backtest_breaches` / `backtest_breach_rate`
- `kupiec_lr` / `kupiec_p`
- `christoffersen_lr` / `christoffersen_p`
- `traffic_light` — Basel bucket
- `var_path` / `loss_path` — for plotting

## Refit frequency

- **Daily (positions + VaR):** `refit_frequency="daily"` on the class.
  HS-VaR is recomputed every trading day with the rolling window updated
  by the `t-1` close.
- **Daily (parametric covariance):** the sample covariance is recomputed
  in the same call. For RiskMetrics-style EWMA (`lambda = 0.94`), wrap
  `parametric_var` with an EWMA preprocessor — this module ships the
  sample-covariance variant only.
- **Quarterly:** review the validation block; if Kupiec rejects or the
  traffic-light is red, escalate to a model review.

## Known limitations (from spec)

1. **Tail blindness.** VaR ignores everything beyond the quantile. The
   spec's recommended companion is Expected Shortfall (Model 18).
2. **Non-subadditivity.** `VaR(L_1 + L_2)` can exceed `VaR(L_1) +
   VaR(L_2)`. The model does not flag this; combine sub-book VaRs
   carefully.
3. **Gaussian parametric VaR** undershoots fat-tailed equity losses by
   factors of 2–5x at `alpha = 0.99`. Prefer HS for material option /
   credit exposure.
4. **Historical simulation regime lag.** The empirical CDF carries the
   last ~1–2 years of returns; in regime change (Aug 2007, Mar 2020) it
   lags reality by weeks. This module does **not** implement filtered
   historical simulation; the spec describes the standard fix
   (GARCH-standardize returns, re-scale by current-vol forecast) and
   that integration is left for a follow-up that wires this model to the
   Model 8 (GARCH) public surface.
5. **Sqrt-h horizon scaling.** Exact under iid Gaussian increments;
   wrong under volatility clustering and serial correlation. The
   `horizon_scale` helper applies the spec's iid recipe; document the
   simplification before reporting to regulators.
6. **Monte Carlo is Gaussian only.** The spec mentions Student-t and
   GARCH-filtered MC. This module ships the Gaussian generator;
   fat-tailed MC is an obvious extension via the existing `_safe_cholesky`
   path.
7. **Procyclicality.** Like every quantile-based risk metric, VaR
   contracts as vol falls and balloons in a stress event — this is a
   feature of the metric, not a bug in this code, but the spec's "August
   2007 quant quake" warning applies.
8. **Survivorship.** `yfinance` silently omits delisted names. The
   `fetch_data` step raises if any portfolio ticker is missing; verify
   the universe is complete before relying on the backtest.

## Implementation notes

- **Returns convention:** decimal simple returns (`pct_change`), per the
  spec's algorithm. The GARCH-family module's percent-units convention
  is *not* used here — portfolio P&L is `w_dollar @ r` and the resulting
  VaR is in dollars.
- **scipy-free:** `normal_ppf` is Acklam's rational approximation and
  `chi2_cdf` uses the regularized lower incomplete gamma function
  (series for small `x`, continued fraction for large `x`). Same
  approach as the GARCH module so behavior is consistent across the
  ecosystem.
- **Monte Carlo seeding:** `mc_seed` is plumbed through; identical
  seeds reproduce identical MC results across runs.
- **Cholesky safety net:** near-singular sample covariances (e.g.
  duplicated holdings) trigger a small ridge fallback inside
  `monte_carlo_var` rather than crashing.
- **VaR non-negativity clip:** when the drift term dominates the
  standard-deviation term (deeply long-biased + strong realized
  drift), the closed-form parametric formula can produce a tiny
  negative number. The module clips at zero — VaR is non-negative by
  construction.

## Open SHARED-CHANGE-REQUEST

None. The current `DataProvider.fetch_prices(ticker, period, interval)`
surface is sufficient for a portfolio of N tickers (one call per
ticker, joined on date). A future batch endpoint would speed up large
portfolios but isn't required.
