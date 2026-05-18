# Stress Tests (Model 19)

Layer 6 risk / capital model implementing three stress flavors against a
portfolio of dollar positions:

1. **Historical replay** of canonical crisis windows (Black Monday 1987,
   LTCM 1998, Lehman 2008, COVID 2020, etc.). For each crisis the engine
   pulls the per-ticker percentage change over the window, falling back to a
   sector-ETF proxy when the ticker pre-dates its IPO and to `^GSPC` scaled
   by the historical equity beta when no sector proxy is available.
2. **Hypothetical** board-level scenarios (Equity -20%, Rates +100bp,
   Combined "1-in-100", ...). Shocks are applied through a per-asset OLS
   beta map onto a small canonical factor set (equity, rates, vol, dxy,
   IG, HY).
3. **Reverse stress** — solve the closed-form min-Mahalanobis factor shock
   that produces a `loss_target = capital * loss_target_fraction` loss
   (default 30% of capital). Reports the shock in factor-sigma units and a
   plausibility flag at the spec's 4-sigma threshold.

Spec: [`models/layer6_risk/19_stress_tests.md`](../../../models/layer6_risk/19_stress_tests.md).

## Public API

- `StressTests(BaseModel)` — orchestration class. `fetch_data` -> `calibrate`
  -> `predict` -> `validate`, registered as `"stress_tests"`.
- `StressTests.run_full_report(data) -> StressReport` — single call that
  returns the full historical + hypothetical + reverse decomposition.
- `calibrate(asset_returns, factor_returns, capital)` — pure-function
  calibration; returns a `CalibrationResult` whose `parameters["fit"]` is
  a `StressFit`.
- Math primitives in `signal.py`: `window_factor_change`, `apply_proxy_fill`,
  `historical_replay_pnl`, `estimate_betas_ols`, `hypothetical_pnl`,
  `portfolio_factor_sensitivities`, `reverse_stress_shock`,
  `scenario_coverage_matrix`, `top_contributors`,
  `window_endpoint_sensitivity`, `is_plausible`.
- Types: `CrisisWindow`, `HypotheticalScenario`, `ScenarioPnL`,
  `ReverseStressResult`, `StressFit`, `StressInputs`, `StressReport`.

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.stress_tests import StressTests

provider = YFinanceProvider()
model = StressTests(
    positions={"AAPL": 1_000_000, "JPM": 500_000, "SPY": -500_000},
    capital=5_000_000,
    calibration_period="2y",
)
data = model.fetch_data(provider)
model.calibrate(data)
risk = model.predict(data)
print(risk.value, risk.metadata["worst_case_name"])
report = model.run_full_report(data)
```

## Refit frequency

- **Daily**: re-run scenario P&L on current positions.
- **Weekly**: refresh beta and sector-proxy mappings (`StressTests.calibrate`).
- **Quarterly**: review the crisis-window set; add new crises as they occur.
- **Annual**: regulatory CCAR scenario refresh.

## Known limitations (from the spec)

- Selection bias of historical scenarios — the next crisis may not resemble a
  past crisis (e.g. COVID's policy-response speed).
- Linearization error on convex/short-gamma books. The current implementation
  is delta-only via the beta map; vega and gamma are not modeled.
- Correlation breakdown in stress regimes — the factor covariance is sample
  estimated over the calibration window and embeds an optimistic dynamic.
- Sensitivity stability — a stock's beta to SPY rises in crashes; static
  sensitivities understate this.
- Survivorship of crisis tickers — names alive in 2008 that are now delisted
  cannot be replayed without a historical universe database.
- Reverse-stress combinatorics — the Mahalanobis-distance objective is a
  heuristic; specific low-probability vulnerabilities may be missed.

## Open SHARED-CHANGE-REQUESTs

None. The existing `DataProvider.fetch_prices(ticker, "max", "1d")` plus
local slicing handles the long-history requirement (e.g. `^GSPC` back to
1928 for the Black Monday 1987 window).
