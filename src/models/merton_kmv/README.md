# Merton-KMV Structural Credit Model

Implementation of Model 03 in the multi-agent quant ecosystem. Treats equity
as a European call on firm-asset value and infers distance-to-default, PD,
and a theoretical credit spread from observable market cap, equity vol, and
balance-sheet debt.

**Spec:** [`models/layer2_structural/03_merton_kmv.md`](../../../models/layer2_structural/03_merton_kmv.md)

## Public API

Imported from `src.models.merton_kmv`:

| Symbol | Kind | Purpose |
|---|---|---|
| `MertonKMV` | class | `BaseModel` implementation; the model registered as `"merton_kmv"`. |
| `MertonInputs` | dataclass | Bundle of fetched inputs (`fetch_data` output). |
| `MertonResult` | dataclass | Full structural snapshot (DD, PD, spread, quality). |
| `KMVSolution` | dataclass | Output of the iterative KMV solve. |
| `BalanceSheetSnapshot` | dataclass | ST/LT debt + cash, fed into the default point. |
| `calibrate` | function | `MertonInputs -> CalibrationResult`. |
| `iterative_kmv` | function | Core KMV iteration on `(V, sigma_V)`. |
| `joint_solve` | function | One-shot Newton-Raphson on the 2-equation system. |
| `merton_call_price`, `merton_d1_d2` | function | Black-Scholes building blocks. |
| `solve_for_v_given_e` | function | Newton inversion of BS for `V` given `E`. |
| `distance_to_default`, `prob_default` | function | DD and PD under either drift. |
| `credit_spread` | function | `-log(1 - PD_Q * LGD) / T`. |
| `default_point` | function | KMV default point: `ST + w * LT`. |
| `equity_vol_from_asset_vol` | function | Ito-lemma equity/asset vol link. |
| `realized_equity_volatility` | function | Annualized std of daily log returns. |
| `estimate_drift_from_assets` | function | `mu = mean(diff(log V)) * 252`. |
| `classify_credit_quality` | function | DD -> {IG, HY, distressed, default_imminent}. |
| `cap_struct_arb_stance` | function | Model-vs-market CDS divergence label. |
| `compute_merton_result` | function | Assemble `MertonResult` from inputs + solution. |
| `norm_cdf` | function | Standard normal CDF (scipy-free). |

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.merton_kmv import MertonKMV

provider = YFinanceProvider()
model = MertonKMV(ticker="GS", horizon_years=1.0, history_days=252)

inputs = model.fetch_data(provider)
calibration = model.calibrate(inputs)   # iterative KMV solve
signal = model.predict(inputs)           # Signal with DD/PD/spread metadata
diagnostics = model.validate(inputs)     # residuals, vol ratio, credit quality

print(signal.direction, diagnostics["dd_physical"], diagnostics["credit_spread_bps"])
```

## Refit frequency

- `sigma_E` and the KMV iteration: **daily** (252-day rolling window).
- Default point `D`: refreshed on each quarterly balance-sheet release.
- Drift `mu`: monthly (3-year window); the default `drift_mode="risk_free"`
  sets `mu = r` per the spec's pragmatic recommendation.
- DD-to-EDF empirical mapping: not currently implemented (this build returns
  the Gaussian `PD = N(-DD)` directly).

## Known limitations (from spec)

1. **Single-bullet maturity** with `T = 1y`; no Geske-style multi-bullet.
2. **Gaussian asset returns** — the Gaussian `PD = N(-DD)` understates tail risk
   relative to an empirical EDF curve.
3. **Continuous monitoring assumption** — default only at `T`, no first-passage
   (Black-Cox) variant.
4. **Financial firms** — the equity-as-call analogy breaks down for banks /
   insurers. Either skip or apply a sector-specific Bank-Merton variant.
5. **Negative-book-value firms** can produce `V < D` and break the Newton step.
6. **No empirical EDF mapping** — needs a default-event database (not in scope
   for this implementation).

## Open SHARED-CHANGE-REQUEST

```
SHARED-CHANGE-REQUEST
File: src/core/data_provider.py
Reason: Merton-KMV needs a structured balance-sheet feed (short-term debt,
        long-term debt, cash) per quarter. Today we extract these from the
        `info` dict via `fetch_fundamentals`, which is partially populated
        and inconsistent across tickers. yfinance exposes `Ticker.balance_sheet`
        (and `quarterly_balance_sheet`) — a first-class method on `DataProvider`
        would let the model stop scraping `info` and would let other Layer 2
        credit/leverage models share the same path.
Proposed change:
    Add an abstract method on `DataProvider`:
        fetch_balance_sheet(ticker: str, frequency: Literal["annual", "quarterly"]
            = "quarterly") -> pd.DataFrame
    Implement in `YFinanceProvider` via `yf.Ticker(t).quarterly_balance_sheet`
    (with `balance_sheet` fallback). Use `InMemoryProvider` keying like the
    other fetch_* methods.
Workaround until merged:
    `src/models/merton_kmv/model.py::_resolve_balance_sheet` consumes the
    `info` dict and falls back to a 30/70 ST/LT split of `totalDebt` when
    component fields are missing. This is good enough for the integration
    test, but downstream PD numbers can shift several bps when the split is
    wrong. Names without ANY debt fields will RuntimeError on fetch_data().
```
