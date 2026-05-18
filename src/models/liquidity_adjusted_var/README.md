# Liquidity-Adjusted VaR — FRTB Horizons + BDSS Spread Cost

Implementation of Model 20 in the multi-agent quant ecosystem. Combines a
historical-simulation price-risk VaR with the Bangia-Diebold-Schuermann-
Stroughair (BDSS) spread component, FRTB liquidity-horizon-scaled ES, and
an Almgren-Chriss execution-cost overlay. Portfolio-level: one instance
owns a `{ticker -> dollar position}` dictionary and produces a single
`RiskMetric` plus a per-name diagnostics block.

**Spec:** [`models/layer6_risk/20_liquidity_adjusted_var.md`](../../../models/layer6_risk/20_liquidity_adjusted_var.md)

## Public API

Imported from `src.models.liquidity_adjusted_var`:

| Symbol | Kind | Purpose |
|---|---|---|
| `LiquidityAdjustedVaR` | class | `BaseModel` implementation; registered as `"liquidity_adjusted_var"`. |
| `Position` | dataclass | Signed dollar position in one ticker. |
| `TickerLiquidityStats` | dataclass | Calibrated per-ticker liquidity stats (ADV, spread, FRTB bucket, tier, η). |
| `LiquidityVaRInputs` | dataclass | `fetch_data` output: positions + returns panel + ticker stats. |
| `LVaRResult` | dataclass | Structured pipeline output (price VaR, FRTB ES, spread cost, AC cost, L-VaR, per-name diagnostics). |
| `calibrate` | function | Build per-ticker stats from prices / volumes / fundamentals panels. |
| `compute_l_var` | function | Pure pipeline: `LiquidityVaRInputs -> LVaRResult`. |
| `historical_var`, `historical_es` | function | Quantile-based VaR and tail-mean ES on a loss vector. |
| `frtb_horizon_scaled_es` | function | BCBS d457/d518 liquidity-horizon ES combination. |
| `bangia_liquidity_cost` | function | Half-spread × \|position\|. |
| `almgren_chriss_eta`, `almgren_chriss_simple_cost` | function | Simplified linear-impact execution cost. |
| `liquidation_horizon_days` | function | `\|shares\| / (κ · ADV)`, floored at 1d. |
| `assign_frtb_bucket`, `market_cap_bucket_days`, `snap_to_frtb_endpoint` | function | Bucketing into the FRTB endpoint table (10/20/40/60/120 days). |
| `tier_for_adv_ratio` | function | Internal tier 1-4 from ADV-to-position ratio. |
| `relative_spread_from_bid_ask`, `fallback_spread_from_high_low` | function | Spread sourcing with H/L proxy fallback. |
| `concentration_ratio`, `horizon_coverage_violations`, `stress_spread_l_var` | function | Spec validation block 3-5 diagnostics. |
| `compute_arithmetic_returns` | function | Wide-panel close → daily-return preprocessing. |
| `FRTB_HORIZON_ENDPOINTS`, `FRTB_T_BASE` | constant | Regulatory horizon table. |

## Usage

```python
from src.core.data_provider import YFinanceProvider
from src.models.liquidity_adjusted_var import LiquidityAdjustedVaR

provider = YFinanceProvider()
model = LiquidityAdjustedVaR(
    positions={"SPY": 1_000_000.0, "AAPL": 500_000.0, "MSFT": 500_000.0},
    alpha=0.975,            # FRTB level; 0.99 also supported
    max_participation=0.15, # 15% of ADV per day
    history_period="2y",
)

inputs = model.fetch_data(provider)
calibration = model.calibrate(inputs)
risk = model.predict(inputs)          # RiskMetric: .value is dollar L-VaR
diagnostics = model.validate(inputs)  # concentration, horizon coverage, stress

print(risk.value, risk.metadata["es_frtb"], risk.metadata["liquidity_cost_spread"])
```

`predict` returns a single `RiskMetric` with `ticker == "PORTFOLIO"`,
`metric_name == "liquidity_adjusted_var"`, and `value` equal to the
headline L-VaR in dollars (`ES_FRTB + Bangia spread cost`). The
`metadata` block carries the spec's STEP 8 output surface (price VaR,
FRTB ES, spread cost, AC cost, gross / net exposure, and per-name
diagnostics).

## Refit frequency

Class attribute: `refit_frequency="daily"`. Spec cadence:

- **Daily**: full L-VaR recompute with rolling 60-day ADV and current
  bid/ask. The class is registered at this frequency.
- **Daily**: FRTB bucket assignment refreshed if a ticker's market cap
  crosses 10B / 2B boundary.
- **Weekly**: tier-assignment review; tickers moving between tiers
  trigger desk-level alerts.
- **Quarterly**: Almgren-Chriss `ac_impact_coef` recalibrated against TCA
  data (out of scope for a yfinance-only pipeline).
- **Annual**: FRTB horizon endpoint table reviewed against regulator
  updates.

## Known limitations (from spec)

1. **ADV is a poor stress proxy** — volume often rises during a crisis
   while depth at touch collapses. Normal-time ADV understates true
   liquidation time in stress.
2. **Spread is not depth** — quoted spread says nothing about size
   available at that quote. A $0.01 NBBO on a $50 stock with 100-share
   depth is illiquid for a $10M position.
3. **Almgren-Chriss is linear-impact** — real impact follows a sub-linear
   (square-root) law; AC overestimates cost for medium-large orders and
   underestimates for very large ones. The `ac_impact_coef` heuristic
   should be replaced with a TCA-fitted η in production.
4. **Crowded-trade contagion not captured** — simultaneous unwind by
   many participants is much costlier than individual ADV would suggest.
5. **yfinance bid/ask is end-of-day** — `info["bid"]` / `info["ask"]`
   are stale NBBO snapshots when queried after the close; for intraday
   L-VaR, replace with a live feed.
6. **Bucket discontinuities (FRTB)** — a stock crossing $10B market cap
   moves from 10d to 20d horizon, doubling its capital contribution
   overnight. Smoothing is operationally helpful.
7. **Procyclical spread feedback** — when L-VaR rises in stress,
   positions are reduced, widening spreads further, raising L-VaR
   further.
8. **Off-exchange flow excluded** — dark-pool and retail-internalized
   volume is not in yfinance volume; ADV-based horizons are biased
   upward (slower) for names with heavy off-exchange flow.
9. **`almgren_chriss_simple_cost` units are loose** — the spec heuristic
   `η · x₀² / T` doesn't deliver a clean dollar number without a TCA
   recalibration; the value is reported in `LVaRResult.liquidity_cost_ac`
   but the headline L-VaR uses the dimensionally clean BDSS spread cost
   instead.

## Implementation notes

- **Returns convention:** arithmetic decimal daily returns (not percent).
  VaR / ES land in dollars when multiplied by the dollar-position
  vector — matches the additive BDSS spread cost without extra rescaling.
- **FRTB combination:** cumulative endpoints `(10, 20, 40, 60, 120)`,
  base horizon `T = 10`. Per the spec's STEP 5, the bucket-j ES is
  computed on the 1-day historical-sim panel restricted to tickers whose
  assigned FRTB bucket is `>= LH_j`, then scaled by `sqrt((LH_j -
  LH_{j-1}) / T)` and combined via L2 norm.
- **Bucket assignment:** `max(market_cap_bucket, ceil(position_horizon))`
  snapped **up** to the nearest endpoint. Missing market cap is treated
  as micro-cap (60-day default).
- **Spread sourcing:** prefers `info["bid"]` / `info["ask"]`; falls back
  to the spec's `(High/Low - 1) × 0.25` proxy on the trailing 20-day
  window when quotes are zero or missing. The `spread_source` field on
  `TickerLiquidityStats` records which path was taken.
- **Tier table:** ADV-to-position ratio in trading days at 100%
  participation: `<= 0.05` is tier 1, `<= 0.5` is tier 2, `<= 2.0` is
  tier 3, else tier 4. Matches the spec table examples (index futures,
  mid-cap, small-cap, exotic).

## Open SHARED-CHANGE-REQUEST

None. The implementation is self-contained on the public
`DataProvider` surface (`fetch_prices` for OHLCV bars,
`fetch_fundamentals` for bid/ask and market cap).
