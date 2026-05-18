# Model 05 — Cost-of-Carry and Discrete Dividends

No-arbitrage pricing model for equity-index futures and ETF baskets. Links
the futures price to the spot index via term-matched financing cost net of
discrete dividends paid between trade and expiry. Emits a basis-trade signal
(future rich / cheap / flat) and an optional parallel ETF
creation/redemption decision.

**Layer:** 2 — Structural / cross-sectional.
**Spec:** [`models/layer2_structural/05_cost_of_carry_dividends.md`](../../../models/layer2_structural/05_cost_of_carry_dividends.md).
**Class:** `CostOfCarry` (`src.models.cost_of_carry_dividends.model`).
**Registered as:** `"cost_of_carry_dividends"` in the global model registry.

## Public API

From `src.models.cost_of_carry_dividends`:

- `CostOfCarry` — `BaseModel` subclass; orchestration entry point.
- `calibrate(...)` — fits the cost band and projects forward dividends;
  returns `CalibrationResult` with `cost_params` and `dividend_stream`.
- `compute_result(inputs)` — pure end-to-end transform on a populated
  `CostOfCarryInputs`, returning a `CostOfCarryResult`.
- Math building blocks (all pure):
  `time_to_maturity`, `interpolate_rate`,
  `pv_dividend_stream` / `fv_dividend_stream`,
  `theoretical_futures_discrete`, `theoretical_futures_continuous`,
  `equivalent_continuous_yield`, `theoretical_futures_price`,
  `implied_repo_rate`, `basis`, `classify_basis_action`, `signal_strength`,
  `forward_dividend_strip`.
- ETF basket path: `compute_etf_basket_result`, `basket_nav`,
  `accrued_dividends`, `etf_premium`, `classify_etf_action`.
- Helpers: `aggregate_constituent_dividends`, `project_forward_dividends`,
  `build_default_expiry`.
- Dataclasses: `Dividend`, `DividendStream`, `RateCurve`, `BasketHolding`,
  `IndexConstituent`, `CostParameters`, `BasisDecomposition`,
  `CostOfCarryInputs`, `CostOfCarryResult`, `ETFBasketResult`.

## Usage (10 lines)

```python
from datetime import datetime, timezone
from src.core.data_provider import YFinanceProvider
from src.models.cost_of_carry_dividends import CostOfCarry, IndexConstituent, build_default_expiry

today = datetime.now(timezone.utc)
model = CostOfCarry(
    futures_ticker="ES=F", spot_ticker="^GSPC",
    expiry=build_default_expiry(today, months_out=3),
    constituents=(IndexConstituent("AAPL", 0.07, 1.5), IndexConstituent("MSFT", 0.07, 1.5)),
)
data = model.fetch_data(YFinanceProvider())
model.calibrate(data)
signal = model.predict(data)   # Signal with basis / implied_repo in metadata
```

## Refit frequency

| Parameter                       | Cadence                |
|---------------------------------|------------------------|
| Spot $S_t$, futures $F_t$       | real-time (1–5 sec)    |
| Financing rate $r$              | daily (market open)    |
| Announced dividends             | daily                  |
| Projected dividends             | weekly                 |
| Transaction-cost band $\kappa+\tau$ | weekly             |
| Index weights / divisor         | monthly / on rebalance |

The class is declared with `refit_frequency = "daily"` to cover the dividend
and band refresh; the orchestration layer triggers `fetch_data + predict` on
the intraday pulse and `calibrate` once per day.

## Known limitations (carried from spec)

1. **Special and one-time dividends.** Projection routines use trailing
   quarterly history and miss special distributions. Mitigation: feed a
   curated corporate-actions calendar via `dividend_stream_override`.
2. **Tax effects.** The model uses pre-tax dividends; foreign holders
   experience a 15–30% withholding bias. Mitigation: scale dividends by
   `(1 - τ_w)` before passing into the stream.
3. **Stock-loan spread.** `stock_loan_spread_bps` is a flat add-on; in
   reality the spread is name-dependent and varies with hard-to-borrow
   conditions.
4. **Continuous-yield approximation.** For `tau >= switch_threshold_years`
   the model uses the spec's first-order equivalent yield, which matches the
   discrete formula only to `O((PV/S)^2)`. The difference is sub-bp for
   broad indices but can be material on illiquid baskets near ex-date.
5. **Repo squeeze.** The term-interpolated rate misses quarter-end / year-end
   dealer-balance-sheet pressure. The `implied_repo` diagnostic flags these
   episodes when the implied rate exits the SOFR ±50 bp band.
6. **Index methodology changes.** Divisor recalculations and total-return
   vs price-return switches change the constituent-dividend → index-point
   mapping; `shares_factor` must be updated externally on each rebalance.
7. **Currency for non-USD indices.** USD `^IRX`/`^TNX` rates are wrong for
   FTSE / DAX / Nikkei futures; pass a `RateCurve` denominated in the local
   currency via a `dividend_stream_override` flow if needed.
8. **Cross-currency basis.** USD-denominated futures on foreign indices
   require the CIP-adjusted formula, not the simple cost-of-carry; this
   model uses the simple form.
9. **Liquidity at expiry.** The basis widens mechanically in the final
   ~week before expiry; downstream strategy code should roll positions
   ~10 days out rather than trade the basis at this point.

## Cross-model dependencies

This model is *upstream* of Model 01 (iNAV / Equity ETF Fair Value) and
Model 02 (Futures-Implied NAV): both consume the dividend stream and
financing rate produced here. There are no upstream dependencies — all
inputs come from `DataProvider`.
