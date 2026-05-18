# Model 01 — iNAV / Equity ETF Fair Value

Real-time fair-value model for an equity ETF. Decomposes the per-share value
into the live basket leg, the stale basket leg (delegated to Model 2 in
production), accrued cash, and accrued liabilities. Compares against the
secondary-market mid to emit a CREATE / REDEEM / FLAT signal for AP
arbitrage.

**Layer:** 2 — Structural / cross-sectional.
**Spec:** [`models/layer2_structural/01_inav_equity_etf.md`](../../../models/layer2_structural/01_inav_equity_etf.md).
**Class:** `INAVEquityETF` (`src.models.inav_equity_etf.model`).
**Registered as:** `"inav_equity_etf"` in the global model registry.

## Public API

From `src.models.inav_equity_etf`:

- `INAVEquityETF` — `BaseModel` subclass; the orchestration entry point.
- `calibrate(...)` — refreshes basket, seeds, and the cost band; returns
  `CalibrationResult`.
- `compute_inav_result(inputs)` — pure end-to-end transform on a populated
  `INAVInputs`, returning a full `INAVResult` decomposition.
- `compute_decomposition`, `compute_premium`, `classify_action`,
  `signal_strength` — building blocks reusable by Model 2 and downstream.
- Dataclasses: `BasketHolding`, `ConstituentQuote`, `CashLedgerSeed`,
  `CostParameters`, `INAVDecomposition`, `INAVInputs`, `INAVResult`.

## Usage (10 lines)

```python
from datetime import datetime, timedelta, timezone
from src.core.data_provider import YFinanceProvider
from src.models.inav_equity_etf import INAVEquityETF, BasketHolding, CashLedgerSeed

holdings = [BasketHolding("AAPL", 1000.0, "USD"), BasketHolding("MSFT", 800.0, "USD")]
seed = CashLedgerSeed(eod_nav=400.0, eod_shares_out=1_000_000.0,
                     eod_release_time=datetime.now(timezone.utc) - timedelta(hours=18),
                     prior_cash=0.0, prior_liabilities=0.0)
model = INAVEquityETF(etf_ticker="SPY", holdings=holdings, seed=seed)
data = model.fetch_data(YFinanceProvider(cache=None))
signal = model.predict(data)  # Signal with action / iNAV / premium in metadata
```

## Refit frequency

| Parameter                       | Cadence            |
|---------------------------------|--------------------|
| Basket composition `{n_i}`      | daily (pre-open)   |
| Cash / liabilities seeds        | daily (NAV strike) |
| Cost band `κ + τ`               | weekly             |
| Intraday `iNAV_t` recomputation | every 1–5 seconds  |

The class is declared with `refit_frequency = "intraday"` to cover the most
frequent of these; the orchestration layer triggers `calibrate()` daily and
`fetch_data + predict` on the intraday pulse.

## Known limitations (carried from spec)

1. **Holdings depth.** `DataProvider.fetch_holdings` exposes only `[symbol,
   weight]`. Production callers must supply the full share-count basket via
   the `INAVEquityETF(holdings=...)` constructor argument (typically loaded
   from the issuer's daily holdings JSON). See the SHARED-CHANGE-REQUEST in
   the implementation report.
2. **Stale international constituents.** When `(t - last_print) >
   staleness_threshold_seconds` a constituent is routed to the STALE leg of
   the decomposition. The fair-value of that leg is *not* re-estimated here
   — Model 2 (Futures-Implied NAV) is responsible for replacing it.
3. **Dividend pay-dates.** yfinance only exposes ex-dates; cash accrual
   credits the dividend at ex-date without modelling the T+30 pay-date lag.
4. **Foreign withholding tax.** Gross dividends are accrued; per-country net
   adjustments are not applied.
5. **Mid proxy.** The ETF mid is taken from `(High+Low)/2` of the last 1-min
   bar — a proxy for the true quote midpoint. Production needs an L1 feed.
6. **1-minute lag.** Free-tier yfinance bars lag the live market by 1–15
   minutes; the model is wired correctly but downstream latency-sensitive
   uses must swap in a direct feed.
