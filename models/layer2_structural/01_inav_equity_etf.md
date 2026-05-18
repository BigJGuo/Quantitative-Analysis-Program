# iNAV (Indicative NAV) and Equity ETF Fair Value

> Real-time fair-value model for equity ETFs that decomposes the ETF price into the sum of its constituents plus cash and accruals. Layer 2 — Structural / Cross-Sectional. Primary use: ETF primary-market arbitrage and authorized participant (AP) creation/redemption decisioning.

## Mathematical formulation

The Indicative Net Asset Value (iNAV) of an ETF at time $t$ is defined as the per-share fair value implied by the current market prices of its underlying constituents. For an ETF holding $N$ constituent securities indexed by $i$:

$$
\text{iNAV}_t \;=\; \frac{\displaystyle\sum_{i=1}^{N} n_i \cdot P_{i,t} \cdot \text{FX}_{i,t} \;+\; C_t \;-\; L_t}{S_t}
$$

where:

- $n_i$ — number of shares of constituent $i$ in the creation unit basket (held constant between rebalances).
- $P_{i,t}$ — last traded price of constituent $i$ at time $t$ in its local currency.
- $\text{FX}_{i,t}$ — spot FX rate converting constituent $i$'s local currency into the ETF's reporting currency (USD for US-listed ETFs).
- $C_t$ — cash held by the fund (T-bills, settlement cash, dividend receivables).
- $L_t$ — total liabilities (management fee accrual, expenses payable).
- $S_t$ — shares outstanding at time $t$ (changes only on creation/redemption settlement).

The exchange-disseminated iNAV (published by NYSE, Cboe, Nasdaq every 15–60 seconds) uses last-prints from the constituent primary exchanges. The ETF secondary-market mid $M_t$ trades around iNAV with a stochastic spread:

$$
M_t \;=\; \text{iNAV}_t \;+\; \epsilon_t
$$

where $\epsilon_t$ captures arbitrage friction, AP inventory effects, and (critically) iNAV staleness when constituents are not actively trading.

### Cash and dividend accrual

Between ex-dates and pay-dates, the ETF accrues a dividend receivable that must be added to NAV:

$$
C_t \;=\; C_0 \;+\; \sum_{i : t_i^{ex} \le t < t_i^{pay}} n_i \cdot d_i \;-\; \sum_{j} \text{Expense}_j
$$

where $d_i$ is the per-share cash dividend declared on constituent $i$ with ex-date $t_i^{ex}$ and pay-date $t_i^{pay}$.

### Arbitrage condition

Define the AP creation/redemption boundary. Let $\kappa$ be the per-share creation fee and $\tau$ be transaction costs in the basket. No-arbitrage between primary and secondary markets implies:

$$
\text{iNAV}_t - \kappa - \tau \;\le\; M_t \;\le\; \text{iNAV}_t + \kappa + \tau
$$

When $M_t > \text{iNAV}_t + \kappa + \tau$, the AP creates new shares (delivers basket, receives ETF, sells ETF at $M_t$). When $M_t < \text{iNAV}_t - \kappa - \tau$, the AP redeems (buys ETF at $M_t$, delivers ETF to issuer, receives basket, sells basket).

### Decomposition into observable and unobservable

Split constituents into a "live" set $\mathcal{L}_t$ (markets open, prices fresh) and a "stale" set $\mathcal{S}_t$ (markets closed or illiquid):

$$
\text{iNAV}_t \;=\; \underbrace{\frac{\sum_{i \in \mathcal{L}_t} n_i P_{i,t} \text{FX}_{i,t}}{S_t}}_{\text{live component}} \;+\; \underbrace{\frac{\sum_{i \in \mathcal{S}_t} n_i \tilde{P}_{i,t} \text{FX}_{i,t}}{S_t}}_{\text{stale / modelled component}} \;+\; \frac{C_t - L_t}{S_t}
$$

where $\tilde{P}_{i,t}$ is the fair-value estimate for stale constituents, handled by the futures-implied NAV model (see Model 2).

## Intuition and what this model addresses

The iNAV is the cleanest example of a structural model in equity markets. An ETF is contractually a wrapper around a basket: by the terms of the issuer's authorized participant agreement, the AP can exchange the basket for ETF shares (or vice versa) at end-of-day NAV. This contractual link enforces a no-arbitrage band on the secondary-market price.

Why the model exists:

1. **The published iNAV is stale.** Even for US-only ETFs, constituent prints can be seconds to minutes old during low-liquidity periods. For a fast-moving market, the published iNAV lags the true fair value.
2. **For international ETFs, iNAV is essentially frozen.** EFA (MSCI EAFE) and EEM (MSCI Emerging Markets) hold securities listed in Tokyo, Hong Kong, London, and other markets that are closed during US trading hours. The published iNAV uses last close prices from those markets and does not update with current information. The real fair value can drift several percent away from iNAV intraday.
3. **Market makers must internalize cash and dividend accrual.** A naive observer treats the ETF as worth the sum of constituent prices; a market maker tracks accrued dividends, T+2 cash settlement, and management fee accrual to the basis point.
4. **AP creation/redemption is the enforcement mechanism.** Without continuous AP arbitrage, the ETF price would untether from NAV. The model tells the AP when the arbitrage band is breached after costs.

The model addresses the failure of simple "ETF price = basket price" reasoning by adding the dividend/cash/expense ledger and by flagging when constituents are stale enough that a separate proxy model (Model 2) is required.

## Calibration approach

Most parameters are not statistically estimated — they are either contractually fixed (the basket composition $n_i$) or directly observed (constituent prices, FX rates). The genuine calibration steps are:

**1. Basket composition $\{n_i\}$.** Pulled from the issuer's published creation unit file (typically updated daily before market open). For yfinance use `Ticker.funds_data.top_holdings`, but this returns top-10 weights only; for full baskets pull the issuer's daily holdings JSON (iShares, Vanguard, State Street all publish these).

**2. Cash and liabilities $C_t, L_t$.** Reconstruct from end-of-day NAV released by the issuer the prior day, plus expected dividend ex/pay flows from each constituent's announced calendar:

$$
\hat{C}_t \;=\; C_{T-1} \cdot (1 + r \cdot \Delta t) \;+\; \sum_{i : t_i^{ex} \in (T-1, t]} n_i d_i
$$

where $r$ is the overnight rate from `^IRX` and $\Delta t$ is the elapsed business-day fraction.

**3. Expense accrual.** Daily expense ratio is $f/252$ where $f$ is the published expense ratio from `Ticker.info["annualReportExpenseRatio"]`. Accrued expense is $L_t = L_{T-1} + f/252 \cdot \text{NAV}_{T-1}$.

**4. FX rates.** For non-USD constituents, pull spot FX from `=X` tickers (e.g., `EURUSD=X`, `JPYUSD=X`). If the constituent's local market is closed, use a forward-implied FX or the last available spot.

**5. Bid/ask half-spread $\kappa + \tau$.** Estimate from realized cost of round-trip basket trades. For SPY-class ETFs $\kappa + \tau \approx 1\text{–}3$ bps; for less liquid international ETFs, $5\text{–}15$ bps.

No optimization or loss function is required for iNAV per se — the model is an accounting identity. The empirical work goes into the *staleness adjustment* (Model 2) and into the spread-cost estimation, which is a quantile regression of realized round-trip costs on basket size and volatility.

## Algorithm outline

```
Inputs:
    etf_ticker         : string, e.g. "SPY"
    timestamp t        : current wall-clock time
    holdings_file      : DataFrame with columns [ticker, shares, weight, currency]
    eod_nav            : prior-day end-of-day NAV per share (from issuer)
    eod_shares_out     : prior-day shares outstanding
    expense_ratio f    : float, annualized

Procedure:

1. Pull current basket
   - Read holdings_file for date T-1 (issuer publishes before open).
   - For each ticker i, normalize n_i = shares_i (per creation unit).

2. Pull constituent quotes
   - For i in basket:
        bar_i = yf.Ticker(i).history(period="1d", interval="1m").iloc[-1]
        P_i   = bar_i["Close"]
        ts_i  = bar_i.name        # timestamp of last print
   - Flag i as STALE if (t - ts_i) > threshold (e.g. 5 minutes for US, market-closed for intl).

3. Pull FX rates
   - For each non-USD currency c:
        FX_c = yf.Ticker(f"{c}USD=X").history(period="1d", interval="1m").iloc[-1]["Close"]

4. Compute cash accrual
   - r = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100 / 360
   - elapsed = business_seconds_since(eod_release_time)
   - C_t = C_{T-1} * (1 + r * elapsed/86400)
   - For each i with ex_date in (T-1, t]:
        C_t += n_i * dividend_i

5. Compute liabilities accrual
   - L_t = L_{T-1} + (f / 252) * eod_nav * eod_shares_out * (elapsed / 86400)

6. Decompose constituents
   - basket_live  = sum(n_i * P_i * FX_i for i in LIVE)
   - basket_stale = sum(n_i * P_i_last * FX_i for i in STALE)
     (Model 2 will overwrite basket_stale with futures-implied value.)

7. Aggregate
   - iNAV_t = (basket_live + basket_stale + C_t - L_t) / S_t

8. Compute trade signals
   - Pull live ETF mid M_t = yf.Ticker(etf_ticker).history(period="1d", interval="1m").iloc[-1]
   - premium = (M_t - iNAV_t) / iNAV_t
   - if premium > kappa + tau:  signal = "CREATE"  (sell ETF, buy basket)
   - if premium < -(kappa + tau): signal = "REDEEM" (buy ETF, sell basket)
   - else: signal = "FLAT"

9. Log diagnostics
   - Record (t, iNAV_t, M_t, premium, stale_fraction, n_live, n_stale).
```

The hot loop (steps 2–8) should run every 1–5 seconds for liquid US ETFs and every 10–30 seconds for international ETFs.

## yfinance data requirements

| Data | yfinance call | Notes |
|------|---------------|-------|
| Top-10 holdings | `yf.Ticker(etf).funds_data.top_holdings` | Returns DataFrame with `holdingPercent`. Insufficient for full basket — must supplement with issuer JSON. |
| Fund description | `yf.Ticker(etf).funds_data.description` | Confirms ETF mandate. |
| Constituent prices | `yf.Ticker(stock).history(period="1d", interval="1m")` | 1-minute bars for intraday; switch to `"5m"` for non-US tickers. |
| FX spot | `yf.Ticker("EURUSD=X").history(...)` | Use `=X` suffix; check `info["regularMarketTime"]` for staleness. |
| Dividends declared | `yf.Ticker(stock).dividends` | Series indexed by ex-date with cash amount. No pay-date — assume T+30. |
| Short rate | `yf.Ticker("^IRX").history(period="5d")` | 13-week T-bill yield as proxy for overnight financing. |
| Expense ratio | `yf.Ticker(etf).info["annualReportExpenseRatio"]` | Float, e.g. 0.0945 = 9.45 bps. |
| ETF mid | `yf.Ticker(etf).history(period="1d", interval="1m")` | Use `(High+Low)/2` of last bar as mid proxy. |

**Limitations and workarounds:**

- yfinance does not expose full basket weights — supplement with issuer-published holdings CSV/JSON downloaded once daily.
- 1-minute granularity is the finest yfinance offers and lags by 1–15 minutes for free users. For production use, replace yfinance with a direct feed (Polygon, IEX, exchange UDP) once the model is validated.
- Dividend pay-dates are not in yfinance; use a corporate-actions vendor (or hard-code T+30 from ex-date as an approximation).
- For international constituents, yfinance returns the *local* exchange's last close when that market is shut — this is the entire reason Model 2 exists.

## Calibration / refit frequency

- **Basket composition $\{n_i\}$:** refit daily, pre-open, from the issuer's holdings file.
- **Cash and liabilities seeds $C_{T-1}, L_{T-1}$:** refit daily at the prior day's NAV strike (4 PM ET for US ETFs).
- **Bid/ask cost parameters $\kappa, \tau$:** refit weekly from realized round-trip costs.
- **Intraday recomputation of $\text{iNAV}_t$:** every 1–5 seconds during market hours.

The choice is driven by the operational frequency of each input. Basket weights are contractually static intraday. Cash and liabilities are deterministic functions of time given the seed. Only constituent prices and FX move on a sub-second cadence.

## Validation and diagnostics

1. **EOD reconciliation.** After 4 PM ET, the issuer publishes the official NAV. Compute the model's $\text{iNAV}_{4\text{PM}}$ and compare. Residual should be < 1 bp for US-only ETFs; < 5 bps for international ETFs (the residual is the staleness gap from Model 2).
2. **Constituent coverage.** Track $\sum_i n_i P_i$ as a fraction of (NAV − cash + liabilities). If coverage drops below 99%, the holdings file is stale or a constituent has been corp-actioned.
3. **Premium stationarity.** $(M_t - \text{iNAV}_t)/\text{iNAV}_t$ should be a mean-reverting process with mean near zero and half-life of seconds for liquid ETFs. Persistent non-zero premium indicates a model bug or a market dislocation.
4. **Creation/redemption realization.** When the model signals CREATE/REDEEM and the trade is executed, the realized P&L should be approximately $(M_t - \text{iNAV}_t) \cdot S_{\text{traded}} - \text{fees}$. Systematic shortfall indicates underestimated $\kappa + \tau$.
5. **Cash drift.** Track $C_t$ vs the issuer's reported cash holding once weekly. Drift > 50 bps means dividend accrual is being mis-applied.

## Connections to other models

- **Model 2 (Futures-Implied NAV):** Replaces the stale-component term $\tilde{P}_{i,t}$ when underlying markets are closed. Without Model 2, iNAV is useless for EFA, EEM, EWJ, FXI, etc.
- **Model 5 (Cost-of-Carry):** Provides the financing-rate and dividend-accrual inputs ($r$, $D_i$) used in cash accrual. The discrete-dividend correction is shared between the two models.
- **Layer 1 microstructure models:** The bid/ask half-spread $\kappa + \tau$ is calibrated from microstructure cost estimates (e.g., Roll's estimator on the ETF).
- **Layer 4 execution:** The CREATE/REDEEM signal feeds the AP execution router, which sizes orders against available basket liquidity.

## Limitations and failure modes

1. **Stale prints on constituents.** A constituent that has not traded in minutes still appears in last-print fields. The model treats it as live and produces a biased iNAV. Mitigation: monitor print timestamps and route to Model 2 above a staleness threshold.
2. **Corporate actions.** Splits, special dividends, spin-offs, and rebalances change $\{n_i\}$. If the holdings file is not refreshed, iNAV is wrong by the full action amount. Mitigation: cross-check basket against issuer file daily and against constituent corporate-action calendar.
3. **T+2 vs same-day cash.** Pending creation/redemption settlements cause $S_t$ to lag the AP's economic exposure. For a fast-creating ETF (e.g., during a flash event) the model's denominator can be wrong by 1–3%. Mitigation: track pending creation orders separately.
4. **Foreign withholding tax on dividends.** International ETFs accrue dividends net of withholding tax. Using gross dividends overstates $C_t$ by 15–30%. Mitigation: apply per-country withholding rates.
5. **Fee waivers and reimbursements.** Some ETFs have temporary fee waivers; using the headline expense ratio overstates $L_t$. Mitigation: use the contractual net expense ratio.
6. **Market-on-close auction.** The published 4 PM NAV uses official closing prices, but the model's last 1-minute bar may pre-date the closing cross. Reconcile carefully.

## References

- Petajisto, A. (2017). "Inefficiencies in the Pricing of Exchange-Traded Funds." *Financial Analysts Journal* 73(1).
- Madhavan, A. (2016). *Exchange-Traded Funds and the New Dynamics of Investing.* Oxford.
- BlackRock iShares Authorized Participant Handbook (technical document).
- Cboe Global Markets, "Indicative Optimized Portfolio Value (IOPV) Specification."
- Engle, R. and Sarkar, D. (2006). "Premiums-Discounts and Exchange Traded Funds." *Journal of Derivatives* 13(4).
