# Cost-of-Carry and Discrete-Dividend Index Basket

> No-arbitrage pricing model for equity index futures and ETF baskets that links the futures price to the spot index via financing cost net of dividend yield, with explicit handling of discrete dividend payments between trade and expiry. Layer 2 — Structural / Cross-Sectional. Primary use: futures-cash basis trading, ETF creation/redemption pricing, and dividend-strip valuation.

## Mathematical formulation

### Continuous-dividend cost-of-carry

For a stock or index paying a continuous dividend yield $q$, the no-arbitrage futures price with maturity $T$ is:

$$
F_t \;=\; S_t \cdot e^{(r - q)(T - t)}
$$

where:

- $S_t$ — spot price of the index at time $t$.
- $r$ — continuously compounded risk-free financing rate, term-matched to $T$.
- $q$ — continuously compounded dividend yield (weighted average across index constituents).
- $T - t$ — time to maturity in years.

The derivation is standard. A long-spot/short-futures portfolio is risk-free if held to expiry: the spot earns $-r$ in financing cost and $+q$ in dividends; the futures locks in $F$ at expiry. Absence of arbitrage requires the portfolio to earn the risk-free rate, giving the formula.

### Discrete-dividend correction

In practice, index constituents pay discrete dividends on specific ex-dates. The accurate cost-of-carry expression replaces the continuous-yield term with a sum over discrete payments:

$$
F_t \;=\; S_t \cdot e^{r(T - t)} \;-\; \sum_{i \,:\, t_i \in [t, T]} D_i \cdot e^{r(T - t_i)}
$$

where:

- $D_i$ — discrete dividend paid at time $t_i$ (in index points, after weighting by constituent index weight).
- $t_i$ — ex-date of dividend $i$.

The second term is the future value (at $T$) of all dividends paid between $t$ and $T$. Each dividend reduces $S$ by $D_i$ on its ex-date, so the futures price must be $S \cdot e^{r(T-t)}$ minus the cumulative future-valued dividend stream.

### Equivalence

Setting $q$ such that $S_t e^{(r-q)(T-t)} = S_t e^{r(T-t)} - \sum_i D_i e^{r(T-t_i)}$:

$$
q \;=\; \frac{1}{T - t} \ln \left( 1 \;+\; \sum_{i} \frac{D_i \cdot e^{-r(t_i - t)}}{S_t} \right)
$$

For small dividend yields and short horizons, $q \approx \frac{1}{T-t} \sum_i D_i / S_t$.

### Implied financing rate

Inverting the cost-of-carry formula gives the futures-implied (repo) rate:

$$
r_{\text{impl}} \;=\; \frac{1}{T - t} \ln \left( \frac{F_t + \sum_i D_i e^{r(T-t_i)}}{S_t} \right)
$$

When $r_{\text{impl}} < r_{\text{LIBOR/SOFR}}$ the future is "cheap" — financing-arbitrage opportunity (buy spot, short future, fund at SOFR, earn $r_{\text{impl}}$ on the trade). When $r_{\text{impl}} > r$, the future is "rich."

### ETF creation basket value

For an ETF holding $N$ constituents with $n_i$ shares each, the basket NAV at time $t$ is:

$$
\text{NAV}_t \;=\; \sum_{i=1}^{N} n_i P_{i,t} \;+\; A_t \;-\; L_t
$$

where:

- $n_i$ — shares of constituent $i$ in the basket.
- $P_{i,t}$ — current price of constituent $i$.
- $A_t$ — accrued dividends receivable (dividends declared ex-date but not yet paid).
- $L_t$ — accrued liabilities (management fees, etc.).

The accrued-dividend term is:

$$
A_t \;=\; \sum_{i, \, j \,:\, t_{i,j}^{\text{ex}} \le t < t_{i,j}^{\text{pay}}} n_i \cdot d_{i,j}
$$

where $d_{i,j}$ is the $j$-th cash dividend per share for constituent $i$.

### AP arbitrage condition

For the authorized participant:

$$
\text{Create when:} \quad M_t > \text{NAV}_t + \kappa + \tau
$$

$$
\text{Redeem when:} \quad M_t < \text{NAV}_t - \kappa - \tau
$$

where $M_t$ is the ETF secondary-market mid, $\kappa$ is the per-share creation fee, and $\tau$ is round-trip basket transaction cost.

### Forward index value (term structure)

For a multi-period futures curve $\{F_t^{(T_1)}, F_t^{(T_2)}, \ldots\}$, each contract obeys its own cost-of-carry:

$$
F_t^{(T_k)} \;=\; S_t e^{r_k (T_k - t)} \;-\; \sum_{i \,:\, t_i \in [t, T_k]} D_i e^{r_k (T_k - t_i)}
$$

The implied forward dividends between $T_k$ and $T_{k+1}$ can be extracted from adjacent contracts:

$$
\sum_{i \,:\, t_i \in [T_k, T_{k+1}]} D_i e^{r(T_{k+1} - t_i)} \;=\; F_t^{(T_k)} e^{r(T_{k+1} - T_k)} \;-\; F_t^{(T_{k+1})}
$$

This is the basis for dividend-strip pricing.

### Continuous-dividend approximation for short horizons

For maturities under ~3 months on broad indices like the S&P 500, the discrete-dividend correction is dominated by 2–3 large constituent ex-dates. Pre-3M, model exactly with discrete dividends; beyond, the continuous-yield approximation is acceptable and computationally cheaper.

## Intuition and what this model addresses

The cost-of-carry relationship is the most fundamental no-arbitrage identity in equity derivatives. If futures and spot decouple by more than transaction costs, a model-free arbitrage exists: long the cheap side, short the rich side, hold to expiry, and earn the basis with zero terminal risk.

The model addresses several specific failures of cruder approximations:

1. **Continuous-yield models miss large discrete dividends.** Apple, Microsoft, JPMorgan, and a handful of other mega-caps pay quarterly dividends that account for a non-trivial fraction of S&P 500 dividend yield. Treating these as continuous smears the cash flow and produces a 1–5 bp daily mispricing around ex-dates.
2. **ETF NAV without dividend accrual is biased.** Between ex-date and pay-date (typically 30 days), the ETF holds a receivable that is not in constituent prices. Ignoring this term mis-prices the basket by 5–20 bps for dividend-rich periods.
3. **Implied repo rate is the cleanest funding signal.** It strips out dividends and gives a direct read on the marginal cost of financing equity exposure. Cross-sectional dispersion across indices (S&P vs Nasdaq vs Russell) signals dealer balance-sheet pressure.
4. **Dividend curve trading.** The term structure of futures encodes the market's view of forward dividends. This is itself a tradable asset (dividend swaps, dividend futures) and the foundation of equity-only fixed-income strategies.
5. **Index roll mechanics.** Futures roll quarterly. The basis differs across maturities, and rolling a position from one contract to the next without accounting for the basis differential produces hidden P&L.

For market makers, this is the spine of equity-index pricing: every quote on S&P futures, every NAV calculation on SPY, every creation/redemption decision flows through this identity.

## Calibration approach

The model has three inputs requiring calibration: the financing rate $r$, the dividend stream $\{D_i, t_i\}$, and the transaction-cost band $\kappa + \tau$.

### Financing rate $r$

For US equity index futures, the relevant rate is the secured overnight financing rate (SOFR) plus a stock-loan spread. From yfinance:

- Short end (< 3 months): use `^IRX` (13-week T-bill yield) as a proxy for SOFR; add ~5–15 bps for general collateral stock loan.
- Mid (3–12 months): interpolate from `^IRX`, `^FVX` (5-year), `^TNX` (10-year), or use SOFR futures (`SR3=F` family) directly.
- Term-match $r$ to the futures expiry $T - t$. A linear or log-linear interpolation between the available tenor points suffices for daily use.

For ETF basket pricing where same-day settlement is assumed, use the overnight rate directly without spread.

### Dividend stream $\{D_i, t_i\}$

The dividend stream has two components: *announced* dividends with known $D_i$ and $t_i$, and *projected* dividends that have not yet been declared but are expected based on the constituent's dividend history.

**Announced (high-confidence):**

For each index constituent, pull historical and announced upcoming dividends:

```python
divs = yf.Ticker(stock).dividends   # historical ex-dates with cash amounts
```

The Series is indexed by ex-date with the per-share cash amount. Future-dated ex-dates (announced but not yet paid) appear in the same series.

**Projected (model-based):**

For ex-dates beyond the announcement horizon (typically ~45 days out), forecast from the constituent's dividend history:

$$
\hat{D}_{i, t} \;=\; D_{i, t-1Q} \cdot (1 + g_i)
$$

with growth rate $g_i$ estimated as the trailing 4-quarter year-over-year change. Robust to outliers via trimmed mean.

**Aggregate to index level:**

For each ex-date $t$, sum across all constituents weighted by their index weight $w_i$:

$$
D_t^{\text{index}} \;=\; \sum_{i \,:\, t_i^{\text{ex}} = t} w_i \cdot d_{i, t} \cdot \frac{\text{Index Divisor}}{\text{Index Level}}
$$

(The divisor converts per-share dividends into index points.)

### Transaction-cost band $\kappa + \tau$

Calibrate from realized round-trip futures-cash basis trades:

$$
\kappa + \tau \;=\; \text{Quantile}_{95\%}\left( |F_{\text{trade}} - F_{\text{theoretical}}| \right)
$$

over a 30-day window. This sets the trigger band for the no-arbitrage strategy.

### Loss function for implied repo

The implied repo rate is a *measurement*, not an estimate to be optimized. The "calibration" is verifying that the rate is sensible:

$$
r_{\text{impl}} \in [r_{\text{SOFR}} - 10\text{bp}, \; r_{\text{SOFR}} + 50\text{bp}]
$$

Excursions outside this band flag either a basis arbitrage opportunity or a data error (typically a missing dividend).

## Algorithm outline

```
Inputs:
    index_ticker         : e.g. "^GSPC" (S&P 500 spot)
    futures_ticker       : e.g. "ES=F" (front-month E-mini)
    constituents         : list of (ticker, index_weight) tuples
    expiry T             : datetime of futures expiry
    today t              : datetime.now()

Procedure:

1. Pull spot
   - S_t = yf.Ticker(index_ticker).history(period="1d", interval="1m").iloc[-1]["Close"]

2. Pull futures
   - F_t = yf.Ticker(futures_ticker).history(period="1d", interval="1m").iloc[-1]["Close"]
   - expiry_T = yf.Ticker(futures_ticker).info["expireDate"]   # may need separate calendar
   - tau = (expiry_T - today).days / 365.25

3. Pull financing rate
   - rate_3m = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100
   - rate_10y = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1] / 100
   - r = interpolate(rate_3m, rate_10y, tau)
   - r += stock_loan_spread   # ~10 bp for SPX

4. Build dividend stream
   div_stream = []
   for (ticker, weight) in constituents:
       divs = yf.Ticker(ticker).dividends
       # Filter announced ex-dates between t and T
       announced = divs[(divs.index > t) & (divs.index < expiry_T)]
       for ex_date, d_amount in announced.items():
           # Convert per-share to index points
           idx_points = d_amount * weight * shares_factor(ticker)
           div_stream.append((ex_date, idx_points))

       # Project unannounced quarters
       last_ex_date = announced.index[-1] if len(announced) > 0 else divs.index[-1]
       while last_ex_date + 90_days < expiry_T:
           projected_ex = last_ex_date + 90_days
           recent_amount = divs.iloc[-4:].mean()   # last 4 quarters mean
           growth = (divs.iloc[-4:].mean() / divs.iloc[-8:-4].mean()) - 1
           projected_amount = recent_amount * (1 + growth)
           idx_points = projected_amount * weight * shares_factor(ticker)
           div_stream.append((projected_ex, idx_points))
           last_ex_date = projected_ex

5. Theoretical futures price
   F_theo = S_t * exp(r * tau)
   for (ex_date, D_i) in div_stream:
       t_i = (ex_date - today).days / 365.25
       F_theo -= D_i * exp(r * (tau - t_i))

6. Basis and implied repo
   basis = F_t - F_theo
   r_impl = log( (F_t + sum(D_i * exp(r*(tau-t_i)) for (t_i,D_i) in div_stream)) / S_t ) / tau

7. Trade signal
   cost_band = kappa + tau_costs   # e.g. 0.5 index points for SPX
   if F_t > F_theo + cost_band:
       signal = "FUTURE_RICH: short future, long basket"
   elif F_t < F_theo - cost_band:
       signal = "FUTURE_CHEAP: long future, short basket"
   else:
       signal = "FLAT"

8. ETF creation basket value (parallel path)
   NAV_t = 0
   for (ticker, n_i) in basket_holdings:
       P_i = yf.Ticker(ticker).history(period="1d", interval="1m").iloc[-1]["Close"]
       NAV_t += n_i * P_i

   # Accrued dividends (declared but not paid)
   A_t = 0
   for (ticker, n_i) in basket_holdings:
       divs = yf.Ticker(ticker).dividends
       for ex_date, d in divs.items():
           pay_date = ex_date + timedelta(days=30)   # T+30 approximation
           if ex_date <= today < pay_date:
               A_t += n_i * d

   NAV_t += A_t
   NAV_t -= L_t   # accrued expenses from Model 1

9. ETF arbitrage signal
   M_t = yf.Ticker(etf_ticker).history(period="1d", interval="1m").iloc[-1] mid
   premium = (M_t - NAV_t / shares_per_creation_unit) / (NAV_t / shares_per_creation_unit)
   if premium > etf_cost_band:    signal_etf = "CREATE"
   elif premium < -etf_cost_band: signal_etf = "REDEEM"
   else:                          signal_etf = "FLAT"

10. Persist (t, S_t, F_t, F_theo, basis, r_impl, signal).
```

### Discrete-vs-continuous switch

Inside the algorithm, parameterize a switch:

```python
if tau < 0.25:  # < 3 months
    # Use full discrete-dividend formula
    F_theo = S_t * exp(r * tau) - sum(D_i * exp(r * (tau - t_i)) for ...)
else:
    # Use continuous-yield approximation
    q = log(1 + sum(D_i * exp(-r * t_i) for ...) / S_t) / tau
    F_theo = S_t * exp((r - q) * tau)
```

The discrete form is more accurate but more expensive; the continuous form saves cycles when dividends are far from ex-date.

## yfinance data requirements

| Data | yfinance call | Notes |
|------|---------------|-------|
| Spot index | `yf.Ticker("^GSPC").history(...)` | S&P 500 spot; use `^NDX`, `^RUT` for Nasdaq, Russell. |
| Index futures | `yf.Ticker("ES=F").history(...)` | E-mini S&P front-month; `NQ=F`, `RTY=F` for others. |
| Futures expiry | `yf.Ticker("ES=F").info["expireDate"]` (when available) | Often missing — maintain a CME contract calendar externally. |
| Constituent dividends | `yf.Ticker(stock).dividends` | Series of historical and announced future ex-dates with cash amounts. |
| Index constituents | Not in yfinance | Maintain externally (S&P, MSCI, FTSE publish daily). |
| Short rate | `yf.Ticker("^IRX").history(period="5d")` | 13-week T-bill yield. |
| Mid-curve rates | `yf.Ticker("^FVX")` (5y), `yf.Ticker("^TNX")` (10y) | For interpolation. |
| SOFR futures | `yf.Ticker("SR3=F").history(...)` | More precise short-end financing. |
| ETF mid | `yf.Ticker("SPY").history(period="1d", interval="1m")` | For ETF basis comparison. |

**Limitations and workarounds:**

- `Ticker.dividends` returns per-share cash amounts but not the ex-pay date gap. Assume T+30 for the pay date, or use a corporate-actions vendor for precision.
- Special dividends and one-time distributions appear in `dividends` but are unpredictable — exclude them from forward projections.
- Futures contracts in yfinance roll automatically on `=F` tickers, which can introduce a discontinuity in the price series. For backtesting, pull historical data for specific contract months (e.g., `ESH24.CME` if available) or maintain your own roll adjustment.
- Stock splits adjust the dividend Series — use `auto_adjust=False` if you need the unadjusted dividend cash amount for index-points conversion.
- For non-US indices, futures and rate proxies need to be matched to local funding (e.g., EUR-denominated futures use ESTR, not SOFR).

## Calibration / refit frequency

- **Spot $S_t$ and futures $F_t$:** real-time, every 1–5 seconds during market hours.
- **Financing rate $r$:** daily, at market open. Term-interpolate from the curve.
- **Announced dividends:** daily update; refresh `Ticker.dividends` once a day.
- **Projected dividends:** weekly. Growth rate $g_i$ updates slowly.
- **Transaction-cost band $\kappa + \tau$:** weekly, from realized round-trip costs.
- **Index weights and divisors:** monthly or on each official index rebalance.

The bulk of computation is real-time evaluation of the formula; the dividend stream and rate are stable enough that daily/weekly refits suffice.

## Validation and diagnostics

1. **Basis stationarity.** $F_t - F_{\text{theo}}$ should be mean-reverting around zero with mean roughly equal to the stock-loan spread (~5–15 bps for SPX). Persistent non-zero mean indicates a missing dividend or financing-rate mis-calibration.
2. **Implied repo stability.** $r_{\text{impl}}$ should track SOFR within ±20 bps. Sudden moves outside this band correspond to dealer-balance-sheet events (quarter-end, year-end, FOMC) and are informative as a stress indicator.
3. **Dividend forecast back-test.** Compare projected $\hat{D}_i$ to the actually-announced $D_i$ for the next quarter. RMSE should be < 5% of announced dividend.
4. **Term-structure consistency.** Implied forward dividends across consecutive contracts (extracted from $F_t^{(T_k)}$ and $F_t^{(T_{k+1})}$) should be smooth in time. Discontinuities indicate quarter-end ex-date clustering or model errors.
5. **EOD reconciliation.** At futures expiry, the settlement should equal the spot index. Track the model's $F_{\text{theo}}$ vs realized settlement — converging error confirms calibration.
6. **Cash-and-carry P&L realization.** When the model signals a basis trade and it's executed, the realized P&L should approximately equal the basis at trade time. Systematic shortfall indicates underestimated $\kappa + \tau$.
7. **Continuous-vs-discrete agreement.** For maturities > 6 months, the discrete and continuous-yield formulas should agree to within 1–2 bps. Large divergence indicates a dividend that needs careful handling.

## Connections to other models

- **Model 1 (iNAV / Equity ETF Fair Value):** This model provides the discrete-dividend correction and financing rate consumed by Model 1's cash-accrual term $C_t$. The two models share the dividend stream and risk-free rate inputs.
- **Model 2 (Futures-Implied NAV):** When the futures proxy used in Model 2 rolls across a contract month, the basis calculation here adjusts the proxy return to be continuous.
- **Model 3 (Merton-KMV):** Provides the risk-free rate $r$.
- **Model 4 (Factor Models):** Sector-ETF baskets used in factor mimicking are priced with this model's basket-NAV formula.
- **Layer 3 stat-arb:** Cash-and-carry signal (rich/cheap basis) is itself a stat-arb signal at quarterly frequency.
- **Layer 4 execution:** Sizing of cash-and-carry trades uses the implied repo rate as the marginal P&L driver.

## Limitations and failure modes

1. **Special and one-time dividends.** A $10$/share special dividend on AAPL would move the S&P futures by ~70 bps if not modelled; standard projection routines miss specials. Mitigation: pull a corporate-actions calendar, not just historical `Ticker.dividends`.
2. **Tax effects.** Different investor classes (US taxable, foreign, retirement) experience dividends differently. The model assumes pre-tax dividends; for absolute-level pricing this is a 15–30% bias for foreign holders. Mitigation: use a tax-adjusted dividend ($D_i \cdot (1 - \tau_w)$ where $\tau_w$ is withholding rate).
3. **Stock-loan spread variation.** For names that are hard to borrow (HTB), the financing rate is significantly above SOFR. The model uses an average spread, missing names like meme stocks during squeeze periods. Mitigation: use index-level HTB index from a securities-lending data vendor.
4. **Ex-date timing in continuous formula.** Approximating discrete dividends as continuous yield smooths the ex-date price drop into a continuous decay; this is a meaningful intraday bias on the ex-date itself (±20 bps depending on the dividend size). Mitigation: switch to discrete formula within 5 trading days of any ex-date.
5. **Repo squeeze.** During quarter-end and year-end, repo rates can spike 100+ bps as dealer balance sheets tighten. The model's term-interpolated rate misses this temporary distortion. Mitigation: monitor SOFR futures and patch in elevated rates at known stress dates.
6. **Index methodology changes.** S&P switching from total-return to price-return calculation, or changes in the divisor calculation, alter the relationship between constituent dividends and index dividends. Mitigation: re-read the index methodology each year.
7. **Currency for non-USD indices.** Cost-of-carry on FTSE futures requires GBP financing, not USD; using `^IRX` is wrong. Mitigation: maintain a currency-specific rate curve.
8. **Cross-currency basis.** For USD-denominated futures on foreign indices, the CIP relationship enters: $F_t = S_t \cdot e^{(r_{\text{USD}} - r_{\text{foreign}} + b)(T-t)}$ where $b$ is the cross-currency basis. Ignored in the simple formula.
9. **Liquidity drying at expiry.** In the last week before expiry, the basis can widen due to roll pressure. Trading the basis at this point is not a clean arbitrage. Mitigation: roll positions ~10 days before expiry.

## References

- Cornell, B. and French, K. R. (1983). "The Pricing of Stock Index Futures." *Journal of Futures Markets* 3, 1–14.
- Cornell, B. and French, K. R. (1983). "Taxes and the Pricing of Stock Index Futures." *Journal of Finance* 38, 675–694.
- Mackinlay, A. C. and Ramaswamy, K. (1988). "Index-Futures Arbitrage and the Behavior of Stock Index Futures Prices." *Review of Financial Studies* 1, 137–158.
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives.* 10th ed. Pearson. Chapters 5 and 7.
- Brennan, M. J. and Schwartz, E. S. (1990). "Arbitrage in Stock Index Futures." *Journal of Business* 63, S7–S31.
- Manaster, S. and Rendleman, R. J. (1982). "Option Prices as Predictors of Equilibrium Stock Prices." *Journal of Finance* 37, 1043–1057.
- Working, H. (1949). "The Theory of Price of Storage." *American Economic Review* 39, 1254–1262. (Original cost-of-carry framework.)
- CME Group (2021). "E-mini S&P 500 Futures Contract Specifications and Settlement Procedures."
