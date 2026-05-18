# Liquidity-Adjusted VaR and FRTB Liquidity Horizons

> Extensions of VaR and ES that incorporate the cost and time required to unwind positions, replacing the textbook assumption of frictionless mark-to-market exits with realistic liquidation dynamics. Layer 6 — Risk & Capital. Primary use: risk reporting on illiquid inventory, FRTB regulatory capital with prescribed liquidity horizons, internal margin tiering, and prime-broker margin sizing.

## Mathematical formulation

Classical VaR assumes positions can be marked to market and instantaneously liquidated at posted mid-quotes. In practice, executing a position incurs **bid-ask spread cost**, **market impact**, and requires **time** during which prices continue to move. Liquidity-adjusted VaR (L-VaR) accounts for these.

### Bangia-Diebold-Schuermann-Stroughair (BDSS) decomposition

L-VaR is decomposed into a price risk component and a liquidity risk component:

$$
\text{L-VaR} \;=\; \underbrace{V_0\,(\mu - \sigma\, z_\alpha)}_{\text{price-risk VaR}} \;+\; \underbrace{\frac{1}{2} V_0\,(\bar S + a\, \sigma_S)}_{\text{liquidity component}}
$$

where $\bar S$ is the mean relative bid-ask spread, $\sigma_S$ is its volatility, $a$ is a percentile scaling (e.g., $a = z_\alpha$) to capture stressed-spread regimes, and $V_0$ is position market value. The factor $\tfrac{1}{2}$ reflects that round-trip exit pays half the spread (mid to bid).

A simpler operational form, ignoring spread volatility:

$$
\boxed{\;\text{L-VaR} \;=\; \text{VaR} \;+\; \tfrac{1}{2}\, \text{BidAskSpread} \times |\text{Position}|\;}
$$

### Almgren-Chriss optimal-liquidation extension

For a position $X_0$ to be liquidated over horizon $T$ in $N$ discrete steps with constant trading rate $v$, the Almgren-Chriss execution-cost model gives expected cost

$$
\mathbb{E}[\text{Cost}] \;=\; \tfrac{1}{2}\, \gamma\, X_0^2 \;+\; \eta\, \sum_{k=1}^N \frac{(\Delta x_k)^2}{\tau}
$$

with permanent-impact coefficient $\gamma$, temporary-impact coefficient $\eta$, step size $\tau = T/N$, and slice $\Delta x_k$. Variance of execution cost is

$$
\text{Var}[\text{Cost}] \;=\; \sigma^2 \sum_{k=1}^N \tau\, x_k^2
$$

where $x_k$ is shares remaining at step $k$. Minimizing $\mathbb{E}[\text{Cost}] + \lambda\,\text{Var}[\text{Cost}]$ yields the optimal trajectory

$$
x_k^* \;=\; X_0\, \frac{\sinh(\kappa(T - k\tau))}{\sinh(\kappa T)}, \qquad \kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}.
$$

L-VaR then incorporates **both** the expected execution cost and the increased exposure window (the position bleeds away over $[0, T]$ rather than vanishing instantly):

$$
\text{L-VaR}_{\text{AC}} \;=\; z_\alpha\, \sigma\, \sqrt{\int_0^T x^*(t)^2\, dt} \;+\; \mathbb{E}[\text{Cost}].
$$

### FRTB Internal Models Approach: liquidity horizons

The FRTB IMA replaces the iid $\sqrt{\text{time}}$ scaling of shocks with **prescribed liquidity horizons** $h_k$ per risk-factor class. The 97.5% Expected Shortfall is computed by scaling each base 10-day shock to the appropriate horizon:

$$
\text{Shock}_{h_k} \;=\; \text{Shock}_{10d} \times \sqrt{\frac{h_k}{10}}.
$$

| Risk-factor class | Liquidity horizon $h$ |
|---|---:|
| Interest rate (major currencies) | 10 d |
| Large-cap equity price | 10 d |
| FX (G10 pairs) | 10 d |
| Small-cap equity price | 20 d |
| Equity volatility (large-cap) | 20 d |
| IG credit spread | 40 d |
| Small-cap equity volatility | 60 d |
| HY credit spread | 60 d |
| Emerging market equity / FX | 60 d |
| Structured product / exotic | 120 d |

The portfolio-level FRTB ES combines bucketed ES values across non-overlapping horizon buckets via the FRTB "Liquidity Horizon Scaled" formula

$$
\text{ES} \;=\; \sqrt{\sum_{j=1}^J \left( \text{ES}_T(P;j) \cdot \sqrt{\frac{\text{LH}_j - \text{LH}_{j-1}}{T}} \right)^2}
$$

where $\text{LH}_j$ are cumulative liquidity-horizon endpoints (10, 20, 40, 60, 120 days), $T = 10$ days, and $\text{ES}_T(P;j)$ is the ES computed shocking only factors with horizon $\geq \text{LH}_j$.

### Tiered haircuts (internal)

Market makers apply additional haircuts $h_T$ by tier $T$ on top of VaR:

| Tier | Examples | Haircut multiple |
|---|---|---:|
| 1 | Index futures, mega-cap equities | $1.0\times$ |
| 2 | Mid-cap equities, liquid ETFs | $1.5\times$ |
| 3 | Small-cap equities, less-liquid ETFs, on-the-run bonds | $2.0$–$3.0\times$ |
| 4 | Off-the-run bonds, structured products, exotic options | $5.0\times$+ |

Tier assignment uses an **average-daily-volume / position-size ratio** as the primary criterion, augmented by spread-based and depth-based signals.

## Intuition and what this model addresses

Textbook VaR implicitly assumes you can exit your book at the close print. Reality:

- A $\$100\text{M}$ position in a stock that trades $\$50\text{M}$/day **cannot** be exited in one day at the close. You will move the price against yourself.
- Bid-ask spreads widen dramatically in stress — small-cap spreads can quadruple in a sell-off.
- The longer your unwind takes, the more **additional** price risk accrues. A 10-day unwind of an illiquid book has roughly $\sqrt{10}$ the price-risk variance of an instant exit.

L-VaR captures all three. The FRTB liquidity-horizon framework operationalizes the third (time-to-unwind) within the regulatory ES; internal tiered haircuts and Almgren-Chriss-based L-VaR add the first two.

For an equity market-maker, the practical implications:

- Long-only large-cap S&P inventory: liquidity adjustment is small (1-2% uplift on VaR).
- Small-cap or thinly traded ETF inventory: liquidity adjustment can **double** the headline VaR.
- Stress-period spreads on small-caps can widen $4$–$10\times$, dwarfing the price-risk uplift.

## Computation approach

Three operational stages:

1. **Per-position liquidation horizon** $T_i$ from $\text{ADV}_i$ (average daily volume): $T_i = \text{Position}_i / (\kappa \cdot \text{ADV}_i)$, where $\kappa$ is a maximum participation rate (typically 10–20%).
2. **Per-position bid-ask cost**: $\text{Spread}_i \times |\text{Position}_i| / 2$ for round-trip closure.
3. **Portfolio L-VaR**: combine bucketed ES with horizon scaling, then add expected liquidation cost.

For yfinance-only data, we can obtain (i) historical volumes for ADV, (ii) live `info['bid']` / `info['ask']` for spread, and (iii) close prices for VaR; this is sufficient for an operational L-VaR.

## Algorithm outline

```
ALGORITHM: Compute_LVaR(positions, alpha, max_participation)
INPUT:
  positions: dict {ticker -> dollar_position}
  alpha: float (0.99 or 0.975)
  max_participation: float (e.g. 0.15 = 15% of ADV/day)
OUTPUT:
  price_VaR, liquidity_cost, L_VaR, per_name_diagnostics

STEP 1 — Pull data
  tickers <- positions.keys()
  data <- yf.download(tickers, period="2y", auto_adjust=True)
  prices <- data["Close"]
  volumes <- data["Volume"]
  rets <- prices.pct_change().dropna()

STEP 2 — Per-name liquidity statistics
  FOR each ticker t:
      ADV_shares <- volumes[t].tail(60).mean()
      last_price <- prices[t].iloc[-1]
      ADV_dollar <- ADV_shares * last_price
      shares_held <- positions[t] / last_price
      # liquidation horizon in trading days
      T_t <- shares_held / (max_participation * ADV_shares)
      T_t <- max(1.0, T_t)
      # bid-ask spread from live quotes
      info <- yf.Ticker(t).info
      bid <- info.get("bid", None)
      ask <- info.get("ask", None)
      IF bid > 0 AND ask > 0:
          spread_rel <- (ask - bid) / ((ask + bid) / 2)
      ELSE:
          # fallback: high-low range as proxy
          spread_rel <- (data["High"][t].tail(20) /
                         data["Low"][t].tail(20) - 1).mean() * 0.25
      liquidity_horizon[t] <- T_t
      spread[t] <- spread_rel

STEP 3 — Assign FRTB-style buckets
  FOR each ticker t:
      mkt_cap <- yf.Ticker(t).info.get("marketCap", 0)
      IF mkt_cap > 10e9:
          h_t <- 10   # large-cap equity
      ELIF mkt_cap > 2e9:
          h_t <- 20   # mid/small-cap (use 20d)
      ELSE:
          h_t <- 60   # micro-cap / EM-like
      # Override if position-based horizon is longer
      h_t <- max(h_t, ceil(liquidity_horizon[t]))
      horizon[t] <- h_t

STEP 4 — Price-risk VaR (base 1-day historical sim)
  w_dollar <- vector of positions[t]
  pnl_vec <- rets.values @ w_dollar
  loss_vec <- -pnl_vec
  VaR_1d <- numpy.quantile(loss_vec, alpha)

STEP 5 — FRTB liquidity-horizon scaling
  # Bucketize by horizon
  buckets <- group tickers by horizon
  ES_components <- []
  FOR each bucket j with horizon LH_j:
      # Compute ES restricted to factors in bucket j
      bucket_pnl <- rets[bucket_tickers].values @ w_bucket
      bucket_loss <- -bucket_pnl
      ES_j_1d <- bucket_loss[bucket_loss > quantile(bucket_loss,alpha)].mean()
      # Scale by sqrt((LH_j - LH_{j-1}) / 10)
      scale_j <- sqrt((LH_j - LH_{j-1}) / 10)
      ES_components.append(ES_j_1d * scale_j)
  ES_FRTB <- sqrt(sum(c^2 for c in ES_components))

STEP 6 — Liquidity cost (Bangia-Diebold)
  liquidity_cost <- 0
  FOR each ticker t:
      liquidity_cost += 0.5 * spread[t] * abs(positions[t])

STEP 7 — Almgren-Chriss execution-cost refinement (optional)
  FOR each ticker t with horizon > 1:
      # Calibrate eta from typical impact-cost model;
      # default eta = 0.1 * sigma_t / ADV_dollar
      sigma_t <- rets[t].std() * last_price
      eta_t <- 0.1 * sigma_t / ADV_dollar
      x0 <- shares_held
      AC_cost[t] <- eta_t * x0^2 / liquidity_horizon[t]
  total_AC_cost <- sum(AC_cost.values())

STEP 8 — Combine
  L_VaR <- ES_FRTB + liquidity_cost     # or replace with total_AC_cost
  return {
    "price_VaR": VaR_1d,
    "ES_FRTB_with_horizons": ES_FRTB,
    "liquidity_cost_spread": liquidity_cost,
    "liquidity_cost_AC": total_AC_cost,
    "L_VaR": L_VaR,
    "per_name": {ticker: {horizon, spread, ADV_dollar}}
  }
```

## yfinance data requirements

- **Daily volume**: `data["Volume"]` for ADV. Average over 30–60 trading days.
- **Bid/Ask**: `yf.Ticker(t).info["bid"]`, `info["ask"]`. Caveats below.
- **Market cap**: `info["marketCap"]` for FRTB bucketing.
- **High/Low**: as fallback for spread when live bid/ask is stale or missing.
- **Float / shares outstanding**: `info["floatShares"]` — useful for cross-checking ADV-based horizons.

**Important caveats on yfinance liquidity data:**

- `info['bid']` and `info['ask']` are **end-of-day quotes** when the API is queried after the close. They reflect the last NBBO snapshot, not live spreads. For intraday risk, use a real market-data feed.
- For ETFs especially, the posted bid/ask can be stale; the implied creation-unit spread is a better measure.
- Bid/ask is zero for many micro-caps in yfinance — implement a fallback (High-Low range proxy, or sector-median spread).
- Volume is exchange-consolidated for US equities but may exclude dark-pool prints.

## Refit / refresh frequency

- **Daily**: full L-VaR recomputation with rolling 30-day ADV and current bid/ask.
- **Daily**: FRTB liquidity-horizon buckets refreshed if a ticker's market cap crosses a bucket boundary.
- **Weekly**: tier assignment review for the inventory; tickers moving between tiers trigger desk-level alerts.
- **Quarterly**: Almgren-Chriss impact-coefficient $\eta$ recalibrated against TCA (transaction cost analysis) data.
- **Annual**: FRTB liquidity-horizon table reviewed against regulator updates.

## Validation and diagnostics

- **Implied vs. realized liquidation cost**: compare L-VaR's liquidity component to actual slippage observed when positions are reduced. Persistent under-estimation is a calibration failure.
- **Spread regime backtest**: bucket historical days by VIX level; verify that stressed bid-ask spreads in the model match historical realizations during VIX > 30 episodes.
- **Horizon coverage**: for each name, the assigned horizon should exceed $\text{Position}/(\kappa\cdot\text{ADV})$. Flag any violations.
- **Concentration ratio**: $\sum_i \mathbf{1}\{\text{Position}_i > 0.2 \cdot \text{ADV}_i\}$ — number of positions that cannot be cleared in one day at 20% participation. A risk-management red flag if it exceeds book-level thresholds.
- **Stress-spread shock**: re-run L-VaR with bid-ask spreads scaled $5\times$; the difference reveals exposure to liquidity-regime shifts.
- **FRTB PLA test extension**: P&L Attribution test under FRTB applies to the liquidity-horizon-scaled ES; failure forces standardized-approach capital.

## Connections to other models

- **Model 17 (VaR)** — L-VaR is a wrapper around price-risk VaR with additive liquidity terms.
- **Model 18 (Expected Shortfall)** — FRTB regulatory ES *is* liquidity-horizon-scaled by construction; L-VaR and FRTB ES are essentially the same calculation under different decompositions.
- **Model 19 (Stress Tests)** — stressed scenarios must combine price shocks with **stressed spread** and **extended horizon**; without liquidity adjustment, stress P&L is optimistic.
- **Layer 5 execution models** — Almgren-Chriss is the primary execution-model bridge. The same $\eta, \gamma$ parameters drive both optimal-execution scheduling and L-VaR liquidity cost.
- **Layer 4 microstructure** — bid-ask spread models (Roll, MRR) feed the spread component of L-VaR with theoretically founded estimates instead of raw quoted spreads.

## Limitations and failure modes

- **ADV is a poor stress proxy**: in a crisis, volume often *rises* while liquidity (depth at touch) *collapses*. Using normal-time ADV during stress understates true liquidation time.
- **Spread is not depth**: a tight quoted spread says nothing about the size available at that quote. A $\$0.01$ NBBO on a $\$50$ stock with $100$-share depth is illiquid for a $\$10\text{M}$ position. Real depth requires order-book or TCA data.
- **Almgren-Chriss is linear-impact**: real impact is sub-linear (square-root law); using AC overestimates cost for medium-large orders and underestimates for very large ones.
- **Crowded-trade contagion**: if many participants share the same position, simultaneous unwind is much costlier than individual ADV would suggest. L-VaR does not capture this without explicit crowding adjustment.
- **yfinance bid/ask reliability**: as noted, end-of-day quotes; for active L-VaR systems, replace with live feed.
- **Bucket discontinuities (FRTB)**: a stock crossing $\$10\text{B}$ market cap shifts from 10-day to 20-day horizon, doubling its capital contribution overnight. Smoothing is operationally helpful.
- **Procyclical spread feedback**: when L-VaR rises in stress, positions are reduced, widening spreads further, raising L-VaR further. Same procyclicality as VaR but with a second mechanism.
- **Off-exchange flow**: dark-pool and retail-internalized volume is not in yfinance volume series; ADV-based horizons are biased upward (slower) for names with heavy off-exchange flow.

## References

- Bangia, A., Diebold, F.X., Schuermann, T., Stroughair, J. (1999). "Modeling Liquidity Risk, with Implications for Traditional Market Risk Measurement and Management." Wharton Financial Institutions Center 99-06.
- Almgren, R., Chriss, N. (2000). "Optimal Execution of Portfolio Transactions." *Journal of Risk* 3.
- Almgren, R. (2003). "Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk." *Applied Mathematical Finance* 10.
- BCBS d457 (2019). *Minimum Capital Requirements for Market Risk*, paragraphs MAR33 on liquidity horizons.
- BCBS d518 (2023). *FRTB: Revised Standards.*
- Acerbi, C., Scandolo, G. (2008). "Liquidity Risk Theory and Coherent Measures of Risk." *Quantitative Finance* 8.
- Stange, S., Kaserer, C. (2009). "The Impact of Liquidity Risk: A Fresh Look." *International Review of Finance* 11.
- Cont, R., Kotlicki, A., Valderrama, L. (2020). "Liquidity at Risk: Joint Stress Testing of Solvency and Liquidity." *Journal of Banking & Finance*.
