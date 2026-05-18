# Kyle Lambda and the Square-Root Impact Law

> Asymmetric-information equilibrium model paired with the empirical square-root law of metaorder impact. Layer 5 — Execution. Primary use: foundational price-impact and adverse-selection modeling, sizing of metaorders, and pre-trade transaction-cost estimation.

## Mathematical formulation

### Kyle (1985): linear-impact equilibrium

Kyle's "Continuous Auctions and Insider Trading" describes a single-period market with three agents:

1. An **informed trader** who observes the terminal asset value $v \sim \mathcal{N}(\mu, \sigma_v^2)$ and submits an order $x$.
2. **Noise traders** who submit a random aggregate order $u \sim \mathcal{N}(0, \sigma_u^2)$, independent of $v$.
3. A **competitive, risk-neutral market maker** who observes only the aggregate order flow $y = x + u$ and sets a price $p$ such that $p = \mathbb{E}[v \mid y]$.

The informed trader maximizes expected profit $\mathbb{E}[(v - p)x \mid v]$. Solving the rational-expectations fixed point yields a linear equilibrium:

$$x^*(v) = \beta (v - \mu), \qquad p(y) = \mu + \lambda y$$

with

$$\beta = \frac{\sigma_u}{\sigma_v}, \qquad \lambda = \frac{1}{2} \frac{\sigma_v}{\sigma_u}$$

The parameter $\lambda$ is **Kyle's lambda** — the per-unit price impact of aggregate signed order flow. It measures both market depth (inverse of $\lambda$) and adverse selection: it scales with the fundamental volatility $\sigma_v$ and inversely with the noise-trader scale $\sigma_u$.

#### Derivation sketch

Conjecture linear strategies $x = \beta(v - \mu)$ and $p = \mu + \lambda y$. Then $y = \beta(v - \mu) + u$ and by the projection theorem for jointly Gaussian variables,

$$\mathbb{E}[v \mid y] = \mu + \frac{\text{Cov}(v, y)}{\mathbb{V}[y]} \, y = \mu + \frac{\beta \sigma_v^2}{\beta^2 \sigma_v^2 + \sigma_u^2} \, y$$

So $\lambda = \beta \sigma_v^2 / (\beta^2 \sigma_v^2 + \sigma_u^2)$. The informed trader's first-order condition on $\mathbb{E}[(v - \mu - \lambda x - \lambda u)x \mid v] = (v - \mu)x - \lambda x^2$ gives $x^* = (v - \mu)/(2\lambda)$, so $\beta = 1/(2\lambda)$. Substituting back and solving yields $\lambda = \sigma_v / (2 \sigma_u)$.

### Empirical Kyle lambda

In data, $\lambda$ is estimated by regressing returns on signed dollar (or share) volume:

$$r_t = \lambda \cdot Q_t^{\text{signed}} + \epsilon_t$$

where $Q_t^{\text{signed}} = \sum_i \text{sign}(\text{trade}_i) \cdot |q_i|$ aggregated over interval $t$. The trade sign comes from a tick-rule or Lee-Ready algorithm on TAQ data. Kyle's $\lambda$ has units of return per unit of signed volume.

A common dimensionless normalization is

$$\lambda^{\text{dim}} = \lambda \cdot \frac{V_{\text{daily}}}{1}$$

so that one full-day's volume traded one-sided produces a return of $\lambda^{\text{dim}}$.

### Square-root impact law

For **metaorders** (parent orders executed by slicing over minutes-to-hours), the empirically dominant law is concave, not linear:

$$\frac{\Delta P}{P} \approx Y \cdot \sigma \cdot \sqrt{\frac{Q}{V}}$$

where

- $\Delta P / P$ is the relative price drift from arrival to completion,
- $Y$ is a dimensionless prefactor of order unity (typically $0.5 \le Y \le 1.5$),
- $\sigma$ is daily volatility,
- $Q$ is metaorder size in shares,
- $V$ is daily volume in shares.

The remarkable property — verified across futures, equities, Bitcoin, and even options — is that impact depends on $Q/V$ alone, not on the participation rate $Q/(V \cdot \tau)$ or the slicing schedule. Only the **total** executed quantity matters.

#### Heuristic derivations

Three classical arguments give the $\sqrt{Q/V}$ scaling:

**(i) Fair pricing.** A metaorder of size $Q$ in a market where the typical price scale of fluctuations over $\tau$ trades is $\sigma \sqrt{\tau / V_{\text{daily}}}$ should pay an impact that matches the information it reveals. Setting $\tau \sim Q$ yields $\sqrt{Q/V}$.

**(ii) Latent liquidity (Toth et al.).** If hidden supply/demand is locally V-shaped around the price, $\text{density}(p) \propto |p - p_0|$, then absorbing $Q$ units requires moving the price by $\Delta p$ with $Q \sim \int_0^{\Delta p} u \, du = \Delta p^2 / 2$, so $\Delta p \sim \sqrt{Q}$.

**(iii) No-arbitrage on round-trip costs.** Permanent impact must scale so that round-trip costs are non-negative on average; combined with the long-memory of order signs, this forces a $\sqrt{Q}$ impact and a power-law decay of temporary impact.

### Propagator (Bouchaud-Gefen-Potters-Wyart) model

Lillo, Mike, and Farmer documented that the autocorrelation of trade signs $\epsilon_t \in \{-1, +1\}$ decays as a power law $\rho(\ell) \sim \ell^{-\gamma_{\text{sign}}}$ with $\gamma_{\text{sign}} \in (0, 1)$ over thousands of trades. Yet the mid-price remains nearly a martingale. The reconciliation: market impact must be **transient**, decaying on a matching timescale.

The propagator model writes mid-price as

$$p_t - p_0 = \sum_{s < t} G(t - s) \, \epsilon_s \, f(q_s) + \text{noise}$$

with $f(q) = q^{\delta}$ for some $\delta \in [0, 1]$ (often $\delta \approx 0.5$) and $G$ a decaying kernel. For the martingale property to hold given long memory in $\epsilon$, $G$ must itself decay as a power law:

$$G(\ell) \sim \ell^{-(1 - \gamma_{\text{sign}})/2}$$

This is the "transient impact" picture: each individual trade nudges the price, but the nudges decay, and the long memory in order signs is exactly what is needed to keep mid prices unpredictable.

## Intuition and what this model addresses

Kyle's $\lambda$ captures **information-driven impact**: the more informative aggregate flow is about the fundamental, the more the market maker moves price per unit of flow. It is linear because the equilibrium under Gaussian uncertainty and risk-neutral, competitive market making is linear.

The square-root law captures the **mechanical reality of metaorder execution**: latent liquidity is finite and locally linear in distance from the mid, so absorbing larger orders pushes price as $\sqrt{Q}$. The law's universality reflects a deep market microstructure property — markets are persistently near-empty at the touch and supply must be excavated from deeper in the book.

The propagator model reconciles the two: signed flow has long memory, individual impact is concave and transient, and prices are still martingales.

Together, these models are used to:

- **Forecast pre-trade costs** for a parent order of size $Q$.
- **Decide whether to trade** by comparing expected alpha to expected impact.
- **Size metaorders** so impact cost is a target fraction of alpha or spread.
- **Detect toxic flow** (high $\lambda$ regimes signal adverse selection).

## Calibration approach

### Kyle's lambda from daily data

For each ticker, on a rolling window of $N$ days (typical $N = 60$ or $N = 120$):

1. Compute daily returns $r_t = \ln(C_t / C_{t-1})$.
2. Build a daily **signed-volume proxy**. Without TAQ, use the close-to-open sign on the same day or the close-to-close sign:
   $$Q_t^{\text{signed}} \approx \text{sign}(C_t - O_t) \cdot V_t$$
   This is the standard Lee-Ready proxy at daily frequency.
3. Regress $r_t = \lambda \cdot Q_t^{\text{signed}} + \epsilon_t$ by OLS (no intercept, or with intercept for drift). Use heteroskedasticity-robust (HC1) standard errors.
4. Report $\hat\lambda$ in units of return per share, plus its 95% CI.

A useful normalization: $\hat\lambda \cdot V_{\text{daily}}^{\text{avg}}$, the return per "one daily-volume" of one-sided flow.

### Square-root law: $Y$ from execution data

If proprietary execution data is available:

1. Collect a panel of metaorders $i = 1, \dots, M$ with size $Q_i$, executed during a day with volume $V_i$ and volatility $\sigma_i$.
2. Compute realized arrival slippage $s_i = \ln(\bar P_i / P_{i,0})$ where $\bar P_i$ is VWAP of the metaorder and $P_{i,0}$ is the arrival mid.
3. Fit $s_i = Y \cdot \sigma_i \sqrt{Q_i / V_i} + \nu_i$ by OLS. Many papers also fit a power $\alpha$: $s_i = Y \sigma_i (Q_i / V_i)^\alpha$ and report both $\hat Y, \hat\alpha$ — empirically $\hat\alpha \approx 0.5$.

Without execution data — the typical yfinance setting — use a literature prior: $Y \approx 1$ for US large-cap equities, $Y \approx 0.5$–$1$ for liquid futures. Use the law as a **pre-trade estimator** rather than a fitted model.

### Propagator: out of reach with yfinance

The propagator kernel $G$ requires tick-level signed trades and a sufficiently long time series of them — typically TAQ or LOBSTER data. yfinance does not deliver this, so the propagator model in this stack remains a documented placeholder.

## Algorithm outline

### Algorithm A — Daily Kyle-lambda estimation (yfinance)

```
INPUT: ticker s, lookback N days, end date d
OUTPUT: lambda_hat, se, ci_low, ci_high, n_obs

1.  df = yf.Ticker(s).history(period=f"{N+5}d", end=d, auto_adjust=False)
2.  df = df.dropna(subset=["Open","Close","Volume"])
3.  df["ret"]    = log(df.Close / df.Close.shift(1))
4.  df["sgn"]    = sign(df.Close - df.Open)            # daily Lee-Ready proxy
5.  df["sgnVol"] = df.sgn * df.Volume                  # signed shares
6.  df = df.dropna().tail(N)
7.  Fit OLS: ret ~ sgnVol (no intercept), HC1 SE
8.  lambda_hat = coefficient on sgnVol
9.  Optionally normalize: lambda_norm = lambda_hat * df.Volume.mean()
10. Return lambda_hat, se(lambda_hat), 95% CI, len(df)
```

A variant uses **dollar-signed volume** $Q_t^\$ = \text{sign}(C_t - O_t) \cdot C_t \cdot V_t$; in that case $\lambda$ has units of return per dollar.

### Algorithm B — Intraday Kyle-lambda (5-minute bars)

yfinance offers 1m bars for the last 7 days, 5m bars for ~60 days, and 1h bars for ~730 days.

```
INPUT: ticker s, interval (e.g. "5m"), lookback days
OUTPUT: per-day lambda_hat, plus pooled estimate

1.  bars = yf.Ticker(s).history(period="60d", interval="5m")
2.  bars["ret"]    = log(bars.Close / bars.Close.shift(1))
3.  bars["sgn"]    = sign(bars.Close - bars.Open)
4.  bars["sgnVol"] = bars.sgn * bars.Volume
5.  For each trading day:
        Fit OLS on within-day bars: ret ~ sgnVol
        Store lambda_d
6.  Pooled estimate: stack all bars, fit panel OLS with day fixed effects
7.  Report median lambda_d (robust to outlier days)
```

Intraday $\lambda$ is **much** closer to the textbook microstructure quantity than daily $\lambda$, because the bar-level sign is closer to the true tick-rule sign.

### Algorithm C — Pre-trade cost from square-root law

```
INPUT: ticker s, parent order size Q (shares), side ∈ {+1,-1}, horizon
OUTPUT: expected impact in bps, expected impact in $

1.  hist = yf.Ticker(s).history(period="120d")
2.  sigma_daily = hist.Close.pct_change().std()
3.  V_daily     = hist.Volume.rolling(20).mean().iloc[-1]   # 20-day ADV
4.  P_now       = hist.Close.iloc[-1]
5.  Y           = 1.0    # literature prior, US large-cap equities
6.  impact_rel  = Y * sigma_daily * sqrt(Q / V_daily)
7.  impact_bps  = 1e4 * impact_rel
8.  impact_dollars = side * impact_rel * P_now * Q
9.  Return impact_bps, impact_dollars
```

Two sanity checks belong in production code: warn if $Q / V > 0.10$ (the law has been calibrated on $Q/V \le 0.1$; extrapolation is dangerous), and warn if the asset's $\sigma$ is unusual relative to its peer group.

### Algorithm D — Combined pre-trade estimator

Use Kyle's $\lambda$ for **small** orders and the square-root law for **large** orders. The crossover point is where the two estimates agree:

$$\lambda \cdot Q^* = Y \sigma \sqrt{Q^* / V} \implies Q^* = \frac{Y^2 \sigma^2}{\lambda^2 V}$$

Below $Q^*$, use $\lambda Q$. Above $Q^*$, use the square-root law. This is the textbook "Kyle for small, Bouchaud for large" recipe.

## yfinance data requirements

| Field | yfinance call | Used for |
|---|---|---|
| Daily OHLCV | `Ticker.history(period="120d")` | Kyle $\lambda$ on daily data; $\sigma$ and $V$ for square-root law |
| Intraday 5m OHLCV | `Ticker.history(period="60d", interval="5m")` | Intraday Kyle $\lambda$; intraday volume profile |
| Intraday 1m OHLCV | `Ticker.history(period="7d", interval="1m")` | Highest-resolution $\lambda$ estimation available |
| Hourly OHLCV | `Ticker.history(period="730d", interval="1h")` | Long-history intraday $\lambda$ |

**Important workarounds:**

1. **No tick data, no trade signs.** Replace true Lee-Ready with the bar-level proxy $\text{sign}(C - O)$. Validate against published estimates for a few benchmark names (SPY, AAPL) to confirm the proxy is in the right order of magnitude.
2. **Bid-ask spread.** yfinance does not provide intraday spread series. For US large-caps assume 1 bp half-spread; for less liquid names use the Corwin-Schultz daily-OHLC spread estimator:
   $$\hat S = 2 \frac{e^\alpha - 1}{1 + e^\alpha}, \quad \alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}$$
   with $\beta = \mathbb{E}[(\ln H_t / L_t)^2 + (\ln H_{t+1} / L_{t+1})^2]$ and $\gamma = \mathbb{E}[(\ln H_{t,t+1} / L_{t,t+1})^2]$.
3. **Volume staleness.** yfinance volume on the most recent (incomplete) trading day is provisional. Use only completed sessions.
4. **Splits and dividends.** Use `auto_adjust=False` for volume comparisons and adjust price returns separately, or use `auto_adjust=True` and accept that historical share counts are scaled.

## Calibration / refit frequency

- **Daily Kyle $\lambda$:** refit weekly on a 60- or 120-day rolling window. Per-ticker $\lambda$ is noisy; pool across a peer group (same sector, similar ADV) for stability.
- **Intraday Kyle $\lambda$:** refit weekly on 30-60 days of 5-minute bars.
- **Square-root prefactor $Y$:** if execution data exists, recalibrate quarterly. Otherwise hold $Y \approx 1$ constant and adjust only $\sigma$ and $V$.
- **Volume average $V_{\text{daily}}$:** roll a 20-day window, refit daily.
- **Volatility $\sigma$:** for pre-trade estimation, prefer a 20-30 day realized vol or a GARCH(1,1) one-day-ahead forecast.

A regime check is essential: if recent realized $\lambda$ deviates from the rolling estimate by more than $2\sigma$, flag the asset as in a microstructure regime change (earnings, M&A, index inclusion) and widen cost estimates.

## Validation and diagnostics

1. **Goodness-of-fit on $\lambda$.** Report $R^2$ from the return-on-signed-volume regression. Cross-sectionally, $R^2$ should be 5-30% for liquid US equities at daily frequency; higher at intraday.
2. **Sign stability.** $\hat\lambda$ must be positive. A negative estimate is a flag for a stale proxy or a thin-trading regime.
3. **Cross-sectional reasonableness.** $\lambda$ should be **decreasing in ADV** and **increasing in $\sigma$**. Plot $\log \lambda$ against $\log(\sigma / V)$ — Kyle's theory predicts slope 1.
4. **Square-root residuals.** On execution data, plot residuals $s_i - Y \sigma_i \sqrt{Q_i/V_i}$ against $Q_i / V_i$. A residual trend signals the wrong power; refit $\alpha$.
5. **Walk-forward backtest.** Use $\hat\lambda$ and $\hat Y$ from window $[t-N, t]$ to forecast cost on $[t, t+h]$. Compute hit rate of forecast within $\pm 25\%$ of realized; should be $\ge 60\%$ for routine sizes.
6. **Stationarity.** Use Chow tests at known structural breaks (March 2020, GameStop January 2021, election dates) to confirm subsample stability.

## Connections to other models

- **Layer 5 — Almgren-Chriss (Model 16):** consumes the temporary-impact coefficient $\eta$, which can be derived from Kyle's $\lambda$ as $\eta \approx \lambda$ (in the limit where permanent and temporary are merged) or from the square-root law as $\eta \approx Y \sigma \sqrt{1/V}$.
- **Layer 5 — Propagator execution:** the propagator kernel $G$ generalizes the impact picture and is required for long-horizon execution under long-memory order flow.
- **Layer 4 — Volatility forecasting (GARCH, HAR-RV):** supplies the $\sigma$ input.
- **Layer 3 — Liquidity-adjusted factor models:** $\lambda$ and the square-root prefactor are themselves candidate liquidity factors (Pastor-Stambaugh, Amihud illiquidity).
- **Layer 6 — Risk:** liquidity-adjusted VaR adds the expected liquidation cost as a deterministic add-on to standard VaR.

## Limitations and failure modes

1. **Trade-sign proxy noise.** Bar-level $\text{sign}(C - O)$ is correct $\approx 70\%$ of the time at 1-minute frequency, but only $\approx 55\%$ at daily frequency. The result is a downward-biased $\hat\lambda$ (attenuation).
2. **Extrapolation beyond calibrated range.** The square-root law is calibrated on $Q/V \in [10^{-4}, 0.1]$. For $Q/V > 0.1$ impact grows faster than $\sqrt{Q}$ (regime change). Refuse to forecast above this range.
3. **Regime dependence.** $\lambda$ doubles or more during stress episodes. A static estimate underprices crisis-day liquidity. Use a volatility-regime gate.
4. **Cross-sectional heterogeneity.** Small-cap, low-ADV names violate Gaussian assumptions and have heavy-tailed impact distributions. The square-root law remains roughly correct but $Y$ varies more.
5. **Hidden orders and dark pools.** The "$V_{\text{daily}}$" in the denominator should ideally include hidden liquidity; using only lit volume overstates impact for names with heavy dark-pool flow.
6. **Mechanical-impact vs. information-impact confound.** A large $\hat\lambda$ on a day with news mixes adverse selection and mechanical depth. Subsample by news flags if possible.
7. **No tick data.** Without TAQ, the textbook Kyle estimation cannot be reproduced and all results are proxy-based. Document this explicitly in any production deployment.

## References

- Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315-1335.
- Glosten, L. R. and Milgrom, P. R. (1985). "Bid, ask and transaction prices in a specialist market." *Journal of Financial Economics*, 14, 71-100.
- Almgren, R., Thum, C., Hauptmann, E., and Li, H. (2005). "Direct Estimation of Equity Market Impact." *Risk*, 18.
- Bouchaud, J.-P., Farmer, J. D., and Lillo, F. (2009). "How markets slowly digest changes in supply and demand." *Handbook of Financial Markets: Dynamics and Evolution*.
- Toth, B., Lemperiere, Y., Deremble, C., De Lataillade, J., Kockelkoren, J., and Bouchaud, J.-P. (2011). "Anomalous price impact and the critical nature of liquidity in financial markets." *Physical Review X*, 1, 021006.
- Bouchaud, J.-P., Gefen, Y., Potters, M., and Wyart, M. (2004). "Fluctuations and response in financial markets: the subtle nature of 'random' price changes." *Quantitative Finance*, 4(2), 176-190.
- Lillo, F., Mike, S., and Farmer, J. D. (2005). "Theory for long memory in supply and demand." *Physical Review E*, 71, 066122.
- Hasbrouck, J. (2009). "Trading costs and returns for US equities: estimating effective costs from daily data." *Journal of Finance*, 64(3), 1445-1477.
- Corwin, S. A. and Schultz, P. (2012). "A simple way to estimate bid-ask spreads from daily high and low prices." *Journal of Finance*, 67(2), 719-760.
