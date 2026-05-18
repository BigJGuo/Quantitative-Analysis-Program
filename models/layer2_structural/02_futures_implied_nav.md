# Futures-Implied NAV

> Multi-source weighted fair-value model that uses liquid futures, ADRs, and FX forwards as proxies for international ETF baskets when the underlying markets are closed. Layer 2 — Structural / Cross-Sectional. Primary use: real-time pricing of EFA, EEM, EWJ, FXI, VWO and similar international ETFs during US trading hours.

## Mathematical formulation

When an ETF's underlying constituents trade in markets that are closed, the published iNAV is stale. The futures-implied fair value replaces the stale basket with a regression-based projection from liquid hedging instruments.

### Single-factor form

The simplest expression updates the close-of-foreign-market NAV by the return on a liquid futures proxy and the contemporaneous FX move:

$$
\text{FV}_t \;=\; \text{NAV}_{\text{close}} \cdot \left[ 1 \;+\; \beta_{\text{basket}} \cdot \frac{F_t - F_{\text{close}}}{F_{\text{close}}} \right] \cdot \frac{\text{FX}_t}{\text{FX}_{\text{close}}}
$$

where:

- $\text{NAV}_{\text{close}}$ — official NAV published by the ETF issuer at the close of the *foreign* market (e.g., Tokyo close for EWJ).
- $F_t$ — current price of a liquid futures contract correlated with the basket (Nikkei 225, FTSE 100, Hang Seng, MSCI EM).
- $F_{\text{close}}$ — futures price at the time NAV was struck.
- $\text{FX}_t / \text{FX}_{\text{close}}$ — currency adjustment between the foreign currency and USD.
- $\beta_{\text{basket}}$ — empirical regression beta of basket returns on futures-proxy returns (typically 0.85–1.05).

### Multi-factor stressed-market form

In a stressed market (March 2020, the 2015 Chinese ETF dislocation, Brexit) a single futures proxy is insufficient. The full model is a weighted combination:

$$
\text{FV}_{\text{ETF},t} \;=\; w_{\text{ETF}} \cdot P_{\text{secondary},t} \;+\; w_{\text{NAV}} \cdot \text{NAV}_{\text{official}} \;+\; w_{\text{Hedge}} \cdot \sum_{j=1}^{J} \beta_j \cdot P_{\text{hedge},j,t}
$$

with $w_{\text{ETF}} + w_{\text{NAV}} + w_{\text{Hedge}} = 1$, and the weights are time-varying.

- $P_{\text{secondary},t}$ — the ETF's own secondary-market mid (weight is non-zero because the ETF itself is a noisy price discovery instrument).
- $\text{NAV}_{\text{official}}$ — the most recently published official NAV.
- $\sum_j \beta_j \cdot P_{\text{hedge},j,t}$ — linear combination of liquid hedges. Index $j$ runs over futures, ADRs, country ETFs, currency forwards.

### Derivation of the single-factor form

Let $r^B_t = \log(\text{NAV}_t / \text{NAV}_{\text{close}})$ be the unobservable basket log-return since the foreign close. Decompose into a futures-driven and idiosyncratic part:

$$
r^B_t \;=\; \beta \cdot r^F_t \;+\; r^{\text{FX}}_t \;+\; \eta_t, \qquad \eta_t \sim (0, \sigma_\eta^2)
$$

where $r^F_t$ is the futures log-return and $r^{\text{FX}}_t$ is the currency log-return. The minimum-mean-squared-error estimator of the basket level is:

$$
\mathbb{E}[\text{NAV}_t \mid F_t, \text{FX}_t] \;=\; \text{NAV}_{\text{close}} \cdot \exp(\beta \cdot r^F_t + r^{\text{FX}}_t)
$$

A first-order expansion of the exponential gives the working formula. The error variance is $\sigma_\eta^2 \cdot \text{NAV}_{\text{close}}^2$, which sets the model's confidence band.

### Multivariate regression

For $J$ hedges, the basket return is regressed in a vector model:

$$
r^B_t \;=\; \boldsymbol{\beta}^\top \mathbf{r}^H_t \;+\; \eta_t
$$

with $\boldsymbol{\beta} \in \mathbb{R}^J$ and $\mathbf{r}^H_t \in \mathbb{R}^J$. The OLS estimator (computed on overlapping US-hour windows) is:

$$
\hat{\boldsymbol{\beta}} \;=\; (\mathbf{R}_H^\top \mathbf{R}_H)^{-1} \mathbf{R}_H^\top \mathbf{r}_B
$$

A ridge-regularized variant is preferable when hedges are correlated:

$$
\hat{\boldsymbol{\beta}}_\lambda \;=\; (\mathbf{R}_H^\top \mathbf{R}_H + \lambda \mathbf{I})^{-1} \mathbf{R}_H^\top \mathbf{r}_B
$$

### Bayesian weight blending

The weights $(w_{\text{ETF}}, w_{\text{NAV}}, w_{\text{Hedge}})$ in the multi-source form are derived from the inverse-variance of each source:

$$
w_k \;\propto\; \frac{1}{\sigma_k^2}, \qquad \sum_k w_k = 1
$$

where $\sigma_k^2$ is the conditional variance of source $k$'s estimate of the true fair value. During normal hours $\sigma_{\text{Hedge}}^2 \ll \sigma_{\text{NAV}}^2$, so most weight is on hedges. When hedges blow out (futures limit-down, exchange halt) $\sigma_{\text{Hedge}}^2$ explodes and weight shifts to $\text{NAV}_{\text{official}}$ and the ETF's own price.

## Intuition and what this model addresses

For US-listed international ETFs, the published iNAV is essentially frozen during US hours. The constituents — Japanese equities for EWJ, Hong Kong equities for FXI, EM-wide for EEM — last printed at their local close, which can be 6–14 hours earlier. Anything that has happened in the world since then (currency moves, futures repricing, US macro data) is not reflected in iNAV.

Three empirical phenomena force the model:

1. **Persistent intraday premium/discount cycles.** EFA and EEM consistently trade at 10–80 bps away from iNAV during US hours, with the deviation reverting on the next foreign market open. This is not an arbitrage — it is the market correctly pricing the missing information.
2. **Asymmetric information at the close.** US closing prints (4 PM ET) for international ETFs encode 4–10 hours of news that the foreign markets will react to on their next open. The futures-implied fair value extracts the consensus on what the foreign opening prices will be.
3. **Stress-period regime shifts.** Under normal conditions a single futures proxy explains 90%+ of basket variance. In a stress event (limit moves, exchange halts), the relationship breaks: hedges become unreliable, the ETF's own secondary market becomes the most informative price, and weights must adapt.

The model addresses the failure of static iNAV by providing a Bayesian update: take the most recent reliable NAV, propagate it forward by the futures-implied move, and blend in the ETF's own price as a sanity check.

This is the single largest source of structural alpha in international ETF market making. The bid/ask Jane Street shows on EEM at 3:55 PM ET is a quote on its internal futures-implied fair value, not on the published iNAV.

## Calibration approach

The calibration target is $\boldsymbol{\beta}$ — the vector of regression coefficients on hedges.

**Estimation window.** Use overlapping US-hour windows. For each US trading session $d$, compute the basket return as $r^B_d = (\text{NAV}_{\text{open of day } d+1}^{\text{foreign}} / \text{NAV}_{\text{close of day } d}^{\text{foreign}}) - 1$ — i.e., the change in NAV across the overnight US session. Pair this with hedge returns over the same US window: $r^H_d = (P^{\text{hedge}}_{4\text{PM ET}, d} / P^{\text{hedge}}_{\text{NAV strike}, d}) - 1$.

**Loss function.** Minimize MSE on the held-out next-day overnight basket return:

$$
\mathcal{L}(\boldsymbol{\beta}) \;=\; \sum_d \left( r^B_d - \boldsymbol{\beta}^\top \mathbf{r}^H_d \right)^2 \;+\; \lambda \|\boldsymbol{\beta}\|_2^2
$$

with $\lambda$ chosen by leave-one-day-out cross-validation on a 90-day rolling window.

**Constraints.** For interpretability, constrain $0 \le \beta_j \le 1.5$ and $\sum_j \beta_j \le 1.2$ (basket is not allowed to be implausibly long-overall). Solve as a quadratic program when constraints bind.

**Robustness.** Use Huber loss or trim the top/bottom 1% of days to reduce influence of regime breaks. Reject any day on which a hedge had a circuit-breaker halt.

**FX handling.** FX is treated separately and not regressed: the term $\text{FX}_t / \text{FX}_{\text{close}}$ is applied multiplicatively. This avoids contaminating $\beta$ with currency loadings that change discontinuously across sessions.

**Weight calibration $(w_{\text{ETF}}, w_{\text{NAV}}, w_{\text{Hedge}})$.** Estimate $\sigma_k^2$ from realized prediction errors over the last 30 days. Recompute weights nightly.

## Algorithm outline

```
Inputs:
    etf_ticker            : e.g. "EEM"
    hedge_tickers H       : list of futures, ETFs, ADRs (e.g. ["^N225","^HSI","^FTSE","EWZ","FXI","DXY"])
    fx_tickers            : currency proxies for FX adjustment
    history_days          : 90
    refit_period          : 1 day

Daily nightly refit (after foreign close, before US open):

1. Fetch official NAV history (issuer file) for last `history_days` business days.
2. For each pair of adjacent foreign trading days (d, d+1):
       r_B[d] = NAV_open(d+1) / NAV_close(d) - 1
       For each hedge j in H:
           r_H[d, j] = P_hedge_j(US 4PM ET, day d) / P_hedge_j(NAV strike time, day d) - 1
3. Run ridge regression r_B ~ R_H, choose lambda by 5-fold CV.
4. Apply constraints: clip beta_j to [0, 1.5]; QP solve if sum(beta) > 1.2.
5. Compute residuals e_d = r_B[d] - beta @ r_H[d].
6. Estimate variances:
       sigma_Hedge^2 = Var(e)
       sigma_NAV^2   = (time since NAV strike) * intraday_basket_variance  (grows with staleness)
       sigma_ETF^2   = realized variance of ETF mid around fair value (60-min lookback, online)
7. Persist beta, sigma_k^2 for the next session.

Intraday hot loop (every 1-5 s during US hours):

1. Fetch hedges and FX
   - For j in H:  P_j_t = yf.Ticker(hedge_j).history(period="1d", interval="1m").iloc[-1]["Close"]
   - For currency c: FX_c_t similarly.

2. Compute hedge returns since NAV strike
   - For j: rH_j_t = P_j_t / P_j_close - 1

3. Compute futures-implied basket return
   - rB_hat = beta @ rH_t

4. FX adjustment
   - fx_ratio = FX_t / FX_close

5. Single-factor fair value
   - FV_hedge = NAV_close * (1 + rB_hat) * fx_ratio

6. Fetch official NAV (most recent) and ETF secondary mid
   - NAV_off = NAV_close * fx_ratio   (no intraday update during foreign close)
   - M_t     = yf.Ticker(etf_ticker).history(period="1d", interval="1m").iloc[-1] mid

7. Update weights (inverse-variance blend, normalized)
   - w_k = (1/sigma_k^2) / sum_j (1/sigma_j^2)

8. Multi-source FV
   - FV_t = w_ETF * M_t + w_NAV * NAV_off + w_Hedge * FV_hedge

9. Generate signals
   - premium = (M_t - FV_t) / FV_t
   - if |premium| > threshold + cost_band:  trade signal
   - publish bid = FV_t - half_spread, ask = FV_t + half_spread

10. Log diagnostics: (t, FV_t, M_t, premium, w_*, residual, hedge_used).
```

## yfinance data requirements

| Asset | Ticker examples | Notes |
|-------|-----------------|-------|
| International ETF | `EFA`, `EEM`, `VWO`, `EWJ`, `FXI`, `EWZ`, `EWG`, `EWY` | Pull 1-minute bars during US hours. |
| Japan futures | `^N225` (Nikkei index); for futures use CME `NKD=F` | yfinance has spot index; CME mini-Nikkei is the tradable. |
| Hong Kong | `^HSI` (Hang Seng index); `HHI=F` for HSCEI futures | Both are 1-min. |
| UK / Europe | `^FTSE`, `^STOXX50E`, futures `Z=F` (FTSE), `FESX=F` (Euro Stoxx) | Local closes overlap part of US AM. |
| Emerging markets | MSCI EM futures not in yfinance directly; use EEM itself plus liquid country ETFs (`EWZ`, `EWY`, `EWT`, `INDA`, `FXI`) as a synthetic basket | This is a yfinance limitation. |
| FX spot | `JPYUSD=X`, `GBPUSD=X`, `EURUSD=X`, `CNYUSD=X` | Pull current spot for FX adjustment term. |
| Risk-free rate | `^IRX` | 13-week T-bill, used for any term-structure carry adjustments. |
| ETF NAV | Not available in yfinance — pull from issuer JSON (iShares, Vanguard) | Treat as exogenous daily input. |

**Limitations and workarounds:**

- yfinance does not carry MSCI EM futures directly. Synthesize an EM proxy from a basket of country ETFs weighted to match EEM's country exposure (typically: 30% CN, 15% TW, 13% IN, 12% KR, 5% BR, plus smaller weights).
- Free-tier yfinance lags by 15 minutes for many tickers. The model still works for backtesting and post-trade analysis; live deployment needs a paid feed.
- Futures contracts roll quarterly. `=F` tickers in yfinance generally point to the front-month, but check `Ticker.info["expireDate"]` and roll-adjust before computing returns spanning a roll.
- The issuer's official NAV is published once daily after the foreign close; for intraday refresh, pull the issuer's published `iNAV` field instead and treat as the staleness anchor.

## Calibration / refit frequency

- **Regression $\boldsymbol{\beta}$:** refit nightly after the foreign market closes. The basket composition can change daily but $\boldsymbol{\beta}$ is typically stable to within $\pm 0.05$ over a week.
- **Weights $w_k$:** recompute nightly from a rolling 30-day window of realized residuals.
- **Bid/ask half-spread:** weekly, from realized round-trip costs.
- **Intraday model evaluation:** every 1–5 seconds.

Faster $\boldsymbol{\beta}$ refits (intraday) are tempting but unstable — the underlying basket changes too slowly to support intraday parameter updates, and intraday refits over-fit to short-horizon noise.

## Validation and diagnostics

1. **Next-day overnight P&L.** The acid test: after the foreign market reopens, compare the model's $\text{FV}_t$ at 4 PM ET against the realized NAV at the next foreign open. The mean absolute error should be < 30 bps for liquid country ETFs (EWJ, EWG, EWU) and < 60 bps for broad EM (EEM, VWO).
2. **R-squared on basket return regression.** $R^2 \ge 0.85$ is the calibration acceptance threshold. Below 0.75, the hedge set is misspecified.
3. **Residual stationarity.** ADF test on residuals $e_d$ should reject unit root. A non-stationary residual means a missing factor (typically a country or sector that has detached from the broad proxy).
4. **Weight stability.** $w_{\text{Hedge}}$ should sit around 0.7–0.9 in normal markets. Sudden shifts to $w_{\text{NAV}} > 0.5$ flag a stress regime and trigger manual review.
5. **Cross-hedge consistency.** Compute $\text{FV}_t$ using two disjoint hedge sets (e.g., futures-only vs ADR-only). The two estimates should agree to within 20 bps; large divergence indicates a hedge has broken.
6. **Backtest premium decay.** Verify that the model's $(M_t - \text{FV}_t)/\text{FV}_t$ signal mean-reverts on the next foreign open. The half-life should be the time-until-foreign-open.

## Connections to other models

- **Model 1 (iNAV / ETF Fair Value):** Model 2 replaces the stale-component term $\tilde{P}_{i,t}$ inside Model 1 with a hedge-implied estimate. They are operationally coupled: Model 1 handles the basket accounting; Model 2 handles the basket projection.
- **Model 5 (Cost-of-Carry):** Used to roll-adjust futures prices when the proxy rolls across a contract month. The carry differential is consumed by Model 2 in computing $r^F_t$.
- **Model 4 (Factor Models / PCA):** When the hedge set $H$ contains many overlapping instruments (e.g., 12 country ETFs), reduce $H$ to its first few principal components before regression to stabilize $\boldsymbol{\beta}$.
- **Layer 3 stat-arb models:** The residual $e_d$ is itself a tradeable signal — international ETFs that systematically over- or under-shoot their futures-implied fair value are pair candidates.
- **Layer 4 execution:** Drives quote placement in international ETF market making.

## Limitations and failure modes

1. **Beta regime shift.** A change in the basket's sector composition (e.g., EEM rebalance increasing China weight) shifts $\boldsymbol{\beta}$ discontinuously. The model takes 10–30 days to adapt. Mitigation: trigger an emergency refit on any index rebalance announcement.
2. **Futures dislocation.** Limit-down/up moves in futures break the linear projection. Mitigation: monitor futures bid/ask width — when it exceeds 3× normal, downgrade $w_{\text{Hedge}}$.
3. **FX detachment.** During FX market stress (e.g., emerging market FX during a crisis), the FX move can overshoot the basket adjustment. Mitigation: cap the FX-adjustment magnitude per session, or model FX with a separate regime indicator.
4. **Currency hedging in the ETF.** Some ETFs are currency-hedged (HEFA, HEWJ); the FX term must be turned off. Always check the prospectus.
5. **Holiday mismatches.** When the foreign market is closed for a local holiday (Golden Week, Lunar New Year), the basket has no fresh information for 1–7 days. The model continues to extrapolate from futures, but uncertainty grows linearly with time-since-close.
6. **Stale issuer NAV.** If the issuer fails to publish NAV for a day (rare but happens around month-end and corporate actions), the anchor $\text{NAV}_{\text{close}}$ is missing. Use the prior-day NAV plus an extrapolation, and widen quotes.
7. **Single-asset shocks.** A single mega-cap constituent (e.g., Samsung in EWY) can move 5% on news with no corresponding futures move. The futures proxy under-estimates the basket move. Mitigation: include ADRs of the top 5 constituents as separate hedges.

## References

- Petajisto, A. (2017). "Inefficiencies in the Pricing of Exchange-Traded Funds." *Financial Analysts Journal*.
- Madhavan, A., and Sobczyk, A. (2016). "Price Dynamics and Liquidity of Exchange-Traded Funds." *Journal of Investment Management*.
- Engle, R. F. and Sarkar, D. (2006). "Premiums-Discounts and Exchange Traded Funds." *Journal of Derivatives*.
- Levy, A. and Lieberman, O. (2013). "Overreaction of Country ETFs to US Market Returns." *Journal of Banking & Finance*.
- Pennathur, A., Delcoure, N. and Anderson, D. (2002). "Diversification Benefits of iShares and Closed-End Country Funds." *Journal of Financial Research*.
