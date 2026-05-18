# Merton-KMV Structural Credit Model

> Structural credit-risk model that treats a firm's equity as a European call option on its asset value, struck at the face value of debt; used to infer asset value, asset volatility, distance-to-default, and the implied probability of default from observable stock prices and balance-sheet liabilities. Layer 2 — Structural / Cross-Sectional. Primary use: distance-to-default and expected default frequency on individual stocks, capital-structure arbitrage signal generation, and cross-check on reduced-form CDS-implied PDs.

## Mathematical formulation

### Merton's structural framework

Merton (1974) models the firm as having a single zero-coupon liability of face value $D$ maturing at time $T$. The firm's asset value $V_t$ follows a geometric Brownian motion under the physical measure $\mathbb{P}$:

$$
dV_t \;=\; \mu V_t \, dt \;+\; \sigma_V V_t \, dW_t
$$

where $\mu$ is the asset drift, $\sigma_V$ is the asset volatility, and $W_t$ is a standard Brownian motion. Default occurs at $T$ if and only if $V_T < D$. Equity holders receive $\max(V_T - D, 0)$ at maturity, so equity is a European call on $V$ with strike $D$:

$$
E_t \;=\; V_t \cdot N(d_1) \;-\; D \cdot e^{-r(T-t)} \cdot N(d_2)
$$

where:

$$
d_1 \;=\; \frac{\ln(V_t / D) + (r + \tfrac{1}{2}\sigma_V^2)(T-t)}{\sigma_V \sqrt{T-t}}, \qquad d_2 \;=\; d_1 - \sigma_V \sqrt{T-t}
$$

and:

- $E_t$ — market value of equity at time $t$ (observable from market cap).
- $V_t$ — firm asset value (unobservable).
- $D$ — face value of debt at maturity (proxied by short-term debt + a fraction of long-term debt).
- $r$ — risk-free rate to horizon $T$.
- $\sigma_V$ — instantaneous asset volatility (unobservable).
- $N(\cdot)$ — standard normal CDF.

### Equity volatility link

Because $E_t = f(V_t, t)$ is a smooth function of $V_t$, Ito's lemma gives a deterministic link between equity and asset volatilities:

$$
\sigma_E \cdot E_t \;=\; \frac{\partial E}{\partial V} \cdot \sigma_V \cdot V_t \;=\; N(d_1) \cdot \sigma_V \cdot V_t
$$

Rearranging:

$$
\sigma_E \;=\; \frac{V_t}{E_t} \cdot N(d_1) \cdot \sigma_V
$$

This identity is the second equation in the iterative KMV solve.

### Simultaneous system

Treating $E_t$ and $\sigma_E$ as observed and $V_t, \sigma_V$ as unknown, KMV solves the two-equation nonlinear system:

$$
\begin{cases}
E_t \;=\; V_t N(d_1) - D e^{-r(T-t)} N(d_2) \\[4pt]
\sigma_E \;=\; \dfrac{V_t}{E_t} N(d_1) \sigma_V
\end{cases}
$$

for $(V_t, \sigma_V)$. Typically $T - t = 1$ year (the standard KMV convention).

### Distance-to-default

Once $(V_t, \sigma_V)$ are recovered, define the Distance-to-Default (DD) under the physical measure:

$$
\text{DD} \;=\; \frac{\ln(V_t / D) + (\mu - \tfrac{1}{2}\sigma_V^2)(T-t)}{\sigma_V \sqrt{T-t}}
$$

Note the substitution $r \to \mu$: DD is computed under the *physical* (real-world) drift, not the risk-neutral drift. The drift $\mu$ is typically estimated as the trailing mean log-return on assets.

### Probability of default

Under the lognormality assumption:

$$
\text{PD}_{\text{Merton}} \;=\; \mathbb{P}(V_T < D) \;=\; N(-\text{DD})
$$

KMV's empirical contribution was to *not* trust this Gaussian PD. Instead, they built a mapping from DD to an empirical Expected Default Frequency (EDF) using a proprietary database of historical defaults:

$$
\text{EDF}(\text{DD}) \;=\; \hat{F}_{\text{empirical}}(\text{DD})
$$

In practice the empirical EDF curve is monotone decreasing in DD, fatter-tailed than the Gaussian, and lower-bounded by a floor (typically 1–5 bps) to avoid implausibly small numbers.

### KMV default point

KMV's default point $D$ is *not* total liabilities. Empirical work showed that firms default when assets fall below short-term debt plus roughly half of long-term debt:

$$
D \;=\; \text{Short-Term Debt} \;+\; 0.5 \cdot \text{Long-Term Debt}
$$

The 0.5 weight reflects that not all long-term debt is callable at the moment of distress.

### Risk-neutral version

For pricing applications, replace $\mu$ with $r$ in the DD formula:

$$
\text{DD}^{\mathbb{Q}} \;=\; \frac{\ln(V_t / D) + (r - \tfrac{1}{2}\sigma_V^2)(T-t)}{\sigma_V \sqrt{T-t}}, \qquad \text{PD}^{\mathbb{Q}} \;=\; N(-\text{DD}^{\mathbb{Q}})
$$

The risk-neutral PD is what feeds into theoretical credit spreads via $s \approx -\frac{1}{T-t} \ln(1 - \text{PD}^{\mathbb{Q}} \cdot \text{LGD})$, where LGD is loss-given-default.

## Intuition and what this model addresses

Reduced-form credit models (Jarrow-Turnbull, Duffie-Singleton) treat default as an exogenous hazard process — they require CDS or bond data to calibrate and say nothing about *why* a firm defaults. Merton's structural insight is that equity holders have the option to walk away when assets fall below liabilities; therefore equity must be a call option on the firm, and the call's parameters can be backed out from the option price (equity market cap) and its observed volatility (equity vol).

The model addresses several failures of simpler approaches:

1. **Equity-only models miss the leverage signal.** A stock down 30% on light leverage is not the same credit story as a stock down 30% on heavy leverage. Merton-KMV explicitly normalizes by leverage through the $V/D$ ratio.
2. **Balance-sheet-only models miss the equity vol signal.** Two firms with identical balance sheets but different equity volatilities have different default risk. Merton-KMV uses $\sigma_E$ to extract $\sigma_V$.
3. **Reduced-form models need credit market data.** For private companies or those without traded CDS, structural models are the only way to get a PD.
4. **Capital-structure arbitrage.** When the structural model's PD diverges materially from the CDS-implied PD, there is a tradeable disagreement between the equity and credit markets.

For equity-focused systems built on yfinance, Merton-KMV is the natural credit lens: it consumes only equity prices and balance-sheet data, both readily available.

## Calibration approach

The core calibration problem is the simultaneous solve for $(V_t, \sigma_V)$ from $(E_t, \sigma_E, D, r, T)$. Two approaches are standard.

### Iterative KMV procedure

This is the original Moody's KMV approach:

1. Initialize $\sigma_V^{(0)} = \sigma_E \cdot E_t / (E_t + D)$.
2. Given $\sigma_V^{(k)}$, invert the Black-Scholes formula (e.g., by Newton on $V$) to recover $V_t^{(k+1)}$ from $E_t$.
3. Build a time series of $V_t^{(k+1)}$ over the calibration window (typically 1 year, daily).
4. Recompute $\sigma_V^{(k+1)} = \text{stdev}(\log V_t^{(k+1)} - \log V_{t-1}^{(k+1)}) \cdot \sqrt{252}$.
5. Repeat until $|\sigma_V^{(k+1)} - \sigma_V^{(k)}| < 10^{-5}$.

This converges quickly (typically 5–15 iterations) and is preferred because it uses the *empirical* asset return distribution rather than the model-implied one.

### Joint nonlinear solve

For a single point estimate (no time series), solve the 2-equation system directly. Define:

$$
g_1(V, \sigma_V) \;=\; V N(d_1) - D e^{-rT} N(d_2) - E
$$

$$
g_2(V, \sigma_V) \;=\; \frac{V}{E} N(d_1) \sigma_V - \sigma_E
$$

Use Newton-Raphson or Levenberg-Marquardt on $(g_1, g_2)$. Jacobian entries are analytic. This is faster but conflates the model's implied $\sigma_V$ with the iterated empirical one.

### Equity volatility input

$\sigma_E$ can be:

- **Realized:** $\sigma_E = \sqrt{252} \cdot \text{stdev}(\log P_t - \log P_{t-1})$ over a 252-day window. Standard for KMV.
- **Implied:** average at-the-money implied vol from listed options. More forward-looking but introduces options-market noise; only viable for liquid optionable names.

Use realized vol for the iterative KMV procedure (consistency); use implied vol for stressed-firm cross-checks.

### EDF mapping

The empirical DD-to-EDF mapping requires a default database. For an internal implementation without proprietary KMV data:

1. Use academic default panels (Moody's URD, Compustat default flags).
2. Bucket DD into 0.5-unit bins.
3. Compute the realized 1-year default frequency in each bucket.
4. Smooth via isotonic regression to enforce monotonicity.
5. Apply floors (EDF $\ge$ 2 bps) and ceilings (EDF $\le$ 35%).

### Drift estimation

$\mu$ is notoriously hard to estimate. Three options:

- **CAPM-implied:** $\mu = r + \beta_{\text{asset}} \cdot (\mathbb{E}[r_M] - r)$.
- **Trailing mean:** average log-return on $V$ over the calibration window (noisy in short windows).
- **Set $\mu = r$:** report only the risk-neutral DD. Common in practice because the absolute PD is less actionable than the relative DD ranking.

## Algorithm outline

```
Inputs:
    ticker            : equity ticker, e.g. "GS"
    horizon T         : 1.0 (years)
    history_days      : 252
    weight_LT_debt    : 0.5
    n_iter_max        : 50
    tol               : 1e-5

Procedure:

1. Pull market cap and equity volatility
   - info  = yf.Ticker(ticker).info
   - E_t   = info["marketCap"]    # in USD
   - hist  = yf.Ticker(ticker).history(period=f"{history_days+30}d")
   - log_r = np.log(hist["Close"]).diff().dropna()[-history_days:]
   - sigma_E = log_r.std() * sqrt(252)

2. Pull balance sheet
   - bs       = yf.Ticker(ticker).balance_sheet     # most recent quarter
   - ST_debt  = bs.loc["Current Debt"].iloc[0]
   - LT_debt  = bs.loc["Long Term Debt"].iloc[0]
   - D        = ST_debt + weight_LT_debt * LT_debt

3. Pull risk-free rate
   - r = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100
   - (use ^TNX for T = 1y if preferred)

4. Initialize
   - sigma_V_0 = sigma_E * E_t / (E_t + D)
   - sigma_V   = sigma_V_0

5. Iterative KMV loop
   for k = 0 .. n_iter_max:

       # Step a: invert Black-Scholes for V_t at each day in history
       V_series = []
       for each P_s in hist["Close"][-history_days:]:
           E_s = P_s * shares_out
           V_s = solve_for_V_given_E(E_s, D, r, T, sigma_V)
                 # Newton on f(V) = V*N(d1) - D*exp(-rT)*N(d2) - E_s
           V_series.append(V_s)

       # Step b: compute empirical sigma_V
       logV    = np.log(V_series)
       new_sig = np.std(np.diff(logV)) * sqrt(252)

       if abs(new_sig - sigma_V) < tol: break
       sigma_V = new_sig

6. Final V_t
   - V_t = V_series[-1]

7. Compute drift mu
   - Option A: mu = r  (risk-neutral DD)
   - Option B: mu = mean(diff(logV)) * 252
   - Recommended: report both

8. Distance-to-default
   - DD = ( log(V_t / D) + (mu - 0.5*sigma_V^2) * T ) / (sigma_V * sqrt(T))

9. Probability of default
   - PD_Merton = norm.cdf(-DD)
   - EDF       = empirical_DD_to_EDF_curve(DD)    # if calibrated; else PD_Merton

10. Theoretical credit spread (risk-neutral)
    - PD_Q = norm.cdf(-DD_Q)
    - LGD  = 0.6   # default assumption
    - spread = -log(1 - PD_Q * LGD) / T

11. Capital-structure arbitrage signal
    - cds_spread_market = (exogenous, e.g. from another data source)
    - if cds_spread_market - spread > threshold:  "CDS rich vs equity"
    - if cds_spread_market - spread < -threshold: "CDS cheap vs equity"

12. Persist (date, ticker, V_t, sigma_V, DD, PD, spread).
```

### Newton subroutine `solve_for_V_given_E`

```
def solve_for_V_given_E(E, D, r, T, sigma_V):
    V = E + D * exp(-r*T)          # initial guess: V ~ E + PV(D)
    for _ in range(100):
        d1 = (log(V/D) + (r + 0.5*sigma_V**2)*T) / (sigma_V*sqrt(T))
        d2 = d1 - sigma_V*sqrt(T)
        f  = V * Ncdf(d1) - D * exp(-r*T) * Ncdf(d2) - E
        fp = Ncdf(d1)                       # dE/dV = N(d1)
        V_new = V - f / fp
        if abs(V_new - V) < 1e-7 * V: break
        V = V_new
    return V
```

## yfinance data requirements

| Data | yfinance call | Field | Notes |
|------|---------------|-------|-------|
| Market cap | `yf.Ticker(t).info["marketCap"]` | scalar | Snapshot value; for historical $E_s$, multiply daily close by shares outstanding. |
| Shares outstanding | `yf.Ticker(t).info["sharesOutstanding"]` | scalar | Assumed constant over calibration window (small bias). |
| Daily prices | `yf.Ticker(t).history(period="1y")` | `Close` (adjusted) | Used for both $\sigma_E$ and for constructing historical $E_s$. |
| Balance sheet | `yf.Ticker(t).balance_sheet` | `Current Debt`, `Long Term Debt`, `Total Debt` | Quarterly; fall back to `quarterly_balance_sheet` if annual not populated. |
| Cash and equivalents | `yf.Ticker(t).balance_sheet.loc["Cash And Cash Equivalents"]` | scalar | For variants of KMV that net cash from debt. |
| Risk-free rate | `yf.Ticker("^IRX").history(...)` for 1-year T-bill | `Close` | Use `^TNX` for 10-year if a longer horizon is desired. |
| Sector / industry | `yf.Ticker(t).info["industry"]` | string | For sector-conditioned EDF mapping. |

**Limitations and workarounds:**

- yfinance balance sheet often has missing fields for non-US or small-cap names. Fall back to `quarterly_balance_sheet` or estimate $D$ from `info["totalDebt"]`.
- `info["marketCap"]` is point-in-time; for historical $E_s$, reconstruct as `Close[s] * sharesOutstanding`. Better, use `Close[s] * shares_history[s]` if a shares-history vendor is available — share-count changes (buybacks, issuance) bias $V_t$ otherwise.
- Adjusted close already incorporates splits and dividends, which is appropriate for vol estimation but not for level-based market-cap reconstruction. Use unadjusted close × current shares-outstanding, or adjusted close × inflation-deflated split factor.
- Balance sheet is reported at quarter-end with a 30–90 day reporting lag; treat $D$ as a step function with quarterly jumps, and use the most recent published value.
- For financials (banks, insurers), the Merton model is structurally inappropriate — leverage is qualitatively different. Either skip financials or use a sector-specific variant (e.g., Bank-Merton with deposit liabilities split out).

## Calibration / refit frequency

- **Equity volatility $\sigma_E$:** recompute daily on a 252-day rolling window.
- **Asset volatility $\sigma_V$:** recompute daily via the iterative KMV loop. The iteration is cheap (< 1 second per name).
- **Default point $D$:** update on each quarterly balance-sheet release (typically 4–6 times per year for US names with quarterly reports).
- **DD-to-EDF mapping:** annual refit, or after a regime break (e.g., update after the COVID-era default cluster).
- **Drift $\mu$:** monthly, from a 3-year trailing window.

Daily refit is appropriate because equity volatility and equity level change daily; quarterly $D$ updates are sufficient because debt structure is sticky.

## Validation and diagnostics

1. **Convergence of iterative loop.** Track number of iterations to convergence. Typical: 5–15. Names that fail to converge in 50 iterations are usually deep in distress (DD < −2) or have data errors — flag for manual review.
2. **In-sample fit.** Verify $g_1$ and $g_2$ residuals are near zero at convergence (< $10^{-4}$ relative).
3. **DD-EDF calibration check.** On the calibration sample, the EDF-bucketed mean DD should match the empirical 1-year default rate. Plot a reliability curve.
4. **Out-of-sample default prediction.** Hold out a forward year. Stocks with DD < 1 should default at much higher rates than stocks with DD > 4. Compute the Cumulative Accuracy Profile (CAP) and Accuracy Ratio (AR). Target AR > 0.6.
5. **Cross-check vs CDS.** For names with traded CDS, compare model spread to market spread. Persistent large divergence is either a model failure or an arbitrage opportunity — look for evidence in both directions.
6. **Parameter stability.** $\sigma_V$ should be more stable than $\sigma_E$ (asset returns are less volatile than equity returns). Sudden jumps in $\sigma_V$ (week-over-week) usually indicate a balance-sheet restatement or a data error.
7. **DD level reasonableness.** Investment-grade firms typically have DD between 3 and 6; high-yield between 1 and 3; distressed below 1. Names outside these bands warrant a manual look.

## Connections to other models

- **Model 4 (Factor Models / PCA):** $\sigma_E$ can be decomposed into systematic and idiosyncratic components using the factor model; the idiosyncratic vol is what KMV should arguably use (the firm's true asset risk is largely idiosyncratic). Advanced KMV variants substitute $\sigma_{E,\text{idio}}$ for $\sigma_E$.
- **Model 5 (Cost-of-Carry):** The risk-free rate $r$ is shared.
- **Layer 1 vol models (GARCH, realized-vol):** Provide higher-frequency $\sigma_E$ inputs than the simple realized 1-year std.
- **Layer 3 stat-arb:** A pair trade between a stock and its CDS uses the model spread vs market spread as a signal.
- **Layer 4 hedging:** The model's $V_t$ and $\sigma_V$ drive capital-structure hedge ratios (e.g., how much equity to short against a long bond position).

## Limitations and failure modes

1. **Single-bullet maturity.** The model assumes all debt matures at $T$. In reality firms have laddered maturities. Multi-bullet variants (Geske 1977) exist but are rarely used in practice; the KMV horizon $T = 1$ year and the $\text{ST debt} + 0.5 \cdot \text{LT debt}$ rule are a pragmatic compromise.
2. **Gaussian asset returns.** Real asset returns are fat-tailed. The empirical EDF mapping partly absorbs this, but the Gaussian PD $N(-\text{DD})$ understates tail risk.
3. **Continuous monitoring assumption.** Default can only occur at $T$, but real firms can default any time. First-passage variants (Black-Cox 1976) handle this; the BC model gives systematically higher PDs but requires solving a barrier option, not a vanilla call.
4. **Financial firms.** Banks have deposit liabilities, off-balance-sheet exposures, and regulatory capital constraints that break the simple equity-as-call analogy. Treat financials with a sector-specific variant or skip.
5. **Negative-book-value firms.** When liabilities exceed reported assets, the Newton inversion can fail or produce $V_t < D$. The model still produces a DD (negative), but cross-section comparability degrades.
6. **Asset substitution and dividends.** The model ignores the fact that equity holders can change the firm's asset risk or pay dividends out of assets. Both effects increase the option value to equity at debtholders' expense and are unmodeled.
7. **Sparse balance-sheet data.** Small-cap and OTC names often have stale or incomplete balance-sheet data on yfinance. Cross-check $D$ against SEC filings (10-K, 10-Q) for any name where the PD would drive a sizable position.
8. **Drift estimation noise.** $\mu$ enters the DD linearly; a 5% misestimate of $\mu$ shifts DD by $0.05/\sigma_V$, which on a low-vol name can be 0.3–0.5 units. Prefer reporting risk-neutral DD when $\mu$ is uncertain.

## References

- Merton, R. C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates." *Journal of Finance* 29(2), 449–470.
- Black, F. and Cox, J. C. (1976). "Valuing Corporate Securities: Some Effects of Bond Indenture Provisions." *Journal of Finance* 31(2), 351–367.
- Geske, R. (1977). "The Valuation of Corporate Liabilities as Compound Options." *Journal of Financial and Quantitative Analysis* 12, 541–552.
- Crosbie, P. and Bohn, J. (2003). "Modeling Default Risk." *Moody's KMV Technical Document.*
- Bharath, S. T. and Shumway, T. (2008). "Forecasting Default with the Merton Distance to Default Model." *Review of Financial Studies* 21(3), 1339–1369.
- Duffie, D. and Singleton, K. (2003). *Credit Risk: Pricing, Measurement, and Management.* Princeton.
- Vassalou, M. and Xing, Y. (2004). "Default Risk in Equity Returns." *Journal of Finance* 59(2), 831–868.
