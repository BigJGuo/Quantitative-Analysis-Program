# Engle-Granger and Kalman Cointegration

> Pair-trading mean reversion built on linear combinations of non-stationary price series that are themselves stationary. Layer 4 — Signals & ML. Primary use: pair-trading signal generation, dynamic hedge-ratio estimation, basket relative value.

## Mathematical formulation

### Cointegration definition

Let $P_t^A$ and $P_t^B$ be two price series, each integrated of order one ($I(1)$): their levels are non-stationary but their first differences $\Delta P_t^A = P_t^A - P_{t-1}^A$ are stationary. Series $P_t^A$ and $P_t^B$ are said to be **cointegrated** if there exists $\beta \in \mathbb{R} \setminus \{0\}$ such that the linear combination

$$Z_t = P_t^A - \beta P_t^B$$

is stationary ($I(0)$). The vector $(1, -\beta)$ is the **cointegrating vector**, and $Z_t$ is the **spread** or **residual**. Working with log prices $p_t = \log P_t$ is standard for equities because it makes $\beta$ scale-invariant to share-price level.

### Engle-Granger two-step procedure

**Step 1 — Estimate $\beta$ via OLS:**

$$P_t^A = \alpha + \beta P_t^B + Z_t, \qquad t = 1, \ldots, T$$

The OLS estimator is

$$\hat{\beta} = \frac{\sum_{t=1}^{T} (P_t^B - \bar{P}^B)(P_t^A - \bar{P}^A)}{\sum_{t=1}^{T} (P_t^B - \bar{P}^B)^2}$$

Under cointegration, $\hat{\beta}$ is **super-consistent**: it converges to the true cointegrating coefficient at rate $T$ rather than the usual $\sqrt{T}$. The OLS standard errors, however, are wrong because $Z_t$ is correlated with $P_t^B$ in finite samples.

**Step 2 — Test the residual $\hat{Z}_t = P_t^A - \hat{\alpha} - \hat{\beta} P_t^B$ for stationarity** using the augmented Dickey-Fuller (ADF) regression

$$\Delta \hat{Z}_t = \phi \hat{Z}_{t-1} + \sum_{j=1}^{p} \psi_j \Delta \hat{Z}_{t-j} + u_t$$

with the null $H_0: \phi = 0$ (unit root, no cointegration) versus $H_1: \phi < 0$ (stationary, cointegrated). Because $\hat{Z}_t$ is itself an estimated residual rather than an observed series, the Dickey-Fuller distribution does not apply directly; one uses Engle-Granger (or MacKinnon) critical values, which are more conservative than the standard ADF.

### Ornstein-Uhlenbeck representation and half-life

If $Z_t$ is stationary it can be approximated as a discrete OU (Ornstein-Uhlenbeck) process:

$$dZ_t = \kappa (\mu - Z_t) \, dt + \sigma \, dW_t$$

where $\kappa > 0$ is the mean-reversion rate, $\mu$ is the long-run mean, $\sigma$ is the diffusion coefficient and $W_t$ is a standard Brownian motion. The discrete-time analogue is the AR(1)

$$Z_t = c + \phi Z_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$

with $\phi = e^{-\kappa \Delta t}$, $c = \mu (1 - \phi)$ and $\sigma_\varepsilon^2 = \sigma^2 (1 - e^{-2\kappa \Delta t}) / (2\kappa)$. The **half-life of mean reversion** — the expected time for $Z_t - \mu$ to halve in magnitude — is

$$\tau_{1/2} = \frac{\ln 2}{\kappa} = -\frac{\ln 2}{\ln \phi}.$$

For daily data ($\Delta t = 1$ trading day), $\tau_{1/2}$ is expressed in trading days. Equilibrium variance:

$$\mathrm{Var}[Z_\infty] = \frac{\sigma^2}{2\kappa} = \frac{\sigma_\varepsilon^2}{1 - \phi^2}.$$

### Z-score trading signal

The standardized signal is

$$s_t = \frac{Z_t - \mu_Z}{\sigma_Z}$$

where $\mu_Z, \sigma_Z$ are either the OU long-run moments or rolling-window estimates. Symmetric entry at $|s_t| = s_{\text{in}}$ (commonly $1.5$ to $2.5$), exit at $|s_t| < s_{\text{out}}$ (e.g. $0.0$ to $0.5$), and a hard stop at $|s_t| > s_{\text{stop}}$ (e.g. $4$) constitute the canonical rule.

### Kalman filter dynamic hedge ratio

Static OLS $\beta$ fails across structural breaks (regime changes, dividend reinvestment policy changes, ETF rebalances). Allow $\beta_t$ to evolve as a random walk:

$$
\begin{aligned}
\beta_t &= \beta_{t-1} + \eta_t, & \eta_t &\sim \mathcal{N}(0, Q) \\
P_t^A &= \beta_t P_t^B + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, R).
\end{aligned}
$$

This is a linear Gaussian state-space model with state $\beta_t$, observation $P_t^A$, observation matrix $H_t = P_t^B$, state transition $F = 1$, process noise variance $Q$ and observation noise variance $R$. The Kalman recursions give:

**Predict step:**
$$\hat{\beta}_{t|t-1} = \hat{\beta}_{t-1|t-1}, \qquad P_{t|t-1} = P_{t-1|t-1} + Q$$

**Update step:**
$$
\begin{aligned}
y_t &= P_t^A - P_t^B \hat{\beta}_{t|t-1} & &\text{(innovation)} \\
S_t &= P_t^B P_{t|t-1} P_t^B + R & &\text{(innovation variance)} \\
K_t &= P_{t|t-1} P_t^B / S_t & &\text{(Kalman gain)} \\
\hat{\beta}_{t|t} &= \hat{\beta}_{t|t-1} + K_t y_t \\
P_{t|t} &= (1 - K_t P_t^B) P_{t|t-1}.
\end{aligned}
$$

The filtered innovation $y_t$ is the dynamic spread; its standardized form $y_t / \sqrt{S_t}$ replaces the static z-score.

### Johansen procedure (multi-asset extension)

For a vector $\mathbf{P}_t \in \mathbb{R}^N$ of $N$ price series, the vector error-correction model (VECM) is

$$\Delta \mathbf{P}_t = \Pi \mathbf{P}_{t-1} + \sum_{j=1}^{p-1} \Gamma_j \Delta \mathbf{P}_{t-j} + \boldsymbol{\varepsilon}_t.$$

The cointegration rank $r = \mathrm{rank}(\Pi)$ is tested via the **trace statistic**

$$\mathcal{T}_r = -T \sum_{i=r+1}^{N} \ln(1 - \hat{\lambda}_i)$$

where $\hat{\lambda}_i$ are the eigenvalues of a generalized eigenvalue problem on the residual moment matrices (see Johansen 1991). If $\mathrm{rank}(\Pi) = r > 0$, $\Pi = \alpha \beta^\top$ where $\beta \in \mathbb{R}^{N \times r}$ contains $r$ cointegrating vectors.

## Intuition and what this model addresses

Two stocks driven by the same fundamental factor — for example, Coca-Cola (KO) and PepsiCo (PEP), or two energy ETFs — share a common stochastic trend. Their individual price levels wander randomly, but a particular weighted difference should not. Cointegration formalizes this: it is the long-run economic equilibrium baked into a statistical test.

The trading thesis is purely **statistical mean reversion**, not directional: when the spread deviates from its long-run mean, short the over-priced leg, long the under-priced leg, and wait for convergence. Profit derives from $\sigma_Z$ — the volatility of the spread — and from the mean-reversion speed $\kappa$, which determines how quickly capital recycles.

The Kalman extension recognises that $\beta$ is rarely truly constant. Mergers, capital structure changes, and shifts in cross-correlations move the hedge ratio. A random-walk prior on $\beta_t$ provides graceful adaptation without imposing a fixed window.

## Calibration approach

1. **Universe selection.** Restrict to economically related pairs (same sector, same supply chain, same index). Without an economic prior, multiple-testing bias dominates and "cointegrated" pairs found by brute screening rarely survive out of sample.
2. **Lookback selection.** Use 1–2 years of daily data ($T \approx 250$–$500$) for initial cointegration testing. Too short: low test power. Too long: regime drift.
3. **Engle-Granger $\beta$.** Run OLS in both directions ($P^A$ on $P^B$ and $P^B$ on $P^A$); pair with stronger ADF rejection on the residual is the preferred dependent-variable orientation. Alternatives include total least squares (orthogonal regression) if neither series is naturally exogenous.
4. **ADF test on residuals.** Use Engle-Granger critical values from MacKinnon (1996). Reject null at the 5% level as the default threshold for trading; at 1% for a tighter universe.
5. **OU fit.** Estimate $\kappa, \mu, \sigma$ from AR(1) on $Z_t$. Compute $\tau_{1/2}$; reject pairs with $\tau_{1/2} > 30$ days (capital tie-up too long) or $\tau_{1/2} < 0.5$ days (microstructure noise, not a real signal).
6. **Kalman extension.** Initialize $\hat{\beta}_{0|0}$ at the static OLS estimate; initialize $P_{0|0}$ at the OLS variance. Tune the signal-to-noise ratio $Q/R$ via maximum likelihood or by treating it as a hyperparameter (typical $Q/R \in [10^{-5}, 10^{-3}]$).

## Algorithm outline

```
INPUT: tickers A, B; price history T_hist days; trading window T_trade days
OUTPUT: positions p_A(t), p_B(t) over T_trade

# --- Stage 1: Cointegration screening ---
1. Fetch daily log-close: p_A[1..T_hist], p_B[1..T_hist] via yfinance.
2. Run ADF on p_A and p_B individually; require both fail to reject I(1) at 5%.
3. OLS: regress p_A on (1, p_B) -> (alpha_hat, beta_hat).
4. Residuals: Z[t] = p_A[t] - alpha_hat - beta_hat * p_B[t].
5. ADF on Z with Engle-Granger critical values; require rejection at chosen level.
6. Fit AR(1): Z[t] = c + phi * Z[t-1] + eps[t]; estimate phi, sigma_eps.
7. kappa = -log(phi); mu = c / (1 - phi); tau_half = log(2) / kappa.
8. If tau_half not in [0.5, 30] days OR phi >= 1: REJECT pair.

# --- Stage 2a: Static-beta trading loop ---
9. For each trading day t in T_trade:
     a. Update Z[t] = p_A[t] - alpha_hat - beta_hat * p_B[t].
     b. Update rolling (mu_Z, sigma_Z) over last 60 days OR use OU equilibrium.
     c. s[t] = (Z[t] - mu_Z) / sigma_Z.
     d. Apply trading rules:
          if s[t] >  s_in   and no position: short 1 share A, long beta_hat shares B.
          if s[t] < -s_in   and no position: long  1 share A, short beta_hat shares B.
          if |s[t]| < s_out and  in position: close position.
          if |s[t]| > s_stop:                 close (stop loss), blacklist pair 30 days.

# --- Stage 2b: Kalman dynamic-beta variant ---
10. Initialize beta_hat[0] = OLS beta; P[0] = Var(OLS beta).
11. Choose Q (process noise), R (obs noise) by MLE or hyperparameter grid.
12. For each trading day t:
      a. beta_pred  = beta_hat[t-1]; P_pred = P[t-1] + Q.
      b. y_t        = p_A[t] - p_B[t] * beta_pred.
      c. S_t        = p_B[t]^2 * P_pred + R.
      d. K_t        = P_pred * p_B[t] / S_t.
      e. beta_hat[t] = beta_pred + K_t * y_t.
      f. P[t]       = (1 - K_t * p_B[t]) * P_pred.
      g. s[t]       = y_t / sqrt(S_t).
      h. Apply trading rules on s[t] as in Stage 2a.

# --- Stage 3: Risk overlays ---
13. Cap gross exposure per pair; cap aggregate exposure across pairs.
14. Re-run Stage 1 monthly; drop pairs that lose cointegration at 10% level.
```

## yfinance data requirements

Daily close-to-close log prices are the workhorse. yfinance daily data has effectively unlimited history (back to IPO) and is free.

```python
import yfinance as yf
import numpy as np

# Five years of KO and PEP daily bars
ko  = yf.Ticker("KO").history(period="5y",  interval="1d", auto_adjust=True)
pep = yf.Ticker("PEP").history(period="5y", interval="1d", auto_adjust=True)

# Align and take log close
df = ko[["Close"]].join(pep[["Close"]], lsuffix="_KO", rsuffix="_PEP").dropna()
p_A = np.log(df["Close_KO"].values)
p_B = np.log(df["Close_PEP"].values)
```

Required fields: `Close` (adjusted; `auto_adjust=True` folds in dividends and splits — essential for cointegration so that corporate actions do not create artificial breaks). For ETF pairs (e.g. XLF/XLB sector rotation), the same call structure applies.

**Intraday caveat:** yfinance restricts `interval="1m"` to 7 days and `interval="5m"` to 60 days of lookback. Cointegration of daily-close series is the standard production setup; intraday pair trading requires a separate higher-frequency data feed.

Use `statsmodels.tsa.stattools.coint` for the Engle-Granger residual ADF (it returns the proper EG critical values), `adfuller` for the unit-root check on individual legs, and `statsmodels.tsa.vector_ar.vecm` for Johansen on baskets.

## Calibration / refit frequency

- **Engle-Granger $\beta$:** refit weekly on a 1-year rolling window. Daily refits over-fit micro-noise; monthly refits miss regime drift.
- **OU parameters ($\kappa, \mu, \sigma$):** refit weekly on the same window used for $\beta$.
- **Rolling z-score moments:** updated daily on a 60-day trailing window.
- **Kalman variant:** $\beta_t$ updates continuously; $Q, R$ re-tuned quarterly via MLE on the most recent year.
- **Universe screening:** re-screen the candidate pool monthly; require continuous cointegration to remain in the active set.

## Validation and diagnostics

1. **Out-of-sample backtest.** Split history 70/30; calibrate on first 70%, trade on last 30%. Sharpe, max drawdown, win rate, average holding period (compare with $\tau_{1/2}$).
2. **Spread stationarity stability.** Re-run ADF on rolling windows; track $p$-value over time. Sudden loss of stationarity is the leading indicator of pair breakdown.
3. **Hedge-ratio drift.** Plot Kalman $\hat{\beta}_t$; sharp jumps indicate structural breaks (acquisitions, index reconstitutions).
4. **Innovation whiteness.** Ljung-Box test on Kalman innovations $y_t / \sqrt{S_t}$; rejection means the model is mis-specified (process noise wrong, observation noise wrong, or no genuine cointegration).
5. **Cross-pair correlation.** If running a portfolio of pairs, monitor spread-return correlations; high cross-correlation collapses diversification.
6. **Half-life realisation.** Empirical mean reversion time should bracket $\tau_{1/2}$ within a factor of two; large discrepancy invalidates the OU fit.

## Connections to other models

- **Avellaneda-Lee PCA-Residual Stat Arb (Model 7).** Generalises cointegration from pairs to the full equity panel: PCA replaces hand-picked pairs with statistically extracted common factors, and each stock's residual is treated as its own OU process.
- **OU/Vasicek mean-reversion models (Layer 2).** Same continuous-time machinery; cointegration provides the empirical justification for treating spreads as OU.
- **Kalman filter (Layer 3 state-space).** The dynamic hedge-ratio variant is a direct application; same filter is used in regime-switching and latent-factor models.
- **VECM / Johansen (Layer 4 multivariate).** Vector generalisation underlying basket trades and ETF-vs-constituents arbitrage.
- **Risk parity / portfolio construction (Layer 5).** Pair P\&L variances feed the portfolio-level covariance estimate.

## Limitations and failure modes

1. **Spurious cointegration.** Brute-force screening over thousands of pairs yields false positives at the test's nominal level. Bonferroni correction or strict economic priors are essential.
2. **Structural breaks.** Acquisitions, spin-offs, dividend policy changes destroy $\beta$. Kalman helps but cannot react instantaneously to discrete events; manual filters on corporate-action calendars are required.
3. **Asymmetric mean reversion.** Real spreads often mean-revert faster from one side than the other (e.g. one leg has higher borrow cost, asymmetric short squeeze risk). The symmetric OU is a simplification.
4. **Carry costs.** Borrow fees on the short leg, dividends paid on the short, financing on the long can erase the alpha for spreads with low volatility and long half-lives.
5. **Regime-dependence.** Cointegration weakens during stress (correlation $\to 1$ kills idiosyncratic dispersion) — exactly when leverage is most dangerous.
6. **Survivorship and look-ahead.** Historical screens that use only currently-listed tickers exclude delisted firms; one or both legs going to zero is the largest tail risk for pair trades.
7. **Multiple-testing on universe screens.** Across an $N$-stock universe there are $\binom{N}{2}$ candidate pairs; at the 5% level, $0.05 \binom{N}{2}$ false positives are expected by chance alone.

## References

- Engle, R. F. and Granger, C. W. J. (1987). *Co-integration and error correction: representation, estimation, and testing.* Econometrica 55(2): 251–276.
- Johansen, S. (1991). *Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models.* Econometrica 59(6): 1551–1580.
- MacKinnon, J. G. (1996). *Numerical distribution functions for unit root and cointegration tests.* Journal of Applied Econometrics 11(6): 601–618.
- Gatev, E., Goetzmann, W. N. and Rouwenhorst, K. G. (2006). *Pairs trading: performance of a relative-value arbitrage rule.* Review of Financial Studies 19(3): 797–827.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.* Wiley.
- Elliott, R. J., van der Hoek, J. and Malcolm, W. P. (2005). *Pairs trading.* Quantitative Finance 5(3): 271–276 (Kalman formulation).
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*, Chapter 4. Wiley.
