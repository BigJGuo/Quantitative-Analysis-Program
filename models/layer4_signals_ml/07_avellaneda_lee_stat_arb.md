# Avellaneda-Lee PCA-Residual Statistical Arbitrage

> Cross-sectional mean reversion of idiosyncratic residuals after projecting equity returns onto principal components or sector-ETF factors, each residual modelled as an Ornstein-Uhlenbeck process. Layer 4 — Signals & ML. Primary use: large-universe equity stat arb, ETF basket relative value, sector-neutral long/short signal generation.

## Mathematical formulation

### Panel factor model

Let $r_{i,t}$ be the log return of stock $i \in \{1, \ldots, N\}$ on day $t \in \{1, \ldots, T\}$:

$$r_{i,t} = \log P_{i,t} - \log P_{i,t-1}.$$

Decompose each return into a systematic component (driven by common factors) and an idiosyncratic residual:

$$r_{i,t} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} F_{k,t} + \tilde{r}_{i,t}$$

where $F_{k,t}$ is the realised return of factor $k$ at time $t$, $\beta_{i,k}$ is stock $i$'s loading on factor $k$, and $\tilde{r}_{i,t}$ is the idiosyncratic residual return. The factors $F_{k,t}$ can be:

- **PCA factors:** the top $K$ eigenportfolios of the empirical return covariance matrix.
- **Sector ETFs:** the SPDR sector ETFs (XLF, XLE, XLK, XLV, XLY, XLP, XLI, XLU, XLB, XLRE, XLC), optionally augmented with size/style factors.
- **Fama-French factors:** Mkt-RF, SMB, HML, RMW, CMA, plus momentum.

### PCA construction

Let $\mathbf{R} \in \mathbb{R}^{T \times N}$ be the matrix of standardized returns over a rolling window of length $T_w$ (typically 252 trading days):

$$\tilde{R}_{i,t} = \frac{r_{i,t} - \bar{r}_i}{\sigma_i}, \qquad \bar{r}_i = \frac{1}{T_w} \sum_{t} r_{i,t}, \quad \sigma_i^2 = \frac{1}{T_w - 1} \sum_{t} (r_{i,t} - \bar{r}_i)^2.$$

The empirical correlation matrix is $\boldsymbol{\Sigma} = \tilde{\mathbf{R}}^\top \tilde{\mathbf{R}} / (T_w - 1)$. Eigendecompose:

$$\boldsymbol{\Sigma} = \sum_{k=1}^{N} \lambda_k \mathbf{v}_k \mathbf{v}_k^\top, \qquad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N.$$

The $k$-th **eigenportfolio return** is the weighted return

$$F_{k,t} = \sum_{i=1}^{N} \frac{v_{k,i}}{\sigma_i} r_{i,t}.$$

Retain the top $K$ eigenportfolios. Avellaneda and Lee use $K \in [5, 15]$ for the S\&P 500 universe; the cutoff is often guided by the Marchenko-Pastur edge $\lambda_{\text{MP}}^+ = (1 + \sqrt{N/T_w})^2$, keeping eigenvalues above this threshold as economically meaningful.

### Residual extraction

For each stock $i$, regress its returns on the chosen factors over the same rolling window:

$$r_{i,t} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} F_{k,t} + \tilde{r}_{i,t}, \qquad t \in \text{window}.$$

OLS provides $\hat{\alpha}_i, \hat{\beta}_{i,k}$. The cumulative residual process is

$$X_i(t) = \sum_{s=1}^{t} \tilde{r}_{i,s}.$$

This integrated residual is the **idiosyncratic price process** — the part of the stock's path not explained by common factors.

### OU calibration on residuals

Model $X_i(t)$ as an Ornstein-Uhlenbeck process:

$$dX_i(t) = \kappa_i \bigl( m_i - X_i(t) \bigr) \, dt + \sigma_i \, dW_i(t)$$

with mean-reversion rate $\kappa_i > 0$, long-run mean $m_i$ and diffusion $\sigma_i$. The discrete-time analogue is AR(1):

$$X_i(t) = a_i + b_i X_i(t-1) + \zeta_{i,t}, \qquad \zeta_{i,t} \sim \mathcal{N}(0, \sigma_{\zeta,i}^2),$$

with the mapping

$$b_i = e^{-\kappa_i \Delta t}, \qquad a_i = m_i (1 - b_i), \qquad \sigma_{\zeta,i}^2 = \sigma_i^2 \, \frac{1 - e^{-2 \kappa_i \Delta t}}{2 \kappa_i}.$$

Solving for the OU parameters:

$$\hat{\kappa}_i = -\frac{\ln \hat{b}_i}{\Delta t}, \qquad \hat{m}_i = \frac{\hat{a}_i}{1 - \hat{b}_i}, \qquad \hat{\sigma}_i^2 = \hat{\sigma}_{\zeta,i}^2 \, \frac{2 \hat{\kappa}_i}{1 - \hat{b}_i^2}.$$

The **equilibrium standard deviation** — the long-run dispersion of $X_i$ around $m_i$ — is

$$\sigma_{\mathrm{eq},i} = \sigma_i \sqrt{\frac{1}{2 \kappa_i}}.$$

The **half-life** is

$$\tau_{1/2,i} = \frac{\ln 2}{\kappa_i}.$$

### S-score signal

The Avellaneda-Lee **s-score** standardizes the current residual against its equilibrium dispersion:

$$s_i(t) = \frac{X_i(t) - m_i}{\sigma_{\mathrm{eq},i}}.$$

Under the OU dynamics, $s_i \to \mathcal{N}(0, 1)$ in the stationary distribution. Trading rules are symmetric:

- **Open short** stock $i$ when $s_i(t) > s_{\text{short-open}}$ (typical: $+1.25$).
- **Open long** stock $i$ when $s_i(t) < -s_{\text{long-open}}$ (typical: $-1.25$).
- **Close short** when $s_i(t) < s_{\text{short-close}}$ (typical: $+0.50$ or $+0.75$).
- **Close long** when $s_i(t) > -s_{\text{long-close}}$ (typical: $-0.50$ or $-0.75$).

A position in stock $i$ is hedged by shorting (or longing) a basket replicating $\sum_k \beta_{i,k} F_k$, making the position factor-neutral.

### Drift correction

Stocks earn a small unconditional drift that contaminates the residual. Avellaneda-Lee subtract the modified s-score

$$s_i^{\mathrm{mod}}(t) = s_i(t) - \frac{\alpha_i}{\kappa_i \sigma_{\mathrm{eq},i}}$$

so that long-term alpha (or carry) is not mistaken for transient deviation.

## Intuition and what this model addresses

Engle-Granger pair trading scales poorly: with $N \approx 500$ stocks there are $\binom{500}{2} \approx 125{,}000$ candidate pairs, and multiple-testing destroys the signal. PCA replaces hand-picked pairs with statistically identified common factors. After projecting out market, sector, size, and style factors, the residual is — by construction — the part of the return that is **uncorrelated with systematic risk**. If that residual mean-reverts, it is pure idiosyncratic alpha.

The economic mechanism: institutional flow (index rebalances, mutual-fund redemptions, ETF creation/redemption) creates temporary price pressure on individual names that is gradually undone by liquidity provision. The OU process captures both the dislocation and the recovery in a single parametric form.

Avellaneda and Lee reported a Sharpe ratio of 1.44 over 1997–2007 on the US equity universe, with the strategy degrading post-2007 as HFT compressed the half-life and inventory costs rose.

## Calibration approach

1. **Universe.** Liquid US equities — typically the S\&P 500 constituents or Russell 1000 — filtered for minimum dollar volume (e.g. \$5M daily) and minimum price (e.g. \$5) to exclude microcaps and stale quotes.
2. **Lookback window $T_w$.** 252 trading days (one calendar year) is the canonical choice. Avellaneda-Lee also use 60-day windows for the OU fit even when the PCA uses a longer window; the rationale is that mean reversion is shorter-lived than factor structure.
3. **Factor selection.** PCA with $K$ chosen so that the top $K$ components explain ~50% of cross-sectional variance, typically $K \in [5, 15]$. Sector-ETF factors are an interpretable alternative that produces qualitatively similar residuals.
4. **OU fit window.** 60 trading days, rolled daily. Shorter windows are noisy; longer windows miss regime shifts.
5. **Half-life filter.** Reject stocks with $\tau_{1/2}$ outside $[5, 30]$ trading days. Below $5$ days: signal indistinguishable from microstructure noise; above $30$ days: capital tie-up too long, model assumptions unreliable.
6. **R-squared filter.** Require the factor regression to explain at least 20%–30% of return variance; below that, the "residual" is essentially the raw return and the OU assumption is questionable.
7. **Position sizing.** Equal-dollar, capital-weighted by inverse $\sigma_{\mathrm{eq}}$, or Kelly-fraction based on $\kappa_i$ and $\sigma_{\mathrm{eq},i}$.

## Algorithm outline

```
INPUT:  universe of N stocks, factor specification F (PCA or sector ETFs),
        PCA window T_w, OU window T_ou, signal thresholds
OUTPUT: daily positions q_i(t) for i = 1..N

# --- Daily refit ---
for each trading day t:

  # Step 1: Fetch and standardize the return panel.
  R[t-T_w+1 : t, 1..N] = log returns over rolling window.
  Drop stocks with insufficient history; forward-fill missing within tolerance.
  Standardize each column: R_tilde[s,i] = (R[s,i] - mean_i) / std_i.

  # Step 2: Build factors.
  if factor_mode == "PCA":
      Sigma = R_tilde.T @ R_tilde / (T_w - 1)
      eigvals, eigvecs = eig(Sigma)         # sorted descending
      sigma_i = std of returns over window for stock i
      for k = 1..K:
          F[s, k] = sum_i (eigvecs[i,k] / sigma_i) * R[s, i]   for s in window
  elif factor_mode == "ETF":
      F[s, k] = log return of sector ETF k at day s, k = 1..K.

  # Step 3: Regress each stock on factors.
  for i = 1..N:
      Run OLS  r_i = alpha_i + sum_k beta_ik * F_k + eps_i  over window.
      Compute residual returns eps_i[s] over the OU window (last T_ou days).
      X_i[s] = cumulative sum of eps_i over the OU window.

  # Step 4: Fit AR(1) to X_i over T_ou days.
  for i = 1..N:
      Run OLS  X_i[s] = a_i + b_i * X_i[s-1] + zeta_i[s].
      if b_i >= 1 or b_i <= 0:    flag as non-stationary; SKIP.
      kappa_i  = -log(b_i)
      m_i      = a_i / (1 - b_i)
      sigma_zeta_i = sd(zeta_i)
      sigma_eq_i   = sigma_zeta_i / sqrt(1 - b_i^2)
      tau_half_i   = log(2) / kappa_i
      if tau_half_i < 5 or tau_half_i > 30: SKIP.

  # Step 5: Compute s-score.
  for i = 1..N (surviving):
      s_i = (X_i[t] - m_i) / sigma_eq_i
      Optionally subtract drift: s_i -= alpha_i / (kappa_i * sigma_eq_i).

  # Step 6: Generate positions.
  for i = 1..N:
      prev_position = q_i(t-1)
      if   s_i >  +1.25 and prev_position == 0:   open SHORT in i
      elif s_i <  -1.25 and prev_position == 0:   open LONG  in i
      elif prev_position == SHORT and s_i < +0.50: close
      elif prev_position == LONG  and s_i > -0.50: close
      else: hold.

  # Step 7: Hedge factor exposure.
  Aggregate factor exposure: e_k = sum_i q_i(t) * beta_ik.
  Take offsetting position in factor portfolio k of size -e_k (eigenportfolio
  weights or sector-ETF shares) to enforce factor neutrality.

  # Step 8: Risk overlays.
  Cap |q_i| as fraction of ADV (e.g. 1% of 20-day avg dollar volume).
  Cap gross book and net book at portfolio level.
  Cap per-sector net exposure (even after factor hedge).
```

## yfinance data requirements

```python
import yfinance as yf
import pandas as pd

# Fetch S&P 500 constituents (universe loaded externally).
tickers = ["AAPL", "MSFT", "JPM", ...]   # ~500 names
data = yf.download(tickers,
                   period="2y",
                   interval="1d",
                   auto_adjust=True,
                   group_by="ticker",
                   threads=True)

# Build the return panel.
closes  = pd.DataFrame({t: data[t]["Close"] for t in tickers}).dropna(how="all")
returns = np.log(closes / closes.shift(1)).dropna(how="all")

# Sector ETF factors.
etfs = ["XLF","XLE","XLK","XLV","XLY","XLP","XLI","XLU","XLB","XLRE","XLC"]
etf_data    = yf.download(etfs, period="2y", interval="1d", auto_adjust=True)
etf_returns = np.log(etf_data["Close"] / etf_data["Close"].shift(1)).dropna()
```

Fields used: `Close` with `auto_adjust=True` so that dividends and splits are folded into the price; without this, residuals will contain ex-dividend jumps that swamp the OU signal. Volume (`Volume`) is used for the ADV-based position-sizing cap.

**Universe construction caveat:** yfinance does not provide point-in-time index constituents. Using today's S\&P 500 list to backtest 5 years ago introduces **survivorship bias** — bankrupt/delisted names are excluded, biasing returns upward. Production use requires a point-in-time membership history (e.g. CRSP, or a maintained constituent list with addition/deletion dates) and yfinance for the price data of those historical constituents.

**Intraday extension:** Avellaneda-Lee on intraday signals requires `interval="5m"` data (60-day cap on yfinance free tier). The cumulative-residual interpretation carries over but the OU half-life shrinks to hours, and microstructure effects dominate without proper exchange-grade data.

## Calibration / refit frequency

- **PCA / factor regression:** refit daily on a rolling 252-day window.
- **OU parameters:** refit daily on a rolling 60-day window of residual returns.
- **Threshold parameters $s_{\text{open}}, s_{\text{close}}$:** reviewed quarterly; sensitivity-tested annually.
- **Universe membership:** monthly; drop names that fail liquidity or stationarity filters.
- **$K$ (number of factors):** reviewed quarterly via Marchenko-Pastur cutoff or scree plot inspection.

## Validation and diagnostics

1. **Residual stationarity.** ADF test on the cumulative residual $X_i(t)$; require rejection at 10% for stock to be tradable.
2. **OU goodness of fit.** Residuals of the AR(1) fit should be approximately white and Gaussian; check via Ljung-Box (no autocorrelation) and Jarque-Bera (normality).
3. **Half-life distribution.** Plot the cross-sectional histogram of $\tau_{1/2}$; bimodality or fat right tail indicates universe contamination (microcaps, illiquid names sneaking through).
4. **Cross-sectional s-score distribution.** Should be approximately $\mathcal{N}(0,1)$ in aggregate; persistent skew or excess kurtosis flags model mis-specification (typically missing factor).
5. **Factor neutrality of P\&L.** Regress strategy daily returns on Fama-French + sector factors out of sample; loadings should be statistically zero. Non-zero loadings = leakage.
6. **Out-of-sample Sharpe degradation.** Compute in-sample (calibration window) vs out-of-sample (forward window) Sharpe; a degradation worse than 50% suggests over-fitting.
7. **Capacity analysis.** Re-run with halved position sizes and on a slippage-adjusted basis; the s-score signal is highly sensitive to transaction costs.

## Connections to other models

- **Engle-Granger cointegration (Model 6).** The two-asset analogue; PCA-residual is the panel generalisation.
- **Factor models / APT (Layer 2).** Provides the systematic-return decomposition that Avellaneda-Lee inverts to isolate residuals.
- **Ornstein-Uhlenbeck / Vasicek (Layer 2).** Same continuous-time process; here applied to residuals rather than rates or spreads.
- **Kalman filter (Layer 3).** Dynamic-factor and dynamic-loading variants of Avellaneda-Lee swap the rolling OLS for a Kalman filter on $\beta_{i,k,t}$, analogous to the dynamic hedge ratio in pair trading.
- **Black-Litterman / Bayesian portfolio (Layer 5).** S-scores feed as views into a portfolio construction step.
- **GARCH (Model 8).** $\sigma_i$ in the OU fit can be replaced by a GARCH-filtered conditional volatility for time-varying scaling.

## Limitations and failure modes

1. **Survivorship bias.** Backtests using current-membership universes overstate Sharpe by 0.3–0.7 depending on period; corrected histories are mandatory.
2. **Regime collapse.** During market stress, cross-sectional dispersion collapses ($\rho \to 1$) and residual variance falls; the model produces small s-scores and the strategy under-allocates exactly when liquidity-provision premia are largest, or it stays in losing positions through forced de-leveraging.
3. **Factor mis-specification.** Missing factors (e.g. crypto-exposure, ESG, dollar sensitivity) leak into the residual and produce spurious s-scores driven by systematic, not idiosyncratic, moves.
4. **Crowding.** PCA-residual is a well-known strategy; multiple participants targeting similar residuals causes co-movement of "idiosyncratic" returns and synchronized stop-outs.
5. **Transaction costs and borrow.** With $\tau_{1/2} \sim 10$ days and modest $\sigma_{\mathrm{eq}}$, round-trip transaction costs of $20$ bps can absorb the entire signal. Hard-to-borrow names require additional borrow-fee modeling.
6. **Microcap noise.** Even with filtering, illiquid names produce stale prices that look like mean-reverting residuals; require strict ADV thresholds.
7. **Non-stationary covariance.** PCA on a rolling window inherits the noise of the sample covariance; the top eigenvectors are stable, but lower eigenvectors are unstable. Truncating at $K$ where the eigenvalue gap is large mitigates this.
8. **Estimation noise in $\kappa_i$.** With 60 observations, $\hat{\kappa}_i$ has a standard error often comparable to its point estimate; the half-life and $\sigma_{\mathrm{eq}}$ inherit large uncertainty.

## References

- Avellaneda, M. and Lee, J.-H. (2010). *Statistical arbitrage in the U.S. equities market.* Quantitative Finance 10(7): 761–782.
- Marchenko, V. A. and Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1(4): 457–483.
- Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). *Noise dressing of financial correlation matrices.* Physical Review Letters 83: 1467.
- Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques.* Wiley.
- Khandani, A. and Lo, A. (2007). *What happened to the quants in August 2007?* Journal of Investment Management 5(4): 5–54.
- Connor, G. and Korajczyk, R. A. (1988). *Risk and return in an equilibrium APT: Application of a new test methodology.* Journal of Financial Economics 21(2): 255–289.
- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance 1(2): 223–236.
