# Barra-style Factor Models and PCA on Returns

> Linear factor decomposition that reduces high-dimensional asset-return covariance to a small set of systematic factors plus diagonal idiosyncratic risk; supports portfolio risk attribution, hedging, capital allocation, and the projection step in statistical arbitrage. Layer 2 — Structural / Cross-Sectional. Primary use: cross-sectional risk modeling for equity portfolios, signal residualization, and hedge construction.

## Mathematical formulation

### Linear factor model

Let $\mathbf{r}_t \in \mathbb{R}^N$ denote the vector of excess returns for $N$ assets at time $t$. The factor model expresses each asset's return as a linear combination of $K$ factor returns $\mathbf{f}_t \in \mathbb{R}^K$ plus an idiosyncratic component:

$$
\mathbf{r}_t \;=\; \mathbf{B} \mathbf{f}_t \;+\; \boldsymbol{\epsilon}_t
$$

where:

- $\mathbf{B} \in \mathbb{R}^{N \times K}$ — factor exposures (loadings, "betas").
- $\mathbf{f}_t \in \mathbb{R}^K$ — factor returns at time $t$.
- $\boldsymbol{\epsilon}_t \in \mathbb{R}^N$ — idiosyncratic returns, assumed mean-zero with $\mathbb{E}[\boldsymbol{\epsilon}_t \boldsymbol{\epsilon}_t^\top] = \mathbf{D}$ diagonal.

For asset $i$, the scalar form is:

$$
r_{i,t} \;=\; \sum_{k=1}^{K} \beta_{i,k} f_{k,t} \;+\; \epsilon_{i,t}
$$

### Covariance decomposition

If factors are uncorrelated with idiosyncratic residuals ($\mathbb{E}[\mathbf{f}_t \boldsymbol{\epsilon}_t^\top] = 0$), the asset-return covariance is:

$$
\boldsymbol{\Sigma} \;=\; \mathbf{B} \mathbf{F} \mathbf{B}^\top \;+\; \mathbf{D}
$$

where:

- $\mathbf{F} = \mathbb{E}[\mathbf{f}_t \mathbf{f}_t^\top] \in \mathbb{R}^{K \times K}$ — factor return covariance.
- $\mathbf{D} = \mathrm{diag}(\sigma_{\epsilon,1}^2, \ldots, \sigma_{\epsilon,N}^2)$ — diagonal specific-risk matrix.

This reduces the parameter count from $\tfrac{N(N+1)}{2}$ to $NK + \tfrac{K(K+1)}{2} + N$, which for $N = 1000, K = 10$ is a 20-fold reduction.

### Portfolio risk

For portfolio weights $\mathbf{w} \in \mathbb{R}^N$:

$$
\sigma_p^2 \;=\; \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} \;=\; \underbrace{\mathbf{w}^\top \mathbf{B} \mathbf{F} \mathbf{B}^\top \mathbf{w}}_{\text{systematic}} \;+\; \underbrace{\mathbf{w}^\top \mathbf{D} \mathbf{w}}_{\text{specific}}
$$

The portfolio's factor exposures are $\mathbf{x}_p = \mathbf{B}^\top \mathbf{w} \in \mathbb{R}^K$.

### PCA approach (statistical factors)

When factors are not specified ex ante, perform an eigendecomposition of $\boldsymbol{\Sigma}$:

$$
\boldsymbol{\Sigma} \;=\; \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top
$$

with eigenvalues $\lambda_1 \ge \lambda_2 \ge \ldots \ge \lambda_N \ge 0$ on the diagonal of $\boldsymbol{\Lambda}$ and eigenvectors as columns of $\mathbf{U}$. Truncate to the top $K$ components:

$$
\mathbf{B}_{\text{PCA}} \;=\; \mathbf{U}_{1:K} \boldsymbol{\Lambda}_{1:K}^{1/2}, \qquad \mathbf{f}_t^{\text{PCA}} \;=\; \boldsymbol{\Lambda}_{1:K}^{-1/2} \mathbf{U}_{1:K}^\top \mathbf{r}_t
$$

The PCA factors are by construction orthogonal: $\mathbf{F} = \mathbf{I}_K$.

For US Treasury yield changes, the first three principal components are interpreted as **level, slope, and curvature** and explain ~99% of the variance. For equity returns, the first PC is the **market**, the next 5–10 PCs are sector- or style-like, and 6–10 factors typically capture 60–80% of cross-sectional variance.

### Fama-French specification (fundamental factors)

The Fama-French 3-factor model specifies factors a priori:

$$
r_{i,t} - r_{f,t} \;=\; \alpha_i \;+\; \beta_i^{\text{MKT}} (r_{M,t} - r_{f,t}) \;+\; \beta_i^{\text{SMB}} \text{SMB}_t \;+\; \beta_i^{\text{HML}} \text{HML}_t \;+\; \epsilon_{i,t}
$$

with:

- $\text{SMB}_t$ — Small-Minus-Big: return on a portfolio long small-cap, short large-cap.
- $\text{HML}_t$ — High-Minus-Low: return on a portfolio long high book-to-market (value), short low book-to-market (growth).

Extensions (FF5) add profitability (RMW) and investment (CMA) factors. The Carhart 4-factor adds momentum (WML).

### Barra-style cross-sectional regression

Barra models specify exposures $\mathbf{B}$ from fundamentals (industry dummies, size, value, momentum, volatility, etc.) and estimate $\mathbf{f}_t$ by *cross-sectional regression* each period:

$$
\hat{\mathbf{f}}_t \;=\; (\mathbf{B}_t^\top \mathbf{W} \mathbf{B}_t)^{-1} \mathbf{B}_t^\top \mathbf{W} \mathbf{r}_t
$$

with $\mathbf{W}$ typically a diagonal weight matrix using $\sqrt{\text{market cap}}$. This contrasts with time-series regression (used for PCA and Fama-French) which fixes $\mathbf{f}_t$ and estimates $\beta_i$.

### Residualization (orthogonalization of signals)

For a raw signal $s_i$ on asset $i$, the factor-neutralized signal is:

$$
\tilde{s}_i \;=\; s_i \;-\; \mathbf{b}_i^\top \hat{\boldsymbol{\gamma}}, \qquad \hat{\boldsymbol{\gamma}} = (\mathbf{B}^\top \mathbf{B})^{-1} \mathbf{B}^\top \mathbf{s}
$$

This is the projection step used in pairs trading and statistical arbitrage (see Avellaneda-Lee).

## Intuition and what this model addresses

Equity returns are dominated by a handful of pervasive drivers: the overall market, sector rotation, size, value, momentum, and quality. Without recognizing this structure, a portfolio of 500 stocks looks like a 500-dimensional risk problem when in practice 6–10 dimensions explain most of the variance.

The model addresses several concrete failures:

1. **Sample-covariance ill-conditioning.** With $T$ daily returns on $N$ stocks, the sample covariance $\hat{\boldsymbol{\Sigma}}$ is rank-deficient whenever $T < N$, and is poorly conditioned when $T \sim N$. The factor model regularizes by forcing the off-block-diagonal structure $\mathbf{B} \mathbf{F} \mathbf{B}^\top$.
2. **Hedging discovery.** To hedge a long single-stock position, you need to know which factors it loads on. The factor decomposition gives the recipe directly: short $\beta_i$ units of each factor's mimicking portfolio.
3. **Risk attribution.** Portfolio managers need to know where their P&L came from. Decomposing realized P&L into factor contributions (sector, size, etc.) and specific contributions answers "skill vs luck."
4. **Signal residualization.** A raw value or momentum signal is contaminated by market beta. Residualizing through the factor model isolates the orthogonal alpha component.
5. **Cross-sectional ranking.** Statistical arbitrage runs on residuals, not raw returns. The factor model defines what "residual" means.

PCA versus fundamental factors is a tradeoff. PCA is fully data-driven and adapts to the current market regime, but the factors lack interpretation and are not stable (signs and rotations change between calibrations). Fundamental factors (Fama-French, Barra) are interpretable and stable but may miss factors the market is actively pricing (e.g., a new ESG factor emerging during 2018–2020). Production systems run both and use PCA as a residual check on the fundamental specification.

## Calibration approach

### Statistical (PCA)

1. Build a returns panel $\mathbf{R} \in \mathbb{R}^{T \times N}$ on a rolling window (typically 252 days).
2. Center: subtract column means.
3. Optionally standardize (divide by column std) to balance high- and low-vol stocks.
4. Compute the sample covariance $\hat{\boldsymbol{\Sigma}} = \frac{1}{T-1} \mathbf{R}_c^\top \mathbf{R}_c$.
5. Eigendecompose: $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{U}} \hat{\boldsymbol{\Lambda}} \hat{\mathbf{U}}^\top$.
6. Choose $K$ by either (a) cumulative variance threshold (e.g., $K$ such that $\sum_{k=1}^K \lambda_k / \sum_{k=1}^N \lambda_k \ge 0.7$), (b) Marchenko-Pastur upper edge $\lambda^* = \sigma^2 (1 + \sqrt{N/T})^2$ — keep factors above this random-matrix-theory threshold, or (c) parallel analysis (Horn's method).
7. Form $\mathbf{B} = \hat{\mathbf{U}}_{1:K} \hat{\boldsymbol{\Lambda}}_{1:K}^{1/2}$ and $\mathbf{D} = \mathrm{diag}(\hat{\boldsymbol{\Sigma}} - \mathbf{B} \mathbf{B}^\top)$.
8. Floor $\mathbf{D}$ at a minimum specific-vol to avoid singular weights downstream.

### Fundamental (Fama-French)

1. Construct factor return series exogenously (download from Ken French's data library or build them: SMB from a size-sorted long-short portfolio, HML from a B/M-sorted long-short portfolio).
2. For each asset $i$, time-series-regress $r_i$ on the factor returns over a 252-day window. The OLS estimator is $\hat{\boldsymbol{\beta}}_i = (\mathbf{F}^\top \mathbf{F})^{-1} \mathbf{F}^\top \mathbf{r}_i$.
3. Compute residual variance $\sigma_{\epsilon,i}^2 = \mathrm{Var}(r_i - \mathbf{f}^\top \hat{\boldsymbol{\beta}}_i)$.
4. The factor covariance $\mathbf{F}$ is the sample covariance of the factor returns.

### Barra-style

1. Construct exposures $\mathbf{B}_t$ from fundamentals each day:
   - **Size:** $\log(\text{market cap})$, cross-sectionally standardized to mean 0, std 1.
   - **Value:** book-to-market $B/P$, standardized.
   - **Momentum:** trailing 12-month return excluding the most recent month, standardized.
   - **Volatility:** trailing 60-day realized vol, standardized.
   - **Industry dummies:** GICS sector or industry (one-hot encoded).
2. Each day, cross-sectional regression $\mathbf{r}_t = \mathbf{B}_t \mathbf{f}_t + \boldsymbol{\epsilon}_t$ weighted by $\sqrt{\text{market cap}}$.
3. Build a time series of $\hat{\mathbf{f}}_t$.
4. Estimate $\mathbf{F}$ as the sample covariance of $\hat{\mathbf{f}}_t$ over a 252-day window.

### Regularization

Shrinkage (Ledoit-Wolf) toward the constant-correlation target:

$$
\hat{\boldsymbol{\Sigma}}_{\text{shrink}} \;=\; (1 - \alpha) \hat{\boldsymbol{\Sigma}} \;+\; \alpha \boldsymbol{\Sigma}_{\text{target}}
$$

with $\alpha \in [0, 1]$ chosen by Ledoit-Wolf's analytic formula minimizing Frobenius distance to the unknown true covariance.

### Loss function

For factor exposure estimation in time-series regression, OLS (equivalent to minimizing squared residuals):

$$
\mathcal{L}(\boldsymbol{\beta}_i) \;=\; \sum_t (r_{i,t} - \mathbf{f}_t^\top \boldsymbol{\beta}_i)^2
$$

For robustness against outlier days, use Huber loss:

$$
\mathcal{L}_H(\boldsymbol{\beta}_i) \;=\; \sum_t \rho_\delta(r_{i,t} - \mathbf{f}_t^\top \boldsymbol{\beta}_i), \qquad \rho_\delta(x) = \begin{cases} \tfrac{1}{2} x^2 & |x| \le \delta \\ \delta(|x| - \tfrac{1}{2}\delta) & |x| > \delta \end{cases}
$$

## Algorithm outline

```
Inputs:
    universe        : list of tickers, length N
    history_days T  : 252
    K               : 5 (or chosen by RMT)
    method          : "PCA" | "FamaFrench" | "Barra"

Procedure:

1. Build returns panel
   - For ticker in universe:
        bars = yf.Ticker(ticker).history(period=f"{T+30}d", auto_adjust=True)
        log_r = np.log(bars["Close"]).diff().dropna()
   - Align all series on common dates: R is (T_eff x N) DataFrame.
   - Drop columns with > 5% missing; forward-fill the rest.
   - Optionally winsorize each column at the 1%/99% level.

2. If method == "PCA":
   a. R_c = R - R.mean(axis=0)
   b. Sigma_hat = (R_c.T @ R_c) / (T_eff - 1)
   c. eigvals, eigvecs = np.linalg.eigh(Sigma_hat)
   d. Sort in descending order.
   e. Apply Marchenko-Pastur cutoff: keep factors with lambda > sigma^2 * (1 + sqrt(N/T))^2.
      (Estimate sigma^2 from the median eigenvalue.)
   f. B = eigvecs[:, :K] * sqrt(eigvals[:K])    # (N x K)
   g. F = I_K (PCA factors are orthonormal by construction)
   h. fitted = R_c @ eigvecs[:, :K] @ eigvecs[:, :K].T   # (T_eff x N)
   i. resid  = R_c - fitted
   j. D = np.diag(resid.var(axis=0))                     # (N x N)
   k. Floor D entries at 0.0001^2 (annualized 1% specific vol).

3. If method == "FamaFrench":
   a. Load (or construct) Mkt-Rf, SMB, HML series; align with R.
   b. F_panel = (T_eff x 3) factor return DataFrame.
   c. For each ticker i:
         beta_i = OLS(R[:,i] - Rf, F_panel).coef
         resid_i = R[:,i] - Rf - F_panel @ beta_i
         sigma_eps_i = std(resid_i)
   d. B = stack(beta_i for i)
      D = diag(sigma_eps_i^2 for i)
      F = cov(F_panel)

4. If method == "Barra":
   a. Construct exposures (loop over dates):
        For each date t in window:
            size_i      = log(market_cap_i(t))
            value_i     = book_to_market_i(t)
            mom_i       = sum(log_returns_i over [t-252, t-21])
            vol_i       = std(log_returns_i over [t-60, t])
            industry_i  = one-hot of GICS sector
        Standardize size, value, mom, vol cross-sectionally each day.
        B_t = column-stack(intercept, size, value, mom, vol, industry dummies)
   b. For each t:
        W_t = diag(sqrt(market_cap_i(t)))
        f_hat_t = solve(B_t.T @ W_t @ B_t, B_t.T @ W_t @ r_t)
        resid_t = r_t - B_t @ f_hat_t
   c. F = cov(stack(f_hat_t for t))
      D = diag(var(resid for each i across t))

5. Compute Sigma_full = B @ F @ B.T + D

6. Portfolio applications:
   a. For weights w (long-short, dollar-neutral):
        sigma_port_sys = sqrt(w.T @ B @ F @ B.T @ w)
        sigma_port_spec = sqrt(w.T @ D @ w)
        sigma_port = sqrt(sigma_port_sys^2 + sigma_port_spec^2)
        factor_exposures x = B.T @ w

7. Signal residualization (for any cross-sectional signal s):
   a. gamma = solve(B.T @ B, B.T @ s)
   b. s_neutral = s - B @ gamma

8. Hedge construction:
   a. For a long position vector w_long, find short weights w_short such that:
        B.T @ (w_long + w_short) = 0 (factor neutral)
   b. Often parametrized via a hedge basket (e.g., SPY, sector ETFs).

9. Diagnostics:
   - Variance explained: cumsum(eigvals) / sum(eigvals)
   - Factor stability: correlation of B_t and B_{t-21} columns
   - Residual autocorrelation: should be near zero
```

## yfinance data requirements

| Data | yfinance call | Notes |
|------|---------------|-------|
| Adjusted close | `yf.Ticker(t).history(period="2y", auto_adjust=True)["Close"]` | Daily; backbone of returns panel. |
| Market cap (current) | `yf.Ticker(t).info["marketCap"]` | Snapshot; for historical, use `Close * shares_outstanding`. |
| Book value | `yf.Ticker(t).balance_sheet.loc["Common Stock Equity"]` | For book-to-market in Barra. |
| Sector / industry | `yf.Ticker(t).info["sector"]`, `info["industry"]` | For industry dummies. |
| Risk-free rate | `yf.Ticker("^IRX").history(period="2y")["Close"] / 100 / 252` | Daily risk-free. |
| Market return | `yf.Ticker("SPY").history(period="2y")["Close"]` (returns) or `^GSPC` | For market factor. |
| Universe membership | Maintain externally; e.g., S&P 500 constituents | yfinance does not provide index membership history. |

**Limitations and workarounds:**

- yfinance returns sometimes have NaN gaps for thinly traded names or around corporate actions. Forward-fill carefully, but drop names with > 5% missing to avoid contaminating the covariance.
- Survivorship bias: yfinance only returns currently-listed tickers. Backtests on a current S&P 500 list overstate returns by 1–3% annually. Maintain a point-in-time universe externally.
- For SMB and HML factor construction, building these from yfinance requires the full cross-section of US stocks, which yfinance is not optimized for. Easier to download Kenneth French's published factor returns and align dates.
- `info["marketCap"]` is a single snapshot; reconstructing historical market caps from `Close * sharesOutstanding` ignores share-count changes from buybacks and issuance. Use `Ticker.get_shares_full(...)` if precision matters.
- `auto_adjust=True` adjusts for splits and dividends — appropriate for returns but distorts level-based signals (e.g., reconstructed market cap). Pull both adjusted and unadjusted if needed.

## Calibration / refit frequency

- **PCA covariance:** refit weekly on a 252-day rolling window. Daily refit is computationally cheap but adds noise; weekly balances stability against regime adaptation.
- **Fama-French betas:** monthly refit. Betas drift slowly and monthly is the standard convention.
- **Barra exposures:** daily refit (exposures are themselves time-varying through fundamentals).
- **Barra factor returns $\hat{\mathbf{f}}_t$:** daily, by cross-sectional regression.
- **Specific-risk $\mathbf{D}$:** weekly, from realized residuals.
- **Shrinkage parameter $\alpha$:** monthly.

PCA is sometimes refit daily during stress periods when factor structure is changing rapidly; this is acceptable as long as the daily PCA is anchored (sign and ordering aligned) against the prior week's solution to avoid spurious flips.

## Validation and diagnostics

1. **Variance explained.** $\sum_{k=1}^K \lambda_k / \sum_{k=1}^N \lambda_k$. For US equities, expect 30–45% for the first PC (market), 50–65% cumulative through K=5, 65–80% through K=10.
2. **Eigenvalue stability.** Compare top-K eigenvalues to Marchenko-Pastur upper edge $(1 + \sqrt{N/T})^2 \sigma^2$. Eigenvalues above this edge are "real" factors; those below are statistical noise.
3. **Factor loadings stability.** Compute correlation of $\mathbf{b}_k$ between adjacent calibrations. Stable factors (market, size) should have correlation > 0.95 week-to-week. PCs 4+ are typically less stable.
4. **Residual autocorrelation.** Idiosyncratic residuals should be near white noise. Ljung-Box test on residuals should not reject at lag 5.
5. **Cross-sectional residual orthogonality.** Pairwise correlations of residuals across assets should be small — large pair correlations indicate a missing factor.
6. **Out-of-sample portfolio vol forecast.** Forecast next-month portfolio vol from $\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$ vs realized. Forecast bias should be < 5% on average.
7. **Factor mimicking portfolio R²:** Each factor should be reproducible by a long-short portfolio of stocks with $R^2 > 0.9$.

## Connections to other models

- **Model 2 (Futures-Implied NAV):** When the hedge set is large, PCA-reduce hedges before regression to stabilize $\boldsymbol{\beta}_{\text{basket}}$.
- **Model 3 (Merton-KMV):** Use the idiosyncratic component of equity vol $\sigma_{E,\text{idio}}$ (from this model's $\mathbf{D}$) as the input to KMV instead of total equity vol. Reduces noise from market-wide vol moves.
- **Layer 3 stat-arb (Avellaneda-Lee):** This model defines the residualization step. The Ornstein-Uhlenbeck mean reversion is fit on the residuals $\boldsymbol{\epsilon}_t$.
- **Layer 4 portfolio construction:** Mean-variance optimization uses $\boldsymbol{\Sigma}$ directly. Risk parity uses factor exposures $\mathbf{x}_p = \mathbf{B}^\top \mathbf{w}$.
- **Layer 4 execution:** Sector ETFs (XLK, XLF, etc.) are interpreted as mimicking portfolios for industry factors; sized using $\mathbf{B}$ and $\mathbf{F}$.

## Limitations and failure modes

1. **Regime changes.** Factor loadings shift during crises (e.g., 2008 the financials sector loading flipped sign relative to non-stressed periods). Rolling estimation lags the regime by half the window length.
2. **Factor zoo and overfitting.** With enough fundamental factors, in-sample $R^2$ approaches 1 but out-of-sample performance degrades. Strict CV and parsimony are required.
3. **PCA sign and rotation indeterminacy.** Adjacent recalibrations can flip the sign of any eigenvector. Always anchor against the prior solution.
4. **Missing factors and pricing errors.** If the true model has $K$ factors and the model uses $K-1$, the omitted factor variance loads into the residuals and biases all downstream applications. Use diagnostic tests for residual cross-correlation.
5. **Linearity assumption.** True exposures may be nonlinear in fundamentals (e.g., size effect is concave). Use piecewise-linear basis functions or quantile-bucketed exposures for non-linear factors.
6. **Specific-risk diagonal assumption.** In practice, residuals show clustering (e.g., a sector-specific shock that the sector dummies didn't fully capture). Block-diagonal $\mathbf{D}$ variants exist.
7. **Survivorship and look-ahead bias.** Calibrating on a current universe introduces both biases; both inflate apparent factor premia and reduce apparent specific risk.
8. **Crowding.** When many funds use similar factor exposures, factor returns become correlated with funding flows rather than fundamental risk — leading to factor crashes (e.g., August 2007 quant crisis).

## References

- Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium." *Journal of Finance.*
- Ross, S. A. (1976). "The Arbitrage Theory of Capital Asset Pricing." *Journal of Economic Theory* 13, 341–360.
- Fama, E. and French, K. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics* 33, 3–56.
- Fama, E. and French, K. (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics* 116, 1–22.
- Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance." *Journal of Finance* 52, 57–82.
- Connor, G. and Korajczyk, R. (1986). "Performance Measurement with the Arbitrage Pricing Theory." *Journal of Financial Economics* 15, 373–394.
- Ledoit, O. and Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis* 88, 365–411.
- Avellaneda, M. and Lee, J. (2010). "Statistical Arbitrage in the US Equities Market." *Quantitative Finance* 10, 761–782.
- MSCI Barra (2011). *USE4 Model Empirical Notes.*
- Marchenko, V. A. and Pastur, L. A. (1967). "Distribution of Eigenvalues for Some Sets of Random Matrices." *Mathematics of the USSR-Sbornik* 1, 457–483.
