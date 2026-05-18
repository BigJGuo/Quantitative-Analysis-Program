# Bayesian Hierarchical and Kalman Dynamic Factor Models

> Partial-pooling Bayesian regression and linear-Gaussian state-space estimation. Layer 4 — Signals & ML. Primary use: cross-sectional alpha with short per-asset histories, time-varying factor loadings (rolling beta), and dynamic-factor models for production risk and signal aggregation.

## Mathematical formulation

### Kalman filter (linear-Gaussian state space)

Latent state $\mathbf{x}_t \in \mathbb{R}^n$, observation $\mathbf{y}_t \in \mathbb{R}^m$:

$$\mathbf{x}_{t+1} = \mathbf{F}_t \mathbf{x}_t + \mathbf{c}_t + \mathbf{w}_t, \qquad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t),$$

$$\mathbf{y}_t = \mathbf{H}_t \mathbf{x}_t + \mathbf{d}_t + \mathbf{v}_t, \qquad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t).$$

Assume Gaussian initial state $\mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_{0|0}, \mathbf{P}_{0|0})$, independence of $\{\mathbf{w}_t\}, \{\mathbf{v}_t\}$, and cross-independence.

**Predict step.**

$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}_{t-1} \hat{\mathbf{x}}_{t-1|t-1} + \mathbf{c}_{t-1},$$

$$\mathbf{P}_{t|t-1} = \mathbf{F}_{t-1} \mathbf{P}_{t-1|t-1} \mathbf{F}_{t-1}^\top + \mathbf{Q}_{t-1}.$$

**Update step.** Innovation $\boldsymbol{\nu}_t = \mathbf{y}_t - \mathbf{H}_t \hat{\mathbf{x}}_{t|t-1} - \mathbf{d}_t$, innovation covariance $\mathbf{S}_t = \mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^\top + \mathbf{R}_t$, Kalman gain

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^\top \mathbf{S}_t^{-1},$$

$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \boldsymbol{\nu}_t, \qquad \mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1}.$$

Use the Joseph form $\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)\mathbf{P}_{t|t-1}(\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)^\top + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^\top$ for numerical stability under finite precision.

**RTS (Rauch-Tung-Striebel) smoother** (backward pass, for offline estimation):

$$\mathbf{C}_t = \mathbf{P}_{t|t} \mathbf{F}_t^\top \mathbf{P}_{t+1|t}^{-1},$$

$$\hat{\mathbf{x}}_{t|T} = \hat{\mathbf{x}}_{t|t} + \mathbf{C}_t (\hat{\mathbf{x}}_{t+1|T} - \hat{\mathbf{x}}_{t+1|t}),$$

$$\mathbf{P}_{t|T} = \mathbf{P}_{t|t} + \mathbf{C}_t (\mathbf{P}_{t+1|T} - \mathbf{P}_{t+1|t}) \mathbf{C}_t^\top.$$

**Log-likelihood** for parameter learning:

$$\log p(\mathbf{y}_{1:T} \mid \boldsymbol{\theta}) = -\tfrac{1}{2} \sum_{t=1}^T \left[ m \log 2\pi + \log |\mathbf{S}_t| + \boldsymbol{\nu}_t^\top \mathbf{S}_t^{-1} \boldsymbol{\nu}_t \right].$$

Maximize over hyperparameters $\boldsymbol{\theta} = (\mathbf{F}, \mathbf{H}, \mathbf{Q}, \mathbf{R}, \mathbf{x}_0, \mathbf{P}_0)$ by L-BFGS or EM.

### Dynamic factor model (Stock–Watson)

For $N$ assets and $r$ factors,

$$\mathbf{y}_t = \boldsymbol{\Lambda} \mathbf{f}_t + \boldsymbol{\epsilon}_t, \qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi}),$$

$$\mathbf{f}_t = \boldsymbol{\Phi} \mathbf{f}_{t-1} + \boldsymbol{\eta}_t, \qquad \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}).$$

Cast as a state-space ($\mathbf{x}_t = \mathbf{f}_t$, $\mathbf{H} = \boldsymbol{\Lambda}$, $\mathbf{R} = \boldsymbol{\Psi}$, $\mathbf{F} = \boldsymbol{\Phi}$) and run the Kalman filter/smoother. Identify with $\mathbf{Q} = \mathbf{I}$ and $\boldsymbol{\Lambda}^\top \boldsymbol{\Lambda}$ diagonal, or PC-initialize.

**Time-varying beta (special case).** For asset $i$ vs market,

$$r_{i,t} = \alpha_t + \beta_t r_{m,t} + \epsilon_t,$$

with state $\mathbf{x}_t = (\alpha_t, \beta_t)^\top$, $\mathbf{F} = \mathbf{I}$ (random walk) or diagonal AR(1) $\text{diag}(\rho_\alpha, \rho_\beta)$, $\mathbf{H}_t = (1, r_{m,t})$, $\mathbf{Q} = \text{diag}(\sigma_\alpha^2, \sigma_\beta^2)$, $R = \sigma_\epsilon^2$.

### Bayesian hierarchical regression (partial pooling)

For asset $i \in 1..N$ over times $t \in 1..T_i$:

$$y_{i,t} \mid \boldsymbol{\theta}_i, \sigma_i \sim \mathcal{N}(\mathbf{X}_{i,t}^\top \boldsymbol{\theta}_i, \sigma_i^2),$$

$$\boldsymbol{\theta}_i \mid \boldsymbol{\mu}, \boldsymbol{\Sigma} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),$$

$$\boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}_0, \tau_0^2 \mathbf{I}), \quad \boldsymbol{\Sigma} \sim \text{LKJ-corr} \otimes \text{Half-Cauchy(scale)}, \quad \sigma_i \sim \text{Half-Normal}(s_0).$$

The posterior $p(\boldsymbol{\theta}_i \mid \mathbf{y})$ is shrunk toward $\boldsymbol{\mu}$ — strongly for short-history $i$, weakly for long-history $i$. For Gaussian-Gaussian conjugacy and known $\boldsymbol{\Sigma}, \sigma_i^2$, the posterior is

$$\boldsymbol{\theta}_i \mid \mathbf{y}_i \sim \mathcal{N}(\mathbf{m}_i, \mathbf{V}_i),$$

$$\mathbf{V}_i^{-1} = \boldsymbol{\Sigma}^{-1} + \sigma_i^{-2} \mathbf{X}_i^\top \mathbf{X}_i, \quad \mathbf{m}_i = \mathbf{V}_i (\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} + \sigma_i^{-2} \mathbf{X}_i^\top \mathbf{y}_i).$$

The shrinkage factor depends on data information $\sigma_i^{-2} \mathbf{X}_i^\top \mathbf{X}_i$ relative to prior precision $\boldsymbol{\Sigma}^{-1}$. This is **James–Stein-style** shrinkage extended to regression.

## Intuition and what this model addresses

**Why partial pooling.** No-pool (OLS per asset) blows up on short histories or sector ETFs with 6 months of data. Complete-pool (one $\boldsymbol{\theta}$ across all assets) ignores cross-section heterogeneity. Partial pooling lets each asset estimate its own $\boldsymbol{\theta}_i$ while borrowing strength from the population. The Bayesian posterior automatically balances the two depending on data quality.

**Why Kalman.** Factor loadings and betas are not constant — sector rotations, leverage changes, macro regime shifts all move $\beta_t$. A rolling-window OLS is a crude truncation: it gives equal weight inside the window and zero outside. The Kalman filter is the *optimal* (MSE-minimizing under Gaussian assumptions) recursive estimator of a slowly drifting parameter, with the smoothness controlled by the ratio $\sigma_w^2 / \sigma_v^2$ (signal-to-noise of state innovations vs. observation noise).

**Production usage.** Stock–Watson dynamic factor models sit under most large risk systems: the latent factors are then conditioned on by Barra-style style/industry factors to produce a full covariance forecast. Hierarchical Bayesian models give well-calibrated alpha and factor-loading point estimates with uncertainty — important inputs for portfolio optimization that respects parameter risk.

## Architecture / model design

### Time-varying beta (Kalman)

- State: $\mathbf{x}_t = (\alpha_t, \beta_t)$, $n=2$.
- Transition: random walk, $\mathbf{F} = \mathbf{I}_2$, $\mathbf{Q} = \text{diag}(q_\alpha, q_\beta)$.
- Observation: $r_{i,t} = (1, r_{m,t}) \mathbf{x}_t + v_t$, $R = \sigma_\epsilon^2$.
- Hyperparameters $(q_\alpha, q_\beta, \sigma_\epsilon^2)$ chosen by max-likelihood or fixed via signal-to-noise heuristic ($q_\beta \approx (0.05/252)^2$ for slow drift).

### Dynamic factor model

- Run PCA on standardized returns to choose $r$ (eigenvalue scree).
- Initial $\boldsymbol{\Lambda}^{(0)}$ = PCA loadings; $\boldsymbol{\Phi}^{(0)}$ = AR(1) on PC scores; $\boldsymbol{\Psi}^{(0)}$ = diag of residual variances.
- EM iterations: E-step = Kalman smoother on current $\boldsymbol{\theta}$; M-step = MLE updates of $(\boldsymbol{\Lambda}, \boldsymbol{\Psi}, \boldsymbol{\Phi}, \mathbf{Q})$.

### Hierarchical Bayesian regression (PyMC)

```
priors:
  mu       ~ Normal(0, 1)            (vector of p factor coefs)
  sd_eta   ~ HalfCauchy(1)           (between-asset SD)
  L_corr   ~ LKJCholeskyCov(eta=2)   (between-asset correlation)
  Sigma    = sd_eta * L_corr * L_corr.T * sd_eta
  theta[i] ~ MvNormal(mu, Sigma)     for i in 1..N
  sigma[i] ~ HalfNormal(0.05)
likelihood:
  y[i,t]   ~ Normal(X[i,t] @ theta[i], sigma[i])
```

Non-centered parameterization (Matt Trick): $\boldsymbol{\theta}_i = \boldsymbol{\mu} + \mathbf{L} \mathbf{z}_i$, $\mathbf{z}_i \sim \mathcal{N}(0, \mathbf{I})$. Avoids funnel pathologies in NUTS.

## Calibration / training approach

### Kalman

- **MLE.** Optimize $\boldsymbol{\theta} = (q_\alpha, q_\beta, \sigma_\epsilon^2)$ by L-BFGS on the marginal log-likelihood from the filter (formula above). Reparameterize variances as $q = \exp(2 \ell)$ to keep positivity.
- **EM.** E-step: smoother gives $\hat{\mathbf{x}}_{t|T}, \mathbf{P}_{t|T}, \mathbf{P}_{t,t-1|T}$. M-step: closed-form updates for $\mathbf{F}, \mathbf{H}, \mathbf{Q}, \mathbf{R}$ via lag-1 covariance.
- **Bayesian.** Wishart priors on $\mathbf{Q}, \mathbf{R}^{-1}$; Gibbs sample states via FFBS (forward-filter, backward-sample).

### Hierarchical Bayes

- **Sampler.** NUTS (No-U-Turn HMC) via PyMC or NumPyro.
- **Chains.** 4 chains, 2000 warmup + 2000 sampling iterations.
- **Diagnostics.** $\hat{R} < 1.01$, ESS > 400, no divergences. If divergences, increase `target_accept` to 0.99 and re-parameterize.
- **Hand-rolled Gibbs (alternative).** With Gaussian likelihood and conjugate Normal-Inverse-Wishart priors: $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ updates closed form; $\boldsymbol{\theta}_i$ updates closed form; $\sigma_i^2$ updates via Inverse-Gamma.

## Algorithm outline

### Kalman: rolling beta of AAPL vs SPY

```
INPUT: r_AAPL[1..T], r_SPY[1..T], hyperparams (q_a, q_b, R)
INIT:  x_hat = (0, 1)^T; P = diag(0.01, 0.1)
For t = 1..T:
  # Predict
  x_pred = F @ x_hat            # F = I, so x_pred = x_hat
  P_pred = F @ P @ F.T + Q      # Q = diag(q_a, q_b)
  # Observe
  H_t   = (1, r_SPY[t])
  nu    = r_AAPL[t] - H_t @ x_pred
  S     = H_t @ P_pred @ H_t.T + R
  K     = P_pred @ H_t.T / S
  # Update
  x_hat = x_pred + K * nu
  P     = (I - K @ H_t) @ P_pred  (Joseph form for stability)
  Save beta_t = x_hat[1]
RETURN beta_t time series
```

### Hierarchical Bayes (Gibbs sketch)

```
INPUT: data {X_i, y_i}, prior hyperparams
INIT:  mu = 0, Sigma = I, sigma_i^2 = 1 for all i, theta_i = OLS(X_i, y_i)
For iter in 1..N_iter:
  # 1. Sample theta_i | rest:
  For each asset i:
     V_i = (Sigma^-1 + X_i^T X_i / sigma_i^2)^-1
     m_i = V_i (Sigma^-1 mu + X_i^T y_i / sigma_i^2)
     theta_i ~ N(m_i, V_i)
  # 2. Sample mu | rest:
  V_mu = (tau_0^-2 I + N Sigma^-1)^-1
  m_mu = V_mu (Sigma^-1 sum_i theta_i)
  mu ~ N(m_mu, V_mu)
  # 3. Sample Sigma | rest:  Inverse-Wishart conjugate
  S_sum = sum_i (theta_i - mu)(theta_i - mu)^T
  Sigma ~ IW(nu_0 + N, S_0 + S_sum)
  # 4. Sample sigma_i^2 | rest:
  For each i:
     a = a_0 + T_i / 2
     b = b_0 + 0.5 * ||y_i - X_i theta_i||^2
     sigma_i^2 ~ InvGamma(a, b)
  Store draws after burn-in.
RETURN posterior samples
```

## yfinance data requirements and feature engineering

**Universe.** Stocks of interest, plus market proxy (SPY), sector ETFs (XLK, XLF, XLE, ..., 11 SPDR sectors), risk-free proxy (^IRX or BIL).

**Data calls.**

```python
import yfinance as yf
panel = yf.download(['AAPL','SPY','XLK', ...], start='2015-01-01',
                    auto_adjust=True, interval='1d')['Close']
returns = panel.pct_change().dropna()
```

**Features.**

1. **Daily log returns.** $r_{i,t} = \log(P_{i,t}/P_{i,t-1})$.
2. **Excess returns.** $r_{i,t} - r_{f,t}$ for asset and market.
3. **Factor proxies.** Fama-French style: market = SPY, size = IWM − SPY, value = IWD − IWF, momentum = MTUM − SPY, low-vol = USMV − SPY.
4. **Sector loadings.** Project each stock onto its sector ETF.
5. **Macro state proxies.** VIX (^VIX), 10Y yield (^TNX), USD index (DX-Y.NYB) — used as exogenous inputs $\mathbf{u}_t$ in extended state-space.
6. **Fundamentals (slow).** From `yf.Ticker(t).info`: marketCap, beta, P/E, P/B — used as features for asset-level prior $\boldsymbol{\mu}_i = h(\text{fundamentals}_i)$ in a meta-regression layer.

**Use cases.**

- **Rolling beta of AAPL to SPY.** Kalman with state $(\alpha, \beta)$. Compare to 60-day OLS rolling window.
- **Time-varying multi-factor model.** State $(\alpha, \beta_\text{mkt}, \beta_\text{size}, \beta_\text{val}, \beta_\text{mom})$; $n=5$, $\mathbf{H}_t$ row of factor returns.
- **Short-history shrinkage.** A newly-IPOed stock with 6 months of data: Bayesian hierarchical pools its $\boldsymbol{\theta}$ toward the sector mean.
- **Latent-factor extraction.** Run dynamic factor model on a panel of 500 returns; extract 5–10 factors that explain most cross-section variance; use factor returns as features downstream.

## Refit frequency

- **Kalman filter.** Recursive — no refit needed; only the hyperparameters $(\mathbf{Q}, \mathbf{R})$ are refit. Re-tune $\mathbf{Q}, \mathbf{R}$ monthly by MLE on the past 2 years.
- **Hierarchical Bayes.** Full posterior re-sample weekly or monthly. Incremental updates are possible via sequential Monte Carlo / particle filters, but weekly batch is operationally simpler.
- **Dynamic factor model.** EM re-fit monthly. The filter is run daily with parameters held fixed between refits.

## Validation and diagnostics

**Kalman diagnostics.**
- **Innovation whiteness.** Standardized innovations $\boldsymbol{\nu}_t / \sqrt{\mathbf{S}_t}$ should be iid $\mathcal{N}(0,1)$. Run Ljung-Box on the series.
- **Normality.** QQ plot of standardized innovations.
- **Log-likelihood gain.** Compare against a static-OLS baseline; require positive in-sample and out-of-sample gain.
- **Parameter recovery on simulation.** Generate synthetic data with known $(\mathbf{F}, \mathbf{H}, \mathbf{Q}, \mathbf{R})$; check that MLE recovers them.

**Bayesian diagnostics.**
- **$\hat{R}$ (Gelman–Rubin).** Should be < 1.01 for all parameters.
- **ESS.** Effective sample size > 400 for primary parameters.
- **Posterior predictive checks.** Simulate $\tilde{\mathbf{y}}$ from the posterior; compare distribution of summary statistics (mean, var, autocorr) against observed.
- **LOO / WAIC.** Pareto-smoothed importance sampling LOO for model comparison.
- **Funnel diagnostics.** Plot $\log \sigma_i$ vs $\theta_i$ — divergences cluster in funnel necks.

**Trading diagnostics.**
- **Out-of-sample $R^2$.** On predicted vs realized factor returns.
- **Shrinkage benefit.** Compare per-asset MSE for hierarchical vs OLS vs full-pool; expect monotone improvement for short histories.

## Connections to other models

- **GBDT / Transformer (Model 11).** Predict residuals from the hierarchical-Bayes mean ($y - \mathbf{X}^\top \mathbf{m}_i$) with GBDT — captures non-linearity on top of well-shrunk linear baseline.
- **SAE + MLP (Model 10).** The bottleneck $\mathbf{h}$ becomes the latent factor in a dynamic factor model with non-linear loadings.
- **Online learning (Model 13).** Kalman filter *is* an online Bayesian update; FTRL on a regression can be derived from a Kalman filter on a slowly-drifting parameter.
- **GARCH (Layer 3).** Kalman beta + GARCH idiosyncratic vol → conditional CAPM.
- **Risk model (Layer 5).** Dynamic factor model produces the factor covariance used in optimization.

## Limitations and failure modes

1. **Linear-Gaussian assumption.** Real returns are heavy-tailed and skewed. Innovations show fat tails; switch to Student-t observation noise (use scaled-mixture trick to retain conjugacy).
2. **Q/R identifiability.** Only the *ratio* is identified from data on short series; pin one variance by domain knowledge.
3. **Filter divergence.** Numerical loss of positive-definiteness of $\mathbf{P}$ in single precision. Use Joseph form or square-root filter.
4. **Bayesian sampler pathologies.** Funnel in the hierarchical scale parameter — fix with non-centered parameterization.
5. **Prior misspecification.** A too-tight $\boldsymbol{\Sigma}$ over-shrinks; too-loose acts like no pooling. Tune prior via cross-validated LOO score.
6. **Stationarity violation.** Random-walk state model is non-stationary; AR(1) on state with $|\rho| < 1$ is safer and often empirically better.
7. **yfinance survivorship bias.** Hierarchical mean is biased upward if the universe excludes delisted names — use a point-in-time index membership feed.
8. **Computation.** Hierarchical Bayes on 1000+ assets with $T = 2500$ takes minutes per NUTS sample; use variational inference (ADVI) for first-pass exploration and reserve NUTS for production.

## References

1. Kalman, R. E. (1960). *A new approach to linear filtering and prediction problems.* J. Basic Engineering.
2. Rauch, H., Tung, F., & Striebel, C. (1965). *Maximum likelihood estimates of linear dynamic systems.* AIAA Journal.
3. Stock, J. H., & Watson, M. W. (2002). *Macroeconomic forecasting using diffusion indexes.* JBES.
4. Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods.* OUP.
5. Gelman, A. et al. (2013). *Bayesian Data Analysis,* 3rd ed., chapters on hierarchical models.
6. Betancourt, M., & Girolami, M. (2015). *Hamiltonian Monte Carlo for hierarchical models.* arXiv:1312.0906.
7. Carvalho, C., Polson, N., & Scott, J. (2010). *The horseshoe estimator for sparse signals.* Biometrika.
8. Doan, T., Litterman, R., & Sims, C. (1984). *Forecasting and conditional projection using realistic prior distributions.* (BVAR foundations).
