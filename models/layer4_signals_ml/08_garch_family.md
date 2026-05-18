# GARCH Family — GARCH, GJR-GARCH, EGARCH

> Autoregressive conditional heteroskedasticity models for one-step-ahead daily volatility forecasting, with extensions capturing leverage and asymmetric volatility response. Layer 4 — Signals & ML. Primary use: pre-trade volatility scaling, option-quote conditioning, position sizing under risk constraints, VaR/ES estimation.

## Mathematical formulation

### Common framework

Let $r_t$ be the daily log return $r_t = \log(P_t / P_{t-1})$. Decompose into conditional mean and innovation:

$$r_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \overset{\mathrm{iid}}{\sim} \mathcal{D}(0, 1)$$

where $\mu_t = \mathbb{E}[r_t \mid \mathcal{F}_{t-1}]$ is the conditional mean (often a constant or a low-order AR), $\sigma_t^2 = \mathrm{Var}[r_t \mid \mathcal{F}_{t-1}]$ is the conditional variance, $\mathcal{F}_{t-1}$ is the information set through $t-1$, and $z_t$ is a standardized innovation with distribution $\mathcal{D}$ — Gaussian or Student-$t$ being the standard choices.

The GARCH family specifies the dynamics of $\sigma_t^2$.

### GARCH(p, q) — Bollerslev (1986)

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2.$$

The benchmark **GARCH(1,1)**:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2.$$

Parameter constraints for positivity and stationarity:

$$\omega > 0, \quad \alpha \geq 0, \quad \beta \geq 0, \quad \alpha + \beta < 1.$$

Under stationarity, the **unconditional variance** is

$$\bar{\sigma}^2 = \mathbb{E}[\sigma_t^2] = \frac{\omega}{1 - \alpha - \beta}.$$

The persistence $\alpha + \beta$ governs how slowly shocks decay; for daily equity returns, typical estimates lie in $[0.95, 0.99]$.

**Multi-step variance forecast.** Iterating the recursion forward, the $h$-step-ahead variance forecast is

$$\hat{\sigma}_{t+h \mid t}^2 = \bar{\sigma}^2 + (\alpha + \beta)^{h-1} (\sigma_{t+1 \mid t}^2 - \bar{\sigma}^2).$$

This converges geometrically to $\bar{\sigma}^2$ at rate $(\alpha + \beta)$ — the **mean reversion of volatility**.

### GJR-GARCH — Glosten, Jagannathan and Runkle (1993)

Adds a leverage term so that negative shocks contribute more to future variance than positive shocks of equal magnitude:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \gamma I(\varepsilon_{t-1} < 0) \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where $I(\cdot)$ is the indicator function. The effective ARCH coefficient is $\alpha$ on positive return days and $\alpha + \gamma$ on negative days. For equity indices, $\gamma > 0$ is essentially universal: the **leverage effect**.

Stationarity condition (under symmetric $z_t$ with $\mathbb{P}(z_t < 0) = 1/2$):

$$\alpha + \tfrac{1}{2} \gamma + \beta < 1.$$

Unconditional variance:

$$\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \tfrac{1}{2} \gamma - \beta}.$$

### EGARCH — Nelson (1991)

Models log-variance so positivity is automatic; captures asymmetry through a linear function of standardized innovations:

$$\log \sigma_t^2 = \omega + \alpha \bigl( |z_{t-1}| - \mathbb{E}|z_{t-1}| \bigr) + \gamma z_{t-1} + \beta \log \sigma_{t-1}^2$$

with $z_{t-1} = \varepsilon_{t-1} / \sigma_{t-1}$. Under Gaussian $z$, $\mathbb{E}|z| = \sqrt{2/\pi}$; under Student-$t_\nu$,

$$\mathbb{E}|z| = \frac{2 \sqrt{\nu - 2} \, \Gamma\bigl(\tfrac{\nu+1}{2}\bigr)}{\sqrt{\pi} \, (\nu - 1) \, \Gamma\bigl(\tfrac{\nu}{2}\bigr)}.$$

The term $\gamma z_{t-1}$ generates asymmetry: $\gamma < 0$ implies negative shocks raise future log-variance more than positive shocks of equal size. EGARCH is stationary whenever $|\beta| < 1$, with no positivity constraints on $\omega, \alpha, \gamma$ — a major practical advantage.

### Innovation distributions

Equity returns exhibit fat tails that Gaussian innovations cannot match. The standard alternative is the **standardized Student-$t$** with $\nu$ degrees of freedom:

$$f(z; \nu) = \frac{\Gamma\bigl(\tfrac{\nu + 1}{2}\bigr)}{\sqrt{\pi (\nu - 2)} \, \Gamma\bigl(\tfrac{\nu}{2}\bigr)} \, \bigl(1 + \tfrac{z^2}{\nu - 2}\bigr)^{-(\nu + 1)/2}, \qquad \nu > 2.$$

Typical MLE estimates on US equity indices place $\hat{\nu} \in [5, 10]$. The **Hansen skewed-$t$** further captures conditional skewness; the **Generalized Error Distribution (GED)** is a Gaussian alternative parameterized by tail-shape $\nu_g$.

### Likelihood

Conditional on parameters $\boldsymbol{\theta} = (\omega, \alpha, \gamma, \beta, \nu)$, the log-likelihood for $T$ observations is

$$\ell(\boldsymbol{\theta}) = \sum_{t=1}^{T} \log f\bigl(\varepsilon_t / \sigma_t; \nu\bigr) - \log \sigma_t.$$

Under Gaussian innovations:

$$\ell(\boldsymbol{\theta}) = -\tfrac{1}{2} \sum_{t=1}^{T} \Bigl[ \log(2\pi) + \log \sigma_t^2 + \frac{\varepsilon_t^2}{\sigma_t^2} \Bigr].$$

Maximize numerically (BFGS, with positivity constraints handled via reparameterization, or interior-point).

## Intuition and what this model addresses

Daily equity returns are nearly uncorrelated but their **squared returns are strongly autocorrelated** — volatility clusters. Calm periods follow calm periods; turbulent periods follow turbulent periods. ARCH (Engle 1982) captured this with a finite memory of past squared returns. GARCH added an autoregressive term on past variance, which provides effectively infinite, geometrically decaying memory with only two parameters.

The conditional-variance forecast $\sigma_t^2$ is the primary deliverable. It feeds:

- **Position sizing** under a target-volatility constraint: scale exposure inversely with $\sigma_t$.
- **VaR / ES** at horizon $h$: combine the GARCH forecast with the innovation distribution to give quantile estimates that respond to current conditions, not the unconditional sample.
- **Option pricing** under a stochastic-volatility wrapper: GARCH option-pricing models (Duan 1995) embed the discrete dynamics in a no-arbitrage framework.
- **Strategy gating**: turn off mean-reversion strategies when conditional vol crosses thresholds — empirical mean reversion collapses in high-vol regimes.

GJR and EGARCH extensions encode the **leverage effect**: empirically, negative returns of size $|r|$ produce ~2× the next-day variance impact of positive returns of the same size. Ignoring asymmetry leaves systematic bias in down-market vol forecasts, exactly where forecasts matter most.

## Calibration approach

1. **Mean model.** Use a constant or a small AR(1) on returns. Higher-order ARMA on daily equity returns is usually rejected; the marginal benefit to vol estimation is small.
2. **Sample window.** Minimum 4–5 years of daily returns (~1000 observations) for stable MLE; 10 years is preferable. Too short: parameter estimates are noisy; too long: structural breaks contaminate.
3. **Initial variance.** Set $\sigma_1^2 = $ sample variance of the first 30 returns, or back-cast using the long-run unconditional variance.
4. **Specification choice.**
   - GARCH(1,1) for plain vanilla applications, simplest benchmark.
   - GJR-GARCH(1,1) when leverage matters and is statistically significant (likelihood ratio test against GARCH).
   - EGARCH(1,1) when positivity constraints in GARCH bind, or when the parameter sign of $\gamma$ is of interest.
5. **Innovation distribution.** Default to standardized Student-$t$ with $\nu$ estimated by MLE. Use Gaussian only for testing or where speed matters.
6. **Optimization.** BFGS with analytic or finite-difference gradients; reparameterize $\omega = \exp(\tilde{\omega})$, $\alpha = \mathrm{logit}^{-1}(\tilde{\alpha}) \cdot (1 - \beta)$ to enforce positivity and stationarity. Standard package: `arch` (Kevin Sheppard).
7. **Robust standard errors.** Use the Bollerslev-Wooldridge (1992) sandwich estimator if Gaussian innovations are used as quasi-MLE.

## Algorithm outline

```
INPUT:  daily log-return series r[1..T], specification in {GARCH, GJR, EGARCH},
        innovation distribution in {Gaussian, Student-t, GED}
OUTPUT: parameter estimates, in-sample sigma^2 path, h-step-ahead forecasts

# --- Step 1: Pre-processing ---
1. Demean: r_centered = r - mean(r)   (or fit AR(1) mean and use residuals).
2. Initialize sigma_1^2 = var(r_centered[1:30]).

# --- Step 2: Likelihood function ---
function negloglik(theta):
   parse theta -> (omega, alpha, gamma, beta, nu)   # gamma=0 if pure GARCH
   for t = 2..T:
     if spec == "GARCH":
        sigma2[t] = omega + alpha*eps[t-1]^2 + beta*sigma2[t-1]
     elif spec == "GJR":
        I_neg = 1 if eps[t-1] < 0 else 0
        sigma2[t] = omega + alpha*eps[t-1]^2 + gamma*I_neg*eps[t-1]^2
                          + beta*sigma2[t-1]
     elif spec == "EGARCH":
        z = eps[t-1] / sqrt(sigma2[t-1])
        log_sigma2[t] = omega + alpha*(|z| - E|z|) + gamma*z
                              + beta*log_sigma2[t-1]
        sigma2[t]     = exp(log_sigma2[t])
     eps[t] = r_centered[t]      # innovation feed for next step
   ll = sum of log f(eps[t]/sigma[t]; nu) - log sigma[t]
   return -ll

# --- Step 3: Optimize ---
3. Initial guess: omega = 0.01*var(r), alpha = 0.05, gamma = 0.05, beta = 0.9,
   nu = 8.
4. Reparameterize to enforce constraints; minimize negloglik via BFGS.
5. Inverse-transform to recover parameter estimates.
6. Compute standard errors from numerical Hessian (or Bollerslev-Wooldridge
   sandwich).

# --- Step 4: Diagnostics ---
7. Compute standardized residuals z_t = eps[t] / sigma[t].
8. Ljung-Box on z_t (test of mean specification) and z_t^2 (test of variance
   specification): no autocorrelation -> well-specified.
9. ARCH-LM test on z_t: should fail to reject no remaining ARCH effect.
10. QQ plot of z_t vs assumed distribution.

# --- Step 5: Forecast ---
11. One-step: sigma^2_{T+1|T} via the recursion using current eps_T, sigma_T.
12. h-step: iterate recursion; for GARCH(1,1) closed form is
       sigma^2_{T+h|T} = sigma_bar^2 + (alpha+beta)^{h-1}*(sigma^2_{T+1|T}-sigma_bar^2).
13. Annualize: sigma_ann = sigma_daily * sqrt(252).
```

## yfinance data requirements

```python
import yfinance as yf
import numpy as np
from arch import arch_model

# 10 years of SPY daily bars
data = yf.Ticker("SPY").history(period="10y",
                                interval="1d",
                                auto_adjust=True)
returns = 100 * np.log(data["Close"]).diff().dropna()  # in percent

# GARCH(1,1) with Student-t innovations
model_garch = arch_model(returns, mean="Constant",
                         vol="GARCH",  p=1, q=1, dist="t")
res_garch = model_garch.fit(disp="off")
print(res_garch.summary())

# GJR-GARCH(1,1)
model_gjr = arch_model(returns, mean="Constant",
                       vol="GARCH",  p=1, o=1, q=1, dist="t")

# EGARCH(1,1)
model_egarch = arch_model(returns, mean="Constant",
                          vol="EGARCH", p=1, o=1, q=1, dist="t")

# 10-day variance forecast
fc = res_garch.forecast(horizon=10, reindex=False)
sigma_fc = np.sqrt(fc.variance.values[-1])
```

Required fields: `Close` with `auto_adjust=True`. Returns expressed in percent (`100 * log diff`) is the convention in the `arch` package; this rescales the likelihood for numerical stability — coefficients reported by the package then refer to percent-return units.

**Why daily, not intraday:** GARCH is a daily-vol model. Intraday equivalents (HEAVY, multiplicative-component GARCH) exist but use realized-vol inputs and are better captured by the HAR-RV framework (Model 9). For intraday vol with GARCH, scale via $\sigma_{\text{intraday}} \approx \sigma_{\text{daily}} \sqrt{\Delta t / 1 \text{ day}}$ only as a rough approximation; intraday seasonality (U-shape) breaks this badly.

**Data quality:** Use a single-source price series; mixing data providers introduces inconsistent dividend treatments and spurious volatility jumps that contaminate GARCH parameters. yfinance's `auto_adjust=True` provides a consistent total-return series.

## Calibration / refit frequency

- **Parameter re-estimation:** monthly is standard; weekly is acceptable for fast-moving markets; daily refits over-fit and produce unstable parameter time series.
- **Rolling window length:** 5–10 years of daily returns. Anchored windows (expanding) are also used and have lower variance but adapt more slowly to regime change.
- **Specification choice:** review annually. Run likelihood-ratio tests of nested models (GARCH vs GJR), and information criteria (AIC, BIC) for non-nested comparisons (GJR vs EGARCH).
- **Distribution parameter $\nu$:** can be re-estimated within the same MLE step at each refit; in stress periods $\hat{\nu}$ drops (tails fatten).
- **Forecast horizon:** typically 1, 5, 10, 22 days (one trading day, week, fortnight, month). For longer horizons, the mean-reversion to $\bar{\sigma}^2$ dominates and the model collapses to its unconditional variance.

## Validation and diagnostics

1. **Standardized residual whiteness.** Ljung-Box($Q$) on $\hat{z}_t = \varepsilon_t / \hat{\sigma}_t$ at lags 10, 20; rejection indicates a misspecified mean.
2. **Squared standardized residual whiteness.** Ljung-Box on $\hat{z}_t^2$; rejection indicates residual ARCH effects — the variance model is too parsimonious.
3. **ARCH-LM test.** Engle's LM test on $\hat{z}_t^2$; should fail to reject (no remaining ARCH).
4. **Sign-bias test (Engle-Ng).** Regress $\hat{z}_t^2$ on $I(\varepsilon_{t-1} < 0)$, $I(\varepsilon_{t-1} < 0) \varepsilon_{t-1}$, $I(\varepsilon_{t-1} \geq 0) \varepsilon_{t-1}$. Significant slopes flag missing asymmetry — switch GARCH to GJR or EGARCH.
5. **Mincer-Zarnowitz regression.** Realized variance proxy $\mathrm{RV}_t$ regressed on $\hat{\sigma}_t^2$; intercept zero, slope one indicates well-calibrated forecasts.
6. **VaR backtest.** Compute $\mathrm{VaR}_t^\alpha = \hat{\mu}_t + \hat{\sigma}_t F^{-1}(\alpha; \nu)$. Kupiec unconditional-coverage test; Christoffersen independence test. Failure-rate divergence from $\alpha$ at the 5% significance level signals model breakdown.
7. **Out-of-sample loss.** QLIKE loss $\sum_t (\mathrm{RV}_t / \hat{\sigma}_t^2 - \log(\mathrm{RV}_t / \hat{\sigma}_t^2) - 1)$ is robust to RV proxy noise; compare across specifications with Diebold-Mariano test.
8. **Parameter stability.** Plot $\hat{\alpha} + \hat{\beta}$ over rolling refits; values approaching 1 signal integrated-GARCH behavior (IGARCH) and indicate structural shift in volatility persistence.

## Connections to other models

- **HAR-RV (Model 9).** Direct competitor for daily vol forecasting; HAR-RV typically dominates when intraday data is available.
- **Stochastic volatility models (Layer 2 derivatives).** Heston is the continuous-time analogue; GARCH(1,1) is the diffusion limit of a particular GARCH family (Nelson 1990).
- **Implied volatility (Layer 2 options).** GARCH gives physical-measure vol; the difference between IV and GARCH-forecast vol is the volatility risk premium.
- **Avellaneda-Lee (Model 7).** $\sigma_i$ in the residual OU can be GARCH-filtered for time-varying scaling.
- **Risk management (Layer 5).** Conditional VaR/ES feed position-sizing constraints and capital allocation.
- **Regime-switching models (Layer 4).** Markov-switching GARCH allows discrete jumps in $(\omega, \alpha, \beta)$ across regimes.
- **Option pricing (Layer 2).** GARCH option pricing (Duan 1995) provides a no-arbitrage embedding for hedging.

## Limitations and failure modes

1. **Slow response to volatility regime shifts.** GARCH adapts geometrically at rate $\alpha + \beta \approx 0.98$; the effective memory is ~50 days. Realized vol drops within hours of a regime change while GARCH lags by weeks.
2. **Identification issues at high persistence.** When $\alpha + \beta \to 1$ (IGARCH), the unconditional variance is undefined and forecasts beyond a few days are unreliable.
3. **Microstructure noise in input returns.** Stale closes and gap risk inflate $\sigma_t^2$ estimates; the GARCH then overreacts to artifacts.
4. **Single-day jumps.** Earnings, FOMC announcements, and exogenous events produce one-off large returns that GARCH treats as persistent volatility; this is the wrong interpretation. Robust GARCH variants (Boudt-Croux-Laurent) downweight jumps.
5. **Asymmetry mis-specification.** Plain GARCH underestimates post-crash volatility; GJR/EGARCH correct this but only partially — leverage effects are themselves regime-dependent.
6. **Distribution mis-specification.** Gaussian innovations produce VaR estimates that are systematically too narrow; Student-$t$ helps but extreme tail estimates ($\alpha < 1\%$) remain unreliable.
7. **Parameter instability across refits.** Estimated $(\omega, \alpha, \beta)$ can jump between refits when the rolling window crosses a major event (e.g. March 2020 enters/exits the window).
8. **Aggregation invariance.** GARCH(1,1) calibrated on daily data does not aggregate cleanly to weekly or monthly horizons; the implied long-horizon vol is biased.
9. **Comparison to realized vol.** When intraday data is available, RV-based forecasts (HAR) systematically outperform daily-return GARCH; GARCH is the right tool only when intraday data is unavailable or too noisy.

## References

- Engle, R. F. (1982). *Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.* Econometrica 50(4): 987–1007.
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity.* Journal of Econometrics 31(3): 307–327.
- Nelson, D. B. (1991). *Conditional heteroskedasticity in asset returns: A new approach.* Econometrica 59(2): 347–370.
- Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). *On the relation between the expected value and the volatility of the nominal excess return on stocks.* Journal of Finance 48(5): 1779–1801.
- Bollerslev, T. and Wooldridge, J. M. (1992). *Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances.* Econometric Reviews 11(2): 143–172.
- Duan, J.-C. (1995). *The GARCH option pricing model.* Mathematical Finance 5(1): 13–32.
- Engle, R. F. and Ng, V. K. (1993). *Measuring and testing the impact of news on volatility.* Journal of Finance 48(5): 1749–1778.
- Hansen, P. R. and Lunde, A. (2005). *A forecast comparison of volatility models: Does anything beat a GARCH(1,1)?* Journal of Applied Econometrics 20(7): 873–889.
- Sheppard, K. (2023). `arch` Python package documentation. https://arch.readthedocs.io.
