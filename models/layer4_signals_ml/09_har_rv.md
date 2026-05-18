# HAR-RV — Heterogeneous Autoregressive Realized Volatility (Corsi 2009)

> A three-component linear regression of realized variance on its daily, weekly, and monthly lags that mimics long memory in volatility while remaining a simple OLS model. Layer 4 — Signals & ML. Primary use: one-day-ahead realized-volatility forecasts for liquid US equities and ETFs; superior to short-memory GARCH when intraday data is available.

## Mathematical formulation

### Realized variance from intraday returns

Within a trading day $t$, partition the trading session into $M$ equal-length intraday intervals (typical: $M = 78$ five-minute bars for a US 9:30–16:00 session). Let $r_{t,i}$ be the log return over the $i$-th interval of day $t$:

$$r_{t,i} = \log P_{t,i} - \log P_{t,i-1}, \qquad i = 1, \ldots, M.$$

The **daily realized variance** is the sum of squared intraday returns:

$$\mathrm{RV}_t^{(d)} = \sum_{i=1}^{M} r_{t,i}^2.$$

Realized volatility is its square root: $\mathrm{RVol}_t = \sqrt{\mathrm{RV}_t^{(d)}}$.

**Theoretical justification (Andersen-Bollerslev-Diebold-Labys 2003).** If the log-price follows a continuous semimartingale

$$d \log P_t = \mu_t \, dt + \sigma_t \, dW_t,$$

then as $M \to \infty$,

$$\mathrm{RV}_t^{(d)} \xrightarrow{p} \int_{t-1}^{t} \sigma_s^2 \, ds = \mathrm{IV}_t,$$

the **integrated variance** over day $t$. RV is therefore an asymptotically consistent estimator of the (unobserved) integrated variance — a model-free quantity, not dependent on any parametric volatility specification.

### Multi-scale aggregations

Define the **weekly** and **monthly** averages of daily realized variance:

$$\mathrm{RV}_t^{(w)} = \frac{1}{5} \sum_{k=0}^{4} \mathrm{RV}_{t-k}^{(d)}, \qquad \mathrm{RV}_t^{(m)} = \frac{1}{22} \sum_{k=0}^{21} \mathrm{RV}_{t-k}^{(d)}.$$

The constants $5$ and $22$ are standard for US trading days per week and per month respectively.

### HAR-RV regression

Corsi's HAR-RV forecasts one-day-ahead realized variance as a linear combination of these three components:

$$\mathrm{RV}_{t+1}^{(d)} = c + \beta_d \mathrm{RV}_t^{(d)} + \beta_w \mathrm{RV}_t^{(w)} + \beta_m \mathrm{RV}_t^{(m)} + \epsilon_{t+1}.$$

The conditional mean is the volatility forecast:

$$\widehat{\mathrm{RV}}_{t+1 \mid t}^{(d)} = \hat{c} + \hat{\beta}_d \mathrm{RV}_t^{(d)} + \hat{\beta}_w \mathrm{RV}_t^{(w)} + \hat{\beta}_m \mathrm{RV}_t^{(m)}.$$

The three regressors are highly correlated by construction (they share many of the same daily RV values), but OLS remains consistent — the issue is on standard errors, which require Newey-West correction.

**Heterogeneous-agent interpretation.** Different market participants operate at different horizons: high-frequency traders react to daily volatility, swing traders to weekly, asset allocators to monthly. Each group's volatility expectation feeds back into the next-day price. The three-component decomposition is a parsimonious proxy for this heterogeneity. Empirically, the regression mimics the **long-memory** feature of realized volatility (autocorrelations decaying as $\rho(h) \sim h^{-d}$, $d \approx 0.4$) without resorting to fractional integration.

### Log-HAR-RV

Because $\mathrm{RV}_t^{(d)}$ is right-skewed and roughly log-normally distributed, the **log-HAR** variant is preferred for forecasting:

$$\log \mathrm{RV}_{t+1}^{(d)} = c + \beta_d \log \mathrm{RV}_t^{(d)} + \beta_w \log \mathrm{RV}_t^{(w)} + \beta_m \log \mathrm{RV}_t^{(m)} + \epsilon_{t+1}.$$

Convert back via the lognormal correction:

$$\widehat{\mathrm{RV}}_{t+1 \mid t}^{(d)} = \exp\Bigl( \widehat{\log \mathrm{RV}_{t+1}^{(d)}} + \tfrac{1}{2} \hat{\sigma}_\epsilon^2 \Bigr)$$

where $\hat{\sigma}_\epsilon^2$ is the OLS residual variance.

### Multi-horizon HAR

Multi-step forecasts replace the dependent variable with the multi-period average:

$$\mathrm{RV}_{t+1:t+h}^{(d)} = \frac{1}{h} \sum_{k=1}^{h} \mathrm{RV}_{t+k}^{(d)}.$$

Direct ("flat") forecasts regress this average on the same right-hand side, avoiding the iterative compounding of one-step errors.

### HAR-CJ — continuous and jump components (Andersen, Bollerslev, Diebold 2007)

**Bipower variation** isolates the continuous component of variance:

$$\mathrm{BV}_t = \mu_1^{-2} \, \frac{M}{M - 1} \sum_{i=2}^{M} |r_{t,i}| |r_{t,i-1}|, \qquad \mu_1 = \sqrt{2/\pi} = \mathbb{E}|z| \text{ for } z \sim \mathcal{N}(0,1).$$

Under a jump-diffusion model $d \log P_t = \mu_t \, dt + \sigma_t \, dW_t + dJ_t$ with $J_t$ a finite-activity jump process:

$$\mathrm{RV}_t \xrightarrow{p} \int_{t-1}^{t} \sigma_s^2 \, ds + \sum_{t-1 < s \leq t} \kappa_s^2, \qquad \mathrm{BV}_t \xrightarrow{p} \int_{t-1}^{t} \sigma_s^2 \, ds$$

where $\kappa_s$ is the size of a jump at time $s$. The **jump variation**:

$$J_t = \max(\mathrm{RV}_t - \mathrm{BV}_t, 0).$$

HAR-CJ regresses $\mathrm{RV}_{t+1}$ on three components of continuous variation $C_t = \mathrm{RV}_t - J_t$ (daily/weekly/monthly) and the analogous components of $J_t$. Empirically the jump components carry weak forecast power — jumps are largely transient — while continuous variation drives persistence.

### Realized semivariance — downside / upside decomposition

Barndorff-Nielsen, Kinnebrock and Shephard (2010):

$$\mathrm{RV}_t^- = \sum_{i=1}^{M} r_{t,i}^2 I(r_{t,i} < 0), \qquad \mathrm{RV}_t^+ = \sum_{i=1}^{M} r_{t,i}^2 I(r_{t,i} > 0).$$

Substituting $\mathrm{RV}_t^-$ for $\mathrm{RV}_t$ in the daily lag (semivariance-HAR) captures the leverage effect: negative-return variance forecasts next-day vol better than positive-return variance.

## Intuition and what this model addresses

Realized volatility built from intraday returns is observable and accurate — it sidesteps the model-dependence of GARCH or the look-ahead of implied vol. Once you have $\mathrm{RV}_t$, forecasting it becomes a straightforward time-series problem.

Empirically, realized vol has two stylized facts that simple AR(1) on RV fails to capture:

1. **Long memory.** The autocorrelation function decays slowly (hyperbolically), not exponentially.
2. **Multi-scale persistence.** Shocks at the daily scale dissipate quickly; weekly-scale persistence is intermediate; monthly-scale is very long-lived.

HAR captures both with three terms and one OLS regression — no fractional integration, no MCMC, no iterative MLE. Corsi (2009) showed that this simple specification out-forecasts both AR(p) on RV and standard GARCH variants on essentially every horizon and every asset class tested.

The economic interpretation through heterogeneous agents is what gives the model its name, but the practical content is statistical: the three lags span the empirical decay structure of realized variance well enough to crowd out more sophisticated long-memory specifications.

## Calibration approach

1. **Intraday sampling frequency.** $5$ minutes is the empirical sweet spot for liquid US equities. Higher frequency increases microstructure noise (bid-ask bounce, stale quotes); lower frequency increases estimator variance. For less liquid names, $15$- or $30$-minute returns reduce noise.
2. **RV estimator choice.**
   - **Plain RV** for liquid, high-volume names.
   - **Two-scales realized volatility (Zhang-Mykland-Aït-Sahalia 2005)** or **realized kernel (Barndorff-Nielsen-Hansen-Lunde-Shephard 2008)** for noise-robust estimation when microstructure is significant.
   - **Overnight return handling.** Either (a) treat overnight as one extra observation: $\mathrm{RV}_t = r_{\text{overnight},t}^2 + \sum_i r_{t,i}^2$, (b) ignore overnight and scale up by $252/T_{\text{trading}}$ ratio, or (c) model overnight separately. Convention (a) is most common.
3. **Outliers and zero-RV days.** Half-trading days (e.g. day after Thanksgiving) produce structurally low RV; either include with proper accounting (fewer intraday bars) or exclude.
4. **Sample window.** Minimum 2 years of daily RV for stable OLS; 5+ years is preferable. HAR is robust to window length because of its small parameter count (four).
5. **Regression specification.** Default to log-HAR for forecasting; level-HAR for in-sample interpretation. Newey-West HAC standard errors with $\sim 5$ lags.
6. **Extensions.**
   - HAR-CJ when jump activity is significant (test via $z$-statistic on $\mathrm{RV}_t - \mathrm{BV}_t$ following Huang and Tauchen 2005).
   - Semivariance-HAR for asymmetric forecasting.
   - HAR with implied-vol regressor (HAR-IV) when option data is available.
   - LASSO or random forest on a wider set of HAR lags for higher-dimensional extensions.

## Algorithm outline

```
INPUT:  ticker, intraday sampling frequency (e.g. "5m"), forecast horizon h
OUTPUT: parameter estimates, in-sample fitted RV path, h-step forecast

# --- Step 1: Build daily realized variance series ---
1. Fetch intraday bars for the lookback period.
2. For each trading day t:
     a. Compute log returns r[t,i] = log(P[t,i] / P[t,i-1]) for i = 1..M.
     b. Optionally append the overnight return r_overnight to the series.
     c. RV_d[t] = sum_i r[t,i]^2.
     d. (Optional) BV[t] = (pi/2) * (M/(M-1)) * sum_{i=2..M} |r[t,i]| * |r[t,i-1]|.
     e. (Optional) Compute realized semivariances RV_minus, RV_plus.

# --- Step 2: Build multi-scale aggregations ---
3. For t = 22..T_total:
     RV_w[t] = mean(RV_d[t-4 .. t])
     RV_m[t] = mean(RV_d[t-21 .. t])

# --- Step 3: OLS regression ---
4. Construct response y[t] = log(RV_d[t]) (or RV_d[t] for level model) for
   t = 23..T_total.
5. Construct regressor matrix:
     X[t] = [1, log(RV_d[t-1]), log(RV_w[t-1]), log(RV_m[t-1])]   (log spec)
     X[t] = [1, RV_d[t-1],      RV_w[t-1],      RV_m[t-1]     ]   (level spec)
6. Estimate (c, beta_d, beta_w, beta_m) by OLS: theta_hat = (X^T X)^{-1} X^T y.
7. Compute Newey-West HAC standard errors with lag = floor(4 * (T/100)^(2/9)).

# --- Step 4: One-step forecast ---
8. At time T:
     compute RV_d[T], RV_w[T], RV_m[T]
     log_forecast = c + beta_d*log(RV_d[T]) + beta_w*log(RV_w[T])
                      + beta_m*log(RV_m[T])
     RV_forecast  = exp(log_forecast + 0.5 * sigma_resid^2)        (log spec)
     RVol_forecast = sqrt(RV_forecast)
     Annualize: sigma_ann = sqrt(252 * RV_forecast).

# --- Step 5: Multi-step forecast ---
9. Direct method: re-fit with dependent variable
     y_h[t] = mean(RV_d[t+1 .. t+h])  (level)  or  log thereof  (log)
   for each horizon h of interest. Avoids compounded errors from iteration.

# --- Step 6: Diagnostics ---
10. Residual analysis: Ljung-Box on epsilon, ARCH-LM on epsilon^2.
11. Mincer-Zarnowitz regression of realized RV on forecast.
12. QLIKE loss vs benchmark (GARCH).
```

## yfinance data requirements

```python
import yfinance as yf
import numpy as np
import pandas as pd

# 5-minute bars: yfinance allows max 60 days of lookback at this resolution
data_5m = yf.Ticker("SPY").history(period="60d",
                                   interval="5m",
                                   auto_adjust=True)

# Build daily RV from 5-minute log returns
data_5m["log_return"] = np.log(data_5m["Close"]).diff()
data_5m["date"]       = data_5m.index.date
rv_daily = (data_5m["log_return"]**2
            .groupby(data_5m["date"]).sum())
```

**Fundamental data limitation:** yfinance restricts intraday history:

| Interval | Maximum lookback |
|----------|-------------------|
| 1m       | 7 days            |
| 2m       | 60 days           |
| 5m       | 60 days           |
| 15m      | 60 days           |
| 30m      | 60 days           |
| 60m / 1h | 730 days          |
| 1d       | Unlimited (back to IPO) |

For a production HAR-RV model requiring multi-year RV history, this is a real constraint. Three strategies:

### Strategy A — Rolling 60-day intraday refresh

Pull 5-minute bars in rolling 60-day windows daily; persist daily RV values locally. After ~3 months of operation, you have a usable history; after a year, robust. This is the **standard yfinance-only approach**.

### Strategy B — 1-hour intraday for longer history

`interval="1h"` (or `60m`) provides 730 days of lookback. RV from hourly returns is noisier than 5-minute RV (fewer observations per day: ~7 bars) but immediately available. Use as a transition while Strategy A accumulates data.

### Strategy C — OHLC range estimators on daily data

When intraday data is unavailable for the required history, use OHLC-based volatility estimators:

**Garman-Klass (1980):**

$$\mathrm{RV}_t^{\mathrm{GK}} = \tfrac{1}{2} [\log(H_t/L_t)]^2 - (2 \log 2 - 1) [\log(C_t / O_t)]^2$$

where $O, H, L, C$ are the day's open, high, low, close. Variance-efficient: roughly 5× more efficient than close-to-close.

**Yang-Zhang (2000):** combines overnight, open-to-close, and Garman-Klass components; minimum-variance estimator that handles overnight gaps:

$$\mathrm{RV}_t^{\mathrm{YZ}} = \sigma_{\text{overnight}}^2 + k \sigma_{\text{open-close}}^2 + (1 - k) \sigma_{\text{RS}}^2$$

with $\sigma_{\text{RS}}^2$ the Rogers-Satchell estimator and $k = 0.34 / (1.34 + (n+1)/(n-1))$.

OHLC estimators are biased proxies for true integrated variance — they tend to underestimate it by 10–20% — but for HAR fitting the bias is absorbed into the intercept and the regression coefficients still capture the persistence structure. **Practical recommendation:** use OHLC-based RV (via the daily yfinance feed) for the long-history HAR fit, and refresh with 5-minute RV (Strategy A) for the most recent 60 days. Validate consistency by overlapping both methods on the same window.

```python
# Garman-Klass from daily OHLC
data_d = yf.Ticker("SPY").history(period="5y", interval="1d",
                                  auto_adjust=True)
hi  = np.log(data_d["High"]  / data_d["Low"])**2
oc  = np.log(data_d["Close"] / data_d["Open"])**2
rv_gk = 0.5 * hi - (2*np.log(2) - 1) * oc
```

## Calibration / refit frequency

- **OLS coefficient re-estimation:** monthly; HAR is parsimonious and stable, so weekly or daily refits add noise without improving forecast accuracy.
- **Rolling sample window:** 2–5 years of daily RV. Longer windows are acceptable because HAR has only 4 free parameters; over-fitting is not a concern.
- **Intraday RV update:** daily after market close; can be performed at 16:30 ET once the 5-minute bars settle.
- **Sampling frequency choice:** reviewed annually; if microstructure noise changes (e.g. tick size reform), re-validate optimal frequency via volatility signature plot — plot $\mathrm{RV}(\Delta)$ against $\Delta$ and pick the smallest $\Delta$ where the curve is flat.
- **Extension specifications (HAR-CJ, semivariance-HAR):** assess quarterly via out-of-sample QLIKE/Diebold-Mariano comparisons.

## Validation and diagnostics

1. **In-sample $R^2$.** Log-HAR typically achieves $R^2 \in [0.6, 0.8]$ on liquid US equities — well above any GARCH-based proxy.
2. **Mincer-Zarnowitz regression.** Run $\mathrm{RV}_{t+1} = a + b \widehat{\mathrm{RV}}_{t+1 \mid t} + u_{t+1}$; well-calibrated forecasts have $a = 0$, $b = 1$. Joint test via Wald statistic.
3. **QLIKE loss.** Robust to RV measurement error (Patton 2011). Compare HAR vs GARCH vs RV-AR(p) out-of-sample; HAR almost always wins on liquid equities.
4. **Diebold-Mariano test.** Pairwise comparison of forecast losses across competing models.
5. **Residual diagnostics.** Ljung-Box on regression residuals; ARCH-LM on squared residuals — should fail to reject (vol of RV captured).
6. **Forecast bias by regime.** Bin forecasts by realized regime (calm vs turbulent); HAR tends to underforecast during transitions from calm to turbulent. Stratified Mincer-Zarnowitz flags this.
7. **Volatility signature plot.** Confirms chosen sampling frequency is in the flat region of $\mathrm{RV}(\Delta)$ — neither dominated by microstructure noise (low $\Delta$) nor by estimator variance (high $\Delta$).
8. **Stability of coefficients.** Recursive estimation of $\hat{\beta}_d, \hat{\beta}_w, \hat{\beta}_m$ over time; coefficients should be approximately stable. Sudden jumps indicate structural shift.

## Connections to other models

- **GARCH family (Model 8).** Direct competitor; HAR generally dominates when intraday data is available. The two can be combined: GARCH on the HAR residual to capture volatility of volatility (HAR-GARCH).
- **Realized covariance / multivariate HAR.** Vector-HAR forecasts the full realized covariance matrix; portfolio applications.
- **Implied volatility (Layer 2 options).** HAR-IV augments the regressor set with VIX or option-implied vol; the combination beats either alone in many studies.
- **HEAVY models (Shephard-Sheppard 2010).** Multivariate framework that uses RV as a measurement equation for the latent variance — closer to a state-space generalization of HAR.
- **Jump-diffusion models (Layer 2 derivatives).** HAR-CJ provides empirical inputs to jump-intensity estimation.
- **Avellaneda-Lee (Model 7).** Time-varying $\sigma_{\text{eq},i}$ in the OU fit can use HAR-RV-based vol filtering for crisper s-scores.
- **VaR / ES (Layer 5 risk).** HAR forecasts feed directly into one-day-ahead quantile estimates via the empirical RV distribution.

## Limitations and failure modes

1. **Intraday data dependence.** The full power of HAR requires intraday returns; without them, OHLC-range proxies must be used, with attendant biases. yfinance's 60-day cap on 5-minute history is the binding constraint.
2. **Microstructure noise.** At sampling frequencies below ~$5$ minutes, bid-ask bounce inflates RV; the volatility signature plot must be checked for each asset before fixing $\Delta$.
3. **Jumps treated as continuous variation.** Plain HAR-RV does not distinguish; jump days produce inflated forecasts for several days afterward. HAR-CJ partially fixes this but jumps remain hard to forecast.
4. **Overnight gap.** US equity overnight returns can dominate intraday RV; treatment of the overnight return is a non-trivial modeling choice and changes the forecast level materially.
5. **Half-trading days.** Shortened sessions (Black Friday, day before Christmas) have structurally different RV; either drop or scale by the ratio of trading minutes.
6. **Regime change.** HAR is linear and stationary; sudden volatility regime shifts (March 2020) cause forecast errors that persist for weeks until the monthly average updates.
7. **Single-asset focus.** Plain HAR is univariate; multivariate extensions are higher-dimensional and require care with positive-definiteness of forecast realized covariance.
8. **Coefficient identification.** $\hat{\beta}_d, \hat{\beta}_w, \hat{\beta}_m$ individually are highly collinear; only their sum and the overall fit are robust. Avoid economic interpretation of individual coefficients.
9. **Forecast horizon decay.** One-day forecasts are excellent; weekly forecasts are good; monthly forecasts collapse to the sample mean and any model beats nothing only marginally.
10. **Liquidity dependence.** For thinly traded names, 5-minute returns contain many zero-return intervals; RV is biased downward. Switch to longer intervals or use OHLC range estimators.

## References

- Corsi, F. (2009). *A simple approximate long-memory model of realized volatility.* Journal of Financial Econometrics 7(2): 174–196.
- Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2003). *Modeling and forecasting realized volatility.* Econometrica 71(2): 579–625.
- Andersen, T. G., Bollerslev, T. and Diebold, F. X. (2007). *Roughing it up: Including jump components in the measurement, modeling, and forecasting of return volatility.* Review of Economics and Statistics 89(4): 701–720.
- Barndorff-Nielsen, O. E. and Shephard, N. (2004). *Power and bipower variation with stochastic volatility and jumps.* Journal of Financial Econometrics 2(1): 1–37.
- Barndorff-Nielsen, O. E., Kinnebrock, S. and Shephard, N. (2010). *Measuring downside risk: realised semivariance.* In *Volatility and Time Series Econometrics*, Oxford University Press.
- Zhang, L., Mykland, P. A. and Aït-Sahalia, Y. (2005). *A tale of two time scales: Determining integrated volatility with noisy high-frequency data.* Journal of the American Statistical Association 100(472): 1394–1411.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2008). *Designing realized kernels to measure the ex post variation of equity prices in the presence of noise.* Econometrica 76(6): 1481–1536.
- Garman, M. B. and Klass, M. J. (1980). *On the estimation of security price volatilities from historical data.* Journal of Business 53(1): 67–78.
- Yang, D. and Zhang, Q. (2000). *Drift-independent volatility estimation based on high, low, open, and close prices.* Journal of Business 73(3): 477–492.
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies.* Journal of Econometrics 160(1): 246–256.
- Huang, X. and Tauchen, G. (2005). *The relative contribution of jumps to total price variance.* Journal of Financial Econometrics 3(4): 456–499.
