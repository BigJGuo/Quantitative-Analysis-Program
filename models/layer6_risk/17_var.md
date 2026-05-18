# Value-at-Risk (Parametric, Historical-Sim, Monte Carlo)

> A quantile-based market-risk measure giving the loss threshold not exceeded with confidence $\alpha$ over a fixed horizon. Layer 6 — Risk & Capital. Primary use: firmwide daily P&L risk reporting, trader limit enforcement, and the legacy Basel internal-models capital metric.

## Mathematical formulation

Let $L$ denote the portfolio **loss** over a horizon $\Delta t$ — by convention $L = -\Delta V$ so a positive $L$ is an adverse outcome. For a confidence level $\alpha \in (0, 1)$ (typically $\alpha = 0.95$ or $0.99$), Value-at-Risk is the $\alpha$-quantile of the loss distribution:

$$
\text{VaR}_\alpha(L) \;=\; \inf \{\, x \in \mathbb{R} : P(L \leq x) \geq \alpha \,\} \;=\; F_L^{-1}(\alpha)
$$

Equivalently, $P(L > \text{VaR}_\alpha) = 1 - \alpha$. When the loss distribution is continuous, $F_L^{-1}$ is just the inverse CDF.

### Parametric (variance-covariance) VaR

Assume single-period log-returns $r$ satisfy $r \sim \mathcal{N}(\mu, \sigma^2)$, so $L = -r$ has distribution $\mathcal{N}(-\mu, \sigma^2)$. Then

$$
\text{VaR}_\alpha \;=\; -\mu + \sigma\, z_\alpha, \qquad z_\alpha = \Phi^{-1}(\alpha).
$$

For small horizons one often drops $\mu$ (it is dominated by noise at daily frequency), giving the clean form $\text{VaR}_\alpha = \sigma z_\alpha$.

For a portfolio with weights $\mathbf{w} \in \mathbb{R}^n$ on $n$ assets with covariance matrix $\boldsymbol{\Sigma}$,

$$
\sigma_P \;=\; \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma}\, \mathbf{w}}, \qquad \text{VaR}_\alpha^P \;=\; V_0 \cdot \sigma_P\, z_\alpha
$$

where $V_0$ is current portfolio market value. The derivation uses the fact that any linear combination of joint-normal returns is itself normal.

### Historical Simulation (HS) VaR

Let $\{r_{t-N+1}, \ldots, r_t\}$ be the most recent $N$ daily return vectors. The HS one-day P&L vector under today's positions $\mathbf{w}_t$ (in dollar exposures) is

$$
\text{PnL}_i \;=\; \mathbf{w}_t^\top \mathbf{r}_{t - N + i}, \quad i = 1, \ldots, N.
$$

VaR is the empirical $(1-\alpha)$-quantile of $\{-\text{PnL}_i\}$. With $N = 250$ and $\alpha = 0.99$, this is roughly the second or third worst loss in the sample.

### Monte Carlo (MC) VaR

Specify a generative model for risk-factor returns — multivariate Student-$t$, GARCH-filtered residuals, Gaussian copula with $t$-marginals, or a more elaborate SDE. Draw $K$ independent samples $\mathbf{r}^{(k)}$, revalue the portfolio $V^{(k)} = V(\mathbf{r}^{(k)})$, form losses $L^{(k)} = V_0 - V^{(k)}$, and report

$$
\widehat{\text{VaR}}_\alpha \;=\; \widehat F_L^{-1}(\alpha)
$$

as the $\lceil \alpha K \rceil$-th order statistic.

### Horizon scaling

Under iid Gaussian increments, $\sigma$ scales as $\sqrt{\Delta t}$, so $\text{VaR}_\alpha^{(h)} = \sqrt{h}\,\text{VaR}_\alpha^{(1)}$. The Basel regulatory standard until 2023 was 99% confidence at a 10-day horizon, yielding the famous $\sqrt{10}$ scaling. This is wrong under volatility clustering and serial correlation; the standard fix is **filtered historical simulation**: standardize each historical return by its GARCH-implied vol at that date, then re-scale the resulting iid-looking residuals by the current GARCH vol forecast.

## Intuition and what this model addresses

VaR answers a single, executive-readable question: *on a "normal-bad" day, how much could we lose?* It compresses an entire P&L distribution into one number that is easy to compare across desks, days, and counterparties.

Three things VaR is good at: (i) **comparability** across heterogeneous books because the units (dollars or % of NAV) are universal; (ii) **limit enforcement** — every trader gets a VaR budget which is easy to monitor intraday; (iii) **rough regulatory capital** — it scales naturally with position size.

Three things VaR is structurally bad at: (i) it says **nothing about losses beyond the quantile** — a book that loses $\text{VaR}_\alpha + \$1$ and a book that loses $100\times \text{VaR}_\alpha$ look identical; (ii) it is **not subadditive** — $\text{VaR}(L_1 + L_2)$ can exceed $\text{VaR}(L_1) + \text{VaR}(L_2)$, so diversification can paradoxically increase reported risk; (iii) under heavy-tailed returns it badly underestimates true loss probabilities because the Gaussian assumption (or short-window empirical CDF) is missing the rare-event mass.

## Computation approach

Three implementations differ in their bias-variance trade-off:

| Method | Strengths | Weaknesses |
|---|---|---|
| Parametric | Fastest; closed form; easy to attribute | Bad tails; assumes linearity in factors |
| Historical Sim | No distributional assumption; preserves correlations; auditable | Stuck in recent regime; tails undersampled |
| Monte Carlo | Handles options/non-linearity; flexible distributions | Slow; model-dependent; convergence of tail quantile is $O(1/\sqrt{K})$ |

Equity market-makers typically prefer historical simulation because (i) options books have material gamma/vega that parametric VaR ignores, (ii) traders can audit every "scenario day" against actual price tape, (iii) fat tails and correlation breaks during stress are preserved naturally without parametrization.

## Algorithm outline

```
ALGORITHM: Compute_VaR(positions, lookback_N, alpha, method)
INPUT:
  positions: dict {ticker -> dollar_notional}
  lookback_N: integer (e.g. 250)
  alpha: float in (0,1)
  method: "parametric" | "historical" | "monte_carlo"
OUTPUT:
  VaR_1d, VaR_10d, full P&L vector

STEP 1 — Pull data
  tickers <- positions.keys()
  prices <- yfinance.download(tickers, period=lookback_N+50 days, auto_adjust=True)["Close"]
  rets <- prices.pct_change().dropna().tail(lookback_N)   # shape (N, n_assets)

STEP 2 — Build weight vector
  V0 <- sum(positions.values())
  w_dollar <- vector aligned with rets.columns of positions[ticker]

STEP 3 — Branch by method

  IF method == "parametric":
    Sigma <- rets.cov().values         # (n,n)
    mu_p <- (w_dollar . mean(rets))    # negligible at daily frequency
    sigma_p <- sqrt( w_dollar^T Sigma w_dollar )
    z <- scipy.stats.norm.ppf(alpha)
    VaR_1d <- -mu_p + sigma_p * z

  ELIF method == "historical":
    pnl_vec <- rets.values @ w_dollar        # shape (N,)
    loss_vec <- -pnl_vec
    VaR_1d <- numpy.quantile(loss_vec, alpha)

  ELIF method == "monte_carlo":
    mu_hat <- rets.mean().values
    Sigma_hat <- rets.cov().values
    # OPTIONAL: fit GARCH(1,1) per asset to get conditional vol
    #          and standardize residuals before sampling
    L <- numpy.linalg.cholesky(Sigma_hat)
    K <- 50_000
    Z <- numpy.random.standard_normal((K, n_assets))
    # For fat tails, replace with multivariate-t draws
    sim_rets <- mu_hat + Z @ L.T
    pnl_sim <- sim_rets @ w_dollar
    VaR_1d <- numpy.quantile(-pnl_sim, alpha)

STEP 4 — Horizon scaling
  VaR_10d_iid <- sqrt(10) * VaR_1d
  # If filtered HS: replace sqrt(10) with cumulative GARCH variance forecast

STEP 5 — Return
  return {VaR_1d, VaR_10d_iid, pnl_vector}
```

For filtered historical simulation, replace the raw `rets` with $\tilde r_t = r_t / \hat\sigma_t$ (GARCH-standardized), compute HS VaR on $\tilde r$, then multiply by $\hat\sigma_{t+1}$ to rescale to current-vol units.

## yfinance data requirements

- **Adjusted close prices** for all portfolio tickers and any benchmark factors. Use `yf.download(tickers, period="5y", auto_adjust=True)["Close"]`.
- **Minimum window**: $N = 250$ days for HS-VaR at $\alpha = 0.99$ is the regulatory floor — at $\alpha = 0.99$ only $\approx 2$-$3$ observations populate the tail, so use $N = 500$ or $750$ when feasible.
- **Currency consistency**: convert all positions to a single base currency before aggregating.
- **Survivorship**: yfinance silently omits delisted tickers; for production work supplement with a delisted-name database.
- **Corporate actions**: `auto_adjust=True` adjusts splits and dividends. Verify on a known split (e.g., AAPL 2020-08-31, 4-for-1) before trusting.

## Refit / refresh frequency

- **Daily**: HS-VaR recomputed every trading day, rolling-window updated with $t-1$ close.
- **Daily**: parametric covariance $\boldsymbol{\Sigma}$ — EWMA with $\lambda = 0.94$ (RiskMetrics) or 60-day rolling cov.
- **Weekly**: full Monte Carlo refit (GARCH parameters, copula calibration, correlation regime).
- **Quarterly**: model review — Kupiec / Christoffersen backtests over trailing 250 days.

## Validation and diagnostics

### Kupiec POF (proportion-of-failures) test

Under the null that VaR is correctly specified, the indicator $I_t = \mathbf{1}\{L_t > \text{VaR}_\alpha\}$ is Bernoulli with $p = 1 - \alpha$. Over $T$ days, the number of breaches $x$ is Binomial$(T, p)$. The likelihood-ratio statistic is

$$
\text{LR}_{\text{POF}} \;=\; -2 \ln \!\left( \frac{p^x (1-p)^{T-x}}{\hat p^x (1-\hat p)^{T-x}} \right) \;\sim\; \chi^2_1
$$

where $\hat p = x/T$. For $T = 250$ days and $\alpha = 0.99$, expected breaches $= 2.5$; the Basel **traffic-light**:

- **Green**: 0–4 breaches (no capital penalty)
- **Yellow**: 5–9 breaches (additive scalar 0.40–0.85)
- **Red**: $\geq 10$ breaches (penalty plus mandatory model review)

### Christoffersen independence test

Tests serial dependence of breaches. Under correct conditional VaR, breaches should not cluster. Define $\pi_{ij} = P(I_t = j \mid I_{t-1} = i)$; the null is $\pi_{01} = \pi_{11}$. LR statistic is $\chi^2_1$.

### Diagnostic plots

- Empirical breach rate vs. nominal $1-\alpha$.
- Time-series of $L_t$ and $\text{VaR}_\alpha^{(t)}$ overlay; mark breaches.
- QQ-plot of standardized P&L $L_t / \widehat{\sigma}_t$ against $\mathcal{N}(0,1)$ or $t_\nu$.

## Connections to other models

- **Model 18 (Expected Shortfall)** — the regulatory successor; ES integrates VaR over $\alpha$ and so always dominates VaR in tail content.
- **Model 19 (Stress Tests)** — VaR is the statistical sister; stress tests are the deterministic complement.
- **Model 20 (Liquidity-Adjusted VaR)** — extends VaR with bid-ask cost and FRTB liquidity horizons.
- **Layer 3 covariance estimators** — parametric VaR depends critically on $\boldsymbol{\Sigma}$ (Ledoit-Wolf shrinkage, factor models).
- **Layer 4 GARCH/HAR vol forecasts** — feed filtered HS and parametric VaR.

## Limitations and failure modes

- **Tail blindness**: VaR ignores everything beyond the quantile. A $\$10\text{M}$ VaR book that occasionally loses $\$1\text{B}$ tests as "fine" until it doesn't.
- **Non-subadditivity**: pathological for combining portfolios; one of the chief reasons FRTB switched to ES.
- **Gaussian parametric VaR** undershoots fat-tailed losses by factors of $2$–$5\times$ for equity index returns at $\alpha = 0.99$.
- **Historical simulation** assumes the past 1–2 years contain the future; in regime change (Aug 2007 quant quake, Mar 2020 COVID gap) the empirical CDF lags reality by weeks.
- **Procyclicality**: VaR contracts as volatility falls, encouraging leverage, then explodes in a stress event forcing forced unwinds. See August 2007 quant quake.
- **Model risk concentration**: when many institutions use similar VaR engines, breaches cluster, producing feedback loops.

## References

- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd ed., McGraw-Hill.
- Hull, J. (2018). *Risk Management and Financial Institutions*, 5th ed., Wiley.
- Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives* 3.
- Christoffersen, P. (1998). "Evaluating Interval Forecasts." *International Economic Review* 39.
- Barone-Adesi, G., Giannopoulos, K., Vosper, L. (1999). "VaR Without Correlations for Portfolios of Derivative Securities." *Journal of Futures Markets* 19. [Filtered HS.]
- BCBS (1996, 2009). *Amendment to the Capital Accord to Incorporate Market Risks* and *Revisions to the Basel II Market Risk Framework*.
