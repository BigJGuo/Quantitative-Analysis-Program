# Expected Shortfall (CVaR) under FRTB

> A coherent tail-conditional risk measure equal to the expected loss given that loss exceeds VaR, replacing VaR as the regulatory metric in Basel III's Fundamental Review of the Trading Book. Layer 6 — Risk & Capital. Primary use: regulatory market-risk capital under the FRTB Internal Models Approach (IMA), tail-risk reporting, and internal capital allocation.

## Mathematical formulation

For a loss random variable $\mathcal{L}$ with continuous distribution, Expected Shortfall at level $\alpha \in (0,1)$ is

$$
\text{ES}_\alpha \;=\; \mathbb{E}[\mathcal{L} \mid \mathcal{L} > \text{VaR}_\alpha] \;=\; \frac{1}{1 - \alpha} \int_\alpha^1 \text{VaR}_u(\mathcal{L})\, du.
$$

The second form (the average of $\text{VaR}_u$ over the tail quantile range $[\alpha, 1]$) is the **definition that survives under discrete distributions**, where conditional expectation is ambiguous when $P(\mathcal{L} = \text{VaR}_\alpha) > 0$. Equivalently, with $q_\alpha = \text{VaR}_\alpha$,

$$
\text{ES}_\alpha \;=\; \frac{1}{1-\alpha} \left( \mathbb{E}[\mathcal{L} \mathbf{1}\{\mathcal{L} > q_\alpha\}] + q_\alpha\, (P(\mathcal{L} \leq q_\alpha) - \alpha) \right).
$$

### Coherence axioms (Artzner et al. 1999)

A risk measure $\rho$ is **coherent** if it satisfies:

1. **Monotonicity**: $\mathcal{L}_1 \leq \mathcal{L}_2$ a.s. $\implies \rho(\mathcal{L}_1) \leq \rho(\mathcal{L}_2)$ — strictly worse losses imply strictly worse risk.
2. **Sub-additivity**: $\rho(\mathcal{L}_1 + \mathcal{L}_2) \leq \rho(\mathcal{L}_1) + \rho(\mathcal{L}_2)$ — diversification cannot increase risk.
3. **Positive homogeneity**: $\rho(c\mathcal{L}) = c\,\rho(\mathcal{L})$ for $c > 0$ — scaling exposure scales risk.
4. **Translation invariance**: $\rho(\mathcal{L} + c) = \rho(\mathcal{L}) + c$ for deterministic cash $c$ — adding certain loss adds the same to risk.

VaR fails sub-additivity in general (one can construct two-asset examples where $\text{VaR}(L_1+L_2) > \text{VaR}(L_1) + \text{VaR}(L_2)$). ES satisfies all four axioms; this is the theoretical lever Basel used to replace VaR.

### Parametric ES (Gaussian)

With $\mathcal{L} \sim \mathcal{N}(-\mu, \sigma^2)$ and $z_\alpha = \Phi^{-1}(\alpha)$,

$$
\boxed{\;\text{ES}_\alpha \;=\; -\mu \;+\; \sigma \cdot \frac{\phi(z_\alpha)}{1 - \alpha}\;}
$$

where $\phi$ is the standard normal pdf. The factor $\phi(z_\alpha)/(1-\alpha)$ is the *Mills-ratio-like* tail-conditional moment. **Derivation**: for $X \sim \mathcal{N}(0,1)$,

$$
\mathbb{E}[X \mid X > z] = \frac{\int_z^\infty x\, \phi(x)\, dx}{1 - \Phi(z)} = \frac{\phi(z)}{1 - \Phi(z)}
$$

using $\int_z^\infty x\, \phi(x)\, dx = \phi(z)$ since $\phi'(x) = -x\phi(x)$.

### Parametric ES (Student-$t_\nu$)

For losses scaled to $t_\nu$ with $\nu > 1$,

$$
\text{ES}_\alpha \;=\; -\mu + \sigma \cdot \frac{f_\nu(t_\alpha)}{1-\alpha} \cdot \frac{\nu + t_\alpha^2}{\nu - 1}
$$

where $t_\alpha = T_\nu^{-1}(\alpha)$ and $f_\nu$ is the $t_\nu$ density. This is the form used when a heavy-tailed marginal is calibrated, e.g., $\nu \in [4, 7]$ for daily equity returns.

### Historical-simulation ES

Given empirical losses $\{L_1, \ldots, L_N\}$ and HS VaR $\widehat q_\alpha$,

$$
\widehat{\text{ES}}_\alpha \;=\; \frac{1}{|\{i : L_i > \widehat q_\alpha\}|}\sum_{i\,:\,L_i > \widehat q_\alpha} L_i.
$$

With $N = 250$ and $\alpha = 0.975$, this is the mean of the worst $\approx 6$ days.

### Monte Carlo ES

Identical to HS but applied to $K$ simulated path losses $\{L^{(1)}, \ldots, L^{(K)}\}$.

## Intuition and what this model addresses

VaR tells you the *boundary* of the bad region. ES tells you the *expected depth* once you're in it.

Three problems ES addresses:

1. **Tail severity**. Two books with identical 99% VaR can have radically different expected losses in the worst 1% of days. ES separates them.
2. **Diversification consistency**. Sub-additivity means a portfolio manager who combines two books cannot game the risk metric by splitting; with VaR this gaming exists.
3. **Fat-tail sensitivity**. At $\alpha = 0.975$, ES under a Gaussian is approximately equal to 99% VaR; but under realistic equity tails (Student-$t_4$, jump-diffusion), ES grows sharply faster than VaR. Basel chose $\alpha = 0.975$ ES specifically so that *under normal conditions the capital number resembles legacy VaR, but under fat tails it correctly punishes severity*.

### FRTB context

The Basel III Fundamental Review of the Trading Book (FRTB), finalized in BCBS d457 (2019) and revised d518 (2023), replaces 99% one-day VaR with **97.5% Expected Shortfall** computed over **liquidity-horizon-adjusted shocks** (see Model 20) and stressed over a historical period containing material financial stress for the desk's risk factors. The IMA capital charge is

$$
\text{IMCC} \;=\; \rho\, \text{IMCC}(C) + (1-\rho)\sum_i \text{IMCC}(C_i)
$$

where $\text{IMCC}(C)$ is portfolio-level ES and $\text{IMCC}(C_i)$ are non-diversifiable bucket ES values, with $\rho = 0.5$.

## Computation approach

Three implementations parallel the VaR analogues:

- **Parametric (Gaussian or $t$)**: closed-form once $\mu, \sigma$ (and $\nu$) are estimated.
- **Historical simulation**: empirical tail-mean over the rolling window.
- **Monte Carlo**: distributional simulation, then tail-mean.

For equity portfolios, historical-sim ES is the cleanest because no parametric assumption is needed and the calculation is auditable, but it requires sufficient tail observations: at $\alpha = 0.975$ with $N = 250$, only $\approx 6$ observations populate the tail mean — so $N \geq 500$ is recommended.

## Algorithm outline

```
ALGORITHM: Compute_ES(positions, lookback_N, alpha, method)
INPUT:
  positions: dict {ticker -> dollar_notional}
  lookback_N: integer (>= 500 recommended)
  alpha: float (typically 0.975 for FRTB)
  method: "parametric_normal" | "parametric_t" | "historical" | "monte_carlo"
OUTPUT:
  VaR_alpha, ES_alpha, tail_losses

STEP 1 — Pull and align data
  tickers <- positions.keys()
  prices <- yf.download(tickers, period="3y", auto_adjust=True)["Close"]
  rets <- prices.pct_change().dropna().tail(lookback_N)
  V0 <- sum(positions.values())
  w_dollar <- vector of {ticker -> dollar_position}

STEP 2 — Branch by method

  IF method == "parametric_normal":
    Sigma <- rets.cov().values
    mu_p <- 0                                     # daily drift negligible
    sigma_p <- sqrt(w_dollar^T Sigma w_dollar)
    z <- scipy.stats.norm.ppf(alpha)
    pdf_z <- scipy.stats.norm.pdf(z)
    VaR <- sigma_p * z
    ES  <- sigma_p * pdf_z / (1 - alpha)

  ELIF method == "parametric_t":
    # Fit Student-t to portfolio P&L series
    port_pnl <- rets @ w_dollar
    nu, loc, scale <- scipy.stats.t.fit(-port_pnl)
    t_a <- scipy.stats.t.ppf(alpha, df=nu)
    f_a <- scipy.stats.t.pdf(t_a, df=nu)
    VaR <- loc + scale * t_a
    ES  <- loc + scale * (f_a / (1 - alpha)) * (nu + t_a^2)/(nu - 1)

  ELIF method == "historical":
    pnl_vec <- rets.values @ w_dollar
    loss_vec <- -pnl_vec
    VaR <- numpy.quantile(loss_vec, alpha)
    tail <- loss_vec[loss_vec > VaR]
    ES <- tail.mean()

  ELIF method == "monte_carlo":
    mu_hat <- rets.mean().values
    Sigma_hat <- rets.cov().values
    L <- numpy.linalg.cholesky(Sigma_hat)
    K <- 100_000
    # Heavy-tailed: sample multivariate-t via Gaussian / chi-squared mixture
    nu <- 5
    g <- scipy.stats.chi2.rvs(nu, size=K) / nu
    Z <- numpy.random.standard_normal((K, n_assets))
    sim_rets <- mu_hat + (Z @ L.T) / sqrt(g)[:,None]
    pnl_sim <- sim_rets @ w_dollar
    loss <- -pnl_sim
    VaR <- numpy.quantile(loss, alpha)
    ES  <- loss[loss > VaR].mean()

STEP 3 — Liquidity scaling (FRTB)
  # If FRTB IMA, scale each risk factor shock by sqrt(h_k / 10)
  # before computing ES.  See Model 20.

STEP 4 — Return
  return {VaR, ES, tail_losses}
```

## yfinance data requirements

- Daily adjusted close for all portfolio tickers, lookback of **3–5 years** to ensure adequate tail observations at $\alpha = 0.975$.
- For FRTB compliance the *stressed period* must include a one-year window of material stress; combining a 2008 GFC window with a 2020 COVID window from yfinance histories satisfies this for US equities.
- Currency-converted prices for non-USD names; yfinance returns native-currency prices.
- For ETF positions, use the ETF ticker directly — do not look through to constituents at this layer.

## Refit / refresh frequency

- **Daily** ES recomputation with rolling window update.
- **Quarterly** model recalibration: $\nu$ for $t$-marginals, GARCH parameters, copula correlation regime.
- **Annual** P&L Attribution (PLA) test for FRTB IMA model eligibility (see below).
- **Whenever a stressed period of greater severity emerges**, the FRTB stressed-ES reference window must be updated.

## Validation and diagnostics

ES is **harder to backtest than VaR** because conditional expectations are unobservable from a single tail draw. Three active approaches:

### Acerbi-Szekely Z1 test

For each breach day $t$ with $I_t = 1$, define

$$
Z_1 \;=\; \frac{1}{n_T} \sum_{t : I_t = 1} \frac{L_t}{\text{ES}_\alpha^{(t)}} - 1
$$

where $n_T = \sum I_t$. Under the null $\mathbb{E}[Z_1] = 0$. Under-prediction of ES yields $Z_1 > 0$. Use Monte Carlo to get critical values — there is no closed-form null distribution.

### Acerbi-Szekely Z2 test

$$
Z_2 \;=\; \sum_t \frac{L_t \cdot I_t}{T (1-\alpha)\, \text{ES}_\alpha^{(t)}} - 1
$$

This integrates over the full sample (not just breach days), giving more power but requiring fewer assumptions.

### FRTB-specific tests

- **Mean of tail violations** vs. predicted ES, with traffic-light bands analogous to VaR's Basel scheme.
- **P&L Attribution (PLA) test**: compares the front-office "hypothetical P&L" against the risk-engine "risk-theoretical P&L". Failure of PLA forces a desk into the standardized approach with materially higher capital.

### Diagnostic plots

- ES vs. realized tail-mean loss over rolling 250-day windows.
- Empirical tail histogram overlaid with parametric tail density.
- Time-series of ES / VaR ratio; spikes flag regime shifts to heavier tails.

## Connections to other models

- **Model 17 (VaR)** — direct predecessor and component (ES is computed from the same simulated/empirical P&L vector).
- **Model 19 (Stress Tests)** — FRTB capital combines stressed-ES with stress scenarios; the two are complementary regulatory inputs.
- **Model 20 (Liquidity-Adjusted VaR)** — FRTB ES uses liquidity-horizon-scaled risk factors; one cannot be computed without the other.
- **Layer 4 (GARCH, jump-diffusion)** — supply the conditional distributions used in parametric and Monte Carlo ES.
- **Layer 3 (factor models, copulas)** — define the joint loss distribution.

## Limitations and failure modes

- **Backtesting weakness**: with only $\approx 6$ tail observations per 250 days at $\alpha = 0.975$, statistical power to detect mis-specified ES is limited. Acerbi-Szekely tests typically need 500+ trading days.
- **Sample-mean instability**: empirical ES is the mean of a small number of order statistics and is highly sensitive to a single extreme observation. Bootstrap confidence intervals are wide.
- **Liquidity blindness**: standard ES assumes mark-to-market exits at posted prices; in a tail event the actual exit price is materially worse (see Model 20).
- **Model risk**: parametric ES is more sensitive than parametric VaR to tail-index assumptions; misspecified $\nu$ produces large errors in ES even when VaR is approximately right.
- **Elicitability**: ES is **not elicitable** as a single statistic (Gneiting 2011) — there is no scoring function that is uniquely minimized in expectation by the true ES. This means ranking competing ES models is mathematically subtle and is the chief technical reason backtesting is hard. The "jointly elicitable with VaR" result of Fissler-Ziegel (2016) gives a workable but two-dimensional scoring rule.
- **Procyclicality** persists from VaR: ES contracts when volatility is low, expands when volatility spikes, contributing to the same forced-deleveraging dynamic.

## References

- Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). "Coherent Measures of Risk." *Mathematical Finance* 9(3), 203–228.
- Acerbi, C., Tasche, D. (2002). "On the Coherence of Expected Shortfall." *Journal of Banking & Finance* 26.
- Acerbi, C., Szekely, B. (2014). "Back-Testing Expected Shortfall." *Risk Magazine*, December 2014.
- Gneiting, T. (2011). "Making and Evaluating Point Forecasts." *Journal of the American Statistical Association* 106.
- Fissler, T., Ziegel, J. (2016). "Higher Order Elicitability and Osband's Principle." *Annals of Statistics* 44.
- BCBS d457 (2019). *Minimum Capital Requirements for Market Risk*. Basel Committee.
- BCBS d518 (2023). *FRTB: Revised Standards*. Basel Committee.
- McNeil, A., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*, 2nd ed., Princeton.
