# Stress Tests (Historical Replay, Hypothetical, Reverse)

> Deterministic scenario-based loss assessments that complement statistical risk measures by quantifying portfolio P&L under specific, large, named shocks. Layer 6 — Risk & Capital. Primary use: firmwide capital adequacy under CCAR/EBA, trader risk limits, prime-broker margin negotiation, and identification of hidden tail exposures.

## Mathematical formulation

Let the portfolio value be a function of $K$ risk factors $\mathbf{f} = (f_1, \ldots, f_K)$:

$$
V \;=\; V(\mathbf{f}).
$$

A **stress scenario** is a prescribed factor change $\Delta \mathbf{f}^{(s)} = (\Delta f_1^{(s)}, \ldots, \Delta f_K^{(s)})$. The scenario P&L is

$$
\text{PnL}^{(s)} \;=\; V(\mathbf{f} + \Delta\mathbf{f}^{(s)}) - V(\mathbf{f}).
$$

For linear (delta-only) approximation, with sensitivities $\Delta_k = \partial V / \partial f_k$,

$$
\text{PnL}^{(s)} \;\approx\; \sum_{k=1}^K \Delta_k\, \Delta f_k^{(s)}.
$$

For non-linear exposures (options books) include gamma and vega:

$$
\text{PnL}^{(s)} \;\approx\; \sum_k \Delta_k\, \Delta f_k^{(s)} + \tfrac{1}{2} \sum_{j,k} \Gamma_{jk}\, \Delta f_j^{(s)} \Delta f_k^{(s)} + \sum_k \mathcal{V}_k \Delta \sigma_k^{(s)} + \cdots
$$

For maximum accuracy in production, full revaluation $V(\mathbf{f} + \Delta\mathbf{f}^{(s)})$ replaces the Taylor expansion. The three stress flavors differ in how $\Delta\mathbf{f}^{(s)}$ is constructed.

### Historical Replay

Pick a historical crisis window $[t_1, t_2]$ and define

$$
\Delta\mathbf{f}^{(s)} \;=\; \mathbf{f}(t_2) - \mathbf{f}(t_1).
$$

For percentage changes (the standard for equity factors), $\Delta f_k^{(s)} = f_k(t_2)/f_k(t_1) - 1$ and the shocked level is $f_k \cdot (1 + \Delta f_k^{(s)})$. Canonical windows:

| Crisis | Window | Headline shock |
|---|---|---|
| Black Monday 1987 | 1987-10-19 (single day) | S&P 500: $-22.6\%$ |
| LTCM 1998 | 1998-08-17 to 1998-10-08 | S&P: $-19\%$, credit spreads $+200$bp |
| Lehman 2008 | 2008-09-12 to 2008-10-10 | S&P: $-28\%$, VIX 25 to 70 |
| COVID Mar 2020 | 2020-02-19 to 2020-03-23 | S&P: $-34\%$ in 23 trading days |
| Volmageddon | 2018-02-02 to 2018-02-09 | VIX $17 \to 50$ in 4 days; XIV blow-up |
| GameStop unwind | 2021-01-25 to 2021-02-05 | meme-name multi-sigma intraday squeezes |
| SEBI India ban | 2025-07-04 | India F&O liquidity event (Jane Street ref) |

### Hypothetical Scenarios

Hand-specified factor moves, not tied to a historical episode. Standard board-level scenarios:

| Scenario | Shock |
|---|---|
| Rate parallel up | $+100$ bp across the yield curve |
| Equity crash | $-20\%$ on all equity factors |
| Credit widening | IG $+200$ bp, HY $+500$ bp |
| Vol spike | All implied-vol surfaces $+5$ vol points |
| USD strength | $+10\%$ on USD trade-weighted index |
| Combined "1-in-100" | Equity $-25\%$, vol $+10$ pts, IG $+150$ bp, HY $+400$ bp, USD $+8\%$ |

These map onto the regulatory CCAR "severely adverse" scenario, which the Federal Reserve publishes annually for each bank holding company $\geq \$100\text{B}$ in assets.

### Reverse Stress Tests

Specify a catastrophic loss target $L^*$ (e.g., 30% of regulatory capital) and search for the **smallest** factor shock that produces $L^*$. Formally,

$$
\min_{\Delta\mathbf{f}}\; \lVert \Delta\mathbf{f} \rVert_{\boldsymbol{\Sigma}^{-1}}^2 \quad \text{s.t.}\quad V(\mathbf{f}) - V(\mathbf{f} + \Delta\mathbf{f}) \geq L^*,
$$

where $\lVert\cdot\rVert_{\boldsymbol{\Sigma}^{-1}}$ is the Mahalanobis distance with respect to the factor covariance $\boldsymbol{\Sigma}$. The solution $\Delta\mathbf{f}^*$ is the **most-likely path to disaster**. Lagrangian conditions give

$$
\Delta\mathbf{f}^* \;=\; \lambda\, \boldsymbol{\Sigma} \nabla_{\mathbf{f}} \text{PnL}(\mathbf{f}^*)
$$

i.e., the shock direction aligns with the gradient of P&L scaled by the covariance — the multivariate analogue of "move along the steepest descent in standardized units."

## Intuition and what this model addresses

VaR and ES describe **average tail behavior** under stationary assumptions. They fail in two ways: (i) they are conditioned on the historical sample, which by construction contains insufficient extreme observations; (ii) they conceal *which factors* drive the tail.

Stress tests answer different questions:

- **Historical replay**: *Are we exposed to the same vulnerability as in 2008/2020?* This is the only credible answer when management asks "what if it happens again?"
- **Hypothetical**: *Are we exposed to a plausible event that has not yet occurred?* (e.g., a synchronized US/China credit shock).
- **Reverse**: *What set of shocks could destroy us, and is it plausible?* This catches concentration risks that no statistical measure surfaces — e.g., a book that looks low-VaR because its volatility positions hedge each other, but loses catastrophically under a specific cross-asset move.

Stress tests are also the **only** risk number traders take seriously when communicating with prime brokers. A PB margin call says "if your book reproduces March 2020, you'll owe us $X$"; arguing that 99% VaR is small is futile.

## Computation approach

Three computational regimes:

1. **Delta-only**: pre-compute factor sensitivities $\Delta_k$, dot with scenario shocks. Suitable for linear equity books, fast (microseconds), used for real-time pre-trade checks.
2. **Delta-gamma-vega**: include second-order terms for options books. Suitable for vanilla options portfolios.
3. **Full revaluation**: re-price every instrument under shocked factor levels. Required for exotic options, structured products, and regulatory CCAR submissions. Computationally expensive — typically batched overnight.

For an equity-only book under yfinance data, the natural risk factors are individual stock returns (or factor returns: market, size, value, momentum, sector dummies). Historical replay reduces to: pull the return panel for the crisis window, dot with current dollar positions.

## Algorithm outline

```
ALGORITHM: Run_Stress_Tests(positions, scenarios)
INPUT:
  positions: dict {ticker -> dollar_position}
  scenarios: list of scenario objects
OUTPUT:
  scenario_pnls, worst_case, decomposition

--------------------------------------------------
ROUTINE A — HISTORICAL REPLAY
--------------------------------------------------
crisis_windows = {
  "Black Monday 1987":  ("1987-10-15", "1987-10-20"),
  "LTCM 1998":           ("1998-08-17", "1998-10-08"),
  "Lehman 2008":         ("2008-09-12", "2008-10-10"),
  "Eurozone 2011":       ("2011-08-01", "2011-08-19"),
  "China devaluation":   ("2015-08-17", "2015-08-26"),
  "Volmageddon 2018":    ("2018-02-02", "2018-02-09"),
  "Q4 2018 selloff":     ("2018-10-03", "2018-12-24"),
  "COVID crash":         ("2020-02-19", "2020-03-23"),
  "GameStop unwind":     ("2021-01-25", "2021-02-05"),
  "2022 bond rout":      ("2022-01-03", "2022-10-14"),
}

FOR each (name, (t1, t2)) in crisis_windows:
    prices <- yf.download(positions.keys(), start=t1, end=t2,
                          auto_adjust=True)["Close"]
    # Some tickers did not exist in older windows; drop missing
    available <- prices.columns
    factor_change <- prices.iloc[-1] / prices.iloc[0] - 1
    # For names that did not exist, fall back to a sector ETF proxy
    FOR each missing ticker t:
        proxy <- sector_map[t]   # e.g. AAPL -> XLK
        proxy_change <- yf.download(proxy, start=t1, end=t2).pct_change_total()
        factor_change[t] <- proxy_change

    scenario_pnl <- sum_over_tickers( positions[t] * factor_change[t] )
    results[name] <- scenario_pnl

--------------------------------------------------
ROUTINE B — HYPOTHETICAL
--------------------------------------------------
hypotheticals = [
  ("Equity -20%",       {"all_equity_factor": -0.20}),
  ("Equity -30%",       {"all_equity_factor": -0.30}),
  ("Equity -10%, Vol+5",{"all_equity_factor": -0.10, "vol": +0.05}),
  ("Rates +100bp",      {"yield_curve": +0.01}),
  ("USD +10%",          {"dxy": +0.10}),
]

# For an equity book, "all equity factor" can be applied directly to all
# equity exposures, or refined via a beta map:
betas <- estimate_beta_to_SPY(positions.keys(), lookback=252)
FOR each (name, shocks) in hypotheticals:
    pnl <- 0
    IF "all_equity_factor" in shocks:
        FOR ticker t:
            pnl += positions[t] * betas[t] * shocks["all_equity_factor"]
    # add other factor contributions for rates/USD/vol
    results[name] <- pnl

--------------------------------------------------
ROUTINE C — REVERSE STRESS
--------------------------------------------------
L_star <- 0.30 * capital
# Estimate factor cov from rolling window
Sigma <- yf returns of factors over last 5y
# Linearized P&L gradient w.r.t. factors
g <- factor_sensitivities(positions)
# Closed-form solution for min-Mahalanobis shock
#   minimise dF^T Sigma^{-1} dF s.t. g^T dF = L_star
lambda <- L_star / (g^T Sigma g)
dF_star <- lambda * Sigma @ g
# Interpret: how many standard deviations in each factor?
std_devs <- dF_star / sqrt(diag(Sigma))
report(dF_star, std_devs, plausibility=norm(std_devs))
```

### Missing-ticker handling for old crises

A common practical issue: Black Monday 1987 has yfinance data for very few names. For each missing ticker, map to a sector ETF proxy (e.g., `AAPL -> XLK`, `JPM -> XLF`, `XOM -> XLE`). For pre-ETF crises, map to a benchmark index (`^GSPC` for broad-market equity exposure) scaled by the historical beta of the ticker.

## yfinance data requirements

- **Daily adjusted close** for all portfolio tickers and proxy ETFs / indices.
- **Long histories** — at least back to 1987 for Black Monday replay. yfinance generally supplies `^GSPC` and `^DJI` to 1928, and individual large-caps to their IPO date.
- **Sector ETF proxies**: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB, XLRE, XLC plus broad-market `SPY`, `QQQ`, `IWM`.
- **Macro factors** for hypothetical: `^TNX` (10Y yield), `DX-Y.NYB` (DXY), `^VIX`, `HYG`, `LQD`.
- **Pre-trade caching**: scenario shock vectors should be persisted; the heavy lifting is data pull and beta estimation, not the dot product.

## Refit / refresh frequency

- **Daily**: scenario P&L on current positions; reuses cached shock vectors.
- **Weekly**: refresh beta and sector-proxy mappings using a rolling 1-year window.
- **Quarterly**: review the historical-replay scenario set; add new crises as they occur (e.g., 2022 bond rout added in early 2023).
- **Annual**: regulatory submission of CCAR scenarios; recalibrate hypothetical scenarios to current macro environment.
- **Ad hoc**: reverse stress tests run on demand during board risk reviews or in response to specific exposure questions.

## Validation and diagnostics

Stress tests are not statistical estimators, so traditional p-values do not apply. Diagnostic practices:

- **Scenario coverage matrix**: each major risk factor should appear with a material shock in at least one scenario. Empty cells reveal gaps.
- **P&L decomposition**: scenario loss decomposed into factor contributions; the top-3 contributors should match intuition. If they don't, sensitivities are likely wrong.
- **Re-run on past data**: for a date 6 months ago, run today's stress engine on that date's positions; compare the historical-replay scenario for the most recent crisis to the *actual* P&L of that window. Engineering discrepancy reveals revaluation errors or missing factor coverage.
- **Sensitivity to crisis-window endpoints**: shift $t_1, t_2$ by $\pm 2$ days; the scenario loss should be roughly stable. Large jumps reveal regime-dependent dynamics that warrant a finer scenario.
- **Reverse-stress plausibility**: any reverse-stress shock with Mahalanobis distance $< 4\sigma$ in $\boldsymbol{\Sigma}$-units is "plausible" and demands risk-committee attention.

## Connections to other models

- **Model 17 (VaR)** — VaR is the average tail; stress tests are named tails. Used together.
- **Model 18 (Expected Shortfall)** — FRTB capital charge has a separate "Stressed ES" component computed on a historical stress window; this is a hybrid of stress test and ES.
- **Model 20 (Liquidity-Adjusted VaR)** — stress scenarios in a market-maker context must also include the unwind cost over the stress; combines stress shocks with Almgren-Chriss liquidation.
- **Layer 3 (factor models)** — provide the factor sensitivities $\Delta_k$ used in linearized stress.
- **Layer 5 (option pricing)** — provides full-revaluation P&L for options books.
- **Layer 4 (volatility models)** — supply the shocked-vol surfaces for vega-bearing books.

## Limitations and failure modes

- **Selection bias of historical scenarios**: replaying past crises assumes the next crisis resembles a past crisis. The 2020 COVID shock did not resemble any prior scenario in its policy-response speed.
- **Linearization error**: delta-only stress understates losses on convex (short-gamma) books and overstates on long-gamma. Required full revaluation is expensive.
- **Correlation breakdown**: in stress, correlations move to $\pm 1$. A scenario constructed from a historical pre-stress correlation matrix mis-states diversification.
- **Sensitivity stability**: factor sensitivities themselves change in stress (e.g., a stock's beta to SPY rises in crashes). Static sensitivities embed an optimistic dynamic.
- **Reverse-stress combinatorics**: the search space is enormous; the Mahalanobis-distance objective is a heuristic and can miss low-probability-but-feasible vulnerabilities.
- **Scenario fatigue**: institutions accumulate dozens of legacy scenarios that obscure rather than illuminate risk. Periodic pruning is required.
- **Survivorship of crisis tickers**: many names alive in 2008 are now delisted (Lehman, Bear Stearns); analogously many today will be gone in 15 years. Reverse the survivorship by populating delisted names from a historical universe database.

## References

- Federal Reserve Board (annual). *Comprehensive Capital Analysis and Review (CCAR) Supervisory Scenarios.*
- European Banking Authority (biennial). *EU-Wide Stress Test Methodology.*
- BCBS (2018). *Stress Testing Principles.* BIS publication d450.
- Berkowitz, J. (2000). "A Coherent Framework for Stress Testing." *Journal of Risk* 2(2).
- Quagliariello, M. (ed.) (2009). *Stress-Testing the Banking System: Methodologies and Applications.* Cambridge.
- Glasserman, P., Kang, C., Kang, W. (2015). "Stress Scenario Selection by Empirical Likelihood." *Quantitative Finance* 15.
- Breuer, T., Krenn, G. (1999). "Identifying Stress Test Scenarios." *Risk Magazine*. [Reverse stress.]
