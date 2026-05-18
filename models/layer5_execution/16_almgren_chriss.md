# Almgren-Chriss Optimal Execution

> Closed-form mean-variance optimal liquidation schedule for a parent order subject to linear permanent and temporary market impact. Layer 5 — Execution. Primary use: parent-order scheduling, ETF basket liquidation, and large-position unwinds.

## Mathematical formulation

### Setup and notation

A trader holds $X$ shares (long) and must liquidate to zero by horizon $T$. Let $x_t$ denote the remaining inventory and $v_t = -\dot x_t \ge 0$ the (continuous) trading rate. Boundary conditions:

$$x_0 = X, \qquad x_T = 0$$

Equivalently in discrete time with $N$ slices of length $\tau = T/N$, let $x_k = x_{k\tau}$ and $n_k = x_{k-1} - x_k$ be the shares traded in slice $k$, with $\sum_{k=1}^N n_k = X$.

### Price dynamics

Let $S_t$ be the unaffected mid-price evolving as arithmetic Brownian motion:

$$dS_t = \sigma \, dW_t$$

(Geometric BM is the equally common alternative; for short horizons and small $\sigma T$ the difference is negligible.) The **effective mid** $\tilde S_t$ includes permanent impact:

$$\tilde S_t = S_t - \gamma \cdot (X - x_t) = S_t - \gamma \int_0^t v_s \, ds$$

i.e. each share sold pushes the mid down by $\gamma$ permanently. The **execution price** $\hat S_t$ adds temporary (slippage) impact paid only on the current child:

$$\hat S_t = \tilde S_t - \eta v_t$$

In discrete form, slice $k$ executes at

$$\hat S_k = S_{k\tau} - \gamma \sum_{j \le k} n_j - \eta \frac{n_k}{\tau}$$

### Implementation shortfall

Cost is measured as the implementation shortfall relative to the arrival price $S_0$:

$$C = X S_0 - \sum_{k=1}^{N} n_k \hat S_k$$

Substituting and taking expectations under $\mathbb{E}[W_t] = 0$:

$$\mathbb{E}[C] = \gamma \sum_{k=1}^N n_k \sum_{j \le k} n_j - \sum_k n_k (S_{k\tau} - S_0)_{\mathbb{E}=0} + \eta \sum_k \frac{n_k^2}{\tau}$$

After simplification with $\sum_k n_k = X$ and $\sum_k n_k \sum_{j \le k} n_j = X^2/2 + \sum_k n_k^2/2$, the permanent-impact term collapses to a constant $\gamma X^2 / 2$ plus a small correction absorbed into $\eta$. In continuous time:

$$\mathbb{E}[C] = \frac{\gamma}{2} X^2 + \eta \int_0^T v_t^2 \, dt$$

The variance comes from price uncertainty on the residual inventory:

$$\mathbb{V}[C] = \sigma^2 \int_0^T x_t^2 \, dt$$

### Mean-variance objective

The trader minimizes

$$\mathcal{U}[x] = \mathbb{E}[C] + \lambda \, \mathbb{V}[C] = \frac{\gamma}{2} X^2 + \int_0^T \left( \eta v_t^2 + \lambda \sigma^2 x_t^2 \right) dt$$

over admissible paths with $x_0 = X$, $x_T = 0$. The permanent-impact term $\gamma X^2/2$ is **path-independent** under the linear model and drops out of the optimization. Only the trade-off between temporary impact (penalizes fast trading via $\eta v_t^2$) and inventory risk (penalizes slow trading via $\lambda \sigma^2 x_t^2$) remains.

### Closed-form solution

The Euler-Lagrange equation of $L = \eta \dot x^2 + \lambda \sigma^2 x^2$ is

$$\eta \ddot x - \lambda \sigma^2 x = 0$$

with general solution $x_t = A \cosh(\kappa t) + B \sinh(\kappa t)$, where

$$\kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}$$

Applying $x_0 = X$ and $x_T = 0$:

$$\boxed{ \; x_t^* = X \cdot \frac{\sinh(\kappa (T - t))}{\sinh(\kappa T)} \; }$$

with trading rate

$$v_t^* = X \kappa \cdot \frac{\cosh(\kappa (T - t))}{\sinh(\kappa T)}$$

#### Limits

- $\lambda \to 0$: $\kappa \to 0$, $\sinh(\kappa s) \approx \kappa s$, so $x_t^* \to X(T - t)/T$ — **TWAP** (linear inventory).
- $\lambda \to \infty$: $\kappa \to \infty$, $x_t^* \to X \cdot e^{-\kappa t}$ for $t \ll T$ — **front-loaded** exponential liquidation.

#### Discrete-time form

For $N$ slices of length $\tau$, the discrete optimal inventory satisfies a difference equation

$$x_{k+1} - 2 \cosh(\tilde\kappa \tau) x_k + x_{k-1} = 0$$

with the same boundary conditions, yielding

$$x_k^* = X \cdot \frac{\sinh(\tilde\kappa (T - k\tau))}{\sinh(\tilde\kappa T)}$$

where $\tilde\kappa$ satisfies $2(\cosh(\tilde\kappa \tau) - 1) = \tau^2 \lambda \sigma^2 / \eta$. As $\tau \to 0$, $\tilde\kappa \to \kappa$.

Child-order sizes:

$$n_k = x_{k-1}^* - x_k^*$$

### Efficient frontier

Sweeping $\lambda$ traces out the **efficient frontier** of (expected cost, cost variance) pairs:

$$E(\lambda) = \mathbb{E}[C \mid x^*(\lambda)] = \frac{\gamma X^2}{2} + \eta X^2 \kappa \cdot \frac{\sinh(\kappa T) \cosh(\kappa T) - \kappa T}{2 \sinh^2(\kappa T)} \cdot \frac{1}{T}$$

$$V(\lambda) = \mathbb{V}[C \mid x^*(\lambda)] = \sigma^2 X^2 \cdot \frac{T \cosh(\kappa T) \sinh(\kappa T) - \kappa T^2}{2 \kappa \sinh^2(\kappa T)} \cdot \frac{1}{T}$$

As $\lambda$ varies from $0$ to $\infty$, $(V, E)$ moves from (high variance, low cost — TWAP-like) to (low variance, high cost — front-loaded).

## Intuition and what this model addresses

Almgren-Chriss formalizes the **trader's dilemma**: liquidating too fast costs slippage; liquidating too slowly exposes you to price moves. With quadratic temporary impact ($\eta v^2$ in dollar cost — i.e. linear in rate per share, $\eta v$ per share) and Gaussian price moves, the problem is a linear-quadratic optimal control problem with a closed-form solution.

The model is the workhorse for:

- **Parent-order schedules** in algorithmic execution: pick a horizon, set a risk aversion, get a slice schedule.
- **Block trade pricing**: integrate cost over the schedule for an upfront principal quote.
- **ETF basket liquidation**: apply per-leg with a shared $\lambda$, or solve a multi-asset version with the covariance matrix $\Sigma$ replacing $\sigma^2$.
- **L-VaR**: liquidity-adjusted Value-at-Risk uses the optimal liquidation cost as the deterministic add-on.

The key insight: the optimal path is parameterized by a **single dimensionless quantity** $\kappa T = T \sqrt{\lambda \sigma^2 / \eta}$. Large $\kappa T$ → front-load. Small $\kappa T$ → TWAP. The economics collapse to one number.

## Calibration approach

### Volatility $\sigma$

From daily yfinance close prices over a 30-day rolling window:

$$\hat\sigma_{\text{annual}} = \text{std}\left(\ln \frac{C_t}{C_{t-1}}\right) \cdot \sqrt{252}$$

For intraday execution at slice length $\tau$ in trading-day fractions:

$$\hat\sigma_{\tau} = \hat\sigma_{\text{annual}} \cdot \sqrt{\tau / 252}$$

Alternative: compute realized intraday volatility directly from 5-minute returns (Andersen-Bollerslev RV estimator) and scale.

### Temporary impact $\eta$

Without execution data, use one of three rules of thumb:

1. **Half-spread rule.** $\eta \approx \dfrac{1}{2} \cdot \dfrac{s}{V_{\text{daily}}}$, where $s$ is the typical half-spread in dollars. Units: dollars per (share per unit time) per share — i.e. if you trade $v$ shares per unit time, you pay $\eta v$ extra per share.
2. **Square-root law mapping.** From Model 15, $\eta \approx Y \sigma \sqrt{1/V_{\text{daily}}} \cdot P$ in appropriate units. The square-root law gives total cost; differentiate w.r.t. $Q$ to get marginal impact and identify the linearized coefficient near the typical $Q/V$.
3. **Almgren et al. (2005) calibration.** $\eta = \eta_0 \cdot \sigma \cdot V_{\text{daily}}^{-\beta}$ with $\eta_0 \approx 0.142$ and $\beta \approx 0.6$ for US equities. This is the de-facto industry default.

### Permanent impact $\gamma$

Permanent impact is typically a fraction of temporary impact. A common rule:

$$\gamma \approx \frac{\eta}{10 \cdot \tau_{\text{half-life}}}$$

For practical purposes the schedule is invariant to $\gamma$ (it drops from the optimization), so $\gamma$ matters only for the final cost estimate, not for the slice sizes.

### Risk aversion $\lambda$

Hand-tuned. Common ranges:

- $\lambda \sim 10^{-7}$ — large institutional (low risk aversion, long horizons).
- $\lambda \sim 10^{-6}$ — typical sell-side execution desk.
- $\lambda \sim 10^{-5}$ — risk-averse, short horizon.
- $\lambda \sim 10^{-4}$ and above — retail-scale and crisis regimes.

A principled approach: pick $\kappa T$ directly. $\kappa T = 1$ is the "balanced" point; $\kappa T = 3$ is "front-loaded"; $\kappa T = 0.3$ is "near-TWAP".

## Algorithm outline

### Algorithm A — Single-asset Almgren-Chriss schedule

```
INPUT:
  X       : shares to liquidate (positive for sell, negative for buy)
  T       : horizon in seconds (or trading-day fraction)
  N       : number of slices
  sigma   : volatility, scaled to units of T
  eta     : temporary impact coefficient ($ per share per share/sec)
  gamma   : permanent impact coefficient ($ per share per share)
  lam     : risk aversion (1/$)

OUTPUT:
  x[0..N] : optimal inventory path
  n[1..N] : child-order sizes
  E[C], V[C], frontier_point

1.  tau    = T / N
2.  kappa  = sqrt(lam * sigma^2 / eta)
3.  For k = 0..N:
        t      = k * tau
        x[k]   = X * sinh(kappa * (T - t)) / sinh(kappa * T)
4.  For k = 1..N:
        n[k]   = x[k-1] - x[k]
5.  E_cost   = 0.5*gamma*X^2
             + eta * sum_k (n[k]/tau)^2 * tau
6.  V_cost   = sigma^2 * sum_k x[k]^2 * tau
7.  Return x, n, E_cost, V_cost
```

### Algorithm B — Discrete optimal with explicit difference equation

```
1.  tau    = T / N
2.  Solve  2*(cosh(kappa_tilde * tau) - 1) = tau^2 * lam * sigma^2 / eta
         for kappa_tilde (Newton's method, initial guess kappa).
3.  x[k]   = X * sinh(kappa_tilde * (T - k*tau)) / sinh(kappa_tilde * T)
4.  Remainder as in Algorithm A.
```

The discrete form matters when $N$ is small (e.g., $N = 13$ for half-hour slices over a US session); for $N \ge 50$ the continuous form is within 0.5% of the discrete.

### Algorithm C — Volume-shaped Almgren-Chriss

Real markets have a U-shaped intraday volume profile. Pure A-C ignores this; **VWAP-aware A-C** modifies the cost to weight by the local volume fraction $u_t$:

$$\eta_t = \eta_0 / u_t$$

The Euler-Lagrange becomes $\eta_0 / u_t \cdot \ddot x_t + (\text{lower order}) - \lambda \sigma^2 x_t = 0$, solved numerically (no closed form). Practically:

```
1.  Build intraday volume profile u[k] from yfinance 5m bars over last 30 days.
2.  Set eta[k] = eta_0 / u[k].
3.  Discretize objective sum_k (eta[k] * (n[k]/tau)^2 * tau + lam * sigma^2 * x[k]^2 * tau).
4.  Solve quadratic program with linear constraints x[0]=X, x[N]=0, x[k] >= 0 (no over-shoot).
5.  Output n[k] = x[k-1] - x[k].
```

The QP is convex and small (N variables, 2 boundary constraints). Solvable with `cvxpy` or `scipy.optimize.minimize` in milliseconds.

### Algorithm D — Multi-asset basket liquidation

For a basket of $M$ assets with inventory vector $\mathbf{x}_t \in \mathbb{R}^M$, covariance matrix $\Sigma$, diagonal $\eta = \text{diag}(\eta_1, \dots, \eta_M)$:

$$\min_{\mathbf{v}} \int_0^T \left( \mathbf{v}_t^\top \eta \mathbf{v}_t + \lambda \mathbf{x}_t^\top \Sigma \mathbf{x}_t \right) dt$$

The solution decouples in the eigenbasis of $\eta^{-1/2} \Sigma \eta^{-1/2}$:

$$\mathbf{x}_t^* = \eta^{-1/2} U \cdot \text{diag}\!\left(\frac{\sinh(\kappa_i (T-t))}{\sinh(\kappa_i T)}\right) U^\top \eta^{1/2} \mathbf{X}$$

where $U \Lambda U^\top = \eta^{-1/2} \Sigma \eta^{-1/2}$ and $\kappa_i = \sqrt{\lambda \Lambda_{ii}}$. Each principal component liquidates on its own timescale.

## yfinance data requirements

| Field | yfinance call | Used for |
|---|---|---|
| Daily close, $30$-day window | `Ticker.history(period="30d")` | $\sigma_{\text{daily}}$ |
| Daily volume, $20$-day window | same | $V_{\text{daily}}$ for $\eta$ calibration |
| Intraday 5m bars, $30$-day window | `Ticker.history(period="30d", interval="5m")` | Intraday volume profile $u_k$ |
| Intraday 1m bars, $7$-day window | `Ticker.history(period="7d", interval="1m")` | Fine-grained slice sizing on the trading day |
| Last price | `Ticker.fast_info["last_price"]` | Arrival reference; dollar cost conversion |

**Workarounds for missing data:**

1. **No spread series.** Estimate $s$ from daily Corwin-Schultz or assume 1 bp for US large-caps. Feed into the half-spread rule for $\eta$.
2. **No real-time book depth.** A-C does not require book depth at runtime — only calibrated $\eta$. The model is therefore implementable on yfinance alone.
3. **Volume-profile staleness.** The intraday $u_k$ from 30 days of 5m bars is stable except on FOMC/CPI days; flag those days and revert to TWAP shape.
4. **Holiday and early-close days.** Truncate $T$ to actual session length; the schedule remains optimal under the same equations.

## Calibration / refit frequency

- **$\sigma$**: refit daily (close-of-day) using 30-day realized vol. For intraday horizons ($T < 1$ day), use a 5-minute RV estimator over the last 10 sessions.
- **$\eta$**: refit weekly per ticker using the chosen rule. If a panel of trades is available, refit monthly with a regression.
- **$\gamma$**: refit quarterly; permanent-impact estimates are noisy and benefit from long windows.
- **$\lambda$**: review monthly. Tied to firm-level risk budget and trader preference, not to data.
- **Intraday volume profile $u_k$**: refit weekly. Use weekday-conditional means (Monday vs. Friday differ) and exclude FOMC/earnings days from the calibration set.

## Validation and diagnostics

1. **Schedule sanity.** Plot $x_t^*$ and $n_k$. Confirm $x_t$ is monotone decreasing, $n_k > 0$, $\sum n_k = X$.
2. **$\kappa T$ in reasonable range.** $\kappa T \in [0.3, 3]$ for most production cases. Outside this range, either revert to TWAP ($\kappa T < 0.1$) or do a block trade ($\kappa T > 5$).
3. **Realized-vs-predicted cost.** On historical executions, compute realized implementation shortfall and compare to $\mathbb{E}[C]$. Ratio should be 0.8-1.3; systematic over/underpredict signals miscalibrated $\eta$.
4. **Efficient frontier check.** Sweep $\lambda$ over 4-6 decades, plot $(V, E)$. Curve should be convex and monotone (E decreasing, V increasing). Concavity is a bug.
5. **Stress test under regime shift.** Multiply $\sigma$ by 3 and re-run. Schedule should accelerate (more front-loading). If it does not, check $\kappa$ computation.
6. **Comparison to TWAP and VWAP.** Backtest A-C, TWAP, and VWAP over the same horizon on historical data. A-C should beat TWAP on cost-variance frontier; if not, calibration is wrong.
7. **Numerical stability.** For $\kappa T > 30$, $\sinh(\kappa T)$ overflows in double precision. Switch to the asymptotic form $x_t^* \approx X e^{-\kappa t}$ for large $\kappa T$.

## Connections to other models

- **Layer 5 — Kyle Lambda and Square-Root Impact Law (Model 15):** supplies the temporary-impact coefficient $\eta$ either directly (Kyle's $\lambda$) or via differentiating the square-root law.
- **Layer 5 — Propagator execution models:** generalize A-C by replacing the instantaneous-temporary-impact assumption with a memory kernel $G$. Required when long-memory of order flow matters (long horizons, large $Q/V$).
- **Layer 5 — Implementation Shortfall and Arrival Price algos:** A-C is the analytical core; production algos add limit-order placement on top.
- **Layer 4 — GARCH / HAR-RV volatility forecasts:** supply $\sigma$ for the horizon-matched volatility input.
- **Layer 6 — L-VaR / liquidation risk:** uses A-C's $\mathbb{E}[C]$ and $\mathbb{V}[C]$ to extend standard VaR with liquidation cost.
- **Layer 7 — Portfolio construction:** the basket version (Algorithm D) is the link between portfolio rebalancing and execution.
- **Layer 8 — Deep RL execution (Nevmyvaka et al. 2006):** A-C provides the baseline cost; RL extensions learn to place each child as a limit order at a chosen depth.

## Limitations and failure modes

1. **Linear impact assumption.** Real impact is concave (square-root). A-C overprices impact for small orders and underprices for large orders. Use Almgren (2003) extensions with concave impact for $Q/V > 0.05$.
2. **Constant $\sigma$.** Volatility clustering (GARCH effects) is ignored. For multi-day horizons during stress, the realized $\mathbb{V}[C]$ will exceed the model's prediction.
3. **No drift or signal.** A-C is purely defensive — it assumes mid is a martingale. With short-term alpha, the optimal schedule should tilt against (for buys) or with (for sells) the predicted drift; this is the alpha-adjusted Almgren-Chriss of Garleanu-Pedersen.
4. **Constant impact coefficients.** $\eta$ varies intraday (lunchtime thinness) and across regimes. The VWAP-shaped extension (Algorithm C) handles intraday; regime-conditional $\eta$ handles the rest.
5. **No price-limit constraints.** A-C does not respect "do not buy above $P + 50$ bps" rules common in client mandates. Add as inequality constraints in the QP form.
6. **Independence of executions.** A-C assumes each child trades at the mid plus impact. Real fills are random — partial fills, queue position, adverse selection on limit orders — all absent.
7. **Quadratic temporary impact has no upper bound.** In thin markets, A-C may prescribe a rate that the market cannot absorb. Always cap $v_t \le \rho V_{\text{daily}}/T$ for some participation rate $\rho \le 0.25$.
8. **Mean-variance is not log-utility.** For high-risk-aversion regimes, mean-variance disagrees with expected utility. Stick to $\lambda$ in the calibrated range where the disagreement is small.
9. **Permanent impact drops from optimization.** While this is mathematically elegant, real permanent impact can be schedule-dependent (information leakage). The textbook model misses this.

## References

- Almgren, R. and Chriss, N. (2000). "Optimal execution of portfolio transactions." *Journal of Risk*, 3, 5-39.
- Almgren, R. (2003). "Optimal execution with nonlinear impact functions and trading-enhanced risk." *Applied Mathematical Finance*, 10, 1-18.
- Almgren, R., Thum, C., Hauptmann, E., and Li, H. (2005). "Direct estimation of equity market impact." *Risk*, 18.
- Bertsimas, D. and Lo, A. W. (1998). "Optimal control of execution costs." *Journal of Financial Markets*, 1, 1-50.
- Garleanu, N. and Pedersen, L. H. (2013). "Dynamic Trading with Predictable Returns and Transaction Costs." *Journal of Finance*, 68(6), 2309-2340.
- Obizhaeva, A. and Wang, J. (2013). "Optimal trading strategy and supply/demand dynamics." *Journal of Financial Markets*, 16(1), 1-32.
- Cartea, A., Jaimungal, S., and Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
- Gueant, O. (2016). *The Financial Mathematics of Market Liquidity: From Optimal Execution to Market Making.* Chapman & Hall/CRC.
- Nevmyvaka, Y., Feng, Y., and Kearns, M. (2006). "Reinforcement learning for optimized trade execution." *Proceedings of the 23rd International Conference on Machine Learning*.
- Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management.* Academic Press.
