# Online Learning (Hedge / FTRL / FTPL)

> Adversarial-regret online optimization algorithms for updating signal weights and feature coefficients in real time under non-stationarity. Layer 4 — Signals & ML. Primary use: combine multiple predictive signals (GARCH, HAR-RV, momentum, ML ensembles) into a meta-prediction; online logistic/linear regression on streaming features.

## Mathematical formulation

### Setup: the experts / online convex optimization (OCO) protocol

At each round $t = 1, 2, \dots, T$:
1. Learner picks $\mathbf{w}_t \in \mathcal{W}$ (a convex action set — a probability simplex for Hedge, $\mathbb{R}^d$ for FTRL).
2. Adversary reveals loss $\ell_t : \mathcal{W} \to \mathbb{R}$.
3. Learner incurs $\ell_t(\mathbf{w}_t)$.

**Regret** against the best fixed action in hindsight:

$$R_T = \sum_{t=1}^T \ell_t(\mathbf{w}_t) - \min_{\mathbf{w} \in \mathcal{W}} \sum_{t=1}^T \ell_t(\mathbf{w}).$$

A useful algorithm is one whose regret grows sublinearly in $T$, so the per-round average regret $R_T / T \to 0$ — the learner is "asymptotically as good as the best fixed expert."

### Hedge / Exponential Weights (Freund–Schapire)

Maintain weights $w_i^{(t)} \ge 0$ on $N$ experts, $\sum_i w_i^{(t)} = 1$. On round $t$, expert $i$ incurs loss $\ell_i^{(t)} \in [0,1]$.

$$w_i^{(t+1)} = \frac{w_i^{(t)} \exp(-\eta \ell_i^{(t)})}{Z^{(t)}}, \quad Z^{(t)} = \sum_j w_j^{(t)} \exp(-\eta \ell_j^{(t)}).$$

Equivalently with cumulative loss $L_i^{(t)} = \sum_{s \le t} \ell_i^{(s)}$,

$$w_i^{(t+1)} \propto \exp(-\eta L_i^{(t)}).$$

**Regret bound.** For $\eta = \sqrt{8 \log N / T}$,

$$R_T \le \sqrt{\frac{T \log N}{2}} = O(\sqrt{T \log N}).$$

Doubling trick or adaptive $\eta_t = \sqrt{\log N / t}$ removes the dependence on knowing $T$ in advance.

### Follow-the-Regularized-Leader (FTRL)

Generalization of Hedge to arbitrary convex losses with a strongly convex regularizer $\Psi$:

$$\mathbf{w}_{t+1} = \arg\min_{\mathbf{w} \in \mathcal{W}} \left\{ \sum_{s=1}^t \langle \mathbf{g}_s, \mathbf{w} \rangle + \Psi_t(\mathbf{w}) \right\},$$

where $\mathbf{g}_s = \nabla \ell_s(\mathbf{w}_s)$ is the gradient at round $s$.

**FTRL-Proximal (McMahan, 2011)** — the workhorse for online linear/logistic regression with sparse output:

$$\boldsymbol{\theta}_{t+1} = \arg\min_{\boldsymbol{\theta}} \left\{ \mathbf{z}_t^\top \boldsymbol{\theta} + \lambda_1 \|\boldsymbol{\theta}\|_1 + \tfrac{1}{2} \sum_{s=1}^t \sigma_s \|\boldsymbol{\theta} - \boldsymbol{\theta}_s\|_2^2 \right\},$$

where $\mathbf{z}_t = \sum_{s=1}^t \mathbf{g}_s - \sum_{s=1}^t \sigma_s \boldsymbol{\theta}_s$ and $\sigma_s = (1/\eta_s - 1/\eta_{s-1})$ is the per-step learning-rate increment, $1/\eta_t = \sum_s \sigma_s$.

Coordinate-wise closed form per coordinate $j$:

$$\theta_{t+1, j} = \begin{cases} 0 & \text{if } |z_{t,j}| \le \lambda_1 \\ -\eta_t \big(z_{t,j} - \lambda_1 \, \text{sign}(z_{t,j})\big) & \text{otherwise} \end{cases}.$$

**Per-coordinate learning rates** (the Google CTR-prediction paper):

$$\eta_{t,j} = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^t g_{s,j}^2}},$$

with $\alpha \in [0.01, 0.5]$, $\beta = 1$. This is **AdaGrad-style** — high-frequency features get smaller steps. Combined with L1 regularization it yields sparse models on huge feature sets at scale.

**Regret bound.** For online convex losses with $G$-bounded gradients and domain radius $D$,

$$R_T \le O(GD \sqrt{T}).$$

### Follow-the-Perturbed-Leader (FTPL)

Sample perturbation $\boldsymbol{\xi}_t$ (e.g., Gumbel or Gaussian) and play

$$\mathbf{w}_t = \arg\min_{\mathbf{w}} \left\{ \sum_{s=1}^{t-1} \langle \mathbf{g}_s, \mathbf{w} \rangle - \langle \boldsymbol{\xi}_t, \mathbf{w} \rangle \right\}.$$

For Hedge-like setups, Gumbel-distributed $\boldsymbol{\xi}_t$ yields the same expected updates as exponential weights (Gumbel-max trick). FTPL is preferred when the regularizer-based FTRL update is computationally hard (e.g., combinatorial action sets).

### Optimistic / dynamic regret variants

When the loss sequence is partially predictable via a hint $\mathbf{m}_t$ (e.g., expected next gradient), **Optimistic FTRL** plays

$$\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left\{ \sum_{s=1}^t \langle \mathbf{g}_s, \mathbf{w} \rangle + \langle \mathbf{m}_{t+1}, \mathbf{w} \rangle + \Psi_t(\mathbf{w}) \right\}.$$

If $\mathbf{m}_t$ is accurate, regret reduces to $O(\sqrt{\sum_t \|\mathbf{g}_t - \mathbf{m}_t\|^2})$ — small when the world is predictable.

**Dynamic regret** against a sequence of comparators $\{\mathbf{u}_t\}$:

$$R_T^{\text{dyn}} = \sum_{t=1}^T \ell_t(\mathbf{w}_t) - \sum_{t=1}^T \ell_t(\mathbf{u}_t),$$

bounded by $O(\sqrt{T(1 + P_T)})$ where $P_T = \sum_t \|\mathbf{u}_t - \mathbf{u}_{t-1}\|$ is the path length.

## Intuition and what this model addresses

Markets are non-stationary: signal quality drifts, feature distributions shift, alpha decays. Batch retraining on a fixed window is inherently lagged. Online learning **does not assume stationarity** — its regret bound holds *adversarially*, against any sequence of losses.

Three concrete deployment patterns:

1. **Hedge over signal-experts.** Suppose you have signals $\hat{r}^{(1)}_t, \dots, \hat{r}^{(N)}_t$ from $N$ models (GARCH vol forecast, HAR-RV, momentum, SAE+MLP, GBDT, etc.). Treat each as an "expert." Each round, observe realized return $r_t$ and incur loss $\ell^{(i)}_t = (\hat{r}^{(i)}_t - r_t)^2$ for each expert. Hedge gives a meta-prediction $\hat{r}_t = \sum_i w_i^{(t)} \hat{r}^{(i)}_t$ that is asymptotically as good as the *best* signal. Weights adapt as the best signal changes across regimes.

2. **FTRL-Proximal for online logistic regression on sparse features.** Predict next-period sign of return from one-hot encoded calendar features, sector dummies, news-keyword indicators. The L1 component produces a sparse coefficient vector that updates one example at a time — no batch retrain.

3. **FTPL for portfolio simplex.** Adversarial portfolio selection over $N$ assets: pick portfolio weights on the simplex each day; Cover's Universal Portfolio is a Bayesian relative; FTPL with Gumbel noise produces a tractable approximation.

## Architecture / model design

Hedge is parameter-free given $N$ experts and $\eta$. FTRL-Proximal maintains, per feature coordinate $j$:

- $z_j$: accumulated gradient minus learning-rate increment adjustments.
- $n_j$: accumulated squared gradient $\sum_s g_{s,j}^2$.
- Current $\theta_j$: derived on-the-fly from $z_j$, $n_j$.

State size: $O(d)$ for $d$ features. No batches, no epochs, no validation set — only a streaming evaluation buffer.

Hyperparameters of FTRL-Proximal: $\alpha$ (per-coordinate base rate), $\beta$ (per-coordinate offset), $\lambda_1$ (L1), $\lambda_2$ (optional L2). For Hedge: $\eta$ (or adaptive schedule).

## Calibration / training approach

- **Loss function.** Whatever round-by-round loss makes operational sense: squared error for regression, log-loss for probability, hinge for classification, $-$utility for trade-quality.
- **Learning rate.** For Hedge in stationary regime: $\eta = \sqrt{8 \log N / T_0}$ with $T_0$ = recent window length. For non-stationary: use **shifting experts** (Herbster–Warmuth, fixed-share update) where after the multiplicative step you mix with uniform: $w_i^{(t+1)} \leftarrow (1-\beta) w_i^{(t+1)} + (\beta/N)$ with $\beta \approx 0.01$.
- **Initialization.** Hedge: uniform $w_i^{(0)} = 1/N$. FTRL: $\boldsymbol{\theta}_0 = 0$, $\mathbf{z}_0 = 0$, $\mathbf{n}_0 = 0$.
- **Warm-up.** Apply FTRL-Proximal on a historical batch first (1–2 passes) to set $\mathbf{z}, \mathbf{n}$ before going live.
- **Bias features.** Always include an intercept (constant feature) and a per-instance bias.
- **Feature hashing.** For sparse high-cardinality categorical features, hash to $d = 2^{20}$ slots; FTRL's per-coordinate state size keeps memory fixed.
- **Validation.** Streaming progressive validation — predict $\hat{y}_t$ *before* seeing $y_t$, accumulate prequential loss.

## Algorithm outline

### Hedge over signal-experts

```
INPUT: N experts producing predictions yhat_i_t each round, learning rate eta
INIT:  w_i = 1/N for i = 1..N
For t = 1, 2, ...:
  Receive predictions yhat_1, ..., yhat_N at time t
  Output meta-prediction yhat_meta = sum_i w_i * yhat_i
  Observe realized y_t
  Compute losses l_i = (yhat_i - y_t)^2  (clipped to [0, L_max])
  Normalize: l_i_tilde = l_i / L_max  in [0,1]
  Multiplicative update:
     w_i <- w_i * exp(-eta * l_i_tilde)
  Renormalize: w_i <- w_i / sum_j w_j
  Optional fixed-share mixing for non-stationarity:
     w_i <- (1 - beta) * w_i + beta / N
```

### FTRL-Proximal for online logistic regression

```
INPUT: streaming examples (x_t in R^d sparse, y_t in {0,1})
HYPERPARAMS: alpha, beta, lambda1, lambda2
INIT: For each j: z_j = 0, n_j = 0

For each example (x_t, y_t):
  # 1. Compute current theta_j for active features
  For each j where x_{t,j} != 0:
     if |z_j| <= lambda1:
        theta_j = 0
     else:
        theta_j = -(z_j - sign(z_j) * lambda1) /
                  ((beta + sqrt(n_j)) / alpha + lambda2)

  # 2. Predict
  score = sum_j theta_j * x_{t,j}
  phat  = sigmoid(score)
  Emit phat (this is your live prediction)

  # 3. Observe y_t, compute gradient
  For each j where x_{t,j} != 0:
     g_j     = (phat - y_t) * x_{t,j}        (gradient of log-loss)
     sigma_j = (sqrt(n_j + g_j^2) - sqrt(n_j)) / alpha
     z_j    += g_j - sigma_j * theta_j
     n_j    += g_j^2
```

### FTPL with Gumbel noise (simplex)

```
INPUT: N actions, cumulative losses L_i_t
At round t+1:
  Draw xi_i ~ Gumbel(0,1) i.i.d. for i=1..N
  Play action i* = argmin_i { L_i_t - xi_i / eta }
```

Equivalent in expectation to Hedge with rate $\eta$ (Gumbel-max identity).

## yfinance data requirements and feature engineering

**Use case — Hedge across vol-forecast experts.**

Experts:
- HAR-RV from intraday `yf.download(t, interval='5m', period='60d')`.
- GARCH(1,1) from daily returns.
- Implied vol proxy from `yf.Ticker(t).options` (ATM-IV).
- Realized vol over 20-day window.
- ML ensemble (SAE+MLP, GBDT) point forecasts of $\hat{\sigma}_t$.

Round $t$ (daily): each expert outputs $\hat{\sigma}^{(i)}_t$ before close. After close, observe realized $\sigma_t = \sqrt{\sum_s r_{t,s}^2}$. Compute squared-log loss $\ell^{(i)}_t = (\log \hat{\sigma}^{(i)}_t - \log \sigma_t)^2$. Update Hedge weights.

**Use case — FTRL on a wide feature set for return-sign prediction.**

Features per (ticker, date):
- Sparse one-hot of sector, market-cap bucket, day-of-week, month.
- Hashed bigrams from news headlines via `yf.Ticker(t).news`.
- Quantile-binned numerical features (5 quantiles each: $r^{(5)}, r^{(20)}, \hat{\sigma}^{(20)}, V/MA_{20}(V)$).

After hashing: $d \approx 2^{20}$. FTRL fits naturally — only active features incur work per example.

**yfinance pipeline.**

```python
import yfinance as yf
for date in trading_days:
    for tkr in universe:
        bar = yf.download(tkr, start=date, end=date+1d, interval='1d')
        features = featurize(bar, news=yf.Ticker(tkr).news)
        # stream into online learner
```

## Refit frequency

- **Online — no batch refit.** That's the point.
- **Periodic checkpoint.** Snapshot state $(\mathbf{z}, \mathbf{n})$ for FTRL or $\mathbf{w}$ for Hedge daily to disk for crash recovery and offline analysis.
- **Hyperparameter retune.** $\alpha, \beta, \lambda_1$ for FTRL and $\eta$ for Hedge retuned monthly by grid search on the past 60 days with progressive validation.
- **Reset on regime break.** If a structural break is detected (e.g., CUSUM on prequential loss), inject fresh uniform mass into Hedge or partially zero $\mathbf{z}$ in FTRL — equivalent to fixed-share.

## Validation and diagnostics

- **Prequential (online) loss.** Cumulative loss $L_T = \sum_{t=1}^T \ell_t(\mathbf{w}_t)$, normalized $L_T / T$.
- **Empirical regret.** $L_T - \min_i L_i^{(T)}$ for Hedge; compare to theoretical bound $\sqrt{T \log N / 2}$.
- **Weight entropy.** $H_t = -\sum_i w_i^{(t)} \log w_i^{(t)}$. Drops to 0 if Hedge over-commits to one expert (often a sign $\eta$ is too aggressive).
- **Coefficient sparsity (FTRL).** Fraction of $\theta_j = 0$; expected $> 95\%$ for properly tuned $\lambda_1$.
- **Online ROC-AUC** for classification — streaming via reservoir sampling or buffered windows.
- **Stability under permutations.** Shuffle the recent prequential window and refit; large variance in final $\mathbf{w}$ flags overfitting to order.

## Connections to other models

- **SAE+MLP (Model 10), GBDT (Model 11), Bayesian (Model 12).** Each contributes a signal; Hedge combines them.
- **Kalman filter (Model 12).** Kalman is the Bayesian online update for a Gaussian linear-state model; FTRL is the regret-minimizing version that drops the Gaussian assumption.
- **GARCH (Layer 3) & HAR-RV.** Their vol forecasts become Hedge experts.
- **Anomaly detection (Model 14).** Anomaly scores can themselves be experts; Hedge learns when to trust them.
- **Execution (Layer 7).** FTRL on routing features for SOR decisions — venue selection updated per child order.

## Limitations and failure modes

1. **Adversarial bound is loose in practice.** Worst-case regret $O(\sqrt{T \log N})$ may be much better than realized — and worse on truly adversarial sequences (e.g., during a flash crash).
2. **Loss scale matters.** Hedge requires losses bounded in $[0,1]$; uncontrolled MSE on tail returns blows up updates. Always clip or rescale.
3. **Slow adaptation under abrupt regime shift.** Vanilla Hedge with constant $\eta$ takes $O(\log N / \eta)$ rounds to forget a previously-best expert. Use fixed-share / shifting-experts or restarts.
4. **Tied experts.** Highly correlated experts inflate effective $N$ without diversification benefit; cluster experts and run Hedge on cluster medoids.
5. **L1 sparsity instability (FTRL).** A feature near the L1 boundary toggles between 0 and nonzero; consider Group-Lasso or proximal averaging.
6. **Latency.** Hedge prediction is $O(N)$; FTRL is $O(\text{nnz}(\mathbf{x}_t))$. Both fast, but logging gradients and checkpointing state add overhead.
7. **No uncertainty estimates.** Outputs are point predictions; pair with online conformal prediction for intervals.
8. **yfinance data freshness.** Live online learning requires low-latency data; yfinance is best for daily/15-min refresh; for tick-level use a vendor feed.

## References

1. Freund, Y., & Schapire, R. (1997). *A decision-theoretic generalization of online learning.* JCSS.
2. McMahan, H. B. (2011). *Follow-the-regularized-leader and mirror descent: equivalence and L1 regularization.* AISTATS.
3. McMahan, H. B., et al. (2013). *Ad click prediction: a view from the trenches.* KDD.
4. Cesa-Bianchi, N., & Lugosi, G. (2006). *Prediction, Learning, and Games.* Cambridge.
5. Hazan, E. (2016). *Introduction to Online Convex Optimization.* Foundations and Trends in Optimization.
6. Kalai, A., & Vempala, S. (2005). *Efficient algorithms for online decision problems.* JCSS. (FTPL)
7. Herbster, M., & Warmuth, M. K. (1998). *Tracking the best expert.* Machine Learning. (Fixed-share)
8. Rakhlin, A., & Sridharan, K. (2013). *Online learning with predictable sequences.* COLT. (Optimistic algorithms)
9. Zinkevich, M. (2003). *Online convex programming and generalized infinitesimal gradient ascent.* ICML.
