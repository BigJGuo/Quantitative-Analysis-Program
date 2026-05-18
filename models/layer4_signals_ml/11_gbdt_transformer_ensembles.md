# Gradient-Boosted Trees and Transformer Ensembles

> Stacked ensembles of LightGBM/XGBoost/CatBoost with Transformer encoders and GRU/MLPs over a tabular panel. Layer 4 — Signals & ML. Primary use: cross-sectional and time-series return forecasting on a large (date, time, symbol) panel.

## Mathematical formulation

### Gradient-Boosted Decision Trees

GBDT fits an additive model

$$F_M(\mathbf{x}) = F_0 + \sum_{m=1}^{M} \eta \, f_m(\mathbf{x}),$$

where $f_m$ is a decision tree (a piecewise-constant function on a partition of $\mathbb{R}^p$ into $J_m$ leaves), $\eta \in (0,1]$ is the shrinkage/learning rate, and $F_0$ is a constant initialization (often the loss-minimizing constant — the mean for MSE, the log-odds for logistic).

At stage $m$, fit $f_m$ to the negative gradient (pseudo-residual) of the loss:

$$g_i^{(m)} = -\left.\frac{\partial \ell(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right|_{F = F_{m-1}}.$$

For squared loss $\ell = \tfrac{1}{2}(y - F)^2$, $g_i^{(m)} = y_i - F_{m-1}(\mathbf{x}_i)$ (ordinary residuals).

**XGBoost** uses a second-order Taylor expansion. Let $g_i = \partial_F \ell$, $h_i = \partial_F^2 \ell$. The structure score of a tree $T$ with leaves $\{I_j\}_{j=1}^J$ is

$$\mathcal{L}^{(m)}(T) = -\frac{1}{2}\sum_{j=1}^{J} \frac{\left(\sum_{i \in I_j} g_i\right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma J,$$

where $\lambda$ is L2 leaf-weight regularization and $\gamma$ is the per-leaf complexity cost. Optimal leaf value: $w_j^* = -\sum_{i\in I_j} g_i / (\sum_{i \in I_j} h_i + \lambda)$. Split gain: difference between parent score and the sum of child scores.

**LightGBM** uses histogram-based split finding (each feature bucketed into $b \approx 255$ bins so the gain scan is $O(b)$ not $O(N)$) and **leaf-wise** (best-first) growth — at each step, split the leaf with the highest gain, regardless of depth. Faster and often more accurate than level-wise growth but prone to overfit narrow regions; control with `num_leaves` and `min_data_in_leaf`.

**CatBoost** uses **ordered boosting**: for each example $i$, gradients are computed using a model trained only on the prefix of the permuted dataset that excludes $i$. This eliminates target leakage in target-encoded categorical features. Trees are **oblivious** (same split feature at every node of a given depth), which acts as a strong regularizer.

### Transformer encoder

Input sequence (e.g., a per-symbol history of length $T$ with $d$ features) is embedded to $\mathbf{X} \in \mathbb{R}^{T \times d_\text{model}}$ plus positional encoding $\mathbf{P}$:

$$\mathbf{Z}_0 = \mathbf{X} \mathbf{W}_e + \mathbf{P}.$$

Each encoder block computes

$$\mathbf{Q} = \mathbf{Z} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}^V,$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V},$$

with causal mask $\mathbf{M}$ ($M_{ij} = -\infty$ for $j > i$) to prevent look-ahead. Multi-head: split into $H$ heads, concatenate, project. Then position-wise FFN $\text{FFN}(\mathbf{z}) = \mathbf{W}_2 \, \text{GELU}(\mathbf{W}_1 \mathbf{z})$, with residual connections and LayerNorm around each sub-block.

### Ensemble combination

Given $M$ base models with predictions $\hat{y}_m(\mathbf{x})$, two combination rules dominate:

- **Simple average / median:** $\hat{y}(\mathbf{x}) = \tfrac{1}{M}\sum_m \hat{y}_m(\mathbf{x})$ or $\text{median}_m \hat{y}_m(\mathbf{x})$.
- **Stacking:** train a meta-learner $g_\theta$ on out-of-fold predictions: $\hat{y}(\mathbf{x}) = g_\theta(\hat{y}_1(\mathbf{x}), \dots, \hat{y}_M(\mathbf{x}))$, typically linear or LightGBM with ~5–10 inputs.

## Intuition and what this model addresses

GBDTs capture interactions and non-linearities on tabular data with minimal preprocessing: splits handle missingness, monotone scaling is irrelevant, and trees discover thresholds naturally (which is essential when alpha lives in tail regions — e.g., "if VIX > 25 AND drawdown > 5% then ..."). They are the de facto winner on most heterogeneous tabular Kaggle competitions.

Transformers complement GBDTs by modeling **time structure**: long-range dependencies in a symbol's history, cross-symbol attention if you tokenize symbols as positions, and intraday patterns via positional encodings of `time_id`. GBDTs see each row independently (modulo lagged features); a Transformer sees the *sequence*.

Ensembling the two families is robust because their error correlations are low: GBDT errors are largely from regions with few training examples; Transformer errors are largely from regime changes outside the training horizon.

The 2024–25 Jane Street competition had 47M+ rows on a `(date_id, time_id, symbol_id)` grid with 79 features and 9 responder targets clipped to $[-5, 5]$. Public top write-ups all used GBDT + DL blends with careful time-series cross-validation.

## Architecture / model design

### GBDT base learners

| Hyperparameter | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| Learning rate $\eta$ | 0.02–0.05 | 0.02–0.05 | 0.03 |
| Num leaves / max depth | 64–256 leaves | depth 6–10 | depth 6–8 |
| Min data per leaf | 1000–5000 | (via min_child_weight) | 1 |
| Subsample (rows) | 0.7 | 0.7 | 0.8 (bagging_temperature) |
| Colsample (features) | 0.7 | 0.7 | 0.8 |
| L2 ($\lambda$) | 1.0 | 1.0 | 3.0 |
| Trees ($M$) | 3000–10000 w/ early stop | 3000–10000 | 3000–10000 |

### Transformer encoder for time-series

- Input: per-symbol window of length $T = 60$, feature dim $d = 79$.
- Embedding: linear $79 \to d_\text{model}=128$ + learned positional embedding on $T$.
- 4 encoder blocks, 8 heads, $d_k = d_v = 16$, FFN hidden 512, GELU, dropout 0.1, pre-LayerNorm.
- Output head: take last-token representation $\mathbf{z}_T \in \mathbb{R}^{128}$ → Linear(128 → 9) → 9 responders.

### GRU + wide-and-deep alternative

- GRU(input=79, hidden=128, layers=2) over the $T$-step window.
- Wide path: raw cross-sectional rank features $\mathbf{r}_t$ → Linear → concat with GRU final hidden.
- Deep path: 3-layer MLP with dropout 0.3.
- Output: Linear → 9 responders.

### Stacking meta-learner

After base models, fit a level-2 LightGBM with input features = base predictions (5–10 columns) + a handful of context features (date_id sine/cosine, symbol_id one-hot summary, VIX-proxy).

## Calibration / training approach

**Loss functions.**
- Regression on clipped responders: Huber loss $\ell_\delta(r) = \tfrac{1}{2} r^2$ for $|r| \le \delta$, else $\delta(|r| - \tfrac{\delta}{2})$, with $\delta = 1$, robust to clipping plateaus.
- Alternative: weighted MSE with weights from the competition's sample-weight column.

**Optimizers.**
- GBDTs: their internal Newton step.
- DL models: AdamW with $\eta = 3\times 10^{-4}$, cosine schedule with warmup 1000 steps, weight decay $10^{-2}$.

**Regularization.**
- GBDT: shrinkage ($\eta$ small), subsample, colsample, min-data-in-leaf, max-bin.
- DL: dropout 0.1–0.3, gradient clipping at 1.0, label smoothing not used (regression).

**Batch sizes.**
- LightGBM: not applicable (the whole dataset is held in memory in histograms; for 47M rows use `two_round` loading).
- Transformer: batch 256 sequences of length 60.

**Validation scheme — critical.**
- Use **purged group K-fold** by `date_id`. Group = date. Purge: remove training dates within $h$ days *on either side* of the validation block, where $h$ matches the maximum forward horizon of the target. Embargo: an additional gap of 1–2 days after the validation block.
- 5 folds with embargo 1, purge 5 is typical.
- Final model retrained on full data with the same hyperparameters and number-of-trees fixed from the CV optimum.

**Sample weighting.** Most competitions ship sample weights $w_i$. Pass to LightGBM `sample_weight`, XGBoost `weight`, CatBoost `weight`. For DL, multiply the per-row loss by $w_i / \bar{w}$.

## Algorithm outline

```
INPUT: panel P = {(date_id, time_id, symbol_id, x in R^79, y in R^9, w)}, splits K=5

1. Feature engineering (per symbol_id, sorted by date_id, time_id):
     - Lagged responders: lag_k(y_j) for k in {1,2,3,5}, j in {0..8}
     - Rolling mean and std of each feature over windows {5, 20, 60} within symbol
     - Cross-sectional rank per date_id: rank(x_j) / N_d
     - Time-of-day encoding: sin(2 pi * time_id / T_day), cos(...)
     - Symbol mean-encoding via target-mean from prior dates only (or use CatBoost ordered)
     - NaN -> -1 sentinel for GBDT; NaN -> 0 with mask channel for DL

2. Build purged K-fold splits on date_id with purge=5, embargo=1.

3. For each fold f:
     (a) Train LightGBM_f on responders[0] (primary), early-stop on val MSE
     (b) Train XGBoost_f similarly
     (c) Train CatBoost_f similarly
     (d) Train Transformer_f on length-T windows ending at the predicted timestamp
     (e) Train GRU_f similarly
     Store out-of-fold preds for each base model.

4. Stack: fit meta-learner g on OOF predictions (5-col table) -> meta CV.

5. Refit each base model on all data with best hyperparameters and trees/epochs from CV.

6. Inference at time (d, t, s):
     For each base model b: yhat_b = b.predict(x_{d,t,s})
     yhat = g(yhat_1, ..., yhat_M) or simple mean across b.

7. Risk-truncate: clip yhat to the 1%/99% empirical quantiles of training preds.
```

## yfinance data requirements and feature engineering

Build a multi-ticker daily panel; intraday is optional via `interval='1h'` or `'5m'` (limited history). Recommended universe: S&P 500 + sector ETFs (~510 tickers).

**yfinance calls:**

```python
import yfinance as yf
data = yf.download(tickers=universe, start='2010-01-01', end='today',
                   interval='1d', group_by='ticker', auto_adjust=True,
                   threads=True)
# Per ticker: Open, High, Low, Close, Volume
info = {t: yf.Ticker(t).info for t in universe}   # sector, marketCap, beta
```

**Feature matrix (per row = (ticker, date)):**

1. **Return features.** $r^{(k)}_t = \log P_t - \log P_{t-k}$ for $k \in \{1, 5, 10, 21, 63, 126, 252\}$.
2. **Lagged returns.** $r^{(1)}_{t-1}, \dots, r^{(1)}_{t-10}$ — 10 columns.
3. **Rolling volatility.** $\hat{\sigma}^{(k)}_t = \text{std}(r^{(1)}_{t-k+1:t})$ for $k \in \{5, 21, 63\}$.
4. **Rolling means.** $\bar{r}^{(k)}_t$ same windows.
5. **High-low range.** $(H_t - L_t) / C_t$, also $(C_t - O_t)/O_t$.
6. **Volume.** $\log V_t$, $V_t / \text{MA}_{20}(V_t)$.
7. **Amihud illiquidity.** $|r^{(1)}_t| / (V_t \cdot C_t)$.
8. **Cross-sectional ranks.** For each feature $f$, $\text{rank}_t(f) / N_t$ over the universe — gives Transformers explicit cross-sectional context.
9. **Sector features.** One-hot sector, sector-mean of $r^{(k)}_t$, sector-mean of $\hat{\sigma}^{(k)}_t$.
10. **Calendar.** Day-of-week, month, days-to-quarter-end.
11. **Benchmark spread.** $r^{(k)}_t(\text{ticker}) - r^{(k)}_t(\text{SPY})$.
12. **Lagged target.** Previous-period $r^{(5)}_{t-5}$ as a feature (allowed if predicting $r^{(5)}_{t+5}$).

Target: forward log return $y_t = \log P_{t+h} - \log P_t$ for $h \in \{1, 5, 21\}$. Clip to $\pm 5$ standard deviations or to fixed $[-0.2, 0.2]$ to prevent outliers dominating MSE.

Drop tickers with < 2 years of history. Forward-fill missing OHLC up to 3 days; longer gaps become a missingness feature.

## Refit frequency

- **GBDTs:** full retrain weekly on a rolling 5-year window. Daily incremental fitting (`init_model=...`) is possible but risks compounding error — prefer weekly clean retrain.
- **Transformer/GRU:** retrain monthly; fine-tune nightly for 1 epoch on the past 30 days.
- **Stacking meta-learner:** refit weekly with the latest OOF window.
- **Trigger-based:** if rolling out-of-sample $R^2$ on the past 20 days drops below the 5th percentile of in-sample, force retrain.

## Validation and diagnostics

- **Primary metric.** Weighted $R^2$: $1 - \sum w_i (y_i - \hat{y}_i)^2 / \sum w_i (y_i - \bar{y})^2$.
- **Cross-sectional rank IC.** Spearman correlation between $\hat{y}_t$ and $y_t$ within each date.
- **Time-series IC.** Pearson correlation per ticker over time, then averaged.
- **Decile spread.** Long top decile, short bottom decile; track Sharpe and turnover.
- **Feature importance.** LightGBM `gain`/`split`, SHAP values for top contributors.
- **Stability across folds.** Variance of feature importance across the 5 folds — high variance = unstable signal.
- **Calibration plot.** Bin predictions into deciles, plot vs realized.
- **Per-regime decomposition.** Performance split by VIX quartile, by sector, by trend/range regime.
- **Concept drift.** Population Stability Index on each feature month over month.

## Connections to other models

- **SAE+MLP (Model 10).** Latent codes $\mathbf{h}$ become features for LightGBM; LightGBM predictions become features for the stacker that already contains the SAE prediction.
- **Bayesian hierarchical (Model 12).** Hierarchical-Bayes shrinkage targets serve as prior means for the GBDT residual model (predict $y - \mu_\text{Bayes}$).
- **Online learning (Model 13).** Run Hedge across base learners (LightGBM, XGBoost, CatBoost, Transformer, GRU) — the meta-weights adapt to regime.
- **GARCH (Layer 3).** Vol forecasts become features and also rescale targets (predict standardized return $y / \hat{\sigma}$).
- **Anomaly detection (Model 14).** Filter training rows with anomaly score above a threshold to clean labels.

## Limitations and failure modes

1. **Look-ahead leakage.** Cross-sectional ranks computed using the validation date's data are fine; using future dates is leakage. Rolling features computed across the fold boundary leak.
2. **Tree extrapolation.** GBDTs cannot extrapolate beyond observed feature ranges; new regime values (e.g., negative oil) get mapped to leaf boundaries.
3. **Target clipping artifacts.** Trees learn to predict the clip value as a mode; the loss landscape becomes flat. Use Huber, not pure MSE.
4. **Transformer overfit on small N.** With 47M rows it works; with 5M rows the Transformer is usually beaten by LightGBM. Don't fight the data.
5. **Inference latency.** Production stack: LightGBM is sub-millisecond; XGBoost similar; CatBoost slower; Transformer needs GPU + batching.
6. **Memory pressure.** 47M × 200 features × 8 bytes ≈ 70 GB. Use `float32` or quantized features; LightGBM's binning solves this naturally.
7. **Hyperparameter staleness.** Best params drift; re-tune monthly with Optuna on the most recent fold.
8. **Sample weight misuse.** Forgetting to pass weights inflates loss on low-weight rows that are quiet by design.

## References

1. Friedman, J. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
3. Ke, G. et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS.
4. Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS.
5. Vaswani, A. et al. (2017). *Attention is all you need.* NeurIPS.
6. Xu, J. et al. (2021). *Autoformer / Informer for time-series forecasting.* NeurIPS / AAAI.
7. de Prado, M. L. (2018). *Advances in Financial Machine Learning,* purged K-fold CV.
8. Public 2024–25 Jane Street Real-Time Market Data Forecasting Kaggle write-ups.
