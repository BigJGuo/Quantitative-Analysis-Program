# Anomaly Detection (Autoencoders, Isolation Forests) + FinBERT NLP

> Unsupervised outlier scoring on numerical features (autoencoders, isolation forests, one-class SVM) paired with transformer-based sentiment classification of news. Layer 4 — Signals & ML. Primary use: fat-finger and adverse-selection monitoring, training-data cleaning, and news sentiment as an alpha feature.

## Mathematical formulation

### Autoencoder reconstruction-error anomaly score

Encoder $E_\phi : \mathbb{R}^p \to \mathbb{R}^d$, decoder $D_\psi : \mathbb{R}^d \to \mathbb{R}^p$. Trained on a "clean" reference set with reconstruction loss

$$\mathcal{L}_\text{recon}(\mathbf{x}) = \|\mathbf{x} - D_\psi(E_\phi(\mathbf{x}))\|_2^2.$$

At inference,

$$s_\text{AE}(\mathbf{x}) = \|\mathbf{x} - \text{AE}(\mathbf{x})\|_2^2.$$

For per-feature attribution, use the squared residual vector $\mathbf{r}(\mathbf{x}) = (\mathbf{x} - \text{AE}(\mathbf{x}))^{\circ 2}$ and standardize by the in-sample variance of $\mathbf{r}$:

$$\tilde{r}_j(\mathbf{x}) = r_j(\mathbf{x}) / \hat{\sigma}_{r_j}^2.$$

A row is flagged if $s_\text{AE}(\mathbf{x}) > Q_{1-\alpha}(s_\text{AE} \mid \text{train})$ — empirical quantile of in-sample scores at level $1-\alpha$ (e.g., $\alpha = 0.005$).

**Variational autoencoder (VAE) score.** Density estimate via the ELBO:

$$\log p(\mathbf{x}) \ge \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - \text{KL}(q(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z})).$$

Anomaly score $s_\text{VAE}(\mathbf{x}) = -\text{ELBO}(\mathbf{x})$.

### Isolation Forest

An **iTree** is built by recursively splitting a sample of size $\psi$ along randomly chosen features at random split points until leaves contain one example or reach `max_depth = log_2(\psi)`. Average path length to isolate $\mathbf{x}$ across $t$ trees:

$$\bar{h}(\mathbf{x}) = \frac{1}{t} \sum_{i=1}^t h_i(\mathbf{x}).$$

Normalized anomaly score:

$$s_\text{iForest}(\mathbf{x}) = 2^{-\bar{h}(\mathbf{x}) / c(\psi)},$$

where $c(\psi) = 2 H(\psi - 1) - 2(\psi-1)/\psi$ and $H(k) \approx \ln k + 0.5772$ is the harmonic number. Score in $[0,1]$: $\to 1$ for short paths (anomalous), $\to 0.5$ for normal.

**Extended Isolation Forest (EIF).** Replaces axis-aligned splits with random hyperplanes $\mathbf{n}^\top (\mathbf{x} - \mathbf{p}) = 0$; mitigates "ghost regions" of low score along feature axes.

### One-class SVM

For a feature map $\phi$ into RKHS, solve

$$\min_{\mathbf{w}, \rho, \boldsymbol{\xi}} \tfrac{1}{2} \|\mathbf{w}\|^2 + \frac{1}{\nu n} \sum_{i=1}^n \xi_i - \rho$$

$$\text{s.t. } \mathbf{w}^\top \phi(\mathbf{x}_i) \ge \rho - \xi_i, \quad \xi_i \ge 0.$$

The parameter $\nu \in (0,1]$ is an upper bound on the fraction of training outliers and a lower bound on the fraction of support vectors. Decision function: $f(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x}) - \rho$; score = $-f$. Typically with RBF kernel $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x}-\mathbf{y}\|^2)$.

### FinBERT (transformer sentiment)

BERT-base backbone: 12 transformer encoder layers, hidden size $h = 768$, 12 attention heads, FFN intermediate size 3072, ~110M parameters. Input is a tokenized sentence prepended with `[CLS]`. After 12 layers,

$$\mathbf{H} \in \mathbb{R}^{L \times 768},$$

with $\mathbf{H}_0$ the contextual representation of `[CLS]`. Classification head:

$$\mathbf{p} = \text{softmax}(\mathbf{W}_c \mathbf{H}_0 + \mathbf{b}_c) \in \Delta^3$$

over labels $\{\text{positive}, \text{neutral}, \text{negative}\}$. Loss = cross-entropy on the Financial PhraseBank (Malo et al., 2014) — ~5000 sentences from financial news annotated by domain experts. Reported test accuracy ~85% (Araci, 2019).

**Sentence score for trading.** Convert to scalar:

$$\text{sent}(\text{text}) = p_\text{pos} - p_\text{neg} \in [-1, 1].$$

Aggregate across multiple sentences/articles for ticker $t$ on day $d$ using weighted average:

$$S_{t,d} = \frac{\sum_k w_k \cdot \text{sent}(\text{article}_k)}{\sum_k w_k},$$

with weights $w_k$ from recency exponential decay $e^{-\lambda \Delta t_k}$ and source reliability multipliers.

## Intuition and what this model addresses

**Two distinct jobs:**

1. **Numerical anomaly detection.** Catch bad inputs (corrupted ticks, stale prices, fat-finger orders, data-vendor outages), adverse-selection patterns (toxic flow signature in microstructure features), and outlier training rows that distort ML models. Autoencoders capture multivariate dependencies and flag points outside the typical manifold; isolation forests are model-free and fast; one-class SVM works well in low dimension.

2. **News sentiment.** Headlines move stocks. Aggregated sentiment is a stale-but-real alpha feature. FinBERT is a finance-tuned BERT that outperforms general-domain sentiment on earnings-call transcripts and analyst reports. Important: the signal half-life on a typical macro headline is **10–30 minutes**, so inference latency and data-feed latency dominate the deployment design — slow daily sentiment is more useful as a slow-moving feature (sector tone) than as a fast trade trigger.

**Why ensemble.** Autoencoder catches subtle non-linear deviations from manifold structure; iForest catches feature-axis outliers cheaply; one-class SVM catches density-based outliers in low-d projections. Use the union (or a calibrated score combination) for monitoring.

## Architecture / model design

### Numerical anomaly autoencoder

```
Input x in R^p  (p ~ 50 features: returns, vol, volume, spread proxies)
  |
  v
Linear(p -> 64) + BN + ELU + Dropout(0.1)
  |
  v
Linear(64 -> 32) + BN + ELU + Dropout(0.1)
  |
  v
Linear(32 -> 16) + BN + ELU      ==  z (bottleneck)
  |
  v
Linear(16 -> 32) + BN + ELU + Dropout(0.1)
  |
  v
Linear(32 -> 64) + BN + ELU + Dropout(0.1)
  |
  v
Linear(64 -> p)                  ==  x_hat
```

Symmetric, untied weights, ~25k params. Train on "clean" rows only — pre-filter the training set by removing rows whose univariate features are beyond ±5 robust z-scores.

### Isolation forest

- Number of trees $t = 100$.
- Subsample size $\psi = 256$ (theoretical optimum for unbiased path-length estimate).
- `contamination = 'auto'` (sklearn) or set manually $\approx 0.005$ for daily data.

### FinBERT pipeline

```
News article text
  -> sentence split (regex on . ! ? respecting abbreviations)
  -> WordPiece tokenize (vocab 30k)
  -> truncate to 256 tokens per sentence; prepend [CLS] append [SEP]
  -> BERT-base encoder (12 layers, h=768, heads=12)
  -> [CLS] representation H_0
  -> Linear(768 -> 3) + softmax  ==  (p_pos, p_neu, p_neg)
sentiment_scalar = p_pos - p_neg
aggregate per (ticker, date) via exp-decay weights
```

## Calibration / training approach

### Autoencoder

- **Loss.** MSE per feature, optionally re-weighted by inverse feature variance to give equal weight to all dimensions.
- **Optimizer.** Adam, learning rate $10^{-3}$, cosine schedule.
- **Batch size.** 1024.
- **Epochs.** 100, early stop on held-out reconstruction MSE.
- **Threshold tuning.** Compute $s_\text{AE}$ on a labeled validation set (rare — synthetic anomalies injected by perturbation if no labels). Set threshold at the 99.5th percentile under in-sample distribution, or to maximize F1 on injected anomalies.

### Isolation forest

- No iterative training; build trees once. Re-fit weekly on rolling window.
- Score threshold at empirical 99th percentile.

### One-class SVM

- $\nu = 0.005$ (expected anomaly rate).
- $\gamma = 1 / (p \cdot \text{Var}(\mathbf{X}))$ (sklearn `scale`).
- Re-fit weekly; quadratic in $n$, so subsample to $n \le 10000$.

### FinBERT

- **Pretrained checkpoint.** ProsusAI/finbert from Hugging Face — no further training needed unless adapting to a specialized corpus (e.g., crypto-specific or sector-specific).
- **Fine-tuning (optional).** AdamW, learning rate $2 \times 10^{-5}$, linear warmup over 10% of steps, weight decay $0.01$, batch size 32, 3 epochs, max sequence length 256, gradient clipping at 1.0.
- **Inference batching.** Group sentences by length bin and pad within bin; on GPU, batch 64 sentences for ~5ms/sentence latency.
- **Quantization for production.** INT8 dynamic quantization halves latency at <1% accuracy loss; ONNX export gives a 2–3x throughput gain.

## Algorithm outline

### End-to-end anomaly monitor on streaming bars

```
TRAIN:
  Pull 2 years of clean (ticker, date) rows.
  Build feature matrix X in R^{N x p} (returns, vol, volume, spread proxies, range).
  Standardize per feature: median/MAD; clip at +/-5.
  Train autoencoder AE on X.
  Train Isolation Forest IF on X.
  Train One-class SVM OCS on a 10k random subsample of X.
  Compute in-sample scores s_AE, s_IF, s_OCS; store the 99.5%, 99%, 99% quantiles q*.

INFERENCE per new row x_t:
  Compute scores s_AE(x_t), s_IF(x_t), s_OCS(x_t).
  Per-detector flag f_d = 1{s_d(x_t) > q_d}.
  Combined flag F = (f_AE + f_IF + f_OCS) >= 2  (majority vote).
  If F == 1:
     log alert: feature attribution = top-3 j with largest |x_{t,j} - AE(x_t)_j|
     route to monitoring; do NOT use this row to train downstream models.
  Else:
     pass through.
```

### FinBERT sentiment pipeline

```
INPUT: ticker t, date d, news list news = yf.Ticker(t).news
OUTPUT: scalar sentiment S_{t,d}
TOKENIZER = AutoTokenizer.from_pretrained('ProsusAI/finbert')
MODEL     = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').eval()

scores = []
For article in news:
  body = article['title'] + '. ' + article.get('summary', '')
  sentences = split_sentences(body)
  inputs = TOKENIZER(sentences, padding=True, truncation=True,
                     max_length=256, return_tensors='pt')
  with torch.no_grad():
     logits = MODEL(**inputs).logits      # (n_sent, 3)
     probs  = softmax(logits, dim=-1)
  sent_scalar_per_sentence = probs[:,0] - probs[:,2]   # pos - neg  (label order: pos, neg, neu)
  article_sent = mean(sent_scalar_per_sentence)
  age_hours    = (now - article['providerPublishTime']) / 3600
  weight       = exp(-age_hours / 24)
  scores.append((article_sent, weight))

S_{t,d} = sum(s*w for s,w in scores) / max(sum(w for _,w in scores), eps)
```

(Verify label index order at runtime — FinBERT label order is `[positive, negative, neutral]`.)

## yfinance data requirements and feature engineering

### Numerical anomaly inputs

| Feature | yfinance source | Notes |
|---|---|---|
| 1-day log return | `Close` | Catch fat-finger ticks (\|r\| > 5 z-scores) |
| Intraday range | `(High - Low) / Close` | Stale-price detection |
| Open-close gap | `(Open - prev Close)/prev Close` | Pre-market news effect |
| Log volume | `Volume` | Volume spike detection |
| Volume / 20d MA | derived | Relative volume |
| Realized vol (20d) | derived from `Close` | Regime indicator |
| Drift from MA | $(C_t - \text{MA}_{50}(C_t))/\text{MA}_{50}$ | Mean-reversion residual |
| Beta-residual | $r_t - \hat{\beta}_t r_{m,t}$ | After-beta idiosyncratic move |
| Cross-sectional rank of return | rank/N | Catch universe-wide events |

For data-quality checks specifically: also check `Adj Close` consistency, missing values, repeated values (sign of stale feed), and weekday alignment (no Saturday bars).

### News sentiment inputs

- `yf.Ticker(t).news` returns recent news items with fields `title`, `summary` (sometimes), `link`, `publisher`, `providerPublishTime`.
- Limitations: yfinance news is shallow (no full article body) and rate-limited; for serious sentiment work pair with a dedicated news vendor (Benzinga, FMP, NewsAPI) and use yfinance news only as backup or for low-frequency aggregation.
- Sentiment feature per (ticker, date): $S_{t,d}$ scalar plus volume $V^{\text{news}}_{t,d}$ (article count) — combined feature $S_{t,d} \cdot \log(1 + V^{\text{news}}_{t,d})$.
- Sector tone: average $S_{t,d}$ across tickers in the same sector — sometimes more useful than ticker-level due to noise.

## Refit frequency

- **Autoencoder.** Monthly retrain on past 24 months. Daily incremental fine-tune of 1 epoch on the latest 30 days (set learning rate $10^{-4}$).
- **Isolation forest, OCSVM.** Weekly retrain on past 12 months.
- **Score thresholds.** Recompute quantiles daily on the trailing 90-day score history; thresholds should track the regime.
- **FinBERT.** No retrain unless you fine-tune on a proprietary labeled corpus; refresh checkpoint when ProsusAI releases an update. Recompute sentiment scores per news item once at publish time; cache.

## Validation and diagnostics

**Anomaly detector validation.**
- **Injected anomalies.** Hold out a clean set; inject synthetic anomalies (e.g., multiply random returns by 10, add Gaussian outliers, swap features between rows). Measure precision/recall, PR-AUC.
- **Per-feature attribution.** For each flagged row, check that the top reconstruction-residual features correspond to the actual perturbation in the synthetic test.
- **Score stability.** Distribution of $s_\text{AE}$ on the validation set should be stationary across time when the input distribution is stable.
- **Calibration.** Plot in-sample score CDF; threshold should sit on a steep part of the tail to be informative.
- **False positives.** Live track flagged rows; classify each manually (true outlier, regime change, data corruption, model drift). Drift of false-positive rate triggers retrain.

**FinBERT validation.**
- **Sentence-level held-out accuracy** on Financial PhraseBank held-out split — expect ~85%.
- **Event-study correlation.** Compute $S_{t,d}$ from news on day $d$; correlate with $r_{t, d+1}$. Expect small but positive correlation (information-coefficient 0.02–0.05 on liquid names; higher on event days).
- **Coverage.** Fraction of (ticker, day) cells with at least one article — sparse for small caps.
- **Latency profile.** Measure end-to-end latency from `providerPublishTime` to feature in the model; aim for < 5 minutes to retain signal.
- **Stability of label order.** Probe the model with known positive/negative example sentences at startup to verify the label index ordering matches downstream code.

## Connections to other models

- **SAE+MLP (Model 10).** The supervised autoencoder's reconstruction error is itself an anomaly score. Pre-train the anomaly AE and the SAE jointly with shared encoder weights for efficient compute.
- **GBDT (Model 11).** Flagged rows are excluded from training; sentiment score $S_{t,d}$ becomes a feature.
- **Bayesian (Model 12).** Anomaly scores can downweight observations in the Bayesian likelihood (robust regression via mixture of Gaussians with anomaly component).
- **Online learning (Model 13).** Sentiment-based signal becomes a Hedge expert; FTRL features can include FinBERT scores hashed by ticker.
- **GARCH (Layer 3).** Realized-vol anomaly score complements GARCH's likelihood-based outlier detection.
- **Execution (Layer 7).** Real-time anomaly alerts halt routing on fat-finger child orders; sentiment spikes pause aggressive execution.

## Limitations and failure modes

1. **AE memorizes training anomalies.** If the training set has not been filtered for outliers, the AE learns to reconstruct them well and they no longer score high. Pre-filter aggressively.
2. **Isolation forest axis bias.** Standard iForest under-scores anomalies along diagonal directions; use EIF.
3. **One-class SVM kernel tuning.** $\gamma$ too large = overfit (everything looks normal); too small = underfit. Cross-validate on injected anomalies.
4. **Threshold instability.** Empirical quantile thresholds drift with regime; pair with absolute "hard rule" thresholds (e.g., \|r\| > 20%) for safety.
5. **FinBERT domain shift.** Trained on financial *phrases*; performance on tweets, Reddit, or sell-side jargon may degrade. Validate on your actual news source.
6. **Causal direction.** Sentiment $S_{t,d}$ is correlated with $r_{t,d}$ contemporaneously — careful not to use day-$d$ sentiment to predict day-$d$ return without timestamp guarantees.
7. **Stale yfinance news cache.** `yf.Ticker(t).news` returns a small set of recent items; check `providerPublishTime` to avoid re-using old articles.
8. **Label-order surprises.** FinBERT's `id2label` mapping is sometimes `{0:'positive', 1:'negative', 2:'neutral'}` rather than alphabetical — read it from the model config explicitly.
9. **Compute cost at scale.** FinBERT on every news item across S&P 500 is ~1 GPU-hour/day; quantize and batch aggressively, or fall back to a distilled student (FinDistilBERT).
10. **Regulatory.** Sentiment-derived signals from premium news may be governed by data-license terms — yfinance news is generally permissive but verify provider terms.

## References

1. An, J., & Cho, S. (2015). *Variational autoencoder based anomaly detection using reconstruction probability.* SNU Tech Report.
2. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* ICDM.
3. Hariri, S., Kind, M. C., & Brunner, R. J. (2019). *Extended Isolation Forest.* IEEE TKDE.
4. Schölkopf, B., et al. (2001). *Estimating the support of a high-dimensional distribution.* Neural Computation. (One-class SVM)
5. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of deep bidirectional transformers.* NAACL.
6. Araci, D. (2019). *FinBERT: Financial sentiment analysis with pre-trained language models.* arXiv:1908.10063.
7. Malo, P., et al. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST.
8. Kingma, D. P., & Welling, M. (2014). *Auto-encoding variational Bayes.* ICLR.
9. Tetlock, P. C. (2007). *Giving content to investor sentiment: The role of media in the stock market.* J. Finance.
10. Ke, Z., Kelly, B. T., & Xiu, D. (2019). *Predicting returns with text data.* NBER WP 26186.
