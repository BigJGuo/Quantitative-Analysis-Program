# Supervised Autoencoder + MLP (Jane Street 2021 Kaggle Winner)

> Tabular deep-learning architecture jointly trained with denoising reconstruction and binary classification heads. Layer 4 — Signals & ML. Primary use: trade-selection alpha (decide whether to take a given trade) from a wide, highly-correlated feature vector.

## Mathematical formulation

Let $\mathbf{x} \in \mathbb{R}^{p}$ be the raw feature vector ($p \approx 130$ in the original Kaggle setup) and $\mathbf{y} \in \{0,1\}^{K}$ a vector of $K$ binarized targets derived from forward returns. The supervised autoencoder (SAE + MLP) has four sub-modules: an input noise layer, an encoder $E_\phi$, a decoder $D_\psi$, and a classifier MLP $C_\omega$.

**Noise injection.** A Gaussian "swap noise" or additive noise is applied during training:

$$\tilde{\mathbf{x}} = \mathbf{x} + \mathbf{n}, \qquad \mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}), \quad \sigma \approx 0.03.$$

**Encoder.** A stack of fully-connected layers reduces dimension to $d \ll p$ (typically $d = 64$):

$$\mathbf{h} = E_\phi(\tilde{\mathbf{x}}) = g_L \circ g_{L-1} \circ \dots \circ g_1(\tilde{\mathbf{x}}), \qquad g_\ell(\mathbf{z}) = \text{SiLU}(\text{BN}(\mathbf{W}_\ell \mathbf{z} + \mathbf{b}_\ell)).$$

**Decoder.** Reconstructs the (clean) input from $\mathbf{h}$:

$$\hat{\mathbf{x}} = D_\psi(\mathbf{h}), \qquad \mathcal{L}_\text{recon} = \frac{1}{p} \|\hat{\mathbf{x}} - \mathbf{x}\|_2^2.$$

**Auxiliary classifier on $\mathbf{h}$.** A linear head $C_{\omega_1}$ predicts the targets from the bottleneck alone:

$$\hat{\mathbf{y}}^{(\text{aux})} = \sigma\!\left( C_{\omega_1}(\mathbf{h}) \right).$$

**Main classifier on $[\mathbf{x}; \mathbf{h}]$.** The full MLP head $C_{\omega_2}$ uses the concatenation of the raw features and the encoded bottleneck:

$$\hat{\mathbf{y}} = \sigma\!\left( C_{\omega_2}([\mathbf{x} ; \mathbf{h}]) \right) \in (0,1)^{K}.$$

**Total loss.** Joint optimization with two cross-entropy heads and a reconstruction term:

$$\mathcal{L} = \mathcal{L}_\text{recon}(\hat{\mathbf{x}}, \mathbf{x}) + \alpha_1 \, \mathcal{L}_\text{BCE}(\hat{\mathbf{y}}^{(\text{aux})}, \mathbf{y}) + \alpha_2 \, \mathcal{L}_\text{BCE}(\hat{\mathbf{y}}, \mathbf{y}),$$

where binary cross-entropy is

$$\mathcal{L}_\text{BCE}(\hat{\mathbf{y}}, \mathbf{y}) = -\frac{1}{K}\sum_{k=1}^K \big[ y_k \log \hat{y}_k + (1-y_k) \log(1-\hat{y}_k) \big].$$

Typical weighting: $\alpha_1, \alpha_2 \in [0.5, 2]$.

**Action rule.** At inference, average ensemble outputs $\bar{y}_k$ across seeds (median or mean) and trade if any binarized horizon exceeds $\tau \approx 0.5$:

$$\text{take trade} \iff \max_k \bar{y}_k > \tau.$$

## Intuition and what this model addresses

Two structural facts about the Kaggle 2021 dataset drove the architecture choice:

1. **Heavy feature correlation.** Anonymized features were known to share linear and non-linear structure. An autoencoder bottleneck $\mathbf{h}$ exploits this — it discovers a compact latent that explains shared variance, denoising idiosyncratic noise in any single feature.
2. **A margin-based target.** The utility score $u = \sum_i w_i r_i \, \mathbb{1}\{a_i = 1\}$ rewards correctly *skipping* low-quality trades. A classification objective on binarized future returns aligns with this discrete action better than direct regression on the noisy continuous responder.

The "supervised" twist is critical: a vanilla autoencoder learns features that minimize reconstruction error, which need not correlate with the *predictive* axes of $\mathbf{x}$. Adding a classification loss on $\mathbf{h}$ steers the encoder toward a representation that is jointly compressible *and* discriminative. This is the same logic as Le et al. (2018) supervised autoencoders and aligns with semi-supervised representation learning.

Why a neural network — and not the usual Kaggle GBDT winners — won is partly explained by these two factors plus a third: the action surface is highly *smooth* in the latent space, and MLPs interpolate smoothly while trees produce piecewise-constant outputs that fight the noise.

## Architecture / model design

A canonical configuration (faithful to the 2021 winning entry):

```
Input x in R^{130}
  |
  +-- Gaussian noise sigma=0.03  -->  x_tilde
                                         |
                                         v
                              Linear(130 -> 96) + BN + SiLU + Dropout(0.2)
                                         |
                                         v
                              Linear(96 -> 96)  + BN + SiLU + Dropout(0.2)
                                         |
                                         v
                              Linear(96 -> 64)  + BN + SiLU                 == h (bottleneck)
                                         |
                 +-----------------------+--------------------------+
                 |                                                  |
            Decoder D_psi                                  Aux classifier on h
            Linear(64 -> 96) + BN + SiLU                   Linear(64 -> K=3) + sigmoid
            Linear(96 -> 130)             == x_hat                  |
                                                                    |
                                                                    v
                                               Concatenate [x ; h] in R^{130+64}
                                                                    |
                                                                    v
                                            Linear(194 -> 256) + BN + SiLU + Dropout(0.3)
                                                                    |
                                                                    v
                                            Linear(256 -> 256) + BN + SiLU + Dropout(0.3)
                                                                    |
                                                                    v
                                            Linear(256 -> K=3) + sigmoid   == y_hat
```

Activation choice: **SiLU (Swish)** is common but ReLU/GELU work; the original used Swish. BatchNorm before activation. Dropout rates: 0.2–0.5 on the wider MLP layers, lower (0.1–0.2) inside the encoder. The decoder is *not* tied to the encoder weights (untied decoder, in contrast to a symmetric stacked autoencoder).

Parameter count is small by modern standards (~150k), so training is cheap.

## Calibration / training approach

- **Optimizer.** Adam with learning rate $\eta = 10^{-3}$ initially, cosine annealing to $10^{-5}$ over training.
- **Batch size.** 4096 (large batches are important for stable BatchNorm on tabular data).
- **Epochs.** 100–200; early stopping on validation utility.
- **Weight decay.** $\lambda = 10^{-5}$ on all linear layers.
- **Loss weights.** $\alpha_1 = 1$, $\alpha_2 = 1$ (reconstruction weight implicit at 1; some implementations scale recon down to 0.3–0.5 because it dominates early).
- **Targets.** From the five horizon responders $r^{(1)}, \dots, r^{(5)}$, build three binary labels: e.g., $y_1 = \mathbb{1}\{r^{(1)} > 0\}$, $y_2 = \mathbb{1}\{\text{med}(r^{(1..5)}) > 0\}$, $y_3 = \mathbb{1}\{\sum_h r^{(h)} > 0\}$.
- **Validation scheme.** Purged time-series K-fold (5 folds with embargo $\approx 1$ day). No random K-fold — that leaks across time.
- **Seeds.** Train 5–10 seeds with different random initializations; blend predictions by **median** (not mean — more robust to seed-level failures).
- **Sample weighting.** Each row has a known weight $w_i$; include in classification loss as $\mathcal{L}_\text{BCE} \cdot w_i$ during training.

Regularization stack: noise injection + dropout + weight decay + BatchNorm + ensemble blend. Each tool addresses a different overfitting mode.

## Algorithm outline

```
INPUT: feature panel X (N x p), targets Y (N x K), weights w (N), seeds S = [s1..s10]
OUTPUT: ensemble of trained models, blend rule

1.  Build features:
      X_raw <- yfinance panel (returns, vol, rolling stats, fundamentals)
      X     <- standardize per-feature: x_j <- (x_j - median_j) / MAD_j
      Impute NaN with feature median; add missing-indicator columns.

2.  Build binarized targets:
      For each row i: y_{i,1} = 1{r1d_i > 0},
                      y_{i,2} = 1{median(r1d..r20d) > 0},
                      y_{i,3} = 1{sum(r1d..r5d) > 0}.

3.  Purged 5-fold split on date_id with embargo of 1 trading day.

4.  For each fold f in 1..5:
      For each seed s in S:
         set torch seed, init model M_{f,s}
         For epoch in 1..E:
            For minibatch (x_b, y_b, w_b):
               x_tilde <- x_b + N(0, sigma^2 I)
               h       <- E_phi(x_tilde)
               x_hat   <- D_psi(h)
               y_aux   <- sigmoid(C_omega1(h))
               y_hat   <- sigmoid(C_omega2([x_b ; h]))
               L_rec   <- MSE(x_hat, x_b)
               L_aux   <- weighted_BCE(y_aux, y_b, w_b)
               L_main  <- weighted_BCE(y_hat, y_b, w_b)
               L       <- L_rec + alpha1 * L_aux + alpha2 * L_main
               L.backward(); opt.step(); opt.zero_grad()
            Evaluate utility on val fold; early-stop on plateau.

5.  Inference on new row x*:
      For each (f, s):  yhat_{f,s}(x*) = M_{f,s}(x*)
      blend(x*) = median over (f, s) of yhat_{f,s}(x*)
      action(x*) = 1 if max_k blend_k(x*) > tau else 0.
```

## yfinance data requirements and feature engineering

Build a per-(ticker, date) panel of ~80–150 features.

**Raw yfinance fields:**

- `yf.download(tickers, period, interval='1d')` → OHLCV: `Open, High, Low, Close, Adj Close, Volume`.
- `yf.Ticker(t).info` → market cap, sector, P/E, beta (static or slowly varying).
- `yf.Ticker(t).financials`, `.balance_sheet`, `.cashflow` → fundamental aggregates.
- `yf.Ticker(t).options` chains → IV proxies if needed.

**Feature transforms** (compute per ticker, then standardize cross-sectionally per date):

| Family | Feature | Formula |
|---|---|---|
| Returns | $r^{(k)}_t$ | $\log(P_t / P_{t-k})$ for $k \in \{1, 5, 10, 20, 60\}$ |
| Volatility | $\hat{\sigma}^{(k)}_t$ | rolling std of $r^{(1)}$ over window $k$ |
| Realized vol | $\text{RV}_t$ | $\sqrt{\sum_{s=t-k}^{t} (r^{(1)}_s)^2}$ |
| Volume | $\tilde{V}_t$ | $V_t / \text{MA}_{20}(V_t)$ |
| High-low range | $\text{HL}_t$ | $(H_t - L_t) / C_t$ |
| Momentum | $\text{MOM}_t^{(k)}$ | $r^{(k)}_t$ z-scored cross-sectionally |
| Reversal | $\text{REV}_t$ | $-r^{(5)}_t$ |
| Liquidity | $\text{ILLIQ}_t$ | $|r^{(1)}_t| / (V_t \cdot C_t)$ (Amihud) |
| Cross-sectional rank | $\text{rank}_t(f)$ | rank of feature $f$ across universe at date $t$ |
| Sector dummies | one-hot | from `info['sector']` |

Per-feature preprocessing: winsorize at 1%/99%, then standardize with median/MAD (robust to fat tails).

## Refit frequency

- **Full retrain:** weekly, on a sliding ~5-year window.
- **Fine-tune (1–3 epochs only):** daily, after the close, on the most recent 60 days.
- **Ensemble refresh:** rotate one seed out and one in each week to keep diversity without drift.
- **Trigger-based retrain:** if 20-day rolling Sharpe of the live signal drops below the 5th percentile of its in-sample distribution, force a full retrain.

## Validation and diagnostics

- **Primary metric.** Sharpe-weighted utility on held-out folds: $u = \sum_i w_i r_i a_i / \sqrt{\sum_i w_i^2}$.
- **Calibration.** Reliability diagram of $\hat{y}$ vs empirical hit rate; Brier score per horizon.
- **Stability.** Variance of predictions across seeds (high variance row → low confidence).
- **Latent inspection.** Plot $\mathbf{h}$ via UMAP, colored by sector and by realized return — clusters should align with regimes.
- **Decoder sanity.** Per-feature reconstruction error; spikes flag drift in input distribution.
- **Permutation importance.** Shuffle each input feature $j$, measure utility drop $\Delta u_j$.
- **Saliency.** Integrated gradients on $\hat{y}$ w.r.t. $\mathbf{x}$ to attribute trade decisions.

## Connections to other models

- **GBDT/Transformer ensembles (Model 11).** SAE+MLP outputs become an input feature or an ensemble member alongside LightGBM. Median blend across model families is robust.
- **Bayesian hierarchical (Model 12).** The bottleneck $\mathbf{h}$ can serve as a low-rank factor for hierarchical regression — partial-pool across tickers in the latent space.
- **Online learning (Model 13).** The classifier head probability $\hat{y}$ becomes one of $N$ "experts" combined by Hedge for the production trading rule.
- **Anomaly detection (Model 14).** Reconstruction error $\|\hat{\mathbf{x}} - \mathbf{x}\|^2$ is itself a powerful anomaly score — fat-finger inputs cause spikes here.
- **GARCH / HAR-RV (Layer 3).** Volatility forecasts from these models feed in as features $\hat{\sigma}^{(k)}_t$.

## Limitations and failure modes

1. **Regime change.** A retrain window that doesn't span the new regime causes the latent to drift; reconstruction error spikes are an early warning.
2. **Label leakage via target construction.** If targets use overlapping forward windows without embargo, validation utility is inflated.
3. **BatchNorm at inference.** Live single-sample inference needs batch statistics frozen — verify `model.eval()` is called.
4. **Seed collapse.** Two seeds drifting to the same local optimum kills ensemble benefit; monitor pairwise prediction correlation.
5. **Class imbalance.** If the universe is biased toward up days, $\hat{y}$ is biased high; check base rates per fold.
6. **yfinance survivorship bias.** The Tickers API returns only currently-listed names — train on a point-in-time universe to avoid look-ahead.
7. **Noise sigma sensitivity.** $\sigma$ too large destroys signal; too small offers no regularization. Tune by validation utility.
8. **Bottleneck collapse.** If $\alpha_2$ dominates, the encoder ignores reconstruction and $\mathbf{h}$ degenerates to a classifier embedding — lose representational benefit.

## References

1. Le, L., Patterson, A., & White, M. (2018). *Supervised autoencoders: Improving generalization performance with unsupervised regularizers.* NeurIPS.
2. Zhang, Y. (2021). *1st place solution write-up.* Jane Street Market Prediction, Kaggle.
3. Vincent, P. et al. (2010). *Stacked denoising autoencoders.* JMLR.
4. Ioffe, S., & Szegedy, C. (2015). *Batch normalization.* ICML.
5. Ramachandran, P., Zoph, B., & Le, Q. (2017). *Swish: a self-gated activation function.* arXiv:1710.05941.
6. de Prado, M. L. (2018). *Advances in Financial Machine Learning,* chapters on purged K-fold cross-validation.
7. Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning.* Review of Financial Studies, 33(5).
