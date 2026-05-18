# Model 14 — Anomaly Detection + FinBERT

Layer-4 signal/ML model that fuses **numerical anomaly detection** (three
detectors voting) with **transformer-based news sentiment** on top of
yfinance data.

Full specification: [`models/layer4_signals_ml/14_anomaly_detection_finbert.md`](../../../models/layer4_signals_ml/14_anomaly_detection_finbert.md).

## What this implementation does

Two jobs in one model:

1. **Numerical anomaly ensemble.** Per-row score from three independent
   detectors, combined by majority vote (≥ 2 of 3 fire):
   - `s_AE`  — squared PCA reconstruction error (linear autoencoder).
   - `s_iF`  — Liu/Ting/Zhou isolation-forest score with the `c(ψ)`
     path-length normalizer.
   - `s_OC`  — shrinkage Mahalanobis distance (the Gaussian
     one-class density baseline the spec's OCSVM-with-RBF approximates).

   Thresholds are the in-sample `1 − contamination` empirical quantile
   per detector. Each alert carries top-feature attribution from the
   PCA reconstruction residuals.

2. **FinBERT sentiment.** Per (ticker, day) scalar `S ∈ [-1, 1]` from
   `p_pos − p_neg`, aggregated across articles with the spec's
   `exp(-Δt / decay_hours)` recency weights. The runtime `label2id` is
   read from the model config so the documented "label-order surprise"
   (`{0: positive, 1: negative, 2: neutral}` vs. alphabetical) is
   handled automatically.

The model is registered as `"anomaly_detection_finbert"` (layer 4,
monthly refit).

## Public API

```python
from src.models.anomaly_detection_finbert import (
    AnomalyDetectionFinBERT,
    AnomalyConfig,
    SentimentConfig,
)
```

Classes:

- `AnomalyDetectionFinBERT(BaseModel)` — main orchestration shell.
- `AnomalyConfig`, `SentimentConfig` — hyperparameter bundles.
- `AnomalyInputs`, `AnomalyFit`, `DetectorFit`, `AnomalyAlert`,
  `SentimentScore`, `ITreeRecord` — model-internal dataclasses.

Public helpers (pure numpy / pandas; importable for ad-hoc analysis):

- Feature engineering: `build_anomaly_feature_panel`,
  `cross_sectional_rank`, `log_return_1d`, `intraday_range`,
  `open_close_gap`, `log_volume`, `vol_ratio`, `realized_vol`,
  `ma_drift`, `abs_return_zscore`.
- Preprocessing: `median_mad_standardize`, `impute_and_clip`.
- Detectors: `pca_fit`, `pca_reconstruction_score`,
  `pca_residual_attribution`, `fit_iforest`, `iforest_score`, `c_psi`,
  `fit_mahalanobis`, `mahalanobis_score`.
- Ensemble glue: `score_all_detectors`, `threshold_from_quantile`,
  `majority_vote_flags`, `top_feature_attribution`.
- Sentiment math: `sentiment_scalar`, `exp_decay_weight`,
  `weighted_sentiment`, `split_sentences`.

## Usage example

```python
from src.core.data_provider import YFinanceProvider
from src.models.anomaly_detection_finbert import (
    AnomalyDetectionFinBERT,
    AnomalyConfig,
    SentimentConfig,
)

model = AnomalyDetectionFinBERT(
    universe=["SPY", "QQQ", "IWM"],
    period="2y",
    anomaly_config=AnomalyConfig(contamination=0.005),
    sentiment_config=SentimentConfig(decay_hours=24.0),
)
provider = YFinanceProvider()
data = model.fetch_data(provider)        # pulls bars + news
model.calibrate(data)                    # fits AE + iForest + Mahalanobis
signal = model.predict(data)             # Signal: direction from sentiment
diag = model.validate(data)              # injected-anomaly recall, etc.
```

## Refit frequency

`monthly` — autoencoder retrained on past 24 months per the spec, and
iForest / Mahalanobis trained weekly in production. The `BaseModel`
class attribute records the slowest cadence; faster refits are the
orchestration layer's job. Thresholds can be cheaply recomputed from
`AnomalyFit.in_sample_scores` without re-training.

## Known limitations

1. **Optional heavy backends.** This implementation runs end-to-end with
   only numpy / pandas / yfinance:
   - Autoencoder is a *linear* AE (PCA reconstruction). The deep
     BatchNorm / ELU / Dropout AE in the spec needs `torch`. A future
     extension can swap `pca_*` for a torch-backed deep AE without
     changing the public API.
   - The OCSVM is approximated by a shrinkage Mahalanobis distance.
     The spec's exact OCSVM-with-RBF needs `sklearn` and is not
     wired in here.
2. **FinBERT requires `transformers`.** Without it, sentiment falls
   back to a deterministic stub returning 0.0 (`backend="stub"`).
   Production must install `transformers` + `torch` and verify the
   `label2id` ordering against the runtime checkpoint.
3. **iForest axis bias.** Standard iForest under-scores anomalies along
   diagonal directions (spec limitation #2). Extended Isolation Forest
   would help; not implemented.
4. **AE memorization of training anomalies.** PCA only sees the dominant
   covariance; aggressive pre-filtering of the training set is the
   user's responsibility per spec.
5. **Empirical quantile thresholds drift with regime.** The spec
   recommends pairing them with absolute "hard rule" thresholds
   (e.g., `|r| > 20%`); only the empirical thresholds are wired here.
6. **Causal direction.** Same-day sentiment is correlated with same-day
   returns — callers using `S_{t,d}` as an alpha feature must guarantee
   timestamps for the news come before any trading-decision cutoff.
7. **yfinance news payload is shallow.** `fetch_news` returns title +
   optional summary; full article bodies need a dedicated news vendor
   (Benzinga, FMP, NewsAPI).

See the spec doc's "Limitations and failure modes" section for the
full list.
