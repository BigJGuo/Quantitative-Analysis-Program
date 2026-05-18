"""Pure numpy / pandas math layer for Model 14.

Nothing in this module imports torch, sklearn, or transformers. The three
anomaly detectors specified in the spec — autoencoder, isolation forest,
one-class SVM — are implemented here as their robust pure-numpy analogues:

- ``pca_*``           — PCA reconstruction error (linear autoencoder).
- ``iforest_*``       — Isolation forest with the spec's `c(ψ)` normalizer.
- ``mahalanobis_*``   — Mahalanobis distance with shrinkage (one-class
                        Gaussian density score; the OCSVM-with-RBF baseline).

Heavier backends (deep AE, sklearn OCSVM) can wrap these as drop-in
replacements; the public output shape is the same.

Sentiment helpers below are independent of any model checkpoint — they
take pre-computed (p_pos, p_neu, p_neg) probability triples and apply the
spec's scalar / exp-decay aggregation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.models.anomaly_detection_finbert.types import (
    DetectorFit,
    ITreeRecord,
)

_MAD_FLOOR: float = 1e-12
_EULER_GAMMA: float = 0.5772156649015329


# ---------------------------------------------------------------------------
# Feature engineering — spec's "Numerical anomaly inputs" table
# ---------------------------------------------------------------------------


def log_return_1d(close: pd.Series) -> pd.Series:
    """One-day log return `log(C_t / C_{t-1})`."""

    if close.empty:
        return close.astype(float)
    s = close.astype(float).sort_index()
    return cast(pd.Series, pd.Series(np.log(s.to_numpy()), index=s.index).diff(1))


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """`(High - Low) / Close` — stale-price / fat-finger range proxy."""

    if close.empty:
        return close.astype(float)
    denom = close.astype(float).replace(0.0, np.nan)
    return cast(pd.Series, (high.astype(float) - low.astype(float)) / denom)


def open_close_gap(open_: pd.Series, prev_close: pd.Series) -> pd.Series:
    """`(Open_t - Close_{t-1}) / Close_{t-1}` — overnight gap."""

    if open_.empty:
        return open_.astype(float)
    denom = prev_close.astype(float).replace(0.0, np.nan)
    return cast(pd.Series, (open_.astype(float) - prev_close.astype(float)) / denom)


def log_volume(volume: pd.Series) -> pd.Series:
    """`log(1 + V)` — robust to zero-volume bars."""

    v = volume.astype(float).clip(lower=0.0)
    return pd.Series(np.log1p(v.to_numpy()), index=v.index)


def vol_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume divided by its trailing simple moving average."""

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    v = volume.astype(float)
    ma = v.rolling(window=window, min_periods=window).mean()
    return v / ma.replace(0.0, np.nan)


def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling-window realized vol: `sqrt(sum r^2)` over `window` returns."""

    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    r = returns.astype(float)
    squared = r * r
    summed = squared.rolling(window=window, min_periods=window).sum()
    return pd.Series(np.sqrt(summed.to_numpy()), index=summed.index)


def ma_drift(close: pd.Series, window: int = 50) -> pd.Series:
    """`(C_t - MA_w(C)) / MA_w(C)` — mean-reversion residual proxy."""

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    c = close.astype(float)
    ma = c.rolling(window=window, min_periods=window).mean()
    return (c - ma) / ma.replace(0.0, np.nan)


def abs_return_zscore(returns: pd.Series, window: int = 60) -> pd.Series:
    """`|r_t| / rolling_std(r, window)` — robust outlier z-score."""

    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    r = returns.astype(float)
    std = r.rolling(window=window, min_periods=window).std(ddof=1)
    return r.abs() / std.replace(0.0, np.nan)


def build_anomaly_feature_panel(bars: pd.DataFrame) -> pd.DataFrame:
    """Build the spec's per-date feature panel for a single ticker.

    Columns (in the order documented by `DEFAULT_FEATURE_NAMES`):

    - ``log_return_1d``
    - ``intraday_range``
    - ``open_close_gap``
    - ``log_volume``
    - ``vol_ratio_20d``
    - ``realized_vol_20d``
    - ``ma_drift_50``
    - ``abs_return_zscore``

    Caller appends the cross-sectional rank in a separate pass after pooling
    rows from all tickers (it requires the universe-wide return panel).
    """

    if bars is None or bars.empty:
        raise ValueError("build_anomaly_feature_panel requires non-empty bars")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")

    close = bars["Close"].astype(float)
    high = bars["High"].astype(float)
    low = bars["Low"].astype(float)
    open_ = bars["Open"].astype(float)
    volume = bars["Volume"].astype(float)
    prev_close = close.shift(1)

    r1 = log_return_1d(close)
    return pd.DataFrame(
        {
            "log_return_1d": r1,
            "intraday_range": intraday_range(high, low, close),
            "open_close_gap": open_close_gap(open_, prev_close),
            "log_volume": log_volume(volume),
            "vol_ratio_20d": vol_ratio(volume, 20),
            "realized_vol_20d": realized_vol(r1, 20),
            "ma_drift_50": ma_drift(close, 50),
            "abs_return_zscore": abs_return_zscore(r1, 60),
        },
        index=bars.index,
    )


def cross_sectional_rank(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional rank of one-day returns across tickers.

    `returns_panel` has shape (T, n_tickers); the output has the same shape
    with each row replaced by its rank in `[0, 1]`. NaNs propagate.
    """

    if returns_panel.empty:
        return returns_panel.astype(float)
    # `rank` ranks within columns by default — rank within rows by transposing.
    return returns_panel.rank(axis=1, method="average", pct=True)


# ---------------------------------------------------------------------------
# Robust preprocessing — median / MAD standardization plus clip
# ---------------------------------------------------------------------------


def median_mad_standardize(
    values: np.ndarray,
    *,
    median: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column robust standardization: `(x - median) / (1.4826 * MAD)`.

    Returns ``(standardized, median, scale)``. If `median` / `scale` are
    supplied they are reused (test-time standardization with train stats);
    otherwise both are computed column-wise on the input.

    `scale = 1.4826 * MAD` — Gaussian-consistent so the output is comparable
    to a z-score on Gaussian-ish features. A floor of 1.0 is substituted when
    MAD is below `_MAD_FLOOR` so constant columns standardize to 0.
    """

    if values.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {values.shape}")
    if median is None:
        median = np.nanmedian(values, axis=0)
    if scale is None:
        mad = np.nanmedian(np.abs(values - median), axis=0)
        scale = 1.4826 * mad
    safe_scale = np.where(scale > _MAD_FLOOR, scale, 1.0)
    out = (values - median) / safe_scale
    return out.astype(float), median.astype(float), safe_scale.astype(float)


def impute_and_clip(values: np.ndarray, clip_z: float) -> np.ndarray:
    """Replace NaN with 0 (post-standardization) and clip to `[-clip_z, clip_z]`.

    Used on the already-standardized panel: NaNs in the *original* features
    propagate through standardization, and after standardization the column
    median is 0, so 0 is a natural imputation.
    """

    if clip_z <= 0:
        raise ValueError(f"clip_z must be > 0, got {clip_z}")
    out = np.where(np.isnan(values), 0.0, values)
    return np.clip(out, -clip_z, clip_z).astype(float)


# ---------------------------------------------------------------------------
# Detector 1: PCA reconstruction error (linear autoencoder)
# ---------------------------------------------------------------------------


def pca_fit(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a PCA on `X` (N, p) and return `(mean, components, explained_var)`.

    `components` has shape `(d, p)` with rows ordered by descending
    explained variance. `explained_var` is the corresponding length-d
    eigenvalue array. Uses an SVD on the centered input (numpy only).
    """

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n, p = X.shape
    if n_components < 1 or n_components > p:
        raise ValueError(
            f"n_components must lie in [1, {p}], got {n_components}"
        )
    if n < 2:
        raise ValueError(f"PCA requires N >= 2 rows, got {n}")
    mean = X.mean(axis=0)
    centered = X - mean
    # SVD: centered = U S Vt. Components are rows of Vt.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (s**2) / max(n - 1, 1)
    components = vt[:n_components]
    return (
        mean.astype(float),
        components.astype(float),
        explained[:n_components].astype(float),
    )


def pca_reconstruct(
    X: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Return `X_hat = mean + (X - mean) @ V.T @ V` for components V (d, p)."""

    if X.ndim != 2 or X.shape[1] != mean.shape[0]:
        raise ValueError(
            f"X shape {X.shape} incompatible with mean shape {mean.shape}"
        )
    if components.shape[1] != mean.shape[0]:
        raise ValueError(
            f"components shape {components.shape} incompatible with mean "
            f"shape {mean.shape}"
        )
    centered = X - mean
    projected = centered @ components.T  # (N, d)
    reconstructed = projected @ components  # (N, p)
    return cast(np.ndarray, (reconstructed + mean).astype(float))


def pca_reconstruction_score(
    X: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """`s_AE(x) = ||x - PCA(x)||^2`. Spec's reconstruction-error score."""

    x_hat = pca_reconstruct(X, mean, components)
    diff = X - x_hat
    return cast(np.ndarray, np.sum(diff * diff, axis=1).astype(float))


def pca_residual_attribution(
    x: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Return the squared per-feature residual vector `(x - PCA(x))^2`."""

    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    x_hat = pca_reconstruct(x.reshape(1, -1), mean, components)[0]
    return cast(np.ndarray, ((x - x_hat) ** 2).astype(float))


# ---------------------------------------------------------------------------
# Detector 2: Isolation Forest (pure numpy)
# ---------------------------------------------------------------------------


def c_psi(psi: int) -> float:
    """Spec's normalizer for iForest path lengths.

    `c(ψ) = 2 * (H(ψ - 1)) - 2 * (ψ - 1) / ψ` with the harmonic number
    `H(k) ≈ ln(k) + γ`. Falls back to 1.0 for `ψ <= 1` to avoid log(0).
    """

    if psi <= 1:
        return 1.0
    harmonic = math.log(psi - 1) + _EULER_GAMMA
    return 2.0 * harmonic - 2.0 * (psi - 1) / psi


def _build_itree(
    X: np.ndarray, rng: np.random.Generator, max_depth: int
) -> ITreeRecord:
    """Build one iTree on the supplied subsample (rows already drawn).

    We walk the construction iteratively (a small stack) and emit four
    parallel arrays — left, right, feature, split — plus a `size` array
    recording how many points landed in each leaf.
    """

    n, _p = X.shape
    if n == 0:
        raise ValueError("cannot build iTree on empty input")
    feature_list: list[int] = []
    split_list: list[float] = []
    left_list: list[int] = []
    right_list: list[int] = []
    size_list: list[int] = []

    # Stack entries: (point indices, current depth, parent_id, is_left_child)
    # parent_id = -1 for the root.
    root_indices = np.arange(n)
    stack: list[tuple[np.ndarray, int, int, bool]] = [(root_indices, 0, -1, False)]

    while stack:
        idx, depth, parent_id, is_left = stack.pop()
        node_id = len(feature_list)
        # Reserve a slot.
        feature_list.append(-1)
        split_list.append(0.0)
        left_list.append(-1)
        right_list.append(-1)
        size_list.append(idx.size)

        if parent_id >= 0:
            if is_left:
                left_list[parent_id] = node_id
            else:
                right_list[parent_id] = node_id

        if depth >= max_depth or idx.size <= 1:
            continue
        subset = X[idx]
        # Choose a random axis whose values are not constant. Try a few times
        # before giving up and making this a leaf — saves wasted leaves on
        # data with many redundant columns.
        candidates = list(range(X.shape[1]))
        rng.shuffle(candidates)
        chosen = -1
        lo = hi = 0.0
        for axis in candidates:
            col = subset[:, axis]
            col_min = float(col.min())
            col_max = float(col.max())
            if col_max > col_min:
                chosen = axis
                lo, hi = col_min, col_max
                break
        if chosen < 0:
            # All columns constant — leaf.
            continue
        split_value = float(rng.uniform(lo, hi))
        feature_list[node_id] = chosen
        split_list[node_id] = split_value
        left_mask = subset[:, chosen] < split_value
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        # Push right then left so left is processed first (depth-first prefix).
        stack.append((right_idx, depth + 1, node_id, False))
        stack.append((left_idx, depth + 1, node_id, True))

    return ITreeRecord(
        feature=np.array(feature_list, dtype=int),
        split=np.array(split_list, dtype=float),
        left=np.array(left_list, dtype=int),
        right=np.array(right_list, dtype=int),
        size=np.array(size_list, dtype=int),
        max_depth=max_depth,
    )


def fit_iforest(
    X: np.ndarray,
    *,
    n_trees: int,
    psi: int,
    seed: int,
) -> tuple[ITreeRecord, ...]:
    """Train `n_trees` isolation trees on bootstrap subsamples of size `psi`.

    Returns a tuple of `ITreeRecord` parallel-array trees. `max_depth` is
    fixed at `ceil(log2(psi))` per the original Liu/Ting/Zhou paper.
    """

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if n_trees < 1:
        raise ValueError(f"n_trees must be >= 1, got {n_trees}")
    if psi < 2:
        raise ValueError(f"psi must be >= 2, got {psi}")
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"isolation forest requires N >= 2 rows, got {n}")
    effective_psi = min(psi, n)
    max_depth = max(int(math.ceil(math.log2(effective_psi))), 1)
    rng = np.random.default_rng(seed)
    trees: list[ITreeRecord] = []
    for _ in range(n_trees):
        sample_idx = rng.choice(n, size=effective_psi, replace=False)
        trees.append(_build_itree(X[sample_idx], rng, max_depth))
    return tuple(trees)


def _path_length_one(x: np.ndarray, tree: ITreeRecord) -> float:
    """Path length of `x` through `tree`, with leaf-size correction."""

    node = 0
    depth = 0
    feature = tree.feature
    split = tree.split
    left = tree.left
    right = tree.right
    size = tree.size
    while feature[node] != -1:
        child = left[node] if x[feature[node]] < split[node] else right[node]
        if child == -1:
            break
        node = child
        depth += 1
    # Leaf-size correction per the original paper: add c(leaf_size) so that
    # leaves containing multiple points are not credited with shorter paths.
    leaf_size = int(size[node])
    return float(depth) + c_psi(leaf_size)


def iforest_score(
    X: np.ndarray, trees: Sequence[ITreeRecord], psi: int
) -> np.ndarray:
    """`s(x) = 2^{-mean_h(x) / c(ψ)}` for every row `x` in `X`."""

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if psi < 2:
        raise ValueError(f"psi must be >= 2, got {psi}")
    n = X.shape[0]
    avg_path = np.zeros(n, dtype=float)
    for tree in trees:
        for i in range(n):
            avg_path[i] += _path_length_one(X[i], tree)
    avg_path /= max(len(trees), 1)
    normalizer = c_psi(psi)
    if normalizer <= 0:
        normalizer = 1.0
    return np.power(2.0, -avg_path / normalizer)


# ---------------------------------------------------------------------------
# Detector 3: Mahalanobis distance (Gaussian one-class baseline for OCSVM)
# ---------------------------------------------------------------------------


def fit_mahalanobis(
    X: np.ndarray, shrinkage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit `(mean, inv_cov)` with a ridge `shrinkage * I` added to the covariance.

    Robust to rank-deficient panels: the ridge guarantees a positive-definite
    inverse even when the empirical covariance is singular.
    """

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if shrinkage < 0:
        raise ValueError(f"shrinkage must be >= 0, got {shrinkage}")
    n, p = X.shape
    if n < 2:
        raise ValueError(f"Mahalanobis fit requires N >= 2 rows, got {n}")
    mean = X.mean(axis=0)
    centered = X - mean
    cov = (centered.T @ centered) / max(n - 1, 1)
    cov_reg = cov + shrinkage * np.eye(p)
    inv_cov = np.linalg.pinv(cov_reg)
    return mean.astype(float), inv_cov.astype(float)


def mahalanobis_score(
    X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray
) -> np.ndarray:
    """`(x - μ)ᵀ Σ⁻¹ (x - μ)` per row of `X`."""

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    centered = X - mean
    # Compute (centered @ inv_cov) elementwise-multiplied with centered, summed.
    proj = centered @ inv_cov
    return cast(np.ndarray, np.einsum("ij,ij->i", proj, centered).astype(float))


# ---------------------------------------------------------------------------
# Convenience: full-detector scoring used by both calibration & inference
# ---------------------------------------------------------------------------


def score_all_detectors(
    X: np.ndarray, fit: DetectorFit
) -> dict[str, np.ndarray]:
    """Score `X` (already in the model's preprocessed feature space).

    Returns a dict keyed `"ae"`, `"iforest"`, `"mahalanobis"`.
    """

    return {
        "ae": pca_reconstruction_score(X, fit.pca_mean, fit.pca_components),
        "iforest": iforest_score(X, fit.iforest_trees, fit.iforest_psi),
        "mahalanobis": mahalanobis_score(X, fit.mahal_mean, fit.mahal_inv_cov),
    }


def threshold_from_quantile(scores: np.ndarray, contamination: float) -> float:
    """Empirical `1 - contamination` quantile — spec's threshold rule."""

    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    if not 0.0 < contamination < 1.0:
        raise ValueError(
            f"contamination must lie in (0, 1), got {contamination}"
        )
    return float(np.quantile(scores, 1.0 - contamination))


def majority_vote_flags(flags: dict[str, bool], min_votes: int = 2) -> bool:
    """`>= min_votes` of the detector flags fired — spec's combined rule."""

    if min_votes < 1:
        raise ValueError(f"min_votes must be >= 1, got {min_votes}")
    return sum(1 for v in flags.values() if v) >= min_votes


def top_feature_attribution(
    residuals_sq: np.ndarray,
    feature_names: Sequence[str],
    *,
    k: int = 3,
) -> tuple[tuple[str, float], ...]:
    """Top-`k` features by squared reconstruction residual.

    Returns a tuple of `(name, residual_value)` pairs in descending order.
    """

    if residuals_sq.ndim != 1:
        raise ValueError(f"residuals_sq must be 1-D, got {residuals_sq.shape}")
    if residuals_sq.shape[0] != len(feature_names):
        raise ValueError(
            f"residuals_sq has {residuals_sq.shape[0]} values but "
            f"{len(feature_names)} feature_names"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, residuals_sq.shape[0])
    order = np.argsort(residuals_sq)[::-1][:k]
    return tuple((feature_names[int(i)], float(residuals_sq[int(i)])) for i in order)


# ---------------------------------------------------------------------------
# FinBERT sentiment helpers — independent of the model checkpoint
# ---------------------------------------------------------------------------


# Match a sentence-ending punctuation followed by whitespace and a capital
# letter or digit — coarse but `re.split` requires a fixed-width pattern.
_SENTENCE_BOUNDARY = re.compile(r"([\.\!\?])\s+(?=[A-Z0-9\"\'])")

# Abbreviations that should NOT terminate a sentence when followed by space.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {"mr", "mrs", "ms", "dr", "inc", "ltd", "co", "corp", "vs", "jr", "sr", "st"}
)


def split_sentences(text: str) -> list[str]:
    """Light-weight sentence splitter on `.`, `!`, `?`.

    Not as sharp as a proper NLP segmenter — it just guards a few common
    abbreviations — but good enough for the short article titles + summaries
    that yfinance returns. Spec's algorithm outline allows a regex splitter.
    """

    if not text:
        return []
    cleaned = text.strip()
    if not cleaned:
        return []
    # Split on `re.split` with a captured punctuation; the captured groups
    # interleave between sentences. We then re-attach punctuation and merge
    # back across known abbreviations.
    raw = _SENTENCE_BOUNDARY.split(cleaned)
    sentences: list[str] = []
    buf = ""
    i = 0
    while i < len(raw):
        chunk = raw[i]
        # Punctuation captured groups arrive at odd indices.
        if i + 1 < len(raw):
            buf += chunk + raw[i + 1]
            # Look at the last token before the punctuation in `chunk`.
            tokens = chunk.split()
            last = tokens[-1].rstrip(".!?").lower() if tokens else ""
            if last in _ABBREVIATIONS:
                # Skip the split — keep buffering until the next boundary.
                buf += " "
                i += 2
                continue
            sentences.append(buf.strip())
            buf = ""
            i += 2
        else:
            buf += chunk
            i += 1
    if buf.strip():
        sentences.append(buf.strip())
    return sentences if sentences else [cleaned]


def sentiment_scalar(
    probabilities: np.ndarray,
    label_to_index: dict[str, int],
) -> float:
    """`p_pos - p_neg` ∈ `[-1, 1]`.

    `label_to_index` is the FinBERT `label2id` map at runtime — the spec
    warns that the index order is sometimes `{positive: 0, negative: 1,
    neutral: 2}` rather than alphabetical, so we always look up by name.
    """

    if probabilities.ndim != 1 or probabilities.shape[0] != 3:
        raise ValueError(
            f"probabilities must be length-3 1-D, got shape {probabilities.shape}"
        )
    for label in ("positive", "negative"):
        if label not in label_to_index:
            raise ValueError(
                f"label_to_index missing required key {label!r}; got "
                f"{sorted(label_to_index)}"
            )
    pos = float(probabilities[label_to_index["positive"]])
    neg = float(probabilities[label_to_index["negative"]])
    return pos - neg


def exp_decay_weight(age_hours: float, decay_hours: float) -> float:
    """`exp(-age_hours / decay_hours)`. Spec's recency weight."""

    if decay_hours <= 0:
        raise ValueError(f"decay_hours must be > 0, got {decay_hours}")
    if age_hours < 0:
        # Future-dated article (clock skew); treat as zero-aged.
        age_hours = 0.0
    return float(math.exp(-age_hours / decay_hours))


def weighted_sentiment(
    scores: Sequence[float],
    weights: Sequence[float],
    *,
    eps: float = 1e-9,
) -> float:
    """`sum(w * s) / max(sum(w), eps)` — guarded weighted average.

    Returns 0 when there are no articles or all weights are 0.
    """

    if len(scores) != len(weights):
        raise ValueError(
            f"scores ({len(scores)}) and weights ({len(weights)}) length mismatch"
        )
    if not scores:
        return 0.0
    s_arr = np.asarray(scores, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    denom = float(w_arr.sum())
    if denom < eps:
        return 0.0
    return float((s_arr * w_arr).sum() / denom)


__all__ = [
    "abs_return_zscore",
    "build_anomaly_feature_panel",
    "c_psi",
    "cross_sectional_rank",
    "exp_decay_weight",
    "fit_iforest",
    "fit_mahalanobis",
    "iforest_score",
    "impute_and_clip",
    "intraday_range",
    "log_return_1d",
    "log_volume",
    "ma_drift",
    "mahalanobis_score",
    "majority_vote_flags",
    "median_mad_standardize",
    "open_close_gap",
    "pca_fit",
    "pca_reconstruct",
    "pca_reconstruction_score",
    "pca_residual_attribution",
    "realized_vol",
    "score_all_detectors",
    "sentiment_scalar",
    "split_sentences",
    "threshold_from_quantile",
    "top_feature_attribution",
    "vol_ratio",
    "weighted_sentiment",
]
