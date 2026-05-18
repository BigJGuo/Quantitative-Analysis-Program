"""Pure-numpy / pandas math layer of the Supervised Autoencoder + MLP model.

Nothing in this module performs I/O or imports torch. Everything here is a
deterministic transformation of arrays / DataFrames — feature engineering,
target binarization, purged time-series K-fold, ensemble median blend, and
the spec's action rule. Trainable-graph code lives in `network.py`.

The functions mirror the spec's Algorithm Outline:

- `rolling_log_returns`, `rolling_std`, `build_feature_panel` -> step 1
- `winsorize`, `median_mad_standardize`, `impute_with_median` -> step 1 (preproc)
- `binarize_targets`                                          -> step 2
- `purged_kfold_indices`                                      -> step 3
- `blend_predictions`, `action_rule`                          -> step 5
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

# Numerical floor for the MAD denominator. Below this we substitute 1.0 so a
# constant feature column standardizes to zero rather than NaN.
_MAD_FLOOR: float = 1e-12

# Forward-return horizons (trading days) used by `binarize_targets`. The
# defaults follow the spec example: 1-day, median-of-5-day, sum-of-5-day.
DEFAULT_HORIZONS_DAYS: tuple[int, ...] = (1, 5, 10)


def rolling_log_returns(close: pd.Series, window: int) -> pd.Series:
    """`log(C_t / C_{t-window})`. Returns a Series same index, leading NaNs."""

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    close = close.astype(float)
    log_close = pd.Series(np.log(close.to_numpy()), index=close.index)
    return log_close.diff(window)


def rolling_std(returns: pd.Series, window: int) -> pd.Series:
    """Rolling sample std of a return series."""

    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    return returns.astype(float).rolling(window=window, min_periods=window).std(ddof=1)


def rolling_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """`sqrt(sum r_s^2)` over a trailing window (spec's RV definition)."""

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    squared = returns.astype(float) ** 2
    rolled = squared.rolling(window=window, min_periods=window).sum()
    return pd.Series(np.sqrt(rolled.to_numpy()), index=rolled.index)


def amihud_illiquidity(returns: pd.Series, volume: pd.Series, close: pd.Series) -> pd.Series:
    """`|r_t| / (V_t * C_t)` — Amihud illiquidity proxy."""

    notional = (volume.astype(float) * close.astype(float)).replace(0.0, np.nan)
    return returns.astype(float).abs() / notional


def build_feature_panel(
    bars: pd.DataFrame,
    *,
    return_windows: tuple[int, ...] = (1, 5, 10, 20, 60),
    vol_windows: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """Build a per-date feature panel for one ticker from an OHLCV bars frame.

    Output columns (one DataFrame row per bar date):

    - `ret_k` for each `k` in `return_windows`
    - `vol_k` for each `k` in `vol_windows`        (rolling std of 1-day rets)
    - `rv_k`  for each `k` in `vol_windows`        (rolling realized vol)
    - `hl`                                          (high-low range / close)
    - `vol_ratio`                                   (V / MA_20(V))
    - `illiq`                                       (Amihud)

    The frame is returned with whatever leading NaNs the rolling windows
    produce; callers should drop or impute before feeding to the network.
    """

    if bars is None or bars.empty:
        raise ValueError("build_feature_panel requires a non-empty bars frame")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars frame missing columns: {sorted(missing)}")

    close = bars["Close"].astype(float)
    high = bars["High"].astype(float)
    low = bars["Low"].astype(float)
    volume = bars["Volume"].astype(float)
    r1 = rolling_log_returns(close, 1)

    out: dict[str, pd.Series] = {}
    for k in return_windows:
        out[f"ret_{k}"] = rolling_log_returns(close, k)
    for k in vol_windows:
        out[f"vol_{k}"] = rolling_std(r1, k)
        out[f"rv_{k}"] = rolling_realized_vol(r1, k)
    out["hl"] = (high - low) / close.replace(0.0, np.nan)
    out["vol_ratio"] = volume / volume.rolling(window=20, min_periods=20).mean()
    out["illiq"] = amihud_illiquidity(r1, volume, close)

    panel = pd.DataFrame(out, index=bars.index)
    return panel


def winsorize(values: np.ndarray, quantile: float) -> np.ndarray:
    """Clip each column at the `quantile` / `1 - quantile` levels.

    Operates column-wise on a 2-D array; ignores NaNs when computing the
    quantile via `np.nanquantile`. Returns a fresh array (does not mutate
    the input). `quantile=0` is a no-op.
    """

    if values.ndim != 2:
        raise ValueError(f"winsorize expects a 2-D array, got shape {values.shape}")
    if not 0.0 <= quantile < 0.5:
        raise ValueError(f"quantile must lie in [0, 0.5), got {quantile}")
    if quantile == 0.0:
        return values.astype(float, copy=True)
    lo = np.nanquantile(values, quantile, axis=0)
    hi = np.nanquantile(values, 1.0 - quantile, axis=0)
    return cast(np.ndarray, np.clip(values, lo, hi))


def median_mad_standardize(
    values: np.ndarray,
    *,
    median: np.ndarray | None = None,
    mad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column robust standardization: `(x - median) / (1.4826 * MAD)`.

    Returns ``(standardized, median, scale)``. The `1.4826` constant makes
    MAD * 1.4826 a consistent estimator of the Gaussian standard deviation,
    so the result is comparable to a z-score for normal-ish features.

    If `median` and `mad` are supplied they are reused (test-time
    standardization with train-time stats); otherwise both are computed
    column-wise on the input.
    """

    if values.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {values.shape}")
    if median is None:
        median = np.nanmedian(values, axis=0)
    if mad is None:
        mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    safe_scale = np.where(scale > _MAD_FLOOR, scale, 1.0)
    standardized = (values - median) / safe_scale
    return standardized.astype(float), median.astype(float), safe_scale.astype(float)


def impute_with_median(
    values: np.ndarray, *, median: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace NaNs with the column median; return imputed array + indicator.

    Returns ``(imputed, median, missing_mask)`` where ``missing_mask`` is a
    float array the same shape as the input — 1 where a NaN was filled,
    0 otherwise. The spec recommends concatenating this mask as extra
    features so the network can learn from "missingness".
    """

    if values.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {values.shape}")
    missing = np.isnan(values)
    if median is None:
        median = np.nanmedian(values, axis=0)
    # If a whole column is NaN, `nanmedian` returns NaN — fall back to 0.
    safe_median = np.where(np.isnan(median), 0.0, median)
    imputed = np.where(missing, safe_median, values)
    return imputed.astype(float), safe_median.astype(float), missing.astype(float)


def binarize_targets(
    forward_returns: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS_DAYS,
) -> pd.DataFrame:
    """Build the spec's three binary targets from a forward-return panel.

    `forward_returns` is a (T, H) frame whose columns are 1-day forward
    log-returns at horizons `h in horizons` (typically 1, 5, 10). Targets:

    - ``pos_1d``      = ``1{r^(1) > 0}``
    - ``pos_med_5d``  = ``1{median(r^(h) for h in horizons) > 0}``
    - ``pos_sum_5d``  = ``1{sum(r^(h)) > 0}``

    Rows where any column is NaN propagate NaN through (so calibration can
    drop them downstream); otherwise the result is 0/1 floats.
    """

    if forward_returns.empty:
        raise ValueError("binarize_targets requires non-empty forward returns")
    if forward_returns.shape[1] != len(horizons):
        raise ValueError(
            f"forward_returns has {forward_returns.shape[1]} columns but "
            f"{len(horizons)} horizons were supplied"
        )

    r = forward_returns.astype(float)
    primary = r.iloc[:, 0]
    median = r.median(axis=1)
    total = r.sum(axis=1)
    out = pd.DataFrame(
        {
            "pos_1d": (primary > 0).astype(float),
            "pos_med_5d": (median > 0).astype(float),
            "pos_sum_5d": (total > 0).astype(float),
        },
        index=forward_returns.index,
    )
    # Preserve NaNs in any row where the forward return panel was incomplete.
    nan_mask = forward_returns.isna().any(axis=1)
    if nan_mask.any():
        out.loc[nan_mask, :] = np.nan
    return out


@dataclass(frozen=True)
class FoldSplit:
    """One purged time-series K-fold partition."""

    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def purged_kfold_indices(
    dates: np.ndarray, n_folds: int, *, embargo: int = 1
) -> Iterator[FoldSplit]:
    """Yield purged time-series K-fold (date, train_idx, val_idx) splits.

    Procedure (per de Prado, AFML ch. 7):

    1. Sort the unique calendar dates ascending.
    2. Slice them into `n_folds` contiguous validation blocks.
    3. Train indices are everything outside the validation block *and* outside
       the embargo band of `embargo` trading days before/after the block.

    This is the recommended scheme for the SAE+MLP — random K-fold would leak
    forward information through the rolling features.
    """

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    if dates.ndim != 1:
        raise ValueError(f"dates must be 1-D, got shape {dates.shape}")

    unique_dates = np.array(sorted(set(dates.tolist())))
    if unique_dates.size < n_folds:
        raise ValueError(
            f"need at least n_folds={n_folds} unique dates, got {unique_dates.size}"
        )
    blocks = np.array_split(unique_dates, n_folds)
    date_to_pos = {d: i for i, d in enumerate(unique_dates.tolist())}

    for fold, block in enumerate(blocks):
        block_set = set(block.tolist())
        val_mask = np.array([d in block_set for d in dates], dtype=bool)
        # Embargo: drop dates within `embargo` positions of the block edges.
        first_pos = date_to_pos[block[0].item() if hasattr(block[0], "item") else block[0]]
        last_pos = date_to_pos[block[-1].item() if hasattr(block[-1], "item") else block[-1]]
        lo = max(first_pos - embargo, 0)
        hi = min(last_pos + embargo, unique_dates.size - 1)
        embargo_dates = set(unique_dates[lo : hi + 1].tolist())
        train_mask = np.array(
            [d not in embargo_dates for d in dates],
            dtype=bool,
        )
        yield FoldSplit(
            fold=fold,
            train_idx=np.flatnonzero(train_mask),
            val_idx=np.flatnonzero(val_mask),
        )


def blend_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    """Median-blend a list of (N, K) prediction arrays from ensemble members.

    Per the spec, **median** is preferred over mean because it is robust to
    seed-level failures (a single divergent member doesn't drag the blend).
    """

    if not predictions:
        raise ValueError("blend_predictions requires at least one member")
    stacked = np.stack(predictions, axis=0)  # (M, N, K)
    return cast(np.ndarray, np.median(stacked, axis=0))


def action_rule(blended: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Spec's trade-take rule: `take_trade = max_k blended[i, k] > threshold`.

    `blended` is an (N, K) array of ensemble-blended probabilities. Returns a
    length-N int array of 0/1 actions.
    """

    if blended.ndim != 2:
        raise ValueError(f"blended must be 2-D, got shape {blended.shape}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold}")
    return cast(np.ndarray, (blended.max(axis=1) > threshold).astype(int))


def reconstruction_error(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-row MSE between input and decoded reconstruction.

    Spec validation item 4 ("Decoder sanity") — sustained spikes in this
    score flag input-distribution drift.
    """

    if x.shape != x_hat.shape:
        raise ValueError(
            f"shape mismatch: x {x.shape} vs x_hat {x_hat.shape}"
        )
    diff = x - x_hat
    return cast(np.ndarray, np.mean(diff * diff, axis=1))


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-target Brier score `mean((p - y)^2)` (spec validation item 2).

    `probabilities` and `targets` are both (N, K). Returns a length-K array.
    """

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: probabilities {probabilities.shape} vs "
            f"targets {targets.shape}"
        )
    diff = probabilities - targets
    return cast(np.ndarray, np.mean(diff * diff, axis=0))


def roc_auc(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Mann-Whitney-U based ROC-AUC for a single binary target column.

    Equivalent to `sklearn.metrics.roc_auc_score`. Returns 0.5 in degenerate
    cases (all-zero or all-one labels) so the metric can be reported
    unconditionally without crashing on imbalanced folds.
    """

    if probabilities.ndim != 1 or targets.ndim != 1:
        raise ValueError("roc_auc expects 1-D arrays")
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must share shape")
    y = targets.astype(int)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks; tie-aware via `scipy`-style midrank but implemented inline.
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    # Midrank for ties: average position across the tie block.
    sorted_probs = probabilities[order]
    i = 0
    n = probabilities.size
    while i < n:
        j = i
        while j + 1 < n and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-indexed midrank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = float(ranks[y == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
