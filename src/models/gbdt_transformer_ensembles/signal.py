"""Diagnostics and signal helpers for the ensemble.

Every metric in this module is a pure function of arrays — no side effects,
no model state. They're what `validate()` and `predict()` lean on.

Implements the spec's "Validation and diagnostics" section:

- Weighted R^2 (primary metric).
- Cross-sectional rank IC (Spearman, per date, then averaged).
- Time-series IC (Pearson, per ticker, then averaged).
- Decile spread (long top / short bottom, equal-weighted return).
- Calibration plot bins.
- Population Stability Index (PSI) for concept drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPSILON: float = 1e-12


def weighted_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    """`1 - sum w (y - yhat)^2 / sum w (y - ybar)^2`.

    `ybar` is the weighted mean of `y`. Returns NaN if all weights are
    zero or the variance of `y` is zero.
    """

    if y.shape != yhat.shape or y.shape != w.shape:
        raise ValueError("weighted_r2: y, yhat, w must have the same shape")
    w_pos = np.maximum(w, 0.0)
    s_w = w_pos.sum()
    if s_w <= 0:
        return float("nan")
    y_bar = float((w_pos * y).sum() / s_w)
    ss_res = float((w_pos * (y - yhat) ** 2).sum())
    ss_tot = float((w_pos * (y - y_bar) ** 2).sum())
    if ss_tot <= EPSILON:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def spearman_rank_ic(yhat: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation = Pearson on ranks. NaN-safe."""

    if yhat.shape != y.shape:
        raise ValueError("spearman_rank_ic: shapes must match")
    mask = np.isfinite(yhat) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    yhat_r = pd.Series(yhat[mask]).rank(method="average").to_numpy()
    y_r = pd.Series(y[mask]).rank(method="average").to_numpy()
    return _pearson(yhat_r, y_r)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= EPSILON:
        return float("nan")
    return float((a * b).sum() / denom)


def cross_sectional_rank_ic(
    yhat: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
) -> float:
    """Average Spearman IC across dates (the spec's primary cross-sectional metric)."""

    if yhat.shape != y.shape or len(dates) != yhat.shape[0]:
        raise ValueError("cross_sectional_rank_ic: shape mismatch")
    df = pd.DataFrame({"yhat": yhat, "y": y, "date": dates.to_numpy()})
    ics: list[float] = []
    for _, sub in df.groupby("date", sort=False, observed=True):
        if len(sub) < 2:
            continue
        ic = spearman_rank_ic(sub["yhat"].to_numpy(), sub["y"].to_numpy())
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def time_series_ic(
    yhat: np.ndarray,
    y: np.ndarray,
    tickers: pd.Series,
) -> float:
    """Average per-ticker Pearson IC over time."""

    if yhat.shape != y.shape or len(tickers) != yhat.shape[0]:
        raise ValueError("time_series_ic: shape mismatch")
    df = pd.DataFrame({"yhat": yhat, "y": y, "ticker": tickers.to_numpy()})
    ics: list[float] = []
    for _, sub in df.groupby("ticker", sort=False, observed=True):
        if len(sub) < 3:
            continue
        ic = _pearson(sub["yhat"].to_numpy(), sub["y"].to_numpy())
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def decile_spread(
    yhat: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    *,
    n_buckets: int = 10,
) -> float:
    """Mean realized return of `top decile - bottom decile`, averaged across dates.

    The bucket assignment is per-date so the spread is a pure cross-sectional
    signal — no calendar bias.
    """

    if n_buckets < 2:
        raise ValueError("decile_spread: n_buckets must be >= 2")
    if yhat.shape != y.shape or len(dates) != yhat.shape[0]:
        raise ValueError("decile_spread: shape mismatch")
    df = pd.DataFrame({"yhat": yhat, "y": y, "date": dates.to_numpy()})
    diffs: list[float] = []
    for _, sub in df.groupby("date", sort=False, observed=True):
        if len(sub) < n_buckets:
            continue
        # Use rank-based quantiles to be robust to ties.
        try:
            buckets = pd.qcut(sub["yhat"], q=n_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue
        if buckets is None:
            continue
        top = sub["y"][buckets == buckets.max()].mean()
        bot = sub["y"][buckets == buckets.min()].mean()
        if np.isfinite(top) and np.isfinite(bot):
            diffs.append(float(top - bot))
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def calibration_bins(
    yhat: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin (mean(yhat), mean(y), count) — the spec's "calibration plot"."""

    if yhat.shape != y.shape:
        raise ValueError("calibration_bins: shape mismatch")
    mask = np.isfinite(yhat) & np.isfinite(y)
    if mask.sum() < n_bins:
        return pd.DataFrame(columns=["pred_mean", "realized_mean", "count"])
    yhat_m = yhat[mask]
    y_m = y[mask]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(yhat_m, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["pred_mean", "realized_mean", "count"])
    bin_idx = np.clip(np.searchsorted(edges[1:-1], yhat_m, side="right"), 0, len(edges) - 2)
    df = pd.DataFrame({"bin": bin_idx, "pred": yhat_m, "real": y_m})
    grouped = df.groupby("bin", sort=True, observed=True)
    out = pd.DataFrame(
        {
            "pred_mean": grouped["pred"].mean(),
            "realized_mean": grouped["real"].mean(),
            "count": grouped["pred"].count(),
        }
    )
    return out.reset_index(drop=True)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """PSI = sum_b (p_curr_b - p_ref_b) * log(p_curr_b / p_ref_b).

    Computed on quantile bins of the *reference* sample. NaN-safe; a small
    epsilon protects empty bins.
    """

    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if ref.size < n_bins or cur.size == 0:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    edges_inner = edges[1:-1]
    ref_bins = np.clip(np.searchsorted(edges_inner, ref, side="right"), 0, len(edges) - 2)
    cur_bins = np.clip(np.searchsorted(edges_inner, cur, side="right"), 0, len(edges) - 2)
    nb = len(edges) - 1
    ref_p = (np.bincount(ref_bins, minlength=nb) + 1e-6) / (ref.size + 1e-6 * nb)
    cur_p = (np.bincount(cur_bins, minlength=nb) + 1e-6) / (cur.size + 1e-6 * nb)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def summarize_oof_metrics(
    oof: pd.DataFrame,
    *,
    pred_col: str = "yhat_ensemble",
) -> dict[str, Any]:
    """Bundle the spec's primary diagnostics into a single dict.

    Expects `oof` columns: `y`, `w`, `date`, `ticker`, and `pred_col`.
    """

    y = oof["y"].to_numpy(dtype=float)
    yhat = oof[pred_col].to_numpy(dtype=float)
    w = oof["w"].to_numpy(dtype=float)
    return {
        "weighted_r2": weighted_r2(y, yhat, w),
        "cross_sectional_rank_ic": cross_sectional_rank_ic(yhat, y, oof["date"]),
        "time_series_ic": time_series_ic(yhat, y, oof["ticker"]),
        "decile_spread": decile_spread(yhat, y, oof["date"]),
        "n_samples": int(len(oof)),
        "n_dates": int(oof["date"].nunique()),
        "n_tickers": int(oof["ticker"].nunique()),
    }


__all__ = [
    "calibration_bins",
    "cross_sectional_rank_ic",
    "decile_spread",
    "population_stability_index",
    "spearman_rank_ic",
    "summarize_oof_metrics",
    "time_series_ic",
    "weighted_r2",
]
