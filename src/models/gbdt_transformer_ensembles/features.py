"""Feature engineering for the GBDT + Transformer ensemble.

Pure pandas/numpy — no model state, no I/O. The model class is responsible
for pulling OHLCV bars from the `DataProvider`; everything in this module
takes a dict of per-ticker price frames and produces a long-format
`PanelFrame` that downstream learners consume.

Naming convention:

- ``r{k}``       — k-day log return on `Close`.
- ``lag_r1_{k}`` — `r1` at lag k.
- ``vol_{k}``    — k-day rolling stdev of `r1`.
- ``mean_{k}``   — k-day rolling mean of `r1`.
- ``rng_oc``     — (Close - Open) / Open.
- ``rng_hl``     — (High - Low) / Close.
- ``log_vol``    — log(1 + Volume).
- ``vol_ratio``  — Volume / MA20(Volume).
- ``amihud``     — |r1| / (Volume * Close).
- ``bench_r{k}`` — r{k} minus benchmark r{k}.
- ``cs_rank_*``  — cross-sectional rank in [0, 1] of feature * across the
                   universe at the same date.
- ``cal_*``      — calendar encodings (sin/cos day-of-week, days-to-eom).
- ``y``          — forward log return over `target_horizon` days.
- ``w``          — sample weight (inverse rolling vol so quiet rows weigh
                   the same as loud ones; matches the spec's "sample
                   weight" intent on yfinance data that has none).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.models.gbdt_transformer_ensembles.types import FeatureSpec, PanelFrame


def compute_log_returns(close: pd.Series, horizon: int) -> pd.Series:
    """`r_t^{(k)} = log(P_t) - log(P_{t-k})` per the spec."""

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    log_p = pd.Series(np.log(close.astype(float).to_numpy()), index=close.index)
    return log_p - log_p.shift(horizon)


def _forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """`y_t = log(P_{t+h}) - log(P_t)`. Forward-looking; used only for the target."""

    log_p = pd.Series(np.log(close.astype(float).to_numpy()), index=close.index)
    return log_p.shift(-horizon) - log_p


def _build_single_ticker_features(
    ticker: str,
    bars: pd.DataFrame,
    spec: FeatureSpec,
    benchmark_close: pd.Series | None,
) -> pd.DataFrame:
    """Per-ticker feature matrix indexed by date.

    `bars` must contain `Open`, `High`, `Low`, `Close`, `Volume`. Missing
    OHLC values are forward-filled up to 3 days; longer gaps remain NaN
    (which the GBDT will see as the `nan_sentinel` later).
    """

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(bars.columns):
        missing = required.difference(bars.columns)
        raise ValueError(f"bars for {ticker} missing columns {sorted(missing)}")
    bars = bars.sort_index().copy()
    for col in ("Open", "High", "Low", "Close"):
        bars[col] = bars[col].astype(float).ffill(limit=3)
    bars["Volume"] = bars["Volume"].astype(float).fillna(0.0)

    r1 = compute_log_returns(bars["Close"], 1)
    features: dict[str, pd.Series] = {}

    for h in spec.return_horizons:
        features[f"r{h}"] = compute_log_returns(bars["Close"], h)

    for k in range(1, spec.lag_count + 1):
        features[f"lag_r1_{k}"] = r1.shift(k)

    for w in spec.vol_windows:
        features[f"vol_{w}"] = r1.rolling(window=w, min_periods=w).std(ddof=0)

    for w in spec.mean_windows:
        features[f"mean_{w}"] = r1.rolling(window=w, min_periods=w).mean()

    features["rng_oc"] = (bars["Close"] - bars["Open"]) / bars["Open"].replace(0.0, np.nan)
    features["rng_hl"] = (bars["High"] - bars["Low"]) / bars["Close"].replace(0.0, np.nan)
    features["log_vol"] = np.log1p(bars["Volume"])
    ma20_v = bars["Volume"].rolling(window=20, min_periods=20).mean()
    features["vol_ratio"] = bars["Volume"] / ma20_v.replace(0.0, np.nan)
    denom = bars["Volume"] * bars["Close"]
    features["amihud"] = r1.abs() / denom.replace(0.0, np.nan)

    if benchmark_close is not None:
        bench = benchmark_close.copy()
        if isinstance(bench.index, pd.DatetimeIndex) and bench.index.tz is not None:
            bench.index = bench.index.tz_localize(None)
        b_aligned = bench.reindex(bars.index).ffill(limit=3)
        for h in spec.return_horizons:
            b_r = compute_log_returns(b_aligned, h)
            features[f"bench_r{h}"] = features[f"r{h}"] - b_r

    if spec.include_calendar:
        idx = pd.DatetimeIndex(bars.index)
        # Drop timezone if present so numpy subtraction in days-to-qe works.
        idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
        dow = idx_naive.dayofweek.to_numpy(dtype=float)
        features["cal_dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 5.0), index=bars.index)
        features["cal_dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 5.0), index=bars.index)
        # days-to-quarter-end ∈ [0, ~63]; rescaled to [0, 1]
        q_end = idx_naive.to_period("Q").end_time.to_numpy().astype("datetime64[ns]")
        idx_ns = idx_naive.to_numpy().astype("datetime64[ns]")
        days_to_qe = (q_end - idx_ns).astype("timedelta64[D]").astype(int)
        features["cal_dtq"] = pd.Series(days_to_qe / 63.0, index=bars.index)

    y = _forward_log_return(bars["Close"], spec.target_horizon)
    if spec.clip_target is not None:
        y = y.clip(lower=-spec.clip_target, upper=spec.clip_target)

    # Inverse rolling-vol weight; floors to median to avoid blowups.
    base_vol = r1.rolling(window=21, min_periods=21).std(ddof=0)
    weight = 1.0 / base_vol.replace(0.0, np.nan)
    weight = weight.replace([np.inf, -np.inf], np.nan)
    w_median = weight.median(skipna=True)
    if not np.isfinite(w_median) or w_median <= 0:
        w_median = 1.0
    weight = weight.fillna(w_median)
    weight = weight / w_median  # normalize so the median weight = 1

    out = pd.DataFrame(features, index=bars.index)
    out["y"] = y
    out["w"] = weight
    out["ticker"] = ticker
    return out


def _add_cross_sectional_ranks(panel: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """For each base feature column, append its same-date cross-sectional rank in [0, 1].

    Rank uses `pct=True` so ties go to the average rank; the column is
    populated only when at least 2 non-NaN values exist on that date.
    """

    if not base_cols:
        return panel
    grouped = panel.groupby("date", sort=False, observed=True)
    extras: dict[str, pd.Series] = {}
    for col in base_cols:
        ranks = grouped[col].rank(pct=True, method="average")
        extras[f"cs_rank_{col}"] = ranks
    rank_df = pd.DataFrame(extras, index=panel.index)
    return pd.concat([panel, rank_df], axis=1)


def build_feature_panel(
    prices_by_ticker: Mapping[str, pd.DataFrame],
    *,
    spec: FeatureSpec,
    benchmark_close: pd.Series | None = None,
) -> PanelFrame:
    """Assemble the long-format feature panel across all tickers.

    Rows below `spec.min_history_days` of contiguous history are dropped
    per ticker so rolling features are well-defined. Rows where `y` is
    NaN (the last `target_horizon` days, or rows with no future price) are
    also dropped — we never train on missing targets.
    """

    if not prices_by_ticker:
        raise ValueError("prices_by_ticker is empty")

    per_ticker: list[pd.DataFrame] = []
    for ticker, bars in prices_by_ticker.items():
        df = _build_single_ticker_features(
            ticker=ticker,
            bars=bars,
            spec=spec,
            benchmark_close=benchmark_close,
        )
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        dates = pd.to_datetime(df["date"])
        if isinstance(dates.dtype, pd.DatetimeTZDtype):
            dates = dates.dt.tz_localize(None)
        df["date"] = dates
        per_ticker.append(df)

    panel = pd.concat(per_ticker, axis=0, ignore_index=True)

    # Drop ticker-history rows below the min-history threshold using the
    # row index *within* each ticker (the first row of each ticker has the
    # most rolling NaNs; we drop the first `min_history_days` rows).
    if spec.min_history_days > 0:
        keep_mask = panel.groupby("ticker", sort=False).cumcount() >= spec.min_history_days
        panel = panel.loc[keep_mask]

    base_cols = [
        c for c in panel.columns if c not in {"date", "ticker", "y", "w"}
    ]
    if spec.include_cross_sectional_ranks:
        panel = _add_cross_sectional_ranks(panel, base_cols)

    panel = panel.dropna(subset=["y"]).reset_index(drop=True)
    if panel.empty:
        raise RuntimeError(
            "Feature panel is empty after dropping rows without a target — "
            "check that `target_horizon` and `min_history_days` are compatible "
            "with the price history available."
        )

    feature_names = tuple(
        c for c in panel.columns if c not in {"date", "ticker", "y", "w"}
    )
    X = panel[list(feature_names)].astype(float)
    y = panel["y"].astype(float)
    w = panel["w"].astype(float)
    meta = panel[["date", "ticker"]].reset_index(drop=True)
    return PanelFrame(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        w=w.reset_index(drop=True),
        meta=meta,
        feature_names=feature_names,
    )


def fill_nans_for_trees(X: pd.DataFrame, sentinel: float) -> np.ndarray:
    """GBDTs see NaN as the sentinel (default -1) per the spec."""

    arr = X.to_numpy(dtype=float, copy=True)
    np.nan_to_num(arr, copy=False, nan=sentinel, posinf=sentinel, neginf=sentinel)
    return arr


def fill_nans_for_linear(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Linear models see NaN -> 0, with a binary mask channel per the spec.

    Returns `(X_imputed, mask)` where `mask` is 1 where the input was NaN.
    """

    arr = X.to_numpy(dtype=float, copy=True)
    mask = (~np.isfinite(arr)).astype(float)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, mask


__all__ = [
    "build_feature_panel",
    "compute_log_returns",
    "fill_nans_for_linear",
    "fill_nans_for_trees",
]
