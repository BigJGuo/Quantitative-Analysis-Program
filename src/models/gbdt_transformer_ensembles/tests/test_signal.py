"""Unit tests for diagnostics (weighted R^2, rank IC, decile spread, PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gbdt_transformer_ensembles.signal import (
    calibration_bins,
    cross_sectional_rank_ic,
    decile_spread,
    population_stability_index,
    spearman_rank_ic,
    summarize_oof_metrics,
    time_series_ic,
    weighted_r2,
)
from src.models.gbdt_transformer_ensembles.types import PanelFrame


def test_weighted_r2_perfect_fit_is_one() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert weighted_r2(y, y, np.ones(3)) == pytest.approx(1.0)


def test_weighted_r2_predicting_mean_is_zero() -> None:
    """yhat = ybar -> R^2 = 0 by definition."""

    y = np.array([1.0, 2.0, 3.0])
    yhat = np.full(3, y.mean())
    assert weighted_r2(y, yhat, np.ones(3)) == pytest.approx(0.0)


def test_weighted_r2_can_go_negative() -> None:
    """A worse-than-mean predictor produces negative R^2."""

    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([3.0, 2.0, 1.0])
    assert weighted_r2(y, yhat, np.ones(3)) < 0.0


def test_weighted_r2_weights_matter() -> None:
    y = np.array([0.0, 0.0, 10.0])
    yhat = np.array([0.0, 0.0, 0.0])
    # Heavy weight on first two perfect points -> R^2 close to 1 if we weight them.
    r2_uniform = weighted_r2(y, yhat, np.ones(3))
    r2_weighted = weighted_r2(y, yhat, np.array([10.0, 10.0, 1.0]))
    assert r2_weighted > r2_uniform


def test_spearman_rank_ic_perfectly_correlated_is_one() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([10.0, 20.0, 30.0, 40.0])  # monotone
    assert spearman_rank_ic(yhat, y) == pytest.approx(1.0)


def test_spearman_rank_ic_anti_correlated_is_minus_one() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([40.0, 30.0, 20.0, 10.0])
    assert spearman_rank_ic(yhat, y) == pytest.approx(-1.0)


def test_cross_sectional_rank_ic_averaging() -> None:
    """Per-date IC averaging: two dates, one perfect, one zero -> average ~0.5."""

    dates = pd.Series(pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3))
    yhat = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 3.0, 1.0, 2.0])
    ic = cross_sectional_rank_ic(yhat, y, dates)
    # First date: perfect IC = 1. Second date: ranks (1,2,3) vs (3,1,2) -> IC = -0.5.
    assert ic == pytest.approx((1.0 + (-0.5)) / 2.0, abs=1e-6)


def test_time_series_ic_averages_across_tickers() -> None:
    tickers = pd.Series(["A"] * 4 + ["B"] * 4)
    yhat = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    ic = time_series_ic(yhat, y, tickers)
    # A: perfect correlation = 1. B: perfect anti-correlation = -1. Avg = 0.
    assert ic == pytest.approx(0.0, abs=1e-6)


def test_decile_spread_positive_when_yhat_predicts_y() -> None:
    """If yhat == y, top decile - bottom decile = (max y - min y) > 0."""

    rng = np.random.default_rng(0)
    n_per_date = 20
    n_dates = 10
    yhat_blocks = []
    y_blocks = []
    dates = []
    for d in range(n_dates):
        v = rng.normal(size=n_per_date)
        yhat_blocks.append(v)
        y_blocks.append(v)  # identical
        dates.extend([pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)] * n_per_date)
    yhat = np.concatenate(yhat_blocks)
    y = np.concatenate(y_blocks)
    spread = decile_spread(yhat, y, pd.Series(dates), n_buckets=4)
    assert spread > 0.5


def test_calibration_bins_monotone_when_yhat_equals_y() -> None:
    rng = np.random.default_rng(1)
    n = 500
    yhat = rng.normal(size=n)
    y = yhat + rng.normal(0, 0.1, size=n)
    bins = calibration_bins(yhat, y, n_bins=5)
    assert (bins["pred_mean"].diff().dropna() > 0).all()
    assert (bins["realized_mean"].diff().dropna() > 0).all()


def test_population_stability_index_small_when_identical() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(size=2000)
    cur = rng.normal(size=2000)  # same distribution
    psi = population_stability_index(ref, cur, n_bins=10)
    assert psi < 0.1


def test_population_stability_index_large_when_shifted() -> None:
    rng = np.random.default_rng(3)
    ref = rng.normal(size=2000)
    cur = rng.normal(loc=2.0, scale=1.0, size=2000)  # shifted by 2 stddevs
    psi = population_stability_index(ref, cur, n_bins=10)
    assert psi > 0.5


def test_summarize_oof_metrics_returns_expected_keys(
    fitted_panel: PanelFrame,
) -> None:
    n = len(fitted_panel.y)
    oof = pd.DataFrame(
        {
            "yhat_ensemble": fitted_panel.y.to_numpy() * 0.5 + np.random.default_rng(0).normal(0, 0.01, size=n),
            "y": fitted_panel.y.to_numpy(),
            "w": fitted_panel.w.to_numpy(),
            "date": fitted_panel.meta["date"].to_numpy(),
            "ticker": fitted_panel.meta["ticker"].to_numpy(),
        }
    )
    summary = summarize_oof_metrics(oof)
    for k in ("weighted_r2", "cross_sectional_rank_ic", "time_series_ic", "decile_spread", "n_samples", "n_dates", "n_tickers"):
        assert k in summary
