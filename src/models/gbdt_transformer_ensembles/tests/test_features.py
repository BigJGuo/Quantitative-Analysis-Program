"""Unit tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gbdt_transformer_ensembles.features import (
    build_feature_panel,
    compute_log_returns,
    fill_nans_for_linear,
    fill_nans_for_trees,
)
from src.models.gbdt_transformer_ensembles.types import FeatureSpec, PanelFrame


def test_compute_log_returns_matches_definition() -> None:
    """`r_t^{(k)} = log(P_t) - log(P_{t-k})` should be exact."""

    close = pd.Series([100.0, 101.0, 102.5, 99.5, 100.2])
    r1 = compute_log_returns(close, 1)
    expected = np.log(close.to_numpy()[1:]) - np.log(close.to_numpy()[:-1])
    np.testing.assert_allclose(r1.dropna().to_numpy(), expected)
    assert np.isnan(r1.iloc[0])

    r2 = compute_log_returns(close, 2)
    expected_2 = np.log(close.to_numpy()[2:]) - np.log(close.to_numpy()[:-2])
    np.testing.assert_allclose(r2.dropna().to_numpy(), expected_2)


def test_compute_log_returns_horizon_validation() -> None:
    close = pd.Series([100.0, 101.0])
    with pytest.raises(ValueError):
        compute_log_returns(close, 0)


def test_build_feature_panel_shapes(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> None:
    panel = build_feature_panel(synthetic_prices, spec=small_feature_spec)
    assert isinstance(panel, PanelFrame)
    assert panel.n_rows > 0
    # Every row has both target and weight defined.
    assert panel.y.notna().all()
    assert panel.w.notna().all()
    # Feature names align with columns.
    assert tuple(panel.X.columns) == panel.feature_names
    # Cross-sectional ranks: each cs_rank_* column must be in [0, 1].
    rank_cols = [c for c in panel.feature_names if c.startswith("cs_rank_")]
    assert len(rank_cols) > 0
    for c in rank_cols:
        col = panel.X[c].dropna()
        assert ((col >= 0.0) & (col <= 1.0)).all()


def test_build_feature_panel_clips_target(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> None:
    panel = build_feature_panel(synthetic_prices, spec=small_feature_spec)
    cap = small_feature_spec.clip_target
    assert cap is not None
    assert panel.y.abs().max() <= cap + 1e-12


def test_build_feature_panel_rejects_missing_columns() -> None:
    bad = {"X": pd.DataFrame({"Close": [1.0, 1.1]})}
    with pytest.raises(ValueError):
        build_feature_panel(bad, spec=FeatureSpec())


def test_fill_nans_for_trees_uses_sentinel() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.inf, 2.0, -np.inf]})
    arr = fill_nans_for_trees(df, sentinel=-1.0)
    assert arr.shape == (3, 2)
    assert arr[1, 0] == -1.0
    assert arr[0, 1] == -1.0
    assert arr[2, 1] == -1.0


def test_fill_nans_for_linear_returns_mask() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    arr, mask = fill_nans_for_linear(df)
    assert arr[1, 0] == 0.0
    assert mask[1, 0] == 1.0
    assert mask[0, 0] == 0.0


def test_target_horizon_matches_spec(
    synthetic_prices: dict[str, pd.DataFrame],
) -> None:
    """For one ticker, manually compute `r5_forward` and verify alignment."""

    spec = FeatureSpec(
        return_horizons=(1,),
        lag_count=0,
        vol_windows=(5,),
        mean_windows=(5,),
        target_horizon=5,
        clip_target=None,
        include_cross_sectional_ranks=False,
        include_calendar=False,
        benchmark_ticker=None,
        min_history_days=10,
    )
    one_ticker = {"T0": synthetic_prices["T0"].copy()}
    panel = build_feature_panel(one_ticker, spec=spec)
    close = synthetic_prices["T0"]["Close"]
    log_p = np.log(close.to_numpy())
    expected_y = log_p[5:] - log_p[:-5]  # length T-5

    # The panel drops the first min_history_days rows AND the last
    # target_horizon rows. Reconstruct the slice it should have kept.
    mask = (panel.meta["ticker"] == "T0").to_numpy()
    sub_y = panel.y.to_numpy()[mask]
    start = spec.min_history_days
    end = start + len(sub_y)
    np.testing.assert_allclose(
        sub_y, expected_y[start:end], atol=1e-12
    )
