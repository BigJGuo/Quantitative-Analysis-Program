"""Unit tests for the pure-numpy helpers in `signal.py`.

Verifies the spec's worked-out math on synthetic inputs:

- Robust standardization recovers the right scale on Gaussian data.
- Winsorization clips at the requested quantiles.
- Imputation fills NaNs with the column median.
- Target binarization implements the three spec definitions exactly.
- Purged K-fold respects the embargo and never overlaps train/val dates.
- Median blend matches numpy's np.median over the ensemble axis.
- Action rule fires iff any horizon probability exceeds tau.
- ROC-AUC matches the closed form for tiny hand-computed cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.supervised_autoencoder_mlp.signal import (
    action_rule,
    binarize_targets,
    blend_predictions,
    brier_score,
    build_feature_panel,
    impute_with_median,
    median_mad_standardize,
    purged_kfold_indices,
    reconstruction_error,
    roc_auc,
    rolling_log_returns,
    rolling_realized_vol,
    rolling_std,
    winsorize,
)


class TestRobustStandardize:
    def test_median_zero_unit_scale(self) -> None:
        # On a column with median 5 and MAD 1, the standardized column has
        # median 0 and scale 1.4826 (1 / 1.4826 in the denominator).
        rng = np.random.default_rng(0)
        col = rng.normal(5.0, 1.0, size=500).reshape(-1, 1)
        std, med, scale = median_mad_standardize(col)
        assert abs(med[0] - 5.0) < 0.2
        # scale = 1.4826 * MAD; for Gaussian noise MAD ≈ 0.6745, so scale ≈ 1.
        assert 0.85 < scale[0] < 1.15
        assert abs(float(np.median(std))) < 0.1

    def test_reapplies_supplied_stats(self) -> None:
        a = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        std_a, med, scale = median_mad_standardize(a)
        b = np.array([[10.0], [20.0]])
        # Reapply train-fit stats — caller passes MAD, not scale.
        std_b, _, _ = median_mad_standardize(b, median=med, mad=scale / 1.4826)
        # The transformation must be the same affine map: (x - med) / scale.
        expected = (b - med) / scale
        assert np.allclose(std_b, expected)

    def test_constant_column_does_not_blow_up(self) -> None:
        a = np.full((5, 1), 7.0)
        std, _, _ = median_mad_standardize(a)
        # All inputs equal the median -> standardized output is 0.
        assert np.allclose(std, 0.0)


class TestWinsorize:
    def test_clips_to_quantile_levels(self) -> None:
        col = np.linspace(0.0, 100.0, 101).reshape(-1, 1)
        clipped = winsorize(col, 0.05)
        assert clipped.min() == pytest.approx(5.0)
        assert clipped.max() == pytest.approx(95.0)

    def test_zero_quantile_is_noop(self) -> None:
        col = np.arange(10).reshape(-1, 1).astype(float)
        out = winsorize(col, 0.0)
        assert np.array_equal(out, col)

    def test_rejects_out_of_range(self) -> None:
        col = np.zeros((5, 1))
        with pytest.raises(ValueError, match="quantile"):
            winsorize(col, 0.6)


class TestImpute:
    def test_replaces_nans_with_median(self) -> None:
        a = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0], [np.nan, 8.0]])
        out, med, mask = impute_with_median(a)
        # column 0: median of [1, 2, 3] = 2
        assert out[3, 0] == pytest.approx(2.0)
        # column 1: median of [4, 6, 8] = 6
        assert out[0, 1] == pytest.approx(6.0)
        assert mask.sum() == 2
        assert med.shape == (2,)

    def test_all_nan_column_falls_back_to_zero(self) -> None:
        a = np.array([[np.nan], [np.nan], [np.nan]])
        out, _, _ = impute_with_median(a)
        assert np.all(out == 0.0)


class TestBinarizeTargets:
    def test_three_definitions(self) -> None:
        fwd = pd.DataFrame(
            {
                "fwd_1": [0.01, -0.01, 0.005, 0.0],
                "fwd_5": [0.02, -0.005, -0.001, 0.0],
                "fwd_10": [0.03, -0.02, -0.01, 0.0],
            }
        )
        out = binarize_targets(fwd, horizons=(1, 5, 10))
        assert list(out.columns) == ["pos_1d", "pos_med_5d", "pos_sum_5d"]
        # Row 0: every horizon positive -> all 1.
        assert out.iloc[0].tolist() == [1.0, 1.0, 1.0]
        # Row 1: every horizon negative -> all 0.
        assert out.iloc[1].tolist() == [0.0, 0.0, 0.0]
        # Row 2: primary positive (0.005 > 0); median is -0.001 (not > 0) -> 0;
        #        sum is -0.006 -> 0.
        assert out.iloc[2].tolist() == [1.0, 0.0, 0.0]
        # Row 3: all zero -> not strictly positive on any rule.
        assert out.iloc[3].tolist() == [0.0, 0.0, 0.0]

    def test_nan_propagation(self) -> None:
        fwd = pd.DataFrame(
            {
                "fwd_1": [0.01, np.nan, 0.0],
                "fwd_5": [0.01, 0.01, 0.0],
                "fwd_10": [0.01, 0.01, 0.0],
            }
        )
        out = binarize_targets(fwd, horizons=(1, 5, 10))
        assert out.iloc[1].isna().all()


class TestPurgedKFold:
    def test_train_val_disjoint(self) -> None:
        dates = np.repeat(np.arange(20), 3)  # 20 unique dates, 3 rows each
        splits = list(purged_kfold_indices(dates, n_folds=5, embargo=1))
        assert len(splits) == 5
        for s in splits:
            train_dates = set(dates[s.train_idx].tolist())
            val_dates = set(dates[s.val_idx].tolist())
            assert train_dates.isdisjoint(val_dates)

    def test_embargo_widens_buffer(self) -> None:
        dates = np.arange(50)  # one row per unique date
        for s in purged_kfold_indices(dates, n_folds=5, embargo=2):
            if s.val_idx.size == 0:
                continue
            v_min = dates[s.val_idx].min()
            v_max = dates[s.val_idx].max()
            train_dates = dates[s.train_idx]
            # No train date should sit within 2 of any validation date.
            assert (train_dates < v_min - 2).sum() + (train_dates > v_max + 2).sum() == len(train_dates)

    def test_rejects_bad_args(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            list(purged_kfold_indices(np.arange(5), n_folds=1))
        with pytest.raises(ValueError, match="embargo"):
            list(purged_kfold_indices(np.arange(5), n_folds=2, embargo=-1))


class TestBlendAndAction:
    def test_median_blend(self) -> None:
        a = np.array([[0.1, 0.9], [0.4, 0.6]])
        b = np.array([[0.2, 0.8], [0.5, 0.5]])
        c = np.array([[0.9, 0.1], [0.0, 1.0]])
        blended = blend_predictions([a, b, c])
        # Per-cell median:
        # row 0: [0.1, 0.2, 0.9] -> 0.2 ; [0.9, 0.8, 0.1] -> 0.8
        # row 1: [0.4, 0.5, 0.0] -> 0.4 ; [0.6, 0.5, 1.0] -> 0.6
        assert np.allclose(blended, [[0.2, 0.8], [0.4, 0.6]])

    def test_action_rule(self) -> None:
        probs = np.array([[0.3, 0.4], [0.6, 0.2], [0.49, 0.51]])
        a = action_rule(probs, threshold=0.5)
        assert a.tolist() == [0, 1, 1]

    def test_action_rule_rejects_threshold_extremes(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            action_rule(np.zeros((1, 2)), threshold=0.0)


class TestRocAuc:
    def test_perfect_separation(self) -> None:
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        y = np.array([0, 0, 1, 1])
        assert roc_auc(probs, y) == pytest.approx(1.0)

    def test_all_negative_default_half(self) -> None:
        probs = np.array([0.1, 0.2, 0.3])
        y = np.zeros(3, dtype=int)
        assert roc_auc(probs, y) == 0.5

    def test_random_around_half(self) -> None:
        rng = np.random.default_rng(7)
        probs = rng.uniform(0, 1, size=500)
        y = rng.integers(0, 2, size=500)
        assert 0.4 < roc_auc(probs, y) < 0.6


class TestBrierAndRecon:
    def test_brier_score_per_target(self) -> None:
        p = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[0.0, 1.0], [1.0, 0.0]])
        # Perfect predictions -> 0 Brier on every target.
        assert np.allclose(brier_score(p, y), [0.0, 0.0])

    def test_reconstruction_error_zero_on_match(self) -> None:
        x = np.random.default_rng(1).normal(size=(4, 5))
        assert np.allclose(reconstruction_error(x, x), 0.0)


class TestRollingHelpers:
    def test_rolling_log_returns(self) -> None:
        s = pd.Series([1.0, 2.0, 4.0, 8.0])
        out = rolling_log_returns(s, 1)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1] == pytest.approx(np.log(2.0))
        assert out.iloc[3] == pytest.approx(np.log(2.0))

    def test_rolling_std_and_rv(self) -> None:
        r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
        std = rolling_std(r, 3).dropna()
        assert std.shape[0] == 3
        rv = rolling_realized_vol(r, 3).dropna()
        assert rv.iloc[0] == pytest.approx(np.sqrt(0.01**2 + 0.01**2 + 0.02**2))


class TestFeaturePanel:
    def test_columns_present_and_sized(self) -> None:
        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100.0 * np.exp(np.cumsum(np.full(n, 0.001)))
        bars = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(n, 1_000_000),
            },
            index=idx,
        )
        panel = build_feature_panel(bars)
        for col in ("ret_1", "ret_5", "vol_5", "rv_20", "hl", "vol_ratio", "illiq"):
            assert col in panel.columns
        assert panel.shape[0] == n

    def test_rejects_missing_columns(self) -> None:
        bars = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing"):
            build_feature_panel(bars)
