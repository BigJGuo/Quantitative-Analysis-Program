"""Unit tests for the pure-numpy helpers in `signal.py`.

Verifies the spec's math on synthetic inputs:

- Feature engineering: log returns, intraday range, gap, vol-ratio, etc.
- Robust standardize: median-zero, unit-scale on Gaussian columns.
- PCA reconstruction: zero error on rank-d data with bottleneck >= d.
- Isolation forest: outliers receive a higher score than the clean core.
- Mahalanobis: matches the closed form on diagonal-covariance inputs.
- `c(ψ)` matches the closed form `2H(ψ-1) - 2(ψ-1)/ψ`.
- Sentiment scalar respects the runtime label index.
- Exp-decay weights match the analytical formula.
- Sentence splitter handles abbreviations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detection_finbert.signal import (
    abs_return_zscore,
    build_anomaly_feature_panel,
    c_psi,
    cross_sectional_rank,
    exp_decay_weight,
    fit_iforest,
    fit_mahalanobis,
    iforest_score,
    impute_and_clip,
    intraday_range,
    log_return_1d,
    log_volume,
    ma_drift,
    mahalanobis_score,
    majority_vote_flags,
    median_mad_standardize,
    open_close_gap,
    pca_fit,
    pca_reconstruct,
    pca_reconstruction_score,
    pca_residual_attribution,
    realized_vol,
    score_all_detectors,
    sentiment_scalar,
    split_sentences,
    threshold_from_quantile,
    top_feature_attribution,
    vol_ratio,
    weighted_sentiment,
)
from src.models.anomaly_detection_finbert.types import (
    DetectorFit,
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class TestLogReturn:
    def test_matches_log_diff(self) -> None:
        s = pd.Series([100.0, 101.0, 102.01])
        r = log_return_1d(s)
        assert np.isnan(r.iloc[0])
        assert r.iloc[1] == pytest.approx(math.log(101.0 / 100.0))
        assert r.iloc[2] == pytest.approx(math.log(102.01 / 101.0))


class TestIntradayRange:
    def test_basic(self) -> None:
        out = intraday_range(
            pd.Series([105.0]), pd.Series([95.0]), pd.Series([100.0])
        )
        assert out.iloc[0] == pytest.approx(0.10)


class TestOpenCloseGap:
    def test_basic(self) -> None:
        out = open_close_gap(pd.Series([102.0]), pd.Series([100.0]))
        assert out.iloc[0] == pytest.approx(0.02)


class TestLogVolume:
    def test_zero_safe(self) -> None:
        out = log_volume(pd.Series([0.0, 100.0]))
        assert out.iloc[0] == 0.0
        assert out.iloc[1] == pytest.approx(math.log1p(100.0))


class TestVolRatio:
    def test_ratio_equals_one_when_constant(self) -> None:
        s = pd.Series(np.full(40, 100.0))
        out = vol_ratio(s, 20)
        assert out.dropna().eq(1.0).all()


class TestRealizedVol:
    def test_zero_returns_zero(self) -> None:
        s = pd.Series(np.zeros(30))
        out = realized_vol(s, 20)
        assert out.dropna().eq(0.0).all()


class TestMADrift:
    def test_constant_series_zero_drift(self) -> None:
        s = pd.Series(np.full(60, 100.0))
        out = ma_drift(s, 50)
        assert out.dropna().eq(0.0).all()


class TestAbsZScore:
    def test_constant_returns_nan_zscore(self) -> None:
        # On a constant series the rolling std is 0 -> z-score divides by NaN.
        s = pd.Series(np.zeros(100))
        out = abs_return_zscore(s, 60)
        assert out.iloc[-1] != out.iloc[-1] or out.iloc[-1] == 0.0


class TestBuildPanel:
    def test_columns_match_spec(self, synthetic_yf_bars: pd.DataFrame) -> None:
        # The helper expects a DatetimeIndex.
        bars = synthetic_yf_bars.set_index("Date")
        panel = build_anomaly_feature_panel(bars)
        expected = {
            "log_return_1d",
            "intraday_range",
            "open_close_gap",
            "log_volume",
            "vol_ratio_20d",
            "realized_vol_20d",
            "ma_drift_50",
            "abs_return_zscore",
        }
        assert set(panel.columns) == expected
        assert len(panel) == len(bars)


class TestCrossSectionalRank:
    def test_rank_in_unit_interval(self) -> None:
        df = pd.DataFrame(
            {
                "A": [0.01, -0.02, 0.03],
                "B": [-0.01, 0.02, -0.03],
                "C": [0.02, 0.01, 0.0],
            }
        )
        ranks = cross_sectional_rank(df)
        assert ranks.values.min() >= 0.0
        assert ranks.values.max() <= 1.0
        # On row 0: A=0.01, B=-0.01, C=0.02 -> ranks (2/3, 1/3, 3/3) pct.
        assert ranks.iloc[0]["C"] == pytest.approx(1.0)
        assert ranks.iloc[0]["B"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Robust preprocessing
# ---------------------------------------------------------------------------


class TestRobustStandardize:
    def test_median_zero_unit_scale(self) -> None:
        rng = np.random.default_rng(0)
        col = rng.normal(5.0, 1.0, size=500).reshape(-1, 1)
        std, med, scale = median_mad_standardize(col)
        assert abs(med[0] - 5.0) < 0.2
        assert 0.85 < scale[0] < 1.15
        assert abs(float(np.median(std))) < 0.1

    def test_reapplies_supplied_stats(self) -> None:
        a = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        std_a, med, scale = median_mad_standardize(a)
        b = np.array([[10.0]])
        std_b, _, _ = median_mad_standardize(b, median=med, scale=scale)
        assert std_b[0, 0] == pytest.approx((10.0 - med[0]) / scale[0])

    def test_constant_column_does_not_blow_up(self) -> None:
        a = np.full((5, 1), 7.0)
        std, _, _ = median_mad_standardize(a)
        assert np.allclose(std, 0.0)


class TestImputeAndClip:
    def test_clips_at_boundaries(self) -> None:
        x = np.array([[10.0, -10.0], [np.nan, 2.0]])
        out = impute_and_clip(x, clip_z=5.0)
        assert out[0, 0] == 5.0
        assert out[0, 1] == -5.0
        assert out[1, 0] == 0.0  # NaN -> 0
        assert out[1, 1] == 2.0


# ---------------------------------------------------------------------------
# PCA reconstruction (linear AE)
# ---------------------------------------------------------------------------


class TestPCAReconstruction:
    def test_zero_error_on_full_rank(self, rng: np.random.Generator) -> None:
        X = rng.normal(0.0, 1.0, size=(100, 4))
        mean, V, _ = pca_fit(X, n_components=4)
        scores = pca_reconstruction_score(X, mean, V)
        assert np.allclose(scores, 0.0, atol=1e-8)

    def test_outlier_scores_higher_than_clean(
        self, panel_with_outliers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, outlier_idx = panel_with_outliers
        clean_idx = np.setdiff1d(np.arange(X.shape[0]), outlier_idx)
        mean, V, _ = pca_fit(X[clean_idx], n_components=2)
        scores = pca_reconstruction_score(X, mean, V)
        # Mean outlier score must clearly exceed the clean p99.
        clean_p99 = np.quantile(scores[clean_idx], 0.99)
        assert scores[outlier_idx].mean() > 5 * clean_p99

    def test_reconstruct_round_trip(self, rng: np.random.Generator) -> None:
        # Stuff data along a 1-d line; PCA with d=1 reconstructs exactly.
        n = 50
        direction = np.array([1.0, 2.0, -1.0])
        t = rng.normal(0.0, 1.0, size=n)
        X = t[:, None] * direction[None, :] + 3.0
        mean, V, _ = pca_fit(X, n_components=1)
        recon = pca_reconstruct(X, mean, V)
        assert np.allclose(X, recon, atol=1e-8)

    def test_residual_attribution_matches_difference(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(0.0, 1.0, size=(40, 3))
        mean, V, _ = pca_fit(X, n_components=2)
        x = X[0]
        residuals = pca_residual_attribution(x, mean, V)
        x_hat = pca_reconstruct(x.reshape(1, -1), mean, V)[0]
        assert np.allclose(residuals, (x - x_hat) ** 2)


# ---------------------------------------------------------------------------
# Isolation forest
# ---------------------------------------------------------------------------


class TestCPsi:
    def test_closed_form(self) -> None:
        # c(2) = 2*H(1) - 2*(1)/2; H(1) = ln(1) + gamma ≈ 0.5772.
        gamma = 0.5772156649015329
        assert c_psi(2) == pytest.approx(2.0 * gamma - 1.0)

    def test_monotone_increasing(self) -> None:
        a = c_psi(50)
        b = c_psi(500)
        assert b > a

    def test_handles_psi_le_1(self) -> None:
        assert c_psi(0) == 1.0
        assert c_psi(1) == 1.0


class TestIsolationForest:
    def test_outliers_score_higher(
        self, panel_with_outliers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, outlier_idx = panel_with_outliers
        trees = fit_iforest(X, n_trees=80, psi=64, seed=0)
        scores = iforest_score(X, trees, psi=64)
        clean_idx = np.setdiff1d(np.arange(X.shape[0]), outlier_idx)
        # Outlier mean score should clearly exceed clean p95.
        assert scores[outlier_idx].mean() > np.quantile(scores[clean_idx], 0.95)
        # Scores live in (0, 1].
        assert scores.min() > 0.0
        assert scores.max() <= 1.0

    def test_deterministic_under_seed(self, gaussian_panel: np.ndarray) -> None:
        a = fit_iforest(gaussian_panel, n_trees=10, psi=32, seed=7)
        b = fit_iforest(gaussian_panel, n_trees=10, psi=32, seed=7)
        scores_a = iforest_score(gaussian_panel, a, psi=32)
        scores_b = iforest_score(gaussian_panel, b, psi=32)
        assert np.allclose(scores_a, scores_b)


# ---------------------------------------------------------------------------
# Mahalanobis
# ---------------------------------------------------------------------------


class TestMahalanobis:
    def test_diagonal_matches_closed_form(self, rng: np.random.Generator) -> None:
        X = rng.normal(0.0, 1.0, size=(500, 3))
        mean, inv_cov = fit_mahalanobis(X, shrinkage=1e-6)
        # On standardized Gaussian data, Mahalanobis ≈ sum of squared z-scores.
        z = (X - mean)
        # cov is ~ I, so M(x) ≈ ||z||^2.
        scores = mahalanobis_score(X, mean, inv_cov)
        sq_norm = (z * z).sum(axis=1)
        # The shrinkage and finite-sample cov bias the ratio slightly,
        # but the means should agree within 10%.
        assert abs(scores.mean() - sq_norm.mean()) / sq_norm.mean() < 0.1

    def test_singular_panel_does_not_crash(self) -> None:
        # Constant column => singular covariance; shrinkage saves us.
        X = np.column_stack([np.linspace(0, 1, 20), np.full(20, 0.5)])
        mean, inv_cov = fit_mahalanobis(X, shrinkage=1e-3)
        out = mahalanobis_score(X, mean, inv_cov)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Ensemble glue
# ---------------------------------------------------------------------------


class TestScoreAllAndThresholds:
    def test_score_all_detectors_keys(self, gaussian_panel: np.ndarray) -> None:
        mean, V, _ = pca_fit(gaussian_panel, n_components=3)
        trees = fit_iforest(gaussian_panel, n_trees=10, psi=32, seed=0)
        mm, inv = fit_mahalanobis(gaussian_panel, shrinkage=1e-3)
        # Hand-built DetectorFit to feed score_all_detectors.
        p = gaussian_panel.shape[1]
        fit = DetectorFit(
            pca_mean=mean,
            pca_components=V,
            pca_explained_var=np.ones(V.shape[0]),
            iforest_trees=trees,
            iforest_psi=32,
            iforest_c_psi=c_psi(32),
            mahal_mean=mm,
            mahal_inv_cov=inv,
            feature_median=np.zeros(p),
            feature_scale=np.ones(p),
        )
        out = score_all_detectors(gaussian_panel, fit)
        assert set(out.keys()) == {"ae", "iforest", "mahalanobis"}
        assert all(arr.shape[0] == gaussian_panel.shape[0] for arr in out.values())

    def test_threshold_matches_quantile(self) -> None:
        scores = np.linspace(0.0, 100.0, 1001)
        q = threshold_from_quantile(scores, contamination=0.01)
        assert q == pytest.approx(99.0, abs=0.5)


class TestMajorityVote:
    def test_two_of_three(self) -> None:
        assert majority_vote_flags({"a": True, "b": True, "c": False}, min_votes=2)
        assert not majority_vote_flags(
            {"a": False, "b": True, "c": False}, min_votes=2
        )


class TestTopFeatureAttribution:
    def test_top_k_descending(self) -> None:
        residuals = np.array([0.1, 9.9, 0.5, 4.4])
        names = ("a", "b", "c", "d")
        top = top_feature_attribution(residuals, names, k=2)
        assert top[0][0] == "b"
        assert top[1][0] == "d"


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------


class TestSentimentScalar:
    def test_label_order_a(self) -> None:
        # ProsusAI/FinBERT label order: {positive:0, negative:1, neutral:2}.
        probs = np.array([0.7, 0.2, 0.1])
        idx = {"positive": 0, "negative": 1, "neutral": 2}
        assert sentiment_scalar(probs, idx) == pytest.approx(0.5)

    def test_label_order_b_alphabetical(self) -> None:
        # Alphabetical: {negative:0, neutral:1, positive:2}.
        probs = np.array([0.2, 0.1, 0.7])
        idx = {"negative": 0, "neutral": 1, "positive": 2}
        # Scalar must still be p_pos - p_neg = 0.5 regardless of layout.
        assert sentiment_scalar(probs, idx) == pytest.approx(0.5)


class TestExpDecay:
    def test_zero_age_full_weight(self) -> None:
        assert exp_decay_weight(0.0, 24.0) == pytest.approx(1.0)

    def test_one_period_one_over_e(self) -> None:
        assert exp_decay_weight(24.0, 24.0) == pytest.approx(math.exp(-1.0))

    def test_negative_age_treated_as_zero(self) -> None:
        assert exp_decay_weight(-5.0, 24.0) == pytest.approx(1.0)


class TestWeightedSentiment:
    def test_simple_weighted_average(self) -> None:
        # 0.5 -> weight 1, -0.3 -> weight 2: blended = (0.5 - 0.6) / 3 = -0.0333.
        s = weighted_sentiment([0.5, -0.3], [1.0, 2.0])
        assert s == pytest.approx((-0.1) / 3.0)

    def test_zero_weights_returns_zero(self) -> None:
        assert weighted_sentiment([0.5, -0.3], [0.0, 0.0]) == 0.0


class TestSplitSentences:
    def test_basic_split(self) -> None:
        assert split_sentences("Hello world. Foo bar!") == [
            "Hello world.",
            "Foo bar!",
        ]

    def test_abbreviation_preserved(self) -> None:
        out = split_sentences("Apple Inc. reported earnings. Stock rose 5%.")
        # "Inc." should NOT split.
        assert out[0].startswith("Apple Inc.")
        assert len(out) == 2

    def test_empty_returns_empty(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   ") == []
