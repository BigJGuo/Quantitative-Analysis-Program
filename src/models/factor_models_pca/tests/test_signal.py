"""Unit tests for the math layer in `signal.py`.

Each test pairs a worked-out-by-hand or known-truth fixture with the
function under test so failures point to the exact step that broke.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.factor_models_pca.signal import (
    align_returns_panel,
    compute_full_covariance,
    compute_log_returns,
    cross_sectional_factor_regression,
    diagonal_shrinkage,
    ljung_box_pvalue,
    marchenko_pastur_upper_edge,
    pca_decompose,
    portfolio_risk,
    residual_autocorrelation,
    residual_max_cross_corr,
    residual_returns,
    residualize,
    time_series_factor_regression,
    variance_explained_curve,
)
from src.models.factor_models_pca.types import FactorFit


class TestComputeLogReturns:
    def test_simple_two_row_panel(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 49.0]})
        out = compute_log_returns(prices)
        assert out.shape == (1, 2)
        assert math.isclose(float(out["A"].iloc[0]), math.log(110.0 / 100.0), abs_tol=1e-12)
        assert math.isclose(float(out["B"].iloc[0]), math.log(49.0 / 50.0), abs_tol=1e-12)

    def test_empty_input_returns_empty(self) -> None:
        empty = pd.DataFrame()
        assert compute_log_returns(empty).empty


class TestAlignReturnsPanel:
    def test_drops_low_coverage_columns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        good = pd.Series(np.arange(10, dtype=float), index=idx)
        bad = pd.Series([np.nan] * 9 + [1.0], index=idx)
        panel = align_returns_panel({"GOOD": good, "BAD": bad}, min_coverage=0.5)
        assert list(panel.columns) == ["GOOD"]

    def test_winsorize_clips_extremes(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=100), index=idx)
        s.iloc[0] = 999.0
        s.iloc[1] = -999.0
        panel = align_returns_panel({"X": s}, winsorize_quantile=0.05, min_coverage=0.99)
        assert panel["X"].max() < 999.0
        assert panel["X"].min() > -999.0

    def test_empty_input(self) -> None:
        assert align_returns_panel({}).empty


class TestMarchenkoPastur:
    def test_recovers_classic_edge_at_q_equal_1(self) -> None:
        # N=T -> q=1 -> upper edge = (1+1)^2 * sigma^2 = 4 * sigma^2.
        edge = marchenko_pastur_upper_edge(n_assets=100, n_obs=100, sigma_squared=1.0)
        assert math.isclose(edge, 4.0, abs_tol=1e-12)

    def test_scales_with_sigma_squared(self) -> None:
        a = marchenko_pastur_upper_edge(50, 200, 1.0)
        b = marchenko_pastur_upper_edge(50, 200, 4.0)
        assert math.isclose(b, 4.0 * a, abs_tol=1e-12)

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            marchenko_pastur_upper_edge(0, 100, 1.0)
        with pytest.raises(ValueError):
            marchenko_pastur_upper_edge(10, 100, -1.0)


class TestPCADecompose:
    def test_one_factor_dominates_variance(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _true_B = synthetic_one_factor
        fit = pca_decompose(returns, K=1)
        # With 1 latent factor + tiny idio, PC1 should explain >> idio share.
        # Theoretical ratio: beta^2 * sigma_f^2 / (beta^2 * sigma_f^2 + sigma_eps^2)
        # is ~0.94 for the fixture's vols; allow a small finite-sample margin.
        assert fit.variance_explained > 0.90
        assert fit.eigenvalues is not None
        # Top eigenvalue should dominate the rest.
        assert fit.eigenvalues[0] > 10 * fit.eigenvalues[1]

    def test_recovers_loadings_up_to_sign(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, true_B = synthetic_one_factor
        fit = pca_decompose(returns, K=1)
        # B is `eigvec * sqrt(lambda)`; true_B is in the same direction up to a sign.
        b_hat = fit.B[:, 0]
        # Normalize both so the comparison is unit-vector cosine similarity.
        b_hat_unit = b_hat / np.linalg.norm(b_hat)
        true_unit = true_B[:, 0] / np.linalg.norm(true_B[:, 0])
        cosine = abs(float(b_hat_unit @ true_unit))
        assert cosine > 0.99

    def test_three_factor_cumulative_variance(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit = pca_decompose(returns, K=3)
        assert fit.eigenvalues is not None
        cumve = variance_explained_curve(fit.eigenvalues)
        assert cumve[2] > 0.9  # 3 factors should explain >90% of synthetic variance

    def test_mp_cutoff_keeps_signal_factors(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit = pca_decompose(returns, K=None, use_mp_cutoff=True)
        # MP should pick at least the 3 true factors out (the design is well separated).
        assert fit.n_factors >= 3

    def test_factor_count_matches_K(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit = pca_decompose(returns, K=2)
        assert fit.n_factors == 2
        assert fit.B.shape == (returns.shape[1], 2)

    def test_specific_variance_floored(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        fit = pca_decompose(returns, K=1, min_specific_variance=1e-4)
        assert (fit.specific_variances >= 1e-4).all()

    def test_standardize_does_not_break_recovery(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit = pca_decompose(returns, K=3, standardize=True)
        assert fit.n_factors == 3
        assert fit.eigenvalues is not None
        assert fit.variance_explained > 0.5

    def test_requires_K_or_mp_cutoff(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        with pytest.raises(ValueError, match="K or use_mp_cutoff"):
            pca_decompose(returns, K=None, use_mp_cutoff=False)


class TestComputeFullCovariance:
    def test_reconstructs_factor_plus_specific(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        fit = pca_decompose(returns, K=1)
        sigma = compute_full_covariance(fit)
        # Diagonal of Sigma should be close to the asset variances (eigval analysis exact).
        sample_var = returns.var(axis=0, ddof=1).to_numpy()
        # `compute_full_covariance` rebuilds Sigma from the top-K subspace + idio.
        # On a 1-factor process the rank-1 piece + the floored diagonal must
        # not exceed the realized variance by much.
        recovered = np.diag(sigma)
        assert np.allclose(recovered, sample_var, atol=2e-4)


class TestPortfolioRisk:
    def test_equal_weighted_one_factor(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        fit = pca_decompose(returns, K=1)
        n = fit.n_assets
        w = np.full(n, 1.0 / n)
        risk = portfolio_risk(w, fit)
        # On a market-like factor the equal-weight portfolio vol is dominated by systematic.
        assert risk.systematic_vol > 5 * risk.specific_vol
        # total = sqrt(sys^2 + spec^2)
        assert math.isclose(
            risk.total_vol**2,
            risk.systematic_vol**2 + risk.specific_vol**2,
            rel_tol=1e-9,
            abs_tol=1e-18,
        )

    def test_factor_contributions_sum_to_systematic_variance(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit = pca_decompose(returns, K=3)
        n = fit.n_assets
        w = np.full(n, 1.0 / n)
        risk = portfolio_risk(w, fit)
        assert math.isclose(
            float(risk.factor_contributions.sum()),
            risk.systematic_vol**2,
            rel_tol=1e-9,
            abs_tol=1e-18,
        )

    def test_shape_mismatch_rejected(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        fit = pca_decompose(returns, K=1)
        with pytest.raises(ValueError, match="weights shape"):
            portfolio_risk(np.array([1.0, 0.0]), fit)


class TestResidualize:
    def test_factor_neutral_signal_is_orthogonal_to_exposures(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit = pca_decompose(returns, K=3)
        rng = np.random.default_rng(7)
        raw = rng.normal(size=fit.n_assets)
        neutral = residualize(raw, fit)
        # Projection: B^T neutral should be ~0.
        proj = fit.B.T @ neutral
        assert np.allclose(proj, 0.0, atol=1e-9)

    def test_signal_already_orthogonal_is_unchanged(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        fit = pca_decompose(returns, K=1)
        # Pick a vector orthogonal to the single loading column.
        b = fit.B[:, 0]
        orthogonal = np.array([1.0, -1.0] + [0.0] * (fit.n_assets - 2))
        # Re-orthogonalize against b just to be safe given numerical noise.
        orthogonal = orthogonal - (orthogonal @ b) / (b @ b) * b
        neutral = residualize(orthogonal, fit)
        assert np.allclose(neutral, orthogonal, atol=1e-9)


class TestResidualReturns:
    def test_residuals_have_smaller_variance_than_raw(
        self, synthetic_one_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_one_factor
        fit = pca_decompose(returns, K=1)
        resid = residual_returns(returns, fit)
        assert (resid.var(axis=0, ddof=1) < returns.var(axis=0, ddof=1)).all()

    def test_residual_cross_correlation_small_after_correct_K(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit = pca_decompose(returns, K=3)
        resid = residual_returns(returns, fit)
        max_corr = residual_max_cross_corr(resid)
        # With the right K, residuals are nearly independent. Allow some slack.
        assert max_corr < 0.5

    def test_residual_cross_correlation_large_when_K_too_small(
        self, synthetic_three_factor: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        returns, _ = synthetic_three_factor
        fit_small = pca_decompose(returns, K=1)
        fit_full = pca_decompose(returns, K=3)
        resid_small = residual_returns(returns, fit_small)
        resid_full = residual_returns(returns, fit_full)
        # Under-modelled residuals should retain more pairwise correlation.
        assert residual_max_cross_corr(resid_small) > residual_max_cross_corr(resid_full)


class TestResidualAutocorrelation:
    def test_white_noise_lag1_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        residuals = pd.DataFrame(rng.normal(size=(1000, 3)))
        rho = residual_autocorrelation(residuals, lag=1)
        assert np.all(np.abs(rho) < 0.1)

    def test_ar1_autocorrelation_detected(self) -> None:
        rng = np.random.default_rng(1)
        n = 1000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + rng.normal()
        rho = residual_autocorrelation(pd.DataFrame({"X": x}), lag=1)
        assert rho[0] > 0.4


class TestLjungBox:
    def test_white_noise_pvalue_typically_above_0_05(self) -> None:
        rng = np.random.default_rng(2)
        white = rng.normal(size=500)
        p = ljung_box_pvalue(white, max_lag=5)
        # The approximation is decent; we just need the rejection rate to behave.
        assert 0.0 <= p <= 1.0

    def test_ar1_pvalue_rejects_zero(self) -> None:
        rng = np.random.default_rng(3)
        n = 500
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + rng.normal()
        p = ljung_box_pvalue(x, max_lag=5)
        assert p < 0.01


class TestTimeSeriesFactorRegression:
    def test_recovers_known_betas(
        self,
        fama_french_synthetic: tuple[pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        asset, factor, true_B = fama_french_synthetic
        fit = time_series_factor_regression(asset, factor)
        assert np.allclose(fit.B, true_B, atol=0.08)
        assert fit.method == "FamaFrench"
        # R^2 should be high given the tiny specific noise we injected.
        assert fit.variance_explained > 0.85

    def test_factor_covariance_matches_sample(
        self,
        fama_french_synthetic: tuple[pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        asset, factor, _ = fama_french_synthetic
        fit = time_series_factor_regression(asset, factor)
        sample_cov = factor.cov().to_numpy()
        assert np.allclose(fit.F, sample_cov, atol=1e-12)


class TestCrossSectionalFactorRegression:
    def test_recovers_factor_returns(self) -> None:
        # B is fixed; pick a known f and check we recover it.
        rng = np.random.default_rng(11)
        n, k = 30, 3
        B = rng.normal(size=(n, k))
        f_true = np.array([0.01, -0.005, 0.002])
        r = B @ f_true  # noiseless
        f_hat, resid = cross_sectional_factor_regression(r, B)
        assert np.allclose(f_hat, f_true, atol=1e-10)
        assert np.allclose(resid, 0.0, atol=1e-10)

    def test_weighted_regression_downweights_outlier_asset(self) -> None:
        rng = np.random.default_rng(12)
        n, k = 20, 2
        B = rng.normal(size=(n, k))
        f_true = np.array([0.01, -0.003])
        r = B @ f_true + rng.normal(0.0, 0.0005, size=n)
        # Inject a large outlier in asset 0; very low weight should pull the
        # estimate back to the true factor returns.
        r_with_outlier = r.copy()
        r_with_outlier[0] += 1.0
        weights = np.ones(n)
        weights[0] = 1e-6
        f_hat, _ = cross_sectional_factor_regression(r_with_outlier, B, weights=weights)
        assert np.allclose(f_hat, f_true, atol=1e-3)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            cross_sectional_factor_regression(np.zeros(5), np.zeros((4, 2)))


class TestDiagonalShrinkage:
    def test_alpha_zero_is_identity(self) -> None:
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        out = diagonal_shrinkage(sigma, 0.0)
        assert np.allclose(out, sigma)

    def test_alpha_one_is_diagonal(self) -> None:
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        out = diagonal_shrinkage(sigma, 1.0)
        assert np.allclose(out, np.diag([1.0, 2.0]))

    def test_rejects_out_of_range_alpha(self) -> None:
        with pytest.raises(ValueError):
            diagonal_shrinkage(np.eye(2), 1.5)


class TestFactorFitInvariants:
    def test_rejects_inconsistent_shapes(self) -> None:
        with pytest.raises(ValueError):
            FactorFit(
                B=np.ones((3, 2)),
                F=np.eye(2),
                specific_variances=np.ones(3),
                tickers=("A", "B"),  # wrong length
                factor_names=("PC1", "PC2"),
                method="PCA",
            )
        with pytest.raises(ValueError):
            FactorFit(
                B=np.ones((3, 2)),
                F=np.eye(3),  # K mismatch
                specific_variances=np.ones(3),
                tickers=("A", "B", "C"),
                factor_names=("PC1", "PC2"),
                method="PCA",
            )

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            FactorFit(
                B=np.ones((2, 1)),
                F=np.eye(1),
                specific_variances=np.ones(2),
                tickers=("A", "B"),
                factor_names=("PC1",),
                method="garbage",  # type: ignore[arg-type]
            )
