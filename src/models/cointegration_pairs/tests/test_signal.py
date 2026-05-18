"""Unit tests for the pure math layer (`signal.py`).

Every test that estimates a parameter from synthetic data uses an explicit
ground-truth fixture so the assertion is a numerical bound, not a smoke test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.cointegration_pairs.signal import (
    adf_test,
    align_pair,
    compute_log_prices,
    engle_granger_test,
    fit_ar1,
    fit_ou,
    half_life_from_phi,
    kalman_dynamic_beta,
    ljung_box_pvalue,
    ols_intercept_slope,
    pair_position_path,
    realized_half_life,
    rolling_eg_residual_tstat,
    rolling_zscore_moments,
    select_dependent_orientation,
    static_zscore,
)
from src.models.cointegration_pairs.types import TradingRule


class TestComputeLogPrices:
    def test_log_prices_basic(self) -> None:
        prices = pd.Series([1.0, math.e, math.e**2], index=pd.RangeIndex(3))
        log_p = compute_log_prices(prices)
        assert np.allclose(log_p.to_numpy(), [0.0, 1.0, 2.0])

    def test_drops_nonpositive(self) -> None:
        prices = pd.Series([1.0, 0.0, -1.0, 2.0])
        log_p = compute_log_prices(prices)
        assert len(log_p) == 2

    def test_empty(self) -> None:
        out = compute_log_prices(pd.Series([], dtype=float))
        assert out.empty


class TestAlignPair:
    def test_inner_join(self) -> None:
        a = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3))
        b = pd.Series([10.0, 20.0], index=pd.RangeIndex(2))
        aa, bb = align_pair(a, b)
        assert len(aa) == 2
        assert aa.index.equals(bb.index)

    def test_drops_nans(self) -> None:
        idx = pd.RangeIndex(4)
        a = pd.Series([1.0, np.nan, 3.0, 4.0], index=idx)
        b = pd.Series([10.0, 20.0, 30.0, np.nan], index=idx)
        aa, bb = align_pair(a, b)
        assert len(aa) == 2


class TestOLS:
    def test_recovers_known_line(self) -> None:
        # y = 0.5 + 2.0 * x, no noise -> exact recovery.
        x = np.linspace(0, 10, 50)
        y = 0.5 + 2.0 * x
        alpha, beta, resid = ols_intercept_slope(y, x)
        assert math.isclose(alpha, 0.5, abs_tol=1e-9)
        assert math.isclose(beta, 2.0, abs_tol=1e-9)
        assert np.allclose(resid, 0.0, atol=1e-9)

    def test_zero_variance_regressor_raises(self) -> None:
        x = np.ones(10)
        y = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="zero variance"):
            ols_intercept_slope(y, x)


class TestADF:
    def test_stationary_series_rejects_unit_root(
        self, synthetic_ar1_path: tuple[np.ndarray, dict[str, float]]
    ) -> None:
        z, _ = synthetic_ar1_path
        result = adf_test(z, n_lags=1, regression="c")
        # AR(1) with phi=0.9, T=5000 should comfortably reject I(1) at 5%.
        assert result.rejects_unit_root("5%")
        assert result.test_type == "adf"

    def test_random_walk_fails_to_reject(
        self, rng: np.random.Generator
    ) -> None:
        z = np.cumsum(rng.normal(0.0, 1.0, size=500))
        result = adf_test(z, n_lags=1, regression="c")
        assert not result.rejects_unit_root("5%")

    def test_engle_granger_critical_values_more_conservative(self) -> None:
        adf = adf_test(np.arange(10, dtype=float) + 1.0, n_lags=0, regression="c")
        eg = adf_test(
            np.arange(10, dtype=float) + 1.0, n_lags=0, regression="nc", test_type="engle_granger"
        )
        assert eg.critical_values["5%"] < adf.critical_values["5%"]

    def test_invalid_level_raises(self) -> None:
        z = np.linspace(0, 1, 30)
        result = adf_test(z, n_lags=1)
        with pytest.raises(KeyError):
            result.rejects_unit_root("bogus")

    def test_invalid_regression_raises(self) -> None:
        with pytest.raises(ValueError, match="regression"):
            adf_test(np.zeros(10), regression="invalid")  # type: ignore[arg-type]

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="ADF requires"):
            adf_test(np.zeros(3), n_lags=2)


class TestEngleGranger:
    def test_recovers_known_beta(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        p_a, p_b, truth = synthetic_cointegrated_pair
        fit = engle_granger_test(p_a, p_b, n_lags=1)
        # OLS on a cointegrated pair is super-consistent — tight tolerance.
        assert math.isclose(fit.beta, truth["beta"], abs_tol=0.05)
        assert math.isclose(fit.alpha, truth["alpha"], abs_tol=0.5)

    def test_residual_adf_rejects_for_cointegrated(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        p_a, p_b, _ = synthetic_cointegrated_pair
        fit = engle_granger_test(p_a, p_b, n_lags=1)
        assert fit.adf_residual.rejects_unit_root("5%")

    def test_residual_adf_does_not_reject_for_independent(
        self, synthetic_independent_walks: tuple[pd.Series, pd.Series]
    ) -> None:
        a, b = synthetic_independent_walks
        fit = engle_granger_test(a, b, n_lags=1)
        # Two independent random walks should *not* reject at 1%.
        assert not fit.adf_residual.rejects_unit_root("1%")

    def test_misaligned_inputs_raise(self) -> None:
        a = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
        b = pd.Series([1.0, 2.0], index=pd.RangeIndex(1, 3))
        with pytest.raises(ValueError, match="aligned"):
            engle_granger_test(a, b)


class TestSelectOrientation:
    def test_selects_one_orientation(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        p_a, p_b, _ = synthetic_cointegrated_pair
        orientation = select_dependent_orientation(p_a, p_b)
        assert orientation in ("AonB", "BonA")


class TestAR1AndOU:
    def test_ar1_recovers_phi(
        self, synthetic_ar1_path: tuple[np.ndarray, dict[str, float]]
    ) -> None:
        z, truth = synthetic_ar1_path
        phi, c, sigma = fit_ar1(z)
        assert math.isclose(phi, truth["phi"], abs_tol=0.02)
        assert math.isclose(c, truth["c"], abs_tol=0.02)
        assert math.isclose(sigma, truth["sigma_eps"], abs_tol=0.01)

    def test_half_life_matches_log2_over_kappa(
        self, synthetic_ar1_path: tuple[np.ndarray, dict[str, float]]
    ) -> None:
        z, truth = synthetic_ar1_path
        ou = fit_ou(z)
        # Known closed form for AR(1) phi=0.9 -> half-life = log2 / -log(0.9).
        expected = -math.log(2.0) / math.log(truth["phi"])
        assert math.isclose(ou.half_life, expected, rel_tol=0.05)
        assert ou.kappa > 0
        assert math.isclose(ou.mu, truth["mu"], abs_tol=0.1)

    def test_half_life_from_phi_corner_cases(self) -> None:
        assert math.isinf(half_life_from_phi(1.0))
        assert math.isinf(half_life_from_phi(-0.5))
        # Known value: phi=0.5 -> half_life = -log(2)/log(0.5) = 1.
        assert math.isclose(half_life_from_phi(0.5), 1.0, abs_tol=1e-12)

    def test_unit_root_phi_yields_inf_half_life(
        self, rng: np.random.Generator
    ) -> None:
        random_walk = np.cumsum(rng.normal(0.0, 1.0, size=2000))
        ou = fit_ou(random_walk)
        assert ou.half_life > 50 or math.isinf(ou.half_life)


class TestZScore:
    def test_static_zscore_basic(self) -> None:
        z = np.array([0.0, 1.0, 2.0])
        out = static_zscore(z, mu=1.0, sigma=1.0)
        assert np.allclose(out, [-1.0, 0.0, 1.0])

    def test_static_zscore_zero_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            static_zscore(np.zeros(3), mu=0.0, sigma=0.0)

    def test_rolling_moments_shape(self) -> None:
        z = pd.Series(np.arange(100, dtype=float))
        mu, sigma = rolling_zscore_moments(z, window=10)
        assert mu.shape == (100,)
        assert sigma.shape == (100,)
        assert mu.iloc[:9].isna().all()
        assert not mu.iloc[10:].isna().any()


class TestKalman:
    def test_tracks_drifting_beta(
        self,
        synthetic_kalman_drift: tuple[pd.Series, pd.Series, np.ndarray],
    ) -> None:
        a, b, true_beta = synthetic_kalman_drift
        fit = kalman_dynamic_beta(
            a.to_numpy(), b.to_numpy(),
            Q=1e-5, R=1e-4, beta0=1.0, P0=1.0,
        )
        # After warm-up, filtered beta should be close to the (drifting) truth.
        warm = 200
        err = np.abs(fit.beta_path[warm:] - true_beta[warm:]).mean()
        assert err < 0.05

    def test_static_beta_when_Q_is_zero(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, _ = synthetic_cointegrated_pair
        fit = kalman_dynamic_beta(
            a.to_numpy(), b.to_numpy(),
            Q=0.0, R=1.0, beta0=0.0, P0=10.0,
        )
        # No process noise + huge initial variance -> filter recovers OLS beta
        # asymptotically.
        ols_beta = float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))
        assert math.isclose(fit.beta_path[-1], ols_beta, abs_tol=0.1)

    def test_R_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="R > 0"):
            kalman_dynamic_beta(
                np.zeros(5), np.zeros(5), Q=0.0, R=0.0, beta0=0.0, P0=1.0,
            )

    def test_shape_consistency(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            kalman_dynamic_beta(
                np.zeros(5), np.zeros(4), Q=0.0, R=1.0, beta0=0.0, P0=1.0,
            )


class TestTradingRules:
    def test_no_signal_when_below_entry(self) -> None:
        s = np.array([0.5, 1.0, 1.5, 0.0])
        rule = TradingRule(s_in=2.0, s_out=0.5, s_stop=4.0)
        pos = pair_position_path(s, rule)
        assert np.all(pos == 0)

    def test_short_then_exit(self) -> None:
        rule = TradingRule(s_in=2.0, s_out=0.5, s_stop=4.0)
        # Above entry -> short (-1). Then drop below s_out -> exit (0).
        s = np.array([0.0, 2.5, 2.5, 0.3, 0.3])
        pos = pair_position_path(s, rule)
        assert pos.tolist() == [0, -1, -1, 0, 0]

    def test_long_then_stop(self) -> None:
        rule = TradingRule(s_in=2.0, s_out=0.5, s_stop=4.0)
        s = np.array([0.0, -2.5, -2.5, -4.5, -2.0])
        pos = pair_position_path(s, rule)
        # Open long (+1), then stop (0). After the stop the rule is flat
        # because |-2.0| < s_in is False (|-2.0|=2.0 is NOT > 2.0) -> stay 0.
        assert pos[1] == 1
        assert pos[3] == 0
        assert pos[4] == 0

    def test_nan_carries_position(self) -> None:
        rule = TradingRule(s_in=2.0, s_out=0.5, s_stop=4.0)
        s = np.array([2.5, np.nan, np.nan, 0.0])
        pos = pair_position_path(s, rule)
        assert pos[0] == -1
        assert pos[1] == -1  # NaN carries previous position
        assert pos[2] == -1
        assert pos[3] == 0  # exit because |0.0| < s_out

    def test_invalid_rule_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="TradingRule"):
            TradingRule(s_in=1.0, s_out=1.5, s_stop=2.0)


class TestDiagnostics:
    def test_ljung_box_white_noise_high_p(
        self, rng: np.random.Generator
    ) -> None:
        x = rng.normal(0.0, 1.0, size=500)
        p = ljung_box_pvalue(x, max_lag=10)
        assert p > 0.05

    def test_ljung_box_autocorrelated_low_p(
        self, rng: np.random.Generator
    ) -> None:
        # Strongly autocorrelated AR(1) should be flagged.
        x = np.empty(500)
        x[0] = 0.0
        eps = rng.normal(0.0, 1.0, size=500)
        for t in range(1, 500):
            x[t] = 0.95 * x[t - 1] + eps[t]
        p = ljung_box_pvalue(x, max_lag=10)
        assert p < 0.01

    def test_rolling_eg_tstat_runs(
        self,
        synthetic_cointegrated_pair: tuple[pd.Series, pd.Series, dict[str, float]],
    ) -> None:
        a, b, _ = synthetic_cointegrated_pair
        ts = rolling_eg_residual_tstat(a, b, window=252, step=20)
        assert not ts.empty
        # For a genuinely cointegrated pair every window should be quite negative.
        assert ts.median() < -2.0

    def test_realized_half_life_matches_phi(
        self, synthetic_ar1_path: tuple[np.ndarray, dict[str, float]]
    ) -> None:
        z, truth = synthetic_ar1_path
        rhl = realized_half_life(pd.Series(z))
        ar1_hl = -math.log(2.0) / math.log(truth["phi"])
        # Empirical autocorrelation-crossing should bracket the AR(1) half-life
        # within a factor of 2 (spec validation step 6).
        assert ar1_hl / 2.0 <= rhl <= ar1_hl * 2.0
