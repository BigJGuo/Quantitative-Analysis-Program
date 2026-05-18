"""Unit tests for the pure-math layer of Kyle-lambda / sqrt impact.

Every test pins a closed-form formula spelled out in the spec
(``models/layer5_execution/15_kyle_lambda_sqrt_impact.md``) against a
synthetic input with known ground truth.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.kyle_lambda_sqrt_impact.signal import (
    adv,
    compute_log_returns,
    compute_signed_volume,
    crossover_size,
    hc1_se_no_intercept,
    kyle_equilibrium_beta,
    kyle_equilibrium_lambda,
    linear_impact,
    ols_slope_no_intercept,
    participation_ratio_exceeds_threshold,
    r_squared_no_intercept,
    realized_daily_vol,
    sqrt_law_impact,
    wald_ci_95,
)


class TestKyleEquilibrium:
    def test_lambda_closed_form(self) -> None:
        # Spec: lambda = sigma_v / (2 * sigma_u).
        assert kyle_equilibrium_lambda(1.0, 1.0) == pytest.approx(0.5)
        assert kyle_equilibrium_lambda(2.0, 1.0) == pytest.approx(1.0)
        assert kyle_equilibrium_lambda(1.0, 4.0) == pytest.approx(0.125)

    def test_beta_closed_form(self) -> None:
        # Spec: beta = sigma_u / sigma_v.
        assert kyle_equilibrium_beta(1.0, 1.0) == pytest.approx(1.0)
        assert kyle_equilibrium_beta(2.0, 1.0) == pytest.approx(0.5)

    def test_lambda_times_2beta_is_one(self) -> None:
        # The spec's fixed point gives beta = 1 / (2 lambda).
        for sigma_v, sigma_u in [(1.0, 1.0), (2.5, 3.7), (0.1, 0.5)]:
            lam = kyle_equilibrium_lambda(sigma_v, sigma_u)
            beta = kyle_equilibrium_beta(sigma_v, sigma_u)
            assert 2.0 * lam * beta == pytest.approx(1.0)

    def test_invalid_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            kyle_equilibrium_lambda(0.0, 1.0)
        with pytest.raises(ValueError):
            kyle_equilibrium_lambda(1.0, -1.0)


class TestSignedVolume:
    def test_sign_matches_close_minus_open(self) -> None:
        bars = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0],
                "Close": [101.0, 99.0, 100.0],
                "Volume": [1000.0, 2000.0, 3000.0],
            }
        )
        sv = compute_signed_volume(bars)
        assert list(sv) == [1000.0, -2000.0, 0.0]

    def test_dollar_scaling(self) -> None:
        bars = pd.DataFrame(
            {
                "Open": [50.0],
                "Close": [55.0],
                "Volume": [1000.0],
            }
        )
        sv = compute_signed_volume(bars, dollar=True)
        # sign = +1, dollar volume = 55 * 1000 = 55000.
        assert sv.iloc[0] == pytest.approx(55_000.0)

    def test_missing_column_raises(self) -> None:
        bars = pd.DataFrame({"Open": [100.0], "Close": [101.0]})
        with pytest.raises(ValueError):
            compute_signed_volume(bars)


class TestLogReturns:
    def test_first_row_is_nan(self) -> None:
        bars = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        rets = compute_log_returns(bars)
        assert pd.isna(rets.iloc[0])
        assert rets.iloc[1] == pytest.approx(math.log(101.0 / 100.0))
        assert rets.iloc[2] == pytest.approx(math.log(102.0 / 101.0))


class TestOlsSlope:
    def test_noiseless_recovery(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.5 * x
        assert ols_slope_no_intercept(x, y) == pytest.approx(2.5)

    def test_zero_regressor_raises(self) -> None:
        x = np.zeros(5)
        y = np.ones(5)
        with pytest.raises(ValueError):
            ols_slope_no_intercept(x, y)


class TestHc1Se:
    def test_zero_residual_gives_zero_se(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = 3.0 * x
        beta = ols_slope_no_intercept(x, y)
        assert hc1_se_no_intercept(x, y, beta) == pytest.approx(0.0, abs=1e-12)

    def test_hc1_matches_closed_form(self) -> None:
        # Homoskedastic noise on a 3-point design:
        # x = [1, 2, 3], y = 2 x + [+0.5, -0.5, +0.5].
        # beta_hat = (1*2.5 + 2*3.5 + 3*6.5) / 14 = 29.0 / 14 = 2.0714286
        # residuals e = y - beta_hat * x.
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.5, 3.5, 6.5])
        beta = ols_slope_no_intercept(x, y)
        assert beta == pytest.approx(29.0 / 14.0)
        n = 3
        resid = y - beta * x
        meat = float(np.sum((x * resid) ** 2))
        xtx = 14.0
        expected = math.sqrt((n / (n - 1.0)) * meat / (xtx * xtx))
        assert hc1_se_no_intercept(x, y, beta) == pytest.approx(expected)

    def test_hc1_recovers_lambda_on_synthetic_panel(
        self,
        synthetic_daily_bars: pd.DataFrame,
        true_lambda: float,
        rng: np.random.Generator,
    ) -> None:
        # Reconstruct (signed_volume, log_return) directly and check lambda
        # is within 5 standard errors of the truth. The synthetic series is
        # noisy by construction.
        sv = compute_signed_volume(synthetic_daily_bars).to_numpy()
        ret = compute_log_returns(synthetic_daily_bars).to_numpy()
        # Drop the first NaN return.
        mask = ~np.isnan(ret)
        x = sv[mask]
        y = ret[mask]
        beta = ols_slope_no_intercept(x, y)
        se = hc1_se_no_intercept(x, y, beta)
        z = (beta - true_lambda) / se
        assert abs(z) < 5.0, f"|z|={abs(z)} too large; beta={beta} se={se}"


class TestRSquared:
    def test_perfect_fit_gives_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = 2.0 * x
        assert r_squared_no_intercept(x, y, 2.0) == pytest.approx(1.0)

    def test_zero_y_gives_nan(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.zeros(3)
        assert math.isnan(r_squared_no_intercept(x, y, 0.0))


class TestWaldCi:
    def test_symmetric_around_beta(self) -> None:
        lo, hi = wald_ci_95(2.0, 0.1)
        assert (lo + hi) / 2 == pytest.approx(2.0)
        assert hi - lo == pytest.approx(2 * 1.959963984540054 * 0.1)

    def test_zero_se_collapses(self) -> None:
        lo, hi = wald_ci_95(1.5, 0.0)
        assert lo == pytest.approx(1.5)
        assert hi == pytest.approx(1.5)


class TestSqrtLaw:
    def test_canonical_example(self) -> None:
        # sigma = 0.02, Q = 1000, V = 100_000, Y = 1 => sigma*sqrt(0.01) = 0.002.
        rel = sqrt_law_impact(0.02, 1_000.0, 100_000.0, Y=1.0)
        assert rel == pytest.approx(0.002)

    def test_doubles_at_quadruple_size(self) -> None:
        # Sqrt scaling: 4x size => 2x impact.
        s = 0.03
        v = 1_000_000.0
        rel_small = sqrt_law_impact(s, 1_000.0, v)
        rel_large = sqrt_law_impact(s, 4_000.0, v)
        assert rel_large / rel_small == pytest.approx(2.0)

    def test_alpha_override(self) -> None:
        # alpha = 1 collapses to the linear-in-Q-over-V law (no sqrt).
        rel = sqrt_law_impact(0.02, 1_000.0, 100_000.0, Y=1.0, alpha=1.0)
        assert rel == pytest.approx(0.02 * 0.01)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            sqrt_law_impact(-0.01, 100, 100)
        with pytest.raises(ValueError):
            sqrt_law_impact(0.01, 0, 100)
        with pytest.raises(ValueError):
            sqrt_law_impact(0.01, 100, 0)
        with pytest.raises(ValueError):
            sqrt_law_impact(0.01, 100, 100, Y=0)


class TestParticipationRatio:
    def test_threshold_at_10pct(self) -> None:
        assert not participation_ratio_exceeds_threshold(0.1, 1.0)
        assert participation_ratio_exceeds_threshold(0.11, 1.0)


class TestCrossover:
    def test_crossover_formula(self) -> None:
        # Q* = Y^2 sigma^2 / (lambda^2 V).
        # Y=1, sigma=0.02, V=1e6, lambda=1e-6 => Q* = 4e-4 / 1e-6 = 400.
        assert crossover_size(1e-6, 1.0, 0.02, 1.0e6) == pytest.approx(400.0)

    def test_lambda_zero_gives_infinity(self) -> None:
        assert crossover_size(0.0, 1.0, 0.02, 1.0e6) == math.inf

    def test_continuity_at_crossover(self) -> None:
        # Linear-Kyle and sqrt-law impacts must agree at Q = Q*.
        Y = 1.0
        sigma = 0.02
        V = 1.0e6
        lam = 1e-6
        q_star = crossover_size(lam, Y, sigma, V)
        linear = linear_impact(lam, q_star)
        sqrt_ = sqrt_law_impact(sigma, q_star, V, Y=Y)
        assert linear == pytest.approx(sqrt_, rel=1e-12)


class TestLinearImpact:
    def test_basic(self) -> None:
        assert linear_impact(1e-6, 1_000_000.0) == pytest.approx(1.0)

    def test_unsigned(self) -> None:
        # Returns absolute impact regardless of lambda sign.
        assert linear_impact(-1e-6, 1_000_000.0) == pytest.approx(1.0)


class TestRealizedVolAdv:
    def test_realized_daily_vol_matches_std(self) -> None:
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0, 0.02, size=500)
        closes = pd.Series(100.0 * np.cumprod(1.0 + rets))
        # The function uses pct_change, which differs slightly from
        # log-returns; check it lives in the right ballpark.
        sigma = realized_daily_vol(closes)
        assert 0.015 < sigma < 0.025

    def test_adv_window(self) -> None:
        vols = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # tail(3) = [3, 4, 5] => mean = 4.
        assert adv(vols, window=3) == pytest.approx(4.0)

    def test_adv_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError):
            adv(pd.Series([1.0]), window=0)
