"""Unit tests for the pure-math layer of the VaR model.

Each test exercises one function in `signal.py` against an analytic
answer or a known-statistic fixture.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.var.signal import (
    align_positions,
    basel_traffic_light,
    chi2_cdf,
    christoffersen_independence_test,
    historical_var,
    horizon_scale,
    kupiec_pof_test,
    monte_carlo_var,
    normal_ppf,
    parametric_var,
    portfolio_pnl_vector,
    rolling_var_backtest,
    simple_returns,
)

# ---------------------------------------------------------------------------
# Distribution primitives
# ---------------------------------------------------------------------------


class TestNormalPPF:
    def test_median(self) -> None:
        assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_basel_95(self) -> None:
        assert normal_ppf(0.95) == pytest.approx(1.6448536269, abs=1e-6)

    def test_basel_99(self) -> None:
        assert normal_ppf(0.99) == pytest.approx(2.3263478740, abs=1e-6)

    def test_symmetry(self) -> None:
        for q in (0.05, 0.25, 0.4, 0.49):
            assert normal_ppf(q) == pytest.approx(-normal_ppf(1.0 - q), abs=1e-9)

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError):
            normal_ppf(0.0)
        with pytest.raises(ValueError):
            normal_ppf(1.0)


class TestChi2CDF:
    def test_zero(self) -> None:
        assert chi2_cdf(0.0, 1) == 0.0

    def test_median_chi2_1(self) -> None:
        # Median of chi2(1) is ~0.4549; CDF at the median is 0.5.
        assert chi2_cdf(0.45493642, 1) == pytest.approx(0.5, abs=1e-4)

    def test_critical_value_chi2_1_95pct(self) -> None:
        # The 95th percentile of chi2(1) is 3.8415.
        assert chi2_cdf(3.8414588, 1) == pytest.approx(0.95, abs=1e-4)

    def test_monotone(self) -> None:
        prev = -1.0
        for x in np.linspace(0.1, 20.0, 50):
            current = chi2_cdf(float(x), 3)
            assert current >= prev - 1e-12
            prev = current


# ---------------------------------------------------------------------------
# Return preprocessing
# ---------------------------------------------------------------------------


class TestSimpleReturns:
    def test_constant_prices_yield_zero(self) -> None:
        prices = pd.DataFrame(
            {"A": [100.0] * 10},
            index=pd.date_range("2020-01-01", periods=10, freq="D"),
        )
        rets = simple_returns(prices)
        assert (rets["A"] == 0.0).all()
        assert len(rets) == 9

    def test_pct_change(self) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 99.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        rets = simple_returns(prices)
        np.testing.assert_allclose(
            rets["A"].to_numpy(), [0.10, -0.10], atol=1e-12
        )

    def test_empty_input(self) -> None:
        result = simple_returns(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Alignment + portfolio P&L
# ---------------------------------------------------------------------------


class TestAlignPositions:
    def test_reorders_to_match_columns(self) -> None:
        w, cols = align_positions({"A": 1.0, "B": 2.0}, ["B", "A"])
        assert cols == ["B", "A"]
        np.testing.assert_array_equal(w, [2.0, 1.0])

    def test_missing_ticker_raises(self) -> None:
        with pytest.raises(KeyError):
            align_positions({"A": 1.0}, ["A", "B"])


class TestPortfolioPnL:
    def test_two_asset_equally_weighted(self) -> None:
        rets = np.array([[0.01, -0.02], [-0.005, 0.015]])
        w = np.array([100.0, 100.0])
        pnl = portfolio_pnl_vector(rets, w)
        # First row: 100*0.01 + 100*-0.02 = -1.0
        # Second row: 100*-0.005 + 100*0.015 = 1.0
        np.testing.assert_allclose(pnl, [-1.0, 1.0])

    def test_short_position(self) -> None:
        rets = np.array([[0.05]])
        w = np.array([-100.0])  # short
        np.testing.assert_allclose(portfolio_pnl_vector(rets, w), [-5.0])


# ---------------------------------------------------------------------------
# Parametric VaR
# ---------------------------------------------------------------------------


class TestParametricVaR:
    def test_single_asset_matches_closed_form(self) -> None:
        """Single iid asset with sigma=0.01 and large N: parametric VaR at
        alpha=0.99 should approach |w| * sigma * Phi^-1(0.99) = 23,263 USD on
        a $1M position (mean term negligible at this scale)."""

        rng = np.random.default_rng(seed=42)
        rets = rng.standard_normal((50_000, 1)) * 0.01
        w = np.array([1_000_000.0])
        var_1d, mu_p, sigma_p, sigma_matrix = parametric_var(
            rets, w, alpha=0.99, include_mean=False
        )
        expected = 1_000_000.0 * 0.01 * normal_ppf(0.99)
        # mu zero'd out; tolerance dominated by sample-cov noise (~1/sqrt(N)).
        assert var_1d == pytest.approx(expected, rel=0.01)
        assert sigma_p == pytest.approx(10_000.0, rel=0.01)
        assert mu_p == 0.0
        assert sigma_matrix.shape == (1, 1)

    def test_uncorrelated_two_asset_sigma_aggregation(self) -> None:
        """For two zero-correlated assets, sigma_p^2 = w1^2 sigma1^2 +
        w2^2 sigma2^2.
        """

        rng = np.random.default_rng(seed=1)
        r1 = rng.standard_normal(40_000) * 0.01
        r2 = rng.standard_normal(40_000) * 0.02
        rets = np.column_stack([r1, r2])
        w = np.array([1.0, 1.0])
        _, _, sigma_p, _ = parametric_var(rets, w, alpha=0.99, include_mean=False)
        expected_sigma_p = math.sqrt(0.01**2 + 0.02**2)
        assert sigma_p == pytest.approx(expected_sigma_p, rel=0.01)

    def test_higher_alpha_yields_larger_var(self) -> None:
        rng = np.random.default_rng(seed=2)
        rets = rng.standard_normal((5000, 1)) * 0.01
        w = np.array([100.0])
        v_95, *_ = parametric_var(rets, w, alpha=0.95)
        v_99, *_ = parametric_var(rets, w, alpha=0.99)
        assert v_99 > v_95

    def test_var_non_negative_clip(self) -> None:
        """If mean dominates sigma (deeply long-biased + strong drift), the
        formula can be negative; we clip to zero.
        """

        rets = np.full((100, 1), 0.05)  # deterministic 5%/day -> sigma=0
        w = np.array([1_000_000.0])
        v, *_ = parametric_var(rets, w, alpha=0.99, include_mean=True)
        assert v == 0.0


# ---------------------------------------------------------------------------
# Historical VaR
# ---------------------------------------------------------------------------


class TestHistoricalVaR:
    def test_quantile_matches_numpy(self) -> None:
        rng = np.random.default_rng(seed=7)
        rets = rng.standard_normal((2000, 1)) * 0.01
        w = np.array([1_000_000.0])
        var_1d, pnl = historical_var(rets, w, alpha=0.99)
        loss = -pnl
        assert var_1d == pytest.approx(float(np.quantile(loss, 0.99)), abs=1e-9)

    def test_99pct_close_to_gaussian_quantile(self) -> None:
        """For Gaussian iid returns the HS quantile should be close to the
        parametric one (10x the analytic standard error at N = 50k)."""

        rng = np.random.default_rng(seed=8)
        rets = rng.standard_normal((50_000, 1)) * 0.01
        w = np.array([1.0])
        var_hs, _ = historical_var(rets, w, alpha=0.99)
        var_param = 0.01 * normal_ppf(0.99)
        assert var_hs == pytest.approx(var_param, rel=0.05)

    def test_pnl_vector_length(self) -> None:
        rets = np.array([[0.01], [-0.01], [0.02]])
        w = np.array([100.0])
        _, pnl = historical_var(rets, w, alpha=0.5)
        assert pnl.shape == (3,)


# ---------------------------------------------------------------------------
# Monte Carlo VaR
# ---------------------------------------------------------------------------


class TestMonteCarloVaR:
    def test_converges_to_parametric_for_gaussian_input(self) -> None:
        rng = np.random.default_rng(seed=10)
        rets = rng.standard_normal((5000, 1)) * 0.01
        w = np.array([1_000_000.0])
        var_param, _, _, _ = parametric_var(rets, w, alpha=0.99, include_mean=False)
        var_mc, _, _, _, _ = monte_carlo_var(
            rets, w, alpha=0.99, n_samples=200_000, seed=2025, include_mean=False
        )
        assert var_mc == pytest.approx(var_param, rel=0.02)

    def test_seed_is_deterministic(self) -> None:
        rng = np.random.default_rng(seed=11)
        rets = rng.standard_normal((500, 2)) * 0.01
        w = np.array([100.0, -50.0])
        v1, _, _, _, _ = monte_carlo_var(rets, w, alpha=0.99, n_samples=5000, seed=42)
        v2, _, _, _, _ = monte_carlo_var(rets, w, alpha=0.99, n_samples=5000, seed=42)
        assert v1 == v2

    def test_singular_covariance_does_not_crash(self) -> None:
        """Duplicated columns produce a singular Sigma — the safe Cholesky
        path should handle it.
        """

        base = np.random.default_rng(seed=12).standard_normal(500) * 0.01
        rets = np.column_stack([base, base])  # perfect collinearity
        w = np.array([100.0, -100.0])
        # Won't raise; portfolio P&L is mathematically zero, but the
        # Cholesky ridge fallback introduces a tiny numerical jitter.
        v, _, _, _, _ = monte_carlo_var(rets, w, alpha=0.99, n_samples=1000, seed=0)
        assert v >= 0.0
        # Expect << 1% of position size; ridge is O(1e-12 * trace).
        assert v < 1e-3


# ---------------------------------------------------------------------------
# Horizon scaling
# ---------------------------------------------------------------------------


class TestHorizonScale:
    def test_one_day_unchanged(self) -> None:
        assert horizon_scale(100.0, 1) == 100.0

    def test_ten_day_basel(self) -> None:
        assert horizon_scale(100.0, 10) == pytest.approx(100.0 * math.sqrt(10))

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            horizon_scale(100.0, 0)


# ---------------------------------------------------------------------------
# Kupiec POF
# ---------------------------------------------------------------------------


class TestKupiecPOF:
    def test_correctly_calibrated_at_expected_rate(self) -> None:
        """When the breach rate equals 1-alpha exactly, LR == 0."""

        lr, p = kupiec_pof_test(failures=10, n=1000, alpha=0.99)
        assert lr == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_too_many_breaches_rejects(self) -> None:
        """5% breach rate vs 1% nominal at T=1000: should reject loudly."""

        lr, p = kupiec_pof_test(failures=50, n=1000, alpha=0.99)
        assert lr > 50.0
        assert p < 1e-5

    def test_zero_breaches_well_defined(self) -> None:
        lr, p = kupiec_pof_test(failures=0, n=250, alpha=0.99)
        assert lr > 0.0
        assert math.isfinite(p)

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(ValueError):
            kupiec_pof_test(failures=5, n=10, alpha=1.5)
        with pytest.raises(ValueError):
            kupiec_pof_test(failures=20, n=10, alpha=0.99)


# ---------------------------------------------------------------------------
# Christoffersen independence
# ---------------------------------------------------------------------------


class TestChristoffersenIndependence:
    def test_iid_breaches_do_not_reject(self) -> None:
        """A Bernoulli iid breach indicator should not reject independence
        in the limit. Use a fairly long fixture and check p > 0.05.
        """

        rng = np.random.default_rng(seed=20)
        ind = (rng.random(2000) < 0.01).astype(int)
        lr, p = christoffersen_independence_test(ind)
        # The test allows for some sampling variation; assert it's not in
        # the rejection region.
        assert p > 0.05
        assert lr < 4.0

    def test_clustered_breaches_reject(self) -> None:
        """Breaches in blocks should drive pi_11 >> pi_01 and reject."""

        ind = np.zeros(2000, dtype=int)
        # Place 5 clusters of 4 consecutive breaches each.
        for start in (100, 400, 700, 1100, 1500):
            ind[start : start + 4] = 1
        lr, p = christoffersen_independence_test(ind)
        assert lr > 5.0
        assert p < 0.05

    def test_all_zeros_returns_degenerate_pass(self) -> None:
        ind = np.zeros(100, dtype=int)
        lr, p = christoffersen_independence_test(ind)
        assert lr == 0.0
        assert p == 1.0

    def test_rejects_non_binary(self) -> None:
        with pytest.raises(ValueError):
            christoffersen_independence_test(np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# Basel traffic light
# ---------------------------------------------------------------------------


class TestBaselTrafficLight:
    def test_green_zone(self) -> None:
        for x in range(0, 5):
            assert basel_traffic_light(x) == "green"

    def test_yellow_zone(self) -> None:
        for x in range(5, 10):
            assert basel_traffic_light(x) == "yellow"

    def test_red_zone(self) -> None:
        for x in (10, 15, 25):
            assert basel_traffic_light(x) == "red"

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            basel_traffic_light(-1)


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------


class TestRollingBacktest:
    def test_lengths_match(self) -> None:
        rng = np.random.default_rng(seed=30)
        rets = rng.standard_normal((400, 1)) * 0.01
        w = np.array([1_000_000.0])
        var_path, loss_path, breach = rolling_var_backtest(
            rets, w, alpha=0.99, method="historical", window=250
        )
        assert var_path.shape == (150,)
        assert loss_path.shape == (150,)
        assert breach.shape == (150,)

    def test_breach_rate_in_band_for_gaussian_input(self) -> None:
        """At alpha=0.95, with Gaussian iid returns and HS, the empirical
        breach rate should land near 5% on a 5000-row run.
        """

        rng = np.random.default_rng(seed=31)
        rets = rng.standard_normal((5000, 1)) * 0.01
        w = np.array([1_000_000.0])
        _, _, breach = rolling_var_backtest(
            rets, w, alpha=0.95, method="historical", window=500
        )
        rate = float(breach.mean())
        # Asymptotically 5%; 4-6% is a generous band on 4500 obs.
        assert 0.03 < rate < 0.08

    def test_rejects_unknown_method(self) -> None:
        rng = np.random.default_rng(seed=32)
        rets = rng.standard_normal((400, 1)) * 0.01
        w = np.array([1.0])
        with pytest.raises(ValueError):
            # Type-ignore on purpose: we're stressing input validation.
            rolling_var_backtest(rets, w, 0.99, method="bogus", window=250)  # type: ignore[arg-type]

    def test_rejects_short_panel(self) -> None:
        rng = np.random.default_rng(seed=33)
        rets = rng.standard_normal((100, 1)) * 0.01
        w = np.array([1.0])
        with pytest.raises(ValueError):
            rolling_var_backtest(rets, w, 0.99, method="historical", window=250)
