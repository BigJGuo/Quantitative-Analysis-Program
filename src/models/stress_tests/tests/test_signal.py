"""Unit tests for `stress_tests/signal.py`.

Each math function is exercised against synthetic inputs with closed-form
answers so the worked-out spec formulas are checked exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.stress_tests.signal import (
    apply_proxy_fill,
    estimate_betas_ols,
    historical_replay_pnl,
    hypothetical_pnl,
    is_plausible,
    portfolio_factor_sensitivities,
    reverse_stress_shock,
    scenario_coverage_matrix,
    slice_window,
    top_contributors,
    window_endpoint_sensitivity,
    window_factor_change,
)
from src.models.stress_tests.types import (
    CrisisWindow,
    HypotheticalScenario,
)


class TestSliceWindow:
    def test_inclusive_endpoints(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        s = pd.Series(np.arange(10, dtype=float), index=index)
        sliced = slice_window(s, "2020-01-03", "2020-01-06")
        assert list(sliced.values) == [2.0, 3.0, 4.0, 5.0]

    def test_empty_for_out_of_range(self) -> None:
        s = pd.Series([1.0], index=[pd.Timestamp("2020-01-01")])
        assert slice_window(s, "2024-01-01", "2024-12-31").empty


class TestWindowFactorChange:
    def test_known_pct_change(self) -> None:
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        s = pd.Series([100.0, 110.0, 90.0], index=index)
        # First-to-last: 90/100 - 1 = -0.10
        assert window_factor_change(s) == pytest.approx(-0.10)

    def test_nan_for_singleton(self) -> None:
        s = pd.Series([100.0], index=[pd.Timestamp("2020-01-01")])
        assert math.isnan(window_factor_change(s))

    def test_nan_when_first_is_zero(self) -> None:
        s = pd.Series([0.0, 10.0])
        assert math.isnan(window_factor_change(s))


class TestApplyProxyFill:
    def test_direct_wins_over_proxy(self) -> None:
        filled, source = apply_proxy_fill(
            tickers=["AAA"],
            direct_changes={"AAA": -0.1},
            proxy_changes={"XLK": -0.2},
            fallback_change=-0.3,
            sector_map={"AAA": "XLK"},
        )
        assert filled["AAA"] == pytest.approx(-0.1)
        assert source["AAA"] == "direct"

    def test_falls_to_sector_proxy_when_direct_missing(self) -> None:
        filled, source = apply_proxy_fill(
            tickers=["AAA"],
            direct_changes={},
            proxy_changes={"XLK": -0.2},
            fallback_change=-0.3,
            sector_map={"AAA": "XLK"},
        )
        assert filled["AAA"] == pytest.approx(-0.2)
        assert source["AAA"].startswith("sector_proxy")

    def test_falls_to_benchmark_beta_when_proxy_missing(self) -> None:
        filled, source = apply_proxy_fill(
            tickers=["AAA"],
            direct_changes={},
            proxy_changes={},
            fallback_change=-0.10,
            sector_map={},
            betas_to_fallback={"AAA": 1.5},
        )
        assert filled["AAA"] == pytest.approx(-0.15)
        assert source["AAA"].startswith("benchmark_beta")

    def test_handles_missing_data(self) -> None:
        filled, source = apply_proxy_fill(
            tickers=["AAA"],
            direct_changes={},
            proxy_changes={},
            fallback_change=float("nan"),
            sector_map={},
        )
        assert filled["AAA"] == 0.0
        assert source["AAA"] == "missing"


class TestHistoricalReplayPnL:
    def test_dot_product_pnl(self) -> None:
        window = CrisisWindow(name="W", t1="2020-01-01", t2="2020-01-31")
        positions = {"AAA": 100_000.0, "BBB": -50_000.0}
        factor_changes = {"AAA": -0.10, "BBB": -0.20}
        result = historical_replay_pnl(
            window=window,
            positions=positions,
            factor_changes=factor_changes,
        )
        # 100k * -0.10 + -50k * -0.20 = -10k + 10k = 0
        assert result.total_pnl == pytest.approx(0.0)
        assert result.contributions["AAA"] == pytest.approx(-10_000.0)
        assert result.contributions["BBB"] == pytest.approx(10_000.0)
        assert result.scenario_type == "historical"

    def test_ignores_missing_changes(self) -> None:
        window = CrisisWindow(name="W", t1="2020-01-01", t2="2020-01-31")
        positions = {"AAA": 100_000.0, "BBB": -50_000.0}
        result = historical_replay_pnl(
            window=window,
            positions=positions,
            factor_changes={"AAA": -0.10},
        )
        # BBB has no factor change available, contributes 0.
        assert result.total_pnl == pytest.approx(-10_000.0)
        assert result.contributions["BBB"] == 0.0


class TestEstimateBetasOLS:
    def test_recovers_known_betas(self) -> None:
        rng = np.random.default_rng(0)
        n = 1500
        f = rng.normal(0.0, 0.01, size=n)
        true_betas = np.array([0.5, 1.0, 1.5])
        eps = rng.normal(0.0, 0.001, size=(n, 3))
        R = f[:, None] * true_betas + eps
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        asset = pd.DataFrame(R, index=index, columns=["A", "B", "C"])
        factor = pd.DataFrame({"M": f}, index=index)
        betas = estimate_betas_ols(asset_returns=asset, factor_returns=factor)
        np.testing.assert_allclose(betas["M"].to_numpy(), true_betas, atol=0.05)

    def test_raises_on_no_overlap(self) -> None:
        a = pd.DataFrame({"A": [0.1]}, index=[pd.Timestamp("2020-01-01")])
        f = pd.DataFrame({"M": [0.1]}, index=[pd.Timestamp("2024-01-01")])
        with pytest.raises(ValueError, match="overlapping"):
            estimate_betas_ols(asset_returns=a, factor_returns=f)


class TestHypotheticalPnL:
    def test_applies_equity_shock_through_beta_map(self) -> None:
        scenario = HypotheticalScenario(name="Eq-20", shocks={"equity": -0.20})
        positions = {"AAA": 100.0, "BBB": -50.0}
        betas = pd.DataFrame(
            {"equity": [1.0, 0.5]}, index=["AAA", "BBB"]
        )
        result = hypothetical_pnl(scenario=scenario, positions=positions, betas=betas)
        # AAA: 100 * 1.0 * -0.20 = -20; BBB: -50 * 0.5 * -0.20 = +5
        assert result.contributions["AAA"] == pytest.approx(-20.0)
        assert result.contributions["BBB"] == pytest.approx(5.0)
        assert result.total_pnl == pytest.approx(-15.0)

    def test_drops_unrecognized_factor(self) -> None:
        scenario = HypotheticalScenario(name="x", shocks={"unknown": 0.5})
        betas = pd.DataFrame({"equity": [1.0]}, index=["AAA"])
        result = hypothetical_pnl(
            scenario=scenario,
            positions={"AAA": 100.0},
            betas=betas,
        )
        assert result.total_pnl == 0.0
        assert result.metadata["unrecognized_factors"] == ["unknown"]


class TestReverseStress:
    def test_loss_constraint_holds(self) -> None:
        # Linear book with known sensitivities: equity shock 1 -> g^T dF = loss.
        g = np.array([100.0, 50.0])
        Sigma = np.array([[0.04, 0.0], [0.0, 0.01]])
        result = reverse_stress_shock(
            g=g, factor_covariance=Sigma, loss_target=10.0,
            factor_names=("equity", "rates"),
        )
        # P&L under the closed-form shock = g^T dF; should equal -loss_target.
        pnl = float(g @ result.dF_star)
        assert pnl == pytest.approx(-10.0, rel=1e-9)

    def test_mahalanobis_matches_closed_form(self) -> None:
        g = np.array([1.0, 0.0])
        Sigma = np.eye(2)
        result = reverse_stress_shock(
            g=g, factor_covariance=Sigma, loss_target=3.0,
            factor_names=("a", "b"),
        )
        # quad_form = 1, mahalanobis = L*/sqrt(quad_form) = 3.0.
        assert result.mahalanobis_distance == pytest.approx(3.0)

    def test_min_norm_property(self) -> None:
        rng = np.random.default_rng(1)
        # Construct a random PSD Σ.
        A = rng.normal(size=(4, 4))
        Sigma = A @ A.T + 0.1 * np.eye(4)
        Sigma_inv = np.linalg.inv(Sigma)
        g = rng.normal(size=4)
        L = 5.0
        result = reverse_stress_shock(
            g=g, factor_covariance=Sigma, loss_target=L,
            factor_names=("a", "b", "c", "d"),
        )
        # Pick an arbitrary feasible dF: same loss but different shape.
        dF_alt = np.zeros(4)
        # find any direction that satisfies g^T dF = -L by adjusting one entry.
        nonzero = np.argmax(np.abs(g))
        dF_alt[nonzero] = -L / g[nonzero]
        m_star = float(result.dF_star @ Sigma_inv @ result.dF_star)
        m_alt = float(dF_alt @ Sigma_inv @ dF_alt)
        assert m_star <= m_alt + 1e-9

    def test_rejects_zero_quad_form(self) -> None:
        with pytest.raises(ValueError, match="g\\^T"):
            reverse_stress_shock(
                g=np.zeros(2),
                factor_covariance=np.eye(2),
                loss_target=1.0,
                factor_names=("a", "b"),
            )


class TestPlausibility:
    def test_below_threshold_is_plausible(self) -> None:
        g = np.array([1.0])
        result = reverse_stress_shock(
            g=g, factor_covariance=np.eye(1), loss_target=2.0,
            factor_names=("a",),
        )
        assert is_plausible(result)  # Mahalanobis = 2 < 4

    def test_above_threshold_not_plausible(self) -> None:
        g = np.array([1.0])
        result = reverse_stress_shock(
            g=g, factor_covariance=np.eye(1), loss_target=10.0,
            factor_names=("a",),
        )
        assert not is_plausible(result)  # Mahalanobis = 10 > 4


class TestPortfolioFactorSensitivities:
    def test_aggregates_dollar_betas(self) -> None:
        betas = pd.DataFrame(
            {"equity": [1.0, 0.5], "rates": [0.2, -0.1]},
            index=["AAA", "BBB"],
        )
        g = portfolio_factor_sensitivities(
            positions={"AAA": 100.0, "BBB": -50.0}, betas=betas
        )
        # equity: 100*1.0 + -50*0.5 = 75; rates: 100*0.2 + -50*-0.1 = 25
        assert g[0] == pytest.approx(75.0)
        assert g[1] == pytest.approx(25.0)


class TestScenarioCoverage:
    def test_marks_present_factors(self) -> None:
        scenarios = [
            HypotheticalScenario(name="A", shocks={"equity": -0.2}),
            HypotheticalScenario(name="B", shocks={"rates": 0.01, "dxy": 0.05}),
        ]
        cov = scenario_coverage_matrix(
            scenarios=scenarios, factor_names=["equity", "rates", "dxy", "vol"]
        )
        assert cov.loc["A", "equity"]
        assert not cov.loc["A", "rates"]
        assert cov.loc["B", "rates"]
        assert cov.loc["B", "dxy"]
        # vol is uncovered across the set -> column-empty.
        assert not cov["vol"].any()


class TestTopContributors:
    def test_returns_largest_absolute(self) -> None:
        from src.models.stress_tests.types import ScenarioPnL
        scen = ScenarioPnL(
            scenario_name="x",
            scenario_type="historical",
            total_pnl=0.0,
            contributions={"AAA": -100.0, "BBB": 50.0, "CCC": 5.0, "DDD": -2.0},
            factor_changes={},
        )
        top = top_contributors(scen, k=2)
        assert [t[0] for t in top] == ["AAA", "BBB"]


class TestWindowEndpointSensitivity:
    def test_shifts_are_close(self) -> None:
        index = pd.date_range("2020-01-01", periods=30, freq="D")
        # Monotonic price ramp: pct change is approximately stable under small shifts.
        s = pd.Series(np.linspace(100.0, 110.0, 30), index=index)
        window = CrisisWindow(name="X", t1="2020-01-05", t2="2020-01-25")
        result = window_endpoint_sensitivity(
            window=window,
            prices_by_ticker={"AAA": s},
            positions={"AAA": 1_000_000.0},
            shift_days=2,
        )
        vals = list(result.values())
        # Range should be modest on a smooth path.
        assert max(vals) - min(vals) < 20_000
