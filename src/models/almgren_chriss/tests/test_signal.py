"""Pure-math tests for the Almgren-Chriss signal layer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.almgren_chriss.signal import (
    annual_return_vol_to_dollar_per_day,
    compute_kappa,
    cost_variance_closed_form,
    discrete_kappa,
    efficient_frontier,
    expected_temporary_cost,
    intraday_volume_profile,
    inventory_path_continuous,
    inventory_path_discrete,
    multi_asset_schedule,
    optimal_schedule,
    permanent_cost,
    resample_profile,
    schedule_diagnostics,
    sinh_ratio,
    stress_schedule,
    vwap_shaped_schedule,
)
from src.models.almgren_chriss.types import (
    ImpactParams,
    LiquidationProblem,
    MultiAssetProblem,
)

# ---------------------------------------------------------------------------
# Scalar primitives
# ---------------------------------------------------------------------------


def test_sinh_ratio_matches_naive_for_small_args() -> None:
    a, b = 2.5, 3.0
    expected = math.sinh(a) / math.sinh(b)
    assert sinh_ratio(a, b) == pytest.approx(expected, rel=1e-12)


def test_sinh_ratio_stable_for_large_args() -> None:
    # sinh(40) overflows in double precision; the ratio must remain finite.
    out = sinh_ratio(38.0, 40.0)
    assert math.isfinite(out)
    # Asymptotic: exp(38 - 40) = e^{-2}.
    assert out == pytest.approx(math.exp(-2.0), rel=1e-6)


def test_sinh_ratio_zero_numerator() -> None:
    assert sinh_ratio(0.0, 1.0) == 0.0


def test_compute_kappa_formula() -> None:
    sigma_d, eta, lam = 2.0, 0.5, 1.0e-3
    k = compute_kappa(sigma_d, eta, lam)
    assert k == pytest.approx(math.sqrt(lam * sigma_d ** 2 / eta), rel=1e-12)


def test_compute_kappa_rejects_nonpositive_inputs() -> None:
    with pytest.raises(ValueError):
        compute_kappa(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        compute_kappa(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        compute_kappa(1.0, 1.0, 0.0)


def test_discrete_kappa_matches_target() -> None:
    kappa_cont = 1.5
    tau = 0.05
    k_tilde = discrete_kappa(kappa_cont, tau)
    # By construction, 2(cosh(k_tilde * tau) - 1) = (kappa_cont * tau)^2.
    lhs = 2.0 * (math.cosh(k_tilde * tau) - 1.0)
    rhs = (kappa_cont * tau) ** 2
    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_discrete_kappa_converges_to_continuous_as_tau_shrinks() -> None:
    kappa_cont = 1.0
    err_small_tau = abs(discrete_kappa(kappa_cont, 0.001) - kappa_cont)
    err_large_tau = abs(discrete_kappa(kappa_cont, 0.5) - kappa_cont)
    assert err_small_tau < err_large_tau
    assert err_small_tau < 1.0e-5


def test_annual_return_vol_to_dollar_per_day_formula() -> None:
    sigma_ann, price = 0.25, 100.0
    sigma_d = annual_return_vol_to_dollar_per_day(sigma_ann, price)
    assert sigma_d == pytest.approx(price * sigma_ann / math.sqrt(252), rel=1e-12)


def test_permanent_cost_formula() -> None:
    assert permanent_cost(1.0e5, 1.0e-8) == pytest.approx(
        0.5 * (1.0e5 ** 2) * 1.0e-8
    )


# ---------------------------------------------------------------------------
# Closed-form cost / variance formulas
# ---------------------------------------------------------------------------


def test_expected_temporary_cost_twap_limit() -> None:
    """As κ → 0 the temporary cost should reduce to η X² / T (TWAP)."""

    X, T, eta = 1000.0, 1.0, 1.0e-6
    e = expected_temporary_cost(X, T, kappa=1.0e-10, eta=eta)
    assert e == pytest.approx(eta * X * X / T, rel=1.0e-3)


def test_cost_variance_twap_limit() -> None:
    """As κ → 0 the variance should reduce to σ² X² T / 3 (TWAP)."""

    X, T, sigma_d = 1000.0, 1.0, 2.0
    v = cost_variance_closed_form(X, T, kappa=1.0e-10, sigma_d=sigma_d)
    assert v == pytest.approx(sigma_d ** 2 * X ** 2 * T / 3.0, rel=1.0e-3)


def test_closed_form_matches_numerical_integration() -> None:
    """Closed-form E[C_temp] and V[C] match Riemann integration of x_t, v_t.

    Uses a very fine grid (N = 5000); verifies the textbook formulas.
    """

    X, T, kappa = 1000.0, 1.0, 1.5
    eta, sigma_d = 1.0e-6, 2.0
    N = 5000
    tau = T / N
    x = inventory_path_continuous(X, T, N, kappa)
    n = x[:-1] - x[1:]
    v = n / tau
    # Numerical E[C_temp] = eta * integral v^2 dt ≈ eta * Σ v_k^2 * tau.
    e_num = float(eta * np.sum(v * v) * tau)
    # Numerical V[C] = sigma_d^2 * Σ x_k^2 * tau (k = 1..N; the slice-end inventories).
    v_num = float(sigma_d ** 2 * np.sum(x[1:] ** 2) * tau)
    e_closed = expected_temporary_cost(X, T, kappa, eta)
    v_closed = cost_variance_closed_form(X, T, kappa, sigma_d)
    assert e_closed == pytest.approx(e_num, rel=2.0e-3)
    assert v_closed == pytest.approx(v_num, rel=2.0e-3)


def test_expected_temporary_cost_increases_with_kappa() -> None:
    X, T, eta = 1000.0, 1.0, 1.0e-6
    costs = [expected_temporary_cost(X, T, k, eta) for k in (0.1, 0.5, 1.5, 5.0)]
    assert costs == sorted(costs)


def test_cost_variance_decreases_with_kappa() -> None:
    X, T, sigma_d = 1000.0, 1.0, 2.0
    vs = [cost_variance_closed_form(X, T, k, sigma_d) for k in (0.1, 0.5, 1.5, 5.0)]
    assert vs == sorted(vs, reverse=True)


# ---------------------------------------------------------------------------
# Inventory paths
# ---------------------------------------------------------------------------


def test_inventory_path_boundary_conditions() -> None:
    X, T, N, kappa = 1.0e5, 1.0, 50, 1.5
    x = inventory_path_continuous(X, T, N, kappa)
    assert x[0] == pytest.approx(X)
    assert x[-1] == pytest.approx(0.0)
    assert np.all(np.diff(x) <= 1.0e-9)  # monotone decreasing


def test_inventory_path_twap_at_zero_kappa() -> None:
    X, T, N = 1.0e5, 1.0, 10
    x = inventory_path_continuous(X, T, N, kappa=1.0e-12)
    expected = X * (1.0 - np.arange(N + 1, dtype=float) / N)
    np.testing.assert_allclose(x, expected, rtol=1.0e-9)


def test_inventory_path_front_loaded_at_high_kappa() -> None:
    """At large κT the path should match X exp(-κt) outside the boundary."""

    X, T, N, kappa = 1.0e5, 1.0, 50, 40.0
    x = inventory_path_continuous(X, T, N, kappa)
    # First half: exponential decay.
    t = np.arange(N + 1) * (T / N)
    asymp = X * np.exp(-kappa * t)
    assert np.allclose(x[:N // 2], asymp[:N // 2], rtol=1.0e-6)
    assert x[-1] == 0.0


def test_inventory_path_discrete_returns_kappa_tilde() -> None:
    X, T, N = 1.0e5, 1.0, 10
    kappa_cont = 2.0
    x, k_tilde = inventory_path_discrete(X, T, N, kappa_cont)
    assert x.shape == (N + 1,)
    assert x[0] == pytest.approx(X)
    assert x[-1] == pytest.approx(0.0)
    # κ̃ > 0 because κ > 0.
    assert k_tilde > 0.0
    # κ̃ close to κ for small τ.
    assert abs(k_tilde - kappa_cont) / kappa_cont < 0.05


# ---------------------------------------------------------------------------
# Top-level schedule
# ---------------------------------------------------------------------------


def test_optimal_schedule_balanced_params(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    sched = optimal_schedule(default_problem, default_params)
    assert sched.ticker == "TEST"
    assert sched.inventory.shape == (default_problem.N + 1,)
    assert sched.child_orders.shape == (default_problem.N,)
    assert sched.inventory[0] == pytest.approx(default_problem.X)
    assert sched.inventory[-1] == pytest.approx(0.0)
    np.testing.assert_allclose(
        sched.child_orders.sum(), default_problem.X, rtol=1.0e-10
    )
    # Reasonable kappa_T range
    assert 0.05 < sched.kappa_T < 30.0
    # Positive cost
    assert sched.expected_cost > 0.0
    assert sched.cost_variance > 0.0
    assert sched.expected_cost_bps > 0.0


def test_optimal_schedule_twap_limit(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    """With λ → 0, the schedule should converge to TWAP."""

    p = LiquidationProblem(
        ticker="TEST",
        X=default_problem.X,
        T=default_problem.T,
        N=default_problem.N,
        side="sell",
        lam=1.0e-15,
    )
    sched = optimal_schedule(p, default_params)
    expected = p.X * (1.0 - np.arange(p.N + 1, dtype=float) / p.N)
    np.testing.assert_allclose(sched.inventory, expected, rtol=1.0e-4)


def test_optimal_schedule_front_loaded_at_high_lambda(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    """With large λ, kappa_T grows and the schedule front-loads."""

    p_lo = LiquidationProblem(
        ticker="TEST", X=default_problem.X, T=default_problem.T,
        N=default_problem.N, side="sell", lam=1.0e-7,
    )
    p_hi = LiquidationProblem(
        ticker="TEST", X=default_problem.X, T=default_problem.T,
        N=default_problem.N, side="sell", lam=1.0e-3,
    )
    s_lo = optimal_schedule(p_lo, default_params)
    s_hi = optimal_schedule(p_hi, default_params)
    assert s_hi.kappa_T > s_lo.kappa_T
    # Half-life should be earlier under high λ.
    half = default_problem.X / 2.0
    t_half_lo = float(s_lo.times[np.argmax(s_lo.inventory <= half)])
    t_half_hi = float(s_hi.times[np.argmax(s_hi.inventory <= half)])
    assert t_half_hi < t_half_lo


def test_optimal_schedule_discrete_matches_continuous_for_large_N(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    """Spec: for N >= 50 the discrete and continuous schedules agree within 0.5%."""

    p = LiquidationProblem(
        ticker="TEST", X=default_problem.X, T=default_problem.T,
        N=100, side="sell", lam=default_problem.lam,
    )
    s_cont = optimal_schedule(p, default_params, discretization="continuous")
    s_disc = optimal_schedule(p, default_params, discretization="discrete")
    rel_diff = np.max(
        np.abs(s_cont.inventory - s_disc.inventory) / max(p.X, 1.0)
    )
    assert rel_diff < 5.0e-3


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------


def test_efficient_frontier_is_monotone_in_lambda(
    default_params: ImpactParams,
) -> None:
    """As λ grows, expected cost grows and variance shrinks."""

    lambdas = np.logspace(-9, -3, 7)
    front = efficient_frontier(1.0e5, 1.0, default_params, lambdas=lambdas)
    e = front.expected_costs()
    v = front.variances()
    # As λ grows we move from TWAP (high V, low E) to front-loaded (low V, high E).
    assert np.all(np.diff(e) >= -1.0e-6 * abs(e).max())  # E increasing in λ
    assert np.all(np.diff(v) <= 1.0e-6 * abs(v).max())  # V decreasing in λ
    # kappa_T should be increasing in λ.
    kt = front.kappa_T_values()
    assert np.all(np.diff(kt) > 0)


def test_efficient_frontier_convexity(default_params: ImpactParams) -> None:
    """E vs V on the frontier should be convex (E decreasing, second-diff >= 0)."""

    lambdas = np.logspace(-9, -3, 20)
    front = efficient_frontier(1.0e5, 1.0, default_params, lambdas=lambdas)
    v = front.variances()
    e = front.expected_costs()
    order = np.argsort(v)
    e_sorted = e[order]
    # E should be monotonically decreasing in V.
    assert np.all(np.diff(e_sorted) <= 1.0e-4 * abs(e_sorted).max())


# ---------------------------------------------------------------------------
# Stress / diagnostics
# ---------------------------------------------------------------------------


def test_stress_schedule_accelerates(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    base = optimal_schedule(default_problem, default_params)
    stressed = stress_schedule(default_problem, default_params, sigma_multiplier=3.0)
    assert stressed.kappa_T == pytest.approx(3.0 * base.kappa_T, rel=1.0e-9)


def test_schedule_diagnostics_balanced_params(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    sched = optimal_schedule(default_problem, default_params)
    diag = schedule_diagnostics(sched)
    assert diag["monotone_inventory"] is True
    assert diag["positive_children"] is True
    assert diag["sum_n_minus_X_relative"] < 1.0e-9
    assert diag["regime"] in {"near_twap", "twap_leaning", "balanced"}
    assert diag["numerical_stability_warning"] is False


# ---------------------------------------------------------------------------
# Volume profile / VWAP-shaped
# ---------------------------------------------------------------------------


def test_intraday_volume_profile_falls_back_to_flat_on_empty() -> None:
    import pandas as pd

    profile = intraday_volume_profile(pd.DataFrame(), bars_per_day=10)
    assert profile.shape == (10,)
    assert profile.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(profile, np.full(10, 0.1))


def test_intraday_volume_profile_recovers_known_share() -> None:
    """Synthetic intraday data where bar 0 has 2× the volume of others."""

    import pandas as pd

    bars_per_day = 4
    n_days = 5
    rows = []
    for d in range(n_days):
        for b in range(bars_per_day):
            rows.append(
                {
                    "Datetime": pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
                    + pd.Timedelta(minutes=5 * b),
                    "Volume": 200.0 if b == 0 else 100.0,
                }
            )
    df = pd.DataFrame(rows)
    profile = intraday_volume_profile(df, bars_per_day=bars_per_day)
    assert profile.sum() == pytest.approx(1.0)
    # Bar 0 share should be 200 / 500 = 0.4.
    assert profile[0] == pytest.approx(0.4, rel=1.0e-6)
    for b in (1, 2, 3):
        assert profile[b] == pytest.approx(0.2, rel=1.0e-6)


def test_resample_profile_identity() -> None:
    p = np.asarray([0.1, 0.2, 0.3, 0.4])
    out = resample_profile(p, 4)
    np.testing.assert_allclose(out, p / p.sum())


def test_resample_profile_preserves_normalization() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 1.0, size=10)
    p /= p.sum()
    for N in (3, 5, 10, 20):
        out = resample_profile(p, N)
        assert out.shape == (N,)
        assert out.sum() == pytest.approx(1.0)


def test_vwap_shaped_schedule_matches_ac_on_flat_profile(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    """With a flat profile the VWAP-shaped solution should reproduce the
    discrete Almgren-Chriss path (modulo the small Newton step on κ̃)."""

    N = default_problem.N
    flat = np.full(N, 1.0 / N)
    ac = optimal_schedule(default_problem, default_params, discretization="discrete")
    vwap = vwap_shaped_schedule(
        default_problem, default_params, volume_profile=flat
    )
    rel = np.max(np.abs(ac.inventory - vwap.inventory)) / default_problem.X
    assert rel < 1.0e-6


def test_vwap_shaped_schedule_is_monotone_on_realistic_profile(
    default_problem: LiquidationProblem, default_params: ImpactParams
) -> None:
    """A U-shaped intraday profile should still produce a monotone schedule."""

    N = default_problem.N
    pos = np.arange(N) / max(N - 1, 1)
    profile = 0.6 + 0.4 * (1.0 - 4.0 * pos * (1.0 - pos))
    profile /= profile.sum()
    sched = vwap_shaped_schedule(
        default_problem, default_params, volume_profile=profile
    )
    assert np.all(np.diff(sched.inventory) <= 1.0e-9)
    assert sched.inventory[-1] == pytest.approx(0.0)
    assert sched.inventory[0] == pytest.approx(default_problem.X)


# ---------------------------------------------------------------------------
# Multi-asset (Algorithm D)
# ---------------------------------------------------------------------------


def test_multi_asset_schedule_diagonal_reduces_to_per_asset(
    default_params: ImpactParams,
) -> None:
    """With a diagonal Σ and equal η, the basket schedule equals the
    single-asset schedule applied per leg."""

    tickers = ("A", "B")
    X = np.asarray([1.0e5, 5.0e4])
    sigma_diag = 0.25
    cov = np.diag([sigma_diag ** 2, sigma_diag ** 2])  # decimal, daily-return
    # But the daily-return cov should be sigma_annual^2 / 252 for our convention?
    # The multi-asset function expects DAILY return cov; build it from annual.
    cov_daily_returns = cov / 252.0
    eta_diag = np.asarray([default_params.eta, default_params.eta])
    gamma_diag = np.asarray([default_params.gamma, default_params.gamma])
    last_prices = np.asarray([default_params.last_price, default_params.last_price])
    daily_vols = np.asarray(
        [default_params.daily_volume, default_params.daily_volume]
    )
    problem = MultiAssetProblem(
        tickers=tickers,
        X=X,
        T=1.0,
        N=50,
        sides=("sell", "sell"),
        lam=1.0e-6,
    )
    basket = multi_asset_schedule(
        problem,
        sigma_cov_daily=cov_daily_returns,
        eta_diag=eta_diag,
        gamma_diag=gamma_diag,
        last_prices=last_prices,
        daily_volumes=daily_vols,
    )
    # Per-asset single-asset solutions:
    per_asset = []
    for ticker, x_size, price, vol_, gamma_ in zip(
        tickers, X, last_prices, daily_vols, gamma_diag, strict=True
    ):
        params = ImpactParams(
            sigma=sigma_diag,
            eta=default_params.eta,
            gamma=float(gamma_),
            last_price=float(price),
            daily_volume=float(vol_),
        )
        prob = LiquidationProblem(
            ticker=ticker,
            X=float(x_size),
            T=1.0,
            N=50,
            side="sell",
            lam=1.0e-6,
        )
        per_asset.append(optimal_schedule(prob, params))

    # Boundary checks
    np.testing.assert_allclose(basket.inventory[0, :], X, rtol=1e-10)
    np.testing.assert_allclose(basket.inventory[-1, :], 0.0, atol=1e-6)
    # Inventory paths should match per-asset within tight tolerance.
    for i, ps in enumerate(per_asset):
        rel = np.max(np.abs(basket.inventory[:, i] - ps.inventory)) / X[i]
        assert rel < 1.0e-6


def test_multi_asset_schedule_correlated_basket(
    default_params: ImpactParams,
) -> None:
    """Correlated covariance still yields a valid, monotone basket."""

    sigma_diag = np.asarray([0.25, 0.30])
    rho = 0.6
    corr = np.asarray([[1.0, rho], [rho, 1.0]])
    sigma_outer = np.outer(sigma_diag, sigma_diag)
    cov_annual = sigma_outer * corr
    cov_daily = cov_annual / 252.0
    eta_diag = np.asarray([default_params.eta, default_params.eta * 2.0])
    gamma_diag = np.asarray([default_params.gamma, default_params.gamma])
    last_prices = np.asarray([100.0, 50.0])
    daily_vols = np.asarray([5.0e7, 3.0e7])
    problem = MultiAssetProblem(
        tickers=("A", "B"),
        X=np.asarray([1.0e5, 8.0e4]),
        T=1.0,
        N=40,
        sides=("sell", "sell"),
        lam=1.0e-6,
    )
    basket = multi_asset_schedule(
        problem,
        sigma_cov_daily=cov_daily,
        eta_diag=eta_diag,
        gamma_diag=gamma_diag,
        last_prices=last_prices,
        daily_volumes=daily_vols,
    )
    assert basket.expected_cost > 0.0
    assert basket.cost_variance > 0.0
    assert basket.kappas.shape == (2,)
    assert np.all(basket.kappas >= 0.0)
    # Each asset's inventory monotone decreasing.
    for m in range(2):
        assert np.all(np.diff(basket.inventory[:, m]) <= 1.0e-9)
        assert basket.inventory[0, m] == pytest.approx(problem.X[m])
        assert basket.inventory[-1, m] == pytest.approx(0.0, abs=1.0e-6)
