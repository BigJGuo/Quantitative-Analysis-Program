"""Pure-math unit tests for `src.models.expected_shortfall.signal`.

Every test pins a closed-form or symmetry property: Gaussian ES against its
analytic formula, Student-t ES against numerical integration, historical ES
against a hand-built sample, and the coherence axioms (sub-additivity,
positive homogeneity, translation invariance, monotonicity) against
synthetic loss vectors.
"""

from __future__ import annotations

import math

import numpy as np

from src.models.expected_shortfall.signal import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    apply_liquidity_scaling,
    es_gaussian,
    es_historical,
    es_monte_carlo,
    es_student_t,
    fit_student_t,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    portfolio_pnl,
    rolling_realized_tail_mean,
    student_t_cdf_raw,
    student_t_pdf_raw,
    student_t_ppf_raw,
    traffic_light_band,
)

# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------


def test_normal_ppf_matches_known_quantiles() -> None:
    assert math.isclose(normal_ppf(0.975), 1.959963984540054, abs_tol=1e-8)
    assert math.isclose(normal_ppf(0.99), 2.3263478740408408, abs_tol=1e-8)
    assert math.isclose(normal_ppf(0.5), 0.0, abs_tol=1e-9)


def test_normal_pdf_at_zero() -> None:
    assert math.isclose(normal_pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi), abs_tol=1e-12)


def test_normal_cdf_at_zero() -> None:
    assert math.isclose(normal_cdf(0.0), 0.5, abs_tol=1e-12)


def test_student_t_pdf_integrates_to_one() -> None:
    # Trapezoidal integration on a wide grid.
    xs = np.linspace(-50.0, 50.0, 20_001)
    pdf = np.array([student_t_pdf_raw(float(x), nu=5.0) for x in xs])
    integral = np.trapezoid(pdf, xs)
    assert math.isclose(integral, 1.0, abs_tol=2e-4)


def test_student_t_cdf_matches_known_values() -> None:
    # Symmetric around zero
    assert math.isclose(student_t_cdf_raw(0.0, nu=5.0), 0.5, abs_tol=1e-9)
    # P(T_5 < 2.015) ≈ 0.95
    assert math.isclose(student_t_cdf_raw(2.015048, nu=5.0), 0.95, abs_tol=1e-5)


def test_student_t_ppf_inverts_cdf() -> None:
    for q in (0.05, 0.5, 0.95, 0.975, 0.99):
        x = student_t_ppf_raw(q, nu=5.0)
        back = student_t_cdf_raw(x, nu=5.0)
        assert math.isclose(back, q, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Gaussian ES — closed form
# ---------------------------------------------------------------------------


def test_es_gaussian_matches_boxed_formula() -> None:
    alpha = 0.975
    sigma = 100_000.0
    mu = 0.0
    var, es = es_gaussian(mu, sigma, alpha)
    z = normal_ppf(alpha)
    assert math.isclose(var, sigma * z, rel_tol=1e-10)
    expected_es = sigma * normal_pdf(z) / (1.0 - alpha)
    assert math.isclose(es, expected_es, rel_tol=1e-10)


def test_es_gaussian_at_99_pct_known_factor() -> None:
    # At alpha=0.99 Gaussian, the multiplier phi(z)/(1-alpha) ≈ 2.6652.
    _, es = es_gaussian(0.0, 1.0, 0.99)
    assert math.isclose(es, 2.665214340046402, rel_tol=1e-6)


def test_es_gaussian_translation_invariance() -> None:
    # ES(L + c) == ES(L) + c
    _, es_base = es_gaussian(0.0, 1.0, 0.975)
    _, es_shifted = es_gaussian(10.0, 1.0, 0.975)
    assert math.isclose(es_shifted - es_base, 10.0, rel_tol=1e-9)


def test_es_gaussian_positive_homogeneity() -> None:
    # ES(cL) == c ES(L) for c > 0
    _, es_unit = es_gaussian(0.0, 1.0, 0.975)
    _, es_3x = es_gaussian(0.0, 3.0, 0.975)
    assert math.isclose(es_3x, 3.0 * es_unit, rel_tol=1e-9)


def test_es_gaussian_dominates_var() -> None:
    var, es = es_gaussian(0.0, 1.0, 0.975)
    assert es > var


# ---------------------------------------------------------------------------
# Student-t ES — closed form
# ---------------------------------------------------------------------------


def test_es_student_t_dominates_gaussian_in_tail() -> None:
    # For fixed (loc=0, scale chosen so they have equal variance), Student-t
    # ES should exceed Gaussian ES at the same alpha.
    nu = 5.0
    gauss_sigma = math.sqrt(nu / (nu - 2.0))  # so Var(t-rv) == 1
    _, es_g = es_gaussian(0.0, 1.0, 0.99)
    _, es_t = es_student_t(0.0, 1.0, nu, 0.99)
    # Equal variance, fatter t-tails → t ES should be materially larger.
    # Scale gaussian sigma to t's: gauss_sigma * z_alpha matches t scale = 1
    _, es_g_eq = es_gaussian(0.0, gauss_sigma, 0.99)
    assert es_t > es_g_eq


def test_es_student_t_recovers_gaussian_as_nu_grows() -> None:
    # As nu → ∞, Student-t ES → Gaussian ES with the same scale.
    nu = 200.0
    _, es_t = es_student_t(0.0, 1.0, nu, 0.975)
    _, es_g = es_gaussian(0.0, 1.0, 0.975)
    assert math.isclose(es_t, es_g, rel_tol=2e-2)


def test_es_student_t_translation_and_positive_homogeneity() -> None:
    nu = 6.0
    _, base = es_student_t(0.0, 1.0, nu, 0.975)
    _, shifted = es_student_t(5.0, 1.0, nu, 0.975)
    _, scaled = es_student_t(0.0, 3.0, nu, 0.975)
    assert math.isclose(shifted - base, 5.0, rel_tol=1e-9)
    assert math.isclose(scaled, 3.0 * base, rel_tol=1e-9)


def test_es_student_t_matches_numerical_integration() -> None:
    """Spec formula ES = loc + scale * f_v(t_a)/(1-a) * (v + t_a^2)/(v-1)
    vs. direct numerical conditional-mean integration of the raw t.
    """
    nu = 7.0
    alpha = 0.975
    _, es_closed = es_student_t(0.0, 1.0, nu, alpha)
    t_a = student_t_ppf_raw(alpha, nu)
    xs = np.linspace(t_a, t_a + 50.0, 20_001)
    pdf = np.array([student_t_pdf_raw(float(x), nu) for x in xs])
    integrand = xs * pdf
    es_numerical = float(np.trapezoid(integrand, xs)) / (1.0 - alpha)
    assert math.isclose(es_closed, es_numerical, rel_tol=2e-3)


# ---------------------------------------------------------------------------
# Historical ES
# ---------------------------------------------------------------------------


def test_es_historical_hand_built_sample() -> None:
    # Worst 6 of 200 are the tail at alpha=0.97. Their mean is the ES.
    losses = np.concatenate([np.full(194, 1.0), np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])])
    var, es, tail = es_historical(losses, alpha=0.97)
    # 97th percentile of these 200 sorted losses is at index ~194, which is 5.0.
    assert var <= 5.0
    # Tail is the 6 largest (or 7 if cutoff falls on 5.0).
    expected_tail_mean = np.mean(losses[losses > var])
    assert math.isclose(es, float(expected_tail_mean), rel_tol=1e-12)
    assert tail.size >= 1


def test_es_historical_converges_to_gaussian() -> None:
    rng = np.random.default_rng(seed=12345)
    losses = rng.standard_normal(100_000)
    alpha = 0.975
    _, es_emp, _ = es_historical(losses, alpha)
    _, es_an = es_gaussian(0.0, 1.0, alpha)
    assert math.isclose(es_emp, es_an, rel_tol=0.03)


def test_es_historical_dominates_var() -> None:
    rng = np.random.default_rng(seed=98765)
    losses = rng.standard_normal(10_000)
    var, es, _ = es_historical(losses, alpha=0.95)
    assert es > var


# ---------------------------------------------------------------------------
# Monte Carlo ES
# ---------------------------------------------------------------------------


def test_es_monte_carlo_gaussian_limit() -> None:
    # Large nu in the MC sampler should match the Gaussian closed-form ES.
    mu = np.zeros(2)
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = np.array([1.0, 0.0])
    rng = np.random.default_rng(seed=42)
    _, es_mc, _ = es_monte_carlo(mu, Sigma, w, alpha=0.975, K=200_000, nu=200.0, rng=rng)
    _, es_an = es_gaussian(0.0, 1.0, 0.975)
    assert math.isclose(es_mc, es_an, rel_tol=0.03)


def test_es_monte_carlo_dominates_var() -> None:
    mu = np.zeros(2)
    Sigma = np.eye(2)
    w = np.array([1.0, 1.0])
    rng = np.random.default_rng(seed=7)
    var, es, _ = es_monte_carlo(mu, Sigma, w, alpha=0.99, K=50_000, nu=5.0, rng=rng)
    assert es > var


def test_es_monte_carlo_deterministic_under_seed() -> None:
    mu = np.zeros(2)
    Sigma = np.eye(2)
    w = np.array([1.0, 1.0])
    rng1 = np.random.default_rng(seed=99)
    rng2 = np.random.default_rng(seed=99)
    _, es1, _ = es_monte_carlo(mu, Sigma, w, alpha=0.975, K=20_000, nu=5.0, rng=rng1)
    _, es2, _ = es_monte_carlo(mu, Sigma, w, alpha=0.975, K=20_000, nu=5.0, rng=rng2)
    assert math.isclose(es1, es2, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Coherence axioms (Artzner et al. 1999)
# ---------------------------------------------------------------------------


def test_coherence_subadditivity_empirical() -> None:
    # ES(L1 + L2) <= ES(L1) + ES(L2) for any two loss vectors.
    rng = np.random.default_rng(seed=20260518)
    L1 = rng.standard_normal(20_000)
    L2 = rng.standard_normal(20_000)
    _, es1, _ = es_historical(L1, alpha=0.95)
    _, es2, _ = es_historical(L2, alpha=0.95)
    _, es_sum, _ = es_historical(L1 + L2, alpha=0.95)
    # Allow a small numerical slack for finite-sample empirical estimators.
    assert es_sum <= es1 + es2 + 1e-6 * (abs(es1) + abs(es2) + 1.0)


def test_coherence_monotonicity_empirical() -> None:
    # If L1 <= L2 pointwise, ES(L1) <= ES(L2).
    rng = np.random.default_rng(seed=33)
    L1 = rng.standard_normal(10_000)
    L2 = L1 + 0.5  # uniformly worse losses
    _, es1, _ = es_historical(L1, alpha=0.95)
    _, es2, _ = es_historical(L2, alpha=0.95)
    assert es2 >= es1


def test_coherence_translation_invariance_empirical() -> None:
    rng = np.random.default_rng(seed=44)
    L = rng.standard_normal(10_000)
    c = 2.5
    _, es_base, _ = es_historical(L, alpha=0.95)
    _, es_shift, _ = es_historical(L + c, alpha=0.95)
    assert math.isclose(es_shift - es_base, c, rel_tol=1e-9)


def test_coherence_positive_homogeneity_empirical() -> None:
    rng = np.random.default_rng(seed=55)
    L = rng.standard_normal(10_000)
    _, es_base, _ = es_historical(L, alpha=0.95)
    _, es_scaled, _ = es_historical(3.0 * L, alpha=0.95)
    assert math.isclose(es_scaled, 3.0 * es_base, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------


def test_portfolio_pnl_matches_manual_dot() -> None:
    import pandas as pd

    rets = pd.DataFrame({"A": [0.01, -0.02], "B": [0.005, -0.01]})
    w = np.array([10_000.0, 20_000.0])
    pnl = portfolio_pnl(rets, w)
    expected = np.array(
        [0.01 * 10_000.0 + 0.005 * 20_000.0, -0.02 * 10_000.0 - 0.01 * 20_000.0]
    )
    np.testing.assert_allclose(pnl, expected, rtol=1e-12)


def test_apply_liquidity_scaling_identity_when_none() -> None:
    import pandas as pd

    rets = pd.DataFrame({"A": [0.01], "B": [0.02]})
    out = apply_liquidity_scaling(rets, None)
    assert (out == rets).all().all()


def test_apply_liquidity_scaling_multiplies_columnwise() -> None:
    import pandas as pd

    rets = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]})
    scalars = np.array([2.0, 0.5])
    out = apply_liquidity_scaling(rets, scalars)
    np.testing.assert_allclose(out["A"].to_numpy(), [0.02, 0.04], rtol=1e-12)
    np.testing.assert_allclose(out["B"].to_numpy(), [0.015, 0.02], rtol=1e-12)


# ---------------------------------------------------------------------------
# Acerbi-Szekely tests
# ---------------------------------------------------------------------------


def test_acerbi_szekely_z1_centered_around_expected_mean() -> None:
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    var = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
    es = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    # Breaches at indices 2,3,4 (losses 3,4,5). Z1 = mean(L_t / ES_t) - 1.
    expected = (3.0 / 4.0 + 4.0 / 4.0 + 5.0 / 4.0) / 3.0 - 1.0  # = 0.0
    z1 = acerbi_szekely_z1(losses, var, es)
    assert math.isclose(z1, expected, abs_tol=1e-12)


def test_acerbi_szekely_z1_positive_when_es_underpredicted() -> None:
    losses = np.array([0.0, 0.0, 10.0, 10.0])
    var = np.array([1.0, 1.0, 1.0, 1.0])
    es_pred = np.array([2.0, 2.0, 2.0, 2.0])  # severely under-predicted
    z1 = acerbi_szekely_z1(losses, var, es_pred)
    assert z1 > 1.0


def test_acerbi_szekely_z2_zero_under_perfect_es_calibration() -> None:
    # When P(L > VaR) = 1-alpha exactly and L on breach days = ES, Z2 ≈ 0.
    T = 1000
    alpha = 0.99
    n_breach = int(T * (1.0 - alpha))
    losses = np.concatenate([np.zeros(T - n_breach), np.full(n_breach, 5.0)])
    var = np.full(T, 1.0)
    es_pred = np.full(T, 5.0)
    z2 = acerbi_szekely_z2(losses, var, es_pred, alpha)
    assert math.isclose(z2, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Diagnostics / utilities
# ---------------------------------------------------------------------------


def test_rolling_tail_mean_shape_and_nans() -> None:
    losses = np.arange(100.0)
    out = rolling_realized_tail_mean(losses, window=30, alpha=0.9)
    # First 29 entries should be NaN; later entries finite.
    assert np.all(np.isnan(out[:29]))
    assert np.all(np.isfinite(out[29:]))


def test_traffic_light_band_buckets() -> None:
    assert traffic_light_band(1.0) == "amber-light"
    assert traffic_light_band(1.2) == "green"
    assert traffic_light_band(1.5) == "amber"
    assert traffic_light_band(2.0) == "red"
    assert traffic_light_band(float("nan")) == "undefined"


# ---------------------------------------------------------------------------
# Student-t MLE
# ---------------------------------------------------------------------------


def test_fit_student_t_recovers_known_parameters() -> None:
    rng = np.random.default_rng(seed=1234)
    nu_true, loc_true, scale_true = 5.0, 0.0, 1.0
    # Sample from a Student-t via Gaussian / chi-squared mixture.
    g = rng.chisquare(nu_true, size=5000) / nu_true
    z = rng.standard_normal(size=5000)
    sample = loc_true + (z / np.sqrt(g)) * scale_true
    nu_hat, loc_hat, scale_hat = fit_student_t(sample)
    assert abs(nu_hat - nu_true) <= 2.5
    assert abs(loc_hat - loc_true) < 0.1
    assert abs(scale_hat - scale_true) < 0.15
