"""Unit tests for the pure-math layer of the GARCH-family model.

Each test exercises one function in `signal.py` against a worked-out
analytic answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.garch_family.signal import (
    annualize_vol_pct,
    arch_lm_test,
    compute_sigma2_path,
    expected_abs_z_gaussian,
    expected_abs_z_student_t,
    forecast_variance,
    kupiec_pof_test,
    ljung_box,
    log_returns_pct,
    mincer_zarnowitz_regression,
    negloglik,
    normal_cdf,
    normal_ppf,
    qlike_loss,
    sign_bias_test,
    student_t_logpdf,
    student_t_ppf,
    value_at_risk,
)
from src.models.garch_family.types import GARCHFit, GARCHParams

# ---------------------------------------------------------------------------
# Distribution primitives
# ---------------------------------------------------------------------------


class TestNormal:
    def test_normal_ppf_known_quantiles(self) -> None:
        assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert normal_ppf(0.025) == pytest.approx(-1.959963985, abs=1e-6)
        assert normal_ppf(0.05) == pytest.approx(-1.644853627, abs=1e-6)
        assert normal_ppf(0.975) == pytest.approx(1.959963985, abs=1e-6)

    def test_normal_cdf_at_zero(self) -> None:
        assert normal_cdf(0.0) == pytest.approx(np.asarray(0.5))

    def test_normal_cdf_symmetry(self) -> None:
        x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
        cdf_x = normal_cdf(x)
        cdf_neg = normal_cdf(-x)
        np.testing.assert_allclose(cdf_x + cdf_neg, np.ones_like(x), atol=1e-10)


class TestStudentT:
    def test_expected_abs_z_gaussian(self) -> None:
        assert expected_abs_z_gaussian() == pytest.approx(math.sqrt(2.0 / math.pi))

    def test_expected_abs_z_student_t_limit(self) -> None:
        # As nu -> infinity, E|z| of standardized t -> sqrt(2/pi) (Gaussian).
        e_inf = expected_abs_z_student_t(1e6)
        assert e_inf == pytest.approx(math.sqrt(2.0 / math.pi), abs=1e-3)

    def test_student_t_logpdf_converges_to_gaussian(self) -> None:
        z = np.asarray([0.0, 1.0, 2.0])
        lp_large_nu = student_t_logpdf(z, 1e10)
        gauss = -0.5 * (math.log(2.0 * math.pi) + z * z)
        np.testing.assert_allclose(lp_large_nu, gauss, atol=1e-4)

    def test_student_t_ppf_symmetry(self) -> None:
        # By symmetry the median is zero.
        assert student_t_ppf(0.5, 5.0) == pytest.approx(0.0, abs=1e-6)

    def test_student_t_ppf_converges_to_normal_for_large_nu(self) -> None:
        assert student_t_ppf(0.05, 1000.0) == pytest.approx(-1.6449, abs=1e-2)

    def test_student_t_ppf_known_value(self) -> None:
        # Standardized t at nu=5, alpha=0.05:
        # raw-t 5% quantile ~ -2.015 (df=5); standardize by sqrt(5/3) -> ~ -1.561.
        assert student_t_ppf(0.05, 5.0) == pytest.approx(-1.561, abs=5e-3)


# ---------------------------------------------------------------------------
# Conditional-variance recursion
# ---------------------------------------------------------------------------


class TestComputeSigma2Path:
    def test_garch11_worked_example(self) -> None:
        """Hand-checked: sigma2[1] = omega + alpha * eps[0]^2 + beta * sigma2[0]."""

        params = GARCHParams(omega=0.1, alpha=0.1, beta=0.85)
        r = np.asarray([0.0, 2.0, -1.0, 0.5, -0.3])
        sigma2, eps = compute_sigma2_path(r, params, "GARCH")
        # Init from first 30 obs (or all when shorter): var of [0, 2, -1, 0.5, -0.3]
        np.testing.assert_allclose(eps, r, atol=1e-12)
        var_init = float(np.var(r, ddof=0))
        assert sigma2[0] == pytest.approx(var_init)
        # sigma2[1] = 0.1 + 0.1 * 0 + 0.85 * var_init
        expected_s2_1 = 0.1 + 0.1 * r[0] ** 2 + 0.85 * sigma2[0]
        assert sigma2[1] == pytest.approx(expected_s2_1)
        # sigma2[2] = 0.1 + 0.1 * 4 + 0.85 * sigma2[1]
        expected_s2_2 = 0.1 + 0.1 * r[1] ** 2 + 0.85 * sigma2[1]
        assert sigma2[2] == pytest.approx(expected_s2_2)

    def test_gjr_uses_indicator_on_negative_shock(self) -> None:
        """When eps[t-1] < 0, gamma contributes; when eps[t-1] >= 0 it does not.

        Construct two paths that share an identical 30-obs warm-up prefix so
        the recursion has the same `sigma2_{T-1}`; they then differ only in
        the sign of the shock at time T-1. The variance gap at T is
        `gamma * eps_{T-1}^2`.
        """

        params = GARCHParams(omega=0.1, alpha=0.05, beta=0.85, gamma=0.1)
        prefix = np.full(30, 0.1)
        r_pos = np.concatenate([prefix, np.asarray([2.0, 0.0])])
        r_neg = np.concatenate([prefix, np.asarray([-2.0, 0.0])])
        s2_pos, _ = compute_sigma2_path(r_pos, params, "GJR")
        s2_neg, _ = compute_sigma2_path(r_neg, params, "GJR")
        # Indices 0..30 share the prefix, so sigma2[..31] is identical;
        # they differ first at index 31 (recursion uses eps at index 30).
        np.testing.assert_allclose(s2_pos[:31], s2_neg[:31], atol=1e-12)
        # After the negative shock at t=30, sigma2[31] should be higher.
        assert s2_neg[31] > s2_pos[31]
        # Exact gap: gamma * eps[30]^2 = 0.1 * 4 = 0.4
        assert s2_neg[31] - s2_pos[31] == pytest.approx(0.4, rel=1e-9)

    def test_egarch_recursion_is_in_log_space(self) -> None:
        """log_sigma2_t depends linearly on log_sigma2_{t-1}, so positivity holds
        automatically. Test by constructing a case where omega < 0: variance must
        remain positive.
        """

        params = GARCHParams(omega=-0.05, alpha=0.1, beta=0.95, gamma=-0.1)
        r = np.asarray([0.0, 1.0, -1.5, 0.5, 2.0])
        s2, _ = compute_sigma2_path(r, params, "EGARCH")
        assert np.all(s2 > 0)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------


class TestNegLogLik:
    def test_gaussian_negloglik_matches_closed_form_at_constant_sigma(self) -> None:
        """When alpha = beta = 0 and omega = constant, sigma2 collapses to omega
        for all t >= 2, and the negloglik equals the iid-Gaussian negloglik
        (modulo the first observation's init variance).
        """

        omega = 1.0
        params = GARCHParams(omega=omega, alpha=0.0, beta=0.0)
        rng = np.random.default_rng(seed=7)
        r = rng.standard_normal(500)
        nll = negloglik(r, params, "GARCH", "Gaussian")
        # For t >= 2, sigma2 = omega = 1, so contribution is
        # 0.5 * (log(2pi) + 0 + r^2) per obs.
        # First obs uses init variance ~ var(r[:30]).
        from src.models.garch_family.signal import compute_sigma2_path
        sigma2, _ = compute_sigma2_path(r, params, "GARCH")
        expected = 0.5 * float(np.sum(np.log(2 * math.pi) + np.log(sigma2) + r ** 2 / sigma2))
        assert nll == pytest.approx(expected, abs=1e-9)

    def test_negloglik_returns_inf_on_invalid_student_t_nu(self) -> None:
        params = GARCHParams(omega=0.1, alpha=0.05, beta=0.9, nu=1.5)
        r = np.asarray([0.0, 1.0, -1.0, 0.5])
        nll = negloglik(r, params, "GARCH", "Student-t")
        assert math.isinf(nll) and nll > 0

    def test_negloglik_is_lower_at_true_parameters(self) -> None:
        """The MLE objective must be minimized near the true parameters."""

        rng = np.random.default_rng(seed=11)
        omega, alpha, beta = 0.05, 0.08, 0.9
        t = 1500
        eps = np.empty(t)
        s2 = np.empty(t)
        s2[0] = omega / (1 - alpha - beta)
        for i in range(1, t):
            s2[i] = omega + alpha * eps[i - 1] ** 2 + beta * s2[i - 1]
            eps[i] = float(np.sqrt(s2[i]) * rng.standard_normal())

        nll_true = negloglik(eps, GARCHParams(omega=omega, alpha=alpha, beta=beta), "GARCH", "Gaussian")
        nll_wrong = negloglik(eps, GARCHParams(omega=0.5, alpha=0.3, beta=0.5), "GARCH", "Gaussian")
        assert nll_true < nll_wrong


# ---------------------------------------------------------------------------
# Multi-step forecasts
# ---------------------------------------------------------------------------


def _build_fit(
    spec: str,
    params: GARCHParams,
    sigma2_last: float,
    eps_last: float,
) -> GARCHFit:
    """Minimal GARCHFit needed to call `forecast_variance`."""

    sigma2 = np.asarray([sigma2_last])
    eps = np.asarray([eps_last])
    return GARCHFit(
        params=params,
        spec=spec,  # type: ignore[arg-type]
        distribution="Gaussian",
        mean_model="Zero",
        sigma2_path=sigma2,
        eps_path=eps,
        std_resid=eps / np.sqrt(sigma2),
        log_likelihood=0.0,
        converged=True,
        n_iter=1,
        n_obs=1,
        unconditional_variance=None,
    )


class TestForecast:
    def test_garch_closed_form_matches_recursion(self) -> None:
        """For GARCH(1,1):
            sigma2_{T+h|T} = sigma_bar^2 + (alpha+beta)^{h-1} * (sigma2_{T+1|T} - sigma_bar^2).
        """

        omega, alpha, beta = 0.1, 0.1, 0.85
        params = GARCHParams(omega=omega, alpha=alpha, beta=beta)
        sigma2_last, eps_last = 1.5, 0.8
        fit = _build_fit("GARCH", params, sigma2_last, eps_last)
        horizons = (1, 5, 10, 500)
        fc = forecast_variance(fit, horizons)
        # 1-step:
        sigma2_next = omega + alpha * eps_last ** 2 + beta * sigma2_last
        assert fc[0] == pytest.approx(sigma2_next)
        # Long horizon converges to unconditional variance. At h=500 with
        # persistence 0.95 the residual is 0.95^499 ~ 8e-12, well inside the
        # tolerance.
        unc = omega / (1.0 - alpha - beta)
        assert fc[-1] == pytest.approx(unc, abs=1e-3)
        # Closed form at h=5:
        h = 5
        expected_h = unc + (alpha + beta) ** (h - 1) * (sigma2_next - unc)
        assert fc[1] == pytest.approx(expected_h, abs=1e-12)

    def test_egarch_forecast_converges_to_unconditional_log_variance(self) -> None:
        params = GARCHParams(omega=0.1, alpha=0.1, beta=0.95, gamma=-0.05)
        fit = _build_fit("EGARCH", params, sigma2_last=1.0, eps_last=0.5)
        # Long horizon: beta=0.95 so the geometric decay is ~0.95^h. At
        # h=1000 the residual is ~5e-23 — well inside any reasonable tol.
        fc = forecast_variance(fit, (1, 1000))
        expected_limit = math.exp(params.omega / (1.0 - params.beta))
        assert fc[1] == pytest.approx(expected_limit, rel=1e-3)


class TestAnnualization:
    def test_annualize_vol_pct(self) -> None:
        # 1% daily vol -> 1 * sqrt(252) / 100 ~= 0.1587 annualized.
        assert annualize_vol_pct(1.0) == pytest.approx(math.sqrt(252) / 100.0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestLjungBox:
    def test_iid_fails_to_reject(self) -> None:
        rng = np.random.default_rng(seed=3)
        x = rng.standard_normal(1000)
        q, p = ljung_box(x, lags=10)
        assert p > 0.05  # no significant autocorrelation

    def test_ar1_rejects(self) -> None:
        rng = np.random.default_rng(seed=3)
        x = np.zeros(1000)
        for i in range(1, 1000):
            x[i] = 0.8 * x[i - 1] + rng.standard_normal()
        _, p = ljung_box(x, lags=10)
        assert p < 1e-5


class TestArchLM:
    def test_iid_fails_to_reject(self) -> None:
        rng = np.random.default_rng(seed=4)
        x = rng.standard_normal(1000)
        _, p = arch_lm_test(x, lags=10)
        assert p > 0.05

    def test_arch1_rejects(self) -> None:
        rng = np.random.default_rng(seed=4)
        n = 1000
        x = np.zeros(n)
        for i in range(1, n):
            s = math.sqrt(0.1 + 0.8 * x[i - 1] ** 2)
            x[i] = s * rng.standard_normal()
        _, p = arch_lm_test(x, lags=10)
        assert p < 1e-3


class TestSignBias:
    def test_returns_keys(self) -> None:
        rng = np.random.default_rng(seed=5)
        eps = rng.standard_normal(500)
        z = eps  # iid -> well-specified case
        out = sign_bias_test(z, eps)
        for key in ("joint_F", "joint_p", "sign_bias", "neg_size_bias", "pos_size_bias"):
            assert key in out
        # iid case: sign-bias joint test should not reject.
        assert out["joint_p"] > 0.05


class TestMincerZarnowitz:
    def test_perfect_forecast_recovers_slope_one(self) -> None:
        pred = np.asarray([1.0, 1.5, 2.0, 0.5, 0.75, 1.25])
        rv = pred.copy()
        out = mincer_zarnowitz_regression(rv, pred)
        assert out["slope"] == pytest.approx(1.0, abs=1e-10)
        assert out["intercept"] == pytest.approx(0.0, abs=1e-9)
        assert out["r_squared"] == pytest.approx(1.0, abs=1e-12)

    def test_biased_forecast_recovers_slope_below_one(self) -> None:
        """Predictor systematically twice the realized var -> slope ~ 0.5."""

        rv = np.asarray([1.0, 1.5, 2.0, 0.5, 0.75, 1.25])
        pred = 2.0 * rv
        out = mincer_zarnowitz_regression(rv, pred)
        assert out["slope"] == pytest.approx(0.5, abs=1e-12)
        assert out["intercept"] == pytest.approx(0.0, abs=1e-12)


class TestQLIKE:
    def test_zero_at_perfect_forecast(self) -> None:
        rv = np.asarray([1.0, 1.5, 2.0])
        assert qlike_loss(rv, rv) == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_biased_forecast(self) -> None:
        rv = np.asarray([1.0, 1.5, 2.0])
        assert qlike_loss(rv, 2 * rv) > 0


class TestVaR:
    def test_gaussian_var_matches_z_quantile(self) -> None:
        # VaR_alpha = -(mu + sigma * z_alpha); at mu=0, sigma=1, alpha=5% -> 1.6449.
        assert value_at_risk(1.0, 0.0, 0.05, "Gaussian", None) == pytest.approx(
            1.6448536, abs=1e-5
        )

    def test_student_t_var_is_wider_at_low_nu(self) -> None:
        # Heavier tails at low nu -> the 1% VaR is larger.
        var_gauss = value_at_risk(1.0, 0.0, 0.01, "Gaussian", None)
        var_t = value_at_risk(1.0, 0.0, 0.01, "Student-t", 4.0)
        assert var_t > var_gauss


class TestKupiec:
    def test_kupiec_at_alpha_is_not_significant(self) -> None:
        # Exactly alpha breach rate -> LR = 0, p = 1.
        lr, p = kupiec_pof_test(50, 1000, 0.05)
        assert lr == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_kupiec_rejects_far_off_rate(self) -> None:
        # 20% breaches vs 5% target -> strong rejection.
        _, p = kupiec_pof_test(200, 1000, 0.05)
        assert p < 1e-5


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_log_returns_pct_signature() -> None:
    import pandas as pd

    close = pd.Series(
        [100.0, 101.0, 99.0, 100.0], index=pd.date_range("2024-01-01", periods=4)
    )
    r = log_returns_pct(close)
    assert len(r) == 3
    # 100 * log(101/100) ~= 0.995
    assert r.iloc[0] == pytest.approx(100.0 * math.log(101 / 100), abs=1e-9)


def test_log_returns_pct_on_empty_series() -> None:
    import pandas as pd

    out = log_returns_pct(pd.Series(dtype="float64"))
    assert out.empty
