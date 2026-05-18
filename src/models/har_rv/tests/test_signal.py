"""Unit tests for the pure-math layer of HAR-RV.

Every test checks a worked-out formula against synthetic inputs with known
ground truth, so failures localize precisely to the function under test
(no end-to-end coupling).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.har_rv.signal import (
    aggregate_har_components,
    arch_lm_pvalue,
    build_har_design_matrix,
    compute_daily_rv,
    compute_garman_klass_rv,
    compute_intraday_log_returns,
    compute_yang_zhang_rv,
    diebold_mariano_pvalue,
    fit_har,
    forecast_one_step,
    in_sample_forecasts,
    latest_har_state,
    ljung_box_pvalue,
    mincer_zarnowitz,
    newey_west_covariance,
    nw_optimal_lag,
    ols_fit,
    qlike_loss,
)


class TestIntradayReturns:
    def test_first_bar_of_day_is_nan(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        enriched = compute_intraday_log_returns(synthetic_intraday_bars)
        first_per_day = enriched.groupby("date").head(1)
        assert first_per_day["log_return"].isna().all()

    def test_constant_growth_gives_constant_log_returns(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        """Day 1 in the fixture has price * 1.001 per bar => log return =
        log(1.001) on every non-first bar of the day."""

        enriched = compute_intraday_log_returns(synthetic_intraday_bars)
        day1 = enriched[enriched["date"] == enriched["date"].iloc[0]]
        non_first = day1.dropna(subset=["log_return"])
        np.testing.assert_allclose(
            non_first["log_return"].to_numpy(),
            np.full(len(non_first), np.log(1.001)),
            atol=1e-12,
        )

    def test_rejects_missing_price_column(self) -> None:
        bars = pd.DataFrame({"NotClose": [1.0, 2.0]})
        with pytest.raises(KeyError, match="Close"):
            compute_intraday_log_returns(bars)


class TestComputeDailyRV:
    def test_intraday_rv_matches_manual_sum(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        """Day 1: M=78 bars with 77 non-NaN log returns of log(1.001).

        Without overnight (no prior close), `RV_d[day1] = 77 * log(1.001)^2`.
        """

        components = compute_daily_rv(
            synthetic_intraday_bars, include_overnight=False
        )
        rv_d = components.rv_d
        assert len(rv_d) == 2
        manual_day1 = 77 * np.log(1.001) ** 2
        np.testing.assert_allclose(rv_d.iloc[0], manual_day1, atol=1e-14)

    def test_overnight_inclusion_adds_squared_overnight_return(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        """Day 2's overnight return = log(open_day2 / close_day1)."""

        components = compute_daily_rv(
            synthetic_intraday_bars, include_overnight=True
        )
        no_overnight = compute_daily_rv(
            synthetic_intraday_bars, include_overnight=False
        )
        bars = synthetic_intraday_bars.copy()
        bars["date"] = pd.to_datetime(bars["Datetime"]).dt.date
        day1_close = bars[bars["date"] == bars["date"].iloc[0]]["Close"].iloc[-1]
        day2_open = bars[bars["date"] == bars["date"].iloc[-1]]["Close"].iloc[0]
        overnight = np.log(day2_open / day1_close)
        delta = components.rv_d.iloc[1] - no_overnight.rv_d.iloc[1]
        np.testing.assert_allclose(delta, overnight**2, atol=1e-14)

    def test_semivariances_split_correctly(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        """Day 2: 38 strictly-negative log returns and 39 strictly-positive
        (the boundary bar transitions; the bar where factor flips from 0.999
        to 1.001 contributes positively). Both halves use |log(1.001)|."""

        components = compute_daily_rv(synthetic_intraday_bars)
        assert components.rv_minus is not None
        assert components.rv_plus is not None
        day2_minus = components.rv_minus.iloc[1]
        day2_plus = components.rv_plus.iloc[1]
        # 38 negative log returns of log(0.999), 39 positive log returns of log(1.001)
        # (the first bar of day 2 has NaN return; remaining 77 split 38/39).
        np.testing.assert_allclose(
            day2_minus, 38 * np.log(0.999) ** 2, atol=1e-12
        )
        np.testing.assert_allclose(
            day2_plus, 39 * np.log(1.001) ** 2, atol=1e-12
        )

    def test_bipower_bound_under_no_jump_continuous_returns(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        """Day 1 has constant |r| = R across all 77 returns. The bipower sum
        has M-1 = 76 cross-products of size R^2, the M/(M-1) correction
        rescales to 77 * R^2, and RV = 77 * R^2 — so BV/RV is exactly pi/2.
        """

        components = compute_daily_rv(
            synthetic_intraday_bars, include_overnight=False
        )
        rv = components.rv_d.iloc[0]
        assert components.bipower is not None
        bv = components.bipower.iloc[0]
        ratio = bv / rv
        np.testing.assert_allclose(ratio, math.pi / 2.0, rtol=1e-12)

    def test_empty_input_returns_empty(self) -> None:
        components = compute_daily_rv(pd.DataFrame())
        assert components.rv_d.empty


class TestGarmanKlass:
    def test_constant_inputs_give_constant_gk(
        self, synthetic_daily_ohlc: pd.DataFrame
    ) -> None:
        gk = compute_garman_klass_rv(synthetic_daily_ohlc)
        expected = 0.5 * 0.02**2 - (2.0 * np.log(2.0) - 1.0) * 0.005**2
        np.testing.assert_allclose(gk.to_numpy(), expected, rtol=1e-12)
        assert len(gk) == len(synthetic_daily_ohlc)

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"Open": [1.0], "High": [1.0], "Close": [1.0]})
        with pytest.raises(KeyError):
            compute_garman_klass_rv(df)


class TestYangZhang:
    def test_window_too_small_raises(
        self, synthetic_daily_ohlc: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="window"):
            compute_yang_zhang_rv(synthetic_daily_ohlc, window=1)

    def test_yang_zhang_runs_and_is_positive(
        self, synthetic_daily_ohlc: pd.DataFrame
    ) -> None:
        yz = compute_yang_zhang_rv(synthetic_daily_ohlc, window=10)
        assert (yz > 0).all()
        assert yz.name == "rv_d"


class TestAggregateHARComponents:
    def test_lookahead_burn_in_is_nan(self) -> None:
        rv = pd.Series(
            np.arange(1, 30, dtype=float),
            index=pd.date_range("2025-01-02", periods=29, freq="B"),
            name="rv_d",
        )
        out = aggregate_har_components(rv)
        assert out["rv_w"].iloc[:4].isna().all()
        assert out["rv_w"].iloc[4] == pytest.approx(np.mean([1, 2, 3, 4, 5]))
        assert out["rv_m"].iloc[:21].isna().all()
        assert out["rv_m"].iloc[21] == pytest.approx(np.mean(np.arange(1, 23)))

    def test_rolling_window_does_not_use_future(self) -> None:
        """rv_w[t] should depend only on rv_d[t-4..t]."""

        rv = pd.Series(
            np.arange(1, 50, dtype=float),
            index=pd.date_range("2025-01-02", periods=49, freq="B"),
            name="rv_d",
        )
        out = aggregate_har_components(rv)
        t = 10
        np.testing.assert_allclose(
            out["rv_w"].iloc[t], np.mean(rv.iloc[t - 4 : t + 1])
        )


class TestBuildDesignMatrix:
    def test_log_spec_shapes(self, persistent_log_rv: pd.Series) -> None:
        y, X, idx = build_har_design_matrix(persistent_log_rv, spec="log", horizon=1)
        assert X.shape[1] == 4
        np.testing.assert_allclose(X[:, 0], 1.0)
        assert y.shape[0] == X.shape[0] == len(idx)

    def test_level_spec_does_not_log(self, persistent_log_rv: pd.Series) -> None:
        y, X, _idx = build_har_design_matrix(persistent_log_rv, spec="level")
        # In level spec the daily regressor must equal the corresponding RV_d.
        components = aggregate_har_components(persistent_log_rv).dropna()
        # The valid range loses one row to the dependent-variable shift.
        assert np.allclose(X[:, 1], components["rv_d"].iloc[:-1].to_numpy()[: X.shape[0]])

    def test_horizon_two_uses_two_period_mean(
        self, persistent_log_rv: pd.Series
    ) -> None:
        y1, X1, _i1 = build_har_design_matrix(persistent_log_rv, spec="log", horizon=1)
        y2, X2, _i2 = build_har_design_matrix(persistent_log_rv, spec="log", horizon=2)
        # h=2 has one fewer observation than h=1 because we lose an extra tail.
        assert y2.shape[0] == y1.shape[0] - 1

    def test_rejects_bad_horizon(self, persistent_log_rv: pd.Series) -> None:
        with pytest.raises(ValueError, match="horizon"):
            build_har_design_matrix(persistent_log_rv, horizon=0)

    def test_rejects_too_few_observations(self) -> None:
        rv = pd.Series(
            np.arange(1, 25, dtype=float),
            index=pd.date_range("2025-01-02", periods=24, freq="B"),
            name="rv_d",
        )
        with pytest.raises(ValueError, match="observations"):
            build_har_design_matrix(rv)


class TestOLSAndHAC:
    def test_ols_recovers_known_coefficients(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        X = np.column_stack(
            [np.ones(n), rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)]
        )
        true_theta = np.array([0.5, -1.0, 2.0, 0.3])
        y = X @ true_theta + rng.normal(scale=0.01, size=n)
        theta, _resid, sigma2, r2 = ols_fit(y, X)
        np.testing.assert_allclose(theta, true_theta, atol=5e-3)
        assert r2 > 0.99
        assert sigma2 < 1e-3

    def test_ols_rejects_underdetermined_system(self) -> None:
        X = np.eye(4)
        y = np.zeros(4)
        with pytest.raises(ValueError, match="T > K"):
            ols_fit(y, X)

    def test_nw_lag_matches_formula(self) -> None:
        for T in (50, 100, 200, 1000, 5000):
            assert nw_optimal_lag(T) == int(math.floor(4 * (T / 100) ** (2 / 9)))

    def test_newey_west_reduces_to_white_at_lag_zero(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        e = rng.normal(size=n)
        cov_nw0 = newey_west_covariance(X, e, lag=0)
        XtX_inv = np.linalg.inv(X.T @ X)
        eX = e[:, None] * X
        white = XtX_inv @ (eX.T @ eX) @ XtX_inv
        np.testing.assert_allclose(cov_nw0, white, atol=1e-12)

    def test_newey_west_is_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(1)
        n = 300
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        # AR(1) residuals to exercise the cross-correlation term.
        e = np.empty(n)
        e[0] = rng.normal()
        for t in range(1, n):
            e[t] = 0.7 * e[t - 1] + rng.normal(scale=0.5)
        cov = newey_west_covariance(X, e, lag=5)
        eigs = np.linalg.eigvalsh((cov + cov.T) / 2.0)
        assert eigs.min() > -1e-8


class TestFitHAR:
    def test_persistence_sum_under_strong_persistence(
        self, persistent_log_rv: pd.Series
    ) -> None:
        fit = fit_har(persistent_log_rv, spec="log")
        beta_sum = float(fit.coefficients[1:].sum())
        assert 0.7 < beta_sum < 1.05  # near-unit-root in volatility
        assert fit.r_squared > 0.5
        assert fit.n_obs > 800

    def test_low_persistence_r2_lower(self, low_persistence_rv: pd.Series) -> None:
        fit = fit_har(low_persistence_rv, spec="log")
        # phi=0.2: low autocorrelation, low predictability.
        assert fit.r_squared < 0.20

    def test_level_spec_does_not_take_logs(self, persistent_log_rv: pd.Series) -> None:
        fit_log = fit_har(persistent_log_rv, spec="log")
        fit_level = fit_har(persistent_log_rv, spec="level")
        # The two are different objects; the coefficients should differ by
        # more than rounding.
        assert not np.allclose(fit_log.coefficients, fit_level.coefficients)
        assert fit_level.spec == "level"

    def test_hac_lag_override(self, persistent_log_rv: pd.Series) -> None:
        fit_default = fit_har(persistent_log_rv, spec="log")
        fit_l1 = fit_har(persistent_log_rv, spec="log", hac_lag=1)
        assert fit_default.hac_lag != 1
        assert fit_l1.hac_lag == 1


class TestForecasts:
    def test_log_forecast_applies_lognormal_correction(self) -> None:
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-02", periods=400, freq="B")
        rv = pd.Series(np.exp(rng.normal(-9, 0.3, 400)), index=idx, name="rv_d")
        fit = fit_har(rv, spec="log")
        rv_d_t, rv_w_t, rv_m_t, _ = latest_har_state(rv)
        # The lognormal-corrected prediction must exceed the bare-exp prediction.
        c_, b_d, b_w, b_m = fit.coefficients
        bare = np.exp(
            c_
            + b_d * np.log(rv_d_t)
            + b_w * np.log(rv_w_t)
            + b_m * np.log(rv_m_t)
        )
        corrected = forecast_one_step(fit, rv_d_t, rv_w_t, rv_m_t)
        assert corrected > bare

    def test_level_forecast_is_plain_linear_combo(self) -> None:
        rng = np.random.default_rng(8)
        idx = pd.date_range("2024-01-02", periods=400, freq="B")
        rv = pd.Series(np.exp(rng.normal(-9, 0.3, 400)), index=idx, name="rv_d")
        fit = fit_har(rv, spec="level")
        rv_d_t, rv_w_t, rv_m_t, _ = latest_har_state(rv)
        c_, b_d, b_w, b_m = fit.coefficients
        manual = c_ + b_d * rv_d_t + b_w * rv_w_t + b_m * rv_m_t
        np.testing.assert_allclose(
            forecast_one_step(fit, rv_d_t, rv_w_t, rv_m_t), manual
        )

    def test_in_sample_forecasts_apply_design_matrix(
        self, persistent_log_rv: pd.Series
    ) -> None:
        y, X, _idx = build_har_design_matrix(persistent_log_rv)
        fit = fit_har(persistent_log_rv)
        manual = X @ fit.coefficients
        np.testing.assert_allclose(in_sample_forecasts(fit, X), manual)

    def test_latest_state_uses_last_valid_row(
        self, persistent_log_rv: pd.Series
    ) -> None:
        d, w, m, t = latest_har_state(persistent_log_rv)
        assert t == persistent_log_rv.index[-1]
        assert d == pytest.approx(persistent_log_rv.iloc[-1])
        assert w == pytest.approx(persistent_log_rv.iloc[-5:].mean())
        assert m == pytest.approx(persistent_log_rv.iloc[-22:].mean())

    def test_latest_state_rejects_short_history(self) -> None:
        rv = pd.Series(
            np.arange(1, 10, dtype=float),
            index=pd.date_range("2025-01-02", periods=9, freq="B"),
            name="rv_d",
        )
        with pytest.raises(ValueError, match="not enough"):
            latest_har_state(rv)


class TestDiagnostics:
    def test_mincer_zarnowitz_perfect_forecast(self) -> None:
        realized = np.linspace(0.1, 0.5, 100)
        forecast = realized.copy()
        mz = mincer_zarnowitz(realized, forecast)
        assert mz["a"] == pytest.approx(0.0, abs=1e-9)
        assert mz["b"] == pytest.approx(1.0, abs=1e-9)
        assert mz["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_qlike_is_zero_for_perfect_forecast(self) -> None:
        rng = np.random.default_rng(3)
        realized = np.exp(rng.normal(-9, 0.3, 200))
        assert qlike_loss(realized, realized) == pytest.approx(0.0, abs=1e-12)

    def test_qlike_drops_nonpositive(self) -> None:
        realized = np.array([0.0, 1.0, 2.0, -1.0])
        forecast = np.array([1.0, 1.0, 2.0, 1.0])
        # Only the two strictly-positive pairs contribute; ratios are 1 and 1,
        # so loss must be 0.
        assert qlike_loss(realized, forecast) == pytest.approx(0.0, abs=1e-12)

    def test_qlike_penalizes_biased_forecast(self) -> None:
        realized = np.full(200, 1.0)
        biased = np.full(200, 2.0)
        # QLIKE penalizes both over- and under-forecasting.
        loss_high = qlike_loss(realized, biased)
        loss_low = qlike_loss(realized, np.full(200, 0.5))
        assert loss_high > 0
        assert loss_low > 0

    def test_ljung_box_rejects_strong_ar1(self) -> None:
        rng = np.random.default_rng(4)
        n = 800
        x = np.empty(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = 0.8 * x[t - 1] + rng.normal(scale=0.5)
        p = ljung_box_pvalue(x, max_lag=10)
        assert p < 1e-3

    def test_ljung_box_passes_on_white_noise(self) -> None:
        # Average across seeds: in true white noise, the p-value is uniform
        # on [0, 1], so the mean must hover near 0.5. Asserting on a single
        # draw is fragile (the 5% tail will fire 5% of the time).
        rng = np.random.default_rng(20260518)
        pvals = []
        for _ in range(20):
            white = rng.normal(size=1000)
            pvals.append(ljung_box_pvalue(white, max_lag=10))
        assert np.mean(pvals) > 0.3

    def test_arch_lm_rejects_arch_process(self) -> None:
        rng = np.random.default_rng(6)
        n = 800
        sigma2 = np.empty(n)
        x = np.empty(n)
        sigma2[0] = 1.0
        x[0] = rng.normal()
        for t in range(1, n):
            sigma2[t] = 0.05 + 0.9 * x[t - 1] ** 2
            x[t] = rng.normal() * np.sqrt(sigma2[t])
        p = arch_lm_pvalue(x, lags=5)
        assert p < 1e-3

    def test_arch_lm_passes_on_iid(self) -> None:
        # Single-seed flake-resistant version: average over draws.
        rng = np.random.default_rng(20260518)
        pvals = []
        for _ in range(20):
            x = rng.normal(size=1000)
            pvals.append(arch_lm_pvalue(x, lags=5))
        assert np.mean(pvals) > 0.3

    def test_diebold_mariano_zero_difference(self) -> None:
        loss = np.full(100, 0.5)
        assert math.isnan(diebold_mariano_pvalue(loss, loss))

    def test_diebold_mariano_detects_difference(self) -> None:
        rng = np.random.default_rng(9)
        n = 500
        a = rng.normal(loc=0.5, scale=0.1, size=n)
        b = rng.normal(loc=0.3, scale=0.1, size=n)
        p = diebold_mariano_pvalue(a, b)
        assert 0.0 <= p < 1e-3
