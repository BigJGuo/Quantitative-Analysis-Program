"""Unit tests for the pure-math layer of Merton-KMV.

Each test pins a formula from the spec against a hand-calculated number or
against a Monte Carlo reference computed inline.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.merton_kmv.signal import (
    cap_struct_arb_stance,
    classify_credit_quality,
    credit_spread,
    default_point,
    diagnostic_residuals,
    distance_to_default,
    equity_price_to_market_cap,
    equity_vol_from_asset_vol,
    estimate_drift_from_assets,
    merton_call_price,
    merton_d1_d2,
    norm_cdf,
    prob_default,
    realized_equity_volatility,
    signal_direction_from_quality,
    signal_strength_from_dd,
    solve_for_v_given_e,
)
from src.models.merton_kmv.types import BalanceSheetSnapshot


def test_norm_cdf_pinned_values() -> None:
    # N(0) = 0.5, N(1) ~ 0.8413, N(-1) ~ 0.1587, N(1.96) ~ 0.9750.
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.0) == pytest.approx(0.8413447460685429, abs=1e-9)
    assert norm_cdf(-1.0) == pytest.approx(0.15865525393145707, abs=1e-9)
    assert norm_cdf(1.96) == pytest.approx(0.9750021048517795, abs=1e-9)


def test_default_point_formula() -> None:
    bs = BalanceSheetSnapshot(short_term_debt=30.0, long_term_debt=50.0)
    assert default_point(bs, 0.5) == pytest.approx(55.0)
    assert default_point(bs, 0.0) == pytest.approx(30.0)
    assert default_point(bs, 1.0) == pytest.approx(80.0)


def test_default_point_rejects_bad_weight() -> None:
    bs = BalanceSheetSnapshot(short_term_debt=10.0, long_term_debt=20.0)
    with pytest.raises(ValueError):
        default_point(bs, 1.5)


def test_merton_d1_d2_against_spec() -> None:
    # Hand-calculation:
    #   V=150, D=55, r=0.04, T=1, sigma=0.25
    #   log(150/55) = log(2.727...) = 1.003302
    #   d1 = (1.003302 + (0.04 + 0.5*0.0625)*1) / (0.25 * 1) = (1.003302 + 0.07125)/0.25
    #      = 1.074552 / 0.25 = 4.298207
    #   d2 = d1 - 0.25 = 4.048207
    d1, d2 = merton_d1_d2(150.0, 55.0, 0.04, 1.0, 0.25)
    assert d1 == pytest.approx(4.298207, abs=1e-4)
    assert d2 == pytest.approx(4.048207, abs=1e-4)


def test_merton_call_price_satisfies_inverse() -> None:
    # Calling solve_for_V_given_E on the BS-implied E should round-trip back to V.
    v_true = 200.0
    d, r, t, sig = 80.0, 0.045, 1.0, 0.30
    e = merton_call_price(v_true, d, r, t, sig)
    v_solved = solve_for_v_given_e(e, d, r, t, sig)
    assert v_solved == pytest.approx(v_true, rel=1e-6)


def test_merton_call_price_in_the_money_lower_bound() -> None:
    # When V >> D, call value approaches V - D*exp(-rT).
    v, d, r, t, sig = 1_000.0, 50.0, 0.05, 1.0, 0.20
    e = merton_call_price(v, d, r, t, sig)
    lower = v - d * math.exp(-r * t)
    assert e >= lower - 1e-6
    assert e == pytest.approx(lower, rel=1e-4)


def test_equity_vol_from_asset_vol_identity() -> None:
    v, d, r, t, sig_v = 150.0, 55.0, 0.04, 1.0, 0.25
    e = merton_call_price(v, d, r, t, sig_v)
    sigma_e = equity_vol_from_asset_vol(v, e, d, r, t, sig_v)
    # sigma_E should exceed sigma_V whenever V > E (leveraged equity).
    assert sigma_e > sig_v
    # And match the Ito identity exactly.
    d1, _ = merton_d1_d2(v, d, r, t, sig_v)
    expected = (v / e) * norm_cdf(d1) * sig_v
    assert sigma_e == pytest.approx(expected, rel=1e-12)


def test_distance_to_default_spec_formula() -> None:
    # V=150, D=55, sigma_V=0.25, mu=0.08, T=1
    # numerator = log(150/55) + (0.08 - 0.5*0.0625)*1
    #           = 1.003302 + 0.04875 = 1.052052
    # denominator = 0.25 * 1 = 0.25
    # DD = 4.208207
    dd = distance_to_default(150.0, 55.0, 0.25, 0.08, 1.0)
    assert dd == pytest.approx(4.208207, abs=1e-4)


def test_distance_to_default_under_risk_neutral_drift() -> None:
    # DD under r should differ from DD under mu by (mu - r)/sigma_V*sqrt(T).
    v, d, sig_v, t = 150.0, 55.0, 0.25, 1.0
    dd_p = distance_to_default(v, d, sig_v, 0.08, t)
    dd_q = distance_to_default(v, d, sig_v, 0.04, t)
    assert dd_p - dd_q == pytest.approx(0.04 / 0.25, abs=1e-6)


def test_prob_default_complement_of_dd() -> None:
    pd_value = prob_default(2.0)
    assert pd_value == pytest.approx(norm_cdf(-2.0), abs=1e-12)
    # Healthy firm has tiny PD.
    assert prob_default(5.0) < 1e-6
    # Distressed firm has materially elevated PD.
    assert prob_default(0.0) == pytest.approx(0.5, abs=1e-12)


def test_credit_spread_formula() -> None:
    # PD_Q = 0.02, LGD = 0.6, T = 1
    # spread = -log(1 - 0.02 * 0.6) = -log(0.988) ~ 0.012073
    s = credit_spread(0.02, 1.0, 0.6)
    assert s == pytest.approx(-math.log(1 - 0.02 * 0.6), rel=1e-12)
    assert s == pytest.approx(0.012073, abs=1e-5)


def test_credit_spread_floor_at_certain_default() -> None:
    s = credit_spread(0.999999, 1.0, 1.0)
    assert s > 5.0  # capped sensible upper bound


def test_classify_credit_quality_buckets() -> None:
    assert classify_credit_quality(5.0) == "investment_grade"
    assert classify_credit_quality(3.0) == "investment_grade"
    assert classify_credit_quality(2.5) == "high_yield"
    assert classify_credit_quality(1.0) == "high_yield"
    assert classify_credit_quality(0.5) == "distressed"
    assert classify_credit_quality(-0.5) == "default_imminent"


def test_cap_struct_arb_stance_logic() -> None:
    assert cap_struct_arb_stance(100.0, None) == "no_market_data"
    assert cap_struct_arb_stance(100.0, 200.0, threshold_bps=25.0) == "cds_rich"
    assert cap_struct_arb_stance(100.0, 50.0, threshold_bps=25.0) == "cds_cheap"
    assert cap_struct_arb_stance(100.0, 110.0, threshold_bps=25.0) == "fair"


def test_signal_direction_from_quality_and_arb() -> None:
    # Arb stance dominates when CDS data is present.
    assert signal_direction_from_quality("investment_grade", "cds_rich") == "short"
    assert signal_direction_from_quality("distressed", "cds_cheap") == "long"
    assert signal_direction_from_quality("high_yield", "fair") == "flat"
    # Without arb data, quality drives direction.
    assert signal_direction_from_quality("investment_grade", "no_market_data") == "long"
    assert signal_direction_from_quality("distressed", "no_market_data") == "short"
    assert signal_direction_from_quality("high_yield", "no_market_data") == "flat"


def test_signal_strength_bounded() -> None:
    assert 0.0 <= signal_strength_from_dd(0.0) <= 1.0
    assert 0.0 <= signal_strength_from_dd(10.0) <= 1.0
    # Strength saturates at high DD.
    assert signal_strength_from_dd(10.0) == pytest.approx(1.0, abs=1e-6)
    # And at the pivot DD ~ 2 it should be near 0.
    assert signal_strength_from_dd(2.0) < 0.1


def test_realized_equity_volatility_recovers_known_gbm_vol() -> None:
    rng = np.random.default_rng(seed=7)
    sigma_true = 0.20
    n = 2520  # 10 years of daily samples for a tight estimate
    log_returns = rng.normal(0.0, sigma_true / math.sqrt(252), size=n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(log_returns)))
    sigma_hat = realized_equity_volatility(prices)
    assert sigma_hat == pytest.approx(sigma_true, abs=0.01)


def test_estimate_drift_from_assets_recovers_known_drift() -> None:
    # Drift estimation is noisy: the stderr of the annualized mean over n
    # daily samples is sigma * sqrt(252 / n). With n=2520, sigma=0.2 we have
    # stderr ~ 0.063, so we allow ~2-sigma tolerance below.
    n = 2520
    mu_true = 0.06
    sigma = 0.20
    rng = np.random.default_rng(seed=11)
    dt = 1.0 / 252.0
    log_returns = (
        (mu_true - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
    )
    v = pd.Series(100.0 * np.exp(np.cumsum(log_returns)))
    mu_hat = estimate_drift_from_assets(v)
    mu_corrected = mu_hat + 0.5 * sigma**2
    assert mu_corrected == pytest.approx(mu_true, abs=0.15)


def test_equity_price_to_market_cap_shape() -> None:
    closes = pd.Series([10.0, 11.0, 12.0])
    cap = equity_price_to_market_cap(closes, 1_000_000.0)
    assert cap.tolist() == [10_000_000.0, 11_000_000.0, 12_000_000.0]


def test_diagnostic_residuals_zero_at_consistent_point() -> None:
    v, d, r, t, sig_v = 200.0, 80.0, 0.03, 1.0, 0.30
    e = merton_call_price(v, d, r, t, sig_v)
    sigma_e = equity_vol_from_asset_vol(v, e, d, r, t, sig_v)
    g1, g2 = diagnostic_residuals(v, e, d, r, t, sig_v, sigma_e)
    assert g1 < 1e-10
    assert g2 < 1e-10


def test_solve_for_v_given_e_handles_low_volatility() -> None:
    # Low vol, V deep in the money: f' = N(d1) ~ 1, so Newton converges fast.
    v_true = 500.0
    e = merton_call_price(v_true, 100.0, 0.02, 1.0, 0.05)
    v_solved = solve_for_v_given_e(e, 100.0, 0.02, 1.0, 0.05)
    assert v_solved == pytest.approx(v_true, rel=1e-8)


def test_solve_for_v_given_e_rejects_non_positive_equity() -> None:
    with pytest.raises(ValueError):
        solve_for_v_given_e(0.0, 50.0, 0.03, 1.0, 0.2)
    with pytest.raises(ValueError):
        solve_for_v_given_e(-1.0, 50.0, 0.03, 1.0, 0.2)
