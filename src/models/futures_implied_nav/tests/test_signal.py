"""Unit tests for `src.models.futures_implied_nav.signal`.

Every math function is exercised against a hand-computed answer on small
synthetic data so failures point straight at a formula change.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from src.models.futures_implied_nav.signal import (
    classify_action,
    compute_fair_value,
    compute_premium,
    compute_result,
    fx_adjustment,
    hedge_return,
    implied_basket_return,
    inverse_variance_weights,
    multi_source_fair_value,
    nav_official_usd,
    signal_strength,
    single_factor_fair_value,
)
from src.models.futures_implied_nav.types import (
    BetaVector,
    BlendWeights,
    CostParameters,
    FuturesImpliedNAVInputs,
    HedgeSpec,
    NAVAnchor,
    SourceVariances,
)


def test_hedge_return_basic() -> None:
    assert hedge_return(110.0, 100.0) == pytest.approx(0.10)
    assert hedge_return(95.0, 100.0) == pytest.approx(-0.05)


def test_hedge_return_rejects_nonpositive_close() -> None:
    with pytest.raises(ValueError):
        hedge_return(100.0, 0.0)


def test_implied_basket_return_weighted_sum() -> None:
    beta = BetaVector(
        tickers=("A", "B"),
        betas=(0.5, 0.3),
        ridge_lambda=0.0,
        r_squared=1.0,
        residual_variance=0.0,
    )
    r_h = {"A": 0.02, "B": -0.01}
    expected = 0.5 * 0.02 + 0.3 * -0.01
    assert implied_basket_return(beta, r_h) == pytest.approx(expected)


def test_implied_basket_return_missing_hedge_contributes_zero() -> None:
    beta = BetaVector(
        tickers=("A", "B"),
        betas=(0.5, 0.3),
        ridge_lambda=0.0,
        r_squared=1.0,
        residual_variance=0.0,
    )
    assert implied_basket_return(beta, {"A": 0.02}) == pytest.approx(0.5 * 0.02)


def test_fx_adjustment_single_currency() -> None:
    # 5% JPY move -> exactly 1.05 ratio
    fx_now = {"JPYUSD=X": 1.05}
    fx_close = {"JPYUSD=X": 1.00}
    assert fx_adjustment(fx_now, fx_close) == pytest.approx(1.05)


def test_fx_adjustment_two_currencies_geometric_mean() -> None:
    fx_now = {"JPYUSD=X": 1.10, "EURUSD=X": 0.90}
    fx_close = {"JPYUSD=X": 1.00, "EURUSD=X": 1.00}
    expected = math.exp(0.5 * (math.log(1.10) + math.log(0.90)))
    assert fx_adjustment(fx_now, fx_close) == pytest.approx(expected)


def test_fx_adjustment_no_fx_returns_one() -> None:
    assert fx_adjustment({}, {}) == 1.0


def test_single_factor_fair_value_matches_spec_formula() -> None:
    # FV = NAV_close * (1 + r_B_hat) * fx_ratio
    fv = single_factor_fair_value(nav_close=100.0, implied_return=0.02, fx_ratio=1.01)
    assert fv == pytest.approx(100.0 * 1.02 * 1.01)


def test_inverse_variance_weights_sum_to_one_and_proportional() -> None:
    variances = SourceVariances(sigma2_etf=4.0, sigma2_nav=1.0, sigma2_hedge=2.0)
    w = inverse_variance_weights(variances)
    assert w.w_etf + w.w_nav + w.w_hedge == pytest.approx(1.0)
    # precision ratios: 0.25 : 1.0 : 0.5 -> total 1.75
    total = 0.25 + 1.0 + 0.5
    assert w.w_etf == pytest.approx(0.25 / total)
    assert w.w_nav == pytest.approx(1.0 / total)
    assert w.w_hedge == pytest.approx(0.5 / total)


def test_inverse_variance_weights_lowest_variance_dominates() -> None:
    # When hedges have tiny variance, w_hedge should approach 1.
    variances = SourceVariances(sigma2_etf=1.0, sigma2_nav=1.0, sigma2_hedge=1e-8)
    w = inverse_variance_weights(variances)
    assert w.w_hedge > 0.99


def test_multi_source_fair_value_blends_linearly() -> None:
    weights = BlendWeights(w_etf=0.1, w_nav=0.2, w_hedge=0.7)
    fv = multi_source_fair_value(weights, etf_mid=100.0, nav_official=99.0, fv_hedge=101.0)
    assert fv == pytest.approx(0.1 * 100.0 + 0.2 * 99.0 + 0.7 * 101.0)


def test_compute_premium_positive_when_etf_rich() -> None:
    assert compute_premium(101.0, 100.0) == pytest.approx(0.01)
    assert compute_premium(99.0, 100.0) == pytest.approx(-0.01)


def test_compute_premium_rejects_nonpositive_fv() -> None:
    with pytest.raises(ValueError):
        compute_premium(100.0, 0.0)


def test_classify_action_thresholds() -> None:
    cost = CostParameters(half_spread_bps=2.0, cost_band_bps=3.0)  # 5 bp total
    assert classify_action(0.0006, cost) == "SELL_ETF"   # 6 bp > 5 bp
    assert classify_action(-0.0006, cost) == "BUY_ETF"
    assert classify_action(0.0004, cost) == "FLAT"
    assert classify_action(-0.0004, cost) == "FLAT"


def test_signal_strength_bounded() -> None:
    cost = CostParameters(half_spread_bps=2.0, cost_band_bps=3.0)
    # 5 bp band; premium 5+25=30 bps -> excess 25 bps = scale_bps -> strength 1.0
    assert signal_strength(0.0030, cost, scale_bps=25.0) == pytest.approx(1.0)
    # Within band -> strength 0.
    assert signal_strength(0.0003, cost) == 0.0
    # Far beyond -> still clipped to 1.
    assert signal_strength(0.05, cost) == 1.0


def test_nav_official_scales_with_fx() -> None:
    anchor = NAVAnchor(
        nav_close=100.0,
        strike_time=datetime(2024, 1, 1, 16, 0),
        hedge_close={"X": 1.0},
        fx_close={"FX": 1.0},
    )
    assert nav_official_usd(anchor, 1.02) == pytest.approx(102.0)


def _make_inputs(
    *,
    nav_close: float = 100.0,
    etf_mid: float = 100.5,
    hedge_now: dict[str, float] | None = None,
    hedge_close: dict[str, float] | None = None,
    fx_now: dict[str, float] | None = None,
    fx_close: dict[str, float] | None = None,
    sigma2_etf: float = 1.0,
    sigma2_nav: float = 100.0,
    sigma2_hedge: float = 1e-3,
    cost_params: CostParameters | None = None,
    beta_tickers: tuple[str, ...] = ("H1",),
    beta_values: tuple[float, ...] = (1.0,),
) -> FuturesImpliedNAVInputs:
    if hedge_now is None:
        hedge_now = {"H1": 102.0}
    if hedge_close is None:
        hedge_close = {"H1": 100.0}
    if fx_now is None:
        fx_now = {}
    if fx_close is None:
        fx_close = {}
    anchor = NAVAnchor(
        nav_close=nav_close,
        strike_time=datetime(2024, 1, 1, 16, 0),
        hedge_close=hedge_close,
        fx_close=fx_close,
    )
    beta = BetaVector(
        tickers=beta_tickers,
        betas=beta_values,
        ridge_lambda=1e-3,
        r_squared=0.9,
        residual_variance=sigma2_hedge,
    )
    variances = SourceVariances(
        sigma2_etf=sigma2_etf, sigma2_nav=sigma2_nav, sigma2_hedge=sigma2_hedge
    )
    return FuturesImpliedNAVInputs(
        etf_ticker="EWJ",
        timestamp=datetime(2024, 1, 1, 18, 0),
        hedges=(HedgeSpec(ticker="H1", kind="equity"),),
        hedge_now=hedge_now,
        fx_now=fx_now,
        anchor=anchor,
        etf_mid=etf_mid,
        beta=beta,
        variances=variances,
        cost_params=cost_params or CostParameters(half_spread_bps=2.0, cost_band_bps=3.0),
    )


def test_compute_fair_value_no_fx_single_hedge() -> None:
    inputs = _make_inputs(
        nav_close=100.0,
        etf_mid=102.0,
        hedge_now={"H1": 102.0},
        hedge_close={"H1": 100.0},
    )
    # r_B_hat = 1.0 * 0.02 = 0.02; fx_ratio = 1.0; FV_hedge = 102.0
    decomp = compute_fair_value(inputs)
    assert decomp.implied_basket_return == pytest.approx(0.02)
    assert decomp.fx_ratio == pytest.approx(1.0)
    assert decomp.fv_hedge == pytest.approx(102.0)
    assert decomp.nav_official == pytest.approx(100.0)
    # With sigma2_hedge tiny, w_hedge ~ 1, so FV_total ~ 102
    assert decomp.fv_total == pytest.approx(102.0, rel=1e-3)


def test_compute_result_flags_action_when_rich() -> None:
    # Premium = (M_t - FV_t)/FV_t. Set etf_mid above FV by > band.
    inputs = _make_inputs(
        nav_close=100.0,
        etf_mid=102.5,
        hedge_now={"H1": 102.0},  # FV_hedge = 102.0
        hedge_close={"H1": 100.0},
        sigma2_hedge=1e-12,        # force FV_total to be FV_hedge essentially
        cost_params=CostParameters(half_spread_bps=2.0, cost_band_bps=3.0),  # 5 bp
    )
    result = compute_result(inputs)
    # Premium ~ (102.5 - 102.0)/102.0 ~ 49 bps -> well past 5 bp band, ETF is rich.
    assert result.action == "SELL_ETF"
    assert result.premium > 0
    assert result.fair_value == pytest.approx(102.0, rel=1e-3)


def test_compute_result_action_flat_within_band() -> None:
    inputs = _make_inputs(
        nav_close=100.0,
        etf_mid=102.01,
        hedge_now={"H1": 102.0},
        hedge_close={"H1": 100.0},
        sigma2_hedge=1e-12,
        cost_params=CostParameters(half_spread_bps=5.0, cost_band_bps=10.0),  # 15 bp
    )
    result = compute_result(inputs)
    # Premium ~ 1 bp inside 15 bp band -> FLAT.
    assert result.action == "FLAT"
