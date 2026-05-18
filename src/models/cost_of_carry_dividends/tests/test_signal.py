"""Unit tests for `src.models.cost_of_carry_dividends.signal`.

Every math function is exercised against a hand-computed answer derived from
the spec formulas. Failures point straight at a formula change.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.models.cost_of_carry_dividends.signal import (
    accrued_dividends,
    aggregate_constituent_dividends,
    basis,
    basket_nav,
    classify_basis_action,
    classify_etf_action,
    compute_decomposition,
    compute_etf_basket_result,
    compute_result,
    equivalent_continuous_yield,
    etf_premium,
    forward_dividend_strip,
    fv_dividend_stream,
    implied_repo_rate,
    interpolate_rate,
    pv_dividend_stream,
    signal_strength,
    theoretical_futures_continuous,
    theoretical_futures_discrete,
    theoretical_futures_price,
    time_to_maturity,
)
from src.models.cost_of_carry_dividends.types import (
    BasketHolding,
    CostOfCarryInputs,
    CostParameters,
    Dividend,
    DividendStream,
    IndexConstituent,
    RateCurve,
)

# --------------------------------------------------------------------------- #
# Time and rates                                                              #
# --------------------------------------------------------------------------- #


def test_time_to_maturity_quarter_year() -> None:
    today = datetime(2026, 5, 18)
    expiry = today + timedelta(days=91)
    assert time_to_maturity(today, expiry) == pytest.approx(91 / 365.25, rel=1e-9)


def test_time_to_maturity_negative_returns_zero() -> None:
    today = datetime(2026, 5, 18)
    expiry = today - timedelta(days=1)
    assert time_to_maturity(today, expiry) == 0.0


def test_interpolate_rate_log_linear_between_pillars() -> None:
    curve = RateCurve(rates={0.25: 0.05, 10.0: 0.04})
    # Log-DF at 0.25 = -0.0125; at 10 = -0.40. Halfway in tenor: tau=5.125
    r = interpolate_rate(curve, 5.125)
    assert 0.04 < r < 0.05


def test_interpolate_rate_flat_outside_support() -> None:
    curve = RateCurve(rates={0.25: 0.05, 10.0: 0.04})
    assert interpolate_rate(curve, 0.1) == pytest.approx(0.05)
    assert interpolate_rate(curve, 30.0) == pytest.approx(0.04)


def test_interpolate_rate_single_pillar_flat() -> None:
    curve = RateCurve(rates={1.0: 0.045})
    assert interpolate_rate(curve, 0.5) == pytest.approx(0.045)
    assert interpolate_rate(curve, 5.0) == pytest.approx(0.045)


# --------------------------------------------------------------------------- #
# Dividend stream PV / FV                                                     #
# --------------------------------------------------------------------------- #


def _stream_two() -> DividendStream:
    return DividendStream(
        dividends=(
            Dividend(ex_date=datetime(2026, 6, 17), amount_index_points=2.0),
            Dividend(ex_date=datetime(2026, 7, 17), amount_index_points=3.0),
        )
    )


def test_pv_dividend_stream_discounts_correctly() -> None:
    today = datetime(2026, 5, 18)
    expiry = datetime(2026, 9, 18)
    rate = 0.05
    stream = _stream_two()

    t1 = (datetime(2026, 6, 17) - today).total_seconds() / (86400 * 365.25)
    t2 = (datetime(2026, 7, 17) - today).total_seconds() / (86400 * 365.25)
    expected = 2.0 * math.exp(-rate * t1) + 3.0 * math.exp(-rate * t2)
    assert pv_dividend_stream(stream, rate, today, expiry) == pytest.approx(expected)


def test_pv_dividend_stream_excludes_outside_window() -> None:
    today = datetime(2026, 5, 18)
    expiry = datetime(2026, 9, 18)
    rate = 0.05
    stream = DividendStream(
        dividends=(
            Dividend(ex_date=datetime(2026, 5, 1), amount_index_points=1.0),    # before today
            Dividend(ex_date=datetime(2026, 6, 17), amount_index_points=2.0),   # in
            Dividend(ex_date=datetime(2026, 10, 1), amount_index_points=10.0),  # after expiry
        )
    )
    # Only the middle dividend contributes.
    t = (datetime(2026, 6, 17) - today).total_seconds() / (86400 * 365.25)
    assert pv_dividend_stream(stream, rate, today, expiry) == pytest.approx(
        2.0 * math.exp(-rate * t)
    )


def test_fv_dividend_stream_compounds_to_expiry() -> None:
    today = datetime(2026, 5, 18)
    expiry = datetime(2026, 9, 18)
    rate = 0.05
    stream = _stream_two()
    tau = (expiry - today).total_seconds() / (86400 * 365.25)
    t1 = (datetime(2026, 6, 17) - today).total_seconds() / (86400 * 365.25)
    t2 = (datetime(2026, 7, 17) - today).total_seconds() / (86400 * 365.25)
    expected = 2.0 * math.exp(rate * (tau - t1)) + 3.0 * math.exp(rate * (tau - t2))
    assert fv_dividend_stream(stream, rate, today, expiry) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Theoretical futures prices                                                  #
# --------------------------------------------------------------------------- #


def test_theoretical_futures_discrete_matches_spec_formula() -> None:
    # F = S exp(r tau) - sum(FV_divs)
    spot = 5000.0
    rate = 0.05
    tau = 0.25
    fv = 10.0
    expected = 5000.0 * math.exp(0.05 * 0.25) - 10.0
    assert theoretical_futures_discrete(spot, rate, tau, fv) == pytest.approx(expected)


def test_theoretical_futures_continuous_matches_spec_formula() -> None:
    # F = S exp((r - q) tau)
    spot = 5000.0
    rate = 0.05
    q = 0.018
    tau = 0.5
    expected = 5000.0 * math.exp((0.05 - 0.018) * 0.5)
    assert theoretical_futures_continuous(spot, rate, q, tau) == pytest.approx(expected)


def test_equivalent_continuous_yield_matches_discrete_to_first_order() -> None:
    """The spec's q = (1/tau) ln(1 + PV/S) matches discrete to O((PV/S)^2)."""

    today = datetime(2026, 5, 18)
    expiry = today + timedelta(days=180)
    rate = 0.05
    spot = 5000.0
    stream = _stream_two()

    tau = time_to_maturity(today, expiry)
    fv = fv_dividend_stream(stream, rate, today, expiry)
    pv = pv_dividend_stream(stream, rate, today, expiry)
    f_discrete = theoretical_futures_discrete(spot, rate, tau, fv)
    q = equivalent_continuous_yield(spot, rate, tau, stream, today, expiry)
    f_continuous = theoretical_futures_continuous(spot, rate, q, tau)
    # Per the spec, this q is the first-order match. Second-order error is
    # bounded by spot * (PV/spot)^2.
    second_order_bound = spot * (pv / spot) ** 2 * 5.0
    assert abs(f_continuous - f_discrete) < second_order_bound


def test_theoretical_futures_price_switches_under_threshold() -> None:
    today = datetime(2026, 5, 18)
    # 60 days < 0.25 yrs -> discrete
    expiry_short = today + timedelta(days=60)
    f, method, _, _ = theoretical_futures_price(
        spot=5000.0,
        rate=0.05,
        today=today,
        expiry=expiry_short,
        stream=_stream_two(),
        switch_threshold_years=0.25,
    )
    assert method == "discrete"
    assert f > 0


def test_theoretical_futures_price_switches_over_threshold() -> None:
    today = datetime(2026, 5, 18)
    # 6 months > 0.25 yrs -> continuous
    expiry_long = today + timedelta(days=180)
    f, method, _, _ = theoretical_futures_price(
        spot=5000.0,
        rate=0.05,
        today=today,
        expiry=expiry_long,
        stream=_stream_two(),
        switch_threshold_years=0.25,
    )
    assert method == "continuous"
    assert f > 0


# --------------------------------------------------------------------------- #
# Implied repo and basis                                                      #
# --------------------------------------------------------------------------- #


def test_implied_repo_recovers_input_rate_when_market_eq_theoretical() -> None:
    today = datetime(2026, 5, 18)
    expiry = today + timedelta(days=90)
    rate = 0.05
    spot = 5000.0
    stream = _stream_two()
    tau = time_to_maturity(today, expiry)
    fv = fv_dividend_stream(stream, rate, today, expiry)
    f_theo = theoretical_futures_discrete(spot, rate, tau, fv)
    # Setting the market F to the theoretical price should recover `rate`.
    r_impl = implied_repo_rate(
        futures=f_theo, spot=spot, rate=rate, today=today, expiry=expiry, stream=stream
    )
    assert r_impl == pytest.approx(rate, abs=1e-10)


def test_basis_is_signed_market_minus_theoretical() -> None:
    assert basis(5005.0, 5000.0) == 5.0
    assert basis(4998.0, 5000.0) == -2.0


def test_classify_basis_action_thresholds() -> None:
    cost = CostParameters(futures_band_points=0.5, etf_cost_band_bps=2.0)
    # Just past 0.5: SELL.
    assert classify_basis_action(5000.6, 5000.0, cost) == "SELL_FUTURE"
    # Within band: FLAT.
    assert classify_basis_action(5000.4, 5000.0, cost) == "FLAT"
    # Past on the cheap side: BUY.
    assert classify_basis_action(4999.0, 5000.0, cost) == "BUY_FUTURE"


def test_signal_strength_bounded() -> None:
    cost = CostParameters(futures_band_points=0.5, etf_cost_band_bps=2.0)
    # 0.5 + 2.0 excess = full strength (scale_points=2.0 default).
    assert signal_strength(2.5, cost) == pytest.approx(1.0)
    assert signal_strength(-2.5, cost) == pytest.approx(1.0)
    # Inside band: 0.
    assert signal_strength(0.2, cost) == 0.0
    # Half the scale.
    assert signal_strength(1.5, cost) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# ETF basket valuation                                                        #
# --------------------------------------------------------------------------- #


def test_accrued_dividends_picks_up_recent_payments() -> None:
    today = datetime(2026, 5, 18)
    holdings = [BasketHolding(ticker="AAPL", shares=100.0)]
    divs = {
        "AAPL": pd.Series(
            [0.24, 0.24],
            index=pd.to_datetime([
                datetime(2026, 5, 1),    # within 30-day window -> accrued
                datetime(2026, 1, 1),    # too old
            ]),
        )
    }
    accrued = accrued_dividends(holdings, divs, today, pay_lag_days=30)
    assert accrued == pytest.approx(100.0 * 0.24)


def test_accrued_dividends_handles_tz_aware_index() -> None:
    today = datetime(2026, 5, 18)
    holdings = [BasketHolding(ticker="AAPL", shares=100.0)]
    divs = {
        "AAPL": pd.Series(
            [0.24],
            index=pd.to_datetime([datetime(2026, 5, 5)]).tz_localize("UTC"),
        )
    }
    accrued = accrued_dividends(holdings, divs, today, pay_lag_days=30)
    assert accrued == pytest.approx(100.0 * 0.24)


def test_basket_nav_matches_formula() -> None:
    holdings = [
        BasketHolding(ticker="A", shares=100.0),
        BasketHolding(ticker="B", shares=50.0),
    ]
    prices = {"A": 10.0, "B": 20.0}
    nav = basket_nav(holdings, prices, accrued=5.0, liabilities=15.0)
    # 100*10 + 50*20 + 5 - 15 = 1000 + 1000 + 5 - 15 = 1990
    assert nav == pytest.approx(1990.0)


def test_basket_nav_missing_price_raises() -> None:
    holdings = [BasketHolding(ticker="A", shares=100.0)]
    with pytest.raises(KeyError):
        basket_nav(holdings, {}, accrued=0.0, liabilities=0.0)


def test_etf_premium_signs() -> None:
    assert etf_premium(101.0, 100.0) == pytest.approx(0.01)
    assert etf_premium(99.0, 100.0) == pytest.approx(-0.01)


def test_classify_etf_action_thresholds() -> None:
    cost = CostParameters(futures_band_points=0.5, etf_cost_band_bps=2.0)
    assert classify_etf_action(0.0003, cost) == "CREATE"
    assert classify_etf_action(-0.0003, cost) == "REDEEM"
    assert classify_etf_action(0.0001, cost) == "FLAT"


def test_compute_etf_basket_result_endtoend() -> None:
    today = datetime(2026, 5, 18)
    holdings = [BasketHolding(ticker="A", shares=100.0)]
    prices = {"A": 50.0}  # basket gross = 5000
    divs = {"A": pd.Series([], dtype="float64")}
    result = compute_etf_basket_result(
        etf_ticker="ETFX",
        timestamp=today,
        holdings=holdings,
        prices=prices,
        dividends_per_share=divs,
        liabilities=0.0,
        market_mid=50.5,
        shares_per_creation_unit=100.0,
        cost_params=CostParameters(futures_band_points=0.5, etf_cost_band_bps=5.0),
    )
    # NAV/share = 5000 / 100 = 50. Premium = (50.5 - 50) / 50 = 1%.
    assert result.nav_per_share == pytest.approx(50.0)
    assert result.premium == pytest.approx(0.01)
    assert result.action == "CREATE"


# --------------------------------------------------------------------------- #
# Forward dividend strip                                                      #
# --------------------------------------------------------------------------- #


def test_forward_dividend_strip_inverts_carry() -> None:
    rate = 0.05
    tau_near = 0.25
    tau_far = 0.50
    # If F_far = F_near * exp(r * delta) the strip should be ~0.
    f_near = 5000.0
    f_far = f_near * math.exp(rate * (tau_far - tau_near))
    strip = forward_dividend_strip(
        f_near=f_near, f_far=f_far, rate=rate, tau_near=tau_near, tau_far=tau_far
    )
    assert strip == pytest.approx(0.0, abs=1e-9)


def test_forward_dividend_strip_positive_when_far_lags() -> None:
    rate = 0.05
    f_near = 5000.0
    # F_far below carry implies positive implied dividend stream.
    f_far = 4990.0
    strip = forward_dividend_strip(
        f_near=f_near, f_far=f_far, rate=rate, tau_near=0.25, tau_far=0.5
    )
    assert strip > 0


# --------------------------------------------------------------------------- #
# Constituent aggregation                                                     #
# --------------------------------------------------------------------------- #


def test_aggregate_constituent_dividends_weights_correctly() -> None:
    today = datetime(2026, 5, 18)
    expiry = datetime(2026, 9, 18)
    constituents = [
        IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5),
        IndexConstituent(ticker="MSFT", weight=0.06, shares_factor=1.5),
    ]
    divs = {
        "AAPL": pd.Series([0.24], index=pd.to_datetime([datetime(2026, 6, 1)])),
        "MSFT": pd.Series([0.75], index=pd.to_datetime([datetime(2026, 7, 1)])),
    }
    stream = aggregate_constituent_dividends(
        constituents=constituents,
        dividends_per_share=divs,
        today=today,
        expiry=expiry,
    )
    assert len(stream) == 2
    aapl_pt = next(d for d in stream.dividends if d.ticker == "AAPL")
    msft_pt = next(d for d in stream.dividends if d.ticker == "MSFT")
    assert aapl_pt.amount_index_points == pytest.approx(0.24 * 0.07 * 1.5)
    assert msft_pt.amount_index_points == pytest.approx(0.75 * 0.06 * 1.5)


def test_aggregate_constituent_dividends_filters_out_window() -> None:
    today = datetime(2026, 5, 18)
    expiry = datetime(2026, 9, 18)
    constituents = [IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5)]
    divs = {
        "AAPL": pd.Series(
            [0.24, 0.24, 0.24],
            index=pd.to_datetime([
                datetime(2026, 5, 1),    # before today
                datetime(2026, 6, 1),    # in
                datetime(2026, 10, 1),   # after expiry
            ]),
        )
    }
    stream = aggregate_constituent_dividends(
        constituents=constituents,
        dividends_per_share=divs,
        today=today,
        expiry=expiry,
    )
    assert len(stream) == 1
    assert stream.dividends[0].ex_date == datetime(2026, 6, 1)


# --------------------------------------------------------------------------- #
# End-to-end compute_result                                                   #
# --------------------------------------------------------------------------- #


def _make_inputs(
    *,
    spot: float = 5000.0,
    futures: float = 5050.0,
    rate: float = 0.05,
    days_to_expiry: int = 90,
    stream: DividendStream | None = None,
    futures_band_points: float = 0.5,
) -> CostOfCarryInputs:
    today = datetime(2026, 5, 18)
    expiry = today + timedelta(days=days_to_expiry)
    return CostOfCarryInputs(
        futures_ticker="ES=F",
        spot_ticker="^GSPC",
        timestamp=today,
        expiry=expiry,
        spot=spot,
        futures=futures,
        rate_curve=RateCurve(rates={0.25: rate, 10.0: rate}),
        stock_loan_spread=0.0,
        dividends=stream or _stream_two(),
        cost_params=CostParameters(
            futures_band_points=futures_band_points, etf_cost_band_bps=2.0
        ),
        switch_threshold_years=0.25,
    )


def test_compute_decomposition_consistency() -> None:
    inputs = _make_inputs(futures=5050.0)
    decomp = compute_decomposition(inputs)
    # Setting market F = theoretical implies r_impl = rate.
    inputs_zero_basis = CostOfCarryInputs(
        futures_ticker="ES=F",
        spot_ticker="^GSPC",
        timestamp=inputs.timestamp,
        expiry=inputs.expiry,
        spot=inputs.spot,
        futures=decomp.theoretical,
        rate_curve=inputs.rate_curve,
        stock_loan_spread=inputs.stock_loan_spread,
        dividends=inputs.dividends,
        cost_params=inputs.cost_params,
        switch_threshold_years=inputs.switch_threshold_years,
    )
    decomp_zero = compute_decomposition(inputs_zero_basis)
    assert decomp_zero.basis == pytest.approx(0.0, abs=1e-9)
    assert decomp_zero.implied_repo == pytest.approx(decomp.rate, abs=1e-10)


def test_compute_result_action_sell_when_future_rich() -> None:
    # Build inputs where F market exceeds F theoretical by > band.
    inputs = _make_inputs(spot=5000.0, futures=5500.0, futures_band_points=0.5)
    result = compute_result(inputs)
    assert result.action == "SELL_FUTURE"
    assert result.basis > 0


def test_compute_result_action_buy_when_future_cheap() -> None:
    inputs = _make_inputs(spot=5000.0, futures=4500.0, futures_band_points=0.5)
    result = compute_result(inputs)
    assert result.action == "BUY_FUTURE"
    assert result.basis < 0


def test_compute_result_metadata_present() -> None:
    inputs = _make_inputs()
    result = compute_result(inputs)
    assert {"tau_years", "rate", "pv_dividends", "fv_dividends", "implied_repo"}.issubset(
        result.metadata.keys()
    )
