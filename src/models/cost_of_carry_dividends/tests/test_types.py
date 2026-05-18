"""Validation tests for `types.py` dataclasses."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.models.cost_of_carry_dividends.types import (
    BasketHolding,
    CostParameters,
    Dividend,
    DividendStream,
    IndexConstituent,
    RateCurve,
)


def test_dividend_validates_amount_non_negative() -> None:
    with pytest.raises(ValueError):
        Dividend(ex_date=datetime(2026, 6, 1), amount_index_points=-1.0)


def test_dividend_validates_source() -> None:
    with pytest.raises(ValueError):
        Dividend(ex_date=datetime(2026, 6, 1), amount_index_points=1.0, source="weird")  # type: ignore[arg-type]


def test_dividend_stream_sorts_by_ex_date() -> None:
    d1 = Dividend(ex_date=datetime(2026, 6, 10), amount_index_points=1.0)
    d2 = Dividend(ex_date=datetime(2026, 6, 1), amount_index_points=2.0)
    d3 = Dividend(ex_date=datetime(2026, 6, 5), amount_index_points=3.0)
    stream = DividendStream(dividends=(d1, d2, d3))
    assert stream.dividends == (d2, d3, d1)
    assert stream.total_index_points == pytest.approx(6.0)


def test_dividend_stream_filter_window_inclusive_upper() -> None:
    d_in = Dividend(ex_date=datetime(2026, 6, 5), amount_index_points=1.0)
    d_out_left = Dividend(ex_date=datetime(2026, 5, 30), amount_index_points=1.0)
    d_out_right = Dividend(ex_date=datetime(2026, 7, 5), amount_index_points=1.0)
    stream = DividendStream(dividends=(d_in, d_out_left, d_out_right))
    filtered = stream.filter_window(datetime(2026, 6, 1), datetime(2026, 6, 30))
    assert filtered.dividends == (d_in,)


def test_rate_curve_rejects_empty() -> None:
    with pytest.raises(ValueError):
        RateCurve(rates={})


def test_rate_curve_rejects_non_positive_tenor() -> None:
    with pytest.raises(ValueError):
        RateCurve(rates={0.0: 0.05})


def test_rate_curve_rejects_extreme_rates() -> None:
    with pytest.raises(ValueError):
        RateCurve(rates={1.0: 0.50})


def test_rate_curve_tenors_sorted() -> None:
    curve = RateCurve(rates={5.0: 0.04, 0.25: 0.05, 10.0: 0.045})
    assert curve.tenors == (0.25, 5.0, 10.0)


def test_index_constituent_validation() -> None:
    with pytest.raises(ValueError):
        IndexConstituent(ticker="", weight=0.1, shares_factor=1.0)
    with pytest.raises(ValueError):
        IndexConstituent(ticker="AAPL", weight=-0.1, shares_factor=1.0)
    with pytest.raises(ValueError):
        IndexConstituent(ticker="AAPL", weight=0.1, shares_factor=0.0)


def test_basket_holding_validation() -> None:
    with pytest.raises(ValueError):
        BasketHolding(ticker="", shares=10.0)
    with pytest.raises(ValueError):
        BasketHolding(ticker="AAPL", shares=0.0)


def test_cost_parameters_validation() -> None:
    with pytest.raises(ValueError):
        CostParameters(futures_band_points=-1.0, etf_cost_band_bps=1.0)
    with pytest.raises(ValueError):
        CostParameters(futures_band_points=1.0, etf_cost_band_bps=-1.0)
    cost = CostParameters(futures_band_points=0.5, etf_cost_band_bps=5.0)
    assert cost.etf_band_fraction == pytest.approx(5.0e-4)
