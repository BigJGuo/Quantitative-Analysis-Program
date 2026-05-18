"""Validation tests for the model-internal dataclasses."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.models.inav_equity_etf.types import (
    BasketHolding,
    ConstituentQuote,
    CostParameters,
    INAVDecomposition,
)


class TestBasketHolding:
    def test_negative_shares_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BasketHolding(ticker="AAPL", shares=-1.0, currency="USD")

    def test_empty_currency_rejected(self) -> None:
        with pytest.raises(ValueError, match="currency"):
            BasketHolding(ticker="AAPL", shares=1.0, currency="")


class TestConstituentQuote:
    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ConstituentQuote(ticker="AAPL", price=-1.0, timestamp=datetime(2026, 1, 1))


class TestINAVDecomposition:
    def test_zero_shares_rejected(self) -> None:
        with pytest.raises(ValueError, match="shares_outstanding"):
            INAVDecomposition(
                basket_live=1.0, basket_stale=0.0, cash=0.0, liabilities=0.0,
                shares_outstanding=0.0, inav=0.0,
            )


class TestCostParameters:
    def test_negative_bps_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CostParameters(kappa_bps=-1.0, tau_bps=0.0)

    def test_total_band_in_fractional_units(self) -> None:
        cost = CostParameters(kappa_bps=1.0, tau_bps=2.0)
        assert cost.total_band == pytest.approx(3e-4)
