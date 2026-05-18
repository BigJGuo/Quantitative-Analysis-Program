"""Unit tests for the pure-math functions in `signal.py`.

The two-constituent fixture has a closed-form iNAV the assertions check
against, so any drift in the accrual or aggregation logic shows up here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.models.inav_equity_etf.signal import (
    accrue_cash,
    accrue_liabilities,
    action_to_signal_direction,
    basket_value_usd,
    classify_action,
    compute_decomposition,
    compute_inav_result,
    compute_premium,
    is_quote_stale,
    signal_strength,
)
from src.models.inav_equity_etf.types import (
    BasketHolding,
    CashLedgerSeed,
    ConstituentQuote,
    CostParameters,
    INAVInputs,
)


def _build_inputs(
    *,
    now: datetime,
    basket: tuple[BasketHolding, ...],
    seed: CashLedgerSeed,
    quotes: dict[str, ConstituentQuote],
    fx_rates: dict[str, float],
    etf_mid: float = 23.50,
    short_rate: float = 0.0525,
    expense_ratio: float = 0.0009,
    cost: CostParameters | None = None,
) -> INAVInputs:
    return INAVInputs(
        etf_ticker="ETFX",
        timestamp=now,
        basket=basket,
        quotes=quotes,
        fx_rates=fx_rates,
        dividends_since_eod={},
        short_rate_annual=short_rate,
        expense_ratio=expense_ratio,
        seed=seed,
        etf_mid=etf_mid,
        staleness_threshold_seconds=300.0,
        cost_params=cost or CostParameters(kappa_bps=1.0, tau_bps=2.0),
    )


class TestStaleness:
    def test_fresh_quote_is_live(self, now: datetime) -> None:
        q = ConstituentQuote("AAPL", 200.0, now - timedelta(seconds=10))
        assert is_quote_stale(q, now, 300.0) is False

    def test_old_quote_is_stale(self, now: datetime) -> None:
        q = ConstituentQuote("AAPL", 200.0, now - timedelta(seconds=600))
        assert is_quote_stale(q, now, 300.0) is True

    def test_explicit_flag_wins(self, now: datetime) -> None:
        q = ConstituentQuote("AAPL", 200.0, now, is_stale=True)
        assert is_quote_stale(q, now, 300.0) is True


class TestAccruals:
    def test_cash_with_zero_elapsed_returns_seed(self) -> None:
        assert accrue_cash(
            prior_cash=50_000.0,
            short_rate_annual=0.05,
            elapsed_seconds=0.0,
            dividends_received=0.0,
        ) == pytest.approx(50_000.0)

    def test_cash_interest_accrual_matches_spec_formula(self) -> None:
        # One full day at 5% ACT/360 -> 0.05/360 * 50_000.
        out = accrue_cash(
            prior_cash=50_000.0,
            short_rate_annual=0.05,
            elapsed_seconds=86_400.0,
            dividends_received=0.0,
        )
        expected = 50_000.0 * (1.0 + 0.05 / 360.0)
        assert out == pytest.approx(expected)

    def test_cash_dividends_add_linearly(self) -> None:
        out = accrue_cash(
            prior_cash=0.0,
            short_rate_annual=0.0,
            elapsed_seconds=0.0,
            dividends_received=123.45,
        )
        assert out == pytest.approx(123.45)

    def test_liabilities_accrual_pro_rated(self) -> None:
        # Half a day of 0.09 bps/yr expense on 100 * 10_000 NAV*shares.
        out = accrue_liabilities(
            prior_liabilities=200.0,
            expense_ratio_annual=0.0009,
            eod_nav=100.0,
            eod_shares_out=10_000.0,
            elapsed_seconds=43_200.0,
        )
        daily = (0.0009 / 252.0) * 100.0 * 10_000.0
        expected = 200.0 + daily * 0.5
        assert out == pytest.approx(expected)


class TestBasketValue:
    def test_live_partition_sums_correctly(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        quotes: dict[str, ConstituentQuote],
        fx_rates: dict[str, float],
    ) -> None:
        live_value, count = basket_value_usd(
            basket, quotes, fx_rates, now=now,
            staleness_threshold_seconds=300.0, live=True,
        )
        expected = 1000.0 * 200.0 + 200.0 * 150.0 * 1.10
        assert live_value == pytest.approx(expected)
        assert count == 2

    def test_stale_constituents_split_out(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        fx_rates: dict[str, float],
    ) -> None:
        stale_quotes: dict[str, ConstituentQuote] = {
            "AAPL": ConstituentQuote("AAPL", 200.0, now - timedelta(seconds=30)),
            "SAP.DE": ConstituentQuote(
                "SAP.DE", 150.0, now - timedelta(seconds=999), is_stale=True
            ),
        }
        live_value, n_live = basket_value_usd(
            basket, stale_quotes, fx_rates, now=now,
            staleness_threshold_seconds=300.0, live=True,
        )
        stale_value, n_stale = basket_value_usd(
            basket, stale_quotes, fx_rates, now=now,
            staleness_threshold_seconds=300.0, live=False,
        )
        assert n_live == 1 and n_stale == 1
        assert live_value == pytest.approx(1000.0 * 200.0)
        assert stale_value == pytest.approx(200.0 * 150.0 * 1.10)

    def test_missing_quote_silently_skipped(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        fx_rates: dict[str, float],
    ) -> None:
        partial: dict[str, ConstituentQuote] = {
            "AAPL": ConstituentQuote("AAPL", 200.0, now)
        }
        live_value, count = basket_value_usd(
            basket, partial, fx_rates, now=now,
            staleness_threshold_seconds=300.0, live=True,
        )
        assert count == 1
        assert live_value == pytest.approx(200_000.0)


class TestDecompositionAndPremium:
    def test_decomposition_matches_hand_calc(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        quotes: dict[str, ConstituentQuote],
        fx_rates: dict[str, float],
    ) -> None:
        inputs = _build_inputs(
            now=now, basket=basket, seed=seed, quotes=quotes, fx_rates=fx_rates
        )
        decomp = compute_decomposition(inputs)

        basket_live_expected = 1000.0 * 200.0 + 200.0 * 150.0 * 1.10
        assert decomp.basket_live == pytest.approx(basket_live_expected)
        assert decomp.basket_stale == pytest.approx(0.0)

        elapsed = (now - seed.eod_release_time).total_seconds()
        day_frac = elapsed / 86_400.0
        expected_cash = seed.prior_cash * (1.0 + 0.0525 * day_frac / 360.0)
        expected_liab = (
            seed.prior_liabilities
            + (0.0009 / 252.0) * seed.eod_nav * seed.eod_shares_out * day_frac
        )
        expected_inav = (
            basket_live_expected + expected_cash - expected_liab
        ) / seed.eod_shares_out

        assert decomp.cash == pytest.approx(expected_cash)
        assert decomp.liabilities == pytest.approx(expected_liab)
        assert decomp.inav == pytest.approx(expected_inav)

    def test_premium_zero_when_mid_equals_inav(self) -> None:
        assert compute_premium(100.0, 100.0) == pytest.approx(0.0)

    def test_premium_sign(self) -> None:
        assert compute_premium(101.0, 100.0) > 0
        assert compute_premium(99.0, 100.0) < 0

    def test_premium_zero_inav_rejected(self) -> None:
        with pytest.raises(ValueError, match="iNAV must be positive"):
            compute_premium(100.0, 0.0)


class TestActionClassification:
    def test_inside_band_is_flat(self) -> None:
        cost = CostParameters(kappa_bps=1.0, tau_bps=2.0)  # 3 bps band
        assert classify_action(0.0001, cost) == "FLAT"
        assert classify_action(-0.0001, cost) == "FLAT"

    def test_above_band_is_create(self) -> None:
        cost = CostParameters(kappa_bps=1.0, tau_bps=2.0)
        assert classify_action(0.01, cost) == "CREATE"

    def test_below_band_is_redeem(self) -> None:
        cost = CostParameters(kappa_bps=1.0, tau_bps=2.0)
        assert classify_action(-0.01, cost) == "REDEEM"

    def test_action_direction_mapping(self) -> None:
        assert action_to_signal_direction("CREATE") == "short"
        assert action_to_signal_direction("REDEEM") == "long"
        assert action_to_signal_direction("FLAT") == "flat"
        with pytest.raises(ValueError):
            action_to_signal_direction("WIBBLE")

    def test_signal_strength_clipped_to_unit_interval(self) -> None:
        cost = CostParameters(kappa_bps=1.0, tau_bps=2.0)
        assert signal_strength(0.0, cost) == 0.0
        assert signal_strength(1.0, cost) == 1.0
        s = signal_strength(0.0010, cost)  # 10 bps premium, 25 bps scale
        assert 0.0 < s < 1.0


class TestComputeINAVResult:
    def test_end_to_end_pure_pipeline_with_create_signal(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        quotes: dict[str, ConstituentQuote],
        fx_rates: dict[str, float],
    ) -> None:
        # Mid set well above iNAV (~22) to force CREATE.
        inputs = _build_inputs(
            now=now, basket=basket, seed=seed,
            quotes=quotes, fx_rates=fx_rates, etf_mid=30.0,
        )
        result = compute_inav_result(inputs)
        assert result.action == "CREATE"
        assert result.premium > 0
        assert result.n_live == 2
        assert result.n_stale == 0
        assert result.stale_fraction == pytest.approx(0.0)
        assert result.decomposition.inav > 0

    def test_stale_fraction_reflects_mix(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        fx_rates: dict[str, float],
    ) -> None:
        quotes = {
            "AAPL": ConstituentQuote("AAPL", 200.0, now - timedelta(seconds=30)),
            "SAP.DE": ConstituentQuote(
                "SAP.DE", 150.0, now - timedelta(seconds=30), is_stale=True
            ),
        }
        inputs = _build_inputs(
            now=now, basket=basket, seed=seed, quotes=quotes, fx_rates=fx_rates
        )
        result = compute_inav_result(inputs)
        assert result.n_live == 1
        assert result.n_stale == 1
        assert 0.0 < result.stale_fraction < 1.0
