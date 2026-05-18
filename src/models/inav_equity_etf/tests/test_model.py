"""Integration tests for the `INAVEquityETF` class against the in-memory provider."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, Signal
from src.models.inav_equity_etf.model import INAVEquityETF
from src.models.inav_equity_etf.types import (
    BasketHolding,
    CashLedgerSeed,
    INAVInputs,
)


def make_bars(*, timestamp: datetime, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Datetime": [pd.Timestamp(timestamp)],
            "Open": [close],
            "High": [close * 1.001],
            "Low": [close * 0.999],
            "Close": [close],
            "Volume": [1000],
        }
    )


def _make_model(
    now: datetime,
    basket: tuple[BasketHolding, ...],
    seed: CashLedgerSeed,
    **kwargs: Any,
) -> INAVEquityETF:
    return INAVEquityETF(
        etf_ticker="ETFX",
        holdings=basket,
        seed=seed,
        now_func=lambda: now,
        **kwargs,
    )


class TestFetchData:
    def test_fetch_data_returns_inputs_with_all_quotes(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        assert isinstance(data, INAVInputs)
        assert set(data.quotes.keys()) == {"AAPL", "SAP.DE"}
        assert "EUR" in data.fx_rates
        assert data.short_rate_annual == pytest.approx(0.0525)
        assert data.expense_ratio == pytest.approx(0.0009)
        assert data.etf_mid == pytest.approx(23.50, rel=1e-3)

    def test_fetch_data_handles_missing_intraday(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        # Drop AAPL bars so it has no quote.
        provider.intraday.pop(("AAPL", "1m", 1))
        provider.intraday[("AAPL", "1m", 1)] = pd.DataFrame()
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        assert "AAPL" not in data.quotes
        assert "SAP.DE" in data.quotes


class TestPredict:
    def test_predict_returns_signal_with_metadata(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        signal = model.predict(data)
        assert isinstance(signal, Signal)
        assert signal.ticker == "ETFX"
        assert signal.direction in {"long", "short", "flat"}
        assert 0.0 <= signal.strength <= 1.0
        assert "inav" in signal.metadata
        assert "premium" in signal.metadata

    def test_predict_signs_create(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        # Spike ETF mid well above iNAV -> CREATE -> short ETF.
        provider.intraday[("ETFX", "1m", 1)] = make_bars(timestamp=now, close=1_000.0)
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        signal = model.predict(data)
        assert signal.metadata["action"] == "CREATE"
        assert signal.direction == "short"
        assert signal.strength > 0.0

    def test_predict_signs_redeem(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        provider.intraday[("ETFX", "1m", 1)] = make_bars(timestamp=now, close=0.50)
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        signal = model.predict(data)
        assert signal.metadata["action"] == "REDEEM"
        assert signal.direction == "long"

    def test_predict_rejects_wrong_input_type(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
    ) -> None:
        model = _make_model(now, basket, seed)
        with pytest.raises(TypeError, match="INAVInputs"):
            model.predict({"not": "inputs"})


class TestSnapshotAndValidate:
    def test_snapshot_exposes_full_decomposition(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        snap = model.snapshot(data)
        assert snap.decomposition.basket_live > 0
        assert snap.decomposition.inav > 0
        assert snap.n_live == 2

    def test_validate_reports_coverage_and_residual(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        diag = model.validate(data)
        assert diag["coverage"] == pytest.approx(1.0)
        assert diag["coverage_ok"] is True
        assert "premium_bps" in diag
        assert "eod_residual_bps" in diag

    def test_validate_flags_low_coverage_when_constituent_missing(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        provider.intraday[("AAPL", "1m", 1)] = pd.DataFrame()
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        diag = model.validate(data)
        assert diag["coverage"] == pytest.approx(0.5)
        assert diag["coverage_ok"] is False


class TestCalibrationAndRegistry:
    def test_calibrate_updates_cost_params_from_realized_premia(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
    ) -> None:
        model = _make_model(now, basket, seed)
        # Inject 100 historical premia averaging |p|=15 bps.
        model.recent_premia = [0.0015 if i % 2 == 0 else -0.0015 for i in range(100)]
        result: CalibrationResult = model.calibrate(None)
        # Band ~ 15 bps; minus 1 bp issuer fee gives tau ~ 14 bps.
        assert result.fit_metrics["cost_band_bps"] == pytest.approx(15.0, abs=0.5)
        assert model.cost_params.tau_bps == pytest.approx(14.0, abs=0.5)

    def test_calibrate_falls_back_to_defaults_with_short_history(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
    ) -> None:
        model = _make_model(now, basket, seed)
        model.recent_premia = [0.01]  # too few samples
        result = model.calibrate(None)
        assert result.fit_metrics["kappa_bps"] == pytest.approx(1.0)
        assert result.fit_metrics["tau_bps"] == pytest.approx(2.0)

    def test_model_is_registered_in_global_registry(self) -> None:
        cls = get_model("inav_equity_etf")
        assert cls is INAVEquityETF
        assert cls.layer == 2
        assert cls.refit_frequency == "intraday"


class TestStaleConstituents:
    def test_intl_constituent_goes_to_stale_leg_after_threshold(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        old_ts = now - timedelta(hours=10)
        provider.intraday[("SAP.DE", "1m", 1)] = make_bars(timestamp=old_ts, close=150.0)
        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        snap = model.snapshot(data)
        assert snap.n_live == 1
        assert snap.n_stale == 1
        assert snap.decomposition.basket_stale > 0


class TestDiagnosticReconciliation:
    """The spec's `Validation and diagnostics` section requires an EOD residual
    check (point #1) and a constituent-coverage check (point #2). This test
    exercises both together against a clean fixture and confirms that the
    residual is within the spec's 5-bp tolerance for international ETFs."""

    def test_eod_reconciliation_within_spec_tolerance(
        self,
        now: datetime,
        basket: tuple[BasketHolding, ...],
        seed: CashLedgerSeed,
        provider: InMemoryProvider,
    ) -> None:
        # Tune the basket so iNAV ≈ eod_nav, then check residual is small.
        per_share_nav = seed.eod_nav  # target 100.0
        target_total = per_share_nav * seed.eod_shares_out
        # Approximate: pick a basket value close to target_total minus cash plus liab.
        elapsed = (now - seed.eod_release_time).total_seconds()
        day_frac = elapsed / 86_400.0
        expected_cash = seed.prior_cash * (1.0 + 0.0525 * day_frac / 360.0)
        expected_liab = (
            seed.prior_liabilities
            + (0.0009 / 252.0) * seed.eod_nav * seed.eod_shares_out * day_frac
        )
        required_basket_usd = target_total - expected_cash + expected_liab
        # Distribute equally across the two constituents at known prices/FX.
        # AAPL USD share + SAP.DE EUR share at FX=1.10.
        sap_value = required_basket_usd * 0.4
        aapl_value = required_basket_usd * 0.6
        new_aapl_price = aapl_value / basket[0].shares
        new_sap_price = sap_value / (basket[1].shares * 1.10)
        provider.intraday[("AAPL", "1m", 1)] = make_bars(
            timestamp=now - timedelta(seconds=30), close=new_aapl_price
        )
        provider.intraday[("SAP.DE", "1m", 1)] = make_bars(
            timestamp=now - timedelta(seconds=30), close=new_sap_price
        )

        model = _make_model(now, basket, seed)
        data = model.fetch_data(provider)
        diag = model.validate(data)
        # EOD residual should now be near zero (well within 5 bps).
        assert abs(diag["eod_residual_bps"]) < 5.0
        assert diag["coverage_ok"] is True
