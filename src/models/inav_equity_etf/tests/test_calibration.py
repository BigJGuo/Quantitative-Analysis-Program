"""Tests for the calibration helper that refreshes seeds and the cost band."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from src.core.types import CalibrationResult
from src.models.inav_equity_etf.calibration import MODEL_NAME, calibrate
from src.models.inav_equity_etf.types import (
    BasketHolding,
    CashLedgerSeed,
    CostParameters,
)


def _seed() -> CashLedgerSeed:
    return CashLedgerSeed(
        eod_nav=100.0,
        eod_shares_out=10_000.0,
        eod_release_time=datetime(2026, 5, 17, 20, 0),
        prior_cash=10_000.0,
        prior_liabilities=50.0,
    )


def _basket() -> tuple[BasketHolding, ...]:
    return (
        BasketHolding(ticker="AAPL", shares=1000.0, currency="USD"),
        BasketHolding(ticker="SAP.DE", shares=200.0, currency="EUR"),
    )


class TestCalibrate:
    def test_returns_calibration_result_with_expected_shape(self) -> None:
        result = calibrate(basket=_basket(), seed=_seed())
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        assert "basket" in result.parameters
        assert "seed" in result.parameters
        assert "cost_params" in result.parameters
        assert isinstance(result.parameters["cost_params"], CostParameters)

    def test_default_cost_band_used_when_no_history(self) -> None:
        result = calibrate(
            basket=_basket(),
            seed=_seed(),
            default_kappa_bps=1.5,
            default_tau_bps=3.5,
        )
        cost: CostParameters = result.parameters["cost_params"]
        assert cost.kappa_bps == pytest.approx(1.5)
        assert cost.tau_bps == pytest.approx(3.5)
        assert result.fit_metrics["cost_band_bps"] == pytest.approx(5.0)

    def test_short_history_falls_back_to_defaults(self) -> None:
        result = calibrate(
            basket=_basket(),
            seed=_seed(),
            realized_premia=[0.001] * 5,
            default_kappa_bps=1.0,
            default_tau_bps=2.0,
        )
        cost: CostParameters = result.parameters["cost_params"]
        assert cost.kappa_bps == pytest.approx(1.0)
        assert cost.tau_bps == pytest.approx(2.0)

    def test_quantile_of_long_history_drives_band(self) -> None:
        # 100 premia: 50 at +5 bps, 50 at +25 bps; 95th percentile |p| = 25 bps.
        history = [0.0005] * 50 + [0.0025] * 50
        result = calibrate(
            basket=_basket(),
            seed=_seed(),
            realized_premia=history,
            default_kappa_bps=1.0,
            default_tau_bps=2.0,
        )
        cost: CostParameters = result.parameters["cost_params"]
        assert cost.kappa_bps == pytest.approx(1.0)
        # Empirical 95th quantile sits at 25 bps, minus 1 bp kappa = 24 bps tau.
        assert math.isclose(cost.tau_bps, 24.0, abs_tol=0.5)

    def test_lower_quantile_yields_tighter_band(self) -> None:
        history = [0.0005] * 50 + [0.0025] * 50
        loose = calibrate(
            basket=_basket(), seed=_seed(), realized_premia=history, cost_quantile=0.95
        )
        tight = calibrate(
            basket=_basket(), seed=_seed(), realized_premia=history, cost_quantile=0.50
        )
        loose_cost: CostParameters = loose.parameters["cost_params"]
        tight_cost: CostParameters = tight.parameters["cost_params"]
        assert tight_cost.tau_bps < loose_cost.tau_bps

    def test_empty_basket_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one BasketHolding"):
            calibrate(basket=[], seed=_seed())

    def test_invalid_seed_rejected(self) -> None:
        bad_seed = CashLedgerSeed(
            eod_nav=-1.0,
            eod_shares_out=10.0,
            eod_release_time=datetime(2026, 5, 17),
            prior_cash=0.0,
            prior_liabilities=0.0,
        )
        with pytest.raises(ValueError, match="eod_nav"):
            calibrate(basket=_basket(), seed=bad_seed)

    def test_quantile_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="cost_quantile"):
            calibrate(basket=_basket(), seed=_seed(), cost_quantile=1.5)
