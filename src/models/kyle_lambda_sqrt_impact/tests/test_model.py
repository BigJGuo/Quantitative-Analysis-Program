"""Tests for the ``KyleSqrtImpact`` BaseModel subclass.

Uses ``InMemoryProvider`` so the integration test is the only one that
hits yfinance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.kyle_lambda_sqrt_impact import KyleSqrtImpact
from src.models.kyle_lambda_sqrt_impact.types import (
    CombinedImpactPrediction,
    KyleInputs,
    SqrtImpactPrediction,
)


def _make_provider(
    daily: pd.DataFrame, intraday: pd.DataFrame | None = None
) -> InMemoryProvider:
    prices = {("ABC", "120d", "1d"): daily}
    intraday_map = (
        {("ABC", "5m", 60): intraday} if intraday is not None else {}
    )
    return InMemoryProvider(prices=prices, intraday=intraday_map)


class TestConstruction:
    def test_defaults(self) -> None:
        m = KyleSqrtImpact("abc")
        assert m.ticker == "ABC"
        assert m.frequency == "daily"
        assert m.layer == 5
        assert m.refit_frequency == "weekly"
        assert m.name == "kyle_lambda_sqrt_impact"

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError):
            KyleSqrtImpact("")
        with pytest.raises(ValueError):
            KyleSqrtImpact("abc", side=2)
        with pytest.raises(ValueError):
            KyleSqrtImpact("abc", Y_prefactor=0)
        with pytest.raises(ValueError):
            KyleSqrtImpact("abc", parent_order_shares=-1)
        with pytest.raises(ValueError):
            KyleSqrtImpact("abc", daily_lookback=10)


class TestPipeline:
    def test_full_pipeline(self, synthetic_daily_bars: pd.DataFrame) -> None:
        provider = _make_provider(synthetic_daily_bars)
        model = KyleSqrtImpact(
            "abc",
            daily_lookback=180,
            parent_order_shares=1_000.0,
            Y_prefactor=1.0,
        )
        inputs = model.fetch_data(provider)
        assert isinstance(inputs, KyleInputs)
        assert inputs.ticker == "ABC"

        result = model.calibrate(inputs)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "kyle_lambda_sqrt_impact"

        risk = model.predict(inputs)
        assert isinstance(risk, RiskMetric)
        assert risk.ticker == "ABC"
        assert risk.metric_name == "expected_impact_bps"
        # 1k-share parent order on a 1M-share book: small ratio, linear regime
        # expected. impact_bps must be finite and positive.
        assert np.isfinite(risk.value)
        assert risk.value > 0.0
        assert risk.metadata["regime"] in ("linear", "sqrt")

    def test_validate_reports_spec_fields(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        provider = _make_provider(synthetic_daily_bars)
        model = KyleSqrtImpact("abc", daily_lookback=180)
        inputs = model.fetch_data(provider)
        model.calibrate(inputs)
        diag = model.validate(inputs)
        assert diag["ticker"] == "ABC"
        assert diag["frequency"] == "daily"
        assert diag["lambda_positive"] is True  # synthetic data has positive lambda
        assert 0.0 < diag["r_squared"] < 1.0
        for key in (
            "sigma_daily",
            "adv_window_volume",
            "sigma_over_volume",
            "lambda_normalized",
        ):
            value = diag[key]
            assert isinstance(value, float)
            assert np.isfinite(value)

    def test_predict_uses_combined_regime(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        provider = _make_provider(synthetic_daily_bars)
        # Tiny parent order -> linear regime; huge parent order -> sqrt.
        small = KyleSqrtImpact("abc", parent_order_shares=100.0, daily_lookback=180)
        big = KyleSqrtImpact("abc", parent_order_shares=5_000_000.0, daily_lookback=180)
        inputs_small = small.fetch_data(provider)
        inputs_big = big.fetch_data(provider)
        risk_small = small.predict(inputs_small)
        risk_big = big.predict(inputs_big)
        # Big order must in fact be on the sqrt side (huge participation).
        assert risk_big.metadata["regime"] == "sqrt"
        # Big order must report extrapolation warning (Q/V >> 0.10).
        assert risk_big.metadata["extrapolation_warning"] is True
        # Small order is below threshold.
        assert risk_small.metadata["extrapolation_warning"] is False

    def test_forecast_sqrt_helper(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        provider = _make_provider(synthetic_daily_bars)
        model = KyleSqrtImpact(
            "abc", parent_order_shares=10_000.0, daily_lookback=180
        )
        inputs = model.fetch_data(provider)
        pred = model.forecast_sqrt(inputs)
        assert isinstance(pred, SqrtImpactPrediction)
        # Sqrt impact in bps is always >= 0 (we take side * relative).
        assert pred.impact_bps > 0

    def test_forecast_combined_helper(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        provider = _make_provider(synthetic_daily_bars)
        model = KyleSqrtImpact(
            "abc", parent_order_shares=10_000.0, daily_lookback=180
        )
        inputs = model.fetch_data(provider)
        # Cannot call forecast_combined before fitting.
        with pytest.raises(RuntimeError):
            _ = model.fit
        model.calibrate(inputs)
        pred = model.forecast_combined(inputs)
        assert isinstance(pred, CombinedImpactPrediction)
        assert pred.regime in ("linear", "sqrt")
        assert np.isfinite(pred.Q_crossover) or pred.Q_crossover == float("inf")

    def test_intraday_pipeline(
        self,
        synthetic_daily_bars: pd.DataFrame,
        synthetic_intraday_bars: pd.DataFrame,
    ) -> None:
        provider = _make_provider(synthetic_daily_bars, synthetic_intraday_bars)
        model = KyleSqrtImpact(
            "abc",
            frequency="intraday",
            daily_lookback=180,
            intraday_lookback_days=60,
            parent_order_shares=10_000.0,
        )
        inputs = model.fetch_data(provider)
        assert inputs.intraday_bars is not None
        result = model.calibrate(inputs)
        assert result.metadata["frequency"] == "intraday"
        diag = model.validate(inputs)
        assert diag["frequency"] == "intraday"


class TestDiagnostic:
    """Spec validation section: lambda sign, R^2, sigma/V monotonicity."""

    def test_lambda_sign_and_r2_in_band(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        provider = _make_provider(synthetic_daily_bars)
        model = KyleSqrtImpact("abc", daily_lookback=180)
        inputs = model.fetch_data(provider)
        model.calibrate(inputs)
        diag = model.validate(inputs)
        # Spec: lambda > 0 in healthy markets; negative = stale proxy.
        assert diag["lambda_positive"] is True
        # Spec: R^2 in [0.05, 0.30] for liquid US equities at daily freq.
        # On synthetic data with engineered ~95% signal share, R^2 will be
        # high — assert merely positive here.
        assert diag["r_squared"] > 0.0
