"""Unit tests for the `HARRVModel` orchestration class.

Uses `InMemoryProvider` so the full `fetch_data -> calibrate -> predict ->
validate` pipeline runs without touching yfinance. Synthetic OHLC fixtures
give the regression enough history to fit and the predictions/validations
enough samples to be non-trivial.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import CalibrationResult, Forecast
from src.models.har_rv.model import HARRVModel
from src.models.har_rv.types import HARRVInputs, RVComponents


def _make_synthetic_daily_ohlc(
    n_days: int = 600, seed: int = 11
) -> pd.DataFrame:
    """OHLC bars with random but realistic intraday range / drift."""

    rng = np.random.default_rng(seed)
    log_close = np.cumsum(rng.normal(0.0005, 0.012, n_days))
    closes = 100.0 * np.exp(log_close)
    opens = closes / np.exp(rng.normal(0.0, 0.005, n_days))
    ranges = np.abs(rng.normal(0.0, 0.015, n_days))
    highs = np.maximum(closes, opens) * np.exp(ranges)
    lows = np.minimum(closes, opens) / np.exp(np.abs(rng.normal(0.0, 0.015, n_days)))
    index = pd.date_range("2022-01-03", periods=n_days, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=index,
    ).reset_index().rename(columns={"index": "Date"})


def _make_synthetic_intraday(
    n_days: int = 10, bars_per_day: int = 78, seed: int = 13
) -> pd.DataFrame:
    """Random walk 5-minute bars over `n_days` US trading sessions."""

    rng = np.random.default_rng(seed)
    rows: list[tuple[pd.Timestamp, float]] = []
    price = 100.0
    base = pd.Timestamp("2026-03-02 09:30")
    for d in range(n_days):
        day_start = base + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=5 * b)
            price *= np.exp(rng.normal(0.0, 0.0015))
            rows.append((ts, price))
    df = pd.DataFrame(rows, columns=["Datetime", "Close"])
    df["Open"] = df["Close"]
    df["High"] = df["Close"]
    df["Low"] = df["Close"]
    return df


@pytest.fixture
def provider_with_daily_only() -> InMemoryProvider:
    """Provider returning empty intraday + a multi-year daily OHLC frame."""

    return InMemoryProvider(
        intraday={("SPY", "5m", 60): pd.DataFrame()},
        prices={("SPY", "5y", "1d"): _make_synthetic_daily_ohlc(n_days=600)},
    )


@pytest.fixture
def provider_full() -> InMemoryProvider:
    """Provider with both intraday + multi-year daily OHLC."""

    return InMemoryProvider(
        intraday={("SPY", "5m", 60): _make_synthetic_intraday(n_days=10)},
        prices={("SPY", "5y", "1d"): _make_synthetic_daily_ohlc(n_days=600)},
    )


class TestHARRVModelConstruction:
    def test_class_attributes(self) -> None:
        assert HARRVModel.name == "har_rv"
        assert HARRVModel.layer == 4
        assert HARRVModel.refit_frequency == "monthly"

    def test_rejects_empty_ticker(self) -> None:
        with pytest.raises(ValueError, match="ticker"):
            HARRVModel(ticker="")

    def test_rejects_invalid_spec(self) -> None:
        with pytest.raises(ValueError, match="spec"):
            HARRVModel(ticker="SPY", spec="exponential")  # type: ignore[arg-type]

    def test_rejects_invalid_windows(self) -> None:
        with pytest.raises(ValueError, match="w_window"):
            HARRVModel(ticker="SPY", w_window=10, m_window=5)


class TestHARRVModelEndToEnd:
    def test_daily_fallback_pipeline(
        self, provider_with_daily_only: InMemoryProvider
    ) -> None:
        model = HARRVModel(ticker="SPY")
        inputs = model.fetch_data(provider_with_daily_only)
        assert isinstance(inputs, HARRVInputs)
        assert isinstance(inputs.rv_components, RVComponents)
        assert inputs.rv_components.source == "garman_klass"
        assert len(inputs.rv_components.rv_d) > 100

        result = model.calibrate(inputs)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "har_rv"
        assert result.fit_metrics["n_obs"] > 100

        forecast = model.predict(inputs)
        assert isinstance(forecast, Forecast)
        assert forecast.ticker == "SPY"
        assert forecast.value > 0
        assert "rv_forecast_daily" in forecast.metadata
        assert forecast.horizon.endswith("annualized")

    def test_intraday_plus_daily_splice(self, provider_full: InMemoryProvider) -> None:
        model = HARRVModel(ticker="SPY")
        inputs = model.fetch_data(provider_full)
        # Intraday dates should be present at the tail.
        assert inputs.metadata["intraday_days"] > 0
        # The combined RV series must be at least as long as the daily series.
        assert len(inputs.rv_components.rv_d) > 100
        # Intraday data should populate bipower over the intraday window.
        assert inputs.rv_components.bipower is not None
        intraday_dates = inputs.rv_components.bipower.dropna()
        # Bipower should be nonzero for days that have intraday bars.
        assert (intraday_dates > 0).any()

    def test_validate_returns_expected_keys(
        self, provider_with_daily_only: InMemoryProvider
    ) -> None:
        model = HARRVModel(ticker="SPY")
        inputs = model.fetch_data(provider_with_daily_only)
        model.calibrate(inputs)
        diag = model.validate(inputs)
        for key in (
            "n_obs",
            "r_squared",
            "residual_variance",
            "coefficient_persistence",
            "coefficient_standard_errors",
            "mincer_zarnowitz_a",
            "mincer_zarnowitz_b",
            "qlike_loss_in_sample",
            "ljung_box_residuals_pvalue",
            "arch_lm_residuals_pvalue",
        ):
            assert key in diag, f"Missing diagnostic key: {key}"
        assert diag["n_obs"] > 100
        assert 0.0 <= diag["r_squared"] <= 1.0
        assert diag["qlike_loss_in_sample"] >= 0.0

    def test_level_spec_pipeline(
        self, provider_with_daily_only: InMemoryProvider
    ) -> None:
        model = HARRVModel(ticker="SPY", spec="level")
        inputs = model.fetch_data(provider_with_daily_only)
        result = model.calibrate(inputs)
        assert result.metadata["spec"] == "level"
        forecast = model.predict(inputs)
        assert forecast.value > 0

    def test_horizon_h_uses_h_period_target(
        self, provider_with_daily_only: InMemoryProvider
    ) -> None:
        model = HARRVModel(ticker="SPY", horizon=5)
        inputs = model.fetch_data(provider_with_daily_only)
        result = model.calibrate(inputs)
        assert result.metadata["horizon"] == 5
        assert result.parameters["horizon"] == 5

    def test_predict_without_calibrate_raises(
        self, provider_with_daily_only: InMemoryProvider
    ) -> None:
        model = HARRVModel(ticker="SPY")
        inputs = model.fetch_data(provider_with_daily_only)
        with pytest.raises(RuntimeError, match="has not been calibrated"):
            model.predict(inputs)

    def test_input_type_mismatch_raises(self) -> None:
        model = HARRVModel(ticker="SPY")
        with pytest.raises(TypeError, match="HARRVInputs"):
            model.calibrate("nope")
        with pytest.raises(TypeError, match="HARRVInputs"):
            model.predict(42)

    def test_no_data_raises(self) -> None:
        """When both intraday and daily come back empty, fetch_data fails."""

        empty_provider = InMemoryProvider(
            intraday={("SPY", "5m", 60): pd.DataFrame()},
            prices={("SPY", "5y", "1d"): pd.DataFrame()},
        )
        model = HARRVModel(ticker="SPY")
        with pytest.raises(RuntimeError, match="no usable RV history"):
            model.fetch_data(empty_provider)

    def test_inputs_timestamp_is_utc(
        self, provider_with_daily_only: InMemoryProvider
    ) -> None:
        model = HARRVModel(ticker="SPY")
        inputs = model.fetch_data(provider_with_daily_only)
        assert isinstance(inputs.timestamp, datetime)
        assert inputs.timestamp.tzinfo == UTC
