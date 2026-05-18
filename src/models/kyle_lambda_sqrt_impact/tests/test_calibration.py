"""Calibration-layer tests: spec Algorithms A and B against synthetic bars."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.kyle_lambda_sqrt_impact.calibration import (
    MODEL_NAME,
    calibrate,
    calibrate_daily,
    calibrate_intraday,
    estimate_intraday_pooled,
)
from src.models.kyle_lambda_sqrt_impact.types import KyleLambdaFit


class TestCalibrateDaily:
    def test_recovers_lambda_within_ci(
        self, synthetic_daily_bars: pd.DataFrame, true_lambda: float
    ) -> None:
        fit = calibrate_daily(synthetic_daily_bars)
        # Truth should lie inside the Wald CI (95%) with reasonably high
        # probability; the synthetic series is designed for it.
        assert fit.ci_low <= true_lambda <= fit.ci_high

    def test_fit_is_positive_and_well_specified(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        fit = calibrate_daily(synthetic_daily_bars)
        assert fit.lambda_hat > 0.0
        assert fit.se > 0.0
        assert fit.r_squared > 0.0
        assert fit.frequency == "daily"
        assert fit.n_obs > 100

    def test_lookback_trims_history(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        fit = calibrate_daily(synthetic_daily_bars, lookback=60)
        # 60-day window => up to 60 observations after dropping the first NaN.
        assert fit.n_obs <= 60

    def test_dollar_signed_changes_units(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        share_fit = calibrate_daily(synthetic_daily_bars, dollar=False)
        dollar_fit = calibrate_daily(synthetic_daily_bars, dollar=True)
        # Lambda in dollar units is much smaller (per-dollar instead of
        # per-share). On a $100 stock, dollar lambda should be roughly
        # share-lambda / price.
        assert dollar_fit.lambda_hat < share_fit.lambda_hat
        assert dollar_fit.mean_volume > share_fit.mean_volume


class TestCalibrateIntraday:
    def test_pooled_and_per_day_returned(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        pooled, per_day = calibrate_intraday(synthetic_intraday_bars)
        assert isinstance(pooled, KyleLambdaFit)
        assert pooled.frequency == "intraday"
        assert len(per_day) == 2  # two trading days in the fixture
        # All per-day lambdas should be positive by fixture construction.
        assert all(v > 0 for v in per_day.values())

    def test_helper_returns_pooled(
        self, synthetic_intraday_bars: pd.DataFrame
    ) -> None:
        pooled = estimate_intraday_pooled(synthetic_intraday_bars)
        assert pooled.frequency == "intraday"
        assert pooled.n_obs > 100

    def test_intraday_requires_data(self) -> None:
        with pytest.raises(ValueError):
            calibrate_intraday(
                pd.DataFrame(
                    {"Open": [], "Close": [], "Volume": [], "Datetime": []}
                )
            )


class TestCalibrateDispatcher:
    def test_daily_dispatch(self, synthetic_daily_bars: pd.DataFrame) -> None:
        res = calibrate(synthetic_daily_bars, frequency="daily")
        assert isinstance(res, CalibrationResult)
        assert res.model_name == MODEL_NAME
        assert "fit" in res.parameters
        assert isinstance(res.parameters["fit"], KyleLambdaFit)
        assert res.fit_metrics["n_obs"] > 0

    def test_intraday_dispatch(
        self,
        synthetic_daily_bars: pd.DataFrame,
        synthetic_intraday_bars: pd.DataFrame,
    ) -> None:
        res = calibrate(
            synthetic_daily_bars,
            frequency="intraday",
            intraday_bars=synthetic_intraday_bars,
        )
        fit = res.parameters["fit"]
        assert isinstance(fit, KyleLambdaFit)
        assert fit.frequency == "intraday"
        # Per-day diagnostics should be in metadata.
        assert "per_day_median_lambda" in res.metadata
        assert res.metadata["per_day_count"] == 2

    def test_intraday_without_bars_raises(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            calibrate(synthetic_daily_bars, frequency="intraday")

    def test_unknown_frequency_raises(
        self, synthetic_daily_bars: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            calibrate(synthetic_daily_bars, frequency="weekly")  # type: ignore[arg-type]
