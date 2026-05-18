"""Validation tests for the model-internal dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.models.kyle_lambda_sqrt_impact.types import (
    KyleInputs,
    KyleLambdaFit,
)


def _minimal_bars() -> pd.DataFrame:
    n = 30
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.5] * n,
            "Volume": [1_000.0] * n,
        }
    )


class TestKyleLambdaFit:
    def test_valid_construction(self) -> None:
        fit = KyleLambdaFit(
            lambda_hat=1e-9,
            se=2e-10,
            ci_low=0.0,
            ci_high=2e-9,
            r_squared=0.1,
            n_obs=100,
            frequency="daily",
            mean_signed_volume_abs=1.0e6,
            mean_volume=1.0e6,
        )
        assert fit.lambda_normalized == pytest.approx(1.0e-3)

    def test_invalid_frequency(self) -> None:
        with pytest.raises(ValueError):
            KyleLambdaFit(
                lambda_hat=1e-9,
                se=0.0,
                ci_low=0.0,
                ci_high=0.0,
                r_squared=0.0,
                n_obs=10,
                frequency="weekly",  # type: ignore[arg-type]
                mean_signed_volume_abs=0.0,
                mean_volume=0.0,
            )

    def test_ci_ordering(self) -> None:
        with pytest.raises(ValueError):
            KyleLambdaFit(
                lambda_hat=1.0,
                se=0.0,
                ci_low=5.0,
                ci_high=1.0,
                r_squared=0.0,
                n_obs=10,
                frequency="daily",
                mean_signed_volume_abs=0.0,
                mean_volume=0.0,
            )

    def test_negative_se(self) -> None:
        with pytest.raises(ValueError):
            KyleLambdaFit(
                lambda_hat=1.0,
                se=-0.1,
                ci_low=0.0,
                ci_high=2.0,
                r_squared=0.0,
                n_obs=10,
                frequency="daily",
                mean_signed_volume_abs=0.0,
                mean_volume=0.0,
            )


class TestKyleInputs:
    def test_valid_daily(self) -> None:
        inputs = KyleInputs(
            ticker="ABC",
            daily_bars=_minimal_bars(),
            frequency="daily",
            parent_order_shares=1_000.0,
            side=1,
            Y_prefactor=1.0,
            timestamp=datetime(2026, 5, 18, tzinfo=UTC),
        )
        assert inputs.ticker == "ABC"
        assert inputs.intraday_bars is None

    def test_intraday_requires_bars(self) -> None:
        with pytest.raises(ValueError):
            KyleInputs(
                ticker="ABC",
                daily_bars=_minimal_bars(),
                frequency="intraday",
                parent_order_shares=1_000.0,
                side=1,
                Y_prefactor=1.0,
                timestamp=datetime(2026, 5, 18, tzinfo=UTC),
            )

    def test_invalid_side(self) -> None:
        with pytest.raises(ValueError):
            KyleInputs(
                ticker="ABC",
                daily_bars=_minimal_bars(),
                frequency="daily",
                parent_order_shares=1_000.0,
                side=0,
                Y_prefactor=1.0,
                timestamp=datetime(2026, 5, 18, tzinfo=UTC),
            )

    def test_missing_column_in_daily(self) -> None:
        bars = _minimal_bars().drop(columns=["Volume"])
        with pytest.raises(ValueError):
            KyleInputs(
                ticker="ABC",
                daily_bars=bars,
                frequency="daily",
                parent_order_shares=1_000.0,
                side=1,
                Y_prefactor=1.0,
                timestamp=datetime(2026, 5, 18, tzinfo=UTC),
            )
