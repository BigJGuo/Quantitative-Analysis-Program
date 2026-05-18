"""Unit tests for dataclass validation in `src.models.var.types`."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.var.types import (
    VaRBacktestResult,
    VaRFit,
    VaRInputs,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 18, 14, 30, tzinfo=UTC)


@pytest.fixture
def returns_2x2() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=50, freq="B")
    return pd.DataFrame(
        {"A": np.zeros(50), "B": np.zeros(50)}, index=idx
    )


class TestVaRInputs:
    def test_valid_inputs(
        self, now: datetime, returns_2x2: pd.DataFrame
    ) -> None:
        inputs = VaRInputs(
            positions={"A": 100.0, "B": 200.0},
            returns=returns_2x2,
            timestamp=now,
            method="historical",
            alpha=0.99,
            horizon_days=10,
        )
        assert inputs.alpha == 0.99
        assert inputs.method == "historical"

    def test_rejects_mismatched_columns(
        self, now: datetime, returns_2x2: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="must match"):
            VaRInputs(
                positions={"X": 100.0, "Y": 200.0},
                returns=returns_2x2,
                timestamp=now,
                method="historical",
                alpha=0.99,
                horizon_days=10,
            )

    def test_rejects_invalid_alpha(
        self, now: datetime, returns_2x2: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="alpha"):
            VaRInputs(
                positions={"A": 100.0, "B": 200.0},
                returns=returns_2x2,
                timestamp=now,
                method="historical",
                alpha=2.0,
                horizon_days=10,
            )

    def test_rejects_invalid_horizon(
        self, now: datetime, returns_2x2: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="horizon_days"):
            VaRInputs(
                positions={"A": 100.0, "B": 200.0},
                returns=returns_2x2,
                timestamp=now,
                method="historical",
                alpha=0.99,
                horizon_days=0,
            )

    def test_rejects_unknown_method(
        self, now: datetime, returns_2x2: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="method"):
            VaRInputs(
                positions={"A": 100.0, "B": 200.0},
                returns=returns_2x2,
                timestamp=now,
                method="bogus",  # type: ignore[arg-type]
                alpha=0.99,
                horizon_days=10,
            )

    def test_rejects_short_returns(self, now: datetime) -> None:
        short = pd.DataFrame(
            {"A": np.zeros(5), "B": np.zeros(5)},
            index=pd.date_range("2024-01-02", periods=5, freq="B"),
        )
        with pytest.raises(ValueError, match=">= 30"):
            VaRInputs(
                positions={"A": 100.0, "B": 200.0},
                returns=short,
                timestamp=now,
                method="historical",
                alpha=0.99,
                horizon_days=10,
            )


class TestVaRFit:
    def _base_kwargs(self) -> dict[str, object]:
        return dict(
            method="historical",
            alpha=0.99,
            horizon_days=10,
            n_obs=500,
            n_assets=2,
            portfolio_value=1_500_000.0,
            gross_exposure=1_500_000.0,
            var_1d_dollar=15_000.0,
            var_horizon_dollar=15_000.0 * np.sqrt(10),
            pnl_vector=np.zeros(500),
        )

    def test_valid_fit(self) -> None:
        fit = VaRFit(**self._base_kwargs())  # type: ignore[arg-type]
        assert fit.method == "historical"

    def test_rejects_negative_var(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["var_1d_dollar"] = -1.0
        with pytest.raises(ValueError, match="var_1d_dollar"):
            VaRFit(**kwargs)  # type: ignore[arg-type]

    def test_rejects_bad_pnl_shape(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["pnl_vector"] = np.zeros((10, 2))
        with pytest.raises(ValueError, match="1-D"):
            VaRFit(**kwargs)  # type: ignore[arg-type]

    def test_rejects_bad_sigma_shape(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["sigma_matrix"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="sigma_matrix"):
            VaRFit(**kwargs)  # type: ignore[arg-type]


class TestVaRBacktestResult:
    def test_valid(self) -> None:
        bt = VaRBacktestResult(
            method="historical",
            alpha=0.99,
            n=250,
            breaches=3,
            breach_rate=0.012,
            expected_breaches=2.5,
            kupiec_lr=0.1,
            kupiec_p=0.75,
            christoffersen_lr=0.5,
            christoffersen_p=0.48,
            traffic_light="green",
        )
        assert bt.traffic_light == "green"

    def test_rejects_bad_traffic_light(self) -> None:
        with pytest.raises(ValueError, match="traffic_light"):
            VaRBacktestResult(
                method="historical",
                alpha=0.99,
                n=250,
                breaches=3,
                breach_rate=0.012,
                expected_breaches=2.5,
                kupiec_lr=0.1,
                kupiec_p=0.75,
                christoffersen_lr=0.5,
                christoffersen_p=0.48,
                traffic_light="blue",  # type: ignore[arg-type]
            )
