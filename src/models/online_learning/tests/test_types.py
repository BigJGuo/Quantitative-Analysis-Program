"""Validation logic on the model's dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.online_learning.types import (
    ExpertStream,
    FTRLConfig,
    FTRLState,
    HedgeConfig,
    HedgeState,
    OnlineLearningInputs,
)


class TestHedgeConfig:
    def test_defaults_validate(self) -> None:
        cfg = HedgeConfig(n_experts=5)
        assert cfg.eta == pytest.approx(0.1)
        assert cfg.fixed_share == 0.0

    def test_rejects_singleton(self) -> None:
        with pytest.raises(ValueError, match="n_experts"):
            HedgeConfig(n_experts=1)

    def test_rejects_non_positive_eta(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            HedgeConfig(n_experts=3, eta=0.0)

    def test_rejects_invalid_schedule(self) -> None:
        with pytest.raises(ValueError, match="eta_schedule"):
            HedgeConfig(n_experts=3, eta_schedule="banana")  # type: ignore[arg-type]

    def test_rejects_out_of_range_fixed_share(self) -> None:
        with pytest.raises(ValueError, match="fixed_share"):
            HedgeConfig(n_experts=3, fixed_share=1.5)

    def test_rejects_invalid_loss_kind(self) -> None:
        with pytest.raises(ValueError, match="loss_kind"):
            HedgeConfig(n_experts=3, loss_kind="banana")  # type: ignore[arg-type]


class TestFTRLConfig:
    def test_defaults_validate(self) -> None:
        cfg = FTRLConfig(dim=10)
        assert cfg.lambda1 > 0
        assert cfg.loss_kind == "log"

    def test_rejects_invalid_loss_kind(self) -> None:
        with pytest.raises(ValueError, match="loss_kind"):
            FTRLConfig(dim=10, loss_kind="log_squared")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dim": 0}, "dim"),
            ({"dim": 10, "alpha": 0.0}, "alpha"),
            ({"dim": 10, "beta": -0.1}, "beta"),
            ({"dim": 10, "lambda1": -1.0}, "lambda1"),
            ({"dim": 10, "lambda2": -1.0}, "lambda2"),
        ],
    )
    def test_rejects_bad_params(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            FTRLConfig(**kwargs)


class TestHedgeState:
    def test_default_cumulative_losses_zeroed(self) -> None:
        w = np.array([0.5, 0.5])
        state = HedgeState(weights=w)
        assert np.array_equal(state.cumulative_expert_losses, np.zeros(2))

    def test_rejects_non_1d_weights(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            HedgeState(weights=np.eye(2))

    def test_rejects_mismatched_cumulative(self) -> None:
        with pytest.raises(ValueError, match="same"):
            HedgeState(
                weights=np.array([0.5, 0.5]),
                cumulative_expert_losses=np.zeros(3),
            )


class TestFTRLState:
    def test_shapes_must_match(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            FTRLState(z=np.zeros(4), n=np.zeros(5))

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            FTRLState(z=np.zeros((2, 2)), n=np.zeros((2, 2)))


class TestExpertStream:
    def test_rejects_singleton_expert(self) -> None:
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        preds = pd.DataFrame({"only": [1.0, 1.0, 1.0]}, index=idx)
        target = pd.Series([1.0, 1.0, 1.0], index=idx)
        with pytest.raises(ValueError, match="at least 2 experts"):
            ExpertStream(predictions=preds, targets=target)

    def test_rejects_misaligned_index(self) -> None:
        idx_a = pd.date_range("2024-01-02", periods=3, freq="B")
        idx_b = pd.date_range("2024-02-02", periods=3, freq="B")
        preds = pd.DataFrame({"a": [1.0] * 3, "b": [2.0] * 3}, index=idx_a)
        target = pd.Series([1.0] * 3, index=idx_b)
        with pytest.raises(ValueError, match="index"):
            ExpertStream(predictions=preds, targets=target)

    def test_rejects_wrong_targets_type(self) -> None:
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        preds = pd.DataFrame({"a": [1.0] * 3, "b": [2.0] * 3}, index=idx)
        with pytest.raises(TypeError, match="Series"):
            ExpertStream(predictions=preds, targets=[1.0, 2.0, 3.0])  # type: ignore[arg-type]


class TestOnlineLearningInputs:
    def _stream(self) -> ExpertStream:
        idx = pd.date_range("2024-01-02", periods=4, freq="B")
        preds = pd.DataFrame({"a": [0.1] * 4, "b": [0.2] * 4}, index=idx)
        target = pd.Series([0.15] * 4, index=idx)
        return ExpertStream(predictions=preds, targets=target)

    def test_rejects_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            OnlineLearningInputs(
                ticker="SPY",
                stream=self._stream(),
                kind="banana",  # type: ignore[arg-type]
                timestamp=datetime.now(UTC),
            )

    def test_rejects_non_positive_year(self) -> None:
        with pytest.raises(ValueError, match="trading_days_per_year"):
            OnlineLearningInputs(
                ticker="SPY",
                stream=self._stream(),
                kind="hedge",
                timestamp=datetime.now(UTC),
                trading_days_per_year=0,
            )
