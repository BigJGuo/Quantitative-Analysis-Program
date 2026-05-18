"""Calibration entry-point tests for the Online Learning model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.online_learning.calibration import (
    MODEL_NAME,
    calibrate,
    calibrate_ftrl,
)
from src.models.online_learning.types import (
    ExpertStream,
    FTRLConfig,
    FTRLFit,
    HedgeConfig,
    HedgeFit,
)


class TestCalibrateHedge:
    def test_returns_calibration_result(
        self,
        dominant_expert_stream: ExpertStream,
    ) -> None:
        cfg = HedgeConfig(
            n_experts=2,
            eta=1.0,
            eta_schedule="constant",
            loss_kind="squared",
        )
        result = calibrate(stream=dominant_expert_stream, config=cfg)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        assert "hedge_fit" in result.parameters
        assert isinstance(result.parameters["hedge_fit"], HedgeFit)
        assert result.fit_metrics["n_rounds"] == 200
        assert result.fit_metrics["n_experts"] == 2

    def test_n_passes_can_run_multiple_replays(
        self,
        dominant_expert_stream: ExpertStream,
    ) -> None:
        cfg = HedgeConfig(
            n_experts=2, eta=0.5, eta_schedule="constant", loss_kind="squared"
        )
        single = calibrate(stream=dominant_expert_stream, config=cfg, n_passes=1)
        double = calibrate(stream=dominant_expert_stream, config=cfg, n_passes=2)
        # More passes → tighter concentration on the dominant expert.
        single_fit: HedgeFit = single.parameters["hedge_fit"]
        double_fit: HedgeFit = double.parameters["hedge_fit"]
        assert double_fit.weights[0] >= single_fit.weights[0]

    def test_rejects_mismatched_expert_count(
        self,
        dominant_expert_stream: ExpertStream,
    ) -> None:
        cfg = HedgeConfig(n_experts=5, loss_kind="squared")
        with pytest.raises(ValueError, match="experts"):
            calibrate(stream=dominant_expert_stream, config=cfg)

    def test_rejects_zero_passes(
        self,
        dominant_expert_stream: ExpertStream,
    ) -> None:
        cfg = HedgeConfig(n_experts=2, loss_kind="squared")
        with pytest.raises(ValueError, match="n_passes"):
            calibrate(stream=dominant_expert_stream, config=cfg, n_passes=0)

    def test_metadata_carries_ticker(
        self,
        dominant_expert_stream: ExpertStream,
    ) -> None:
        cfg = HedgeConfig(n_experts=2, loss_kind="squared")
        result = calibrate(
            stream=dominant_expert_stream, config=cfg, metadata={"ticker": "ABC"}
        )
        assert result.metadata["ticker"] == "ABC"
        assert result.metadata["kind"] == "hedge"

    def test_empirical_regret_is_finite_and_non_negative(
        self,
        dominant_expert_stream: ExpertStream,
    ) -> None:
        cfg = HedgeConfig(
            n_experts=2, eta=1.0, eta_schedule="constant", loss_kind="squared"
        )
        result = calibrate(stream=dominant_expert_stream, config=cfg)
        regret = result.fit_metrics["empirical_regret"]
        assert math.isfinite(regret)
        assert regret >= 0


class TestCalibrateFTRL:
    def test_returns_calibration_result(
        self,
        linear_regression_stream: tuple[
            list[tuple[np.ndarray, np.ndarray]], np.ndarray
        ],
    ) -> None:
        features, targets = linear_regression_stream
        cfg = FTRLConfig(dim=20, lambda1=0.1, loss_kind="squared")
        result = calibrate_ftrl(features=features, targets=targets, config=cfg)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == MODEL_NAME
        assert isinstance(result.parameters["ftrl_fit"], FTRLFit)
        assert result.fit_metrics["n_rounds"] == len(features)
        assert 0 <= result.fit_metrics["sparsity"] <= 1

    def test_rejects_mismatched_shapes(self) -> None:
        cfg = FTRLConfig(dim=4, lambda1=0.1, loss_kind="squared")
        features = [(np.array([0], dtype=np.int64), np.array([1.0]))]
        targets = np.array([1.0, 2.0])  # wrong length
        with pytest.raises(ValueError, match="align"):
            calibrate_ftrl(features=features, targets=targets, config=cfg)

    def test_rejects_zero_passes(
        self,
        linear_regression_stream: tuple[
            list[tuple[np.ndarray, np.ndarray]], np.ndarray
        ],
    ) -> None:
        features, targets = linear_regression_stream
        cfg = FTRLConfig(dim=20, lambda1=0.1, loss_kind="squared")
        with pytest.raises(ValueError, match="n_passes"):
            calibrate_ftrl(
                features=features, targets=targets, config=cfg, n_passes=0
            )

    def test_logistic_calibration_metadata(
        self,
        logistic_regression_stream: tuple[
            list[tuple[np.ndarray, np.ndarray]], np.ndarray
        ],
    ) -> None:
        features, targets = logistic_regression_stream
        cfg = FTRLConfig(dim=30, lambda1=0.5, loss_kind="log")
        result = calibrate_ftrl(
            features=features,
            targets=targets,
            config=cfg,
            metadata={"experiment": "smoke"},
        )
        assert result.metadata["experiment"] == "smoke"
        assert result.metadata["kind"] == "ftrl"
        assert result.fit_metrics["mean_prequential_loss"] < math.log(2)


def test_prequential_losses_indexed_by_stream_index(
    dominant_expert_stream: ExpertStream,
) -> None:
    cfg = HedgeConfig(
        n_experts=2, eta=1.0, eta_schedule="constant", loss_kind="squared"
    )
    result = calibrate(stream=dominant_expert_stream, config=cfg)
    fit: HedgeFit = result.parameters["hedge_fit"]
    assert isinstance(fit.prequential_losses, pd.Series)
    assert fit.prequential_losses.index.equals(dominant_expert_stream.predictions.index)
