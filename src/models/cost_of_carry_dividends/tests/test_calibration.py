"""Unit tests for `src.models.cost_of_carry_dividends.calibration`."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.models.cost_of_carry_dividends.calibration import (
    MODEL_NAME,
    calibrate,
    project_forward_dividends,
)
from src.models.cost_of_carry_dividends.types import (
    CostParameters,
    DividendStream,
    IndexConstituent,
)


def test_calibrate_uses_quantile_when_history_is_long_enough() -> None:
    rng = np.random.default_rng(42)
    # Synthetic basis: zero-mean with 1.0 std -> 95% quantile of |x| ~ 1.96.
    history = rng.normal(scale=1.0, size=500).tolist()
    result = calibrate(basis_history=history)
    assert result.model_name == MODEL_NAME
    cost = result.parameters["cost_params"]
    assert isinstance(cost, CostParameters)
    assert 1.5 < cost.futures_band_points < 2.3  # rough sanity around 1.96


def test_calibrate_falls_back_when_history_too_short() -> None:
    result = calibrate(
        basis_history=[0.1, 0.2],   # less than _MIN_BASIS_OBS
        fallback_band_points=0.7,
    )
    cost = result.parameters["cost_params"]
    assert cost.futures_band_points == pytest.approx(0.7)


def test_calibrate_returns_empty_stream_when_constituents_missing() -> None:
    result = calibrate(basis_history=[])
    stream = result.parameters["dividend_stream"]
    assert isinstance(stream, DividendStream)
    assert len(stream) == 0


def test_project_forward_dividends_announces_then_projects() -> None:
    today = datetime(2026, 5, 18)
    expiry = today + timedelta(days=400)

    constituents = [IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5)]
    # Quarterly history: 8 quarters at 0.22, 0.22, 0.22, 0.24, 0.24, 0.24, 0.24, 0.24
    dates = pd.to_datetime(
        [datetime(2024, 5, 10) + timedelta(days=90 * i) for i in range(8)]
    )
    series = pd.Series([0.22, 0.22, 0.22, 0.24, 0.24, 0.24, 0.24, 0.24], index=dates)
    divs = {"AAPL": series}
    stream = project_forward_dividends(
        constituents=constituents,
        dividends_per_share=divs,
        today=today,
        expiry=expiry,
        announcement_horizon_days=45,
    )
    # Should contain a mix of announced and projected dividends.
    sources = {d.source for d in stream.dividends}
    # The series ends in early 2026, so most quarters between today and expiry
    # will be projected; a quarter that falls within ~45 days of today might
    # be flagged announced.
    assert "projected" in sources
    assert all(d.amount_index_points > 0 for d in stream.dividends)


def test_project_forward_dividends_handles_empty_history() -> None:
    today = datetime(2026, 5, 18)
    expiry = today + timedelta(days=180)
    constituents = [IndexConstituent(ticker="NEW", weight=0.01, shares_factor=1.0)]
    stream = project_forward_dividends(
        constituents=constituents,
        dividends_per_share={"NEW": pd.Series([], dtype="float64")},
        today=today,
        expiry=expiry,
    )
    assert len(stream) == 0


def test_calibrate_combines_history_and_projection() -> None:
    today = datetime(2026, 5, 18)
    expiry = today + timedelta(days=180)
    constituents = [IndexConstituent(ticker="AAPL", weight=0.07, shares_factor=1.5)]
    dates = pd.to_datetime(
        [datetime(2024, 5, 10) + timedelta(days=90 * i) for i in range(8)]
    )
    series = pd.Series([0.22] * 4 + [0.24] * 4, index=dates)
    rng = np.random.default_rng(1)
    history = rng.normal(scale=0.5, size=50).tolist()
    result = calibrate(
        basis_history=history,
        constituents=constituents,
        dividends_per_share={"AAPL": series},
        today=today,
        expiry=expiry,
    )
    stream = result.parameters["dividend_stream"]
    assert len(stream) > 0
    # Diagnostics include both flavors of count.
    assert "n_dividends_announced" in result.fit_metrics
    assert "n_dividends_projected" in result.fit_metrics
