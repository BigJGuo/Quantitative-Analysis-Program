"""Tests for the `BayesianKalman` orchestration shell.

These exercise `fetch_data`/`calibrate`/`predict`/`validate` against
`InMemoryProvider` with hand-built fixture data so we don't depend on
yfinance for unit tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.types import Forecast, RiskMetric
from src.models.bayesian_kalman.model import BayesianKalman
from src.models.bayesian_kalman.types import (
    BayesianKalmanInputs,
    DynamicFactorFit,
    HierarchicalFit,
    KalmanFit,
)


def _bars_from_returns(series: pd.Series) -> pd.DataFrame:
    """Build a yfinance-shaped OHLCV-ish frame whose Close is exp(cumsum(r))."""

    prices = np.exp(series.cumsum().to_numpy())
    return pd.DataFrame(
        {"Date": series.index, "Close": prices, "Open": prices, "High": prices, "Low": prices, "Volume": 0.0}
    )


def _make_provider(
    return_series: dict[str, pd.Series], period: str = "2y", interval: str = "1d"
) -> InMemoryProvider:
    prices = {(t, period, interval): _bars_from_returns(s) for t, s in return_series.items()}
    return InMemoryProvider(prices=prices)


def test_model_kalman_beta_end_to_end(
    constant_beta_series: tuple[pd.Series, pd.Series, float]
) -> None:
    asset_r, market_r, true_beta = constant_beta_series
    idx = pd.date_range("2023-01-01", periods=len(asset_r), freq="B")
    asset_r = pd.Series(asset_r.to_numpy(), index=idx, name="AAPL")
    market_r = pd.Series(market_r.to_numpy(), index=idx, name="SPY")

    provider = _make_provider({"AAPL": asset_r, "SPY": market_r})
    model = BayesianKalman(
        ticker="AAPL",
        market_ticker="SPY",
        mode="kalman_beta",
        fix_hyperparameters=(1e-12, 1e-8, 0.005 ** 2),
    )
    data = model.fetch_data(provider)
    assert isinstance(data, BayesianKalmanInputs)
    result = model.calibrate(data)
    assert result.model_name == "bayesian_kalman"
    fit = model.fit
    assert isinstance(fit, KalmanFit)

    pred = model.predict(data)
    assert isinstance(pred, Forecast)
    assert pred.ticker == "AAPL"
    assert pred.horizon == "1d"
    assert abs(pred.value - true_beta) < 0.20
    assert pred.lower is not None and pred.upper is not None
    assert pred.lower < pred.value < pred.upper

    diag = model.validate(data)
    assert diag["n_obs"] >= 400
    assert "ljung_box_pvalue" in diag


def test_model_hierarchical_end_to_end(
    hierarchical_panel: tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]
) -> None:
    designs, targets, _true_thetas = hierarchical_panel
    # Convert dict-of-arrays back into a yfinance-shaped panel so the model's
    # fetch_data path works. We need a market series and one return series
    # per asset such that, after the model rebuilds the design matrix
    # (X = [1, market_returns]), the response is what we already have.
    # We do that by reconstructing market = design_column_1 and constructing
    # asset returns from y = X @ theta + ... already present in `targets`.
    tickers = tuple(sorted(designs.keys()))
    # All assets share the *same* design column-1 because the integration
    # test only needs a market and a per-asset return series. Use the first
    # asset's design as the market series; all other assets' market columns
    # may differ — for this test we just pick the model's first ticker's
    # market column.
    market_arr = designs[tickers[0]][:, 1]
    idx = pd.date_range("2023-01-01", periods=len(market_arr), freq="B")
    market_r = pd.Series(market_arr, index=idx, name="SPY")
    return_series = {"SPY": market_r}
    for t in tickers:
        # Reconstruct y_t when X uses *this* market series: r_t = y_t (the
        # design's market column for asset t may differ from SPY but the model
        # will see this asset's y and the universal market in fetch_data and
        # build its own design matrix from those).
        y_t = targets[t]
        return_series[t] = pd.Series(y_t, index=idx, name=t)

    provider = _make_provider(return_series)
    model = BayesianKalman(
        ticker=tickers[0],
        market_ticker="SPY",
        universe=tickers[1:],
        mode="hierarchical",
        n_iter=200,
        n_burn=100,
    )
    data = model.fetch_data(provider)
    result = model.calibrate(data)
    assert isinstance(model.fit, HierarchicalFit)
    pred = model.predict(data)
    assert isinstance(pred, Forecast)
    assert pred.ticker == tickers[0]

    diag = model.validate(data)
    assert diag["n_assets"] >= 2
    assert diag["mean_sigma_sq"] > 0
    assert result.fit_metrics["n_iter"] == 200


def test_model_dynamic_factor_end_to_end(
    dfm_panel: tuple[pd.DataFrame, np.ndarray, np.ndarray]
) -> None:
    panel, _Lambda, _factors = dfm_panel
    return_series = {
        str(c): pd.Series(panel[c].to_numpy(), index=panel.index, name=str(c))
        for c in panel.columns
    }
    provider = _make_provider(return_series)
    model = BayesianKalman(
        universe=tuple(str(c) for c in panel.columns),
        mode="dynamic_factor",
        n_factors=2,
        n_em_iter=20,
    )
    data = model.fetch_data(provider)
    result = model.calibrate(data)
    assert isinstance(model.fit, DynamicFactorFit)

    pred = model.predict(data)
    assert isinstance(pred, RiskMetric)
    assert pred.ticker == "PORTFOLIO"
    assert pred.value > 0.0

    diag = model.validate(data)
    assert diag["n_factors"] == 2
    assert diag["mean_r_squared"] > 0.5
    assert result.fit_metrics["variance_explained"] > 0.5


def test_model_init_validates_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        BayesianKalman(ticker="AAPL", mode="not_a_mode")  # type: ignore[arg-type]


def test_model_init_requires_ticker_for_kalman_beta() -> None:
    with pytest.raises(ValueError, match="requires a primary ticker"):
        BayesianKalman(mode="kalman_beta")


def test_model_init_requires_universe_for_dfm() -> None:
    with pytest.raises(ValueError, match="universe"):
        BayesianKalman(mode="dynamic_factor", universe=("ONLY_ONE",))


def test_predict_before_calibrate_raises() -> None:
    model = BayesianKalman(
        ticker="AAPL", market_ticker="SPY", mode="kalman_beta"
    )
    inputs = BayesianKalmanInputs(
        mode="kalman_beta",
        timestamp=pd.Timestamp("2024-01-01").to_pydatetime(),
        asset_returns=pd.Series([0.0] * 40),
        market_returns=pd.Series([0.0] * 40),
    )
    with pytest.raises(RuntimeError, match="not been calibrated"):
        model.predict(inputs)


def test_predict_wrong_inputs_type_raises() -> None:
    model = BayesianKalman(ticker="AAPL", mode="kalman_beta")
    with pytest.raises(TypeError, match="BayesianKalmanInputs"):
        model.predict("not the right thing")
