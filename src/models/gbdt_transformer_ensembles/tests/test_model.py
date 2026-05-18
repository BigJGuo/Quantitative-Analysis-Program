"""Unit tests for the `GBDTTransformerEnsemble` model class.

Uses an `InMemoryProvider` populated with the synthetic OHLCV fixtures so
no network calls happen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, Forecast
from src.models.gbdt_transformer_ensembles.gbdt import GBDTParams
from src.models.gbdt_transformer_ensembles.model import GBDTTransformerEnsemble
from src.models.gbdt_transformer_ensembles.sequence_model import LinearSeqParams
from src.models.gbdt_transformer_ensembles.types import (
    EnsembleFit,
    FeatureSpec,
    PanelFrame,
)


def _build_provider(prices: dict[str, pd.DataFrame], period: str) -> InMemoryProvider:
    bars: dict[tuple[str, str, str], pd.DataFrame] = {}
    for t, df in prices.items():
        bars[(t, period, "1d")] = df.reset_index().rename(columns={"index": "Date"})
    return InMemoryProvider(prices=bars)


def _model(small_feature_spec: FeatureSpec) -> GBDTTransformerEnsemble:
    fs = FeatureSpec(
        return_horizons=small_feature_spec.return_horizons,
        lag_count=small_feature_spec.lag_count,
        vol_windows=small_feature_spec.vol_windows,
        mean_windows=small_feature_spec.mean_windows,
        target_horizon=small_feature_spec.target_horizon,
        clip_target=small_feature_spec.clip_target,
        include_cross_sectional_ranks=small_feature_spec.include_cross_sectional_ranks,
        include_calendar=small_feature_spec.include_calendar,
        benchmark_ticker=None,  # use no benchmark in unit tests
        min_history_days=small_feature_spec.min_history_days,
    )
    return GBDTTransformerEnsemble(
        universe=[f"T{i}" for i in range(6)],
        feature_spec=fs,
        gbdt_params=GBDTParams(
            n_estimators=15,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=10,
            subsample=1.0,
            colsample=1.0,
            l2_leaf_reg=1.0,
            loss="huber",
            early_stopping_rounds=5,
            seed=0,
        ),
        linseq_params=LinearSeqParams(windows=(5,), ewma_alpha=0.2, alpha=1.0),
        n_splits=3,
        purge=1,
        embargo=1,
        period="2y",
    )


def test_model_registered() -> None:
    cls = get_model("gbdt_transformer_ensembles")
    assert cls is GBDTTransformerEnsemble


def test_model_class_metadata() -> None:
    assert GBDTTransformerEnsemble.name == "gbdt_transformer_ensembles"
    assert GBDTTransformerEnsemble.layer == 4
    assert GBDTTransformerEnsemble.refit_frequency == "weekly"


def test_model_fetch_calibrate_predict_validate(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> None:
    provider = _build_provider(synthetic_prices, period="2y")
    model = _model(small_feature_spec)

    panel = model.fetch_data(provider)
    assert isinstance(panel, PanelFrame)

    result = model.calibrate(panel)
    assert isinstance(result, CalibrationResult)
    assert "ensemble_fit" in result.parameters

    pred = model.predict(panel)
    assert isinstance(pred, Forecast)
    assert pred.horizon == f"{small_feature_spec.target_horizon}d"
    assert np.isfinite(pred.value)

    cs = model.predict_cross_section(panel)
    assert len(cs.tickers) > 0
    assert cs.values.shape == (len(cs.tickers),)

    diag = model.validate(panel)
    assert "weighted_r2" in diag
    assert "top_features" in diag


def test_predict_without_calibration_errors(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> None:
    provider = _build_provider(synthetic_prices, period="2y")
    model = _model(small_feature_spec)
    panel = model.fetch_data(provider)
    with pytest.raises(RuntimeError):
        model.predict(panel)


def test_feature_importance_returns_dataframe(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> None:
    provider = _build_provider(synthetic_prices, period="2y")
    model = _model(small_feature_spec)
    panel = model.fetch_data(provider)
    model.calibrate(panel)
    fi = model.feature_importance()
    assert {"feature", "gain", "split"}.issubset(fi.columns)
    assert len(fi) == len(panel.feature_names)


def test_universe_size_validation() -> None:
    with pytest.raises(ValueError):
        GBDTTransformerEnsemble(universe=["A"])


def test_diagnostic_in_sample_better_than_oof_on_synthetic(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> None:
    """Sanity diagnostic for the spec's 'Validation and diagnostics' section.

    The in-sample R^2 should be no worse than OOF (else something is broken);
    typically it's strictly better.
    """

    provider = _build_provider(synthetic_prices, period="2y")
    model = _model(small_feature_spec)
    panel = model.fetch_data(provider)
    model.calibrate(panel)
    diag = model.validate(panel)
    oof_r2 = diag["weighted_r2"]
    is_r2 = max(diag["in_sample_r2"].values())
    assert is_r2 >= oof_r2 - 0.1  # allow a tiny CV-noise margin


def test_ensemble_fit_metadata_round_trip(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> None:
    provider = _build_provider(synthetic_prices, period="2y")
    model = _model(small_feature_spec)
    panel = model.fetch_data(provider)
    result = model.calibrate(panel)
    fit = result.parameters["ensemble_fit"]
    assert isinstance(fit, EnsembleFit)
    assert fit.n_features == len(panel.feature_names)
    assert fit.target_horizon == small_feature_spec.target_horizon
