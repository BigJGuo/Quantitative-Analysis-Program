"""End-to-end integration test for the GBDT + Transformer ensemble.

Hits yfinance via `YFinanceProvider` for a small universe of US sector
ETFs (matching the unit-test style of the other models). Auto-skipped
when the network fetch fails so unit-test CI stays green.

The test exercises the spec's full pipeline:

1. Multi-ticker yfinance fetch → feature panel.
2. Purged group K-fold CV with two heterogeneous base learners.
3. Ridge stacking on out-of-fold predictions.
4. Cross-sectional prediction at the most-recent date.
5. The "Validation and diagnostics" section (weighted R^2, rank IC,
   feature importance).
"""

from __future__ import annotations

import numpy as np
import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, Forecast
from src.models.gbdt_transformer_ensembles import (
    EnsembleFit,
    EnsemblePrediction,
    FeatureSpec,
    GBDTParams,
    GBDTTransformerEnsemble,
    LinearSeqParams,
    PanelFrame,
)

pytest.importorskip("yfinance")

# A small sector-ETF universe + SPY as the benchmark. Compact enough that
# yfinance pulls run in a few seconds, broad enough that the ensemble has
# real cross-sectional dispersion to learn from.
_UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU"]


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model() -> GBDTTransformerEnsemble:
    feature_spec = FeatureSpec(
        return_horizons=(1, 5, 21),
        lag_count=5,
        vol_windows=(5, 21),
        mean_windows=(5, 21),
        target_horizon=5,
        clip_target=0.15,
        include_cross_sectional_ranks=True,
        include_calendar=True,
        benchmark_ticker="SPY",
        min_history_days=63,
    )
    gbdt_params = GBDTParams(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        colsample=0.8,
        l2_leaf_reg=1.0,
        loss="huber",
        early_stopping_rounds=10,
        seed=0,
    )
    linseq_params = LinearSeqParams(windows=(5, 21), ewma_alpha=0.1, alpha=1.0)
    return GBDTTransformerEnsemble(
        universe=_UNIVERSE,
        feature_spec=feature_spec,
        gbdt_params=gbdt_params,
        linseq_params=linseq_params,
        n_splits=3,
        purge=2,
        embargo=1,
        period="2y",
    )


def _fetch_or_skip(
    model: GBDTTransformerEnsemble, provider: YFinanceProvider
) -> PanelFrame:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_pipeline_against_yfinance(
    provider: YFinanceProvider, model: GBDTTransformerEnsemble
) -> None:
    panel = _fetch_or_skip(model, provider)
    assert isinstance(panel, PanelFrame)
    assert panel.n_rows > 0
    # Expect ~ (universe size) * (trading days) rows, post-history-cutoff.
    assert panel.n_rows > 300

    result = model.calibrate(panel)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "gbdt_transformer_ensembles"
    fit = result.parameters["ensemble_fit"]
    assert isinstance(fit, EnsembleFit)
    assert set(fit.base_learner_names) == {"gbdt", "linear_seq"}
    assert fit.has_stacker

    pred = model.predict(panel)
    assert isinstance(pred, Forecast)
    assert pred.horizon == "5d"
    assert np.isfinite(pred.value)

    cs = model.predict_cross_section(panel)
    assert isinstance(cs, EnsemblePrediction)
    assert len(cs.tickers) >= len(_UNIVERSE) - 2
    assert cs.values.shape == (len(cs.tickers),)
    assert set(cs.per_learner.keys()) == {"gbdt", "linear_seq"}


@pytest.mark.integration
def test_diagnostics_section_of_spec(
    provider: YFinanceProvider, model: GBDTTransformerEnsemble
) -> None:
    """Cross-check the spec's "Validation and diagnostics" outputs."""

    panel = _fetch_or_skip(model, provider)
    model.calibrate(panel)
    diag = model.validate(panel)

    # Primary metrics must exist and be finite (signal can be weak on a
    # short window of sector ETFs — we don't constrain the sign).
    assert np.isfinite(diag["weighted_r2"])
    assert np.isfinite(diag["cross_sectional_rank_ic"])
    assert np.isfinite(diag["time_series_ic"])

    # In-sample R^2 should be at least as good as OOF — the ensemble must
    # have learned _something_ from the training data.
    is_r2 = max(diag["in_sample_r2"].values())
    assert is_r2 >= diag["weighted_r2"] - 0.1

    # Feature importance for the GBDT base learner: at least one feature
    # must have positive gain, and the top features should include a
    # return- or volatility-type column (the dominant signals in the spec).
    top = diag["top_features"]
    assert "gbdt" in top
    assert len(top["gbdt"]) > 0
    important_names = " ".join(top["gbdt"].keys())
    assert any(k in important_names for k in ("r1", "r5", "vol", "mean", "lag", "cs_rank"))


@pytest.mark.integration
def test_purged_kfold_oof_coverage(
    provider: YFinanceProvider, model: GBDTTransformerEnsemble
) -> None:
    """Every fold's validation set must contribute OOF predictions."""

    panel = _fetch_or_skip(model, provider)
    result = model.calibrate(panel)
    fit = result.parameters["ensemble_fit"]
    oof = fit.oof_predictions
    # All base-learner OOF cols are finite (we drop incomplete rows).
    for name in fit.base_learner_names:
        assert oof[name].notna().all()
    # OOF should cover multiple distinct dates.
    assert oof["date"].nunique() >= 10
