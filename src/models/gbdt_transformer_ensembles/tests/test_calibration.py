"""Unit tests for `calibrate()` — the full CV/stack/refit pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.gbdt_transformer_ensembles.calibration import calibrate
from src.models.gbdt_transformer_ensembles.gbdt import GBDTParams
from src.models.gbdt_transformer_ensembles.sequence_model import LinearSeqParams
from src.models.gbdt_transformer_ensembles.types import (
    EnsembleFit,
    FeatureSpec,
    GBDTFit,
    LinearSeqFit,
    PanelFrame,
)


def _fast_params() -> tuple[GBDTParams, LinearSeqParams]:
    """Minimal params to keep CV runtime short."""

    return (
        GBDTParams(
            n_estimators=20,
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
        LinearSeqParams(windows=(5,), ewma_alpha=0.2, alpha=1.0),
    )


def test_calibrate_returns_calibration_result(
    fitted_panel: PanelFrame,
    small_feature_spec: FeatureSpec,
) -> None:
    gbdt_p, lin_p = _fast_params()
    result = calibrate(
        fitted_panel,
        feature_spec=small_feature_spec,
        n_splits=3,
        purge=1,
        embargo=1,
        gbdt_params=gbdt_p,
        linseq_params=lin_p,
    )
    assert result.model_name == "gbdt_transformer_ensembles"
    fit = result.parameters["ensemble_fit"]
    assert isinstance(fit, EnsembleFit)
    assert set(fit.base_fits.keys()) == {"gbdt", "linear_seq"}
    assert isinstance(fit.base_fits["gbdt"], GBDTFit)
    assert isinstance(fit.base_fits["linear_seq"], LinearSeqFit)


def test_calibrate_oof_predictions_have_required_columns(
    fitted_panel: PanelFrame,
    small_feature_spec: FeatureSpec,
) -> None:
    gbdt_p, lin_p = _fast_params()
    result = calibrate(
        fitted_panel,
        feature_spec=small_feature_spec,
        n_splits=3,
        purge=1,
        embargo=1,
        gbdt_params=gbdt_p,
        linseq_params=lin_p,
    )
    fit = result.parameters["ensemble_fit"]
    oof = fit.oof_predictions
    for col in ("gbdt", "linear_seq", "y", "w", "date", "ticker", "yhat_ensemble"):
        assert col in oof.columns
    assert oof[fit.base_learner_names[0]].notna().all()


def test_calibrate_stacker_attached_when_multiple_learners(
    fitted_panel: PanelFrame,
    small_feature_spec: FeatureSpec,
) -> None:
    gbdt_p, lin_p = _fast_params()
    result = calibrate(
        fitted_panel,
        feature_spec=small_feature_spec,
        base_learners=("gbdt", "linear_seq"),
        n_splits=3,
        purge=1,
        embargo=1,
        gbdt_params=gbdt_p,
        linseq_params=lin_p,
    )
    fit = result.parameters["ensemble_fit"]
    assert fit.has_stacker


def test_calibrate_single_learner_skips_stacker(
    fitted_panel: PanelFrame,
    small_feature_spec: FeatureSpec,
) -> None:
    gbdt_p, lin_p = _fast_params()
    result = calibrate(
        fitted_panel,
        feature_spec=small_feature_spec,
        base_learners=("gbdt",),
        n_splits=3,
        purge=1,
        embargo=1,
        gbdt_params=gbdt_p,
        linseq_params=lin_p,
    )
    fit = result.parameters["ensemble_fit"]
    assert not fit.has_stacker


def test_calibrate_oof_r2_is_finite(
    fitted_panel: PanelFrame,
    small_feature_spec: FeatureSpec,
) -> None:
    """Sanity: the CV R^2 should be a finite real number on synthetic data."""

    gbdt_p, lin_p = _fast_params()
    result = calibrate(
        fitted_panel,
        feature_spec=small_feature_spec,
        n_splits=3,
        purge=1,
        embargo=1,
        gbdt_params=gbdt_p,
        linseq_params=lin_p,
    )
    r2 = result.fit_metrics["weighted_r2"]
    assert np.isfinite(r2)


def test_calibrate_validates_panel_size(
    fitted_panel: PanelFrame,
    small_feature_spec: FeatureSpec,
) -> None:
    """Asking for more folds than rows should fail loudly."""

    gbdt_p, lin_p = _fast_params()
    with pytest.raises(ValueError):
        calibrate(
            fitted_panel,
            feature_spec=small_feature_spec,
            n_splits=fitted_panel.n_rows,
            purge=0,
            embargo=0,
            gbdt_params=gbdt_p,
            linseq_params=lin_p,
        )
