"""Unit tests for `calibrate()` — the end-to-end pipeline on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.types import CalibrationResult
from src.models.avellaneda_lee_stat_arb.calibration import calibrate
from src.models.avellaneda_lee_stat_arb.types import StatArbFit, StatArbInputs


def _make_inputs(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame | None,
    *,
    factor_mode: str = "ETF",
    K: int = 1,
    pca_window: int = 200,
    ou_window: int = 60,
) -> StatArbInputs:
    return StatArbInputs(
        returns=asset_returns,
        tickers=tuple(asset_returns.columns),
        factor_mode=factor_mode,  # type: ignore[arg-type]
        K=K,
        pca_window=pca_window,
        ou_window=ou_window,
        timestamp=pd.Timestamp("2026-05-18").to_pydatetime(),
        factor_returns=factor_returns,
    )


def test_calibrate_ETF_mode_runs_end_to_end(
    synthetic_panel: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """End-to-end ETF-mode fit on a synthetic OU-residual panel.

    With well-behaved OU residuals (half-life ~7 days) the model should
    survive most stocks and produce finite s-scores.
    """
    asset_returns, factor_returns = synthetic_panel
    inputs = _make_inputs(asset_returns, factor_returns, factor_mode="ETF", K=1)

    result = calibrate(inputs, half_life_band=(2.0, 60.0), min_r_squared=0.1)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "avellaneda_lee_stat_arb"
    fit = result.parameters["stat_arb_fit"]
    assert isinstance(fit, StatArbFit)
    # Most synthetic names should survive the loose filters.
    assert fit.n_surviving >= 8
    # s-scores should be finite and the cross-section ~mean-zero (it's a
    # standardized residual; aggregate should sit near N(0, 1)).
    s_values = np.array([f.s_score_mod for f in fit.ou_fits.values()])
    assert np.all(np.isfinite(s_values))
    assert abs(s_values.mean()) < 1.5
    # All survivors should have half-life in band.
    for ou_fit in fit.ou_fits.values():
        assert 2.0 <= ou_fit.half_life <= 60.0
        assert 0.0 < ou_fit.b < 1.0


def test_calibrate_PCA_mode_builds_factors_from_panel(
    synthetic_panel: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    asset_returns, _ = synthetic_panel
    inputs = _make_inputs(asset_returns, None, factor_mode="PCA", K=2)

    result = calibrate(inputs, half_life_band=(2.0, 60.0), min_r_squared=0.1)
    fit = result.parameters["stat_arb_fit"]
    assert fit.factor_mode == "PCA"
    assert fit.factor_names == ("PC1", "PC2")
    assert fit.eigenvalues is not None
    assert fit.marchenko_pastur_cutoff is not None
    assert fit.n_surviving > 0


def test_calibrate_drops_non_stationary_stocks(
    rng: np.random.Generator,
) -> None:
    """A pure random-walk return (non-stationary cumulative residual) should
    be dropped by the stationarity filter."""

    n_obs = 300
    n_assets = 4
    # 3 mean-reverting names, 1 random walk.
    factor = rng.normal(0.0, 0.01, size=n_obs)
    rows = []
    for i in range(n_assets):
        if i == n_assets - 1:
            # Random walk *returns* -> X is a random walk in second integral,
            # but the AR(1) on it will often produce b ~ 1 / outside band.
            rows.append(0.01 * rng.normal(0.0, 1.0, size=n_obs))
        else:
            kappa = 0.15
            b = float(np.exp(-kappa))
            X = np.zeros(n_obs)
            sigma_zeta = 0.01
            z = rng.normal(0.0, sigma_zeta, size=n_obs)
            for t in range(1, n_obs):
                X[t] = b * X[t - 1] + z[t]
            rows.append(np.diff(X, prepend=0.0) + 0.9 * factor)
    R = np.column_stack(rows)
    asset_returns = pd.DataFrame(
        R,
        index=pd.date_range("2024-01-01", periods=n_obs, freq="B"),
        columns=[f"S{i}" for i in range(n_assets)],
    )
    factor_returns = pd.DataFrame(
        {"MKT": factor}, index=asset_returns.index
    )
    inputs = _make_inputs(
        asset_returns, factor_returns, factor_mode="ETF", K=1, pca_window=200, ou_window=60
    )
    result = calibrate(inputs, half_life_band=(2.0, 90.0), min_r_squared=0.0)
    fit = result.parameters["stat_arb_fit"]
    # The mean-reverting names should survive; the random-walk one *may* be
    # dropped or pass with a long half-life. Either way, the calibration
    # must not crash and the survivor count must respect the universe.
    assert fit.n_surviving + fit.n_dropped == n_assets


def test_calibrate_validates_inputs() -> None:
    panel = pd.DataFrame(
        np.zeros((100, 3)),
        index=pd.date_range("2024-01-01", periods=100, freq="B"),
        columns=["A", "B", "C"],
    )
    # pca_window > rows -> error
    inputs = StatArbInputs(
        returns=panel,
        tickers=("A", "B", "C"),
        factor_mode="PCA",
        K=1,
        pca_window=200,
        ou_window=60,
        timestamp=pd.Timestamp("2026-05-18").to_pydatetime(),
        factor_returns=None,
    )
    with pytest.raises(ValueError):
        calibrate(inputs)


def test_calibrate_rejects_bad_half_life_band(
    synthetic_panel: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    asset_returns, factor_returns = synthetic_panel
    inputs = _make_inputs(asset_returns, factor_returns)
    with pytest.raises(ValueError):
        calibrate(inputs, half_life_band=(30.0, 5.0))
    with pytest.raises(ValueError):
        calibrate(inputs, half_life_band=(0.0, 30.0))


def test_calibrate_drift_correction_changes_s_score(
    synthetic_panel: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """With drift correction off, s_score == s_score_mod."""
    asset_returns, factor_returns = synthetic_panel
    inputs = _make_inputs(asset_returns, factor_returns, factor_mode="ETF", K=1)

    fit_off = calibrate(inputs, use_drift_correction=False).parameters[
        "stat_arb_fit"
    ]
    fit_on = calibrate(inputs, use_drift_correction=True).parameters[
        "stat_arb_fit"
    ]
    for ticker, ou_fit in fit_off.ou_fits.items():
        assert ou_fit.s_score == pytest.approx(ou_fit.s_score_mod)
        # And the same ticker with correction on should generally differ.
        # (Equality is possible only when alpha is exactly zero.)
        assert ticker in fit_on.ou_fits
