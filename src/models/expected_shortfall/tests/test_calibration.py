"""Unit tests for `src.models.expected_shortfall.calibration`."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.models.expected_shortfall.calibration import MODEL_NAME, calibrate
from src.models.expected_shortfall.types import ESFit, ESInputs


def _build_inputs(
    returns: pd.DataFrame,
    *,
    method: str = "parametric_normal",
    dollar_positions: np.ndarray | None = None,
    lookback_N: int = 500,
    alpha: float = 0.975,
) -> ESInputs:
    tickers = tuple(returns.columns)
    if dollar_positions is None:
        dollar_positions = np.array([10_000.0] * len(tickers), dtype=float)
    return ESInputs(
        tickers=tickers,
        returns=returns,
        dollar_positions=dollar_positions,
        alpha=alpha,
        method=method,  # type: ignore[arg-type]
        lookback_N=lookback_N,
        timestamp=datetime(2026, 5, 18, tzinfo=UTC),
    )


def test_calibrate_returns_model_name(synthetic_returns) -> None:  # type: ignore[no-untyped-def]
    inp = _build_inputs(synthetic_returns)
    result = calibrate(inp)
    assert result.model_name == MODEL_NAME


def test_calibrate_recovers_sample_moments(synthetic_returns, true_mvn_params) -> None:  # type: ignore[no-untyped-def]
    inp = _build_inputs(synthetic_returns, lookback_N=2000)
    result = calibrate(inp)
    fit: ESFit = result.parameters["fit"]
    # Sample mean of 2000 obs of N(0.0005, ~0.02) has SE ~4.5e-4, so 3-sigma
    # tolerance is ~1.5e-3. We use the absolute tolerance scaled accordingly.
    np.testing.assert_allclose(fit.mu, true_mvn_params["mu"], atol=2e-3)
    # Diagonal terms recover to within ~5%; the small off-diagonal has SE
    # comparable to its magnitude, so use an absolute tolerance scaled to
    # the diagonal.
    np.testing.assert_allclose(np.diag(fit.Sigma), np.diag(true_mvn_params["Sigma"]), rtol=0.05)
    np.testing.assert_allclose(fit.Sigma, true_mvn_params["Sigma"], atol=5e-5)


def test_calibrate_sigma_p_matches_quadratic_form(synthetic_returns) -> None:  # type: ignore[no-untyped-def]
    w = np.array([10_000.0, 20_000.0])
    inp = _build_inputs(synthetic_returns, dollar_positions=w, lookback_N=1000)
    result = calibrate(inp)
    fit: ESFit = result.parameters["fit"]
    expected_sigma = float(np.sqrt(w @ fit.Sigma @ w))
    assert abs(fit.sigma_p - expected_sigma) < 1e-9


def test_calibrate_skips_student_t_under_other_methods(synthetic_returns) -> None:  # type: ignore[no-untyped-def]
    for method in ("parametric_normal", "historical", "monte_carlo"):
        inp = _build_inputs(synthetic_returns, method=method)
        fit: ESFit = calibrate(inp).parameters["fit"]
        assert fit.nu is None
        assert fit.loc is None
        assert fit.scale is None


def test_calibrate_fits_student_t_when_method_is_parametric_t(synthetic_returns) -> None:  # type: ignore[no-untyped-def]
    inp = _build_inputs(synthetic_returns, method="parametric_t", lookback_N=1500)
    fit: ESFit = calibrate(inp).parameters["fit"]
    assert fit.nu is not None and fit.nu > 2.0
    assert fit.loc is not None
    assert fit.scale is not None and fit.scale > 0.0


def test_calibrate_metrics_include_condition_number(synthetic_returns) -> None:  # type: ignore[no-untyped-def]
    inp = _build_inputs(synthetic_returns, lookback_N=1000)
    metrics = calibrate(inp).fit_metrics
    assert "sigma_p" in metrics
    assert "cov_condition_number" in metrics
    assert metrics["n_obs"] == 1000.0
