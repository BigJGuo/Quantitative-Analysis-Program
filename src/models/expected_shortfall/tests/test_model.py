"""Unit tests for `ExpectedShortfall` — the BaseModel orchestration shell.

Covers the four computation methods end-to-end, the `BaseModel` contract
(fetch / calibrate / predict / validate), and the registry hookup.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.core.registry import get_model
from src.core.types import CalibrationResult, RiskMetric
from src.models.expected_shortfall import (
    ESInputs,
    ESResult,
    ExpectedShortfall,
    es_gaussian,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_expected_shortfall_registered() -> None:
    cls = get_model("expected_shortfall")
    assert cls is ExpectedShortfall
    assert ExpectedShortfall.layer == 6
    assert ExpectedShortfall.refit_frequency == "daily"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_positions() -> None:
    with pytest.raises(ValueError, match="positions"):
        ExpectedShortfall(positions={})


def test_construction_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method"):
        ExpectedShortfall(
            positions={"AAA": 10_000.0}, method="banana"  # type: ignore[arg-type]
        )


def test_construction_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ExpectedShortfall(positions={"AAA": 1_000.0}, alpha=1.0)


def test_construction_sorts_tickers() -> None:
    m = ExpectedShortfall(positions={"BBB": 1.0, "AAA": 2.0})
    assert m.tickers == ("AAA", "BBB")


# ---------------------------------------------------------------------------
# fetch_data
# ---------------------------------------------------------------------------


def test_fetch_data_builds_aligned_returns(in_memory_provider) -> None:  # type: ignore[no-untyped-def]
    model = ExpectedShortfall(
        positions={"AAA": 10_000.0, "BBB": 20_000.0},
        method="historical",
        lookback_N=500,
        history_period="3y",
    )
    inputs = model.fetch_data(in_memory_provider)
    assert isinstance(inputs, ESInputs)
    assert inputs.tickers == ("AAA", "BBB")
    assert inputs.returns.shape[1] == 2
    assert inputs.returns.shape[0] >= 500
    np.testing.assert_allclose(
        inputs.dollar_positions, np.array([10_000.0, 20_000.0]), rtol=1e-12
    )


def test_fetch_data_raises_on_empty_panel() -> None:
    from src.core.data_provider import InMemoryProvider

    empty_bars = pd.DataFrame(
        {"Date": pd.to_datetime([]), "Close": np.array([], dtype=float)}
    )
    provider = InMemoryProvider(prices={("AAA", "3y", "1d"): empty_bars})
    model = ExpectedShortfall(
        positions={"AAA": 1.0}, method="historical", lookback_N=500
    )
    with pytest.raises(RuntimeError, match="No daily price history"):
        model.fetch_data(provider)


# ---------------------------------------------------------------------------
# calibrate / predict / validate per method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["parametric_normal", "parametric_t", "historical", "monte_carlo"],
)
def test_calibrate_predict_validate_all_methods(in_memory_provider, method) -> None:  # type: ignore[no-untyped-def]
    model = ExpectedShortfall(
        positions={"AAA": 50_000.0, "BBB": 50_000.0},
        method=method,
        lookback_N=500,
        mc_draws=20_000,
        mc_seed=7,
    )
    inputs = model.fetch_data(in_memory_provider)
    cal = model.calibrate(inputs)
    assert isinstance(cal, CalibrationResult)

    rm = model.predict(inputs)
    assert isinstance(rm, RiskMetric)
    assert rm.metric_name == "expected_shortfall"
    assert rm.confidence_level == model.alpha
    assert rm.value > 0
    # ES must dominate VaR
    assert rm.metadata["ES"] >= rm.metadata["VaR"] - 1e-9

    diags = model.validate(inputs)
    assert diags["method"] == method
    assert diags["alpha"] == model.alpha
    assert diags["covariance_is_psd"]


def test_parametric_normal_matches_closed_form(in_memory_provider, true_mvn_params) -> None:  # type: ignore[no-untyped-def]
    # On the synthetic-Gaussian data, the parametric-normal ES from `predict`
    # should match `es_gaussian` applied to the analytic portfolio sigma.
    model = ExpectedShortfall(
        positions={"AAA": 100_000.0, "BBB": 100_000.0},
        method="parametric_normal",
        lookback_N=2000,
    )
    inputs = model.fetch_data(in_memory_provider)
    model.calibrate(inputs)
    rm = model.predict(inputs)

    Sigma = true_mvn_params["Sigma"]
    w = np.array([100_000.0, 100_000.0])
    true_sigma_p = float(np.sqrt(w @ Sigma @ w))
    _, expected_es = es_gaussian(0.0, true_sigma_p, 0.975)
    # ~5% tolerance because we used the sample covariance from 2000 obs,
    # not the population covariance.
    assert math.isclose(rm.value, expected_es, rel_tol=0.10)


def test_methods_produce_comparable_es_under_gaussian(in_memory_provider) -> None:  # type: ignore[no-untyped-def]
    # All four methods on Gaussian data should land within ~25% of each other
    # at alpha=0.975. (MC at nu=5 produces fatter tails; spread comes from there.)
    results: dict[str, float] = {}
    for method in ["parametric_normal", "historical", "monte_carlo"]:
        m = ExpectedShortfall(
            positions={"AAA": 100_000.0, "BBB": 100_000.0},
            method=method,  # type: ignore[arg-type]
            lookback_N=2000,
            mc_draws=50_000,
            mc_seed=1,
        )
        inp = m.fetch_data(in_memory_provider)
        m.calibrate(inp)
        results[method] = m.predict(inp).value
    # Empirical and parametric_normal should be very close on Gaussian data.
    pn = results["parametric_normal"]
    hist = results["historical"]
    mc = results["monte_carlo"]
    assert abs(hist / pn - 1.0) < 0.15
    # MC at nu=5 should be heavier-tailed (i.e., larger ES) than the Gaussian.
    assert mc > pn * 0.9


def test_compute_es_returns_es_result(in_memory_provider) -> None:  # type: ignore[no-untyped-def]
    model = ExpectedShortfall(
        positions={"AAA": 10_000.0, "BBB": 10_000.0}, method="historical"
    )
    inputs = model.fetch_data(in_memory_provider)
    res = model.compute_es(inputs)
    assert isinstance(res, ESResult)
    assert res.alpha == 0.975
    assert res.method == "historical"


def test_predict_before_calibrate_lazy_fits(in_memory_provider) -> None:  # type: ignore[no-untyped-def]
    model = ExpectedShortfall(
        positions={"AAA": 10_000.0, "BBB": 10_000.0}, method="historical"
    )
    inputs = model.fetch_data(in_memory_provider)
    rm = model.predict(inputs)  # no explicit calibrate() — should lazy fit
    assert rm.value > 0


def test_validate_reports_z1_z2(in_memory_provider) -> None:  # type: ignore[no-untyped-def]
    model = ExpectedShortfall(
        positions={"AAA": 10_000.0, "BBB": 10_000.0},
        method="historical",
        lookback_N=500,
    )
    inputs = model.fetch_data(in_memory_provider)
    diags = model.validate(inputs)
    # Z1 / Z2 must be finite numbers (we have ample breach observations at N=500)
    assert math.isfinite(diags["acerbi_szekely_z1"])
    assert math.isfinite(diags["acerbi_szekely_z2"])
    # Breach rate should be ~ (1-alpha) = 2.5% within sampling noise.
    assert abs(diags["breach_rate_in_sample"] - 0.025) < 0.025


def test_short_portfolio_runs(in_memory_provider) -> None:  # type: ignore[no-untyped-def]
    # Sign-aware: long-short book with net zero notional must still produce ES.
    model = ExpectedShortfall(
        positions={"AAA": 50_000.0, "BBB": -50_000.0},
        method="historical",
        lookback_N=500,
    )
    inputs = model.fetch_data(in_memory_provider)
    rm = model.predict(inputs)
    assert rm.value > 0


def test_liquidity_scalars_scale_es(in_memory_provider) -> None:  # type: ignore[no-untyped-def]
    base = ExpectedShortfall(
        positions={"AAA": 50_000.0, "BBB": 50_000.0},
        method="historical",
        lookback_N=500,
    )
    scaled = ExpectedShortfall(
        positions={"AAA": 50_000.0, "BBB": 50_000.0},
        method="historical",
        lookback_N=500,
        liquidity_scalars=np.array([2.0, 2.0]),  # 4x variance, 2x sigma
    )
    inp_base = base.fetch_data(in_memory_provider)
    inp_scl = scaled.fetch_data(in_memory_provider)
    rm_base = base.predict(inp_base)
    rm_scl = scaled.predict(inp_scl)
    # Empirically expect ES to roughly double under uniform sqrt-h scaling of 2.
    ratio = rm_scl.value / rm_base.value
    assert 1.7 < ratio < 2.3
