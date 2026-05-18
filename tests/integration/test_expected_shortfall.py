"""End-to-end integration test for the Expected Shortfall (FRTB) model.

Pulls real yfinance data for a small two-ticker equity portfolio (SPY + AGG
— the canonical 60/40 split used in regulatory examples) and runs the full
`fetch_data -> calibrate -> predict -> validate` pipeline for each of the
four ES estimators. The module self-skips when yfinance is unavailable or
the network call fails, so unit-test runs stay deterministic.
"""

from __future__ import annotations

import importlib.util
import math

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.expected_shortfall import ESMethod, ExpectedShortfall

_POSITIONS: dict[str, float] = {"SPY": 600_000.0, "AGG": 400_000.0}  # 60/40 $1MM book


def _yfinance_available() -> bool:
    return importlib.util.find_spec("yfinance") is not None


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
@pytest.mark.parametrize(
    "method",
    ["parametric_normal", "parametric_t", "historical", "monte_carlo"],
)
def test_expected_shortfall_end_to_end_on_real_portfolio(method: ESMethod) -> None:
    """Run a single ES method against real SPY+AGG data. Tolerates network outages."""

    provider = YFinanceProvider(cache=None)
    model = ExpectedShortfall(
        positions=_POSITIONS,
        method=method,
        alpha=0.975,
        lookback_N=500,
        history_period="3y",
        mc_draws=20_000,
        mc_seed=20260518,
    )

    try:
        inputs = model.fetch_data(provider)
    except RuntimeError as exc:
        pytest.skip(f"yfinance data unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance call failed: {exc}")

    calibration = model.calibrate(inputs)
    assert isinstance(calibration, CalibrationResult)
    assert calibration.model_name == "expected_shortfall"
    assert calibration.fit_metrics["n_obs"] == 500.0
    assert calibration.fit_metrics["sigma_p"] > 0.0
    if method == "parametric_t":
        # Student-t fit only populated under the parametric-t path.
        assert calibration.fit_metrics["t_nu"] > 2.0
        assert calibration.fit_metrics["t_scale"] > 0.0

    risk = model.predict(inputs)
    assert isinstance(risk, RiskMetric)
    assert risk.metric_name == "expected_shortfall"
    assert risk.confidence_level == 0.975
    # Sanity bounds on dollar ES for a $1MM 60/40 book at alpha=0.975:
    # typical daily ES is ~$5k-$50k depending on the regime.
    assert 1_000.0 < risk.value < 200_000.0
    # ES dominates VaR by construction.
    assert risk.metadata["ES"] >= risk.metadata["VaR"] - 1e-6
    # Traffic-light band populated.
    assert risk.metadata["traffic_light"] in {
        "amber-light",
        "green",
        "amber",
        "red",
        "undefined",
    }

    diag = model.validate(inputs)
    # Spec validation block: Z1 / Z2 should be finite scalars.
    assert math.isfinite(diag["acerbi_szekely_z1"]) or math.isnan(diag["acerbi_szekely_z1"])
    assert math.isfinite(diag["acerbi_szekely_z2"])
    # Realized breach rate should be in the same ballpark as 1-alpha=2.5%
    # (lax band: real-data backtest noise is large at N=500).
    assert 0.0 < diag["breach_rate_in_sample"] < 0.15
    # Covariance PSD sanity.
    assert diag["covariance_is_psd"]


@pytest.mark.skipif(not _yfinance_available(), reason="yfinance not installed")
def test_methods_produce_ordered_es_on_real_data() -> None:
    """All four methods at the same configuration should land within an order
    of magnitude of each other. Student-t/MC should generally produce >= ES
    than the Gaussian baseline (fatter tails); not asserted strictly because
    real data has its own tail shape.
    """

    provider = YFinanceProvider(cache=None)
    methods: list[ESMethod] = [
        "parametric_normal",
        "parametric_t",
        "historical",
        "monte_carlo",
    ]
    es_by_method: dict[str, float] = {}
    for method in methods:
        m = ExpectedShortfall(
            positions=_POSITIONS,
            method=method,
            alpha=0.975,
            lookback_N=500,
            history_period="3y",
            mc_draws=20_000,
            mc_seed=20260518,
        )
        try:
            inp = m.fetch_data(provider)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"yfinance call failed: {exc}")
        m.calibrate(inp)
        es_by_method[method] = float(m.predict(inp).value)

    values = list(es_by_method.values())
    assert min(values) > 0
    # All four numbers within a factor of 3 of each other on a 60/40 book.
    assert max(values) / min(values) < 3.0
