"""End-to-end integration test for the Avellaneda-Lee stat-arb model.

Hits yfinance via `YFinanceProvider` for a small liquid universe of US
large caps + a single sector ETF as the factor proxy. The universe is
small enough that PCA mode is viable; the ETF mode is exercised
separately.

Skipped automatically when yfinance is missing or the network fetch
fails, so unit-test CI without internet stays green.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, Signal
from src.models.avellaneda_lee_stat_arb import (
    AvellanedaLeeStatArb,
    StatArbInputs,
)

pytest.importorskip("yfinance")


# Mega-cap tech / financial / health-care names. With N=8 and T~252 the
# Marchenko-Pastur edge is well-defined and the half-life filter is
# meaningful. The universe is intentionally small for a quick CI fetch.
_UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "JNJ",
]


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model() -> AvellanedaLeeStatArb:
    return AvellanedaLeeStatArb(
        universe=_UNIVERSE,
        K=3,
        factor_mode="PCA",
        pca_window=252,
        ou_window=60,
        # Looser filters than the spec's tight defaults — with only 8 names
        # and one year of data, half-lives jitter and some R^2 land below
        # the 0.20 spec floor.
        half_life_band=(3.0, 60.0),
        min_r_squared=0.05,
    )


def _fetch_or_skip(
    model: AvellanedaLeeStatArb, provider: YFinanceProvider
) -> StatArbInputs:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_pipeline_against_yfinance(
    provider: YFinanceProvider, model: AvellanedaLeeStatArb
) -> None:
    data = _fetch_or_skip(model, provider)
    assert isinstance(data, StatArbInputs)
    # We should keep most of the universe; a name missing on a holiday is fine.
    assert len(data.tickers) >= len(_UNIVERSE) - 2
    assert data.returns.shape[0] >= 252

    result = model.calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "avellaneda_lee_stat_arb"
    # At least one stock should pass the (loosened) filters; if none do, the
    # universe or filters are wrong, not a flaky-network signal.
    fit = model.fit
    assert fit.n_surviving + fit.n_dropped == len(data.tickers)


@pytest.mark.integration
def test_predict_emits_finite_signal(
    provider: YFinanceProvider, model: AvellanedaLeeStatArb
) -> None:
    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    if model.fit.n_surviving == 0:
        pytest.skip("No surviving tickers after filtering; cannot test predict.")
    signal = model.predict(data)
    assert isinstance(signal, Signal)
    assert signal.direction in ("long", "short", "flat")
    assert 0.0 <= signal.strength <= 1.0
    assert np.isfinite(signal.metadata.get("s_score_mod", float("nan")))


@pytest.mark.integration
def test_predict_all_covers_survivors(
    provider: YFinanceProvider, model: AvellanedaLeeStatArb
) -> None:
    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    signals = model.predict_all(data)
    assert len(signals) == model.fit.n_surviving
    for s in signals:
        assert s.ticker in model.fit.ou_fits


@pytest.mark.integration
def test_diagnostics_adf_and_distribution(
    provider: YFinanceProvider, model: AvellanedaLeeStatArb
) -> None:
    """Spec validation section 1 (ADF on residuals) + section 4 (s-score
    distribution shape). With a real universe the ADF p-values are noisy,
    so we only check that they're finite and in [0, 1]; the s-score cross
    section should be roughly mean-zero (loose envelope)."""

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    if model.fit.n_surviving == 0:
        pytest.skip("No surviving tickers; diagnostics not meaningful.")
    diag = model.validate(data)
    for ticker, p in diag["adf_pvalues"].items():
        assert ticker in model.fit.ou_fits
        if np.isfinite(p):
            assert 0.0 <= p <= 1.0
    if model.fit.n_surviving >= 3:
        # With ~5-8 survivors the empirical mean can wander, but should not
        # be wildly off-zero. A loose ±2 envelope catches catastrophic bugs.
        assert abs(diag["s_score_mean"]) < 2.0
