"""End-to-end integration test for the factor / PCA risk model.

Hits yfinance via `YFinanceProvider` for a small universe of US sector ETFs.
Skipped automatically when yfinance is missing or the network fetch fails, so
unit-test CI without internet stays green.

The sector-ETF universe is deliberate: PC1 should explain the market and
land in the spec's "30-45% for the first PC" band, giving us a meaningful
sanity check beyond just "the pipeline runs".
"""

from __future__ import annotations

import numpy as np
import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import CalibrationResult, RiskMetric
from src.models.factor_models_pca import FactorModelPCA
from src.models.factor_models_pca.types import FactorInputs

pytest.importorskip("yfinance")


# 10 US sector ETFs span the broad market with reasonably independent specific
# risk, which is the textbook PCA fixture.
_UNIVERSE = [
    "XLK",  # tech
    "XLF",  # financials
    "XLE",  # energy
    "XLV",  # health care
    "XLY",  # consumer disc.
    "XLP",  # consumer staples
    "XLI",  # industrials
    "XLU",  # utilities
    "XLB",  # materials
    "XLRE",  # real estate
]


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model() -> FactorModelPCA:
    return FactorModelPCA(
        universe=_UNIVERSE,
        K=3,
        method="PCA",
        lookback_days=252,
        use_mp_cutoff=False,
        annualize=True,
    )


def _fetch_or_skip(model: FactorModelPCA, provider: YFinanceProvider) -> FactorInputs:
    try:
        return model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")


@pytest.mark.integration
def test_pipeline_against_yfinance(
    provider: YFinanceProvider, model: FactorModelPCA
) -> None:
    data = _fetch_or_skip(model, provider)
    assert isinstance(data, FactorInputs)
    # We should keep most of the universe; missing one ETF on a weekend pull is fine.
    assert len(data.tickers) >= len(_UNIVERSE) - 2
    assert data.returns.shape[0] > 100  # ~ 1 year of trading days

    result = model.calibrate(data)
    assert isinstance(result, CalibrationResult)
    assert result.model_name == "factor_models_pca"
    fit = model.fit
    assert fit.method == "PCA"
    assert fit.n_factors == 3

    signal = model.predict(data)
    assert isinstance(signal, RiskMetric)
    assert signal.ticker == "PORTFOLIO"
    assert signal.metric_name == "portfolio_volatility"
    assert signal.value > 0.0
    # Annualized portfolio vol for an equal-weight broad sector basket is
    # historically in the 10-40% range. A wider sanity envelope is fine here.
    assert 0.03 < signal.value < 0.80


@pytest.mark.integration
def test_pc1_is_market_like(
    provider: YFinanceProvider, model: FactorModelPCA
) -> None:
    """Spec validation step 1: first PC of an equity panel explains 30-45% of
    cross-sectional variance for US equities. Sector ETFs should sit comfortably
    inside that band (often higher than individual stocks because the names
    are themselves already diversified)."""

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    diag = model.validate(data)
    cumve = diag["cumulative_variance_explained"]
    assert cumve[0] > 0.30
    # 3 components on sector ETFs commonly clear 80% — give a loose floor.
    assert cumve[-1] > 0.60


@pytest.mark.integration
def test_marchenko_pastur_diagnostic(
    provider: YFinanceProvider, model: FactorModelPCA
) -> None:
    """Spec validation step 2: real factors sit above the MP upper edge.

    With N≈10 and T≈252, the MP edge is small relative to the top eigenvalue,
    so at least one PC should clear it.
    """

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    diag = model.validate(data)
    assert diag["mp_upper_edge"] > 0.0
    assert diag["n_factors_above_mp"] >= 1
    assert diag["top_eigenvalue"] > diag["mp_upper_edge"]


@pytest.mark.integration
def test_residualization_strips_factor_beta(
    provider: YFinanceProvider, model: FactorModelPCA
) -> None:
    """Spec algorithm step 7: residualized signal is orthogonal to the loadings."""

    data = _fetch_or_skip(model, provider)
    model.calibrate(data)
    raw_signal = np.random.default_rng(0).normal(size=model.fit.n_assets)
    neutral = model.residualize_signal(raw_signal)
    proj = model.fit.B.T @ neutral
    assert np.allclose(proj, 0.0, atol=1e-8)
