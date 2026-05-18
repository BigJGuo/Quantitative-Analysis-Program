"""Orchestration tests for `AvellanedaLeeStatArb`.

Uses an `InMemoryProvider` populated with synthetic price series so the
fetch -> calibrate -> predict -> validate pipeline runs without network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.core.registry import get_model
from src.core.types import CalibrationResult, Signal
from src.models.avellaneda_lee_stat_arb import (
    AvellanedaLeeStatArb,
    SignalThresholds,
)
from src.models.avellaneda_lee_stat_arb.model import _period_from_window
from src.models.avellaneda_lee_stat_arb.signal import (
    POSITION_LONG,
    POSITION_SHORT,
)
from src.models.avellaneda_lee_stat_arb.types import StatArbFit, StatArbInputs

# Match the model fixture below (pca_window=252).
_PERIOD: str = _period_from_window(252)


def _make_prices_from_returns(
    asset_returns: pd.DataFrame,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Build per-ticker yfinance-shaped frames from a returns panel."""

    log_prices = np.log(100.0) + asset_returns.cumsum(axis=0)
    prices = np.exp(log_prices)
    out: dict[tuple[str, str, str], pd.DataFrame] = {}
    dates = asset_returns.index
    for ticker in asset_returns.columns:
        col = prices[ticker]
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": col.values,
                "High": col.values * 1.001,
                "Low": col.values * 0.999,
                "Close": col.values,
                "Volume": np.full(len(dates), 1_000_000, dtype=int),
            }
        )
        out[(ticker, _PERIOD, "1d")] = df
    return out


@pytest.fixture
def in_memory_provider(
    synthetic_panel: tuple[pd.DataFrame, pd.DataFrame],
) -> InMemoryProvider:
    asset_returns, factor_returns = synthetic_panel
    # Treat the synthetic factor as a single sector ETF for ETF-mode tests.
    factor_returns = factor_returns.rename(columns={"MKT": "XLK"})
    prices: dict[tuple[str, str, str], pd.DataFrame] = {}
    prices.update(_make_prices_from_returns(asset_returns))
    prices.update(_make_prices_from_returns(factor_returns))
    return InMemoryProvider(prices=prices)


class TestAvellanedaLeeStatArbClass:
    def test_registered_in_global_registry(self) -> None:
        assert get_model("avellaneda_lee_stat_arb") is AvellanedaLeeStatArb

    def test_base_model_class_attrs(self) -> None:
        assert AvellanedaLeeStatArb.name == "avellaneda_lee_stat_arb"
        assert AvellanedaLeeStatArb.layer == 4
        assert AvellanedaLeeStatArb.refit_frequency == "daily"

    def test_constructor_rejects_small_universe(self) -> None:
        with pytest.raises(ValueError, match="at least 2 tickers"):
            AvellanedaLeeStatArb(universe=["AAPL"])

    def test_constructor_rejects_bad_K(self) -> None:
        with pytest.raises(ValueError):
            AvellanedaLeeStatArb(universe=["AAA", "BBB"], K=0)

    def test_constructor_rejects_ou_window_too_short(self) -> None:
        with pytest.raises(ValueError):
            AvellanedaLeeStatArb(universe=["AAA", "BBB"], ou_window=5)

    def test_constructor_rejects_ou_window_exceeding_pca(self) -> None:
        with pytest.raises(ValueError):
            AvellanedaLeeStatArb(
                universe=["AAA", "BBB"], pca_window=60, ou_window=120
            )


class TestPipelinePCAMode:
    """fetch -> calibrate -> predict -> validate, PCA factor mode."""

    @pytest.fixture
    def model(
        self, synthetic_panel: tuple[pd.DataFrame, pd.DataFrame]
    ) -> AvellanedaLeeStatArb:
        asset_returns, _ = synthetic_panel
        return AvellanedaLeeStatArb(
            universe=list(asset_returns.columns),
            K=2,
            factor_mode="PCA",
            pca_window=252,
            ou_window=60,
            half_life_band=(2.0, 60.0),
            min_r_squared=0.1,
        )

    def test_fetch_data_builds_panel(
        self, model: AvellanedaLeeStatArb, in_memory_provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(in_memory_provider)
        assert isinstance(data, StatArbInputs)
        assert data.factor_mode == "PCA"
        assert data.factor_returns is None
        assert data.returns.shape[0] >= 200

    def test_calibrate_returns_calibration_result(
        self, model: AvellanedaLeeStatArb, in_memory_provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(in_memory_provider)
        result = model.calibrate(data)
        assert isinstance(result, CalibrationResult)
        assert result.model_name == "avellaneda_lee_stat_arb"
        assert result.fit_metrics["n_surviving"] > 0
        # Stored on the model for later predict/validate calls.
        assert isinstance(model.fit, StatArbFit)

    def test_predict_emits_signal_for_focal_ticker(
        self, model: AvellanedaLeeStatArb, in_memory_provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(in_memory_provider)
        model.calibrate(data)
        signal = model.predict(data)
        assert isinstance(signal, Signal)
        # Focal ticker is the first surviving universe member.
        assert signal.ticker in model.universe
        assert signal.direction in ("long", "short", "flat")
        assert 0.0 <= signal.strength <= 1.0
        # Metadata must surface the s-scores and beta.
        assert "s_score" in signal.metadata
        assert "s_score_mod" in signal.metadata
        assert "beta" in signal.metadata

    def test_predict_all_covers_survivors(
        self, model: AvellanedaLeeStatArb, in_memory_provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(in_memory_provider)
        model.calibrate(data)
        signals = model.predict_all(data)
        assert len(signals) == model.fit.n_surviving
        tickers = {s.ticker for s in signals}
        assert tickers == set(model.fit.ou_fits.keys())

    def test_validate_returns_diagnostics(
        self, model: AvellanedaLeeStatArb, in_memory_provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(in_memory_provider)
        model.calibrate(data)
        diag = model.validate(data)
        # Spec validation section 1 - ADF p-values present.
        assert "adf_pvalues" in diag
        assert len(diag["adf_pvalues"]) == model.fit.n_surviving
        # Validation section 4 - s-score distribution stats.
        assert "s_score_mean" in diag
        assert "s_score_std" in diag
        # Validation section 3 - half-life distribution stats.
        assert "half_life_mean" in diag
        assert "half_life_median" in diag

    def test_validate_includes_pca_diagnostics(
        self, model: AvellanedaLeeStatArb, in_memory_provider: InMemoryProvider
    ) -> None:
        data = model.fetch_data(in_memory_provider)
        model.calibrate(data)
        diag = model.validate(data)
        assert "mp_upper_edge" in diag
        assert diag["mp_upper_edge"] > 0
        assert "variance_explained_top_K" in diag


class TestPositionStateMachine:
    """Predict updates the cached prior-position dict; subsequent predictions
    should respect the open/close state machine."""

    def test_extreme_s_score_opens_short(self) -> None:
        model = AvellanedaLeeStatArb(
            universe=["AAA", "BBB"],
            K=1,
            factor_mode="PCA",
            pca_window=60,
            ou_window=30,
            thresholds=SignalThresholds(),
        )
        # Manually inject a fit so we can drive predict_all without I/O.
        from src.models.avellaneda_lee_stat_arb.types import OUFit, StatArbFit

        ou = OUFit(
            ticker="AAA",
            alpha=0.0,
            beta=np.array([1.0]),
            r_squared=0.5,
            a=0.0,
            b=0.9,
            kappa=0.105,
            m=0.0,
            sigma=0.01,
            sigma_zeta=0.01,
            sigma_eq=0.022,
            half_life=6.6,
            X_last=2.0,
            s_score=2.5,  # extreme positive -> open short
            s_score_mod=2.5,
        )
        fit = StatArbFit(
            factor_mode="PCA",
            factor_names=("PC1",),
            universe=("AAA", "BBB"),
            ou_fits={"AAA": ou},
            dropped_tickers={"BBB": "low_r_squared"},
            factor_returns=pd.DataFrame({"PC1": [0.0]}),
            eigenvalues=None,
            marchenko_pastur_cutoff=None,
            half_life_band=(5.0, 30.0),
            min_r_squared=0.2,
            timestamp=pd.Timestamp("2026-05-18").to_pydatetime(),
        )
        model._fit = fit  # noqa: SLF001 — test wires the fit in directly

        # Build a minimal StatArbInputs (predict only reads timestamp + type).
        inputs = StatArbInputs(
            returns=pd.DataFrame({"AAA": [0.0]}, index=[pd.Timestamp("2026-05-18")]),
            tickers=("AAA",),
            factor_mode="PCA",
            K=1,
            pca_window=60,
            ou_window=30,
            timestamp=pd.Timestamp("2026-05-18").to_pydatetime(),
        )
        first = model.predict(inputs)
        assert first.ticker == "AAA"
        assert first.direction == "short"
        assert model._prev_positions["AAA"] == POSITION_SHORT  # noqa: SLF001

    def test_factor_exposure_aggregates_positions(self) -> None:
        model = AvellanedaLeeStatArb(
            universe=["AAA", "BBB"],
            K=1,
            factor_mode="PCA",
            pca_window=60,
            ou_window=30,
        )
        from src.models.avellaneda_lee_stat_arb.types import OUFit, StatArbFit

        ou_a = OUFit(
            ticker="AAA",
            alpha=0.0,
            beta=np.array([1.0]),
            r_squared=0.5,
            a=0.0,
            b=0.9,
            kappa=0.105,
            m=0.0,
            sigma=0.01,
            sigma_zeta=0.01,
            sigma_eq=0.022,
            half_life=6.6,
            X_last=0.0,
            s_score=0.0,
            s_score_mod=0.0,
        )
        ou_b = OUFit(
            ticker="BBB",
            alpha=0.0,
            beta=np.array([0.5]),
            r_squared=0.5,
            a=0.0,
            b=0.9,
            kappa=0.105,
            m=0.0,
            sigma=0.01,
            sigma_zeta=0.01,
            sigma_eq=0.022,
            half_life=6.6,
            X_last=0.0,
            s_score=0.0,
            s_score_mod=0.0,
        )
        model._fit = StatArbFit(  # noqa: SLF001
            factor_mode="PCA",
            factor_names=("PC1",),
            universe=("AAA", "BBB"),
            ou_fits={"AAA": ou_a, "BBB": ou_b},
            dropped_tickers={},
            factor_returns=pd.DataFrame({"PC1": [0.0]}),
            eigenvalues=None,
            marchenko_pastur_cutoff=None,
            half_life_band=(5.0, 30.0),
            min_r_squared=0.2,
            timestamp=pd.Timestamp("2026-05-18").to_pydatetime(),
        )
        model._prev_positions = {"AAA": POSITION_LONG, "BBB": POSITION_SHORT}  # noqa: SLF001
        exposure = model.aggregate_factor_exposure()
        np.testing.assert_allclose(exposure, np.array([1.0 - 0.5]))

    def test_predict_uncalibrated_raises(self) -> None:
        model = AvellanedaLeeStatArb(
            universe=["AAA", "BBB"], K=1, pca_window=60, ou_window=30
        )
        inputs = StatArbInputs(
            returns=pd.DataFrame({"AAA": [0.0], "BBB": [0.0]}),
            tickers=("AAA", "BBB"),
            factor_mode="PCA",
            K=1,
            pca_window=60,
            ou_window=30,
            timestamp=pd.Timestamp("2026-05-18").to_pydatetime(),
        )
        with pytest.raises(RuntimeError, match="not been calibrated"):
            model.predict(inputs)
