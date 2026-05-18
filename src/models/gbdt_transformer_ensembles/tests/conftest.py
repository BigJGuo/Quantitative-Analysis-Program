"""Synthetic fixtures for the GBDT + Transformer ensemble tests.

We generate a tractable but non-trivial price panel: 6 tickers, ~400 daily
business days, with a known latent factor + ticker-specific noise so the
ensemble has a real signal to recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gbdt_transformer_ensembles.types import FeatureSpec, PanelFrame


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


@pytest.fixture
def small_feature_spec() -> FeatureSpec:
    """Reduced spec so tests run quickly."""

    return FeatureSpec(
        return_horizons=(1, 5, 21),
        lag_count=3,
        vol_windows=(5, 21),
        mean_windows=(5,),
        target_horizon=5,
        clip_target=0.2,
        include_cross_sectional_ranks=True,
        include_calendar=True,
        benchmark_ticker=None,
        min_history_days=63,
    )


@pytest.fixture
def synthetic_prices(
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Six tickers, 400 daily bars, one latent factor + per-ticker noise.

    Open == Close (we only need Close for tests); High/Low are Close +/- a
    small spread; Volume is exp(noise) so the log-volume feature has finite
    variance.
    """

    n_obs = 400
    n_assets = 6
    dates = pd.date_range("2024-01-02", periods=n_obs, freq="B")
    beta = rng.uniform(0.7, 1.3, size=n_assets)
    factor_vol = 0.01
    spec_vol = 0.005
    f = rng.normal(0.0, factor_vol, size=n_obs)
    eps = rng.normal(0.0, spec_vol, size=(n_obs, n_assets))
    r = f[:, None] * beta[None, :] + eps  # (T, N)
    log_p = np.log(100.0) + np.cumsum(r, axis=0)
    close = np.exp(log_p)
    spread = 0.002 * close
    volume = np.exp(rng.normal(13.0, 0.5, size=(n_obs, n_assets)))

    out: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        ticker = f"T{i}"
        df = pd.DataFrame(
            {
                "Open": close[:, i],
                "High": close[:, i] + spread[:, i],
                "Low": close[:, i] - spread[:, i],
                "Close": close[:, i],
                "Volume": volume[:, i],
            },
            index=dates,
        )
        out[ticker] = df
    return out


@pytest.fixture
def fitted_panel(
    synthetic_prices: dict[str, pd.DataFrame],
    small_feature_spec: FeatureSpec,
) -> PanelFrame:
    """Pre-built `PanelFrame` for tests that need calibrated data."""

    from src.models.gbdt_transformer_ensembles.features import build_feature_panel

    return build_feature_panel(synthetic_prices, spec=small_feature_spec)
