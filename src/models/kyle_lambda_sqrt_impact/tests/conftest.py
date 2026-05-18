"""Shared synthetic fixtures for the Kyle-lambda / sqrt-impact unit tests.

Each fixture has a worked-out target so the corresponding test can pin a
formula against a known answer rather than an approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260518)


@pytest.fixture
def synthetic_daily_bars(rng: np.random.Generator) -> pd.DataFrame:
    """200 daily bars with returns = lambda * signed_volume + small noise.

    Construction:

    * ``Volume_t`` is drawn from a log-normal with mean ``~1e6``.
    * Trade-sign is independent of price (so the proxy ``sign(C - O)`` will
      recover it almost perfectly).
    * True ``lambda = 5e-9`` per share, so a one-million-share buy lifts
      price by 0.5% (well inside the calibrated range).
    """

    n = 200
    sigma_noise = 1e-4
    true_lambda = 5e-9
    volumes = rng.lognormal(mean=np.log(1.0e6), sigma=0.4, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    signed_volume = signs * volumes
    rets = true_lambda * signed_volume + rng.normal(0.0, sigma_noise, size=n)

    # Build OHLC so that sign(C - O) == signs[t] and Close path matches rets.
    closes = 100.0 * np.cumprod(np.exp(rets))
    # Open is set so sign(Close - Open) == signs[t]. A small offset of
    # 0.01% suffices and keeps spread realistic.
    opens = closes * (1.0 - signs * 0.0001)
    highs = np.maximum(closes, opens) * 1.0005
    lows = np.minimum(closes, opens) * 0.9995
    index = pd.date_range("2025-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=index,
    ).reset_index().rename(columns={"index": "Date"})


@pytest.fixture
def true_lambda() -> float:
    """Matching ``synthetic_daily_bars`` for tests that compare to ground truth."""

    return 5e-9


@pytest.fixture
def synthetic_intraday_bars() -> pd.DataFrame:
    """Two 78-bar trading days of 5-minute bars.

    Day 1: buy-biased — every bar has C > O and positive return.
    Day 2: sell-biased — every bar has C < O and negative return.

    Within each day the sign of ``C - O`` exactly matches the sign of the
    log return so the OLS slope is positive on both days.
    """

    timestamps_d1 = pd.date_range("2026-01-05 09:30", periods=78, freq="5min")
    timestamps_d2 = pd.date_range("2026-01-06 09:30", periods=78, freq="5min")

    p_d1 = 100.0 * (1.001 ** np.arange(78))
    o_d1 = p_d1 / np.exp(0.0005)  # ensures sign(C - O) > 0
    p_d2 = 100.0 * (0.999 ** np.arange(78))
    o_d2 = p_d2 * np.exp(0.0005)  # ensures sign(C - O) < 0

    closes = np.concatenate([p_d1, p_d2])
    opens = np.concatenate([o_d1, o_d2])
    highs = np.maximum(closes, opens) * 1.0001
    lows = np.minimum(closes, opens) * 0.9999
    # Volume: 10_000 shares per bar on each day so the lambda is well-defined.
    vols = np.full(156, 10_000.0)
    timestamps = timestamps_d1.append(timestamps_d2)
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": vols,
        },
        index=timestamps,
    ).reset_index().rename(columns={"index": "Datetime"})
