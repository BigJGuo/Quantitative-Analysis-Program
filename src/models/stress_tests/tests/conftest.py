"""Shared synthetic fixtures for the stress-tests unit suite.

The fixtures construct a tiny universe of positions plus an `InMemoryProvider`
populated with daily-close price panels that cover the spec's canonical crisis
windows, so the end-to-end pipeline runs without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.data_provider import InMemoryProvider
from src.models.stress_tests.types import (
    DEFAULT_FACTORS,
    CrisisWindow,
    HypotheticalScenario,
)

UNIVERSE: list[str] = ["AAA", "BBB", "CCC"]
PROXIES: list[str] = ["XLK", "XLF"]
BENCHMARK: str = "^GSPC"


def _synthetic_close_panel(
    *,
    tickers: list[str],
    start: str = "1985-01-01",
    end: str = "2024-12-31",
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Build a yfinance-shaped daily close panel for every ticker.

    Returns a dict mapping ticker -> DataFrame with the same columns the
    `YFinanceProvider` produces after `.reset_index()`: ``Date``, ``Open``,
    ``High``, ``Low``, ``Close``, ``Volume``.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    out: dict[str, pd.DataFrame] = {}
    # Common market factor injected into every ticker for predictable beta.
    market_log_returns = rng.normal(0.0003, 0.012, size=n)
    for i, ticker in enumerate(tickers):
        beta = 1.0 + 0.1 * i
        idio = rng.normal(0.0, 0.005, size=n)
        log_r = beta * market_log_returns + idio
        log_p = np.log(100.0) + log_r.cumsum()
        close = np.exp(log_p)
        out[ticker] = pd.DataFrame(
            {
                "Date": dates,
                "Open": close,
                "High": close * 1.002,
                "Low": close * 0.998,
                "Close": close,
                "Volume": np.full(n, 1_000_000, dtype=int),
            }
        )
    return out


@pytest.fixture
def positions() -> dict[str, float]:
    return {"AAA": 100_000.0, "BBB": -50_000.0, "CCC": 75_000.0}


@pytest.fixture
def capital() -> float:
    return 1_000_000.0


@pytest.fixture
def crisis_windows() -> tuple[CrisisWindow, ...]:
    # Use 1990+ windows so the synthetic panels (which start in 1985) cover them.
    return (
        CrisisWindow(name="LTCM 1998", t1="1998-08-17", t2="1998-10-08"),
        CrisisWindow(name="Lehman 2008", t1="2008-09-12", t2="2008-10-10"),
        CrisisWindow(name="COVID crash", t1="2020-02-19", t2="2020-03-23"),
    )


@pytest.fixture
def hypotheticals() -> tuple[HypotheticalScenario, ...]:
    return (
        HypotheticalScenario(name="Equity -20%", shocks={"equity": -0.20}),
        HypotheticalScenario(name="Combined", shocks={"equity": -0.10, "rates": 0.01}),
    )


@pytest.fixture
def provider() -> InMemoryProvider:
    all_tickers = UNIVERSE + PROXIES + [BENCHMARK] + list(DEFAULT_FACTORS.values())
    raw = _synthetic_close_panel(tickers=all_tickers)
    prices: dict[tuple[str, str, str], pd.DataFrame] = {}
    for ticker, df in raw.items():
        prices[(ticker, "max", "1d")] = df
    return InMemoryProvider(prices=prices)


@pytest.fixture
def sector_map() -> dict[str, str]:
    return {"AAA": "XLK", "BBB": "XLF"}


@pytest.fixture
def _synthetic_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint asset / factor panels built from the same underlying market path.

    Returning both ensures the OLS path can recover the planted betas exactly;
    splitting them across two fixtures (with different seeds) would defeat the
    test.
    """

    rng = np.random.default_rng(42)
    n = 600
    market = rng.normal(0.0, 0.012, size=n)
    rates = rng.normal(0.0, 0.0008, size=n)
    true_betas = np.array([0.6, 1.0, 1.4])
    eps = rng.normal(0.0, 0.002, size=(n, 3))
    R = market[:, None] * true_betas + eps
    index = pd.date_range("2022-01-01", periods=n, freq="B")
    asset = pd.DataFrame(R, index=index, columns=["AAA", "BBB", "CCC"])
    factor = pd.DataFrame({"equity": market, "rates": rates}, index=index)
    return asset, factor


@pytest.fixture
def asset_returns(_synthetic_returns: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return _synthetic_returns[0]


@pytest.fixture
def factor_returns(_synthetic_returns: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return _synthetic_returns[1]
