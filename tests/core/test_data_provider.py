"""Tests for the data provider abstraction and disk cache.

Network access is never required — `YFinanceProvider` is exercised by stubbing
out `_import_yf` so we never hit the real API.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from src.core.data_provider import (
    CacheConfig,
    DataProvider,
    DiskCache,
    InMemoryProvider,
    YFinanceProvider,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1_000, 1_100, 1_200],
        },
        index=idx,
    )


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(
        CacheConfig(
            cache_dir=tmp_path / "cache",
            default_ttl_seconds=60,
            per_method_ttl_seconds={"fetch_intraday": 5},
        )
    )


class TestDiskCache:
    def test_round_trip_parquet(self, cache: DiskCache, sample_ohlcv: pd.DataFrame) -> None:
        calls = {"n": 0}

        def compute() -> pd.DataFrame:
            calls["n"] += 1
            return sample_ohlcv

        first = cache.get_or_compute("fetch_prices", "AAPL|1y|1d", "parquet", compute)
        second = cache.get_or_compute("fetch_prices", "AAPL|1y|1d", "parquet", compute)

        assert calls["n"] == 1
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True), second.reset_index(drop=True)
        )

    def test_round_trip_json(self, cache: DiskCache) -> None:
        value = {"sector": "Tech", "marketCap": 1_000_000}
        out1 = cache.get_or_compute("fetch_fundamentals", "AAPL", "json", lambda: value)
        out2 = cache.get_or_compute(
            "fetch_fundamentals", "AAPL", "json", lambda: {"should_not_run": True}
        )
        assert out1 == value
        assert out2 == value

    def test_round_trip_pickle(self, cache: DiskCache) -> None:
        series = pd.Series([0.5, 0.6], index=pd.to_datetime(["2024-01-01", "2024-04-01"]))
        out1 = cache.get_or_compute("fetch_dividends", "AAPL", "pickle", lambda: series)
        out2 = cache.get_or_compute(
            "fetch_dividends", "AAPL", "pickle", lambda: pd.Series([999.0])
        )
        pd.testing.assert_series_equal(out1, out2)

    def test_ttl_expiry_recomputes(self, tmp_path: Path) -> None:
        cache = DiskCache(
            CacheConfig(
                cache_dir=tmp_path / "c",
                default_ttl_seconds=60,
                per_method_ttl_seconds={"fetch_intraday": 0},
            )
        )
        calls = {"n": 0}

        def compute() -> dict[str, int]:
            calls["n"] += 1
            return {"v": calls["n"]}

        cache.get_or_compute("fetch_intraday", "AAPL|1m|1", "json", compute)
        cache.get_or_compute("fetch_intraday", "AAPL|1m|1", "json", compute)
        assert calls["n"] == 2  # TTL of 0 forces recompute every call

    def test_per_method_ttl_independent(self, cache: DiskCache) -> None:
        # `fetch_intraday` has TTL 5s, `fetch_prices` has the default 60s. We don't
        # wait — we just confirm separate methods don't share cache slots.
        cache.get_or_compute("fetch_intraday", "AAPL", "json", lambda: {"x": 1})
        cache.get_or_compute("fetch_prices", "AAPL", "json", lambda: {"x": 2})

        assert cache.get_or_compute(
            "fetch_intraday", "AAPL", "json", lambda: {"should_not_run": True}
        ) == {"x": 1}
        assert cache.get_or_compute(
            "fetch_prices", "AAPL", "json", lambda: {"should_not_run": True}
        ) == {"x": 2}


class TestInMemoryProvider:
    def test_round_trip_prices(self, sample_ohlcv: pd.DataFrame) -> None:
        provider = InMemoryProvider(prices={("AAPL", "1y", "1d"): sample_ohlcv})
        out = provider.fetch_prices("AAPL", "1y", "1d")
        pd.testing.assert_frame_equal(out, sample_ohlcv)

    def test_news_respects_max_items(self) -> None:
        items = [{"title": f"n{i}", "published": "2024-01-01"} for i in range(30)]
        provider = InMemoryProvider(news={"AAPL": items})
        assert len(provider.fetch_news("AAPL", max_items=5)) == 5
        assert provider.fetch_news("UNKNOWN") == []

    def test_dividends_and_holdings_and_fundamentals(self) -> None:
        provider = InMemoryProvider(
            dividends={"AAPL": pd.Series([0.2, 0.22])},
            holdings={"SPY": pd.DataFrame({"symbol": ["AAPL"], "weight": [0.07]})},
            fundamentals={"AAPL": {"sector": "Tech"}},
        )
        assert provider.fetch_dividends("AAPL").iloc[-1] == pytest.approx(0.22)
        assert provider.fetch_holdings("SPY").loc[0, "symbol"] == "AAPL"
        assert provider.fetch_fundamentals("AAPL")["sector"] == "Tech"

    def test_satisfies_dataprovider_protocol(self) -> None:
        assert isinstance(InMemoryProvider(), DataProvider)


class TestYFinanceProviderCaching:
    """Exercise `YFinanceProvider` without touching the network."""

    def test_cache_hits_skip_yfinance_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_ohlcv: pd.DataFrame,
    ) -> None:
        cache = DiskCache(CacheConfig(cache_dir=tmp_path / "yfc", default_ttl_seconds=60))
        provider = YFinanceProvider(cache=cache)

        calls = {"n": 0}

        class FakeTicker:
            def __init__(self, ticker: str) -> None:
                self.ticker = ticker

            def history(self, period: str, interval: str) -> pd.DataFrame:
                calls["n"] += 1
                return sample_ohlcv

        class FakeYF:
            Ticker = FakeTicker

        monkeypatch.setattr(YFinanceProvider, "_import_yf", staticmethod(lambda: FakeYF))

        provider.fetch_prices("AAPL", "1y", "1d")
        provider.fetch_prices("AAPL", "1y", "1d")
        assert calls["n"] == 1

    def test_no_cache_means_every_call_hits_backend(
        self, monkeypatch: pytest.MonkeyPatch, sample_ohlcv: pd.DataFrame
    ) -> None:
        provider = YFinanceProvider(cache=None)
        calls = {"n": 0}

        class FakeTicker:
            def __init__(self, ticker: str) -> None: ...

            def history(self, period: str, interval: str) -> pd.DataFrame:
                calls["n"] += 1
                return sample_ohlcv

        class FakeYF:
            Ticker = FakeTicker

        monkeypatch.setattr(YFinanceProvider, "_import_yf", staticmethod(lambda: FakeYF))

        provider.fetch_prices("AAPL", "1y", "1d")
        provider.fetch_prices("AAPL", "1y", "1d")
        assert calls["n"] == 2


def test_unknown_cache_format_raises(tmp_path: Path) -> None:
    cache = DiskCache(CacheConfig(cache_dir=tmp_path / "c"))
    with pytest.raises(ValueError, match="Unknown cache format"):
        cache.get_or_compute(
            "fetch_prices",
            "AAPL",
            "csv",  # type: ignore[arg-type]
            lambda: {"x": 1},
        )


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Belt-and-suspenders: ensure `YFinanceProvider._import_yf` is never called
    without a monkeypatch in place.
    """

    def boom() -> Any:
        raise RuntimeError(
            "Test attempted to import yfinance — patch YFinanceProvider._import_yf"
        )

    monkeypatch.setattr(YFinanceProvider, "_import_yf", staticmethod(boom))
    yield


# Sanity check that the no-network guard works by ensuring the InMemory tests
# above never trigger it (they don't construct YFinanceProvider).
def test_no_network_guard_is_active() -> None:
    with pytest.raises(RuntimeError, match="patch YFinanceProvider._import_yf"):
        YFinanceProvider._import_yf()


# Ensure timestamps used in samples are valid datetimes (catches regressions
# in fixture wiring before tests that depend on them silently pass).
def test_sample_ohlcv_index_is_datetime(sample_ohlcv: pd.DataFrame) -> None:
    assert isinstance(sample_ohlcv.index, pd.DatetimeIndex)
    assert sample_ohlcv.index[0] == datetime(2024, 1, 1)
