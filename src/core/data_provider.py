"""Data access layer for the quant ecosystem.

`DataProvider` is the abstract contract. `YFinanceProvider` is the production
implementation backed by yfinance with on-disk caching. `InMemoryProvider`
is a deterministic fixture-loaded variant used in tests.

The caching layer is decoupled (`DiskCache`) so it can be reused or swapped
without touching provider logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import pandas as pd

logger = logging.getLogger(__name__)

CacheFormat = Literal["parquet", "json", "pickle"]
_VALID_FORMATS: frozenset[str] = frozenset({"parquet", "json", "pickle"})
T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for `DiskCache`.

    `per_method_ttl_seconds` lets each `DataProvider` method have its own TTL
    (e.g. fundamentals refreshed weekly, intraday refreshed every few minutes).
    """

    cache_dir: Path = Path(".cache/data")
    default_ttl_seconds: int = 3600
    per_method_ttl_seconds: dict[str, int] = field(default_factory=dict)


class DiskCache:
    """Disk-backed key/value cache with per-method TTL."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ttl_for(self, method: str) -> int:
        return self.config.per_method_ttl_seconds.get(method, self.config.default_ttl_seconds)

    def _path(self, method: str, key: str, fmt: CacheFormat) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.config.cache_dir / method / f"{digest}.{fmt}"

    def _is_fresh(self, path: Path, method: str) -> bool:
        ttl = self._ttl_for(method)
        if ttl <= 0:
            return False
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < ttl

    def get_or_compute(
        self,
        method: str,
        key: str,
        fmt: CacheFormat,
        compute: Callable[[], T],
    ) -> T:
        if fmt not in _VALID_FORMATS:
            raise ValueError(f"Unknown cache format: {fmt}")
        path = self._path(method, key, fmt)
        if self._is_fresh(path, method):
            try:
                return cast(T, self._read(path, fmt))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cache read failed for %s; recomputing. %s", path, exc)
        value = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._write(path, fmt, value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache write failed for %s. %s", path, exc)
        return value

    @staticmethod
    def _read(path: Path, fmt: CacheFormat) -> Any:
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "json":
            return json.loads(path.read_text(encoding="utf-8"))
        if fmt == "pickle":
            with path.open("rb") as fh:
                return pickle.load(fh)
        raise ValueError(f"Unknown cache format: {fmt}")

    @staticmethod
    def _write(path: Path, fmt: CacheFormat, value: Any) -> None:
        if fmt == "parquet":
            if not isinstance(value, pd.DataFrame):
                value = pd.DataFrame(value)
            value.to_parquet(path)
            return
        if fmt == "json":
            path.write_text(json.dumps(value, default=str), encoding="utf-8")
            return
        if fmt == "pickle":
            with path.open("wb") as fh:
                pickle.dump(value, fh)
            return
        raise ValueError(f"Unknown cache format: {fmt}")


class DataProvider(ABC):
    """Abstract data source for all quant models.

    Implementations may cache results; the contract makes no guarantees about
    cache freshness beyond what the implementation documents.
    """

    @abstractmethod
    def fetch_prices(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """OHLCV bars. `period` and `interval` follow yfinance conventions."""

    @abstractmethod
    def fetch_intraday(
        self, ticker: str, interval: str, lookback_days: int
    ) -> pd.DataFrame:
        """High-frequency OHLCV bars over the last `lookback_days`."""

    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Static / slow-moving company info (P/E, market cap, sector, ...)."""

    @abstractmethod
    def fetch_dividends(self, ticker: str) -> pd.Series:
        """Dividend payments indexed by ex-date."""

    @abstractmethod
    def fetch_holdings(self, etf_ticker: str) -> pd.DataFrame:
        """Constituent holdings of an ETF (`symbol`, `weight`, ...)."""

    @abstractmethod
    def fetch_news(self, ticker: str, max_items: int = 20) -> list[dict[str, Any]]:
        """Recent news headlines. Each item must include at least `title` and `published`."""


class YFinanceProvider(DataProvider):
    """Production `DataProvider` backed by yfinance with optional disk cache."""

    def __init__(self, cache: DiskCache | None = None) -> None:
        self.cache = cache

    def _cached(
        self,
        method: str,
        key: str,
        fmt: CacheFormat,
        compute: Callable[[], T],
    ) -> T:
        if self.cache is None:
            return compute()
        return self.cache.get_or_compute(method, key, fmt, compute)

    @staticmethod
    def _import_yf() -> Any:
        # Imported lazily so the package can be imported without yfinance installed
        # (useful for type-checking and for tests that use InMemoryProvider).
        import yfinance as yf

        return yf

    def fetch_prices(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        def compute() -> pd.DataFrame:
            yf = self._import_yf()
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            return cast(pd.DataFrame, df.reset_index() if not df.empty else df)

        return self._cached("fetch_prices", f"{ticker}|{period}|{interval}", "parquet", compute)

    def fetch_intraday(
        self, ticker: str, interval: str, lookback_days: int
    ) -> pd.DataFrame:
        def compute() -> pd.DataFrame:
            yf = self._import_yf()
            df = yf.Ticker(ticker).history(period=f"{lookback_days}d", interval=interval)
            return cast(pd.DataFrame, df.reset_index() if not df.empty else df)

        key = f"{ticker}|{interval}|{lookback_days}"
        return self._cached("fetch_intraday", key, "parquet", compute)

    def fetch_fundamentals(self, ticker: str) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            yf = self._import_yf()
            info = yf.Ticker(ticker).info
            return dict(info) if info else {}

        return self._cached("fetch_fundamentals", ticker, "json", compute)

    def fetch_dividends(self, ticker: str) -> pd.Series:
        def compute() -> pd.Series:
            yf = self._import_yf()
            div = yf.Ticker(ticker).dividends
            return div if isinstance(div, pd.Series) else pd.Series(dtype="float64")

        return self._cached("fetch_dividends", ticker, "pickle", compute)

    def fetch_holdings(self, etf_ticker: str) -> pd.DataFrame:
        def compute() -> pd.DataFrame:
            yf = self._import_yf()
            tkr = yf.Ticker(etf_ticker)
            funds = getattr(tkr, "funds_data", None)
            if funds is not None:
                top = getattr(funds, "top_holdings", None)
                if isinstance(top, pd.DataFrame):
                    return top.reset_index()
            return pd.DataFrame(columns=["symbol", "weight"])

        return self._cached("fetch_holdings", etf_ticker, "parquet", compute)

    def fetch_news(self, ticker: str, max_items: int = 20) -> list[dict[str, Any]]:
        def compute() -> list[dict[str, Any]]:
            yf = self._import_yf()
            items = yf.Ticker(ticker).news or []
            return [dict(item) for item in items[:max_items]]

        key = f"{ticker}|{max_items}"
        return self._cached("fetch_news", key, "json", compute)


class InMemoryProvider(DataProvider):
    """`DataProvider` backed by pre-loaded fixtures. Intended for unit tests.

    Keys mirror the arguments of the corresponding `fetch_*` method.
    """

    def __init__(
        self,
        prices: dict[tuple[str, str, str], pd.DataFrame] | None = None,
        intraday: dict[tuple[str, str, int], pd.DataFrame] | None = None,
        fundamentals: dict[str, dict[str, Any]] | None = None,
        dividends: dict[str, pd.Series] | None = None,
        holdings: dict[str, pd.DataFrame] | None = None,
        news: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.prices = prices or {}
        self.intraday = intraday or {}
        self.fundamentals = fundamentals or {}
        self.dividends = dividends or {}
        self.holdings = holdings or {}
        self.news = news or {}

    def fetch_prices(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        return self.prices[(ticker, period, interval)]

    def fetch_intraday(
        self, ticker: str, interval: str, lookback_days: int
    ) -> pd.DataFrame:
        return self.intraday[(ticker, interval, lookback_days)]

    def fetch_fundamentals(self, ticker: str) -> dict[str, Any]:
        return self.fundamentals[ticker]

    def fetch_dividends(self, ticker: str) -> pd.Series:
        return self.dividends[ticker]

    def fetch_holdings(self, etf_ticker: str) -> pd.DataFrame:
        return self.holdings[etf_ticker]

    def fetch_news(self, ticker: str, max_items: int = 20) -> list[dict[str, Any]]:
        return self.news.get(ticker, [])[:max_items]
