"""End-to-end integration test for the iNAV / equity-ETF model.

Hits yfinance via the production `YFinanceProvider` for a single representative
ticker (SPY). Skipped automatically when yfinance is missing or network access
fails, so unit-test CI without internet stays green.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from src.core.data_provider import YFinanceProvider
from src.core.types import Signal
from src.models.inav_equity_etf import (
    BasketHolding,
    CashLedgerSeed,
    INAVEquityETF,
)

pytest.importorskip("yfinance")


_ETF = "SPY"


def _make_holdings_from_top10(provider: YFinanceProvider) -> list[BasketHolding]:
    """Best-effort basket from yfinance's top-10 holdings.

    The production deployment loads the issuer's full holdings JSON; the
    integration test settles for the top 10 because that is all yfinance
    exposes via `funds_data.top_holdings`. yfinance has shipped several
    column-name conventions over time; the helper picks a reasonable
    symbol column and a numeric weight column dynamically.
    """

    df = provider.fetch_holdings(_ETF)
    if df is None or df.empty:
        pytest.skip("yfinance returned no holdings for SPY")

    weight_candidates = [
        c for c in df.columns
        if c.lower() in {"weight", "holdingpercent", "weighting", "percent"}
    ]
    if not weight_candidates:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            pytest.skip(f"No numeric weight column in yfinance holdings: {list(df.columns)}")
        weight_col = numeric_cols[0]
    else:
        weight_col = weight_candidates[0]

    symbol_candidates = [
        c for c in df.columns if c.lower() in {"symbol", "ticker", "holding"}
    ]
    symbol_col = symbol_candidates[0] if symbol_candidates else df.columns[0]

    holdings: list[BasketHolding] = []
    weights = df[weight_col].astype(float)
    total = float(weights.sum()) or 1.0
    norm = weights / total
    for sym, w in zip(df[symbol_col].astype(str), norm, strict=False):
        # Synthetic share count: 1000 * weight, sufficient to test pipeline plumbing.
        holdings.append(
            BasketHolding(ticker=str(sym), shares=1000.0 * float(w), currency="USD")
        )
    return holdings


def _make_seed(provider: YFinanceProvider, holdings: list[BasketHolding]) -> CashLedgerSeed:
    info = provider.fetch_fundamentals(_ETF)
    nav = float(info.get("navPrice") or info.get("previousClose") or 400.0)
    eod_release = datetime.now(UTC) - timedelta(hours=18)
    return CashLedgerSeed(
        eod_nav=nav,
        eod_shares_out=1_000_000.0,
        eod_release_time=eod_release,
        prior_cash=0.0,
        prior_liabilities=0.0,
    )


@pytest.fixture(scope="module")
def provider() -> YFinanceProvider:
    return YFinanceProvider(cache=None)


@pytest.fixture(scope="module")
def model(provider: YFinanceProvider) -> INAVEquityETF:
    try:
        holdings = _make_holdings_from_top10(provider)
        seed = _make_seed(provider, holdings)
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance setup failed (network or rate limit): {exc!r}")
    return INAVEquityETF(etf_ticker=_ETF, holdings=holdings, seed=seed)


def test_end_to_end_pipeline_against_yfinance(
    provider: YFinanceProvider, model: INAVEquityETF
) -> None:
    try:
        data = model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")

    # The ETF mid must come through — that's the load-bearing input.
    assert data.etf_mid > 0
    # At least half the top-10 should have produced a quote even on weekends.
    assert len(data.quotes) >= max(1, len(model.holdings) // 2)

    signal = model.predict(data)
    assert isinstance(signal, Signal)
    assert signal.ticker == _ETF
    assert signal.direction in {"long", "short", "flat"}
    assert 0.0 <= signal.strength <= 1.0
    assert "inav" in signal.metadata
    assert signal.metadata["inav"] > 0


def test_validate_emits_diagnostics(
    provider: YFinanceProvider, model: INAVEquityETF
) -> None:
    try:
        data = model.fetch_data(provider)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yfinance fetch failed (network or rate limit): {exc!r}")
    diag = model.validate(data)
    for key in ("inav", "premium_bps", "coverage", "n_live", "n_stale", "band_bps"):
        assert key in diag
