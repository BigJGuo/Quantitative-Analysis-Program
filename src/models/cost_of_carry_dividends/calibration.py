"""Calibration entry point for the Cost-of-Carry model.

The model's three calibrated quantities are:

* The transaction-cost band κ + τ, set to the 95% quantile of the realized
  absolute basis over a rolling window. This is the trigger threshold above
  which the model emits a directional trade signal.
* The forward dividend stream — announced ex-dates are observed; unannounced
  quarters are projected from each constituent's trailing dividend history.
* The financing rate, which is *read* rather than *fit* (no optimization);
  calibrate() therefore only sanity-checks the rate vs the SOFR band.

The implied repo rate itself is a measurement, not an estimate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.cost_of_carry_dividends.signal import (
    _to_naive_ts,
    aggregate_constituent_dividends,
)
from src.models.cost_of_carry_dividends.types import (
    CostParameters,
    Dividend,
    DividendStream,
    IndexConstituent,
)

MODEL_NAME: str = "cost_of_carry_dividends"

_QUARTER_DAYS: int = 91
_DEFAULT_QUANTILE: float = 0.95
_DEFAULT_FALLBACK_BAND_POINTS: float = 0.5
_DEFAULT_FALLBACK_ETF_BPS: float = 2.0
_MIN_BASIS_OBS: int = 10
_DIV_HISTORY_LOOKBACK_QUARTERS: int = 8


def _band_from_basis(
    basis_history: Sequence[float],
    *,
    quantile: float,
    fallback: float,
) -> float:
    """Set the cost band to the `quantile`-tile of |basis| in the history.

    Falls back to `fallback` when the history is too short to be reliable.
    """

    if len(basis_history) < _MIN_BASIS_OBS:
        return fallback
    arr = np.abs(np.asarray(list(basis_history), dtype=np.float64))
    return float(np.quantile(arr, quantile))


def _trimmed_mean(values: np.ndarray, trim_pct: float = 0.10) -> float:
    """Robust mean: drop the top and bottom `trim_pct` of observations."""

    if values.size == 0:
        return 0.0
    if values.size < 4:
        return float(values.mean())
    n_trim = max(1, int(values.size * trim_pct))
    sorted_vals = np.sort(values)
    trimmed = sorted_vals[n_trim:-n_trim] if n_trim < sorted_vals.size // 2 else sorted_vals
    return float(trimmed.mean()) if trimmed.size else float(sorted_vals.mean())


def _next_quarterly_ex_date(last_ex_date: datetime) -> datetime:
    return last_ex_date + timedelta(days=_QUARTER_DAYS)


def project_forward_dividends(
    *,
    constituents: Iterable[IndexConstituent],
    dividends_per_share: Mapping[str, pd.Series],
    today: datetime,
    expiry: datetime,
    announcement_horizon_days: int = 45,
) -> DividendStream:
    """Build the forward dividend stream up to `expiry`.

    Announced ex-dates (already in the per-share series, between today and
    expiry) are taken at face value. Beyond the announcement horizon we
    project additional quarterly ex-dates using each constituent's trailing
    history:

        D_hat = trimmed_mean(last 4 quarters) · (1 + g)
        g     = (avg(last 4Q) / avg(prior 4Q)) − 1

    The 8 most recent quarters drive both the level and the growth rate.
    """

    announced = aggregate_constituent_dividends(
        constituents=list(constituents),
        dividends_per_share=dividends_per_share,
        today=today,
        expiry=expiry,
    )
    projected: list[Dividend] = []
    horizon_cutoff = today + timedelta(days=announcement_horizon_days)

    for constituent in constituents:
        series = dividends_per_share.get(constituent.ticker)
        if series is None or len(series) == 0:
            continue
        sorted_series = series.sort_index()
        index = sorted_series.index
        if not isinstance(index, pd.DatetimeIndex):
            continue
        if index.tz is not None:
            index = index.tz_localize(None)
            sorted_series.index = index

        recent_values = sorted_series.to_numpy()[-_DIV_HISTORY_LOOKBACK_QUARTERS:]
        if recent_values.size == 0:
            continue

        last_4 = recent_values[-4:] if recent_values.size >= 4 else recent_values
        recent_mean = _trimmed_mean(np.asarray(last_4, dtype=np.float64))
        if recent_values.size >= 8:
            prior_4 = recent_values[-8:-4]
            prior_mean = _trimmed_mean(np.asarray(prior_4, dtype=np.float64))
            growth = (recent_mean / prior_mean - 1.0) if prior_mean > 0 else 0.0
        else:
            growth = 0.0
        projected_amount = max(recent_mean * (1.0 + growth), 0.0)
        if projected_amount <= 0:
            continue

        # Walk forward from the last observed ex-date, skipping any quarter
        # already covered by an announced ex-date. Normalize all timestamps
        # to tz-naive wall-clock so mixed inputs compare cleanly.
        announced_dates = {
            _to_naive_ts(d.ex_date).date()
            for d in announced.dividends
            if d.ticker == constituent.ticker
        }
        expiry_naive = _to_naive_ts(expiry).to_pydatetime()
        horizon_naive = _to_naive_ts(horizon_cutoff).to_pydatetime()
        last_ex = index[-1].to_pydatetime()
        cursor = _next_quarterly_ex_date(last_ex)
        while cursor < expiry_naive:
            if cursor > horizon_naive and cursor.date() not in announced_dates:
                idx_points = (
                    projected_amount
                    * constituent.weight
                    * constituent.shares_factor
                )
                projected.append(
                    Dividend(
                        ex_date=cursor,
                        amount_index_points=idx_points,
                        source="projected",
                        ticker=constituent.ticker,
                    )
                )
            cursor = _next_quarterly_ex_date(cursor)

    combined = tuple(announced.dividends) + tuple(projected)
    return DividendStream(combined)


def calibrate(
    *,
    basis_history: Sequence[float] = (),
    etf_premium_history: Sequence[float] = (),
    constituents: Iterable[IndexConstituent] = (),
    dividends_per_share: Mapping[str, pd.Series] | None = None,
    today: datetime | None = None,
    expiry: datetime | None = None,
    quantile: float = _DEFAULT_QUANTILE,
    fallback_band_points: float = _DEFAULT_FALLBACK_BAND_POINTS,
    fallback_etf_band_bps: float = _DEFAULT_FALLBACK_ETF_BPS,
    announcement_horizon_days: int = 45,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Calibrate the cost band and project forward dividends.

    Parameters
    ----------
    basis_history:
        Sequence of realized |F_market − F_theoretical| values (index points).
    etf_premium_history:
        Realized ETF premium values (fractions of NAV) for the parallel
        creation/redemption decision.
    constituents, dividends_per_share, today, expiry:
        Inputs to the dividend projection. When any is missing or the window
        is empty, the result's ``dividend_stream`` is an empty stream.
    quantile, fallback_band_points, fallback_etf_band_bps:
        Quantile of the basis distribution used as the cost band; fallbacks
        kick in when the history is shorter than `_MIN_BASIS_OBS`.

    Returns
    -------
    CalibrationResult
        ``parameters`` contains:

        * ``cost_params`` — a `CostParameters` ready for `predict()`.
        * ``dividend_stream`` — the projected `DividendStream`.

        ``fit_metrics`` reports the band quantile and number of basis obs.
    """

    band_points = _band_from_basis(
        basis_history, quantile=quantile, fallback=fallback_band_points
    )
    etf_band_bps = _band_from_basis(
        [p * 1e4 for p in etf_premium_history],  # convert to bps for the quantile
        quantile=quantile,
        fallback=fallback_etf_band_bps,
    )
    cost = CostParameters(
        futures_band_points=band_points,
        etf_cost_band_bps=etf_band_bps,
    )

    constituent_tuple = tuple(constituents)
    if (
        dividends_per_share is not None
        and today is not None
        and expiry is not None
        and constituent_tuple
    ):
        stream = project_forward_dividends(
            constituents=constituent_tuple,
            dividends_per_share=dividends_per_share,
            today=today,
            expiry=expiry,
            announcement_horizon_days=announcement_horizon_days,
        )
    else:
        stream = DividendStream(())

    parameters: dict[str, Any] = {
        "cost_params": cost,
        "dividend_stream": stream,
    }
    fit_metrics: dict[str, float] = {
        "n_basis_obs": float(len(basis_history)),
        "n_etf_obs": float(len(etf_premium_history)),
        "band_points": float(band_points),
        "etf_band_bps": float(etf_band_bps),
        "n_dividends_announced": float(
            sum(1 for d in stream.dividends if d.source == "announced")
        ),
        "n_dividends_projected": float(
            sum(1 for d in stream.dividends if d.source == "projected")
        ),
        "total_pv_index_points": float(stream.total_index_points),
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=timestamp or datetime.now(UTC),
        metadata={
            "quantile": float(quantile),
            "fallback_band_points": float(fallback_band_points),
            "fallback_etf_band_bps": float(fallback_etf_band_bps),
        },
    )
