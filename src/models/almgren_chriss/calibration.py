"""Calibration of Almgren-Chriss inputs from yfinance-style data.

Three quantities are calibrated:

- ``sigma``: annualized log-return volatility, decimal. Computed from a
  daily-close window per the spec's "Volatility σ" section.
- ``eta``: temporary impact coefficient. Implemented are two of the
  spec's three rules — Almgren et al. (2005), and the half-spread
  rule. Default is Almgren 2005.
- ``gamma``: permanent impact coefficient, ``η / (10 · τ_halflife)`` where
  ``τ_halflife`` is in trading-day units. Default ``τ_halflife = 1.0``.

``calibrate(data, **params)`` is the public entry point and returns a
``CalibrationResult`` whose ``parameters`` contain a fully-populated
``ImpactParams``.

References
----------
Almgren, Thum, Hauptmann, Li (2005). "Direct estimation of equity market
impact." *Risk*, 18.  The 2005 paper proposes
``η ≈ η_0 σ V_daily^{-β}`` with ``η_0 ≈ 0.142``, ``β ≈ 0.6`` and ``σ`` in
daily-return decimal units. The result is a per-share dollar cost when
multiplied by the **fraction of daily volume traded per unit time**;
here we report ``η`` in $/share per (share/trading-day) so that
multiplying by ``v_t`` (shares/trading-day) yields per-share dollars.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.almgren_chriss.types import (
    AlmgrenChrissInputs,
    ImpactParams,
)

MODEL_NAME: str = "almgren_chriss"

EtaRule = Literal["almgren_2005", "half_spread"]
VALID_ETA_RULES: frozenset[str] = frozenset({"almgren_2005", "half_spread"})

_TRADING_DAYS_PER_YEAR: int = 252
# Defaults for the Almgren-2005 rule.
_ALMGREN_ETA0: float = 0.142
_ALMGREN_BETA: float = 0.6
# Default half-spread for US large-caps (1 bp on $100 = $0.01).
_DEFAULT_HALF_SPREAD_BPS: float = 1.0
# Default permanent-impact half-life (in trading-day units).
_DEFAULT_GAMMA_HALFLIFE_DAYS: float = 1.0
# Default rolling window for sigma (in trading days).
_DEFAULT_SIGMA_WINDOW: int = 30


def realized_log_return_vol(
    close: pd.Series,
    *,
    window: int = _DEFAULT_SIGMA_WINDOW,
) -> float:
    """Annualized log-return vol over the trailing ``window`` daily bars.

    ``σ_annual = std(ln(C_t / C_{t-1})) · sqrt(252)``.
    """

    if close is None or close.empty:
        raise ValueError("realized_log_return_vol: close must be non-empty")
    if window < 5:
        raise ValueError(f"realized_log_return_vol: window must be >= 5, got {window}")
    s = close.astype(float).sort_index().dropna()
    window_eff = (
        max(min(len(s) - 1, _DEFAULT_SIGMA_WINDOW), 5)
        if len(s) < window + 1
        else window
    )
    log_diff = np.diff(np.log(s.to_numpy()))
    if log_diff.size == 0:
        raise ValueError("realized_log_return_vol: no return observations")
    tail = log_diff[-window_eff:]
    sd = float(np.std(tail, ddof=1))
    return sd * math.sqrt(_TRADING_DAYS_PER_YEAR)


def average_daily_volume(
    volume: pd.Series,
    *,
    window: int = 20,
) -> float:
    """Average daily share volume over the trailing ``window`` sessions.

    Falls back to the full sample if shorter than ``window``.
    """

    if volume is None or volume.empty:
        raise ValueError("average_daily_volume: volume must be non-empty")
    v = volume.astype(float).sort_index().dropna()
    v = v[v > 0]
    if v.empty:
        raise ValueError("average_daily_volume: all volume entries are non-positive")
    tail = v.tail(window) if len(v) >= 5 else v
    return float(tail.mean())


def eta_almgren_2005(
    *,
    sigma_annual: float,
    daily_volume: float,
    last_price: float,
    eta0: float = _ALMGREN_ETA0,
    beta: float = _ALMGREN_BETA,
) -> float:
    """Almgren et al. (2005) temporary-impact coefficient.

    Original form (dimensionful):
        ``Δp̄ / p = η_0 · σ_daily · (Q / V_daily)^β``

    Differentiating with respect to ``Q`` and identifying the linearized
    coefficient near ``Q / V ≈ 1`` gives a marginal-cost rate. To map
    onto the linear-impact ``η v_t`` convention we report ``η`` such
    that ``η · v_t`` is the per-share dollar slippage when ``v_t`` is the
    trading rate in shares per trading-day. The resulting closed form
    used here:

        ``η = β · η_0 · σ_daily · price / V_daily``

    (linearized at the typical trade size ``Q ≈ V_daily``, so
    ``(Q/V)^{β-1} ≈ 1``). This is the per-share dollar cost per unit
    trading rate (shares/day) and the canonical default in production
    Almgren-Chriss implementations.
    """

    if sigma_annual <= 0 or daily_volume <= 0 or last_price <= 0:
        raise ValueError(
            "eta_almgren_2005: sigma_annual, daily_volume, last_price must be > 0"
        )
    sigma_daily = sigma_annual / math.sqrt(_TRADING_DAYS_PER_YEAR)
    return float(beta * eta0 * sigma_daily * last_price / daily_volume)


def eta_half_spread(
    *,
    half_spread_bps: float,
    daily_volume: float,
    last_price: float,
) -> float:
    """Half-spread temporary-impact coefficient.

    ``η ≈ (s / 2) / V_daily`` in the dollar-per-share-per-trading-rate
    units used in the schedule.

    Caller supplies the half-spread in basis points of price.
    """

    if half_spread_bps <= 0 or daily_volume <= 0 or last_price <= 0:
        raise ValueError(
            "eta_half_spread: half_spread_bps, daily_volume, last_price "
            "must be > 0"
        )
    half_spread_dollars = (half_spread_bps / 1.0e4) * last_price
    return float(half_spread_dollars / daily_volume)


def gamma_from_eta(
    eta: float,
    *,
    halflife_days: float = _DEFAULT_GAMMA_HALFLIFE_DAYS,
) -> float:
    """``γ ≈ η / (10 · τ_halflife)`` — spec's permanent-impact rule.

    For practical purposes ``γ`` drops out of schedule optimization and
    matters only for the expected-cost number, so a conservative default
    is fine.
    """

    if eta <= 0 or halflife_days <= 0:
        raise ValueError("gamma_from_eta: eta and halflife_days must be > 0")
    return float(eta / (10.0 * halflife_days))


def calibrate(
    inputs: AlmgrenChrissInputs,
    *,
    sigma_window: int = _DEFAULT_SIGMA_WINDOW,
    volume_window: int = 20,
    eta_rule: EtaRule = "almgren_2005",
    half_spread_bps: float = _DEFAULT_HALF_SPREAD_BPS,
    eta0: float = _ALMGREN_ETA0,
    beta: float = _ALMGREN_BETA,
    gamma_halflife_days: float = _DEFAULT_GAMMA_HALFLIFE_DAYS,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Calibrate ``ImpactParams`` (σ, η, γ) from a single asset's data.

    Inputs come from ``AlmgrenChriss.fetch_data`` — daily close, daily
    volume, last price. Returns a ``CalibrationResult`` whose
    ``parameters["impact"]`` is the fitted ``ImpactParams``.
    """

    if eta_rule not in VALID_ETA_RULES:
        raise ValueError(
            f"eta_rule must be one of {sorted(VALID_ETA_RULES)}, got {eta_rule!r}"
        )
    ts = timestamp or datetime.now(UTC)

    sigma_annual = realized_log_return_vol(inputs.close, window=sigma_window)
    if sigma_annual <= 0 or not math.isfinite(sigma_annual):
        raise RuntimeError(
            f"Calibrated sigma_annual is invalid: {sigma_annual} (ticker={inputs.ticker!r})"
        )

    daily_vol = average_daily_volume(inputs.volume, window=volume_window)

    if eta_rule == "almgren_2005":
        eta = eta_almgren_2005(
            sigma_annual=sigma_annual,
            daily_volume=daily_vol,
            last_price=inputs.last_price,
            eta0=eta0,
            beta=beta,
        )
    else:  # half_spread
        eta = eta_half_spread(
            half_spread_bps=half_spread_bps,
            daily_volume=daily_vol,
            last_price=inputs.last_price,
        )

    gamma = gamma_from_eta(eta, halflife_days=gamma_halflife_days)
    params = ImpactParams(
        sigma=sigma_annual,
        eta=eta,
        gamma=gamma,
        last_price=inputs.last_price,
        daily_volume=daily_vol,
    )
    metrics: dict[str, float] = {
        "sigma_annual": sigma_annual,
        "sigma_daily": sigma_annual / math.sqrt(_TRADING_DAYS_PER_YEAR),
        "daily_volume": daily_vol,
        "eta": eta,
        "gamma": gamma,
        "last_price": inputs.last_price,
        "n_close_observations": float(len(inputs.close)),
        "n_volume_observations": float(len(inputs.volume)),
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={
            "impact": params,
            "sigma_annual": sigma_annual,
            "eta": eta,
            "gamma": gamma,
            "eta_rule": eta_rule,
        },
        fit_metrics=metrics,
        timestamp=ts,
        metadata={
            "ticker": inputs.ticker,
            "eta_rule": eta_rule,
            "sigma_window": sigma_window,
            "volume_window": volume_window,
            "gamma_halflife_days": gamma_halflife_days,
        },
    )


__all__ = [
    "MODEL_NAME",
    "VALID_ETA_RULES",
    "EtaRule",
    "average_daily_volume",
    "calibrate",
    "eta_almgren_2005",
    "eta_half_spread",
    "gamma_from_eta",
    "realized_log_return_vol",
]
