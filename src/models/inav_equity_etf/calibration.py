"""Calibration entry point for the iNAV / equity-ETF model.

Per the spec there is no optimization step — iNAV is an accounting identity.
The "calibration" boils down to three things:

1. Refresh the creation-unit basket {n_i} from the latest holdings file.
2. Seed C_{T-1}, L_{T-1}, NAV_{T-1}, S_{T-1} from the issuer's prior-day strike.
3. Re-estimate the bid/ask half-spread band (kappa + tau) from realized
   round-trip basket trade costs, when such history is supplied.

The first two are deterministic snapshots. The third is a simple robust
quantile of |premium_t| over a recent window — that is the most defensible
empirical proxy for the round-trip cost when realized P&L is unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from src.core.types import CalibrationResult
from src.models.inav_equity_etf.types import (
    BasketHolding,
    CashLedgerSeed,
    CostParameters,
)

MODEL_NAME: str = "inav_equity_etf"
_DEFAULT_COST_QUANTILE: float = 0.95
_BPS_PER_UNIT: float = 1e4


def calibrate(
    *,
    basket: Sequence[BasketHolding],
    seed: CashLedgerSeed,
    realized_premia: Sequence[float] | None = None,
    default_kappa_bps: float = 1.0,
    default_tau_bps: float = 2.0,
    cost_quantile: float = _DEFAULT_COST_QUANTILE,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Refresh the model's static and slow-moving parameters.

    `realized_premia` is an optional series of historical (M_t - iNAV_t)/iNAV_t
    samples. When supplied (and long enough), the high quantile of the absolute
    values is used as the cost band; otherwise we fall back to the defaults
    (1 bp creation fee + 2 bp basket cost for SPY-class liquidity).
    """

    if not basket:
        raise ValueError("calibrate() requires at least one BasketHolding in the basket")
    if seed.eod_nav <= 0:
        raise ValueError(f"seed.eod_nav must be positive, got {seed.eod_nav}")
    if seed.eod_shares_out <= 0:
        raise ValueError(
            f"seed.eod_shares_out must be positive, got {seed.eod_shares_out}"
        )
    if not 0.0 < cost_quantile < 1.0:
        raise ValueError(f"cost_quantile must lie in (0, 1), got {cost_quantile}")

    cost = _estimate_cost_band(
        realized_premia,
        default_kappa_bps=default_kappa_bps,
        default_tau_bps=default_tau_bps,
        cost_quantile=cost_quantile,
    )

    fit_metrics: dict[str, float] = {
        "basket_size": float(len(basket)),
        "eod_nav": float(seed.eod_nav),
        "eod_shares_out": float(seed.eod_shares_out),
        "kappa_bps": cost.kappa_bps,
        "tau_bps": cost.tau_bps,
        "cost_band_bps": cost.kappa_bps + cost.tau_bps,
    }
    if realized_premia is not None and len(realized_premia) > 0:
        fit_metrics["n_realized_premia"] = float(len(realized_premia))

    parameters: dict[str, object] = {
        "basket": tuple(basket),
        "seed": seed,
        "cost_params": cost,
    }

    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=timestamp or datetime.now(UTC),
        metadata={"cost_quantile": cost_quantile},
    )


def _estimate_cost_band(
    realized_premia: Sequence[float] | None,
    *,
    default_kappa_bps: float,
    default_tau_bps: float,
    cost_quantile: float,
) -> CostParameters:
    if realized_premia is None or len(realized_premia) < 20:
        return CostParameters(kappa_bps=default_kappa_bps, tau_bps=default_tau_bps)

    abs_premia_bps = sorted(abs(p) * _BPS_PER_UNIT for p in realized_premia)
    # Robust empirical quantile (linear interpolation) of |premium| in bps.
    pos = cost_quantile * (len(abs_premia_bps) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(abs_premia_bps) - 1)
    frac = pos - lo
    band_bps = abs_premia_bps[lo] * (1.0 - frac) + abs_premia_bps[hi] * frac

    # Issuer creation fee is contractual; the residual is the AP's transaction cost.
    kappa_bps = default_kappa_bps
    tau_bps = max(0.0, band_bps - kappa_bps)
    return CostParameters(kappa_bps=kappa_bps, tau_bps=tau_bps)
