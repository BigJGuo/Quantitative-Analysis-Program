"""Calibration entry point for the cointegration / pair-trading model.

`calibrate()` runs the full Stage-1 screening pipeline on a single pair:

    1. Standalone ADF on each leg                  -> both should fail to reject
    2. OLS `p_A = alpha + beta * p_B + Z`          -> Engle-Granger step 1
    3. Engle-Granger residual ADF                  -> rejection = cointegration
    4. AR(1) -> OU mapping                         -> kappa, mu, sigma, half-life
    5. Half-life filter `[half_life_min, half_life_max]`
    6. (optional) Kalman dynamic-beta initialization

The result is a `CalibrationResult` whose `parameters["pair_fit"]` is a
`PairFit`. Rejected pairs carry `status="rejected"` plus a `rejection_reason`
string so downstream `predict` can emit a flat / zero-strength signal without
crashing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.core.types import CalibrationResult
from src.models.cointegration_pairs.signal import (
    engle_granger_test,
    fit_ou,
    kalman_dynamic_beta,
)
from src.models.cointegration_pairs.types import (
    VALID_METHODS,
    EngleGrangerFit,
    KalmanFit,
    OUFit,
    PairFit,
    PairMethod,
    PairStatus,
)

MODEL_NAME: str = "cointegration_pairs"

_DEFAULT_HALF_LIFE_MIN: float = 0.5
_DEFAULT_HALF_LIFE_MAX: float = 30.0
_DEFAULT_SIGNIFICANCE: str = "5%"
_DEFAULT_KALMAN_Q: float = 1e-4
_DEFAULT_KALMAN_R: float = 1e-3


def calibrate(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    *,
    method: PairMethod = "static",
    significance: str = _DEFAULT_SIGNIFICANCE,
    half_life_min: float = _DEFAULT_HALF_LIFE_MIN,
    half_life_max: float = _DEFAULT_HALF_LIFE_MAX,
    adf_lags: int = 1,
    require_unit_root_legs: bool = True,
    Q: float = _DEFAULT_KALMAN_Q,
    R: float = _DEFAULT_KALMAN_R,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Run the full cointegration screening on a single pair.

    Parameters
    ----------
    log_price_a, log_price_b:
        Aligned log-close series for the two legs. `a` is the dependent
        variable in the cointegrating regression.
    method:
        ``"static"`` keeps the OLS beta fixed and trades the rolling z-score.
        ``"kalman"`` additionally fits the random-walk hedge ratio so the
        standardized innovation can be used as the live signal.
    significance:
        ADF significance level used for both leg I(1) checks and the EG
        residual test. One of ``"1%"`` / ``"5%"`` / ``"10%"``.
    half_life_min, half_life_max:
        Reject pairs whose OU half-life falls outside this band. Defaults
        come straight from the spec: < 0.5 days = noise; > 30 days = capital
        tied up too long.
    adf_lags:
        Number of lagged-difference terms in the ADF regression.
    require_unit_root_legs:
        When True (default), reject the pair if either leg's ADF rejects
        I(1). Useful to disable when calibrating on already-differenced data.
    Q, R:
        Kalman process / observation noise variances. Ignored when
        `method="static"`.

    Returns
    -------
    CalibrationResult
        `parameters` contains ``"pair_fit": PairFit``. `fit_metrics` records
        alpha / beta / EG t-stat / half-life and the like.
    """

    if method not in VALID_METHODS:
        raise ValueError(
            f"calibrate: method must be one of {sorted(VALID_METHODS)}, got {method!r}"
        )
    if half_life_min < 0 or half_life_max <= half_life_min:
        raise ValueError(
            f"half-life band must satisfy 0 <= half_life_min < half_life_max; got "
            f"[{half_life_min}, {half_life_max}]"
        )
    ts = timestamp or datetime.now(UTC)

    eg_fit = engle_granger_test(log_price_a, log_price_b, n_lags=adf_lags)
    ou_fit = fit_ou(eg_fit.residuals.to_numpy())

    status, rejection_reason = _assess_pair(
        eg_fit=eg_fit,
        ou_fit=ou_fit,
        significance=significance,
        half_life_min=half_life_min,
        half_life_max=half_life_max,
        require_unit_root_legs=require_unit_root_legs,
    )

    kalman_fit: KalmanFit | None = None
    if method == "kalman" and status == "accepted":
        kalman_fit = kalman_dynamic_beta(
            log_price_a.to_numpy(),
            log_price_b.to_numpy(),
            Q=Q,
            R=R,
            beta0=eg_fit.beta,
            P0=max(1e-6, ou_fit.sigma_eps * ou_fit.sigma_eps),
        )

    pair_fit = PairFit(
        method=method,
        eg_fit=eg_fit,
        ou_fit=ou_fit,
        kalman_fit=kalman_fit,
        status=status,
        rejection_reason=rejection_reason,
        significance=significance,
        half_life_band=(half_life_min, half_life_max),
    )

    fit_metrics = _build_fit_metrics(eg_fit=eg_fit, ou_fit=ou_fit, status=status)
    metadata: dict[str, object] = {
        "method": method,
        "significance": significance,
        "half_life_band": [half_life_min, half_life_max],
        "rejection_reason": rejection_reason or "",
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters={"pair_fit": pair_fit},
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata=metadata,
    )


def _assess_pair(
    *,
    eg_fit: EngleGrangerFit,
    ou_fit: OUFit,
    significance: str,
    half_life_min: float,
    half_life_max: float,
    require_unit_root_legs: bool,
) -> tuple[PairStatus, str | None]:
    """Apply the spec's Stage-1 acceptance filters in order."""

    if require_unit_root_legs:
        # Each leg should *fail* to reject the unit-root null at the chosen level.
        # If either leg already looks stationary, the cointegration framing is
        # ill-posed (spec algorithm step 2).
        for label, adf in (("A", eg_fit.adf_leg_a), ("B", eg_fit.adf_leg_b)):
            if adf.rejects_unit_root(level=significance):
                return (
                    "rejected",
                    f"leg {label} rejects I(1) null at {significance}; "
                    f"t={adf.t_stat:.3f} < {adf.critical_values[significance]:.3f}",
                )

    if not eg_fit.adf_residual.rejects_unit_root(level=significance):
        return (
            "rejected",
            f"Engle-Granger residual does not reject unit-root at {significance}; "
            f"t={eg_fit.adf_residual.t_stat:.3f} >= "
            f"{eg_fit.adf_residual.critical_values[significance]:.3f}",
        )

    if not (0.0 < ou_fit.phi < 1.0):
        return (
            "rejected",
            f"AR(1) phi={ou_fit.phi:.3f} outside (0, 1); no mean reversion",
        )

    if not (half_life_min <= ou_fit.half_life <= half_life_max):
        return (
            "rejected",
            f"half-life {ou_fit.half_life:.2f} outside band "
            f"[{half_life_min}, {half_life_max}]",
        )

    return "accepted", None


def _build_fit_metrics(
    *,
    eg_fit: EngleGrangerFit,
    ou_fit: OUFit,
    status: PairStatus,
) -> dict[str, float]:
    return {
        "alpha": eg_fit.alpha,
        "beta": eg_fit.beta,
        "adf_leg_a_tstat": eg_fit.adf_leg_a.t_stat,
        "adf_leg_b_tstat": eg_fit.adf_leg_b.t_stat,
        "eg_residual_tstat": eg_fit.adf_residual.t_stat,
        "ou_phi": ou_fit.phi,
        "ou_mu": ou_fit.mu,
        "ou_kappa": ou_fit.kappa,
        "ou_sigma": ou_fit.sigma,
        "ou_sigma_eps": ou_fit.sigma_eps,
        "ou_sigma_eq": ou_fit.sigma_eq,
        "half_life": ou_fit.half_life,
        "accepted": 1.0 if status == "accepted" else 0.0,
    }


__all__ = ["MODEL_NAME", "calibrate"]
