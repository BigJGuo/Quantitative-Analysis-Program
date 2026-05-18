"""`calibrate(...)` for the VaR model.

Unlike GARCH this model has no MLE step — "calibration" here is
parameter selection plus the up-front computation of the chosen VaR
engine. That gives a clean place to validate inputs, build the sample
moments / quantile cache, and stamp the result with a
`CalibrationResult` for the orchestration layer.

The function is method-aware: parametric, historical, and Monte Carlo
each follow the spec's algorithm outline and produce the same
`VaRFit` shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.var.signal import (
    align_positions,
    historical_var,
    horizon_scale,
    monte_carlo_var,
    parametric_var,
    portfolio_pnl_vector,
)
from src.models.var.types import (
    VALID_METHODS,
    VaRFit,
    VaRMethod,
)

MODEL_NAME: str = "var"


def calibrate(
    *,
    returns: pd.DataFrame,
    positions: dict[str, float],
    method: VaRMethod = "historical",
    alpha: float = 0.99,
    horizon_days: int = 10,
    mc_samples: int = 50_000,
    mc_seed: int | None = None,
    timestamp: datetime | None = None,
    include_mean: bool = True,
) -> CalibrationResult:
    """Compute portfolio VaR via the chosen method.

    Parameters
    ----------
    returns:
        ``pd.DataFrame`` of decimal simple returns. Columns must align
        with ``positions.keys()`` (same set, same order).
    positions:
        Mapping ``ticker -> signed dollar notional``.
    method:
        ``"parametric"`` (variance-covariance, Gaussian),
        ``"historical"`` (empirical quantile), or ``"monte_carlo"``
        (Gaussian MC with sample mean / covariance).
    alpha:
        Confidence level in ``(0, 1)`` — typically ``0.95`` or ``0.99``.
    horizon_days:
        Reporting horizon for the sqrt-h scaled VaR.
    mc_samples / mc_seed:
        Only used for ``method == "monte_carlo"``.
    timestamp:
        Optional fit-time stamp; defaults to ``datetime.now(UTC)``.
    include_mean:
        Whether to retain the drift term in parametric / MC. At daily
        frequency the spec recommends dropping it (``False``); the default
        is ``True`` for fidelity to the spec's closed-form expression.

    Returns
    -------
    CalibrationResult
        With ``parameters["fit"]`` populated as a `VaRFit`.
    """

    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(VALID_METHODS)}, got {method!r}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if not positions:
        raise ValueError("positions must be a non-empty mapping.")
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if list(returns.columns) != list(positions.keys()):
        raise ValueError(
            f"returns.columns ({list(returns.columns)}) must match "
            f"positions.keys() ({list(positions.keys())})"
        )
    if len(returns) < 30:
        raise ValueError(
            f"calibrate requires >= 30 return rows, got {len(returns)}"
        )

    ts = timestamp or datetime.now(UTC)
    rets_array = np.asarray(returns.values, dtype=float)
    w, _ = align_positions(positions, returns.columns)
    V0 = float(np.sum(list(positions.values())))
    gross = float(np.sum(np.abs(list(positions.values()))))

    var_1d_dollar: float
    mu_p_dollar: float | None
    sigma_p_dollar: float | None
    sigma_matrix: np.ndarray | None
    pnl_vector: np.ndarray
    mc_n: int | None

    if method == "parametric":
        var_1d_dollar, mu_p, sigma_p, sigma_matrix = parametric_var(
            rets_array, w, alpha, include_mean=include_mean
        )
        # Empirical P&L on the same window for plotting / cross-check.
        pnl_vector = portfolio_pnl_vector(rets_array, w)
        mu_p_dollar = mu_p
        sigma_p_dollar = sigma_p
        mc_n = None
    elif method == "historical":
        var_1d_dollar, pnl_vector = historical_var(rets_array, w, alpha)
        mu_p_dollar = float(pnl_vector.mean())
        # Use ddof=1 to match the parametric branch.
        sigma_p_dollar = float(pnl_vector.std(ddof=1)) if pnl_vector.size > 1 else 0.0
        sigma_matrix = None
        mc_n = None
    else:  # monte_carlo
        var_1d_dollar, mu_p, sigma_p, sigma_matrix, pnl_sim = monte_carlo_var(
            rets_array,
            w,
            alpha,
            n_samples=mc_samples,
            seed=mc_seed,
            include_mean=include_mean,
        )
        pnl_vector = pnl_sim
        mu_p_dollar = mu_p
        sigma_p_dollar = sigma_p
        mc_n = mc_samples

    var_horizon_dollar = horizon_scale(var_1d_dollar, horizon_days)

    fit = VaRFit(
        method=method,
        alpha=alpha,
        horizon_days=horizon_days,
        n_obs=int(rets_array.shape[0]),
        n_assets=int(rets_array.shape[1]),
        portfolio_value=V0,
        gross_exposure=gross,
        var_1d_dollar=var_1d_dollar,
        var_horizon_dollar=var_horizon_dollar,
        pnl_vector=pnl_vector,
        mu_p_dollar=mu_p_dollar,
        sigma_p_dollar=sigma_p_dollar,
        sigma_matrix=sigma_matrix,
        mc_samples=mc_n,
        metadata={"include_mean": include_mean},
    )

    fit_metrics: dict[str, float] = {
        "var_1d_dollar": var_1d_dollar,
        "var_horizon_dollar": var_horizon_dollar,
        "horizon_days": float(horizon_days),
        "alpha": float(alpha),
        "n_obs": float(rets_array.shape[0]),
        "n_assets": float(rets_array.shape[1]),
        "portfolio_value": V0,
        "gross_exposure": gross,
        "pnl_mean": float(pnl_vector.mean()),
        "pnl_std": (
            float(pnl_vector.std(ddof=1)) if pnl_vector.size > 1 else 0.0
        ),
        "pnl_min": float(pnl_vector.min()),
        "pnl_max": float(pnl_vector.max()),
    }
    if mu_p_dollar is not None:
        fit_metrics["mu_p_dollar"] = mu_p_dollar
    if sigma_p_dollar is not None:
        fit_metrics["sigma_p_dollar"] = sigma_p_dollar
    if gross > 0:
        fit_metrics["var_1d_pct_gross"] = var_1d_dollar / gross
        fit_metrics["var_horizon_pct_gross"] = var_horizon_dollar / gross

    parameters: dict[str, object] = {
        "fit": fit,
        "method": method,
        "alpha": alpha,
        "horizon_days": horizon_days,
        "positions": dict(positions),
    }
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata={
            "method": method,
            "include_mean": include_mean,
            "mc_samples": mc_n,
        },
    )


__all__ = ["MODEL_NAME", "calibrate"]
