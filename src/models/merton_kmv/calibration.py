"""Iterative KMV calibration: solve for `(V_t, sigma_V)` from market data.

The core routine `iterative_kmv` follows the spec's "Iterative KMV procedure":

1. Initialize `sigma_V_0 = sigma_E * E_t / (E_t + D)`.
2. Invert Black-Scholes for `V_s` at each day in the calibration window.
3. Recompute `sigma_V` as the annualized std of `diff(log V)`.
4. Repeat until convergence (typically 5-15 iterations).

The joint-nonlinear-solve variant is exposed as `joint_solve` for callers that
want a one-shot point estimate without iterating over the history.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.merton_kmv.signal import (
    default_point,
    diagnostic_residuals,
    equity_price_to_market_cap,
    equity_vol_from_asset_vol,
    estimate_drift_from_assets,
    merton_call_price,
    realized_equity_volatility,
    select_recent_prices,
    solve_for_v_given_e,
)
from src.models.merton_kmv.types import (
    KMVSolution,
    MertonInputs,
)

MODEL_NAME: str = "merton_kmv"

_DEFAULT_HISTORY_DAYS: int = 252
_DEFAULT_MAX_ITER: int = 50
_DEFAULT_TOL: float = 1e-5
_TRADING_DAYS_PER_YEAR: int = 252


def iterative_kmv(
    *,
    equity_market_value: float,
    equity_price_series: pd.Series,
    shares_outstanding: float,
    default_pt: float,
    risk_free_rate: float,
    horizon_years: float,
    history_days: int = _DEFAULT_HISTORY_DAYS,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> KMVSolution:
    """Iterative KMV solve for `(V, sigma_V)`.

    Parameters
    ----------
    equity_market_value:
        Current equity market value `E_t` (= price * shares_out).
    equity_price_series:
        Daily closing-price series used both for `sigma_E` and for
        constructing the historical `E_s` path. Length must be >= 30.
    shares_outstanding:
        Constant shares-outstanding assumption for the calibration window.
    default_pt:
        KMV default point `D` (ST debt + 0.5 * LT debt).
    risk_free_rate:
        Annualized risk-free rate to horizon `T`.
    horizon_years:
        Maturity assumption `T - t` (typically 1.0).
    history_days, max_iter, tol:
        Calibration window and outer-loop stopping rule.
    """

    if equity_market_value <= 0:
        raise ValueError(
            f"equity_market_value must be positive, got {equity_market_value}"
        )
    if default_pt <= 0:
        # Firm with no debt — model is degenerate; treat as risk-free.
        raise ValueError(
            f"default_pt must be positive for the KMV solve, got {default_pt}"
        )

    prices = select_recent_prices(equity_price_series, history_days)
    if len(prices) < 30:
        raise ValueError(
            f"iterative_kmv needs >= 30 price observations, got {len(prices)}"
        )

    equity_series = equity_price_to_market_cap(prices, shares_outstanding)
    sigma_e = realized_equity_volatility(prices)

    # Step 1: initialize sigma_V using the simple leverage-scaled equity vol.
    sigma_v = sigma_e * equity_market_value / (equity_market_value + default_pt)
    if sigma_v <= 0:
        sigma_v = max(sigma_e * 0.5, 1e-4)

    asset_series = pd.Series(dtype=float, index=equity_series.index)
    converged = False
    iterations_used = 0

    for k in range(max_iter):
        iterations_used = k + 1
        # Step 2: invert Black-Scholes for each E_s.
        v_values = np.empty(len(equity_series), dtype=float)
        for i, e_s in enumerate(equity_series.to_numpy(dtype=float)):
            v_values[i] = solve_for_v_given_e(
                e_s,
                default_pt,
                risk_free_rate,
                horizon_years,
                sigma_v,
            )
        asset_series = pd.Series(v_values, index=equity_series.index)

        # Step 3: empirical sigma_V from diff(log V).
        log_v = np.log(v_values)
        diffs = np.diff(log_v)
        new_sigma_v = float(np.std(diffs, ddof=1) * math.sqrt(_TRADING_DAYS_PER_YEAR))
        if new_sigma_v <= 0:
            new_sigma_v = sigma_v  # protect against degenerate runs

        if abs(new_sigma_v - sigma_v) < tol:
            sigma_v = new_sigma_v
            converged = True
            break
        sigma_v = new_sigma_v

    asset_value = float(asset_series.iloc[-1])
    # Resolve g1, g2 residuals at the converged point.
    g1, g2 = diagnostic_residuals(
        asset_value=asset_value,
        equity_value=float(equity_series.iloc[-1]),
        default_pt=default_pt,
        risk_free_rate=risk_free_rate,
        horizon_years=horizon_years,
        asset_volatility=sigma_v,
        equity_volatility=sigma_e,
    )

    return KMVSolution(
        asset_value=asset_value,
        asset_volatility=sigma_v,
        asset_value_series=asset_series,
        equity_volatility=sigma_e,
        default_point=default_pt,
        n_iterations=iterations_used,
        converged=converged,
        g1_residual=g1,
        g2_residual=g2,
    )


def joint_solve(
    *,
    equity_market_value: float,
    equity_volatility: float,
    default_pt: float,
    risk_free_rate: float,
    horizon_years: float,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """One-shot Newton-Raphson on the 2-equation system `(g1, g2)`.

    Returns `(V, sigma_V)`. The Jacobian is computed numerically with central
    differences; for the small 2x2 problem this is faster and more robust than
    deriving the analytic entries. Used by `MertonKMV.predict` when the caller
    explicitly opts out of the iterative loop.
    """

    if equity_market_value <= 0:
        raise ValueError(
            f"equity_market_value must be positive, got {equity_market_value}"
        )
    if default_pt <= 0:
        raise ValueError(f"default_pt must be positive, got {default_pt}")
    if equity_volatility <= 0:
        raise ValueError(
            f"equity_volatility must be positive, got {equity_volatility}"
        )

    v = equity_market_value + default_pt * math.exp(-risk_free_rate * horizon_years)
    sigma_v = (
        equity_volatility * equity_market_value / (equity_market_value + default_pt)
    )
    if sigma_v <= 0:
        sigma_v = max(equity_volatility * 0.5, 1e-4)

    def residuals(v_: float, sig_: float) -> tuple[float, float]:
        e_model = merton_call_price(
            v_, default_pt, risk_free_rate, horizon_years, sig_
        )
        sigma_e_model = equity_vol_from_asset_vol(
            v_,
            equity_market_value,
            default_pt,
            risk_free_rate,
            horizon_years,
            sig_,
        )
        return (
            e_model - equity_market_value,
            sigma_e_model - equity_volatility,
        )

    for _ in range(max_iter):
        g1, g2 = residuals(v, sigma_v)
        if abs(g1) < tol * equity_market_value and abs(g2) < tol * equity_volatility:
            return v, sigma_v

        # Numerical Jacobian via central differences.
        h_v = max(abs(v), 1.0) * 1e-5
        h_s = max(abs(sigma_v), 1e-4) * 1e-3
        g1_v_plus, g2_v_plus = residuals(v + h_v, sigma_v)
        g1_v_minus, g2_v_minus = residuals(v - h_v, sigma_v)
        g1_s_plus, g2_s_plus = residuals(v, sigma_v + h_s)
        g1_s_minus, g2_s_minus = residuals(v, sigma_v - h_s)

        j11 = (g1_v_plus - g1_v_minus) / (2.0 * h_v)
        j12 = (g1_s_plus - g1_s_minus) / (2.0 * h_s)
        j21 = (g2_v_plus - g2_v_minus) / (2.0 * h_v)
        j22 = (g2_s_plus - g2_s_minus) / (2.0 * h_s)

        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-18:
            # Singular Jacobian — fall back to a tiny gradient step.
            v_new = v - 1e-3 * g1
            sigma_v_new = sigma_v - 1e-5 * g2
        else:
            dv = (j22 * g1 - j12 * g2) / det
            ds = (-j21 * g1 + j11 * g2) / det
            v_new = v - dv
            sigma_v_new = sigma_v - ds

        if v_new <= 0:
            v_new = max(v * 0.5, 1e-6 * equity_market_value)
        if sigma_v_new <= 0:
            sigma_v_new = max(sigma_v * 0.5, 1e-4)
        v, sigma_v = v_new, sigma_v_new

    return v, sigma_v


def calibrate(
    data: MertonInputs,
    *,
    history_days: int = _DEFAULT_HISTORY_DAYS,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
    timestamp: datetime | None = None,
) -> CalibrationResult:
    """Top-level entry point: run the iterative KMV solve and package results.

    Returns a shared `CalibrationResult` whose `parameters` dict contains the
    fitted `KMVSolution` plus the asset drift `mu`. The orchestration layer
    persists this; `predict()` consumes it through the model instance.
    """

    default_pt = default_point(data.balance_sheet, weight_lt_debt=data.weight_lt_debt)
    solution = iterative_kmv(
        equity_market_value=data.equity_market_value,
        equity_price_series=data.equity_prices,
        shares_outstanding=data.shares_outstanding,
        default_pt=default_pt,
        risk_free_rate=data.risk_free_rate,
        horizon_years=data.horizon_years,
        history_days=history_days,
        max_iter=max_iter,
        tol=tol,
    )
    mu_trailing = estimate_drift_from_assets(solution.asset_value_series)

    fit_metrics: dict[str, float] = {
        "asset_value": solution.asset_value,
        "asset_volatility": solution.asset_volatility,
        "equity_volatility": solution.equity_volatility,
        "default_point": solution.default_point,
        "drift_trailing": mu_trailing,
        "n_iterations": float(solution.n_iterations),
        "converged": 1.0 if solution.converged else 0.0,
        "g1_residual": solution.g1_residual,
        "g2_residual": solution.g2_residual,
        "leverage": solution.default_point / solution.asset_value,
    }

    parameters: dict[str, object] = {
        "solution": solution,
        "drift_trailing": mu_trailing,
        "risk_free_rate": data.risk_free_rate,
        "horizon_years": data.horizon_years,
    }

    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=timestamp or data.timestamp,
        metadata={
            "ticker": data.ticker,
            "industry": data.industry,
            "history_days": history_days,
        },
    )
