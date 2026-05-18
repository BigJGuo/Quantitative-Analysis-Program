"""Pure-math layer for Almgren-Chriss optimal execution.

Everything here is ``numpy`` / ``pandas`` only — no I/O, no class state. The
``AlmgrenChriss`` orchestration shell in ``model.py`` composes these
functions to build ``ExecutionSchedule`` / ``EfficientFrontier`` /
``MultiAssetSchedule`` outputs.

Sigma convention
----------------
All cost / variance functions here take ``sigma_d``, the **dollar price
volatility per square-root of trading-day** (i.e. so that
``sigma_d * sqrt(T)`` is the standard deviation of the price drift over
``T`` trading-day-fractions). Convert from annualized return vol via
``sigma_d = S_0 * sigma_annual / sqrt(252)``; the helper
``annual_return_vol_to_dollar_per_day`` does this.

Closed-form formulas (derived in the spec; the ``±κT`` term in the spec's
line 102 vs. our derivation: the spec swapped E and V; we use the
textbook form below — both reduce to the TWAP limits ``η X²/T`` and
``σ² X² T/3``):

- ``E_temp(λ) = (η X² κ) / (2 sinh²(κT))  ·  [sinh(κT) cosh(κT) + κT]``
- ``V(λ)    = (σ² X²)  / (2 κ sinh²(κT)) ·  [sinh(κT) cosh(κT) − κT]``

Numerical stability
-------------------
For ``κT`` larger than ``_SINH_OVERFLOW_THRESHOLD`` (≈ 30), ``sinh(κT)``
overflows in double precision. We switch to the asymptotic
``x_t* ≈ X exp(−κt)``, ``E_temp ≈ η X² κ / 2``, ``V ≈ σ² X² / (2κ)``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.models.almgren_chriss.types import (
    DiscretizationMode,
    EfficientFrontier,
    EfficientFrontierPoint,
    ExecutionSchedule,
    ImpactParams,
    LiquidationProblem,
    MultiAssetProblem,
    MultiAssetSchedule,
)

_TRADING_DAYS_PER_YEAR: int = 252
_SINH_OVERFLOW_THRESHOLD: float = 30.0
_KAPPA_T_NEAR_ZERO: float = 1e-6
_NEWTON_MAX_ITER: int = 80
_NEWTON_TOL: float = 1e-12


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def annual_return_vol_to_dollar_per_day(sigma_annual: float, price: float) -> float:
    """Convert annualized log-return vol to per-day dollar price vol.

    ``dS_t = sigma_d dW_t`` in trading-day time, where
    ``sigma_d = price * sigma_annual / sqrt(252)``.
    """

    if sigma_annual <= 0 or price <= 0:
        raise ValueError(
            f"sigma_annual and price must be > 0, got {sigma_annual}, {price}"
        )
    return float(price * sigma_annual / math.sqrt(_TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# Stable sinh-ratio helpers
# ---------------------------------------------------------------------------


def sinh_ratio(a: float, b: float) -> float:
    """Compute ``sinh(a) / sinh(b)`` in a numerically stable way.

    For ``b`` large (``>= _SINH_OVERFLOW_THRESHOLD``) the explicit
    formula overflows in double precision; we factor out ``exp(b)``:

      ``sinh(a) / sinh(b) = e^{a-b} · (1 − e^{−2a}) / (1 − e^{−2b})``

    Valid for ``a, b >= 0``; for ``a < 0`` the caller flips the sign.
    """

    if b <= 0:
        raise ValueError(f"sinh_ratio requires b > 0, got {b}")
    if a == 0.0:
        return 0.0
    if a < 0:
        return -sinh_ratio(-a, b)
    if max(a, b) < _SINH_OVERFLOW_THRESHOLD:
        sb = math.sinh(b)
        if sb == 0.0:
            raise ValueError("sinh_ratio: denominator sinh(b) is zero")
        return math.sinh(a) / sb
    # Stable form for large arguments.
    num = 1.0 - math.exp(-2.0 * a)
    den = 1.0 - math.exp(-2.0 * b)
    return math.exp(a - b) * (num / den)


def _log_sinh(x: float) -> float:
    """``log(sinh(x))`` stably for ``x > 0``.

    For ``x >= _SINH_OVERFLOW_THRESHOLD``, ``sinh(x) ≈ exp(x) / 2``.
    """

    if x <= 0:
        raise ValueError(f"_log_sinh requires x > 0, got {x}")
    if x < _SINH_OVERFLOW_THRESHOLD:
        return math.log(math.sinh(x))
    # log(sinh(x)) = x + log((1 - exp(-2x)) / 2) ≈ x − log(2)
    return x + math.log1p(-math.exp(-2.0 * x)) - math.log(2.0)


# ---------------------------------------------------------------------------
# Core scalar primitives
# ---------------------------------------------------------------------------


def compute_kappa(sigma_d: float, eta: float, lam: float) -> float:
    """``κ = sqrt(λ σ_d² / η)`` — continuous-time decay rate."""

    if eta <= 0 or sigma_d <= 0 or lam <= 0:
        raise ValueError(
            f"compute_kappa: all inputs must be > 0, got "
            f"sigma_d={sigma_d}, eta={eta}, lam={lam}"
        )
    return math.sqrt(lam * sigma_d * sigma_d / eta)


def discrete_kappa(kappa_continuous: float, tau: float) -> float:
    """Solve ``2 (cosh(κ̃ τ) − 1) = (κ τ)²`` for ``κ̃``.

    The discrete optimum (Algorithm B) replaces ``κ`` with ``κ̃`` where
    the equation arises from the second-order finite-difference form of
    the Euler-Lagrange equation. As ``τ → 0``, ``κ̃ → κ``. We solve via
    Newton's method on ``f(u) = 2 (cosh(u) − 1) − (κτ)²`` with
    ``u = κ̃ τ``.
    """

    if kappa_continuous <= 0:
        raise ValueError(
            f"discrete_kappa requires kappa_continuous > 0, got {kappa_continuous}"
        )
    if tau <= 0:
        raise ValueError(f"discrete_kappa requires tau > 0, got {tau}")
    target = (kappa_continuous * tau) ** 2
    u = kappa_continuous * tau  # initial guess = continuous solution
    for _ in range(_NEWTON_MAX_ITER):
        f = 2.0 * (math.cosh(u) - 1.0) - target
        fp = 2.0 * math.sinh(u)
        if fp == 0.0:
            break
        step = f / fp
        u_new = u - step
        if abs(step) < _NEWTON_TOL * max(abs(u), 1.0):
            u = u_new
            break
        u = u_new
    return u / tau


def expected_temporary_cost(
    X: float, T: float, kappa: float, eta: float
) -> float:
    """Closed-form ``η ∫_0^T v_t² dt`` for the optimal continuous schedule.

    ``E_temp = (η X² κ) / (2 sinh²(κT)) · [sinh(κT) cosh(κT) + κT]``

    Numerical stability: for ``κT >= _SINH_OVERFLOW_THRESHOLD`` we use
    the asymptotic ``E_temp ≈ η X² κ / 2`` (front-loaded limit).
    """

    if kappa <= 0:
        # κ→0 limit (TWAP): E_temp = η X² / T.
        return eta * X * X / T
    kt = kappa * T
    if kt < _KAPPA_T_NEAR_ZERO:
        return eta * X * X / T
    if kt >= _SINH_OVERFLOW_THRESHOLD:
        # sinh(κT) cosh(κT) / sinh²(κT) → 1; κT / sinh²(κT) → 0.
        return 0.5 * eta * X * X * kappa
    sh = math.sinh(kt)
    ch = math.cosh(kt)
    return eta * X * X * kappa / (2.0 * sh * sh) * (sh * ch + kt)


def cost_variance_closed_form(
    X: float, T: float, kappa: float, sigma_d: float
) -> float:
    """Closed-form ``σ² ∫_0^T x_t² dt`` for the optimal continuous schedule.

    ``V = (σ² X²) / (2 κ sinh²(κT)) · [sinh(κT) cosh(κT) − κT]``

    Numerical stability: for ``κT >= _SINH_OVERFLOW_THRESHOLD`` we use
    the asymptotic ``V ≈ σ² X² / (2 κ)``.
    """

    if kappa <= 0:
        # κ→0 limit (TWAP): V = σ² X² T / 3.
        return sigma_d * sigma_d * X * X * T / 3.0
    kt = kappa * T
    if kt < _KAPPA_T_NEAR_ZERO:
        return sigma_d * sigma_d * X * X * T / 3.0
    if kt >= _SINH_OVERFLOW_THRESHOLD:
        return 0.5 * sigma_d * sigma_d * X * X / kappa
    sh = math.sinh(kt)
    ch = math.cosh(kt)
    return (
        sigma_d * sigma_d * X * X / (2.0 * kappa * sh * sh) * (sh * ch - kt)
    )


def permanent_cost(X: float, gamma: float) -> float:
    """``γ X² / 2`` — path-independent permanent-impact cost."""

    return 0.5 * gamma * X * X


# ---------------------------------------------------------------------------
# Inventory paths (continuous and discrete)
# ---------------------------------------------------------------------------


def inventory_path_continuous(
    X: float, T: float, N: int, kappa: float
) -> np.ndarray:
    """``x_k = X · sinh(κ(T − k τ)) / sinh(κT)`` for ``k = 0..N``."""

    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    tau = T / N
    k = np.arange(N + 1, dtype=float)
    t = k * tau
    if kappa <= 0 or kappa * T < _KAPPA_T_NEAR_ZERO:
        return X * (1.0 - t / T)
    if kappa * T >= _SINH_OVERFLOW_THRESHOLD:
        # Asymptotic: x_k ≈ X exp(−κ t) for t < T; clamp endpoint to 0.
        x = X * np.exp(-kappa * t)
        x[-1] = 0.0
        return x
    out = np.empty(N + 1, dtype=float)
    sh_total = math.sinh(kappa * T)
    for i, ti in enumerate(t):
        if i == N:
            out[i] = 0.0
        else:
            out[i] = X * math.sinh(kappa * (T - ti)) / sh_total
    return out


def inventory_path_discrete(
    X: float, T: float, N: int, kappa_continuous: float
) -> tuple[np.ndarray, float]:
    """Algorithm B: use ``κ̃`` such that the discrete difference equation
    is exact, then evaluate ``x_k = X · sinh(κ̃(T − kτ)) / sinh(κ̃ T)``.

    Returns ``(x_path, kappa_tilde)``.
    """

    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    tau = T / N
    kt_cont = kappa_continuous * tau
    if kappa_continuous <= 0 or kt_cont < _KAPPA_T_NEAR_ZERO:
        # κ→0 limit: TWAP regardless of discretization.
        k = np.arange(N + 1, dtype=float)
        return X * (1.0 - k / N), 0.0
    kappa_tilde = discrete_kappa(kappa_continuous, tau)
    x = inventory_path_continuous(X, T, N, kappa_tilde)
    return x, kappa_tilde


# ---------------------------------------------------------------------------
# Top-level schedule builder
# ---------------------------------------------------------------------------


def optimal_schedule(
    problem: LiquidationProblem,
    params: ImpactParams,
    *,
    discretization: DiscretizationMode = "continuous",
    timestamp: datetime | None = None,
) -> ExecutionSchedule:
    """Build the Almgren-Chriss optimal schedule for a single asset.

    Implements Algorithm A (``discretization="continuous"``) or
    Algorithm B (``"discrete"``).
    """

    ts = timestamp or datetime.now(UTC)
    if discretization not in {"continuous", "discrete"}:
        raise ValueError(
            f"discretization must be 'continuous' or 'discrete', "
            f"got {discretization!r}"
        )
    sigma_d = annual_return_vol_to_dollar_per_day(params.sigma, params.last_price)
    kappa = compute_kappa(sigma_d, params.eta, problem.lam)
    tau = problem.T / problem.N

    if discretization == "continuous":
        x_path = inventory_path_continuous(problem.X, problem.T, problem.N, kappa)
        kappa_used = kappa
    else:
        x_path, kappa_used = inventory_path_discrete(
            problem.X, problem.T, problem.N, kappa
        )
        if kappa_used <= 0:
            kappa_used = kappa  # TWAP fallback

    n_path = x_path[:-1] - x_path[1:]
    v_path = n_path / tau
    times = np.arange(problem.N + 1, dtype=float) * tau

    e_temp = expected_temporary_cost(problem.X, problem.T, kappa_used, params.eta)
    e_perm = permanent_cost(problem.X, params.gamma)
    e_cost = e_temp + e_perm
    v_cost = cost_variance_closed_form(problem.X, problem.T, kappa_used, sigma_d)
    cost_bps = (
        e_cost / (problem.X * params.last_price) * 1.0e4
        if problem.X * params.last_price > 0
        else float("nan")
    )

    return ExecutionSchedule(
        ticker=problem.ticker,
        side=problem.side,
        discretization=discretization,
        X=problem.X,
        T=problem.T,
        N=problem.N,
        tau=tau,
        times=times,
        inventory=x_path,
        child_orders=n_path,
        trading_rate=v_path,
        expected_cost=float(e_cost),
        cost_variance=float(v_cost),
        expected_cost_bps=float(cost_bps),
        kappa=float(kappa_used),
        kappa_T=float(kappa_used * problem.T),
        params=params,
        lam=problem.lam,
        timestamp=ts,
        metadata={
            "kappa_continuous": float(kappa),
            "sigma_dollar_per_day": float(sigma_d),
            "expected_temporary_cost": float(e_temp),
            "expected_permanent_cost": float(e_perm),
        },
    )


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------


def efficient_frontier(
    X: float,
    T: float,
    params: ImpactParams,
    *,
    lambdas: np.ndarray | tuple[float, ...] | list[float],
    gamma: float | None = None,
) -> EfficientFrontier:
    """Sweep ``λ`` and trace ``(V[C], E[C])`` for the optimal schedules.

    The expected and variance use the **continuous closed forms** (the
    appropriate basis for the frontier — discretization is a fixed-N
    detail that doesn't change the trade-off curve in the limit).

    ``gamma`` defaults to ``params.gamma``; pass an override only when
    studying frontiers with a counterfactual permanent-impact level.
    """

    if X <= 0:
        raise ValueError(f"X must be > 0, got {X}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if len(lambdas) == 0:
        raise ValueError("efficient_frontier: lambdas must be non-empty")
    g = params.gamma if gamma is None else gamma
    sigma_d = annual_return_vol_to_dollar_per_day(params.sigma, params.last_price)
    pts: list[EfficientFrontierPoint] = []
    for lam in lambdas:
        lam_f = float(lam)
        if lam_f <= 0:
            raise ValueError(
                f"efficient_frontier: lambdas must be > 0, got {lam_f}"
            )
        kappa = compute_kappa(sigma_d, params.eta, lam_f)
        e_temp = expected_temporary_cost(X, T, kappa, params.eta)
        e_perm = 0.5 * g * X * X
        e_cost = e_temp + e_perm
        v_cost = cost_variance_closed_form(X, T, kappa, sigma_d)
        bps = e_cost / (X * params.last_price) * 1.0e4
        pts.append(
            EfficientFrontierPoint(
                lam=lam_f,
                kappa=float(kappa),
                kappa_T=float(kappa * T),
                expected_cost=float(e_cost),
                cost_variance=float(v_cost),
                expected_cost_bps=float(bps),
            )
        )
    return EfficientFrontier(points=tuple(pts))


# ---------------------------------------------------------------------------
# Intraday volume profile (Algorithm C input)
# ---------------------------------------------------------------------------


def intraday_volume_profile(
    intraday: pd.DataFrame,
    *,
    bars_per_day: int,
) -> np.ndarray:
    """Estimate the per-bin mean intraday volume share, summing to 1.

    Expects a ``Volume`` column and a ``Datetime`` (or index) column with
    bar timestamps. We group bars by their **position within the trading
    day**, average the share across days, and renormalize to sum to 1.

    Falls back to a flat profile if any of these go wrong:
    - empty frame
    - no ``Volume`` column
    - fewer than ``bars_per_day`` rows on any individual day
    """

    flat = np.full(bars_per_day, 1.0 / bars_per_day)
    if intraday is None or intraday.empty or "Volume" not in intraday.columns:
        return flat
    df = intraday.copy()
    dt_series: pd.Series
    if "Datetime" in df.columns:
        dt_series = pd.to_datetime(df["Datetime"])
    elif "Date" in df.columns:
        dt_series = pd.to_datetime(df["Date"])
    else:
        dt_series = pd.Series(pd.to_datetime(df.index), index=df.index)
    df = df.assign(_dt=dt_series.values).dropna(subset=["_dt", "Volume"])
    if df.empty:
        return flat
    df["_day"] = df["_dt"].dt.date
    df["_bin"] = df.groupby("_day").cumcount()
    df = df[df["_bin"] < bars_per_day]
    df = df[df["Volume"] > 0]
    if df.empty:
        return flat
    # Mean volume share per bin across days.
    grouped = df.groupby(["_day", "_bin"])["Volume"].sum().unstack(fill_value=0)
    day_totals = grouped.sum(axis=1)
    day_totals = day_totals[day_totals > 0]
    if day_totals.empty:
        return flat
    grouped = grouped.loc[day_totals.index]
    shares = grouped.div(day_totals, axis=0)
    mean_share = shares.mean(axis=0)
    profile = np.zeros(bars_per_day, dtype=float)
    for bin_idx, share in mean_share.items():
        bin_i = int(bin_idx)  # type: ignore[call-overload]
        if 0 <= bin_i < bars_per_day:
            profile[bin_i] = float(share)
    total = profile.sum()
    if total <= 0:
        return flat
    return np.asarray(profile / total, dtype=float)


def resample_profile(profile: np.ndarray, N: int) -> np.ndarray:
    """Resample a length-K volume profile to length ``N`` (normalized to 1).

    Uses simple bucket aggregation: split ``[0, 1)`` into ``N`` equal
    sub-intervals and integrate the piecewise-constant profile over
    each. Works for any ``N`` (smaller or larger than ``K``).
    """

    K = profile.size
    if K == 0:
        raise ValueError("resample_profile: profile must be non-empty")
    if N < 1:
        raise ValueError(f"resample_profile: N must be >= 1, got {N}")
    if K == N:
        s = profile.sum()
        return profile / s if s > 0 else np.full(N, 1.0 / N)
    edges_old = np.linspace(0.0, 1.0, K + 1)
    edges_new = np.linspace(0.0, 1.0, N + 1)
    out = np.zeros(N, dtype=float)
    for j in range(N):
        a, b = edges_new[j], edges_new[j + 1]
        # Sum profile[i] * overlap_length((a,b), (edges_old[i], edges_old[i+1])) * K
        lo = np.searchsorted(edges_old, a, side="right") - 1
        hi = np.searchsorted(edges_old, b, side="left")
        total = 0.0
        for i in range(max(lo, 0), min(hi, K)):
            left = max(a, edges_old[i])
            right = min(b, edges_old[i + 1])
            if right > left:
                # profile[i] is share over a 1/K-width bin; per unit length it's K * profile[i]
                total += profile[i] * K * (right - left)
        out[j] = total
    s = out.sum()
    return out / s if s > 0 else np.full(N, 1.0 / N)


# ---------------------------------------------------------------------------
# VWAP-shaped (Algorithm C)
# ---------------------------------------------------------------------------


def vwap_shaped_schedule(
    problem: LiquidationProblem,
    params: ImpactParams,
    *,
    volume_profile: np.ndarray,
    timestamp: datetime | None = None,
) -> ExecutionSchedule:
    """Algorithm C: volume-weighted Almgren-Chriss.

    Per-slice impact ``η_k = η_0 / u_k`` where ``u_k`` is the local volume
    share (normalized so ``Σ u_k = 1``, then scaled to mean 1 across slices
    so the *average* slice has impact ``η_0``).

    The unconstrained equality-constrained quadratic minimization
    reduces to a tridiagonal linear system in the interior knots
    ``x_1, ..., x_{N−1}`` — solved directly via ``numpy.linalg.solve``.
    Non-negativity is checked post-hoc; for sensible parameters the
    solution is monotone decreasing and the constraint is inactive.
    """

    ts = timestamp or datetime.now(UTC)
    N = problem.N
    tau = problem.T / N
    if volume_profile.shape != (N,):
        # Caller can resample first; if not, do it here.
        volume_profile = resample_profile(volume_profile, N)
    if np.any(volume_profile <= 0):
        raise ValueError("vwap_shaped_schedule: volume_profile must be > 0")
    # Normalize so mean(u) = 1. That way the *average* η_k equals η_0.
    u = volume_profile / volume_profile.mean()
    eta_k = params.eta / u  # shape (N,)

    sigma_d = annual_return_vol_to_dollar_per_day(params.sigma, params.last_price)
    lam_sigma2_tau2 = problem.lam * sigma_d * sigma_d * tau * tau

    if N == 1:
        x_path = np.asarray([problem.X, 0.0], dtype=float)
    else:
        # Build tridiagonal system A x_int = b for x_int = (x_1, ..., x_{N-1}).
        # Diag:    A[m,m]   = eta_{m+1} + eta_{m+2} + λ σ² τ²   (m = 0..N-2)
        # Lower:   A[m,m-1] = -eta_{m+1}
        # Upper:   A[m,m+1] = -eta_{m+2}
        # b[0]     = eta_1 * X       ; b[N-2] = 0
        # Indexing: eta_k corresponds to slice k=1..N → eta_k = eta_k[k-1] in 0-indexed array.
        n_int = N - 1
        diag = eta_k[:-1] + eta_k[1:] + lam_sigma2_tau2  # length n_int
        off = -eta_k[1:-1]  # length n_int - 1
        # Construct dense tridiagonal matrix (small N — full solve is fine).
        A = np.zeros((n_int, n_int), dtype=float)
        for i in range(n_int):
            A[i, i] = diag[i]
            if i > 0:
                A[i, i - 1] = off[i - 1]
            if i < n_int - 1:
                A[i, i + 1] = off[i]
        b = np.zeros(n_int, dtype=float)
        b[0] = eta_k[0] * problem.X
        x_int = np.linalg.solve(A, b)
        x_path = np.empty(N + 1, dtype=float)
        x_path[0] = problem.X
        x_path[1:N] = x_int
        x_path[N] = 0.0

    n_path = x_path[:-1] - x_path[1:]
    v_path = n_path / tau
    times = np.arange(N + 1, dtype=float) * tau

    # Cost computation (discretized objective).
    e_temp = float(np.sum(eta_k * v_path * v_path) * tau)
    e_perm = permanent_cost(problem.X, params.gamma)
    e_cost = e_temp + e_perm
    # Variance: discretized σ² Σ x_k² τ, summed over k=1..N (x_N = 0).
    v_cost = float(sigma_d * sigma_d * np.sum(x_path[1:] ** 2) * tau)
    cost_bps = (
        e_cost / (problem.X * params.last_price) * 1.0e4
        if problem.X * params.last_price > 0
        else float("nan")
    )

    # Effective kappa (use the average eta).
    kappa_eff = compute_kappa(sigma_d, params.eta, problem.lam)

    monotone = bool(np.all(np.diff(x_path) <= 1e-9))
    nonneg = bool(np.all(x_path >= -1e-9))

    return ExecutionSchedule(
        ticker=problem.ticker,
        side=problem.side,
        discretization="discrete",
        X=problem.X,
        T=problem.T,
        N=problem.N,
        tau=tau,
        times=times,
        inventory=x_path,
        child_orders=n_path,
        trading_rate=v_path,
        expected_cost=float(e_cost),
        cost_variance=float(v_cost),
        expected_cost_bps=float(cost_bps),
        kappa=float(kappa_eff),
        kappa_T=float(kappa_eff * problem.T),
        params=params,
        lam=problem.lam,
        timestamp=ts,
        metadata={
            "algorithm": "vwap_shaped",
            "volume_profile_normalized": u.tolist(),
            "eta_per_slice": eta_k.tolist(),
            "monotone": monotone,
            "nonnegative": nonneg,
        },
    )


# ---------------------------------------------------------------------------
# Multi-asset (Algorithm D)
# ---------------------------------------------------------------------------


def multi_asset_schedule(
    problem: MultiAssetProblem,
    *,
    sigma_cov_daily: np.ndarray,
    eta_diag: np.ndarray,
    gamma_diag: np.ndarray,
    last_prices: np.ndarray,
    daily_volumes: np.ndarray,
    timestamp: datetime | None = None,
) -> MultiAssetSchedule:
    """Algorithm D: basket liquidation in the eigenbasis of ``η^{−1/2} Σ η^{−1/2}``.

    ``sigma_cov_daily``: M×M return-covariance matrix (decimal, daily).
    ``eta_diag`` / ``gamma_diag``: per-asset impact coefficients, shape (M,).
    ``last_prices`` / ``daily_volumes``: shape (M,).

    The return-covariance is converted to dollar-price covariance
    ``Σ_d = diag(P) Σ_r diag(P)`` (per trading-day), and the problem is
    solved in the eigenbasis.
    """

    ts = timestamp or datetime.now(UTC)
    M = len(problem.tickers)
    if sigma_cov_daily.shape != (M, M):
        raise ValueError(
            f"sigma_cov_daily must have shape ({M}, {M}), "
            f"got {sigma_cov_daily.shape}"
        )
    if eta_diag.shape != (M,):
        raise ValueError(f"eta_diag must have shape ({M},), got {eta_diag.shape}")
    if gamma_diag.shape != (M,):
        raise ValueError(
            f"gamma_diag must have shape ({M},), got {gamma_diag.shape}"
        )
    if last_prices.shape != (M,):
        raise ValueError(
            f"last_prices must have shape ({M},), got {last_prices.shape}"
        )
    if daily_volumes.shape != (M,):
        raise ValueError(
            f"daily_volumes must have shape ({M},), got {daily_volumes.shape}"
        )
    if np.any(eta_diag <= 0):
        raise ValueError("eta_diag entries must all be > 0")
    if np.any(last_prices <= 0):
        raise ValueError("last_prices entries must all be > 0")

    # Convert return covariance to dollar-price covariance per trading-day.
    P = np.diag(last_prices)
    Sigma_d = P @ sigma_cov_daily @ P  # $^2 per trading-day
    eta_half = np.sqrt(eta_diag)
    eta_inv_half = 1.0 / eta_half
    M_mat = np.outer(eta_inv_half, eta_inv_half) * Sigma_d  # η^{-1/2} Σ_d η^{-1/2}
    # Symmetrize against floating-point drift.
    M_mat = 0.5 * (M_mat + M_mat.T)
    eigvals, U = np.linalg.eigh(M_mat)
    eigvals = np.maximum(eigvals, 0.0)
    kappas = np.sqrt(problem.lam * eigvals)
    # X in eigenbasis: y_0 = U^T η^{1/2} X
    X_vec = np.asarray(problem.X, dtype=float)
    y0 = U.T @ (eta_half * X_vec)

    tau = problem.T / problem.N
    times = np.arange(problem.N + 1, dtype=float) * tau

    # Build per-mode decay factors d_i(t) = sinh(κ_i (T − t)) / sinh(κ_i T).
    decay = np.zeros((problem.N + 1, M), dtype=float)
    for i in range(M):
        k_i = float(kappas[i])
        if k_i <= 0 or k_i * problem.T < _KAPPA_T_NEAR_ZERO:
            decay[:, i] = 1.0 - times / problem.T
        elif k_i * problem.T >= _SINH_OVERFLOW_THRESHOLD:
            decay[:, i] = np.exp(-k_i * times)
            decay[-1, i] = 0.0
        else:
            sh_total = math.sinh(k_i * problem.T)
            for j, tj in enumerate(times):
                if j == problem.N:
                    decay[j, i] = 0.0
                else:
                    decay[j, i] = math.sinh(k_i * (problem.T - tj)) / sh_total

    # y_t = diag(d_i(t)) y_0 ; x_t = η^{−1/2} U y_t
    y_path = decay * y0[None, :]  # (N+1, M)
    x_path = (eta_inv_half[None, :]) * (y_path @ U.T)  # (N+1, M)
    # Pin boundaries exactly.
    x_path[0, :] = X_vec
    x_path[-1, :] = 0.0
    n_path = x_path[:-1, :] - x_path[1:, :]

    # Cost: in the eigenbasis the modes decouple.
    e_temp_total = 0.0
    v_total = 0.0
    for i in range(M):
        k_i = float(kappas[i])
        if k_i <= 0:
            # TWAP mode
            e_temp_total += y0[i] * y0[i] / problem.T
            v_total += eigvals[i] * y0[i] * y0[i] * problem.T / 3.0
        else:
            kt = k_i * problem.T
            if kt < _KAPPA_T_NEAR_ZERO:
                e_temp_total += y0[i] * y0[i] / problem.T
                v_total += eigvals[i] * y0[i] * y0[i] * problem.T / 3.0
            elif kt >= _SINH_OVERFLOW_THRESHOLD:
                e_temp_total += 0.5 * y0[i] * y0[i] * k_i
                v_total += 0.5 * eigvals[i] * y0[i] * y0[i] / k_i
            else:
                sh = math.sinh(kt)
                ch = math.cosh(kt)
                e_temp_total += (
                    y0[i] * y0[i] * k_i / (2.0 * sh * sh) * (sh * ch + kt)
                )
                v_total += (
                    eigvals[i]
                    * y0[i]
                    * y0[i]
                    / (2.0 * k_i * sh * sh)
                    * (sh * ch - kt)
                )
    # Permanent impact (γ_i X_i² / 2 per asset).
    e_perm = 0.5 * float(np.sum(gamma_diag * X_vec * X_vec))
    e_cost = float(e_temp_total + e_perm)
    v_cost = float(v_total)

    return MultiAssetSchedule(
        tickers=problem.tickers,
        sides=problem.sides,
        T=problem.T,
        N=problem.N,
        tau=tau,
        times=times,
        inventory=x_path,
        child_orders=n_path,
        expected_cost=e_cost,
        cost_variance=v_cost,
        kappas=kappas,
        lam=problem.lam,
        timestamp=ts,
        metadata={
            "eigenvalues": eigvals.tolist(),
            "expected_temporary_cost": float(e_temp_total),
            "expected_permanent_cost": float(e_perm),
            "daily_volumes": daily_volumes.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def schedule_diagnostics(schedule: ExecutionSchedule) -> dict[str, Any]:
    """Spec validation block 1, 2, 4, 5, 7 condensed into a dict.

    1. Monotone decreasing inventory, n_k > 0, Σ n_k == X.
    2. kappa_T in [0.3, 3] for "typical production".
    4. (Frontier check handled by `efficient_frontier` consumer.)
    5. (Stress test handled by caller.)
    7. Numerical-stability flag for kappa_T > 30.
    """

    x = schedule.inventory
    n = schedule.child_orders
    sum_n = float(np.sum(n))
    monotone = bool(np.all(np.diff(x) <= 1e-9))
    positive_children = bool(np.all(n >= -1e-9))
    sum_check = abs(sum_n - schedule.X) / max(schedule.X, 1.0)
    kappa_T = schedule.kappa_T
    regime: str
    if kappa_T < 0.1:
        regime = "near_twap"
    elif kappa_T < 0.3:
        regime = "twap_leaning"
    elif kappa_T <= 3.0:
        regime = "balanced"
    elif kappa_T <= 5.0:
        regime = "front_loaded"
    else:
        regime = "block_trade"
    return {
        "monotone_inventory": monotone,
        "positive_children": positive_children,
        "sum_n_minus_X_relative": float(sum_check),
        "kappa_T": float(kappa_T),
        "kappa": float(schedule.kappa),
        "regime": regime,
        "expected_cost": float(schedule.expected_cost),
        "expected_cost_bps": float(schedule.expected_cost_bps),
        "cost_std": float(schedule.cost_std),
        "numerical_stability_warning": bool(kappa_T > _SINH_OVERFLOW_THRESHOLD),
    }


def stress_schedule(
    problem: LiquidationProblem,
    params: ImpactParams,
    *,
    sigma_multiplier: float = 3.0,
) -> ExecutionSchedule:
    """Spec validation 5: re-run with ``σ`` scaled by ``sigma_multiplier``.

    A correctly-implemented schedule should accelerate (kappa_T grows by
    ``sigma_multiplier``).
    """

    if sigma_multiplier <= 0:
        raise ValueError(
            f"sigma_multiplier must be > 0, got {sigma_multiplier}"
        )
    stressed = ImpactParams(
        sigma=params.sigma * sigma_multiplier,
        eta=params.eta,
        gamma=params.gamma,
        last_price=params.last_price,
        daily_volume=params.daily_volume,
    )
    return optimal_schedule(problem, stressed, discretization="continuous")


__all__ = [
    "annual_return_vol_to_dollar_per_day",
    "compute_kappa",
    "cost_variance_closed_form",
    "discrete_kappa",
    "efficient_frontier",
    "expected_temporary_cost",
    "intraday_volume_profile",
    "inventory_path_continuous",
    "inventory_path_discrete",
    "multi_asset_schedule",
    "optimal_schedule",
    "permanent_cost",
    "resample_profile",
    "schedule_diagnostics",
    "sinh_ratio",
    "stress_schedule",
    "vwap_shaped_schedule",
]
