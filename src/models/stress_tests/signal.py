"""Pure math layer of the stress-tests risk model.

Nothing in this module performs I/O. Every function is a deterministic
transformation of arrays / Series, so the math is unit-testable with
synthetic data.

The functions roughly mirror the spec's Algorithm Outline:

- ``window_factor_change``                                -> Routine A step 1
- ``apply_proxy_fill``                                    -> Routine A step 2
- ``historical_replay_pnl``                               -> Routine A step 3
- ``estimate_betas_ols``                                  -> Routine B beta map
- ``hypothetical_pnl``                                    -> Routine B step 2
- ``reverse_stress_shock``                                -> Routine C
- ``scenario_coverage_matrix``, ``top_contributors``,
  ``window_endpoint_sensitivity``                         -> validation
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.stress_tests.types import (
    CrisisWindow,
    HypotheticalScenario,
    ReverseStressResult,
    ScenarioPnL,
)

_PLAUSIBILITY_THRESHOLD: float = 4.0  # spec: < 4 sigma Mahalanobis = plausible


def slice_window(series: pd.Series, t1: str, t2: str) -> pd.Series:
    """Return the slice of `series` that falls in the closed range `[t1, t2]`.

    Index must be datetime-like; out-of-range queries return an empty Series.
    """

    if series is None or series.empty:
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex(series.index)
    mask = (idx >= pd.Timestamp(t1)) & (idx <= pd.Timestamp(t2))
    return series.loc[mask].dropna()


def window_factor_change(series: pd.Series) -> float:
    """Percentage change between the first and last observation of `series`.

    Returns NaN if the input has fewer than two non-null points — the caller
    is responsible for routing to a proxy in that case.
    """

    cleaned = series.dropna()
    if len(cleaned) < 2:
        return float("nan")
    first = float(cleaned.iloc[0])
    last = float(cleaned.iloc[-1])
    if first == 0.0:
        return float("nan")
    return last / first - 1.0


def apply_proxy_fill(
    *,
    tickers: list[str],
    direct_changes: dict[str, float],
    proxy_changes: dict[str, float],
    fallback_change: float,
    sector_map: dict[str, str],
    betas_to_fallback: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Combine direct, sector-proxy, and benchmark fallback per-ticker changes.

    For each ticker in `tickers`:

    1. If a finite direct change exists, use it (label ``"direct"``).
    2. Else if `sector_map` points to a proxy with a finite change, use it
       (label ``"sector_proxy"``).
    3. Else fall back to `fallback_change` scaled by `betas_to_fallback[ticker]`
       (default 1.0); label ``"benchmark_beta"``. ``"benchmark"`` if no beta.

    Returns ``(filled_changes, source_label_per_ticker)``.
    """

    filled: dict[str, float] = {}
    source: dict[str, str] = {}
    betas = betas_to_fallback or {}
    for ticker in tickers:
        direct = direct_changes.get(ticker)
        if direct is not None and not math.isnan(direct):
            filled[ticker] = direct
            source[ticker] = "direct"
            continue
        proxy_ticker = sector_map.get(ticker)
        if proxy_ticker is not None:
            proxy_change = proxy_changes.get(proxy_ticker)
            if proxy_change is not None and not math.isnan(proxy_change):
                filled[ticker] = proxy_change
                source[ticker] = f"sector_proxy:{proxy_ticker}"
                continue
        if math.isnan(fallback_change):
            # No benchmark data either — mark the ticker as missing.
            filled[ticker] = 0.0
            source[ticker] = "missing"
            continue
        beta = betas.get(ticker)
        if beta is None:
            filled[ticker] = fallback_change
            source[ticker] = "benchmark"
        else:
            filled[ticker] = beta * fallback_change
            source[ticker] = f"benchmark_beta:{beta:.3f}"
    return filled, source


def historical_replay_pnl(
    *,
    window: CrisisWindow,
    positions: dict[str, float],
    factor_changes: dict[str, float],
    source: dict[str, str] | None = None,
) -> ScenarioPnL:
    """Convert per-ticker percentage changes into dollar P&L.

    `factor_changes[ticker]` is the percentage move of `ticker` over the
    crisis window. `positions[ticker]` is the current dollar exposure (signed).
    Total P&L is the dot product; per-ticker contributions are reported.
    """

    contributions: dict[str, float] = {}
    total = 0.0
    for ticker, dollars in positions.items():
        change = factor_changes.get(ticker)
        if change is None or math.isnan(change):
            contributions[ticker] = 0.0
            continue
        pnl = float(dollars) * float(change)
        contributions[ticker] = pnl
        total += pnl
    metadata: dict[str, object] = {"t1": window.t1, "t2": window.t2}
    if source is not None:
        metadata["source"] = source
    return ScenarioPnL(
        scenario_name=window.name,
        scenario_type="historical",
        total_pnl=total,
        contributions=contributions,
        factor_changes=dict(factor_changes),
        metadata=metadata,
    )


def estimate_betas_ols(
    *,
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Per-asset OLS regression of returns onto factor returns.

    Centers both sides (the intercept absorbs the mean) and solves the
    least-squares system column-wise. Returns a DataFrame of shape `(N, K)`
    indexed by asset ticker with factor columns preserved from `factor_returns`.
    """

    if asset_returns.empty or factor_returns.empty:
        raise ValueError("estimate_betas_ols requires non-empty inputs")
    aligned = asset_returns.join(factor_returns, how="inner").dropna()
    if aligned.empty:
        raise ValueError(
            "asset_returns and factor_returns share no overlapping non-null dates"
        )
    asset_cols = list(asset_returns.columns)
    factor_cols = list(factor_returns.columns)
    F = aligned[factor_cols].to_numpy(dtype=float)
    R = aligned[asset_cols].to_numpy(dtype=float)
    F_c = F - F.mean(axis=0)
    R_c = R - R.mean(axis=0)
    betas, *_ = np.linalg.lstsq(F_c, R_c, rcond=None)  # (K, N)
    return pd.DataFrame(betas.T, index=asset_cols, columns=factor_cols)


def hypothetical_pnl(
    *,
    scenario: HypotheticalScenario,
    positions: dict[str, float],
    betas: pd.DataFrame,
) -> ScenarioPnL:
    """Apply a hypothetical-shock dict to positions through the beta map.

    `shocks` keys must match `betas.columns`. Unrecognized keys are dropped
    with a record in `metadata["unrecognized_factors"]`. Per-ticker P&L is
    `dollars * sum_k beta_{i,k} * shock_k`.
    """

    factor_cols = list(betas.columns)
    used_shocks: dict[str, float] = {}
    unrecognized: list[str] = []
    for key, value in scenario.shocks.items():
        if key in factor_cols:
            used_shocks[key] = float(value)
        else:
            unrecognized.append(key)

    contributions: dict[str, float] = {}
    total = 0.0
    for ticker, dollars in positions.items():
        if ticker not in betas.index:
            contributions[ticker] = 0.0
            continue
        beta_row = betas.loc[ticker]
        eq_return = 0.0
        for factor_name, shock_value in used_shocks.items():
            eq_return += float(beta_row[factor_name]) * shock_value
        pnl = float(dollars) * eq_return
        contributions[ticker] = pnl
        total += pnl

    return ScenarioPnL(
        scenario_name=scenario.name,
        scenario_type="hypothetical",
        total_pnl=total,
        contributions=contributions,
        factor_changes=dict(used_shocks),
        metadata={"unrecognized_factors": unrecognized},
    )


def portfolio_factor_sensitivities(
    *,
    positions: dict[str, float],
    betas: pd.DataFrame,
) -> np.ndarray:
    """Aggregate dollar sensitivity vector `g_k = sum_t positions[t] * beta_{t,k}`.

    Tickers not present in `betas` contribute zero. The returned vector is
    aligned to `betas.columns`.
    """

    factor_cols = list(betas.columns)
    g = np.zeros(len(factor_cols), dtype=float)
    for ticker, dollars in positions.items():
        if ticker not in betas.index:
            continue
        beta_row = betas.loc[ticker].to_numpy(dtype=float)
        g += float(dollars) * beta_row
    return g


def reverse_stress_shock(
    *,
    g: np.ndarray,
    factor_covariance: np.ndarray,
    loss_target: float,
    factor_names: tuple[str, ...],
) -> ReverseStressResult:
    """Closed-form min-Mahalanobis shock that produces `loss_target` of loss.

    Solves

        min_{dF} dF^T Σ⁻¹ dF   s.t.   g^T dF = -L*

    The constraint is `g^T dF = -L*` (negative because L* is a loss). The
    Lagrangian gives `dF = -λ Σ g` with `λ = L* / (g^T Σ g)`. The Mahalanobis
    norm of the solution is `L* / sqrt(g^T Σ g)`.

    Parameters
    ----------
    g:
        `(K,)` portfolio dollar sensitivities to each factor.
    factor_covariance:
        `(K, K)` factor-return covariance Σ.
    loss_target:
        Positive scalar loss in the same currency units as `g`. The sign
        convention is: a *positive* `loss_target` is the magnitude of money the
        book is allowed to lose.
    """

    K = g.shape[0]
    if factor_covariance.shape != (K, K):
        raise ValueError(
            f"factor_covariance shape {factor_covariance.shape} does not match "
            f"g shape {g.shape}"
        )
    if loss_target <= 0:
        raise ValueError(f"loss_target must be > 0, got {loss_target}")
    if len(factor_names) != K:
        raise ValueError(
            f"len(factor_names)={len(factor_names)} does not match K={K}"
        )

    quad_form = float(g @ factor_covariance @ g)
    if quad_form <= 0:
        raise ValueError(
            "g^T Σ g <= 0; portfolio has no factor sensitivity or covariance "
            "is not positive-definite — reverse stress is undefined."
        )
    lam = loss_target / quad_form
    dF_star = -lam * (factor_covariance @ g)
    mahalanobis = loss_target / math.sqrt(quad_form)
    diag = np.diag(factor_covariance)
    sigmas = np.sqrt(np.where(diag > 0, diag, 1.0))
    std_devs = dF_star / sigmas
    return ReverseStressResult(
        dF_star=dF_star,
        std_devs=std_devs,
        mahalanobis_distance=float(mahalanobis),
        loss_target=float(loss_target),
        factor_names=factor_names,
    )


def is_plausible(result: ReverseStressResult, threshold: float = _PLAUSIBILITY_THRESHOLD) -> bool:
    """Spec rule: Mahalanobis < 4 sigma is "plausible" and warrants attention."""

    return result.mahalanobis_distance < threshold


def scenario_coverage_matrix(
    *,
    scenarios: list[HypotheticalScenario],
    factor_names: list[str],
    material_threshold: float = 0.0,
) -> pd.DataFrame:
    """Boolean matrix marking which scenarios shock each factor materially.

    A cell is True when `|scenario.shocks[factor]| > material_threshold`.
    Empty rows / columns flag scenario-coverage gaps.
    """

    if not scenarios:
        return pd.DataFrame(False, index=[], columns=factor_names)
    rows: dict[str, dict[str, bool]] = {}
    for scenario in scenarios:
        row: dict[str, bool] = {f: False for f in factor_names}
        for factor, value in scenario.shocks.items():
            if factor in row and abs(float(value)) > material_threshold:
                row[factor] = True
        rows[scenario.name] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=factor_names)


def top_contributors(scenario: ScenarioPnL, k: int = 3) -> list[tuple[str, float]]:
    """Return the `k` tickers contributing the largest absolute P&L."""

    if k <= 0:
        return []
    items = list(scenario.contributions.items())
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items[:k]


def window_endpoint_sensitivity(
    *,
    window: CrisisWindow,
    prices_by_ticker: dict[str, pd.Series],
    positions: dict[str, float],
    shift_days: int = 2,
) -> dict[str, float]:
    """Re-run the window with `t1, t2` shifted by `±shift_days`.

    Returns a dict of label -> total P&L. The spec validation step wants
    "roughly stable" results; the caller compares these values pairwise.
    """

    out: dict[str, float] = {}
    t1 = pd.Timestamp(window.t1)
    t2 = pd.Timestamp(window.t2)
    shifts = {
        "base": (t1, t2),
        "early_t1": (t1 - pd.Timedelta(days=shift_days), t2),
        "late_t1": (t1 + pd.Timedelta(days=shift_days), t2),
        "early_t2": (t1, t2 - pd.Timedelta(days=shift_days)),
        "late_t2": (t1, t2 + pd.Timedelta(days=shift_days)),
    }
    for label, (a, b) in shifts.items():
        total = 0.0
        for ticker, dollars in positions.items():
            series = prices_by_ticker.get(ticker)
            if series is None:
                continue
            sliced = slice_window(series, a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
            change = window_factor_change(sliced)
            if math.isnan(change):
                continue
            total += float(dollars) * change
        out[label] = total
    return out
