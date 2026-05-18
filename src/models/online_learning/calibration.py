"""Calibration entry point for the Online Learning model.

`calibrate()` is the public, data-only path. For online learners there is no
"batch fit" in the classical sense — calibration is a *warm-up*: replay the
historical stream once (or twice) to bring the state up to date before going
live. The returned `CalibrationResult.parameters` holds the final `HedgeFit`
or `FTRLFit`; `validate` and `predict` then consume it.

Two public functions:

* `calibrate(stream, ...)` — Hedge warm-up on an `ExpertStream`. Default for
  the vol-expert workflow in the spec's 'yfinance data requirements' section.
* `calibrate_ftrl(features, targets, config)` — FTRL-Proximal warm-up on a
  sparse-feature stream.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from src.core.types import CalibrationResult
from src.models.online_learning.signal import (
    empirical_regret,
    ftrl_dense_theta,
    ftrl_step,
    initial_ftrl_state,
    initial_hedge_state,
    run_hedge_stream,
    sparsity_fraction,
    weight_entropy,
)
from src.models.online_learning.types import (
    ExpertStream,
    FTRLConfig,
    FTRLFit,
    HedgeConfig,
    HedgeFit,
)

MODEL_NAME: str = "online_learning"


def calibrate(
    *,
    stream: ExpertStream,
    config: HedgeConfig,
    n_passes: int = 1,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> CalibrationResult:
    """Warm-up Hedge on an `ExpertStream` and return a `CalibrationResult`.

    Parameters
    ----------
    stream:
        Aligned expert-prediction / target panel.
    config:
        Hedge hyperparameters.
    n_passes:
        Number of replays over the same history. `1` is the streaming
        default; `2` is a warm-start for backtesting. Empirical Hedge regret
        is reported on the *last* pass.
    timestamp:
        Calibration time. Defaults to ``datetime.now(UTC)``.
    metadata:
        Extra metadata to attach to the result (e.g. ticker).
    """

    if n_passes < 1:
        raise ValueError(f"calibrate: n_passes must be >= 1, got {n_passes}")
    if stream.predictions.shape[1] != config.n_experts:
        raise ValueError(
            f"calibrate: stream has {stream.predictions.shape[1]} experts but "
            f"config.n_experts = {config.n_experts}"
        )

    ts = timestamp or datetime.now(UTC)
    state = initial_hedge_state(config.n_experts)

    final_history: np.ndarray | None = None
    final_losses: pd.Series | None = None
    for _ in range(n_passes):
        state, final_history, final_losses = run_hedge_stream(
            stream, config, initial_state=state
        )

    assert final_history is not None  # n_passes >= 1 by the check above
    assert final_losses is not None

    regret = empirical_regret(state.cumulative_meta_loss, state.cumulative_expert_losses)
    fit = HedgeFit(
        expert_names=tuple(str(c) for c in stream.predictions.columns),
        weights=state.weights.copy(),
        cumulative_meta_loss=state.cumulative_meta_loss,
        cumulative_expert_losses=state.cumulative_expert_losses.copy(),
        empirical_regret=regret,
        weight_history=final_history,
        prequential_losses=final_losses,
        config=config,
    )
    return _wrap_hedge(fit, ts, metadata or {})


def calibrate_ftrl(
    *,
    features: list[tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
    config: FTRLConfig,
    n_passes: int = 1,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> CalibrationResult:
    """Warm-up FTRL-Proximal on a sparse-feature stream.

    `features[t] = (indices, values)` is the active-coordinate sparse encoding
    of example `t`; `targets[t]` is the realized label (0 / 1 for logistic).
    """

    if len(features) != targets.shape[0]:
        raise ValueError(
            f"calibrate_ftrl: features ({len(features)}) and targets "
            f"({targets.shape[0]}) must align"
        )
    if n_passes < 1:
        raise ValueError(f"calibrate_ftrl: n_passes must be >= 1, got {n_passes}")

    ts = timestamp or datetime.now(UTC)
    state = initial_ftrl_state(config.dim)

    final_losses: list[float] = []
    for _ in range(n_passes):
        final_losses = []
        for (indices, values), y in zip(features, targets, strict=True):
            loss = ftrl_step(state, indices, values, float(y), config)
            final_losses.append(loss)

    theta = ftrl_dense_theta(state, config)
    fit = FTRLFit(
        z=state.z.copy(),
        n=state.n.copy(),
        theta=theta,
        cumulative_meta_loss=state.cumulative_meta_loss,
        sparsity=sparsity_fraction(theta),
        prequential_losses=pd.Series(final_losses, dtype="float64", name="meta_loss"),
        config=config,
    )
    return _wrap_ftrl(fit, ts, metadata or {})


def _wrap_hedge(
    fit: HedgeFit,
    ts: datetime,
    extra_metadata: dict[str, Any],
) -> CalibrationResult:
    parameters: dict[str, Any] = {
        "hedge_fit": fit,
        "kind": "hedge",
        "weights": fit.weights.tolist(),
        "expert_names": list(fit.expert_names),
    }
    t = int(fit.weight_history.shape[0])
    fit_metrics = {
        "n_rounds": float(t),
        "n_experts": float(fit.weights.shape[0]),
        "cumulative_meta_loss": float(fit.cumulative_meta_loss),
        "best_expert_cum_loss": float(fit.cumulative_expert_losses.min()),
        "empirical_regret": float(fit.empirical_regret),
        "final_weight_entropy": weight_entropy(fit.weights),
    }
    metadata = {"kind": "hedge", **extra_metadata}
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata=metadata,
    )


def _wrap_ftrl(
    fit: FTRLFit,
    ts: datetime,
    extra_metadata: dict[str, Any],
) -> CalibrationResult:
    parameters: dict[str, Any] = {
        "ftrl_fit": fit,
        "kind": "ftrl",
        "theta": fit.theta.tolist(),
    }
    losses_arr = cast(np.ndarray, fit.prequential_losses.to_numpy(dtype=float))
    fit_metrics = {
        "n_rounds": float(losses_arr.size),
        "dim": float(fit.theta.shape[0]),
        "cumulative_meta_loss": float(fit.cumulative_meta_loss),
        "sparsity": float(fit.sparsity),
        "mean_prequential_loss": (
            float(np.nanmean(losses_arr)) if losses_arr.size else float("nan")
        ),
    }
    metadata = {"kind": "ftrl", **extra_metadata}
    return CalibrationResult(
        model_name=MODEL_NAME,
        parameters=parameters,
        fit_metrics=fit_metrics,
        timestamp=ts,
        metadata=metadata,
    )


__all__ = ["MODEL_NAME", "calibrate", "calibrate_ftrl"]
