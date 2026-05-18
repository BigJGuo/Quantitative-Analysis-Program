"""Dataclasses specific to the Online Learning model.

Three flavours are supported:

* **Hedge / Exponential Weights** over `N` signal-experts (Freund-Schapire).
  State: a probability vector `weights` on the simplex.
* **FTRL-Proximal** for online linear / logistic regression with L1 + L2
  (McMahan 2011 / 2013). State: per-coordinate `z`, `n`.
* **FTPL** (Follow-the-Perturbed-Leader) — Gumbel-perturbation form
  equivalent in expectation to Hedge. Same `HedgeState` carrier.

`Forecast` / `Signal` / `RiskMetric` / `CalibrationResult` come from
`src.core.types`; this module holds only the model-internal shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

LearnerKind = Literal["hedge", "ftrl", "ftpl"]
EtaSchedule = Literal["constant", "adaptive"]
LossKind = Literal["squared", "log_squared", "log"]

VALID_KINDS: frozenset[str] = frozenset({"hedge", "ftrl", "ftpl"})
VALID_ETA_SCHEDULES: frozenset[str] = frozenset({"constant", "adaptive"})
VALID_LOSS_KINDS: frozenset[str] = frozenset({"squared", "log_squared", "log"})


@dataclass(frozen=True)
class HedgeConfig:
    """Hyperparameters for the Hedge / FTPL learner.

    Attributes
    ----------
    n_experts:
        Number of experts `N`.
    eta:
        Learning rate. With ``eta_schedule="constant"`` this is used as-is;
        with ``"adaptive"`` the per-round rate is `sqrt(log N / t)` (the
        textbook anytime schedule) and `eta` is ignored.
    eta_schedule:
        ``"constant"`` or ``"adaptive"``.
    fixed_share:
        Herbster-Warmuth mixing weight `beta`. After the multiplicative step,
        `w <- (1 - beta) * w + beta / N`. `0` disables (vanilla Hedge);
        typical non-stationary value is `0.01`.
    l_max:
        Upper clip / normalization constant for losses. Hedge requires losses
        in `[0, 1]`; per-round loss is divided by `l_max` after clipping.
    loss_kind:
        How to compute per-expert loss against the realized target:
        * ``"squared"`` — `(yhat - y)^2`
        * ``"log_squared"`` — `(log yhat - log y)^2` (volatility forecasts)
        * ``"log"`` — `-(y log p + (1-y) log (1-p))` (binary y).
    seed:
        Seed for FTPL perturbations. Unused by vanilla Hedge.
    """

    n_experts: int
    eta: float = 0.1
    eta_schedule: EtaSchedule = "constant"
    fixed_share: float = 0.0
    l_max: float = 1.0
    loss_kind: LossKind = "squared"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_experts < 2:
            raise ValueError(
                f"HedgeConfig.n_experts must be >= 2, got {self.n_experts}"
            )
        if self.eta <= 0:
            raise ValueError(f"HedgeConfig.eta must be > 0, got {self.eta}")
        if self.eta_schedule not in VALID_ETA_SCHEDULES:
            raise ValueError(
                f"HedgeConfig.eta_schedule must be one of "
                f"{sorted(VALID_ETA_SCHEDULES)}, got {self.eta_schedule!r}"
            )
        if not 0.0 <= self.fixed_share <= 1.0:
            raise ValueError(
                f"HedgeConfig.fixed_share must be in [0, 1], got {self.fixed_share}"
            )
        if self.l_max <= 0:
            raise ValueError(f"HedgeConfig.l_max must be > 0, got {self.l_max}")
        if self.loss_kind not in VALID_LOSS_KINDS:
            raise ValueError(
                f"HedgeConfig.loss_kind must be one of {sorted(VALID_LOSS_KINDS)}, "
                f"got {self.loss_kind!r}"
            )


@dataclass
class HedgeState:
    """Mutable per-round state of the Hedge learner.

    `weights` is the current probability vector on the `n_experts` simplex.
    `t` is the round counter (number of updates applied so far).
    `cumulative_meta_loss` is the prequential loss `sum_s ell_s(w_s)`.
    `cumulative_expert_losses` is the length-`N` vector of expert cumulative
    losses `L_i^{(t)} = sum_s ell_i^{(s)}` used for empirical-regret reporting.
    """

    weights: np.ndarray
    t: int = 0
    cumulative_meta_loss: float = 0.0
    cumulative_expert_losses: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError(
                f"HedgeState.weights must be 1-D, got shape {self.weights.shape}"
            )
        if self.cumulative_expert_losses.size == 0:
            self.cumulative_expert_losses = np.zeros_like(self.weights)
        if self.cumulative_expert_losses.shape != self.weights.shape:
            raise ValueError(
                "HedgeState.cumulative_expert_losses must have the same "
                f"shape as weights, got {self.cumulative_expert_losses.shape} "
                f"vs {self.weights.shape}"
            )


@dataclass(frozen=True)
class FTRLConfig:
    """Hyperparameters for FTRL-Proximal (McMahan 2013).

    Attributes
    ----------
    dim:
        Number of feature coordinates `d`.
    alpha:
        Per-coordinate base learning rate. Typical range `[0.01, 0.5]`.
    beta:
        Per-coordinate AdaGrad offset. Typical value `1.0`.
    lambda1:
        L1 regularization weight (sparsity).
    lambda2:
        L2 regularization weight.
    loss_kind:
        ``"squared"`` for online linear regression, ``"log"`` for online
        logistic regression. ``"log_squared"`` is not supported for FTRL.
    """

    dim: int
    alpha: float = 0.1
    beta: float = 1.0
    lambda1: float = 1.0
    lambda2: float = 0.0
    loss_kind: LossKind = "log"

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"FTRLConfig.dim must be >= 1, got {self.dim}")
        if self.alpha <= 0:
            raise ValueError(f"FTRLConfig.alpha must be > 0, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"FTRLConfig.beta must be >= 0, got {self.beta}")
        if self.lambda1 < 0:
            raise ValueError(
                f"FTRLConfig.lambda1 must be >= 0, got {self.lambda1}"
            )
        if self.lambda2 < 0:
            raise ValueError(
                f"FTRLConfig.lambda2 must be >= 0, got {self.lambda2}"
            )
        if self.loss_kind not in ("squared", "log"):
            raise ValueError(
                "FTRLConfig.loss_kind must be 'squared' or 'log', "
                f"got {self.loss_kind!r}"
            )


@dataclass
class FTRLState:
    """Mutable per-coordinate state of the FTRL-Proximal learner.

    `z` accumulates the gradient-minus-learning-rate-adjustment running sum.
    `n` accumulates squared gradients (for AdaGrad-style per-coordinate
    rates). `theta` is *not* stored — it is derived on-the-fly per active
    coordinate per round.
    """

    z: np.ndarray
    n: np.ndarray
    t: int = 0
    cumulative_meta_loss: float = 0.0

    def __post_init__(self) -> None:
        if self.z.shape != self.n.shape:
            raise ValueError(
                f"FTRLState.z and FTRLState.n must have the same shape, "
                f"got {self.z.shape} vs {self.n.shape}"
            )
        if self.z.ndim != 1:
            raise ValueError(
                f"FTRLState.z must be 1-D, got shape {self.z.shape}"
            )


@dataclass(frozen=True)
class ExpertStream:
    """Aligned panel of expert predictions plus realized targets.

    Attributes
    ----------
    predictions:
        DataFrame indexed by date with one column per expert; values are the
        expert's prediction for that date (made *before* the target is
        revealed).
    targets:
        Series of realized values aligned to `predictions.index`.
    """

    predictions: pd.DataFrame
    targets: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.predictions, pd.DataFrame):
            raise TypeError(
                f"ExpertStream.predictions must be a DataFrame, "
                f"got {type(self.predictions).__name__}"
            )
        if not isinstance(self.targets, pd.Series):
            raise TypeError(
                f"ExpertStream.targets must be a Series, "
                f"got {type(self.targets).__name__}"
            )
        if self.predictions.shape[1] < 2:
            raise ValueError(
                "ExpertStream.predictions must have at least 2 experts, got "
                f"{self.predictions.shape[1]}"
            )
        if not self.predictions.index.equals(self.targets.index):
            raise ValueError(
                "ExpertStream.predictions index must equal targets index"
            )


@dataclass(frozen=True)
class OnlineLearningInputs:
    """`fetch_data -> calibrate / predict / validate` payload.

    Carries the expert panel for Hedge and the diagnostic / forecast horizon
    metadata. The FTRL branch uses a different data shape (sparse features /
    binary targets) and is not exposed through `BaseModel.fetch_data` in the
    default `HedgeOverVolExperts` configuration — call `calibrate()` directly
    for FTRL workflows.
    """

    ticker: str
    stream: ExpertStream
    kind: LearnerKind
    timestamp: datetime
    horizon: str = "1d"
    annualize: bool = False
    trading_days_per_year: int = 252
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"OnlineLearningInputs.kind must be one of {sorted(VALID_KINDS)}, "
                f"got {self.kind!r}"
            )
        if self.trading_days_per_year <= 0:
            raise ValueError(
                "OnlineLearningInputs.trading_days_per_year must be positive, "
                f"got {self.trading_days_per_year}"
            )


@dataclass(frozen=True)
class HedgeFit:
    """Final Hedge state plus prequential diagnostics.

    Attributes
    ----------
    expert_names:
        Column-aligned to `weights`.
    weights:
        Final probability vector (length `N`).
    cumulative_meta_loss:
        Sum of per-round meta losses over the streamed history.
    cumulative_expert_losses:
        Per-expert cumulative loss over the same window.
    empirical_regret:
        `cumulative_meta_loss - min_i cumulative_expert_losses[i]`. Compare
        against the theoretical `sqrt(T log N / 2)` bound.
    weight_history:
        `(T, N)` array of post-update weights for diagnostics / plotting.
    prequential_losses:
        Length-`T` series of per-round meta losses (the running prequential
        loss series). NaN for any round where the loss could not be evaluated.
    config:
        The HedgeConfig used.
    """

    expert_names: tuple[str, ...]
    weights: np.ndarray
    cumulative_meta_loss: float
    cumulative_expert_losses: np.ndarray
    empirical_regret: float
    weight_history: np.ndarray
    prequential_losses: pd.Series
    config: HedgeConfig

    def __post_init__(self) -> None:
        if len(self.expert_names) != self.weights.shape[0]:
            raise ValueError(
                f"HedgeFit.expert_names ({len(self.expert_names)}) does not "
                f"match weights ({self.weights.shape[0]})"
            )
        if self.weight_history.ndim != 2:
            raise ValueError(
                f"HedgeFit.weight_history must be 2-D, got shape "
                f"{self.weight_history.shape}"
            )
        if self.weight_history.shape[1] != self.weights.shape[0]:
            raise ValueError(
                "HedgeFit.weight_history second axis must match N experts"
            )


@dataclass(frozen=True)
class FTRLFit:
    """Final FTRL state plus prequential diagnostics.

    Attributes
    ----------
    z, n:
        Per-coordinate accumulators (`d`-length).
    theta:
        Derived coefficient vector at the final state (snapshot).
    cumulative_meta_loss:
        Sum of per-round log-loss / squared loss.
    sparsity:
        Fraction of theta coordinates equal to zero.
    prequential_losses:
        Length-`T` series of per-round meta losses.
    config:
        The FTRLConfig used.
    """

    z: np.ndarray
    n: np.ndarray
    theta: np.ndarray
    cumulative_meta_loss: float
    sparsity: float
    prequential_losses: pd.Series
    config: FTRLConfig

    def __post_init__(self) -> None:
        if not (self.z.shape == self.n.shape == self.theta.shape):
            raise ValueError(
                f"FTRLFit z/n/theta shapes must match, got {self.z.shape}, "
                f"{self.n.shape}, {self.theta.shape}"
            )
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError(
                f"FTRLFit.sparsity must be in [0, 1], got {self.sparsity}"
            )
