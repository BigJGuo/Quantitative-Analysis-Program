"""Pure online-learning math.

Functions in this module are stateless (or operate on the explicit `*State`
dataclasses): no I/O, no model class, no global mutation. This keeps every
step verifiable against the spec's worked-out formulas with synthetic inputs.

Three families:

1. **Hedge / Exponential Weights** — multiplicative-weights update on the
   simplex over `N` experts.
2. **FTRL-Proximal** — coordinate-wise prox update for streaming linear /
   logistic regression with L1 + L2 (McMahan 2011 / 2013).
3. **Diagnostics** — prequential / empirical regret, weight entropy,
   CUSUM-based regime-break detector.

Cross-cutting helpers (numerically stable sigmoid / log-loss, loss clipping,
adaptive learning-rate schedules) live alongside the family they pair with.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from src.models.online_learning.types import (
    ExpertStream,
    FTRLConfig,
    FTRLState,
    HedgeConfig,
    HedgeState,
    LossKind,
)

_TINY = 1e-300  # safe lower bound for log() inside numerical routines
_PROB_CLIP = 1e-15  # clip for probabilities in log-loss — tighter than _TINY
                    # because `1 - 1e-300` underflows to 1.0 in IEEE 754.


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Numerically stable logistic function `1 / (1 + exp(-x))`."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_log(x: float, floor: float = _TINY) -> float:
    """`log(max(x, floor))` — avoids `-inf` for zero-valued forecasts."""

    return math.log(x if x > floor else floor)


def log_loss(p: float, y: float) -> float:
    """Binary cross-entropy loss with `p` clipped away from `{0, 1}`.

    `y` may be any value in `[0, 1]`; the convex form is
    `-(y log p + (1-y) log (1-p))`.
    """

    p_clipped = min(max(p, _PROB_CLIP), 1.0 - _PROB_CLIP)
    return -(y * math.log(p_clipped) + (1.0 - y) * math.log(1.0 - p_clipped))


# ---------------------------------------------------------------------------
# Hedge / Exponential Weights
# ---------------------------------------------------------------------------


def initial_hedge_state(n_experts: int) -> HedgeState:
    """Uniform prior `w_i^{(0)} = 1 / N` plus zero-initialized accumulators."""

    if n_experts < 2:
        raise ValueError(f"initial_hedge_state: n_experts must be >= 2, got {n_experts}")
    w = np.full(n_experts, 1.0 / n_experts, dtype=float)
    return HedgeState(weights=w)


def adaptive_eta(t: int, n_experts: int) -> float:
    """`sqrt(log N / t)` — the standard anytime schedule for Hedge.

    Falls back to `sqrt(log N)` for the very first round (`t = 1`) where the
    rate would otherwise be undefined.
    """

    if t < 1:
        raise ValueError(f"adaptive_eta: t must be >= 1, got {t}")
    return math.sqrt(math.log(n_experts) / max(t, 1))


def expert_losses(
    predictions: np.ndarray,
    target: float,
    *,
    loss_kind: LossKind,
) -> np.ndarray:
    """Per-expert raw losses against a scalar realized target.

    `loss_kind`:

    * ``"squared"`` — `(yhat_i - y)^2`.
    * ``"log_squared"`` — `(log yhat_i - log y)^2`. Both predictions and
      target are first floored away from zero. Spec usage: volatility
      forecasts.
    * ``"log"`` — binary cross-entropy. Each `yhat_i` is interpreted as a
      probability in `[0, 1]` and `target` as the `0/1` label.
    """

    if predictions.ndim != 1:
        raise ValueError(
            f"expert_losses: predictions must be 1-D, got shape {predictions.shape}"
        )
    if loss_kind == "squared":
        return cast(np.ndarray, (predictions - target) ** 2)
    if loss_kind == "log_squared":
        floor = _TINY
        log_pred = np.log(np.maximum(predictions, floor))
        log_tgt = math.log(target if target > floor else floor)
        return cast(np.ndarray, (log_pred - log_tgt) ** 2)
    if loss_kind == "log":
        out = np.empty_like(predictions, dtype=float)
        for i, p in enumerate(predictions):
            out[i] = log_loss(float(p), target)
        return out
    raise ValueError(f"expert_losses: unknown loss_kind {loss_kind!r}")


def clip_normalize_losses(losses: np.ndarray, l_max: float) -> np.ndarray:
    """Clip to `[0, l_max]` then rescale to `[0, 1]` in place-safe form.

    Hedge's regret bound is stated for losses in `[0, 1]`; uncontrolled MSE on
    tail returns can blow up the multiplicative update, so the spec's
    'Limitations and failure modes' section explicitly requires clipping.
    """

    if l_max <= 0:
        raise ValueError(f"clip_normalize_losses: l_max must be > 0, got {l_max}")
    clipped = np.clip(losses, 0.0, l_max)
    return cast(np.ndarray, clipped / l_max)


def hedge_predict(weights: np.ndarray, predictions: np.ndarray) -> float:
    """Meta prediction `sum_i w_i * yhat_i`."""

    if weights.shape != predictions.shape:
        raise ValueError(
            "hedge_predict: weights and predictions must share shape, "
            f"got {weights.shape} vs {predictions.shape}"
        )
    return float(np.dot(weights, predictions))


def hedge_step(
    state: HedgeState,
    losses_normalized: np.ndarray,
    eta: float,
    *,
    fixed_share: float = 0.0,
) -> HedgeState:
    """One multiplicative-weights update.

    Implements ``w_i^{(t+1)} = w_i^{(t)} * exp(-eta * l_i) / Z`` followed
    by optional fixed-share mixing ``w <- (1 - beta) w + beta / N``.

    The update is done with a log-domain shift for numerical stability:
    multiplying by ``exp(-eta * l_i)`` and renormalizing is invariant to
    subtracting a constant from the exponent. The constant we use is the
    *minimum* of the log-weights minus `eta * l`, which keeps the largest
    pre-normalized weight at `1` and avoids underflow.
    """

    if eta <= 0:
        raise ValueError(f"hedge_step: eta must be > 0, got {eta}")
    if losses_normalized.shape != state.weights.shape:
        raise ValueError(
            "hedge_step: losses shape must match weights, "
            f"got {losses_normalized.shape} vs {state.weights.shape}"
        )
    if not 0.0 <= fixed_share <= 1.0:
        raise ValueError(
            f"hedge_step: fixed_share must be in [0, 1], got {fixed_share}"
        )

    log_w = np.log(np.maximum(state.weights, _TINY)) - eta * losses_normalized
    log_w -= log_w.max()  # numerical shift; preserves the post-normalization result
    new_w = np.exp(log_w)
    total = new_w.sum()
    new_w = (
        np.full_like(state.weights, 1.0 / state.weights.size)
        if total <= 0
        else new_w / total
    )

    if fixed_share > 0:
        new_w = (1.0 - fixed_share) * new_w + fixed_share / state.weights.size
        # Re-normalize defensively against tiny float drift.
        new_w = new_w / new_w.sum()

    return HedgeState(
        weights=new_w,
        t=state.t + 1,
        cumulative_meta_loss=state.cumulative_meta_loss,
        cumulative_expert_losses=state.cumulative_expert_losses.copy(),
    )


def weight_entropy(weights: np.ndarray) -> float:
    """Shannon entropy `H = -sum w_i log w_i`.

    Equals `log N` when uniform, `0` when concentrated on one expert. The
    spec lists this as a primary diagnostic: collapse to `0` is the textbook
    signal that ``eta`` is too aggressive.
    """

    w = np.maximum(weights, _TINY)
    return float(-np.sum(w * np.log(w)))


def empirical_regret(meta_loss: float, expert_losses_cum: np.ndarray) -> float:
    """`L_T - min_i L_i^{(T)}` — see spec 'Validation and diagnostics'.

    Compare against the theoretical Hedge bound `sqrt(T * log N / 2)`.
    """

    if expert_losses_cum.size == 0:
        raise ValueError("empirical_regret: expert_losses_cum must be non-empty")
    return float(meta_loss - expert_losses_cum.min())


def regret_bound_hedge(t: int, n_experts: int) -> float:
    """Textbook Hedge regret bound `sqrt(T log N / 2)` for `eta` optimally tuned."""

    if t < 1 or n_experts < 2:
        raise ValueError(
            f"regret_bound_hedge: require T >= 1 and N >= 2, got T={t}, N={n_experts}"
        )
    return math.sqrt(t * math.log(n_experts) / 2.0)


def cusum_break_detect(loss_series: pd.Series, *, h: float, k: float = 0.0) -> bool:
    """One-sided CUSUM regime-break detector on a prequential loss series.

    `S_0 = 0; S_t = max(0, S_{t-1} + (ell_t - mean_loss) - k)`. Returns
    ``True`` if `max_t S_t > h`. Used to trigger a Hedge reset / fixed-share
    boost as described in the spec's 'Reset on regime break'.

    Parameters
    ----------
    loss_series:
        Per-round prequential losses. NaNs are dropped.
    h:
        Decision threshold.
    k:
        Slack parameter (typical: `0.5 * sigma_ell`). Default `0`.
    """

    if h <= 0:
        raise ValueError(f"cusum_break_detect: h must be > 0, got {h}")
    clean = loss_series.dropna().to_numpy(dtype=float)
    if clean.size < 2:
        return False
    centered = clean - clean.mean()
    s = 0.0
    for v in centered:
        s = max(0.0, s + v - k)
        if s > h:
            return True
    return False


# ---------------------------------------------------------------------------
# Hedge: streaming driver over an ExpertStream
# ---------------------------------------------------------------------------


def run_hedge_stream(
    stream: ExpertStream,
    config: HedgeConfig,
    *,
    initial_state: HedgeState | None = None,
) -> tuple[HedgeState, np.ndarray, pd.Series]:
    """Replay `stream` through Hedge, returning (final state, history, losses).

    `history` is a `(T, N)` array of post-update weights. `losses` is a
    `pd.Series` indexed by `stream.predictions.index` containing the
    per-round meta squared / log loss (NOT the normalized version), with
    `NaN` for any row where the target is missing.

    The driver:

    1. Reads `yhat_1, ..., yhat_N` for the round.
    2. Emits `yhat_meta = sum_i w_i yhat_i` and records its raw loss against
       the realized target.
    3. Updates the cumulative per-expert losses (raw scale).
    4. Computes normalized losses (clip to `[0, l_max]`, divide by `l_max`).
    5. Runs the multiplicative-weights update with the chosen `eta` schedule.
    6. Optionally fixed-share mixes (Herbster-Warmuth).
    """

    n = config.n_experts
    if stream.predictions.shape[1] != n:
        raise ValueError(
            "run_hedge_stream: stream has "
            f"{stream.predictions.shape[1]} experts but config.n_experts = {n}"
        )

    state = initial_state if initial_state is not None else initial_hedge_state(n)
    if state.weights.shape[0] != n:
        raise ValueError(
            "run_hedge_stream: initial_state.weights size does not match n_experts"
        )
    history = np.empty((stream.predictions.shape[0], n), dtype=float)
    raw_losses = np.full(stream.predictions.shape[0], np.nan, dtype=float)

    preds = stream.predictions.to_numpy(dtype=float)
    tgts = stream.targets.to_numpy(dtype=float)

    for t_idx in range(preds.shape[0]):
        row = preds[t_idx]
        target = tgts[t_idx]
        if not np.all(np.isfinite(row)) or not np.isfinite(target):
            history[t_idx] = state.weights
            continue

        meta_pred = hedge_predict(state.weights, row)
        meta_loss = _scalar_loss(meta_pred, target, config.loss_kind)
        raw_losses[t_idx] = meta_loss

        per_expert = expert_losses(row, target, loss_kind=config.loss_kind)
        state.cumulative_expert_losses += per_expert
        state.cumulative_meta_loss += meta_loss

        normalized = clip_normalize_losses(per_expert, config.l_max)
        eta_t = (
            adaptive_eta(state.t + 1, n)
            if config.eta_schedule == "adaptive"
            else config.eta
        )
        state = hedge_step(
            state, normalized, eta_t, fixed_share=config.fixed_share
        )
        history[t_idx] = state.weights

    losses = pd.Series(raw_losses, index=stream.predictions.index, name="meta_loss")
    return state, history, losses


def _scalar_loss(yhat: float, y: float, loss_kind: LossKind) -> float:
    """Per-round meta loss in the same scale as `expert_losses`."""

    if loss_kind == "squared":
        return float((yhat - y) ** 2)
    if loss_kind == "log_squared":
        return float(
            (safe_log(yhat) - safe_log(y)) ** 2
        )
    if loss_kind == "log":
        return log_loss(yhat, y)
    raise ValueError(f"_scalar_loss: unknown loss_kind {loss_kind!r}")


# ---------------------------------------------------------------------------
# FTPL — Follow-the-Perturbed-Leader (Gumbel form)
# ---------------------------------------------------------------------------


def ftpl_sample_action(
    cumulative_losses: np.ndarray,
    eta: float,
    rng: np.random.Generator,
) -> int:
    """Draw `xi_i ~ Gumbel(0, 1)` and play `argmin_i { L_i - xi_i / eta }`.

    For Hedge-equivalent setups this gives the same expected weights as
    exponential weights (Gumbel-max identity).
    """

    if eta <= 0:
        raise ValueError(f"ftpl_sample_action: eta must be > 0, got {eta}")
    # numpy's gumbel samples from Gumbel(loc, scale); we want standard.
    xi = rng.gumbel(loc=0.0, scale=1.0, size=cumulative_losses.shape)
    return int(np.argmin(cumulative_losses - xi / eta))


# ---------------------------------------------------------------------------
# FTRL-Proximal
# ---------------------------------------------------------------------------


def initial_ftrl_state(dim: int) -> FTRLState:
    """Zero-initialize the per-coordinate `z` and `n` accumulators."""

    if dim < 1:
        raise ValueError(f"initial_ftrl_state: dim must be >= 1, got {dim}")
    return FTRLState(z=np.zeros(dim, dtype=float), n=np.zeros(dim, dtype=float))


def ftrl_theta(
    z_active: np.ndarray,
    n_active: np.ndarray,
    config: FTRLConfig,
) -> np.ndarray:
    """Per-coordinate prox solution.

    Implements the closed-form

        theta_j = 0                                      if |z_j| <= lambda1
                = -(z_j - sign(z_j) * lambda1) /
                  ((beta + sqrt(n_j)) / alpha + lambda2) otherwise.

    Vectorized over the *active* coordinates `j` with `x_{t,j} != 0`.
    """

    theta = np.zeros_like(z_active)
    inactive = np.abs(z_active) <= config.lambda1
    if inactive.all():
        return theta
    active = ~inactive
    sign_z = np.sign(z_active[active])
    denom = (config.beta + np.sqrt(n_active[active])) / config.alpha + config.lambda2
    numer = z_active[active] - sign_z * config.lambda1
    theta[active] = -numer / denom
    return theta


def ftrl_predict(
    state: FTRLState,
    indices: np.ndarray,
    values: np.ndarray,
    config: FTRLConfig,
) -> tuple[float, np.ndarray]:
    """Compute the score `sum_j theta_j * x_j` over active coords.

    For `loss_kind == "log"`, the returned scalar is the *score*; the caller
    applies `sigmoid()` to get a probability.

    Returns ``(score, theta_active)`` so the caller can reuse the
    per-coordinate `theta` when applying the gradient update.
    """

    if indices.shape != values.shape:
        raise ValueError(
            f"ftrl_predict: indices/values shape mismatch, {indices.shape} vs {values.shape}"
        )
    if indices.size == 0:
        return 0.0, np.zeros(0, dtype=float)
    z_active = state.z[indices]
    n_active = state.n[indices]
    theta_active = ftrl_theta(z_active, n_active, config)
    return float(np.dot(theta_active, values)), theta_active


def ftrl_update(
    state: FTRLState,
    indices: np.ndarray,
    values: np.ndarray,
    theta_active: np.ndarray,
    grad_scale: float,
    config: FTRLConfig,
) -> None:
    """Apply the FTRL gradient update for one streamed example *in place*.

    Decomposition: each per-coordinate gradient is
    ``g_j = grad_scale * x_{t,j}`` so logistic and linear regression share a
    single update path. For log-loss ``grad_scale = phat - y``; for squared
    loss ``grad_scale = (yhat - y)``.
    """

    if indices.shape != values.shape or indices.shape != theta_active.shape:
        raise ValueError("ftrl_update: indices/values/theta_active shapes must match")
    if indices.size == 0:
        return

    g = grad_scale * values
    n_active = state.n[indices]
    sigma = (np.sqrt(n_active + g * g) - np.sqrt(n_active)) / config.alpha
    state.z[indices] += g - sigma * theta_active
    state.n[indices] += g * g


def ftrl_step(
    state: FTRLState,
    indices: np.ndarray,
    values: np.ndarray,
    y: float,
    config: FTRLConfig,
) -> float:
    """One full FTRL-Proximal round: predict, observe, update.

    Returns the per-round meta loss (log-loss for logistic, squared error
    for linear). Side effects: mutates ``state``.
    """

    score, theta_active = ftrl_predict(state, indices, values, config)
    if config.loss_kind == "log":
        phat = sigmoid(score)
        loss = log_loss(phat, y)
        grad_scale = phat - y
    elif config.loss_kind == "squared":
        loss = float((score - y) ** 2)
        grad_scale = score - y
    else:  # pragma: no cover -- guarded by FTRLConfig validation
        raise ValueError(f"ftrl_step: unsupported loss_kind {config.loss_kind!r}")

    ftrl_update(state, indices, values, theta_active, grad_scale, config)
    state.cumulative_meta_loss += loss
    state.t += 1
    return loss


def ftrl_dense_theta(state: FTRLState, config: FTRLConfig) -> np.ndarray:
    """Snapshot the full `theta` vector from the current `z`, `n` accumulators.

    Useful for diagnostics and final reporting; not needed inside the
    streaming hot path because per-coord `theta_j` is derived on the fly.
    """

    return ftrl_theta(state.z, state.n, config)


def sparsity_fraction(theta: np.ndarray) -> float:
    """Fraction of theta coordinates that are exactly zero."""

    if theta.size == 0:
        return 1.0
    return float((theta == 0.0).sum()) / theta.size


# ---------------------------------------------------------------------------
# Vol-expert factory used by the BaseModel wrapper
# ---------------------------------------------------------------------------


def realized_vol_target(prices: pd.Series) -> pd.Series:
    """Daily realized vol proxy: `|r_t|` where `r_t = log P_t - log P_{t-1}`.

    On a single trading day, the integrated-variance estimator collapses to
    the absolute log return (or equivalently `sqrt(r^2)`). It is a noisy but
    unbiased proxy of daily realized vol — adequate for Hedge benchmarking
    because all experts in the panel share the same target.
    """

    if prices.empty:
        return pd.Series(dtype="float64")
    log_prices = pd.Series(
        np.log(prices.astype(float).to_numpy(dtype=float)),
        index=prices.index,
    )
    return log_prices.diff().abs().rename("realized_vol")


def rolling_realized_vol(prices: pd.Series, window: int) -> pd.Series:
    """Rolling realized vol over `window` trading days, shifted to be causal.

    Output at time `t` is the standard deviation of the past `window`
    log-returns *strictly before* `t`, so the value is an admissible
    prediction of `|r_t|` made at the close of day `t - 1`.
    """

    if window < 2:
        raise ValueError(f"rolling_realized_vol: window must be >= 2, got {window}")
    log_prices = pd.Series(
        np.log(prices.astype(float).to_numpy(dtype=float)),
        index=prices.index,
    )
    log_returns = log_prices.diff()
    # std over the trailing window, then shift one step so we never look ahead.
    return log_returns.rolling(window=window, min_periods=window).std().shift(1)


def ewma_vol(prices: pd.Series, lam: float) -> pd.Series:
    """RiskMetrics-style EWMA volatility prediction made one step ahead.

    Recursion:
        `sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2`.

    The output at time `t` is the volatility forecast for day `t`, formed
    from information available at the close of day `t - 1`.
    """

    if not 0.0 < lam < 1.0:
        raise ValueError(f"ewma_vol: lam must be in (0, 1), got {lam}")
    log_prices = pd.Series(
        np.log(prices.astype(float).to_numpy(dtype=float)),
        index=prices.index,
    )
    log_returns = log_prices.diff()
    r_sq = (log_returns**2).shift(1)
    out = pd.Series(np.nan, index=prices.index, dtype="float64")
    prev = np.nan
    for i, r2 in enumerate(r_sq.to_numpy(dtype=float)):
        if np.isnan(r2):
            continue
        prev = r2 if np.isnan(prev) else lam * prev + (1.0 - lam) * r2
        out.iloc[i] = math.sqrt(prev) if prev > 0 else 0.0
    return out


def build_vol_expert_panel(
    prices: pd.Series,
    *,
    rolling_windows: tuple[int, ...] = (5, 20, 60),
    ewma_lambdas: tuple[float, ...] = (0.94, 0.97),
) -> ExpertStream:
    """Construct a Hedge-ready panel of vol-forecast experts plus targets.

    Experts: rolling-σ over each window in `rolling_windows`, and EWMA-σ for
    each λ in `ewma_lambdas`. Target: `|r_t|`.

    All rows containing any NaN (predictions or target) are dropped so the
    panel can be replayed deterministically.
    """

    if prices.empty:
        raise ValueError("build_vol_expert_panel: prices must be non-empty")

    target = realized_vol_target(prices)
    cols: dict[str, pd.Series] = {}
    for w in rolling_windows:
        cols[f"rolling_{w}"] = rolling_realized_vol(prices, w)
    for lam in ewma_lambdas:
        cols[f"ewma_{lam:.2f}"] = ewma_vol(prices, lam)

    panel = pd.DataFrame(cols, index=prices.index)
    aligned = pd.concat([panel, target.rename("__target__")], axis=1).dropna()
    preds = aligned.drop(columns="__target__")
    tgt = aligned["__target__"].rename("realized_vol")
    return ExpertStream(predictions=preds, targets=tgt)


__all__ = [
    "adaptive_eta",
    "build_vol_expert_panel",
    "clip_normalize_losses",
    "cusum_break_detect",
    "empirical_regret",
    "expert_losses",
    "ewma_vol",
    "ftpl_sample_action",
    "ftrl_dense_theta",
    "ftrl_predict",
    "ftrl_step",
    "ftrl_theta",
    "ftrl_update",
    "hedge_predict",
    "hedge_step",
    "initial_ftrl_state",
    "initial_hedge_state",
    "log_loss",
    "realized_vol_target",
    "regret_bound_hedge",
    "rolling_realized_vol",
    "run_hedge_stream",
    "safe_log",
    "sigmoid",
    "sparsity_fraction",
    "weight_entropy",
]
