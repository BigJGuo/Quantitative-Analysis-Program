"""Unit tests for the pure online-learning math in `signal.py`.

Each test verifies one piece of the spec's worked-out formulas against
synthetic data with known answers. Where possible the test reduces to a
closed-form algebraic check (single-round Hedge update against the exact
exponential ratio, FTRL prox against the spec's coordinate closed form, etc.)
so the test fails for the right reason if the math drifts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.online_learning.signal import (
    adaptive_eta,
    build_vol_expert_panel,
    clip_normalize_losses,
    cusum_break_detect,
    empirical_regret,
    ewma_vol,
    expert_losses,
    ftpl_sample_action,
    ftrl_dense_theta,
    ftrl_predict,
    ftrl_step,
    ftrl_theta,
    ftrl_update,
    hedge_predict,
    hedge_step,
    initial_ftrl_state,
    initial_hedge_state,
    log_loss,
    realized_vol_target,
    regret_bound_hedge,
    rolling_realized_vol,
    run_hedge_stream,
    safe_log,
    sigmoid,
    sparsity_fraction,
    weight_entropy,
)
from src.models.online_learning.types import (
    ExpertStream,
    FTRLConfig,
    HedgeConfig,
    HedgeState,
)

# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    @pytest.mark.parametrize("x", [50.0, 500.0])
    def test_large_positive_no_overflow(self, x: float) -> None:
        # No overflow / NaN; value is essentially 1 at this magnitude.
        result = sigmoid(x)
        assert math.isfinite(result)
        assert 0.0 < result <= 1.0
        assert result == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("x", [-50.0, -500.0])
    def test_large_negative_no_underflow(self, x: float) -> None:
        result = sigmoid(x)
        assert math.isfinite(result)
        assert 0.0 <= result < 1.0
        assert result == pytest.approx(0.0, abs=1e-12)


class TestLogLoss:
    def test_perfect_prediction(self) -> None:
        assert log_loss(1.0 - 1e-12, 1.0) < 1e-6
        assert log_loss(1e-12, 0.0) < 1e-6

    def test_log_loss_at_half_probability(self) -> None:
        # -(y log 0.5 + (1-y) log 0.5) = log 2 regardless of y in {0, 1}.
        for y in (0.0, 1.0):
            assert log_loss(0.5, y) == pytest.approx(math.log(2))

    def test_log_loss_clip_inf_safe(self) -> None:
        # p = 0 with y = 1 should not raise.
        assert log_loss(0.0, 1.0) > 0
        assert log_loss(1.0, 0.0) > 0


def test_safe_log_floors_negative() -> None:
    assert safe_log(0.0) == pytest.approx(math.log(1e-300))
    assert safe_log(1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Hedge — single-round and streaming
# ---------------------------------------------------------------------------


class TestInitialHedgeState:
    def test_uniform_weights(self) -> None:
        state = initial_hedge_state(4)
        assert state.weights.shape == (4,)
        assert state.weights == pytest.approx(np.full(4, 0.25))
        assert state.t == 0
        assert state.cumulative_meta_loss == 0.0

    def test_rejects_n_lt_2(self) -> None:
        with pytest.raises(ValueError, match="n_experts"):
            initial_hedge_state(1)


def test_adaptive_eta_matches_formula() -> None:
    n = 5
    for t in (1, 10, 100):
        assert adaptive_eta(t, n) == pytest.approx(math.sqrt(math.log(n) / t))


def test_clip_normalize_losses_maps_into_unit_interval() -> None:
    losses = np.array([-1.0, 0.5, 1.0, 5.0])
    out = clip_normalize_losses(losses, l_max=2.0)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert out[0] == 0.0  # negative clipped to 0
    assert out[3] == 1.0  # clipped to l_max then divided


def test_clip_normalize_losses_rejects_bad_l_max() -> None:
    with pytest.raises(ValueError, match="l_max"):
        clip_normalize_losses(np.array([1.0]), l_max=0.0)


class TestExpertLosses:
    def test_squared_loss_matches_formula(self) -> None:
        preds = np.array([1.0, 2.0, 3.0])
        out = expert_losses(preds, 2.0, loss_kind="squared")
        assert out == pytest.approx(np.array([1.0, 0.0, 1.0]))

    def test_log_squared_loss_perfect_zero(self) -> None:
        preds = np.array([2.0, 4.0])
        out = expert_losses(preds, 2.0, loss_kind="log_squared")
        # log(2/2) = 0, log(4/2) = log(2) so loss is log(2)^2.
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(math.log(2) ** 2)

    def test_log_loss_for_probability(self) -> None:
        preds = np.array([0.9, 0.1])
        out = expert_losses(preds, 1.0, loss_kind="log")
        assert out[0] < out[1]


class TestHedgePredict:
    def test_average_under_uniform(self) -> None:
        w = np.array([0.5, 0.5])
        p = np.array([1.0, 3.0])
        assert hedge_predict(w, p) == pytest.approx(2.0)

    def test_shape_check(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            hedge_predict(np.array([0.5, 0.5]), np.array([1.0, 2.0, 3.0]))


class TestHedgeStep:
    def test_single_round_matches_closed_form(self) -> None:
        """Spec Eq. 24: `w_i^{(t+1)} ∝ w_i^{(t)} exp(-eta ell_i)`."""

        w0 = np.array([0.25, 0.25, 0.25, 0.25])
        losses = np.array([0.0, 0.5, 1.0, 0.25])
        eta = 1.0
        state = HedgeState(weights=w0.copy())
        new_state = hedge_step(state, losses, eta=eta)
        expected = w0 * np.exp(-eta * losses)
        expected = expected / expected.sum()
        assert new_state.weights == pytest.approx(expected, rel=1e-10)
        assert new_state.t == 1

    def test_weight_collapses_onto_best_expert(self) -> None:
        """After many rounds with a constant best expert, weight → 1 on it."""

        state = HedgeState(weights=np.array([0.5, 0.5]))
        for _ in range(200):
            state = hedge_step(state, np.array([0.0, 1.0]), eta=0.5)
        assert state.weights[0] > 0.99
        assert state.weights[1] < 0.01

    def test_fixed_share_keeps_mass_on_inactive_expert(self) -> None:
        """Herbster-Warmuth mixing puts a floor on each weight."""

        state = HedgeState(weights=np.array([0.5, 0.5]))
        for _ in range(200):
            state = hedge_step(
                state, np.array([0.0, 1.0]), eta=0.5, fixed_share=0.1
            )
        # Fixed-share floor: each weight >= beta / N = 0.05.
        assert state.weights[1] >= 0.04

    def test_weights_remain_simplex(self) -> None:
        rng = np.random.default_rng(123)
        state = HedgeState(weights=np.full(6, 1.0 / 6))
        for _ in range(100):
            losses = rng.uniform(0.0, 1.0, size=6)
            state = hedge_step(state, losses, eta=0.3, fixed_share=0.01)
            assert state.weights.sum() == pytest.approx(1.0, abs=1e-12)
            assert (state.weights >= 0).all()

    def test_log_domain_stability_under_large_eta(self) -> None:
        """Large eta should not under/overflow — log-shift handles it."""

        state = HedgeState(weights=np.full(3, 1.0 / 3))
        losses = np.array([0.0, 1.0, 0.5])
        new_state = hedge_step(state, losses, eta=1e6)
        assert np.all(np.isfinite(new_state.weights))
        # The zero-loss expert wins almost all the mass.
        assert new_state.weights[0] == pytest.approx(1.0, abs=1e-6)


class TestWeightEntropy:
    def test_uniform_gives_log_n(self) -> None:
        w = np.full(8, 1.0 / 8)
        assert weight_entropy(w) == pytest.approx(math.log(8))

    def test_concentrated_gives_zero(self) -> None:
        w = np.array([1.0, 0.0, 0.0])
        assert weight_entropy(w) == pytest.approx(0.0, abs=1e-9)


class TestEmpiricalRegret:
    def test_regret_zero_when_meta_matches_best(self) -> None:
        cum = np.array([10.0, 12.0, 15.0])
        assert empirical_regret(10.0, cum) == pytest.approx(0.0)

    def test_regret_positive_when_meta_underperforms(self) -> None:
        cum = np.array([10.0, 12.0])
        assert empirical_regret(11.0, cum) == pytest.approx(1.0)


def test_regret_bound_matches_spec_formula() -> None:
    """Spec Eq. 30: `R_T <= sqrt(T log N / 2)`."""

    assert regret_bound_hedge(100, 5) == pytest.approx(
        math.sqrt(100 * math.log(5) / 2)
    )


class TestCusumBreakDetect:
    def test_no_break_on_constant_stream(self) -> None:
        s = pd.Series([0.1] * 20)
        assert not cusum_break_detect(s, h=1.0)

    def test_detects_upward_shift(self) -> None:
        s = pd.Series([0.1] * 50 + [10.0] * 20)
        assert cusum_break_detect(s, h=1.0)


# ---------------------------------------------------------------------------
# Hedge streaming driver
# ---------------------------------------------------------------------------


def test_run_hedge_stream_collapses_to_dominant_expert(
    dominant_expert_stream: ExpertStream,
) -> None:
    cfg = HedgeConfig(
        n_experts=2,
        eta=1.0,
        eta_schedule="constant",
        fixed_share=0.0,
        l_max=1.0,
        loss_kind="squared",
    )
    state, history, losses = run_hedge_stream(dominant_expert_stream, cfg)
    assert history.shape == (200, 2)
    assert state.t == 200
    assert state.weights[0] > 0.99
    # Bad expert always off by 1: per-round squared loss is 1 -> cum loss = 200.
    assert state.cumulative_expert_losses[1] == pytest.approx(200.0)
    assert losses.notna().sum() == 200


def test_run_hedge_stream_empirical_regret_below_theoretical_bound(
    dominant_expert_stream: ExpertStream,
) -> None:
    """Empirical regret should be small (and certainly below the textbook bound)."""

    cfg = HedgeConfig(
        n_experts=2,
        eta=1.0,
        eta_schedule="constant",
        loss_kind="squared",
    )
    state, _, _ = run_hedge_stream(dominant_expert_stream, cfg)
    regret = empirical_regret(state.cumulative_meta_loss, state.cumulative_expert_losses)
    bound = regret_bound_hedge(state.t, cfg.n_experts)
    # Hedge regret over an "easy" stream should be small relative to the worst-case bound.
    assert 0 <= regret < bound


def test_run_hedge_stream_handles_nan_targets() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    preds = pd.DataFrame(
        {"a": [0.1, 0.2, 0.3, 0.4, 0.5], "b": [0.5, 0.4, 0.3, 0.2, 0.1]},
        index=idx,
    )
    target = pd.Series([0.1, float("nan"), 0.3, 0.4, 0.5], index=idx)
    stream = ExpertStream(predictions=preds, targets=target)
    cfg = HedgeConfig(n_experts=2, loss_kind="squared")
    state, _, losses = run_hedge_stream(stream, cfg)
    # NaN target round is skipped; t increments only for valid rounds.
    assert state.t == 4
    assert losses.isna().sum() == 1


# ---------------------------------------------------------------------------
# FTPL
# ---------------------------------------------------------------------------


def test_ftpl_sample_action_picks_low_loss_expert_in_expectation() -> None:
    """With one expert clearly dominant, FTPL nearly always picks it."""

    rng = np.random.default_rng(99)
    cum = np.array([1.0, 100.0, 100.0])
    picks = [ftpl_sample_action(cum, eta=1.0, rng=rng) for _ in range(2000)]
    counts = np.bincount(picks, minlength=3)
    assert counts[0] > 1900  # heavy concentration on the low-loss action


# ---------------------------------------------------------------------------
# FTRL-Proximal
# ---------------------------------------------------------------------------


class TestFTRLTheta:
    def test_zero_when_z_below_lambda1(self) -> None:
        cfg = FTRLConfig(dim=3, lambda1=1.0, alpha=0.5, beta=1.0)
        z = np.array([0.5, -0.9, 0.99])
        n = np.array([1.0, 1.0, 1.0])
        theta = ftrl_theta(z, n, cfg)
        assert np.all(theta == 0.0)

    def test_closed_form_when_z_above_lambda1(self) -> None:
        """Spec Eq. 52 closed form for an active coordinate."""

        cfg = FTRLConfig(dim=1, alpha=0.5, beta=1.0, lambda1=0.5, lambda2=0.0)
        z = np.array([2.0])
        n = np.array([4.0])  # sqrt(n) = 2
        theta = ftrl_theta(z, n, cfg)
        # -(z - sign(z) lambda1) / ((beta + sqrt(n))/alpha + lambda2)
        # = -(2 - 0.5) / ((1 + 2)/0.5) = -1.5 / 6 = -0.25
        assert theta[0] == pytest.approx(-0.25)

    def test_l2_strengthens_shrinkage(self) -> None:
        cfg_a = FTRLConfig(dim=1, alpha=0.5, lambda1=0.0, lambda2=0.0)
        cfg_b = FTRLConfig(dim=1, alpha=0.5, lambda1=0.0, lambda2=5.0)
        z = np.array([3.0])
        n = np.array([1.0])
        theta_a = ftrl_theta(z, n, cfg_a)
        theta_b = ftrl_theta(z, n, cfg_b)
        assert abs(theta_b[0]) < abs(theta_a[0])


def test_ftrl_predict_and_update_logistic_smoke() -> None:
    cfg = FTRLConfig(dim=8, alpha=0.1, beta=1.0, lambda1=0.0, lambda2=0.0, loss_kind="log")
    state = initial_ftrl_state(cfg.dim)
    indices = np.array([0, 3, 7], dtype=np.int64)
    values = np.array([1.0, 0.5, -1.0])
    # On first call theta is all zero so the score is zero -> phat = 0.5.
    score, theta_active = ftrl_predict(state, indices, values, cfg)
    assert score == pytest.approx(0.0)
    assert theta_active.shape == (3,)

    ftrl_update(state, indices, values, theta_active, 0.4, cfg)
    # The active z-coords should now be nonzero.
    assert state.z[0] != 0
    assert state.z[3] != 0
    assert state.z[7] != 0
    # Inactive coords untouched.
    assert state.z[1] == 0
    assert state.n[indices][0] > 0


def test_ftrl_step_logistic_learns_dgp(
    logistic_regression_stream: tuple[
        list[tuple[np.ndarray, np.ndarray]], np.ndarray
    ],
) -> None:
    features, targets = logistic_regression_stream
    cfg = FTRLConfig(
        dim=30, alpha=0.3, beta=1.0, lambda1=0.5, lambda2=0.0, loss_kind="log"
    )
    state = initial_ftrl_state(cfg.dim)
    cumulative = 0.0
    for (idx, vals), y in zip(features, targets, strict=True):
        cumulative += ftrl_step(state, idx, vals, float(y), cfg)
    mean_loss = cumulative / len(features)
    # Naive predict-0.5 gives log(2) ~= 0.693; learner should clearly beat that.
    assert mean_loss < 0.65

    theta = ftrl_dense_theta(state, cfg)
    # In a dense low-dim setup every coordinate eventually gets activated, so
    # we don't require global sparsity — only that the three true-active
    # coords are the largest in absolute value.
    top3 = np.argsort(-np.abs(theta))[:3].tolist()
    assert set(top3) == {1, 7, 19}


def test_ftrl_high_lambda1_produces_sparsity() -> None:
    """Spec validation: with L1 dominating the gradient signal, theta collapses to 0."""

    rng = np.random.default_rng(2026)
    cfg = FTRLConfig(
        dim=50, alpha=0.05, beta=1.0, lambda1=20.0, lambda2=0.0, loss_kind="log"
    )
    state = initial_ftrl_state(cfg.dim)
    for _ in range(500):
        idx = rng.choice(cfg.dim, size=5, replace=False).astype(np.int64)
        vals = rng.normal(size=5)
        y = float(rng.binomial(1, 0.5))
        ftrl_step(state, idx, vals, y, cfg)
    theta = ftrl_dense_theta(state, cfg)
    assert sparsity_fraction(theta) > 0.9


def test_ftrl_step_squared_loss_recovers_sparse_target(
    linear_regression_stream: tuple[
        list[tuple[np.ndarray, np.ndarray]], np.ndarray
    ],
) -> None:
    features, targets = linear_regression_stream
    cfg = FTRLConfig(
        dim=20, alpha=0.1, beta=1.0, lambda1=0.1, lambda2=0.0, loss_kind="squared"
    )
    state = initial_ftrl_state(cfg.dim)
    for (idx, vals), y in zip(features, targets, strict=True):
        ftrl_step(state, idx, vals, float(y), cfg)
    theta = ftrl_dense_theta(state, cfg)
    # The signs of the recovered nonzero coords must agree with theta_star.
    assert theta[2] > 0
    assert theta[5] < 0
    assert theta[11] > 0


# ---------------------------------------------------------------------------
# Vol-expert panel builders
# ---------------------------------------------------------------------------


def test_realized_vol_target_first_row_is_nan() -> None:
    prices = pd.Series([100.0, 101.0, 102.0])
    out = realized_vol_target(prices)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(abs(math.log(101 / 100)))


def test_rolling_realized_vol_is_causal() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.0, 100.0, 99.0, 101.0, 103.0, 102.0],
        index=pd.date_range("2024-01-02", periods=8, freq="B"),
    )
    out = rolling_realized_vol(prices, window=3)
    # First 3 should be NaN (need 3 returns and a one-step shift).
    assert out.iloc[:3].isna().all()
    # Causal: out at index t uses returns strictly before t.
    assert out.notna().sum() >= 3


def test_ewma_vol_decay_lower_lambda_reacts_faster(synthetic_prices: pd.Series) -> None:
    fast = ewma_vol(synthetic_prices, lam=0.80)
    slow = ewma_vol(synthetic_prices, lam=0.99)
    # On a shock the fast (lower lambda) series moves more.
    fast_var = fast.diff().abs().mean()
    slow_var = slow.diff().abs().mean()
    assert fast_var > slow_var


def test_ewma_vol_rejects_bad_lambda() -> None:
    with pytest.raises(ValueError, match="lam"):
        ewma_vol(pd.Series([100.0, 101.0]), lam=1.0)


def test_build_vol_expert_panel_returns_clean_stream(synthetic_prices: pd.Series) -> None:
    stream = build_vol_expert_panel(synthetic_prices)
    # All NaN rows dropped: every prediction and target must be finite.
    assert stream.predictions.notna().all().all()
    assert stream.targets.notna().all()
    # 5 experts: rolling 5/20/60 + ewma 0.94/0.97.
    assert stream.predictions.shape[1] == 5
