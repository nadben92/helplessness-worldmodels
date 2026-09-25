"""End-to-end Phase 2 pipeline on the REAL E3 pixel environment: collect
fixed epsilon=1 data into a replay buffer, then run world-model-only
training steps against it. Uses k=2 (24x24 images, divisible by 8) and a
tiny RSSM so this stays fast on CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from _experiment_loader import load_experiment_module

from helplessness.agents.worldmodel.loop import (
    WorldModelLossConfig,
    collect_fixed_policy_data,
    train_world_model_step,
    world_model_loss,
)
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig
from helplessness.envs.appearance.theme_sampling import build_themes
from helplessness.envs.observation.allocentric_pixel_backend import render_background

e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "test_loop_e3_env")

K = 2  # small render scale -> 24x24 images, fast on CPU


def _build_env_and_background(seed_index: int = 0, max_steps: int = 36):
    rotation = e3_env.rotation_for_seed_index(seed_index)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    layout = e3_env.build_layout(assignment)
    themes = build_themes(e3_env.ZONE_CATEGORIES, goal_state=e3_env.goal_state(), rng=np.random.default_rng(seed_index), min_dist=5.0)
    env = e3_env.build_pixel_env(layout, themes, goal_reward=50.0, max_steps=max_steps, seed=seed_index, k=K)
    background = render_background(layout, themes, k=K)  # (H, W, 3) uint8
    background_t = torch.as_tensor(background, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return env, background_t


def _small_rssm_cfg() -> RSSMConfig:
    return RSSMConfig(
        image_size=(24, 24), in_channels=3, base_depth=4, embed_dim=16, h_dim=16,
        n_vars=3, n_classes=4, n_actions=5,
    )


def test_collect_fixed_policy_data_fills_replay_with_expected_shapes():
    env, _ = _build_env_and_background()
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(0)
    collect_fixed_policy_data(env, buf, n_steps=100, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)
    assert len(buf) == 100

    model_batch, label_batch = buf.sample(batch_size=4, seq_len=16, rng=rng)
    assert model_batch.x.shape == (4, 16, K * 12, K * 12, 3)
    assert model_batch.a.min() >= 0 and model_batch.a.max() < env.n_actions
    assert label_batch.s.min() >= 0 and label_batch.s.max() < env.n_states


def test_collect_fixed_policy_data_never_conflates_terminal_and_truncated():
    """"Not conflated" does NOT mean mutually exclusive -- an episode
    can legitimately reach the goal on the very step max_steps is also
    hit, giving terminal=True AND truncated=True together (both are
    independently, honestly true that step). What must never happen is
    truncated being used as a STAND-IN for terminal, or vice versa: both
    a truncation-only ending and a termination-only ending must occur,
    and terminal must be True only at genuine goal arrivals."""
    env, _ = _build_env_and_background(max_steps=4)
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(1)
    collect_fixed_policy_data(env, buf, n_steps=200, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)
    model_batch, label_batch = buf.sample(batch_size=1, seq_len=200, rng=rng)
    terminal = model_batch.terminal[0]
    truncated = model_batch.truncated[0]
    s = label_batch.s[0]

    assert (truncated & ~terminal).any()  # a truncation-only ending occurs
    assert (terminal & ~truncated).any() or (terminal & truncated).any()  # some genuine termination occurs
    goal_s = e3_env.goal_state()
    # terminal is True only where the state truly is the goal -- s here
    # is the DEPARTURE label (the state the step was taken FROM), so we
    # check goal-arrival by looking at model_batch.r for the goal reward.
    goal_reward = 50.0
    assert (terminal == (model_batch.r[0] == goal_reward)).all()


def test_world_model_loss_runs_end_to_end_and_is_finite():
    torch.manual_seed(0)
    env, background = _build_env_and_background()
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(2)
    collect_fixed_policy_data(env, buf, n_steps=80, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)

    rssm = RSSM(_small_rssm_cfg())
    model_batch, _ = buf.sample(batch_size=2, seq_len=8, rng=rng)
    cfg = WorldModelLossConfig(k=K, w_patch=10.0)
    losses = world_model_loss(rssm, model_batch, background, cfg)

    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} is not finite: {value}"
    assert losses["total"].requires_grad


def test_train_world_model_step_updates_parameters_and_returns_finite_losses():
    torch.manual_seed(0)
    env, background = _build_env_and_background()
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(3)
    collect_fixed_policy_data(env, buf, n_steps=80, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)

    rssm = RSSM(_small_rssm_cfg())
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-4, eps=1e-8)
    cfg = WorldModelLossConfig(k=K, w_patch=10.0)

    before = [p.clone() for p in rssm.parameters()]
    model_batch, _ = buf.sample(batch_size=2, seq_len=8, rng=rng)
    result = train_world_model_step(rssm, optimizer, model_batch, background, cfg)

    for name, value in result.items():
        assert value == value, f"{name} is NaN"  # NaN != NaN
        assert abs(value) < 1e9, f"{name} exploded: {value}"

    after = list(rssm.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_repeated_training_steps_decrease_total_loss_on_a_tiny_fixed_batch():
    """Not a rigorous convergence proof, just a sanity check that the
    pipeline is a genuine, functioning optimisation loop: many steps on
    ONE fixed small batch should reduce that batch's own loss."""
    torch.manual_seed(0)
    env, background = _build_env_and_background()
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(4)
    collect_fixed_policy_data(env, buf, n_steps=80, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)

    rssm = RSSM(_small_rssm_cfg())
    optimizer = torch.optim.Adam(rssm.parameters(), lr=3e-3, eps=1e-8)  # higher lr for a fast, short sanity check
    cfg = WorldModelLossConfig(k=K, w_patch=10.0)
    model_batch, _ = buf.sample(batch_size=2, seq_len=6, rng=np.random.default_rng(99))

    first_loss = None
    last_loss = None
    for step in range(30):
        result = train_world_model_step(rssm, optimizer, model_batch, background, cfg)
        if step == 0:
            first_loss = result["total"]
        last_loss = result["total"]

    assert last_loss < first_loss


# --- training schedules (docs/e3/DECISIONS.md, 2026-09-24) ---------------


def test_gumbel_schedule_anneals_exponentially_from_tau_start_to_tau_end_then_holds():
    from helplessness.agents.worldmodel.loop import EstimatorSchedule

    sched = EstimatorSchedule(kind="gumbel_st", tau_start=1.0, tau_end=0.1, anneal_steps=100)
    assert sched.at(0).tau == pytest.approx(1.0)
    assert sched.at(50).tau == pytest.approx(0.1**0.5)  # geometric midpoint
    assert sched.at(100).tau == pytest.approx(0.1)
    assert sched.at(10_000).tau == pytest.approx(0.1)
    assert sched.at(0).kind == "gumbel_st"


def test_warmup_probs_schedule_switches_to_straight_through_at_warmup_steps():
    from helplessness.agents.worldmodel.loop import EstimatorSchedule

    sched = EstimatorSchedule(kind="warmup_probs", warmup_steps=10)
    assert sched.at(0).kind == "probs"
    assert sched.at(9).kind == "probs"
    assert sched.at(10).kind == "straight_through"


def test_straight_through_schedule_is_constant():
    from helplessness.agents.worldmodel.loop import EstimatorSchedule

    sched = EstimatorSchedule()
    assert {sched.at(t).kind for t in (0, 1, 10_000)} == {"straight_through"}


def test_rep_beta_is_zero_until_the_switch_then_ramps_linearly_then_holds():
    from helplessness.agents.worldmodel.loop import rep_beta

    assert rep_beta(5000, None, 100, 0.1) == 0.0  # no switch yet
    assert rep_beta(999, 1000, 100, 0.1) == 0.0
    assert rep_beta(1000, 1000, 100, 0.1) == 0.0
    assert rep_beta(1050, 1000, 100, 0.1) == pytest.approx(0.05)
    assert rep_beta(1100, 1000, 100, 0.1) == pytest.approx(0.1)
    assert rep_beta(9000, 1000, 100, 0.1) == pytest.approx(0.1)


def test_separation_switch_fires_after_k_consecutive_evaluations_above_threshold():
    from helplessness.agents.worldmodel.loop import SeparationSwitch

    sw = SeparationSwitch(max_steps=10_000, threshold_bits=3.0, consecutive=3, eval_every=50, min_steps=100)
    assert sw.estimator().kind == "probs"
    for step, stat in ((0, 5.0), (50, 5.0), (100, 5.0), (150, 2.0), (200, 3.5), (250, 3.5)):
        sw.update(step, stat)
        assert not sw.switched  # before min_steps, or streak broken at 150, or streak only 2
    sw.update(300, 3.5)
    assert sw.switch_step == 301  # effective from the next step
    assert sw.estimator().kind == "straight_through"
    sw.update(350, 0.0)  # once switched, stays switched
    assert sw.switch_step == 301


def test_separation_switch_raises_explicitly_when_separation_is_not_reached():
    from helplessness.agents.worldmodel.loop import SeparationNotReachedError, SeparationSwitch

    sw = SeparationSwitch(max_steps=200, consecutive=2, eval_every=50, min_steps=0)
    for step in (0, 50, 100, 150):
        sw.update(step, 1.0)
        sw.check_deadline(step + 1)
    with pytest.raises(SeparationNotReachedError, match="non atteinte"):
        sw.check_deadline(200)


def test_separation_switch_never_raises_once_switched():
    from helplessness.agents.worldmodel.loop import SeparationSwitch

    sw = SeparationSwitch(max_steps=100, consecutive=1, eval_every=50, min_steps=0)
    sw.update(0, 9.0)
    sw.check_deadline(10_000)


def test_train_world_model_step_can_return_the_batch_posteriors():
    torch.manual_seed(0)
    env, background = _build_env_and_background()
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(6)
    collect_fixed_policy_data(env, buf, n_steps=60, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)
    rssm = RSSM(_small_rssm_cfg())
    opt = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    mb, _ = buf.sample(batch_size=3, seq_len=5, rng=rng)
    out = train_world_model_step(rssm, opt, mb, background, WorldModelLossConfig(k=K), return_posteriors=True)
    assert out["post_probs"].shape == (15, 3, 4)
    assert np.allclose(out["post_probs"].sum(-1), 1.0, atol=1e-5)
    assert "post_probs" not in train_world_model_step(rssm, opt, mb, background, WorldModelLossConfig(k=K))


def test_train_world_model_step_accepts_an_estimator():
    from helplessness.agents.worldmodel.rssm import LatentEstimator

    torch.manual_seed(0)
    env, background = _build_env_and_background()
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(5)
    collect_fixed_policy_data(env, buf, n_steps=60, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)
    rssm = RSSM(_small_rssm_cfg())
    opt = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    mb, _ = buf.sample(batch_size=2, seq_len=6, rng=rng)
    for est in (LatentEstimator("probs"), LatentEstimator("gumbel_st", tau=0.5), LatentEstimator()):
        out = train_world_model_step(rssm, opt, mb, background, WorldModelLossConfig(k=K), estimator=est)
        assert all(v == v for v in out.values())  # no NaN


def test_separation_switch_rejects_a_deadline_before_the_earliest_possible_switch():
    from helplessness.agents.worldmodel.loop import SeparationSwitch

    # earliest switch: evaluations at 500, 550, ..., 700 (5 consecutive) -> effective at 701
    with pytest.raises(ValueError, match="can never switch"):
        SeparationSwitch(max_steps=700, consecutive=5, eval_every=50, min_steps=500)
    SeparationSwitch(max_steps=701, consecutive=5, eval_every=50, min_steps=500)


def test_collapse_guard_fires_only_after_the_switch_on_consecutive_low_evaluations():
    from helplessness.agents.worldmodel.loop import CollapseAfterSwitchError, SeparationSwitch

    sw = SeparationSwitch(max_steps=10_000, consecutive=2, eval_every=50, min_steps=0, collapse_threshold_bits=1.0,
                          collapse_consecutive=3)
    for step in (0, 50, 100):  # before the switch: low values never trip the guard
        sw.update(step, 0.0)
    sw.update(150, 5.0)
    sw.update(200, 5.0)
    assert sw.switch_step == 201
    sw.update(250, 0.5)
    sw.update(300, 0.5)
    sw.update(350, 2.0)  # streak reset
    sw.update(400, 0.5)
    sw.update(450, 0.5)
    with pytest.raises(CollapseAfterSwitchError, match="collapse after the switch"):
        sw.update(500, 0.5)


def test_phase_learning_rate_and_set_learning_rate():
    from helplessness.agents.worldmodel.loop import phase_learning_rate, set_learning_rate

    assert phase_learning_rate(10, None, 1e-3, 1e-4) == 1e-3
    assert phase_learning_rate(99, 100, 1e-3, 1e-4) == 1e-3
    assert phase_learning_rate(100, 100, 1e-3, 1e-4) == 1e-4
    assert phase_learning_rate(500, 100, 1e-3, None) == 1e-3
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
    assert set_learning_rate(opt, 1e-4) is True
    assert set_learning_rate(opt, 1e-4) is False
    assert opt.param_groups[0]["lr"] == 1e-4


def test_module_grad_norms_hand_values_and_read_only():
    from helplessness.agents.worldmodel.loop import module_grad_norms

    a, b = torch.nn.Linear(2, 1, bias=False), torch.nn.Linear(2, 1, bias=False)
    a.weight.grad = torch.tensor([[3.0, 4.0]])
    norms = module_grad_norms({"a": a, "b": b})
    assert norms == {"a": 5.0, "b": 0.0}
    assert torch.equal(a.weight.grad, torch.tensor([[3.0, 4.0]]))


def test_clip_norm_none_does_not_clip_and_reports_the_same_pre_clip_grad_norm():
    """clip_norm=None (global clipping disabled, DECISIONS.md 2026-09-24):
    the reported grad_norm is the true global norm, identical to a very
    large finite clip_norm, whereas a tiny clip_norm still reports that
    pre-clip norm but moves the parameters less."""
    env, background = _build_env_and_background()
    buf = ReplayBuffer(capacity=200, image_shape=(K * 12, K * 12, 3))
    rng = np.random.default_rng(3)
    collect_fixed_policy_data(env, buf, n_steps=80, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)
    model_batch, _ = buf.sample(batch_size=2, seq_len=8, rng=rng)
    cfg = WorldModelLossConfig(k=K, w_patch=10.0)

    def run(clip_norm):
        torch.manual_seed(0)
        rssm = RSSM(_small_rssm_cfg())
        opt = torch.optim.SGD(rssm.parameters(), lr=1e-3)
        before = torch.cat([p.detach().reshape(-1).clone() for p in rssm.parameters()])
        torch.manual_seed(1)
        result = train_world_model_step(rssm, opt, model_batch, background, cfg, clip_norm=clip_norm)
        after = torch.cat([p.detach().reshape(-1) for p in rssm.parameters()])
        return result["grad_norm"], float((after - before).norm())

    norm_none, move_none = run(None)
    norm_huge, move_huge = run(1e12)
    norm_tiny, move_tiny = run(1e-6)
    assert norm_none == norm_huge == norm_tiny > 0
    assert move_none == move_huge
    assert move_tiny < 1e-3 * move_none
