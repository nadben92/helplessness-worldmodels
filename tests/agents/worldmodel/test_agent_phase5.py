"""Phase 5 agent components (spec section 4, as amended): goal head, critic,
slow critic, the one-step-lookahead epsilon-greedy policy with its relative
tie-break, imagination and the lambda-returns. Small RSSM, CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel import heads, imagination, policy
from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig

CFG = AgentConfig(k_act=6, k_imag=3, horizon=4)


def _model(seed: int = 0):
    torch.manual_seed(seed)
    rssm = RSSM(RSSMConfig(image_size=(24, 24), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4))
    goal = GoalHead(16, 12, hidden_dim=32, zero_init=False)  # random heads: distinct Q-values to test the formulas on
    critic = Critic(16, 12, hidden_dim=32, zero_init=False)
    return rssm, goal, critic


def _states(batch: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(batch, 16, generator=g)
    z = F.one_hot(torch.randint(0, 4, (batch, 3), generator=g), num_classes=4).float()
    return h, z


def test_reward_and_continuation_follow_e2_s_convention():
    p = torch.tensor([0.0, 1.0, 0.5])
    assert heads.reward_from_goal(p, CFG).tolist() == [-1.0, 50.0, 24.5]  # 50 p - 1 (1 - p)
    assert heads.continuation_from_goal(p).tolist() == [1.0, 0.0, 0.5]


def test_heads_send_no_gradient_to_their_inputs():
    _, goal, critic = _model()
    h, z = _states(5)
    h.requires_grad_(True)
    z.requires_grad_(True)
    (goal(h, z).sum() + critic(h, z).sum()).backward()
    assert h.grad is None and z.grad is None
    assert all(p.grad is not None for p in goal.parameters()) and all(p.grad is not None for p in critic.parameters())


def test_slow_critic_is_an_ema_of_the_critic():
    _, _, critic = _model()
    slow = heads.make_slow_critic(critic)
    assert all(not p.requires_grad for p in slow.parameters())
    before = [p.clone() for p in slow.parameters()]
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(1.0)
    heads.update_slow_critic(slow, critic, tau=0.02)
    for b, s, c in zip(before, slow.parameters(), critic.parameters()):
        assert torch.allclose(s, 0.98 * b + 0.02 * c)


def test_q_is_the_mean_over_the_prior_draws_of_r_plus_discounted_value():
    rssm, goal, critic = _model()
    h, z = _states(4)
    Q = policy.q_values(rssm, goal, critic, h, z, CFG.k_act, CFG, torch.Generator().manual_seed(3))
    assert Q.shape == (4, 5)
    h_next, z_next = policy.lookahead(rssm, h, z, CFG.k_act, torch.Generator().manual_seed(3))  # the same draws
    assert h_next.shape == (4, 5, 16) and z_next.shape == (4, 5, 6, 3, 4) and torch.all(z_next.sum(-1) == 1)
    with torch.no_grad():
        hk = h_next.unsqueeze(2).expand(-1, -1, 6, -1)
        p = goal.p_goal(hk, z_next)
        expected = (50.0 * p - (1.0 - p) + 0.9 * (1.0 - p) * critic(hk, z_next)).mean(dim=-1)
    assert torch.allclose(Q, expected, atol=1e-5)
    # h'_a is the RSSM's own GRU step
    a2 = F.one_hot(torch.full((4,), 2), num_classes=5).float()
    assert torch.allclose(h_next[:, 2], rssm.gru_step(h, z, a2), atol=1e-6)


def test_tie_tolerance_is_absolute_from_the_horizon():
    """docs/e3/DECISIONS.md, 2026-10-01 (12): half the smallest Q gap of information at H = 36 (6 x 0.9^35)."""
    assert heads.horizon_information_gap(0.9, 50.0, -1.0, 36) == pytest.approx(6 * 0.9 ** 35)
    assert CFG.tie_tol_abs == pytest.approx(0.5 * 6 * 0.9 ** 35) and CFG.tie_tol_rel is None
    Q = torch.tensor([[0.0, 0.05, 0.01, 0.07, 0.0], [0.0, 0.0, 0.0, 0.0, 0.2], [100.0, 100.07, 99.9, 50.0, 0.0]])
    assert torch.allclose(policy.tie_tolerance(Q, CFG), torch.full((3,), 0.5 * 6 * 0.9 ** 35))  # the same for every row
    ties = policy.tie_set(Q, CFG)
    assert ties[0].all() and ties[1].tolist() == [False] * 4 + [True] and ties[2].tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError, match="choose one"):
        AgentConfig(tie_tol_abs=0.1, tie_tol_rel=0.01)


def test_legacy_relative_tie_tolerance_reads_old_runs():
    legacy = AgentConfig(k_act=6, k_imag=3, horizon=4, tie_tol_rel=0.01)
    assert legacy.tie_tol_abs is None
    Q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 0.999, 0.0, 0.0, 0.5]])
    CFG_ = legacy
    tol = policy.tie_tolerance(Q, CFG_)
    assert tol[0] == pytest.approx(0.01 * 0.4) and tol[1] == pytest.approx(1e-6)  # std 0: the floor
    ties = policy.tie_set(Q, CFG_)
    assert ties[0].tolist() == [False, False, False, False, True] and ties[1].all()
    assert ties[2].tolist() == [True, True, False, False, False]  # 0.999 is within 0.01 * std of the max


def test_real_step_choice_uses_separate_streams_and_stays_in_the_tie_set():
    cfg = AgentConfig(epsilon=0.0)
    Q = torch.tensor([1.0, 1.0, 0.0, 1.0, -3.0])
    picks = [policy.act(Q, cfg, np.random.default_rng(0), np.random.default_rng(seed))["action"] for seed in range(60)]
    assert set(picks) == {0, 1, 3}  # random among the ties, never outside
    out = policy.act(Q, cfg, np.random.default_rng(0), np.random.default_rng(0))
    assert out["tied"] and not out["explored"]
    # epsilon = 1: the explore stream alone decides; the tie-break stream is not consumed
    tiebreak = np.random.default_rng(5)
    state = tiebreak.bit_generator.state
    explore = np.random.default_rng(1)
    actions = [policy.act(Q, AgentConfig(epsilon=1.0), explore, tiebreak) for _ in range(200)]
    assert tiebreak.bit_generator.state == state and all(a["explored"] for a in actions)
    assert set(a["action"] for a in actions) == {0, 1, 2, 3, 4}
    assert not policy.act(torch.tensor([0.0, 0.0, 0.0, 0.0, 9.0]), cfg, explore, tiebreak)["tied"]


def test_batched_choice_has_the_same_tie_set_as_the_real_step_choice():
    torch.manual_seed(0)
    Q = torch.randn(300, 5).round(decimals=0)  # many exact ties
    cfg = AgentConfig(epsilon=0.0)
    choice = policy.epsilon_greedy(Q, cfg, torch.Generator().manual_seed(0))
    ties = policy.tie_set(Q, cfg)
    assert torch.all(ties[torch.arange(300), choice["action"]])  # always a tied-best action
    assert torch.equal(choice["tied"], ties.sum(-1) > 1) and not choice["explored"].any()
    for row in range(20):  # the numpy path picks inside the same set
        a = policy.act(Q[row], cfg, np.random.default_rng(row), np.random.default_rng(row))["action"]
        assert ties[row, a]
    explored = policy.epsilon_greedy(Q, AgentConfig(epsilon=1.0), torch.Generator().manual_seed(0))
    assert explored["explored"].all() and set(explored["action"].tolist()) == {0, 1, 2, 3, 4}


def test_lambda_returns_match_the_recursion_by_hand():
    r = torch.tensor([[0.0], [1.0], [2.0]])  # r(s_0), r(s_1), r(s_2)
    c = torch.tensor([[1.0], [0.5], [1.0]])
    v = torch.tensor([[10.0], [20.0], [30.0]])
    R = imagination.lambda_returns(r, c, v, gamma=0.9, lam=0.95)
    R2 = 30.0
    R1 = 2.0 + 0.9 * 1.0 * (0.05 * 30.0 + 0.95 * R2)
    R0 = 1.0 + 0.9 * 0.5 * (0.05 * 20.0 + 0.95 * R1)
    assert R[:, 0].tolist() == pytest.approx([R0, R1, R2])


def test_imagination_rollout_and_critic_loss():
    rssm, goal, critic = _model()
    slow = heads.make_slow_critic(critic)
    h, z = _states(7)
    rollout = imagination.imagine(rssm, goal, critic, h, z, CFG, torch.Generator().manual_seed(0))
    assert rollout["h"].shape == (5, 7, 16) and rollout["z"].shape == (5, 7, 3, 4) and rollout["actions"].shape == (4, 7)
    assert torch.equal(rollout["h"][0], h) and torch.equal(rollout["z"][0], z)
    again = imagination.imagine(rssm, goal, critic, h, z, CFG, torch.Generator().manual_seed(0))
    assert torch.equal(rollout["actions"], again["actions"]) and torch.equal(rollout["z"], again["z"])  # one stream decides
    out = imagination.critic_loss(critic, slow, goal, rollout, CFG)
    out["critic_loss"].backward()
    assert all(p.grad is not None for p in critic.parameters())
    assert all(p.grad is None for p in rssm.parameters()) and all(p.grad is None for p in goal.parameters())
    assert all(p.grad is None for p in slow.parameters())


def test_a_start_at_the_goal_does_not_count():
    rssm, goal, critic = _model()
    with torch.no_grad():  # a goal head that says "goal" everywhere: c = 0 for every state
        goal.net[-1].weight.zero_()
        goal.net[-1].bias.fill_(30.0)
    h, z = _states(7)
    rollout = imagination.imagine(rssm, goal, critic, h, z, CFG, torch.Generator().manual_seed(0))
    out = imagination.critic_loss(critic, heads.make_slow_critic(critic), goal, rollout, CFG)
    assert float(out["weight_mean"]) < 1e-9 and float(out["critic_loss"].detach()) < 1e-6


def test_goal_head_learns_the_reward_signal_and_the_heads_never_move_the_world_model():
    rssm, goal, critic = _model()
    slow = heads.make_slow_critic(critic)
    goal_opt = torch.optim.Adam(goal.parameters(), lr=1e-2)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    h, z = _states(64)
    h, z = h.reshape(8, 8, 16), z.reshape(8, 8, 3, 4)
    reward = torch.where(z[..., 0, 0] > 0, 50.0, -1.0)  # "goal" = the first variable is in class 0
    before = [p.clone() for p in rssm.parameters()]
    gen = torch.Generator().manual_seed(0)
    for _ in range(150):
        out = imagination.train_heads_step(rssm, goal, critic, slow, goal_opt, critic_opt, h, z, reward, CFG, gen)
    assert out["goal_recall"] == 1.0 and out["goal_false_positive_rate"] == 0.0 and out["goal_loss"] < 0.05
    assert all(torch.equal(a, b) for a, b in zip(before, rssm.parameters()))
    assert set(out) >= {"critic_loss", "critic_grad_norm", "imagined_tie_fraction", "goal_target_rate"}


def test_the_filter_follows_the_training_unroll():
    """LatentFilter = RSSM.observation_step along an episode: the same h as loop.unroll_observation."""
    from helplessness.agents.worldmodel.loop import unroll_observation

    rssm, _, _ = _model()
    x = torch.rand(1, 5, 3, 24, 24)
    a = torch.tensor([[1, 4, 0, 2, 3]])
    is_first = torch.tensor([[True, False, False, True, False]])
    with torch.no_grad():
        ref = unroll_observation(rssm, x, a, is_first, generator=torch.Generator().manual_seed(2), decode=False)
    f = policy.LatentFilter(rssm, torch.device("cpu"), torch.Generator().manual_seed(2))
    for t in range(5):
        h, z = f.observe(x[0, t], bool(is_first[0, t]))
        f.set_action(int(a[0, t]))
        assert torch.allclose(h[0], ref["h"][0, t], atol=1e-6) and torch.equal(z[0].argmax(-1), ref["z"][0, t].argmax(-1))


def test_untrained_heads_have_no_preference_so_the_agent_explores():
    """Zero-initialised output layers (the default): every Q ties, the tie-break is uniform."""
    rssm, _, _ = _model()
    goal, critic = GoalHead(16, 12, hidden_dim=32), Critic(16, 12, hidden_dim=32)
    h, z = _states(50)
    Q = policy.q_values(rssm, goal, critic, h, z, CFG.k_act, CFG, torch.Generator().manual_seed(0))
    assert torch.allclose(Q, Q[:, :1].expand_as(Q)) and policy.tie_set(Q, CFG).all()
    picks = policy.epsilon_greedy(Q, AgentConfig(epsilon=0.0), torch.Generator().manual_seed(0))
    assert picks["tied"].all() and set(picks["action"].tolist()) == {0, 1, 2, 3, 4}
