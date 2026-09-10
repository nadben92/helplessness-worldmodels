"""Tests for online.agent: the restricted-support Dirichlet prior, online
terminal discovery, and the optional constant-step-size update mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.gridworld import GridWorld
from online.agent import OnlineTabularAgent, geographic_support


def test_geographic_support_is_neighbours_plus_self():
    # dedicated wall-free layout so the expected support of an interior cell
    # is exactly {s, up, down, left, right} with no ambiguity from the
    # default (maze-like) layout's walls.
    open_layout = "S....\n.....\n.....\n.....\n....G"
    env = GridWorld(layout=open_layout, p=0.3, seed=0, n_actions=5)
    support = geographic_support(env)
    assert support.shape == (env.n_states, env.n_states)
    assert support.diagonal().all()  # every state includes itself

    r, c = 2, 2  # interior cell, all 4 neighbours in-bounds and open
    s = env.rc_to_s((r, c))
    expected = {s, env.rc_to_s((r - 1, c)), env.rc_to_s((r + 1, c)), env.rc_to_s((r, c - 1)), env.rc_to_s((r, c + 1))}
    assert set(np.flatnonzero(support[s])) == expected


def test_prior_restricted_to_support_for_unvisited_pairs():
    env = GridWorld(p=0.2, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    support = agent.support
    T = agent.T_hat  # nothing observed yet -- pure prior everywhere
    for s in range(env.n_states):
        outside = ~support[s]
        assert np.allclose(T[s, :, outside], 0.0), f"prior mass leaked outside support at s={s}"


def test_terminal_is_learned_not_read_from_env():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    # even though the env knows exactly which states are terminal, the agent
    # must start with no such knowledge.
    assert not agent.terminal.any()

    goal_s = next(s for s in range(env.n_states) if env.goals[env.s_to_rc(s)])
    agent.observe(s=0, a=0, r=1.0, s_next=goal_s, terminated=True)
    assert agent.terminal[goal_s]
    # nothing else was marked terminal by that single observation
    assert agent.terminal.sum() == 1


def test_plan_gives_zero_value_at_discovered_terminal_states():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    goal_s = next(s for s in range(env.n_states) if env.goals[env.s_to_rc(s)])
    agent.observe(s=0, a=0, r=1.0, s_next=goal_s, terminated=True)
    agent.plan(np.random.default_rng(0))
    assert agent.V[goal_s] == pytest.approx(0.0)


def test_eta_update_mode_off_by_default_matches_counts():
    env = GridWorld(p=0.1, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    assert not agent.use_eta_update
    agent.observe(s=3, a=1, r=-0.01, s_next=4, terminated=False)
    assert np.array_equal(agent.T_hat, agent._model.T_hat)


def test_eta_update_mode_moves_towards_observed_successor():
    env = GridWorld(p=0.1, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1, use_eta_update=True, eta=0.5)
    s, a, s_next = 3, 1, 4
    before = agent.T_hat[s, a, s_next]
    agent.observe(s=s, a=a, r=-0.01, s_next=s_next, terminated=False)
    after = agent.T_hat[s, a, s_next]
    assert after > before
    assert np.allclose(agent.T_hat[s, a].sum(), 1.0)


def _trained_agent(seed=0, n_transitions=3000):
    from online.loop import train_online_with_burnin

    env = GridWorld(p=0.0, seed=seed, n_actions=5, step_reward=-1.0, goal_reward=100.0, trap_reward=-100.0)
    env.seed_dynamics(seed)
    agent = OnlineTabularAgent(env, alpha=0.1, gamma=0.95)
    train_online_with_burnin(
        env, agent, n_transitions, 0.1, 200, burn_in_steps=500,
        explore_rng=np.random.default_rng(1), tiebreak_rng=np.random.default_rng(2),
        burn_in_rng=np.random.default_rng(3), learning=True,
    )
    return env, agent


def test_rescale_counts_preserves_perceived_controllability_and_sums_to_cap():
    # tol relaxed vs. the class default: this agent trains for only 3000
    # transitions (for test speed), far below the real 50k Phase-1 budget
    # rescale_counts's default tol was calibrated against -- see its
    # docstring. sums-to-cap and row-proportion checks below don't depend
    # on the tolerance at all.
    _env, agent = _trained_agent()
    before_bits = agent.N.sum()
    assert before_bits > 0
    agent.rescale_counts(n_cap=500.0, tol=2.0)
    assert agent.N.sum() == pytest.approx(500.0)


def test_rescale_counts_preserves_row_proportions():
    _env, agent = _trained_agent()
    visited = np.argwhere(agent.N.sum(axis=2) > 0)
    s, a = visited[0]
    before_row = agent._model.N[s, a].copy()
    agent.rescale_counts(n_cap=500.0, tol=2.0)
    after_row = agent._model.N[s, a]
    # a global scalar rescale must preserve within-row proportions exactly
    nz = before_row > 0
    ratios = after_row[nz] / before_row[nz]
    assert np.allclose(ratios, ratios[0])


def test_save_load_round_trip_reproduces_agent_exactly(tmp_path):
    _env, agent = _trained_agent()
    agent.plan(np.random.default_rng(0))
    path = str(tmp_path / "snapshot.npz")
    agent.save(path)

    loaded = OnlineTabularAgent.load(path)
    assert loaded.n_states == agent.n_states
    assert loaded.n_actions == agent.n_actions
    assert np.array_equal(loaded.support, agent.support)
    assert np.array_equal(loaded.terminal, agent.terminal)
    assert np.array_equal(loaded.N, agent.N)
    assert np.allclose(loaded.T_hat, agent.T_hat)
    assert np.allclose(loaded.R_hat, agent.R_hat)

    loaded.plan(np.random.default_rng(0))
    assert np.array_equal(loaded.policy, agent.policy)


def test_save_load_round_trip_with_naive_untrained_agent(tmp_path):
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1, gamma=0.95)
    path = str(tmp_path / "naive.npz")
    agent.save(path)
    loaded = OnlineTabularAgent.load(path)
    assert loaded.N.sum() == 0
    assert not loaded.terminal.any()
    assert np.allclose(loaded.T_hat, agent.T_hat)
