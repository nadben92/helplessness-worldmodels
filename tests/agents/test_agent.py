"""Tests for the offline protocol (helplessness.agents.tabular, helplessness.agents.collect)."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.agents.collect import collect, uniform_policy
from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.envs.gridworld import GridWorld
from helplessness.planning.value_iteration import value_iteration


def test_T_hat_converges_to_true_tensor_as_samples_grow():
    p = 0.4

    def tv_at_budget(budget):
        env = GridWorld(p=p, seed=0)
        env.seed_dynamics(0)
        rng = np.random.default_rng(1)
        data = collect(env, uniform_policy, budget, rng)

        agent = TabularModelBasedAgent(env.n_states, 4, alpha=0.1)
        for s, a, r, s_next, _terminated in data:
            agent.observe(s, a, r, s_next)

        T_true = env.transition_tensor()
        visited = agent.visited_mask
        tv = 0.5 * np.abs(agent.T_hat - T_true).sum(axis=2)
        return float(tv[visited].mean())

    tv_small = tv_at_budget(1_000)
    tv_large = tv_at_budget(100_000)
    assert tv_large < tv_small
    assert tv_large < 0.05


def test_T_hat_rows_sum_to_one_even_when_unvisited():
    agent = TabularModelBasedAgent(n_states=10, n_actions=4, alpha=0.1)
    assert np.allclose(agent.T_hat.sum(axis=2), 1.0)


def test_collect_returns_exact_transition_count_regardless_of_episode_length():
    for p in (0.0, 0.5, 1.0):
        env = GridWorld(p=p, seed=0, max_steps=5)  # short episodes -> many resets
        env.seed_dynamics(0)
        rng = np.random.default_rng(0)
        data = collect(env, uniform_policy, 777, rng)
        assert len(data) == 777


def test_agent_plan_matches_oracle_when_given_true_tensor(monkeypatch):
    env = GridWorld(p=0.4, seed=0)
    T_true = env.transition_tensor()
    R_true = env.reward_vector()
    terminal = env.terminal_mask()
    V_star, pi_star = value_iteration(T_true, R_true, terminal)

    agent = TabularModelBasedAgent(env.n_states, 4, alpha=0.1)
    # Bypasses the counts: forces T_hat/R_hat to be the true tensor/reward,
    # to check that plan() correctly delegates to value_iteration.
    monkeypatch.setattr(TabularModelBasedAgent, "T_hat", property(lambda self: T_true))
    monkeypatch.setattr(TabularModelBasedAgent, "R_hat", property(lambda self: R_true))
    agent.plan(terminal)

    assert np.allclose(agent.V, V_star)
    assert np.array_equal(agent.policy, pi_star)


def test_determinism_same_seed_same_results():
    def run():
        env = GridWorld(p=0.5, seed=0)
        env.seed_dynamics(42)
        rng = np.random.default_rng(42)
        data = collect(env, uniform_policy, 3000, rng)

        agent = TabularModelBasedAgent(env.n_states, 4, alpha=0.1)
        for s, a, r, s_next, _terminated in data:
            agent.observe(s, a, r, s_next)
        agent.plan(env.terminal_mask())
        return data, agent.T_hat.copy(), agent.policy.copy()

    data1, T1, pi1 = run()
    data2, T2, pi2 = run()
    assert data1 == data2
    assert np.array_equal(T1, T2)
    assert np.array_equal(pi1, pi2)
