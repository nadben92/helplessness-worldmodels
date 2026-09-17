"""Tests for the offline protocol (helplessness.agents.tabular, helplessness.agents.collect)."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.agents.collect import collect, uniform_policy
from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.envs.legacy.gridworld import GridWorld
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


# ---------------------------------------------------------------------
# alpha_support prior normalization -- E2's Dirichlet convention:
# T_hat(s'|s,a) = (N[s,a,s'] + alpha/5) / (N[s,a].sum() + alpha), support
# restricted to the 5 geographically reachable successors. See
# docs/E2_online_4zones.md.
# ---------------------------------------------------------------------


def _five_of_ten_support() -> np.ndarray:
    """A (10, 10) alpha_support mask, 5 True entries per row (states
    0..4), matching the shape of E2's restricted-support Dirichlet."""
    n_states = 10
    support = np.zeros((n_states, n_states), dtype=bool)
    for s in range(n_states):
        support[s, :5] = True
    return support


def test_t_hat_rows_sum_to_one_with_restricted_support():
    support = _five_of_ten_support()
    agent = TabularModelBasedAgent(n_states=10, n_actions=3, alpha=1.0, alpha_support=support)
    assert agent.T_hat.sum(axis=2) == pytest.approx(1.0)


def test_unvisited_pair_is_exactly_uniform_over_the_support_regardless_of_alpha():
    """At zero counts, T_hat(s'|s,a) = (alpha/5)/alpha = 1/5 on each
    support state and exactly 0 elsewhere -- the alpha cancels for a
    symmetric prior, so this holds for ANY alpha > 0, not just alpha=1.
    Distinguishes the correct posterior-mean formula from a naive
    (unnormalized) reading of "alpha/5" as the final probability."""
    support = _five_of_ten_support()
    for alpha in (0.5, 1.0, 5.0):
        agent = TabularModelBasedAgent(n_states=10, n_actions=3, alpha=alpha, alpha_support=support)
        row = agent.T_hat[0, 0]  # never visited: N is all zero
        assert row[:5] == pytest.approx(np.full(5, 0.2))
        assert row[5:] == pytest.approx(np.zeros(5))


def test_alpha_support_prior_mass_is_alpha_not_a_raw_zero_one_mask():
    """Regression check for the specific bug this formula is vulnerable
    to: using alpha_support directly as the per-cell prior pseudo-count
    (an unnormalized 0/1 mask contributing 1 per support cell, total
    prior mass = support size) instead of alpha_support * alpha / 5
    (total prior mass = alpha). Both variants still sum T_hat to 1 (so a
    normalization-only test cannot catch this), but they diverge on the
    EXACT value at a partially-visited pair -- checked here against the
    closed form (N[s,a,s'] + alpha/5) / (N[s,a].sum() + alpha)."""
    support = _five_of_ten_support()
    alpha = 5.0
    agent = TabularModelBasedAgent(n_states=10, n_actions=3, alpha=alpha, alpha_support=support)
    agent.N[0, 0, 1] = 10  # 10 observed transitions to successor state 1 (in the support)

    n_total = 10.0
    expected_visited_successor = (10.0 + alpha / 5) / (n_total + alpha)
    expected_other_support = (0.0 + alpha / 5) / (n_total + alpha)

    row = agent.T_hat[0, 0]
    assert row[1] == pytest.approx(expected_visited_successor)
    for s in (0, 2, 3, 4):
        assert row[s] == pytest.approx(expected_other_support)
    assert row[5:] == pytest.approx(np.zeros(5))
    # the two candidate (buggy vs correct) formulas give numerically
    # distinct answers at this alpha -- pin the actual numbers so a
    # regression to the unnormalized-mask bug is caught even if the
    # closed-form comparison above is ever loosened.
    assert row[1] == pytest.approx(11 / 15)
    assert row[0] == pytest.approx(1 / 15)


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
