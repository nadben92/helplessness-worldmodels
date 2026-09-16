"""Tests for planning.value_iteration.policy_evaluation: the fixed-policy
analogue of value_iteration(), required for E1's Delta(s) = V*(s) -
V^pi_hat(s) so that V^pi_hat is never the planner's own returned value
(computed under the wrong kernel, T_hat) but genuinely pi_hat evaluated
under the true P, per the spec's exact procedure.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from helplessness.planning.value_iteration import policy_evaluation, value_iteration


def test_policy_evaluation_contains_no_action_max():
    """Structural check: no argmax/action-axis max anywhere -- policy is
    fixed and never improved. np.max(...) is still allowed for the
    convergence-tolerance check (a reduction over states, the same pattern
    value_iteration() itself uses), and max_iter is just a parameter name
    -- what must be absent is any max/argmax over the action axis."""
    source = inspect.getsource(policy_evaluation)
    body = source.split('"""', 2)[-1]  # strip the docstring, which discusses value_iteration's argmax in prose
    assert "argmax" not in body
    assert ".max(axis=" not in body
    assert "Q.max" not in body


def test_policy_evaluation_reproduces_value_iteration_for_the_optimal_policy():
    """When policy IS P's own optimal (greedy) policy, policy_evaluation
    must reproduce value_iteration's V exactly (Delta == 0)."""
    rng = np.random.default_rng(0)
    n_states, n_actions = 5, 3
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.normal(size=n_states)
    terminal = np.zeros(n_states, dtype=bool)
    terminal[0] = True

    V_star, policy = value_iteration(P, R, terminal, gamma=0.9)
    V_pi = policy_evaluation(policy, P, R, terminal, gamma=0.9)
    assert V_pi == pytest.approx(V_star, abs=1e-6)


def test_delta_is_nonnegative_for_an_arbitrary_suboptimal_policy():
    """The fundamental MDP fact Delta must never violate: V* >= V^pi for
    ANY fixed policy pi under the true kernel P, always."""
    rng = np.random.default_rng(1)
    n_states, n_actions = 6, 4
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.normal(size=n_states)
    terminal = np.zeros(n_states, dtype=bool)
    terminal[0] = True

    V_star, _ = value_iteration(P, R, terminal, gamma=0.9)
    for _ in range(10):
        random_policy = rng.integers(0, n_actions, size=n_states)
        V_pi = policy_evaluation(random_policy, P, R, terminal, gamma=0.9)
        delta = V_star - V_pi
        assert np.all(delta >= -1e-8)


def test_policy_evaluation_hand_computed_two_state_chain():
    # state 0 -> state 1 (terminal, reward 10) under the fixed policy's one action
    n_states, n_actions = 2, 1
    P = np.zeros((n_states, n_actions, n_states))
    P[0, 0, 1] = 1.0
    P[1, 0, 1] = 1.0  # irrelevant, state 1 is terminal
    R = np.array([0.0, 10.0])
    terminal = np.array([False, True])
    policy = np.array([0, 0])
    V = policy_evaluation(policy, P, R, terminal, gamma=0.9)
    # V(1) = 0 (terminal); V(0) = R[1] (reward on arrival) + gamma * V(1) * 0 = 10
    assert V[1] == 0.0
    assert V[0] == pytest.approx(10.0)


def test_policy_evaluation_matches_geometric_series_on_a_self_loop():
    # single non-terminal state looping to itself forever, reward -1 per step
    n_states, n_actions = 1, 1
    P = np.ones((n_states, n_actions, n_states))
    R = np.array([-1.0])
    terminal = np.array([False])
    policy = np.array([0])
    gamma = 0.9
    V = policy_evaluation(policy, P, R, terminal, gamma=gamma)
    expected = -1.0 / (1 - gamma)  # sum_{t=0..inf} gamma^t * (-1)
    assert V[0] == pytest.approx(expected, abs=1e-6)


def test_terminal_state_gets_reward_on_arrival_but_no_further_bootstrap():
    """Same convention as value_iteration(): R contributes on arrival at a
    terminal state, but V[terminal] itself is forced to 0 (no bootstrapping
    beyond it)."""
    n_states, n_actions = 2, 1
    P = np.zeros((n_states, n_actions, n_states))
    P[0, 0, 1] = 1.0
    P[1, 0, 0] = 1.0
    R = np.array([0.0, 5.0])
    terminal = np.array([False, True])
    policy = np.array([0, 0])
    V = policy_evaluation(policy, P, R, terminal, gamma=0.5)
    assert V[1] == 0.0
    assert V[0] == pytest.approx(5.0)
