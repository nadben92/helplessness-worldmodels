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

from helplessness.planning.value_iteration import policy_evaluation, policy_evaluation_stochastic, value_iteration


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


def test_policy_evaluation_stochastic_reduces_to_deterministic_for_one_hot_rows():
    """One-hot policy_probs must reproduce policy_evaluation() exactly --
    a strict generalisation, not a second implementation (see E2's Delta
    convention, docs/E2_online_4zones.md)."""
    rng = np.random.default_rng(4)
    n_states, n_actions = 6, 4
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.normal(size=n_states)
    terminal = np.zeros(n_states, dtype=bool)
    terminal[0] = True

    policy = rng.integers(0, n_actions, size=n_states)
    V_det = policy_evaluation(policy, P, R, terminal, gamma=0.9)

    policy_probs = np.zeros((n_states, n_actions))
    policy_probs[np.arange(n_states), policy] = 1.0
    V_stoch = policy_evaluation_stochastic(policy_probs, P, R, terminal, gamma=0.9)
    assert V_stoch == pytest.approx(V_det, abs=1e-8)


def test_policy_evaluation_stochastic_hand_computed_uniform_mixture():
    # state 0: two actions, one goes straight to a terminal reward-10 state,
    # the other stays at state 0 forever (reward -1 per step). A uniform
    # mixture must average the two actions' own transition ROWS, not their
    # separately-evaluated values.
    n_states, n_actions = 2, 2
    P = np.zeros((n_states, n_actions, n_states))
    P[0, 0, 1] = 1.0  # action 0: straight to terminal
    P[0, 1, 0] = 1.0  # action 1: self-loop
    P[1, 0, 1] = 1.0
    P[1, 1, 1] = 1.0
    R = np.array([-1.0, 10.0])
    terminal = np.array([False, True])
    gamma = 0.9

    policy_probs = np.array([[0.5, 0.5], [1.0, 0.0]])
    V = policy_evaluation_stochastic(policy_probs, P, R, terminal, gamma=gamma)
    # V(0) = 0.5*(R[1] + gamma*V(1)) + 0.5*(R[0] + gamma*V(0)), V(1) = 0
    # V(0) = 0.5*10 + 0.5*(-1 + gamma*V(0)) => V(0)*(1 - 0.5*gamma) = 5 - 0.5
    expected_v0 = (5.0 - 0.5) / (1 - 0.5 * gamma)
    assert V[1] == 0.0
    assert V[0] == pytest.approx(expected_v0, abs=1e-6)


def test_policy_evaluation_stochastic_matches_e1_delta_when_no_ties(tmp_path):
    """Regression against E1's actual pipeline (frozen config, alpha=0,
    n=100): at this budget, real ties in the planner's Q are effectively
    absent, so Delta computed with the E1 convention (plain argmax,
    planning.value_iteration.value_iteration's own policy) and Delta
    computed with E2's convention (uniform_tie_policy_probs +
    policy_evaluation_stochastic) must agree exactly, to the 1e-9 clamp
    tolerance both conventions share (see
    experiments/E1_offline_4zones/measurement.py's clamp)."""
    import sys
    from pathlib import Path

    from helplessness.evaluation.tiebreak import uniform_tie_policy_probs

    e1_dir = str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones")
    sys.path.insert(0, e1_dir)
    try:
        from env import build_layout, build_env, reward_and_terminal
        from estimation import agent_from_counts, collect_generative
        from helplessness.config import load_yaml_block
    finally:
        sys.path.remove(e1_dir)

    cfg = load_yaml_block("zones/e1_4zones.yaml")
    layout = build_layout(cfg, direction="away_from_goal")
    P_true = build_env(layout).transition_tensor()
    R_true, terminal = reward_and_terminal(cfg, goal_reward=50.0)
    gamma = 0.9

    rng = np.random.default_rng(0)
    N = collect_generative(P_true, n_per_pair=100, rng=rng)
    agent = agent_from_counts(N, R_true, alpha=0.0)
    T_hat, R_hat = agent.T_hat, agent.R_hat

    V_star, _ = value_iteration(P_true, R_true, terminal, gamma=gamma)

    # Both conventions plan on the SAME belief (T_hat, R_hat) -- only how
    # the greedy policy is extracted from the resulting Q differs.
    V_believed, pi_hat = value_iteration(T_hat, R_hat, terminal, gamma=gamma)
    Q_hat = np.einsum("sap,p->sa", T_hat, R_hat + gamma * V_believed * (~terminal))

    # E1 convention: plain argmax policy (value_iteration's own pi_hat)
    delta_e1 = V_star - policy_evaluation(pi_hat, P_true, R_true, terminal, gamma=gamma)
    delta_e1 = np.where(np.abs(delta_e1) < 1e-9, 0.0, delta_e1)

    # E2 convention: uniform mixture over Q-ties from the same Q_hat
    policy_probs = uniform_tie_policy_probs(Q_hat, tol=1e-9)
    delta_e2 = V_star - policy_evaluation_stochastic(policy_probs, P_true, R_true, terminal, gamma=gamma)
    delta_e2 = np.where(np.abs(delta_e2) < 1e-9, 0.0, delta_e2)

    n_ties = int((policy_probs > 0).sum(axis=1).max())
    assert delta_e2 == pytest.approx(delta_e1, abs=1e-6), (
        f"E1 (argmax) vs E2 (uniform-tie) Delta diverge at n=100, alpha=0 "
        f"(max ties per state observed: {n_ties})"
    )


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
