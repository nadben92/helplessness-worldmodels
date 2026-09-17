"""Tests for experiments/E2_online_4zones/agent.py::OnlineTabularAgentKnownReward:
plans on the TRUE reward/terminal (never learns R or terminal), and its
collection policy departages uniformly under the flat prior."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from _experiment_loader import load_experiment_module
from helplessness.envs.zones import geographic_support_mask

agent_mod = load_experiment_module("E2_online_4zones", "agent.py", "_e2_agent_for_agent_test")

GRID_SHAPE = (4, 4)
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5
GOAL_S = 0


def _make_agent(alpha=1.0):
    support = geographic_support_mask(GRID_SHAPE)
    R_true = np.full(N_STATES, -1.0)
    R_true[GOAL_S] = 50.0
    terminal_true = np.zeros(N_STATES, dtype=bool)
    terminal_true[GOAL_S] = True
    return agent_mod.OnlineTabularAgentKnownReward(
        n_states=N_STATES, n_actions=N_ACTIONS, R_true=R_true, terminal_true=terminal_true,
        support=support, alpha=alpha, gamma=0.9,
    )


def test_plan_uses_true_reward_and_terminal_never_r_hat():
    """The planner's Q must be built from R_true/terminal_true, not from
    any learned quantity -- checked by construction: an agent with ZERO
    observations already has a well-defined, non-degenerate V under the
    known reward (an uninformed T_hat still bootstraps a real R signal),
    which would be all r_default=0.0 under the learned-reward convention
    OnlineTabularAgentKnownReward deliberately does NOT use."""
    agent = _make_agent()
    tiebreak_rng = np.random.default_rng(0)
    agent.plan(tiebreak_rng)
    assert agent.V is not None
    # V(goal) forced to 0 (terminal); everywhere else V must reflect the
    # real step_reward/goal_reward mix, not all-zero (r_default's value).
    assert agent.V[GOAL_S] == 0.0
    assert np.any(agent.V != 0.0)


def test_observe_rejects_reward_mismatch_with_known_r_true():
    agent = _make_agent()
    with pytest.raises(AssertionError, match="disagree on the"):
        agent.observe(s=1, a=0, r=999.0, s_next=2, terminated=False)


def test_observe_rejects_terminated_mismatch_with_known_terminal():
    agent = _make_agent()
    with pytest.raises(AssertionError, match="disagree on the"):
        agent.observe(s=1, a=0, r=-1.0, s_next=2, terminated=True)  # state 2 is not the goal


def test_observe_accepts_a_consistent_transition():
    agent = _make_agent()
    agent.observe(s=1, a=0, r=-1.0, s_next=2, terminated=False)
    assert agent.N[1, 0, 2] == 1
    agent.observe(s=5, a=2, r=50.0, s_next=GOAL_S, terminated=True)
    assert agent.N[5, 2, GOAL_S] == 1


def test_collection_policy_is_uniform_under_the_flat_prior():
    """Sous le prior (zero data), every action at a state has the exact
    same Q -- argmax_random_tie must depart uniformly among the 5
    actions, not collapse onto the lowest index. Verified with a
    chi-squared test over many independent replans (each replan draws a
    fresh tie-break)."""
    agent = _make_agent()
    state_to_check = 6
    n_replans = 3000
    tiebreak_rng = np.random.default_rng(123)
    counts = np.zeros(N_ACTIONS, dtype=int)
    for _ in range(n_replans):
        agent.plan(tiebreak_rng)
        counts[agent.policy[state_to_check]] += 1

    chi2, p_value = chisquare(counts)
    assert p_value > 0.01, f"departage looks non-uniform: counts={counts}, chi2={chi2:.2f}, p={p_value:.4f}"
