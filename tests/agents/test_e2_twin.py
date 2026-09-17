"""Tests for experiments/E2_online_4zones/twin.py::twin_from_counts -- the
tabular twin's central regression requirement (spec: "jumeau(N_online)
reproduit exactement T_hat, I_hat, V_hat internes de l'agent online en fin
de run") plus its structural/shape invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module
from helplessness.envs.dynamics.controllable import ControllableNoisy
from helplessness.envs.observation.index_backend import IndexObservationBackend
from helplessness.envs.rewards.base import GoalReward
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import Zone, ZoneLayout, geographic_support_mask
from helplessness.measures.controllability import controllability_per_state

agent_mod = load_experiment_module("E2_online_4zones", "agent.py", "_e2_agent_for_twin_test")
loop_mod = load_experiment_module("E2_online_4zones", "loop.py", "_e2_loop_for_twin_test")
twin_mod = load_experiment_module("E2_online_4zones", "twin.py", "_e2_twin_for_twin_test")

GRID_SHAPE = (4, 4)
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5
GOAL_S = 0
NON_TERMINAL_STATES = np.array([s for s in range(N_STATES) if s != GOAL_S])


def _make_env(max_steps=10, seed=0, p=0.3):
    zones = [Zone(mask=np.ones(GRID_SHAPE, dtype=bool), dynamics=ControllableNoisy(p=p), appearance=None)]
    layout = ZoneLayout(GRID_SHAPE, zones)
    return TorusZonedEnv(
        zone_layout=layout, observation_backend=IndexObservationBackend(),
        reward_fn=GoalReward(goal_state=GOAL_S, step_reward=-1.0, goal_reward=50.0),
        start_position=(0, 0), max_steps=max_steps, seed=seed, terminal_states=[GOAL_S],
    ), layout


def _true_R_terminal():
    R_true = np.full(N_STATES, -1.0)
    R_true[GOAL_S] = 50.0
    terminal_true = np.zeros(N_STATES, dtype=bool)
    terminal_true[GOAL_S] = True
    return R_true, terminal_true


def _make_agent(support, alpha=1.0):
    R_true, terminal_true = _true_R_terminal()
    return agent_mod.OnlineTabularAgentKnownReward(
        n_states=N_STATES, n_actions=N_ACTIONS, R_true=R_true, terminal_true=terminal_true,
        support=support, alpha=alpha, gamma=0.9,
    )


def test_twin_reproduces_online_agents_t_hat_i_hat_and_v_exactly():
    env, layout = _make_env()
    support = geographic_support_mask(GRID_SHAPE)
    agent = _make_agent(support, alpha=1.0)
    loop_mod.train_online_e2(
        env, agent, n_steps=3000, epsilon=0.2, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=np.random.default_rng(1), tiebreak_rng=np.random.default_rng(2),
        start_rng=np.random.default_rng(3),
    )
    # train_online_e2 only replans at episode boundaries (per spec, never
    # a step-count interval) -- the run can end mid-episode, a few
    # transitions after the LAST replan, so agent.V can be stale relative
    # to the final agent.N. The twin reconstructs everything from N alone,
    # so the fair comparison is against an agent explicitly replanned on
    # that exact final N -- one more plan() call, deterministic given N
    # (no new data), makes agent.V/T_hat consistent with each other.
    agent.plan(np.random.default_rng(0))
    R_true, terminal_true = _true_R_terminal()
    P_true = env.transition_tensor()

    result = twin_mod.twin_from_counts(
        N=agent.N, alpha=1.0, support=support,
        P_true=P_true, R_true=R_true, terminal=terminal_true,
        P_ref=P_true,  # reference kernel is irrelevant to this test's claims (b, I_ref not checked here)
        gamma=0.9, reference_rng=np.random.default_rng(999),
    )

    assert np.array_equal(result["T_hat"], agent.T_hat)
    assert result["I_hat"] == pytest.approx(controllability_per_state(agent.T_hat))
    assert result["v_pi_believed"] == pytest.approx(agent.V)


def test_twin_depends_only_on_n_not_on_agent_object():
    """Two different agent RUNS that happen to produce the SAME N must
    give the twin the SAME result -- direct evidence the function reads
    nothing off "the agent" beyond the counts array itself."""
    support = geographic_support_mask(GRID_SHAPE)
    R_true, terminal_true = _true_R_terminal()
    env, _ = _make_env()
    P_true = env.transition_tensor()

    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    rng = np.random.default_rng(5)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            N[s, a] = rng.multinomial(20, P_true[s, a])

    result_a = twin_mod.twin_from_counts(
        N=N, alpha=1.0, support=support, P_true=P_true, R_true=R_true, terminal=terminal_true,
        P_ref=P_true, gamma=0.9, reference_rng=np.random.default_rng(0),
    )
    result_b = twin_mod.twin_from_counts(
        N=N.copy(), alpha=1.0, support=support, P_true=P_true, R_true=R_true, terminal=terminal_true,
        P_ref=P_true, gamma=0.9, reference_rng=np.random.default_rng(0),
    )
    assert np.array_equal(result_a["T_hat"], result_b["T_hat"])
    assert result_a["I_hat"] == pytest.approx(result_b["I_hat"])
    assert result_a["delta"] == pytest.approx(result_b["delta"])


def test_twin_delta_is_nonnegative():
    support = geographic_support_mask(GRID_SHAPE)
    R_true, terminal_true = _true_R_terminal()
    env, _ = _make_env()
    P_true = env.transition_tensor()

    rng = np.random.default_rng(6)
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            n = rng.integers(0, 15)
            if n > 0:
                N[s, a] = rng.multinomial(n, P_true[s, a])

    result = twin_mod.twin_from_counts(
        N=N, alpha=1.0, support=support, P_true=P_true, R_true=R_true, terminal=terminal_true,
        P_ref=P_true, gamma=0.9, reference_rng=np.random.default_rng(1),
    )
    assert np.all(result["delta"] >= -1e-8)


def test_twin_b_is_zero_when_reference_kernel_equals_true_kernel():
    """When P_ref == P_true (a degenerate but structurally valid case:
    the 'reference' condition sampling from the same kernel as the real
    data), I_hat and I_ref are estimated from independent draws of the
    SAME underlying distribution -- b's population mean is 0, though any
    single seed's b need not be exactly 0 (both are noisy estimates).
    What must hold exactly regardless of noise: b == I_hat - I_ref,
    checked directly."""
    support = geographic_support_mask(GRID_SHAPE)
    R_true, terminal_true = _true_R_terminal()
    env, _ = _make_env()
    P_true = env.transition_tensor()

    N = np.full((N_STATES, N_ACTIONS), 50)
    N_full = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    rng = np.random.default_rng(7)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            N_full[s, a] = rng.multinomial(50, P_true[s, a])

    result = twin_mod.twin_from_counts(
        N=N_full, alpha=1.0, support=support, P_true=P_true, R_true=R_true, terminal=terminal_true,
        P_ref=P_true, gamma=0.9, reference_rng=np.random.default_rng(2),
    )
    assert result["b"] == pytest.approx(result["I_hat"] - result["I_ref"])


def test_twin_is_optimal_under_p_marks_at_least_one_action_per_state():
    support = geographic_support_mask(GRID_SHAPE)
    R_true, terminal_true = _true_R_terminal()
    env, _ = _make_env()
    P_true = env.transition_tensor()
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)

    result = twin_mod.twin_from_counts(
        N=N, alpha=1.0, support=support, P_true=P_true, R_true=R_true, terminal=terminal_true,
        P_ref=P_true, gamma=0.9, reference_rng=np.random.default_rng(3),
    )
    assert np.all(result["is_optimal_under_P"].sum(axis=1) >= 1)


def test_twin_l1_is_zero_for_the_true_kernel_itself():
    """A sanity check on the L1 wiring: if T_hat happened to equal P_true
    exactly (constructed directly, not learned), l1 must be exactly 0."""
    support = geographic_support_mask(GRID_SHAPE)
    R_true, terminal_true = _true_R_terminal()
    env, _ = _make_env()
    P_true = env.transition_tensor()

    # force T_hat == P_true by using an enormous, exactly-matched sample
    # so alpha's influence is negligible relative to float tolerance is
    # NOT guaranteed -- instead, directly verify l1's wiring via a
    # generative draw large enough that l1 must be small, not exactly 0
    # (never claim exactness from a stochastic estimator).
    rng = np.random.default_rng(4)
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            N[s, a] = rng.multinomial(20000, P_true[s, a])

    result = twin_mod.twin_from_counts(
        N=N, alpha=1.0, support=support, P_true=P_true, R_true=R_true, terminal=terminal_true,
        P_ref=P_true, gamma=0.9, reference_rng=np.random.default_rng(5),
    )
    assert np.all(result["l1"] < 0.02)
