"""Tests for experiments/E2_online_4zones/loop.py::train_online_e2 --
the load-bearing behaviours the Temps-1 audit found missing in
helplessness.agents.online.loop.train_online (see docs/E2_online_4zones.md):
uniform start states from a dedicated RNG, replanning at episode
boundaries (not a step-count interval), the reset transition never
counted in N, and terminated/truncated never conflated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from _experiment_loader import load_experiment_module
from helplessness.envs.observation.index_backend import IndexObservationBackend
from helplessness.envs.rewards.base import GoalReward
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import Zone, ZoneLayout, geographic_support_mask

agent_mod = load_experiment_module("E2_online_4zones", "agent.py", "_e2_agent_for_loop_test")
loop_mod = load_experiment_module("E2_online_4zones", "loop.py", "_e2_loop_for_loop_test")

from helplessness.envs.dynamics.controllable import ControllableNoisy  # noqa: E402

GRID_SHAPE = (4, 4)
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5
GOAL_S = 0


def _make_env(max_steps=36, seed=0):
    zones = [Zone(mask=np.ones(GRID_SHAPE, dtype=bool), dynamics=ControllableNoisy(p=0.3), appearance=None)]
    layout = ZoneLayout(GRID_SHAPE, zones)
    return TorusZonedEnv(
        zone_layout=layout, observation_backend=IndexObservationBackend(),
        reward_fn=GoalReward(goal_state=GOAL_S, step_reward=-1.0, goal_reward=50.0),
        start_position=(0, 0), max_steps=max_steps, seed=seed, terminal_states=[GOAL_S],
    )


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


NON_TERMINAL_STATES = np.array([s for s in range(N_STATES) if s != GOAL_S])


def test_n_sum_equals_steps_executed_never_counting_resets():
    env = _make_env(max_steps=10)
    agent = _make_agent()
    n_steps = 2000
    loop_mod.train_online_e2(
        env, agent, n_steps=n_steps, epsilon=0.2, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=np.random.default_rng(1), tiebreak_rng=np.random.default_rng(2),
        start_rng=np.random.default_rng(3),
    )
    assert int(agent.N.sum()) == n_steps


def test_shuffling_logged_transitions_reproduces_same_n_and_belief():
    env = _make_env(max_steps=10)
    agent = _make_agent()
    logged = []
    loop_mod.train_online_e2(
        env, agent, n_steps=800, epsilon=0.2, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=np.random.default_rng(10), tiebreak_rng=np.random.default_rng(11),
        start_rng=np.random.default_rng(12),
        on_transition=lambda t, s, a, r, s_next, terminated, truncated: logged.append((s, a, r, s_next, terminated)),
    )
    original_N = agent.N.copy()
    original_T_hat = agent.T_hat.copy()

    rng = np.random.default_rng(999)
    shuffled = logged.copy()
    rng.shuffle(shuffled)

    replay_agent = _make_agent()
    for s, a, r, s_next, terminated in shuffled:
        replay_agent.observe(s, a, r, s_next, terminated)

    assert np.array_equal(replay_agent.N, original_N)
    assert np.allclose(replay_agent.T_hat, original_T_hat)


def test_start_states_uniform_direct_reset_tracking():
    """Direct measurement: wrap env.reset to record every start state
    drawn across the whole run, then chi-square test uniformity over the
    non-terminal states."""
    env = _make_env(max_steps=3)
    agent = _make_agent()
    starts = []
    real_reset = env.reset

    def tracking_reset(*args, **kwargs):
        obs, info = real_reset(*args, **kwargs)
        starts.append(info["state"])
        return obs, info

    env.reset = tracking_reset
    loop_mod.train_online_e2(
        env, agent, n_steps=6000, epsilon=1.0, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=np.random.default_rng(30), tiebreak_rng=np.random.default_rng(31),
        start_rng=np.random.default_rng(32),
    )
    assert GOAL_S not in starts
    counts = np.array([starts.count(s) for s in NON_TERMINAL_STATES])
    assert counts.sum() == len(starts)
    chi2, p_value = chisquare(counts)
    assert p_value > 0.01, f"start states look non-uniform: counts={counts}, p={p_value:.4f}"


def test_replan_happens_at_episode_boundaries_not_a_step_interval():
    env = _make_env(max_steps=3)  # forces frequent truncation -> frequent episode boundaries
    agent = _make_agent()
    plan_calls = {"n": 0}
    real_plan = agent.plan

    def counting_plan(*args, **kwargs):
        plan_calls["n"] += 1
        return real_plan(*args, **kwargs)

    agent.plan = counting_plan

    n_episode_boundaries = {"n": 0}

    def on_transition(t, s, a, r, s_next, terminated, truncated):
        if terminated or truncated:
            n_episode_boundaries["n"] += 1

    loop_mod.train_online_e2(
        env, agent, n_steps=500, epsilon=0.3, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=np.random.default_rng(40), tiebreak_rng=np.random.default_rng(41),
        start_rng=np.random.default_rng(42), on_transition=on_transition,
    )
    # exactly 1 initial plan (on the prior) + 1 replan per episode boundary
    assert plan_calls["n"] == 1 + n_episode_boundaries["n"]


def test_terminated_and_truncated_are_independent_signals_reward_matches_terminated():
    """terminated is purely a function of s_next (arrival at the goal);
    truncated is purely a function of the step counter -- the two CAN
    coincide (goal reached exactly on the step budget's last step, per
    Gymnasium's own convention: terminated takes precedence for
    bootstrapping, see CLAUDE.md "only terminated=True is genuinely
    absorbing"), so they must never be asserted mutually exclusive. What
    must hold unconditionally: terminated is true iff s_next is the goal,
    and the reward always matches which state was actually reached,
    regardless of truncated's value."""
    env = _make_env(max_steps=5)
    agent = _make_agent()
    events = []
    loop_mod.train_online_e2(
        env, agent, n_steps=2000, epsilon=0.3, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=np.random.default_rng(50), tiebreak_rng=np.random.default_rng(51),
        start_rng=np.random.default_rng(52),
        on_transition=lambda t, s, a, r, s_next, terminated, truncated: events.append((s_next, r, terminated, truncated)),
    )
    saw_terminated = False
    saw_truncated = False
    saw_both_at_once = False
    for s_next, r, terminated, truncated in events:
        assert terminated == (s_next == GOAL_S)
        assert r == pytest.approx(50.0 if s_next == GOAL_S else -1.0)
        saw_terminated |= terminated
        saw_truncated |= truncated
        saw_both_at_once |= terminated and truncated
    assert saw_terminated and saw_truncated  # both must actually occur in this setup, or the test is vacuous
