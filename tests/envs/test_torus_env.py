"""Tests for TorusZonedEnv: the model/simulation agreement check CLAUDE.md
requires of any environment's dynamics, torus wraparound, zone routing,
both observation backends, and interop with the existing shared evaluate()/
value_iteration() (proving the Env interface really is enough).
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.appearance.base import AppearanceSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic
from helplessness.envs.observation.index_backend import IndexObservationBackend
from helplessness.envs.observation.pixel_backend import PixelObservationBackend
from helplessness.envs.rewards.base import GoalReward, RewardFn, ZeroReward
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import Zone, ZoneLayout
from helplessness.evaluation.rollout import evaluate
from helplessness.planning.value_iteration import policy_evaluation, value_iteration

GRID_SHAPE = (4, 5)


def _two_zone_layout(top_dynamics, bottom_dynamics):
    n_rows, n_cols = GRID_SHAPE
    top = np.zeros(GRID_SHAPE, dtype=bool)
    top[: n_rows // 2, :] = True
    bottom = ~top
    return ZoneLayout(
        GRID_SHAPE,
        [
            Zone(mask=top, dynamics=top_dynamics, appearance=AppearanceSpec("top")),
            Zone(mask=bottom, dynamics=bottom_dynamics, appearance=AppearanceSpec("bottom")),
        ],
    )


def _make_env(layout, backend=None, seed=0):
    return TorusZonedEnv(
        zone_layout=layout,
        observation_backend=backend or IndexObservationBackend(),
        reward_fn=ZeroReward(),
        start_position=(0, 0),
        max_steps=20,
        seed=seed,
    )


def test_model_simulation_agreement():
    """The load-bearing test: transition_tensor() must be exactly the law
    step() induces -- same discipline as the legacy GridWorld's own test."""
    layout = _two_zone_layout(ControllableNoisy(p=0.4), UncontrollableDeterministic(drift=(0, 1)))
    env = _make_env(layout)
    P = env.transition_tensor()

    n_samples = 4000
    tol = 0.03
    for rc in [(0, 0), (1, 2), (2, 0), (3, 3)]:
        s = env.rc_to_s(rc)
        for a in range(env.n_actions):
            counts = np.zeros(env.n_states)
            for _ in range(n_samples):
                env.position = rc
                env.t = 0
                _, _, _, _, info = env.step(a)
                counts[info["state"]] += 1
            empirical = counts / n_samples
            assert np.allclose(empirical, P[s, a], atol=tol), f"s={s} a={a}: {empirical} vs {P[s, a]}"


def test_torus_wraparound_no_walls():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    env.position = (0, 0)
    env.t = 0
    obs, _r, _term, _trunc, info = env.step(0)  # "up" from row 0 wraps to last row
    assert info["state"] == env.rc_to_s((GRID_SHAPE[0] - 1, 0))


def test_zone_routes_to_correct_dynamics():
    layout = _two_zone_layout(UncontrollableDeterministic(drift=(1, 0)), ControllableDeterministic())
    env = _make_env(layout)
    # top zone: action ignored, fixed drift
    env.position = (0, 0)
    env.t = 0
    _, _, _, _, info = env.step(3)  # "left", irrelevant under the conveyor
    assert info["state"] == env.rc_to_s((1, 0))
    # bottom zone: action determines outcome
    env.position = (GRID_SHAPE[0] - 1, 0)
    env.t = 0
    _, _, _, _, info = env.step(1)  # "right"
    assert info["state"] == env.rc_to_s((GRID_SHAPE[0] - 1, 1))


def test_no_terminal_states_by_default():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    assert not env.terminal_mask().any()


def test_truncation_at_max_steps():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    env.max_steps = 3
    env.reset()
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(4)  # stay
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(4)
    assert not terminated and truncated


def test_index_backend_returns_flat_state():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout, backend=IndexObservationBackend())
    env.position = (2, 3)
    obs = env._obs()
    assert obs == env.rc_to_s((2, 3))


def test_pixel_backend_returns_colored_grid_with_agent_marker():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout, backend=PixelObservationBackend())
    env.position = (0, 0)
    obs = env._obs()
    assert obs.shape == (GRID_SHAPE[0], GRID_SHAPE[1], 3)
    assert np.allclose(obs[0, 0], (1.0, 1.0, 1.0))  # agent marker overrides zone colour
    assert not np.allclose(obs[GRID_SHAPE[0] - 1, 0], (1.0, 1.0, 1.0))  # elsewhere, zone colour


def test_reward_fn_is_pluggable():
    class ConstantReward(RewardFn):
        def __call__(self, s: int) -> float:
            return 7.0

    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    env.reward_fn = ConstantReward()
    assert np.all(env.reward_vector() == 7.0)
    env.reset()
    _, r, _, _, _ = env.step(4)
    assert r == 7.0


def test_controllability_reflects_zone_mix():
    """Full control everywhere -> ceiling log2(n_actions); full
    uncontrollable everywhere -> exactly 0 -- reusing the one existing
    measure, unmodified."""
    all_controllable = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env_hi = _make_env(all_controllable)
    assert env_hi.controllability() == pytest.approx(np.log2(env_hi.n_actions), abs=1e-6)

    all_uncontrollable = _two_zone_layout(
        UncontrollableDeterministic(drift=(1, 0)), UncontrollableDeterministic(drift=(0, 1))
    )
    env_lo = _make_env(all_uncontrollable)
    assert env_lo.controllability() == pytest.approx(0.0, abs=1e-9)


def test_no_terminal_states_when_terminal_states_omitted():
    """terminal_states defaults to None -- every pre-existing call site
    (E1's build_env(), every test above) must keep getting terminal_mask()
    all-False exactly as before this constructor argument existed."""
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = TorusZonedEnv(
        zone_layout=layout, observation_backend=IndexObservationBackend(),
        reward_fn=ZeroReward(), start_position=(0, 0), max_steps=20, seed=0,
    )
    assert not env.terminal_mask().any()


def test_terminal_states_sets_terminal_mask():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    goal_rc = (0, 0)
    goal_s = goal_rc[0] * GRID_SHAPE[1] + goal_rc[1]
    env = TorusZonedEnv(
        zone_layout=layout, observation_backend=IndexObservationBackend(),
        reward_fn=ZeroReward(), start_position=(2, 2), max_steps=20, seed=0,
        terminal_states=[goal_s],
    )
    mask = env.terminal_mask()
    assert mask[goal_s]
    assert mask.sum() == 1


def test_terminal_states_produce_terminated_true_on_arrival_only():
    """terminated must fire exactly on arrival at a terminal state, and
    truncated/terminated must never both be produced by the same
    mechanism -- terminated is purely a function of s_next, truncated
    purely a function of the step counter (see CLAUDE.md: the two are
    never conflated)."""
    n_rows, n_cols = GRID_SHAPE
    goal_rc = (0, 1)
    goal_s = goal_rc[0] * n_cols + goal_rc[1]
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = TorusZonedEnv(
        zone_layout=layout, observation_backend=IndexObservationBackend(),
        reward_fn=GoalReward(goal_state=goal_s, step_reward=-1.0, goal_reward=50.0),
        start_position=(0, 0), max_steps=100, seed=0, terminal_states=[goal_s],
    )
    env.reset()
    # one "right" step from (0,0) lands exactly on the goal (0,1)
    obs, r, terminated, truncated, info = env.step(1)
    assert info["state"] == goal_s
    assert terminated and not truncated
    assert r == pytest.approx(50.0)

    # a non-goal arrival: terminated must be False, reward is step_reward
    env.reset()
    obs, r, terminated, truncated, info = env.step(4)  # "stay" at (0,0), not the goal
    assert not terminated
    assert r == pytest.approx(-1.0)


def test_goal_reward_replaces_step_reward_not_added_to_it():
    """GoalReward must match reward_and_terminal()'s convention exactly:
    R[goal] = goal_reward, never step_reward + goal_reward."""
    reward_fn = GoalReward(goal_state=5, step_reward=-1.0, goal_reward=50.0)
    assert reward_fn(5) == pytest.approx(50.0)
    assert reward_fn(6) == pytest.approx(-1.0)


def test_monte_carlo_return_matches_policy_evaluation():
    """The load-bearing consistency check for the TorusZonedEnv extension:
    the discounted return actually produced by step()/reset() under a
    fixed policy must equal policy_evaluation(policy, P, R, terminal,
    gamma) up to Monte Carlo error, from several start states -- proof
    that the env's real interaction loop reproduces exactly what the
    exact-kernel planner assumes (CLAUDE.md: step() must stay consistent
    with transition_tensor(); this extends that discipline to reward/
    terminal, not just dynamics)."""
    grid_shape = (4, 4)
    n_states = grid_shape[0] * grid_shape[1]
    goal_s = 0
    zones = [Zone(mask=np.ones(grid_shape, dtype=bool), dynamics=ControllableNoisy(p=0.3), appearance=AppearanceSpec("all"))]
    layout = ZoneLayout(grid_shape, zones)

    gamma = 0.9
    step_reward, goal_reward = -1.0, 50.0
    env = TorusZonedEnv(
        zone_layout=layout, observation_backend=IndexObservationBackend(),
        reward_fn=GoalReward(goal_state=goal_s, step_reward=step_reward, goal_reward=goal_reward),
        start_position=(0, 0), max_steps=500, seed=0, terminal_states=[goal_s],
    )

    P = env.transition_tensor()
    R = env.reward_vector()
    terminal = env.terminal_mask()
    rng = np.random.default_rng(0)
    policy = rng.integers(env.n_actions, size=n_states)  # a fixed, arbitrary policy
    V_exact = policy_evaluation(policy, P, R, terminal, gamma=gamma)

    n_episodes = 4000
    for start_s in (1, 2, 5, 10):
        env.seed_dynamics(123)
        returns = []
        n_truncated = 0
        for ep in range(n_episodes):
            s, _ = env.reset(start_state=start_s)
            total, discount, terminated, truncated = 0.0, 1.0, False, False
            while not (terminated or truncated):
                a = int(policy[s])
                s, r, terminated, truncated, _ = env.step(a)
                total += discount * r
                discount *= gamma
            n_truncated += int(truncated)
            returns.append(total)
        # max_steps=500 on a 16-state irreducible chain (full-support noise
        # every step) must make truncation essentially impossible -- assert
        # it outright rather than let a rare truncated (and thus
        # incomplete, un-bootstrapped) return silently bias mc_mean low.
        assert n_truncated == 0, f"start_s={start_s}: {n_truncated}/{n_episodes} episodes truncated"
        mc_mean = float(np.mean(returns))
        mc_se = float(np.std(returns) / np.sqrt(n_episodes))
        assert abs(mc_mean - V_exact[start_s]) < 6 * mc_se, (
            f"start_s={start_s}: MC mean {mc_mean:.4f} vs exact {V_exact[start_s]:.4f}, "
            f"6*SE={6 * mc_se:.4f}"
        )


def test_interop_with_existing_evaluate_and_value_iteration():
    """The whole point of sharing the Env ABC: value_iteration() and
    evaluate(), written for the legacy GridWorld, must work unchanged."""
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    T, R, terminal = env.transition_tensor(), env.reward_vector(), env.terminal_mask()
    V, policy = value_iteration(T, R, terminal, gamma=0.9)
    assert V.shape == (env.n_states,)
    summary, traj = evaluate(env, policy, n_episodes=5, seed=1)
    assert "return_mean" in summary
    assert len(traj) == 5
