"""Tests for helplessness.agents.online.loop.train_online: the four correctness requirements
from CLAUDE.md's online task (step 2) -- truncation vs termination, random
tie-break, replan timing, and (indirectly, via helplessness.agents.online.agent) V=0 at
terminals -- plus the near-optimal single-agent validation run.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.evaluation.rollout import evaluate
from helplessness.planning.value_iteration import value_iteration
from helplessness.agents.online.agent import OnlineTabularAgent
from helplessness.agents.online.loop import train_online, train_online_with_burnin


def test_truncation_never_marks_terminal():
    # max_steps=3 on an otherwise ordinary env: every episode from most
    # start states truncates long before reaching goal/traps.
    env = GridWorld(p=0.0, seed=0, n_actions=5, max_steps=3)
    agent = OnlineTabularAgent(env, alpha=0.1)
    explore_rng = np.random.default_rng(0)
    tiebreak_rng = np.random.default_rng(1)

    truncations_seen = []

    def spy(t, s, a, r, s_next, terminated, truncated):
        if truncated:
            truncations_seen.append(s_next)

    train_online(env, agent, n_steps=300, epsilon=1.0, replan_every=1000,
                 explore_rng=explore_rng, tiebreak_rng=tiebreak_rng, on_transition=spy)

    assert truncations_seen, "test setup should produce at least one truncation"
    for s in truncations_seen:
        assert not agent.terminal[s], f"truncated state {s} was incorrectly marked terminal"


def test_random_tie_break_varies_policy_across_seeds():
    # before any data, the restricted-support prior is action-independent
    # at every state, so Q is exactly tied across all actions everywhere --
    # the ideal condition to expose a non-random (index-biased) argmax.
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)

    agent.plan(np.random.default_rng(1))
    policy_a = agent.policy.copy()
    agent.plan(np.random.default_rng(2))
    policy_b = agent.policy.copy()
    agent.plan(np.random.default_rng(3))
    policy_c = agent.policy.copy()

    assert not (np.array_equal(policy_a, policy_b) and np.array_equal(policy_b, policy_c)), (
        "policy identical across three different tie-break seeds -- argmax looks deterministic"
    )


def test_replan_happens_strictly_after_the_count_update():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    n_at_replan = []

    real_plan = agent.plan
    def spy_plan(rng):
        n_at_replan.append(int(agent.N.sum()))
        real_plan(rng)
    agent.plan = spy_plan

    explore_rng = np.random.default_rng(0)
    tiebreak_rng = np.random.default_rng(1)
    train_online(env, agent, n_steps=6, epsilon=1.0, replan_every=2,
                 explore_rng=explore_rng, tiebreak_rng=tiebreak_rng)

    # initial plan (before the loop) sees 0 transitions; each subsequent
    # plan at t=2,4,6 must see exactly t counted transitions -- i.e. the
    # count update for step t has already happened.
    assert n_at_replan == [0, 2, 4, 6]


def test_learning_false_never_updates_model():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    explore_rng = np.random.default_rng(0)
    tiebreak_rng = np.random.default_rng(1)
    train_online(env, agent, n_steps=50, epsilon=1.0, replan_every=10,
                 explore_rng=explore_rng, tiebreak_rng=tiebreak_rng, learning=False)
    assert agent.N.sum() == 0
    assert not agent.terminal.any()


def _run_phase1(seed, burn_in_steps, n_steps=50_000, epsilon=0.1):
    p = 0.0
    train_env = GridWorld(p=p, seed=seed, n_actions=5, step_reward=-1.0, goal_reward=100.0, trap_reward=-100.0)
    train_env.seed_dynamics(seed)
    agent = OnlineTabularAgent(train_env, alpha=0.1, gamma=0.95)
    explore_rng = np.random.default_rng(1_000_000 + seed)
    tiebreak_rng = np.random.default_rng(2_000_000 + seed)
    burn_in_rng = np.random.default_rng(5_000_000 + seed)
    train_online_with_burnin(
        train_env, agent, n_steps=n_steps, epsilon=epsilon, replan_every=500,
        burn_in_steps=burn_in_steps, explore_rng=explore_rng, tiebreak_rng=tiebreak_rng,
        burn_in_rng=burn_in_rng, learning=True,
    )
    return train_env, agent


def _eval_success(agent, seed_offset=0):
    eval_env = GridWorld(p=0.0, seed=3_000_000 + seed_offset, n_actions=5, step_reward=-1.0, goal_reward=100.0, trap_reward=-100.0)
    summary, _ = evaluate(eval_env, agent.policy, n_episodes=300, seed=4_000_000 + seed_offset)
    return summary


def test_single_agent_reaches_near_optimal_return_at_p_train_zero():
    """Step-2 validation, using the actual Phase-1 recipe (burn-in +
    epsilon-greedy, see train_online_with_burnin): an agent trained online
    at p_train=0 for a realistic budget must reach near-oracle greedy return
    when evaluated at p=0 -- if this fails, nothing built on top of this
    loop is meaningful.

    seed=0 was, without a burn-in, one of the ~40% of seeds that never
    discovered the goal at all in 50k steps at epsilon=0.1 -- a reproducible
    exploration deadlock at the two trap-adjacent chokepoints connecting the
    two halves of the default layout (see online/README.md). Used
    deliberately here (rather than a seed that happened to work anyway) to
    demonstrate the burn-in actually fixes the failure mode it targets."""
    train_env, agent = _run_phase1(seed=0, burn_in_steps=2_000)
    summary = _eval_success(agent)

    T_true = train_env.transition_tensor(p=0.0)
    R_true = train_env.reward_vector()
    terminal_true = train_env.terminal_mask()
    _V_star, pi_star = value_iteration(T_true, R_true, terminal_true, gamma=0.95)
    eval_env = GridWorld(p=0.0, seed=3_000_000, n_actions=5, step_reward=-1.0, goal_reward=100.0, trap_reward=-100.0)
    oracle_summary, _ = evaluate(eval_env, pi_star, n_episodes=300, seed=4_000_000)

    assert summary["success_rate"] >= 0.9 * oracle_summary["success_rate"]
    assert summary["return_mean"] >= 0.8 * oracle_summary["return_mean"]


def test_burnin_fixes_multiple_previously_stuck_seeds():
    """Broader check than the single-seed test above: several seeds
    confirmed to deadlock at epsilon=0.1 with no burn-in must succeed once
    burn_in_steps=2000 is applied, without changing epsilon, replan_every,
    or the total transition budget."""
    for seed in (0, 3, 6):
        _train_env, agent = _run_phase1(seed=seed, burn_in_steps=2_000)
        summary = _eval_success(agent)
        assert summary["success_rate"] == pytest.approx(1.0), f"seed={seed} still stuck after burn-in"


def test_burnin_counts_against_budget_not_added_on_top():
    _train_env, agent = _run_phase1(seed=0, burn_in_steps=2_000, n_steps=10_000)
    assert agent.N.sum() == 10_000


def test_burnin_zero_is_equivalent_to_plain_train_online():
    def fresh():
        env = GridWorld(p=0.0, seed=0, n_actions=5)
        env.seed_dynamics(0)
        agent = OnlineTabularAgent(env, alpha=0.1)
        return env, agent, np.random.default_rng(1), np.random.default_rng(2)

    env, agent, explore_rng, tiebreak_rng = fresh()
    train_online(env, agent, 500, 0.2, 100, explore_rng, tiebreak_rng)
    N_plain, pi_plain = agent.N.copy(), agent.policy.copy()

    env, agent, explore_rng, tiebreak_rng = fresh()
    train_online_with_burnin(
        env, agent, 500, 0.2, 100, burn_in_steps=0,
        explore_rng=explore_rng, tiebreak_rng=tiebreak_rng, burn_in_rng=np.random.default_rng(99),
    )
    N_burnin, pi_burnin = agent.N.copy(), agent.policy.copy()

    assert np.array_equal(N_plain, N_burnin)
    assert np.array_equal(pi_plain, pi_burnin)


def test_burnin_rejects_out_of_range_steps():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    with pytest.raises(ValueError):
        train_online_with_burnin(
            env, agent, n_steps=100, epsilon=0.1, replan_every=10, burn_in_steps=200,
            explore_rng=np.random.default_rng(0), tiebreak_rng=np.random.default_rng(1),
            burn_in_rng=np.random.default_rng(2),
        )
