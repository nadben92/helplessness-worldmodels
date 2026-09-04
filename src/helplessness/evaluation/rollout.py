"""Policy rollout and evaluation.

evaluate() (and rollout(), its simpler predecessor) are the only pieces
outside envs/ that touch an environment, and they do so via the generic
reset()/step()/seed_dynamics() API (duck typing), without reading any
GridWorld-specific attribute (no env.goals or env.agent_rc): success is
detected via the sign of the terminal reward. This is what lets them run
unchanged against any future environment or the neural world model.

Trajectory format: list of episodes, each episode a list of tuples
(s, a, r, s_next, terminated, truncated).
"""

from __future__ import annotations

import numpy as np


def draw_start_states(reachable_states: list[int], n_episodes: int, seed: int) -> np.ndarray:
    """Draws n_episodes start states uniformly among reachable_states.

    Does not exclude states adjacent to a trap: that is exactly where the
    avoidance phenotype lives, excluding them would discard the most
    informative part of the policy. RNG dedicated to this draw, independent
    of any training seed -- calling this function with the same arguments
    (same reachable_states, same n_episodes, same seed) reproduces exactly
    the same sequence, so that ALL agents and ALL conditions share the same
    starts and observed differences are not confounded by different starts.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(reachable_states), size=n_episodes)


def rollout(env, policy, n_episodes=1000, seed=0):
    """Evaluates a policy (action array indexed by state) on the env.

    Simpler predecessor of evaluate(): fixed start, no trajectories
    returned. Kept for the aggregate-only use cases that don't need the
    behavioural metrics computed from raw trajectories.
    """
    env.seed_dynamics(seed)
    returns, successes, lengths, truncations = [], 0, [], 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        total, terminated, truncated, t = 0.0, False, False, 0
        while not (terminated or truncated):
            s, r, terminated, truncated, info = env.step(int(policy[s]))
            total += r
            t += 1
        returns.append(total)
        lengths.append(t)
        successes += int(env.goals[env.agent_rc])
        truncations += int(truncated)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": successes / n_episodes,
        "length_mean": float(np.mean(lengths)),
        "truncation_rate": truncations / n_episodes,
    }


def evaluate(
    env,
    policy: np.ndarray,
    n_episodes: int = 1000,
    seed: int = 0,
    start_states: np.ndarray | None = None,
):
    """Rolls out policy in env, returns (summary, trajectories).

    Provides the raw trajectories in addition to the summary stats so that
    the behavioural metrics don't need a second rollout. Success is detected
    via r > 0 at the terminal step (module convention: goal_reward > 0 >
    trap_reward), not via a GridWorld attribute, so that this function stays
    generic over any env that honours the reset()/step()/seed_dynamics() API.

    start_states: sequence of n_episodes start states, one per episode
    (typically produced by draw_start_states, shared across agents). None
    (default) keeps the historical protocol: fixed start at env.start_rc for
    all episodes -- reproducible with existing results. evaluate() only
    executes policy and start_states as given; the choice of policy
    (deterministic or with random tie-break) and of the starts is made by
    the caller, not here, to avoid duplicating that logic.
    """
    env.seed_dynamics(seed)
    trajectories = []
    returns, lengths, successes, truncations = [], [], 0, 0
    for i in range(n_episodes):
        start = None if start_states is None else int(start_states[i])
        s, _ = env.reset(start_state=start)
        episode = []
        total, terminated, truncated, t, last_r = 0.0, False, False, 0, 0.0
        while not (terminated or truncated):
            a = int(policy[s])
            s_next, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, r, s_next, terminated, truncated))
            total += r
            last_r = r
            t += 1
            s = s_next
        trajectories.append(episode)
        returns.append(total)
        lengths.append(t)
        successes += int(terminated and last_r > 0)
        truncations += int(truncated)

    summary = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": successes / n_episodes,
        "length_mean": float(np.mean(lengths)),
        "truncation_rate": truncations / n_episodes,
    }
    return summary, trajectories
