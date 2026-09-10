"""Behavioural phenotypes -- computed on evaluation trajectories.

Free functions only, operating on trajectories (list of episodes, each
episode a list of tuples (s, a, r, s_next, terminated, truncated)): nothing
here imports the GridWorld class or any agent class, so these functions work
unchanged on trajectories collected from the future neural world model.
"""

from __future__ import annotations

import numpy as np

REVERSALS = {(0, 2), (2, 0), (1, 3), (3, 1)}  # up<->down, right<->left


def policy_entropy(trajectories: list[list[tuple]], n_actions: int) -> float:
    """Average entropy (bits) of the empirical action distribution, per visited state.

    nan if no state was ever visited (empty trajectories) -- not 0.0, to
    avoid conflating "measured and found to be zero" with "not measurable".
    """
    action_counts: dict[int, np.ndarray] = {}
    for episode in trajectories:
        for s, a, *_ in episode:
            action_counts.setdefault(s, np.zeros(n_actions))[a] += 1

    if not action_counts:
        return float("nan")
    entropies = []
    for counts in action_counts.values():
        p = counts / counts.sum()
        nz = p > 0
        entropies.append(-np.sum(p[nz] * np.log2(p[nz])))
    return float(np.mean(entropies))


def perseveration(trajectories: list[list[tuple]]) -> tuple[float, float]:
    """Rate of immediate action reversals (up/down, left/right) and fraction
    of steps that leave the agent on the same cell.

    Returns (reversal_rate, in_place_rate) -- kept separate rather than
    combined into a single index, these are two distinct forms of
    perseveration (hesitation vs. being stuck). nan (not 0.0) if the
    corresponding denominator is zero (no steps, or no pair of consecutive
    actions).
    """
    n_pairs = n_reversals = n_steps = n_in_place = 0
    for episode in trajectories:
        prev_a = None
        for s, a, r, s_next, terminated, truncated in episode:
            n_steps += 1
            if s_next == s:
                n_in_place += 1
            if prev_a is not None:
                n_pairs += 1
                if (prev_a, a) in REVERSALS:
                    n_reversals += 1
            prev_a = a
    reversal_rate = n_reversals / n_pairs if n_pairs else float("nan")
    in_place_rate = n_in_place / n_steps if n_steps else float("nan")
    return reversal_rate, in_place_rate


def mean_trap_distance(
    trajectories: list[list[tuple]], trap_states: list[int], grid_shape: tuple[int, int]
) -> tuple[float, float]:
    """Manhattan distance to the nearest trap cell.

    trap_states: state indices of the traps (not (row,col) positions --
    converted here via grid_shape, like any visited state). Returns
    (average over all steps, average of the minimum reached per episode).
    The second distinguishes an agent that genuinely avoids traps (minimum
    always >= 3) from one that skirts them without falling in (minimum
    close to 0 despite a high average) -- two behaviours that the average
    alone conflates. nan if no trap is given.
    """
    if not trap_states:
        return float("nan"), float("nan")
    n_cols = grid_shape[1]
    trap_rc = np.array([divmod(s, n_cols) for s in trap_states])

    all_dists = []
    per_episode_mins = []
    for episode in trajectories:
        episode_dists = []
        for s, a, r, s_next, terminated, truncated in episode:
            rc = np.array(divmod(s, n_cols))
            episode_dists.append(int(np.abs(trap_rc - rc).sum(axis=1).min()))
        if episode_dists:
            all_dists.extend(episode_dists)
            per_episode_mins.append(min(episode_dists))

    mean_dist = float(np.mean(all_dists)) if all_dists else float("nan")
    mean_min_dist = float(np.mean(per_episode_mins)) if per_episode_mins else float("nan")
    return mean_dist, mean_min_dist


def latency_to_goal(trajectories: list[list[tuple]]) -> tuple[float, float]:
    """Number of steps before first reaching a rewarded state (r > 0 at the
    step -- same convention as evaluate(), no dependency on a separately
    supplied set of goal states).

    Returns (mean latency over non-censored episodes, censoring rate). An
    episode is censored if it ends (trap or truncation) without ever going
    through a step with r > 0 -- reporting it separately prevents an
    artificially low mean latency from masking a majority of failures. Do
    not impute a value to censored episodes.
    """
    latencies = []
    n_censored = 0
    for episode in trajectories:
        latency = None
        for i, (s, a, r, s_next, terminated, truncated) in enumerate(episode, start=1):
            if r > 0:
                latency = i
                break
        if latency is None:
            n_censored += 1
        else:
            latencies.append(latency)
    n_episodes = len(trajectories)
    censoring_rate = n_censored / n_episodes if n_episodes else float("nan")
    mean_latency = float(np.mean(latencies)) if latencies else float("nan")
    return mean_latency, censoring_rate


def outcome_rates(trajectories: list[list[tuple]]) -> dict[str, float]:
    """Fraction of episodes ending in: goal reached, trap, truncation.

    The aggregate return (return_mean) conflates "dies" and "goes in
    circles", two very different phenotypes -- a return of -1.00 on this
    layout can be 100 * step_reward (pure truncation, no terminal state ever
    reached) or a quick trip into a trap; only this breakdown distinguishes
    them. The outcome of each episode is read from its last step: truncated
    -> "truncated", else terminated with r>0 -> "goal", else (terminated,
    r<=0) -> "trap".
    """
    if not trajectories:
        return {"goal": float("nan"), "trap": float("nan"), "truncated": float("nan")}
    counts = {"goal": 0, "trap": 0, "truncated": 0}
    for episode in trajectories:
        s, a, r, s_next, terminated, truncated = episode[-1]
        if truncated:
            counts["truncated"] += 1
        elif r > 0:
            counts["goal"] += 1
        else:
            counts["trap"] += 1
    n = len(trajectories)
    return {k: v / n for k, v in counts.items()}


def _detect_final_cycle(states: list[int], max_period: int = 20) -> int | None:
    """Smallest period k such that the last k states of the trajectory
    exactly equal the k states right before them. None if no proper period
    is detected up to max_period."""
    n = len(states)
    for period in range(1, max_period + 1):
        if n < 2 * period:
            break
        if states[-period:] == states[-2 * period : -period]:
            return period
    return None


def cycle_stats(trajectories: list[list[tuple]], max_period: int = 20) -> dict:
    """Terminal-cycle detection among TRUNCATED episodes only (an episode
    that reaches a goal/trap did not "go in circles", it ended).

    A period-2 cycle signals an in-place oscillation (up/down or left/right
    back-and-forth) -- on a 4-action grid, this is the only way to express
    "staying still" without a cell blocked on every side.

    Returns a dict with, as before (period 1 included -- a self-blocking
    against a wall is a period-1 in this aggregated view, not distinguished):
      n_truncated: number of truncated episodes (denominator)
      n_with_cycle: how many of them have a detected terminal cycle
      fraction_with_cycle: n_with_cycle / n_truncated (nan if n_truncated == 0)
      mean_cycle_length: average of the detected periods (nan if n_with_cycle == 0)
      cycle_length_counts: {period: number of episodes} over detected cycles

    Plus, explicitly distinguishing the two underlying behaviours (cf.
    evaluation protocol v2, where random start + random tie-break allow
    checking that the two are not conflated):
      n_wall_stuck / wall_stuck_rate: period == 1, the agent stays stuck at
        the same state (self-blocking action repeated, e.g. against a
        wall/border).
      n_genuine_cycle / cycle_rate / mean_genuine_cycle_length: period >= 2,
        oscillation between at least two distinct cells (by construction of
        _detect_final_cycle, which looks for the SMALLEST period: if period
        2 is returned, period 1 has already been tested and rejected, so
        the 2 states of the cycle are necessarily distinct).
    """
    n_truncated = 0
    periods = []
    for episode in trajectories:
        if not episode or not episode[-1][5]:  # index 5 = truncated
            continue
        n_truncated += 1
        states = [episode[0][0]] + [step[3] for step in episode]  # s_0, s_1, ..., s_T
        period = _detect_final_cycle(states, max_period=max_period)
        if period is not None:
            periods.append(period)

    cycle_length_counts: dict[int, int] = {}
    for period in periods:
        cycle_length_counts[period] = cycle_length_counts.get(period, 0) + 1

    n_wall_stuck = cycle_length_counts.get(1, 0)
    genuine_periods = [p for p in periods if p >= 2]

    return {
        "n_truncated": n_truncated,
        "n_with_cycle": len(periods),
        "fraction_with_cycle": (len(periods) / n_truncated) if n_truncated else float("nan"),
        "mean_cycle_length": (float(np.mean(periods)) if periods else float("nan")),
        "cycle_length_counts": cycle_length_counts,
        "n_wall_stuck": n_wall_stuck,
        "wall_stuck_rate": (n_wall_stuck / n_truncated) if n_truncated else float("nan"),
        "n_genuine_cycle": len(genuine_periods),
        "cycle_rate": (len(genuine_periods) / n_truncated) if n_truncated else float("nan"),
        "mean_genuine_cycle_length": (float(np.mean(genuine_periods)) if genuine_periods else float("nan")),
    }


def distinct_states_per_episode(
    trajectories: list[list[tuple]], n_reachable_states: int
) -> tuple[float, float]:
    """Average number of distinct states visited per episode, and the same
    normalised by the number of reachable states (cf. coverage()) -- an
    agent stuck in a period-2 cycle visits two, regardless of the layout.
    """
    if not trajectories:
        return float("nan"), float("nan")
    counts = []
    for episode in trajectories:
        if not episode:
            continue
        states = {episode[0][0]}
        states.update(step[3] for step in episode)
        counts.append(len(states))
    if not counts:
        return float("nan"), float("nan")
    mean_distinct = float(np.mean(counts))
    normalized = mean_distinct / n_reachable_states if n_reachable_states else float("nan")
    return mean_distinct, normalized
