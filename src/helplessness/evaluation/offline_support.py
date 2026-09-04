"""Shared plumbing for the offline pipeline (Protocol A), used across the
sweep, its reanalysis without retraining, and every downstream experiment
that loads saved agents (swaps, controllability dial, rescue injection).

Kept out of experiments/ because it is imported by multiple experiment
runners, not just one -- see the src/ vs experiments/ rule in CONTRIBUTING.md.

Renamed relative to the pre-restructure scripts (see restructuring commit
message for the full old -> new mapping):
  _agent_path        -> agent_path
  _write_sweep_readme -> write_sweep_readme
  FIELDNAMES          -> OFFLINE_METRICS_FIELDNAMES
"""

from __future__ import annotations

import os

import numpy as np

from helplessness.envs.gridworld import N_ACTIONS, GridWorld
from helplessness.evaluation.rollout import evaluate
from helplessness.metrics.behavioural import (
    cycle_stats,
    distinct_states_per_episode,
    latency_to_goal,
    mean_trap_distance,
    outcome_rates,
    perseveration,
    policy_entropy,
)
from helplessness.metrics.belief import coverage, model_error, perceived_controllability
from helplessness.metrics.controllability import value_gap, value_gap_second
from helplessness.planning.value_iteration import GAMMA, value_iteration

OFFLINE_METRICS_FIELDNAMES = [
    "p", "seed", "budget", "alpha",
    "perceived_controllability", "tv_visited", "tv_all", "frac_visited_reachable",
    "policy_agreement_reachable",
    "value_gap_mean", "value_gap_second_mean",
    "policy_entropy", "reversal_rate", "in_place_rate",
    "mean_trap_distance", "trap_min_distance_mean",
    "latency_mean", "latency_censoring_rate",
    "frac_goal", "frac_trap", "frac_truncated",
    "n_truncated", "n_with_cycle", "fraction_with_cycle", "mean_cycle_length",
    "distinct_states_mean", "distinct_states_normalized",
    "return_mean", "return_std", "success_rate", "length_mean", "truncation_rate",
    "oracle_return",
]


def compute_oracle(env: GridWorld):
    """True tensor, true reward, terminal mask, V*, Q* and the optimal policy.

    Q* is recomputed from V* (same formula value_iteration uses internally
    for its final backup): this is not a second implementation of value
    iteration, just the exposure of a quantity value_iteration already
    computes but does not return.
    """
    T_true = env.transition_tensor()
    R_true = env.reward_vector()
    terminal = env.terminal_mask()
    V_star, pi_star = value_iteration(T_true, R_true, terminal, gamma=GAMMA)
    Q_star = np.einsum("sap,p->sa", T_true, R_true + GAMMA * V_star * (~terminal))
    return T_true, R_true, terminal, V_star, Q_star, pi_star


def agent_path(agents_dir: str, budget: int, p: float, seed: int) -> str:
    return os.path.join(agents_dir, f"budget{budget}", f"p{p}_seed{seed}.npz")


def shared_context(ref_env: GridWorld):
    """Quantities shared across p/seed/budget: terminal mask (a layout
    property), R_true (independent of p), reference oracle return at p=0,
    trap states, grid shape, number of reachable states."""
    terminal = ref_env.terminal_mask()
    R_true = ref_env.reward_vector()
    _, _, _, _, _, pi_star_ref = compute_oracle(ref_env)
    oracle_summary, _ = evaluate(ref_env, pi_star_ref, n_episodes=2000, seed=0)
    oracle_return = oracle_summary["return_mean"]
    trap_states = [s for s in range(ref_env.n_states) if ref_env.traps[ref_env.s_to_rc(s)]]
    grid_shape = (ref_env.n_rows, ref_env.n_cols)
    n_reachable = int((~terminal).sum())
    return terminal, R_true, oracle_return, trap_states, grid_shape, n_reachable


def agent_metrics_row(
    agent,
    p: float,
    seed: int,
    budget: int,
    alpha: float,
    terminal: np.ndarray,
    R_true: np.ndarray,
    T_true_p: np.ndarray,
    oracle_return: float,
    trap_states: list[int],
    grid_shape: tuple[int, int],
    n_reachable: int,
    n_eval_episodes: int = 2000,
) -> dict:
    """Computes all metrics (belief + behavioural) for an agent that is
    ALREADY trained and planned (agent.V/Q/policy populated). Neither
    collects nor plans anything itself -- pure metrics computation, so it is
    usable both right after training (sweep) and after loading an agent from
    disk (reanalyze, without retraining)."""
    tv_visited, tv_all = model_error(agent.T_hat, T_true_p, agent.visited_mask)

    # policy agreement vs. the oracle under the true T at p_train (not p=0):
    # measures the quality of the learned model independently of the
    # transfer question, restricted to reachable states.
    V_star_p, pi_star_p = value_iteration(T_true_p, R_true, terminal, gamma=GAMMA)
    reachable = ~terminal
    policy_agreement = float(np.mean(agent.policy[reachable] == pi_star_p[reachable]))

    # evaluation: p=0, learning disabled, planner frozen
    eval_env = GridWorld(p=0.0, seed=seed)
    summary, traj = evaluate(eval_env, agent.policy, n_episodes=n_eval_episodes, seed=seed + 2_000_000)

    reversal_rate, in_place_rate = perseveration(traj)
    latency_mean, latency_censoring_rate = latency_to_goal(traj)
    outcomes = outcome_rates(traj)
    cstats = cycle_stats(traj)
    dist_mean, dist_norm = distinct_states_per_episode(traj, n_reachable)
    trap_mean, trap_min_mean = mean_trap_distance(traj, trap_states, grid_shape)

    return {
        "p": p,
        "seed": seed,
        "budget": budget,
        "alpha": alpha,
        "perceived_controllability": perceived_controllability(agent.T_hat, terminal),
        "tv_visited": tv_visited,
        "tv_all": tv_all,
        "frac_visited_reachable": coverage(agent.visited_mask, terminal),
        "policy_agreement_reachable": policy_agreement,
        "value_gap_mean": float(np.mean(value_gap(agent.Q))),
        "value_gap_second_mean": float(np.mean(value_gap_second(agent.Q))),
        "policy_entropy": policy_entropy(traj, N_ACTIONS),
        "reversal_rate": reversal_rate,
        "in_place_rate": in_place_rate,
        "mean_trap_distance": trap_mean,
        "trap_min_distance_mean": trap_min_mean,
        "latency_mean": latency_mean,
        "latency_censoring_rate": latency_censoring_rate,
        "frac_goal": outcomes["goal"],
        "frac_trap": outcomes["trap"],
        "frac_truncated": outcomes["truncated"],
        "n_truncated": cstats["n_truncated"],
        "n_with_cycle": cstats["n_with_cycle"],
        "fraction_with_cycle": cstats["fraction_with_cycle"],
        "mean_cycle_length": cstats["mean_cycle_length"],
        "distinct_states_mean": dist_mean,
        "distinct_states_normalized": dist_norm,
        "return_mean": summary["return_mean"],
        "return_std": summary["return_std"],
        "success_rate": summary["success_rate"],
        "length_mean": summary["length_mean"],
        "truncation_rate": summary["truncation_rate"],
        "oracle_return": oracle_return,
    }


def summary_from_trajectories(traj: list[list[tuple]]) -> dict:
    """Reproduces the summary computation of evaluate(), applicable to a
    subset (stratum) of already-collected trajectories -- no new rollout."""
    if not traj:
        return {
            "return_mean": float("nan"), "return_std": float("nan"),
            "success_rate": float("nan"), "length_mean": float("nan"),
            "truncation_rate": float("nan"),
        }
    returns, lengths, successes, truncations = [], [], 0, 0
    for episode in traj:
        returns.append(sum(step[2] for step in episode))
        lengths.append(len(episode))
        _s, _a, last_r, _sn, terminated, truncated = episode[-1]
        successes += int(terminated and last_r > 0)
        truncations += int(truncated)
    n = len(traj)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": successes / n,
        "length_mean": float(np.mean(lengths)),
        "truncation_rate": truncations / n,
    }


def behavioral_row(traj: list[list[tuple]], trap_states: list[int], grid_shape, n_reachable: int) -> dict:
    """Full belief-free behavioural row for an already-rolled-out set of
    trajectories -- shared by the eval-v2 reevaluation and by both
    intervention sweeps (dial and structure injection), so that the two
    rescue routes are scored by exactly the same metric set."""
    summary = summary_from_trajectories(traj)
    outcomes = outcome_rates(traj)
    cstats = cycle_stats(traj)
    dist_mean, dist_norm = distinct_states_per_episode(traj, n_reachable)
    trap_mean, trap_min_mean = mean_trap_distance(traj, trap_states, grid_shape)
    reversal_rate, in_place_rate = perseveration(traj)
    latency_mean, latency_censoring = latency_to_goal(traj)
    return {
        "n_episodes_in_stratum": len(traj),
        "return_mean": summary["return_mean"],
        "return_std": summary["return_std"],
        "success_rate": summary["success_rate"],
        "length_mean": summary["length_mean"],
        "truncation_rate": summary["truncation_rate"],
        "frac_goal": outcomes["goal"],
        "frac_trap": outcomes["trap"],
        "frac_truncated": outcomes["truncated"],
        "n_truncated": cstats["n_truncated"],
        "n_wall_stuck": cstats["n_wall_stuck"],
        "wall_stuck_rate": cstats["wall_stuck_rate"],
        "n_genuine_cycle": cstats["n_genuine_cycle"],
        "cycle_rate": cstats["cycle_rate"],
        "mean_genuine_cycle_length": cstats["mean_genuine_cycle_length"],
        "distinct_states_mean": dist_mean,
        "distinct_states_normalized": dist_norm,
        "trap_distance_mean": trap_mean,
        "trap_min_distance_mean": trap_min_mean,
        "reversal_rate": reversal_rate,
        "in_place_rate": in_place_rate,
        "latency_mean": latency_mean,
        "latency_censoring_rate": latency_censoring,
    }


def write_sweep_readme(out_path: str, budget: int, alpha: float, p_values, seeds) -> None:
    readme_path = out_path + ".README.txt"
    with open(readme_path, "w") as f:
        f.write(
            "Columns of this sweep -- definitions and state sets used.\n\n"
            f"budget={budget} transitions, alpha={alpha} (Dirichlet prior), "
            f"p in {list(p_values)}, seeds in {list(seeds)}.\n\n"
            "frac_visited_reachable: fraction of (s,a) pairs visited during\n"
            "  collection, among REACHABLE pairs (non-terminal states -- walls,\n"
            "  goals and terminal traps excluded from the denominator). 1.0 =\n"
            "  complete coverage of what is observable, regardless of layout.\n\n"
            "policy_agreement_reachable: fraction of REACHABLE (non-terminal)\n"
            "  states where the agent's greedy policy (under T_hat/R_hat)\n"
            "  matches the oracle policy computed on the TRUE tensor at\n"
            "  p_train (not p=0) -- measures the quality of the learned model,\n"
            "  independently of the transfer to p=0. Does not reach 1.0 even at\n"
            "  convergence: some states have tied actions in Q*, broken\n"
            "  arbitrarily by argmax (see validate() for an explicit check).\n\n"
            "value_gap_mean: max_a Q - mean_a Q, averaged over states.\n"
            "value_gap_second_mean: max_a Q - second max_a Q, averaged over states --\n"
            "  near 0 signals a near-tie between the two best actions\n"
            "  (arbitrary tie-break), regardless of the remaining actions.\n\n"
            "tv_visited / tv_all: average total-variation distance between T_hat and\n"
            "  the true tensor at p_train, respectively restricted to visited (s,a)\n"
            "  pairs and over all pairs (including unvisited ones, where T_hat is\n"
            "  only the Dirichlet prior -- tv_all is structurally higher and not\n"
            "  comparable across layouts).\n\n"
            "mean_trap_distance / trap_min_distance_mean: Manhattan distance to the\n"
            "  nearest trap, averaged over all steps, vs. average of the minimum\n"
            "  reached per episode -- distinguishes avoiding (minimum always large)\n"
            "  from skirting without falling in (minimum small despite a high\n"
            "  average).\n\n"
            "frac_goal / frac_trap / frac_truncated: outcome of each evaluation\n"
            "  episode (mutually exclusive, sums to 1). A return_mean of -1.00 on\n"
            "  this layout can be a pure truncation (frac_truncated=1, no terminal\n"
            "  state ever reached) or a quick fall into a trap (frac_trap=1) --\n"
            "  two very different phenotypes that the return alone conflates.\n\n"
            "n_truncated / n_with_cycle / fraction_with_cycle / mean_cycle_length:\n"
            "  cycle detection on the tail of TRUNCATED episodes only (an episode\n"
            "  that reaches a goal/trap does not \"go in circles\"). Period 2 =\n"
            "  in-place oscillation (the only way to express \"staying still\" with\n"
            "  4 actions). Explicit nan/0 if n_truncated=0 -- never an imputed value.\n\n"
            "distinct_states_mean / distinct_states_normalized: average number of\n"
            "  distinct states visited per episode, raw and normalised by the\n"
            "  number of reachable states (same convention as frac_visited_reachable).\n\n"
            "return_mean / success_rate / etc.: evaluated at p=0, learning\n"
            "  disabled, planner frozen (phase 2 of the protocol).\n\n"
            "oracle_return: mean return of the oracle policy at p=0 on this layout,\n"
            "  identical for every row -- a normalisation reference, not a\n"
            "  per-agent measure.\n"
        )
