"""Runs the swap matrix (Phase 3 -- aetiology, cf.
helplessness.interventions.swaps) for each (low_p, high_p) pair, over the 10
seeds, using agents already trained at budget=50k (experiments/01_offline_sweep/run.py).
Trains nothing: loads agents saved to disk, like reanalyze.py -- this module
also does not import helplessness.agents.collect.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import t as _student_t

    def _t_crit(df: int) -> float:
        return float(_student_t.ppf(0.975, df)) if df > 0 else float("nan")
except ImportError:  # fallback without scipy
    def _t_crit(df: int) -> float:
        return 1.96


from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import SwapsConfig, dump_config, load_config, resolve_path
from helplessness.envs.gridworld import N_ACTIONS, GridWorld
from helplessness.evaluation.offline_support import agent_path
from helplessness.evaluation.rollout import evaluate
from helplessness.interventions.swaps import CONDITIONS, swap_matrix
from helplessness.metrics.behavioural import (
    cycle_stats,
    distinct_states_per_episode,
    latency_to_goal,
    mean_trap_distance,
    outcome_rates,
    perseveration,
    policy_entropy,
)

FIELDNAMES = [
    "low_p", "high_p", "seed", "condition",
    "return_mean", "return_std", "success_rate",
    "frac_goal", "frac_trap", "frac_truncated",
    "n_truncated", "n_with_cycle", "fraction_with_cycle", "mean_cycle_length",
    "distinct_states_mean", "distinct_states_normalized",
    "trap_distance_mean", "trap_min_distance_mean",
    "reversal_rate", "in_place_rate",
    "policy_entropy",
    "latency_mean", "latency_censoring_rate",
]


def run_swaps(cfg: SwapsConfig) -> list[dict]:
    ref_env = GridWorld(p=0.0, seed=0)
    T_true = ref_env.transition_tensor()
    R_true = ref_env.reward_vector()
    terminal = ref_env.terminal_mask()  # TEST env (p=0), not the donor
    trap_states = [s for s in range(ref_env.n_states) if ref_env.traps[ref_env.s_to_rc(s)]]
    grid_shape = (ref_env.n_rows, ref_env.n_cols)
    n_reachable = int((~terminal).sum())

    all_rows = []

    resolved_out = resolve_path(cfg.out)
    resolved_agents_dir = resolve_path(cfg.agents_dir)
    os.makedirs(os.path.dirname(resolved_out) or ".", exist_ok=True)
    with open(resolved_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for high_p in cfg.high_ps:
            for seed in cfg.seeds:
                path_low = agent_path(resolved_agents_dir, cfg.budget, cfg.low_p, seed)
                path_high = agent_path(resolved_agents_dir, cfg.budget, high_p, seed)
                for path in (path_low, path_high):
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"missing agent: {path}. Rerun "
                            f"experiments/01_offline_sweep/run.py (budget={cfg.budget}) to generate "
                            "the saved agents before running the swaps."
                        )
                agent_low = TabularModelBasedAgent.load(path_low)
                agent_high = TabularModelBasedAgent.load(path_high)

                planned = swap_matrix(agent_low, agent_high, T_true, R_true, terminal, gamma=cfg.gamma)

                for condition in CONDITIONS:
                    _V, _Q, policy = planned[condition]
                    # same evaluation seed, hence the same start states and
                    # the same stochastic stream, for ALL conditions of this
                    # (high_p, seed) -- seed fixed per seed, not per condition.
                    eval_env = GridWorld(p=0.0, seed=seed)
                    summary, traj = evaluate(eval_env, policy, n_episodes=cfg.n_episodes, seed=seed)

                    outcomes = outcome_rates(traj)
                    cstats = cycle_stats(traj)
                    dist_mean, dist_norm = distinct_states_per_episode(traj, n_reachable)
                    trap_mean, trap_min_mean = mean_trap_distance(traj, trap_states, grid_shape)
                    reversal_rate, in_place_rate = perseveration(traj)
                    latency_mean, latency_censoring = latency_to_goal(traj)

                    row = {
                        "low_p": cfg.low_p,
                        "high_p": high_p,
                        "seed": seed,
                        "condition": condition,
                        "return_mean": summary["return_mean"],
                        "return_std": summary["return_std"],
                        "success_rate": summary["success_rate"],
                        "frac_goal": outcomes["goal"],
                        "frac_trap": outcomes["trap"],
                        "frac_truncated": outcomes["truncated"],
                        "n_truncated": cstats["n_truncated"],
                        "n_with_cycle": cstats["n_with_cycle"],
                        "fraction_with_cycle": cstats["fraction_with_cycle"],
                        "mean_cycle_length": cstats["mean_cycle_length"],
                        "distinct_states_mean": dist_mean,
                        "distinct_states_normalized": dist_norm,
                        "trap_distance_mean": trap_mean,
                        "trap_min_distance_mean": trap_min_mean,
                        "reversal_rate": reversal_rate,
                        "in_place_rate": in_place_rate,
                        "policy_entropy": policy_entropy(traj, N_ACTIONS),
                        "latency_mean": latency_mean,
                        "latency_censoring_rate": latency_censoring,
                    }
                    writer.writerow(row)
                    all_rows.append(row)

    dump_config(cfg, resolved_out + ".config.yaml")
    print(f"wrote {len(all_rows)} rows to {resolved_out}\n")
    _print_summary(all_rows, cfg.low_p, cfg.high_ps, cfg.seeds)
    return all_rows


def _print_summary(rows: list[dict], low_p: float, high_ps, seeds) -> None:
    df = len(seeds) - 1
    for high_p in high_ps:
        print(f"=== low_p={low_p} high_p={high_p} : mean return per condition (95% CI, n={len(seeds)} seeds) ===")
        by_cond = defaultdict(list)
        for r in rows:
            if r["high_p"] == high_p:
                by_cond[r["condition"]].append(r["return_mean"])
        print(f"{'condition':<18}{'mean return':<16}{'95% CI':<10}")
        for cond in CONDITIONS:
            vals = np.array(by_cond[cond])
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            halfwidth = _t_crit(df) * sem
            print(f"{cond:<18}{mean:<+16.4f}+/-{halfwidth:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component swap matrix (Phase 3).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--low-p", type=float, default=None)
    parser.add_argument("--high-ps", type=float, nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--agents-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(SwapsConfig, args.config)
    if args.low_p is not None:
        cfg.low_p = args.low_p
    if args.high_ps is not None:
        cfg.high_ps = args.high_ps
    if args.budget is not None:
        cfg.budget = args.budget
    if args.agents_dir is not None:
        cfg.agents_dir = args.agents_dir
    if args.out is not None:
        cfg.out = args.out

    run_swaps(cfg)
