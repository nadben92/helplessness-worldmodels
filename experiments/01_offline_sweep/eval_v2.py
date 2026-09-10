"""Evaluation protocol v2: fixes two artefacts of the original protocol
(fixed start + deterministic argmax), without retraining any agent.

This module does NOT import helplessness.agents.collect -- like reanalyze.py
and experiments/02_swaps/run.py, it therefore cannot retrain by
construction. The only operation performed on a loaded agent is
agent.plan() (deterministic value iteration on frozen T_hat/R_hat), never
observe().

The two artefacts:
  1. Fixed start (always S, top-left corner): the behavioural metrics only
     probe a handful of states, not representative of the policy.
  2. Deterministic tie-break (np.argmax): when several actions are tied
     (frequent at high p, cf. value_gap_second_mean near 0), np.argmax
     always returns the lowest index ("up"). Combined with a fixed start in
     the top-left corner (where "up" and "left" are both blocked by the
     border), this produces a lock-up at the start state from the very
     first step -- an indexing-order artefact, not a behavioural result.

Fixes, each available behind a flag to keep the old protocol reproducible:
  - start_mode="random": start drawn uniformly among reachable (non-terminal)
    states, the SAME sequence for all agents/conditions (dedicated RNG,
    independent of the training seed) -- cf. draw_start_states() in
    helplessness.evaluation.rollout. start_mode="fixed" keeps the historical
    behaviour (S).
  - tiebreak_mode="random": argmax_random_tie() instead of np.argmax to
    extract the evaluation policy from Q. Dedicated RNG, seeded identically
    for each agent (not the training seed). tiebreak_mode="deterministic"
    keeps np.argmax.

policy_entropy is removed from this file rather than fixed: under a greedy
policy (either of the two above), each visited state has only one possible
action, entropy is zero by construction -- this is not an artefact of the
evaluation protocol, it is a direct consequence of evaluating a
deterministic policy. Computing it correctly would require a stochastic
policy (softmax(Q/T)), an ADDITIONAL EVALUATION MODE with its own
temperature hyperparameter -- outside the scope of this task (which fixes
the existing protocol, does not add a third one). Staying nan/absent rather
than silently wrong.

Stratification (random start only -- with a single fixed start,
stratification would be trivial and would duplicate the "pooled" row): in
addition to the pooled row (all strata combined), one row per value of
Manhattan distance to the goal, and one row per value of Manhattan distance
to the nearest trap from the episode's start.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
import csv

import numpy as np

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import EvalV2Config, dump_config, load_config, resolve_path
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.evaluation.offline_support import agent_path, behavioral_row
from helplessness.evaluation.rollout import draw_start_states, evaluate
from helplessness.evaluation.tiebreak import argmax_random_tie
from helplessness.planning.value_iteration import GAMMA

CONDITIONS = [
    ("fixed", "deterministic"),  # the old protocol, unchanged
    ("fixed", "random"),
    ("random", "deterministic"),
    ("random", "random"),  # the final v2 protocol
]

FIELDNAMES = [
    "p", "seed", "budget", "alpha", "start_mode", "tiebreak_mode",
    "stratum_type", "stratum_value", "n_episodes_in_stratum",
    "return_mean", "return_std", "success_rate", "length_mean", "truncation_rate",
    "frac_goal", "frac_trap", "frac_truncated",
    "n_truncated", "n_wall_stuck", "wall_stuck_rate",
    "n_genuine_cycle", "cycle_rate", "mean_genuine_cycle_length",
    "distinct_states_mean", "distinct_states_normalized",
    "trap_distance_mean", "trap_min_distance_mean",
    "reversal_rate", "in_place_rate",
    "latency_mean", "latency_censoring_rate",
]


def _manhattan(rc1, rc2) -> int:
    return abs(rc1[0] - rc2[0]) + abs(rc1[1] - rc2[1])


def reevaluate(cfg: EvalV2Config) -> list[dict]:
    ref_env = GridWorld(p=0.0, seed=0)
    terminal = ref_env.terminal_mask()
    reachable_states = [s for s in range(ref_env.n_states) if not terminal[s]]
    trap_states = [s for s in range(ref_env.n_states) if ref_env.traps[ref_env.s_to_rc(s)]]
    goal_state = next(s for s in range(ref_env.n_states) if ref_env.goals[ref_env.s_to_rc(s)])
    grid_shape = (ref_env.n_rows, ref_env.n_cols)
    n_reachable = len(reachable_states)

    goal_rc = ref_env.s_to_rc(goal_state)
    trap_rcs = [ref_env.s_to_rc(t) for t in trap_states]

    # SAME sequence of starts for all agents and all conditions
    shared_starts = draw_start_states(reachable_states, cfg.n_episodes, seed=cfg.eval_start_seed)

    resolved_out = resolve_path(cfg.out)
    resolved_agents_dir = resolve_path(cfg.agents_dir)

    all_rows = []
    os.makedirs(os.path.dirname(resolved_out) or ".", exist_ok=True)
    with open(resolved_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for budget in cfg.budgets:
            for p in cfg.p_values:
                for seed in cfg.seeds:
                    path = agent_path(resolved_agents_dir, budget, p, seed)
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"missing agent: {path}. Rerun "
                            f"run.py sweep(budget={budget}) before reevaluating."
                        )
                    agent = TabularModelBasedAgent.load(path)
                    agent.plan(terminal, gamma=GAMMA)  # replans, does not retrain

                    deterministic_policy = agent.Q.argmax(axis=1)
                    random_tie_policy = argmax_random_tie(
                        agent.Q, np.random.default_rng(cfg.eval_tiebreak_seed)
                    )

                    eval_env = GridWorld(p=0.0, seed=seed)

                    for start_mode, tiebreak_mode in CONDITIONS:
                        policy = deterministic_policy if tiebreak_mode == "deterministic" else random_tie_policy
                        start_states = shared_starts if start_mode == "random" else None

                        _summary, traj = evaluate(
                            eval_env, policy, n_episodes=cfg.n_episodes,
                            seed=seed + 2_000_000, start_states=start_states,
                        )

                        base = {
                            "p": p, "seed": seed, "budget": budget, "alpha": agent.alpha,
                            "start_mode": start_mode, "tiebreak_mode": tiebreak_mode,
                        }

                        pooled = {**base, "stratum_type": "pooled", "stratum_value": float("nan"),
                                  **behavioral_row(traj, trap_states, grid_shape, n_reachable)}
                        writer.writerow(pooled)
                        all_rows.append(pooled)

                        if start_mode == "random":
                            groups_goal = defaultdict(list)
                            groups_trap = defaultdict(list)
                            for episode in traj:
                                start_rc = ref_env.s_to_rc(episode[0][0])
                                groups_goal[_manhattan(start_rc, goal_rc)].append(episode)
                                groups_trap[min(_manhattan(start_rc, trc) for trc in trap_rcs)].append(episode)

                            for dist, sub in sorted(groups_goal.items()):
                                row = {**base, "stratum_type": "dist_to_goal", "stratum_value": dist,
                                       **behavioral_row(sub, trap_states, grid_shape, n_reachable)}
                                writer.writerow(row)
                                all_rows.append(row)

                            for dist, sub in sorted(groups_trap.items()):
                                row = {**base, "stratum_type": "dist_to_trap", "stratum_value": dist,
                                       **behavioral_row(sub, trap_states, grid_shape, n_reachable)}
                                writer.writerow(row)
                                all_rows.append(row)

    dump_config(cfg, resolved_out + ".config.yaml")
    print(f"wrote {len(all_rows)} rows to {resolved_out}")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reevaluates all saved agents under the v2 protocol.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--agents-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(EvalV2Config, args.config)
    if args.budgets is not None:
        cfg.budgets = args.budgets
    if args.agents_dir is not None:
        cfg.agents_dir = args.agents_dir
    if args.out is not None:
        cfg.out = args.out
    elif args.config is None and args.budgets is not None and len(args.budgets) == 1:
        cfg.out = f"results/offline_eval_v2_{args.budgets[0]}.csv"
    if args.n_episodes is not None:
        cfg.n_episodes = args.n_episodes

    reevaluate(cfg)
