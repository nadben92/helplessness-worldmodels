"""Sweeps the controllability dial (helplessness.interventions.controllability_dial)
over all saved agents, plus a reference curve on the true tensor. No
retraining: loads T_hat/R_hat from disk, applies the dial, replans
(value_iteration, not a retraining), evaluates under the fixed protocol
(random starts + random tie-break + fixed seeds, cf. experiments/01_offline_sweep/eval_v2.py)
-- the same shared seeds for EVERY agent and EVERY lam, so that the only
source of variation between rows is the dial itself.

Does not import helplessness.agents.collect: cannot retrain by construction.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import InterventionConfig, dump_config, load_config, resolve_path
from helplessness.envs.gridworld import GridWorld
from helplessness.evaluation.offline_support import agent_path, behavioral_row
from helplessness.evaluation.rollout import draw_start_states, evaluate
from helplessness.evaluation.tiebreak import argmax_random_tie
from helplessness.interventions.controllability_dial import controllability_dial, p_effective
from helplessness.metrics.belief import perceived_controllability
from helplessness.metrics.controllability import value_gap, value_gap_second
from helplessness.planning.value_iteration import GAMMA, value_iteration

FIELDNAMES = [
    "budget", "p_train", "seed", "lam", "p_effective", "clipped_fraction",
    "perceived_controllability", "value_gap_mean", "value_gap_second_mean",
    "return_mean", "return_std", "suboptimality", "success_rate", "length_mean", "truncation_rate",
    "frac_goal", "frac_trap", "frac_truncated",
    "n_truncated", "n_wall_stuck", "wall_stuck_rate",
    "n_genuine_cycle", "cycle_rate", "mean_genuine_cycle_length",
    "distinct_states_mean", "distinct_states_normalized",
    "trap_distance_mean", "trap_min_distance_mean",
    "reversal_rate", "in_place_rate",
    "latency_mean", "latency_censoring_rate",
    "oracle_return_ref",
]


def _eval_policy_from_Q(Q: np.ndarray, eval_tiebreak_seed: int) -> np.ndarray:
    """Evaluation policy: random tie-break, RNG reset to the SAME fixed
    seed on every call -- identical protocol for every (agent, lam)."""
    return argmax_random_tie(Q, np.random.default_rng(eval_tiebreak_seed))


def run_intervention(cfg: InterventionConfig) -> list[dict]:
    ref_env = GridWorld(p=0.0, seed=0)
    terminal = ref_env.terminal_mask()
    reachable_states = [s for s in range(ref_env.n_states) if not terminal[s]]
    trap_states = [s for s in range(ref_env.n_states) if ref_env.traps[ref_env.s_to_rc(s)]]
    grid_shape = (ref_env.n_rows, ref_env.n_cols)
    n_reachable = len(reachable_states)

    shared_starts = draw_start_states(reachable_states, cfg.n_episodes, seed=cfg.eval_start_seed)

    T_det = ref_env.transition_tensor(0.0)
    R_true = ref_env.reward_vector()

    # reference oracle UNDER THE SAME PROTOCOL (shared random starts): used
    # as the basis for suboptimality. Comparing a return_mean under random
    # starts against an oracle_return at a fixed start would confound the
    # protocol with the measured effect.
    V_star, pi_star = value_iteration(T_det, R_true, terminal, gamma=GAMMA)
    oracle_policy = _eval_policy_from_Q(
        np.einsum("sap,p->sa", T_det, R_true + GAMMA * V_star * (~terminal)),
        cfg.eval_tiebreak_seed,
    )
    eval_env_ref = GridWorld(p=0.0, seed=0)
    oracle_summary, _ = evaluate(eval_env_ref, oracle_policy, n_episodes=cfg.n_episodes, seed=0, start_states=shared_starts)
    oracle_return_ref = oracle_summary["return_mean"]

    resolved_out = resolve_path(cfg.out)
    resolved_agents_dir = resolve_path(cfg.agents_dir)

    all_rows = []
    os.makedirs(os.path.dirname(resolved_out) or ".", exist_ok=True)
    with open(resolved_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        def emit(budget, p_train, seed, T_hat, R, eval_seed):
            eval_env = GridWorld(p=0.0, seed=0)
            for lam in cfg.lam_values:
                T_lam, clipped_fraction = controllability_dial(T_hat, lam)
                V, _pi = value_iteration(T_lam, R, terminal, gamma=GAMMA)
                Q = np.einsum("sap,p->sa", T_lam, R + GAMMA * V * (~terminal))
                policy = _eval_policy_from_Q(Q, cfg.eval_tiebreak_seed)

                summary, traj = evaluate(
                    eval_env, policy, n_episodes=cfg.n_episodes, seed=eval_seed, start_states=shared_starts
                )
                behavior = behavioral_row(traj, trap_states, grid_shape, n_reachable)

                row = {
                    "budget": budget,
                    "p_train": p_train,
                    "seed": seed,
                    "lam": lam,
                    "p_effective": p_effective(p_train, lam),
                    "clipped_fraction": clipped_fraction,
                    "perceived_controllability": perceived_controllability(T_lam, terminal),
                    "value_gap_mean": float(np.mean(value_gap(Q))),
                    "value_gap_second_mean": float(np.mean(value_gap_second(Q))),
                    "return_mean": behavior["return_mean"],
                    "return_std": behavior["return_std"],
                    "suboptimality": oracle_return_ref - behavior["return_mean"],
                    "success_rate": behavior["success_rate"],
                    "length_mean": behavior["length_mean"],
                    "truncation_rate": behavior["truncation_rate"],
                    "frac_goal": behavior["frac_goal"],
                    "frac_trap": behavior["frac_trap"],
                    "frac_truncated": behavior["frac_truncated"],
                    "n_truncated": behavior["n_truncated"],
                    "n_wall_stuck": behavior["n_wall_stuck"],
                    "wall_stuck_rate": behavior["wall_stuck_rate"],
                    "n_genuine_cycle": behavior["n_genuine_cycle"],
                    "cycle_rate": behavior["cycle_rate"],
                    "mean_genuine_cycle_length": behavior["mean_genuine_cycle_length"],
                    "distinct_states_mean": behavior["distinct_states_mean"],
                    "distinct_states_normalized": behavior["distinct_states_normalized"],
                    "trap_distance_mean": behavior["trap_distance_mean"],
                    "trap_min_distance_mean": behavior["trap_min_distance_mean"],
                    "reversal_rate": behavior["reversal_rate"],
                    "in_place_rate": behavior["in_place_rate"],
                    "latency_mean": behavior["latency_mean"],
                    "latency_censoring_rate": behavior["latency_censoring_rate"],
                    "oracle_return_ref": oracle_return_ref,
                }
                writer.writerow(row)
                all_rows.append(row)

        # reference on the true tensor T(p_train=0) -- no seed, no real
        # budget: sentinel markers to distinguish it from agents.
        emit("oracle", 0.0, -1, T_det, R_true, eval_seed=0)

        for budget in cfg.budgets:
            for p_train in cfg.p_train_values:
                for seed in cfg.seeds:
                    path = agent_path(resolved_agents_dir, budget, p_train, seed)
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"missing agent: {path}. Rerun "
                            f"experiments/01_offline_sweep/run.py (budget={budget}) before the intervention."
                        )
                    agent = TabularModelBasedAgent.load(path)
                    emit(budget, p_train, seed, agent.T_hat, agent.R_hat, eval_seed=seed + 2_000_000)

    dump_config(cfg, resolved_out + ".config.yaml")
    print(f"wrote {len(all_rows)} rows to {resolved_out}")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweeps the controllability dial over all saved agents.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--agents-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(InterventionConfig, args.config)
    if args.budgets is not None:
        cfg.budgets = args.budgets
    if args.agents_dir is not None:
        cfg.agents_dir = args.agents_dir
    if args.out is not None:
        cfg.out = args.out
    if args.n_episodes is not None:
        cfg.n_episodes = args.n_episodes

    run_intervention(cfg)
