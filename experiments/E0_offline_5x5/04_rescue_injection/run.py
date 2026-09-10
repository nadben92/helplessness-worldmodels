"""Sweeps structure injection (helplessness.interventions.controllability_dial.structure_injection)
over agents trained at high p_train, complementing the lam dial already
explored by experiments/03_intervention/run.py. Same frozen protocol
(random starts + random tie-break + fixed seeds shared across agents/mu, cf.
experiments/01_offline_sweep/eval_v2.py). No retraining: loads T_hat/R_hat
from disk, applies the injection, replans (value_iteration, not a
retraining), evaluates.

Does not import helplessness.agents.collect: cannot retrain by construction.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import RescueInjectionConfig, dump_config, load_config, resolve_path
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.evaluation.offline_support import agent_path, behavioral_row
from helplessness.evaluation.rollout import draw_start_states, evaluate
from helplessness.evaluation.tiebreak import argmax_random_tie
from helplessness.interventions.controllability_dial import structure_injection
from helplessness.measures.belief import perceived_controllability
from helplessness.measures.controllability import value_gap, value_gap_second
from helplessness.planning.value_iteration import GAMMA, value_iteration

FIELDNAMES = [
    "budget", "p_train", "seed", "mu", "p_effective_equiv",
    "policy_change_count", "n_reachable_states",
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
    """Same evaluation protocol as experiments/03_intervention/run.py:
    random tie-break, RNG reset to the same fixed seed on every call."""
    return argmax_random_tie(Q, np.random.default_rng(eval_tiebreak_seed))


def run_rescue_injection(cfg: RescueInjectionConfig) -> list[dict]:
    ref_env = GridWorld(p=0.0, seed=0)
    terminal = ref_env.terminal_mask()
    reachable_states = [s for s in range(ref_env.n_states) if not terminal[s]]
    reachable_mask = ~terminal
    trap_states = [s for s in range(ref_env.n_states) if ref_env.traps[ref_env.s_to_rc(s)]]
    grid_shape = (ref_env.n_rows, ref_env.n_cols)
    n_reachable = len(reachable_states)

    shared_starts = draw_start_states(reachable_states, cfg.n_episodes, seed=cfg.eval_start_seed)

    T_det = ref_env.transition_tensor(0.0)
    R_true = ref_env.reward_vector()

    # same reference oracle as experiments/03_intervention/run.py, so that
    # suboptimality is directly comparable between the two routes.
    V_star, _pi = value_iteration(T_det, R_true, terminal, gamma=GAMMA)
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

        for budget in cfg.budgets:
            for p_train in cfg.p_train_values:
                for seed in cfg.seeds:
                    path = agent_path(resolved_agents_dir, budget, p_train, seed)
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"missing agent: {path}. Rerun "
                            f"experiments/01_offline_sweep/run.py (budget={budget}) before the injection."
                        )
                    agent = TabularModelBasedAgent.load(path)
                    T_hat = agent.T_hat
                    R_hat = agent.R_hat

                    # mu=0 reference policy, DETERMINISTIC argmax (not the
                    # evaluation tie-break): this is the table compared
                    # against every mu to count action changes -- the
                    # requested mechanistic diagnostic, not a return measure.
                    V0, _pi0 = value_iteration(T_hat, R_hat, terminal, gamma=GAMMA)
                    Q0 = np.einsum("sap,p->sa", T_hat, R_hat + GAMMA * V0 * (~terminal))
                    baseline_greedy = Q0.argmax(axis=1)

                    eval_env = GridWorld(p=0.0, seed=0)
                    eval_seed = seed + 2_000_000

                    for mu in cfg.mu_values:
                        T_mu = structure_injection(T_hat, T_det, mu)
                        V, _pi = value_iteration(T_mu, R_hat, terminal, gamma=GAMMA)
                        Q = np.einsum("sap,p->sa", T_mu, R_hat + GAMMA * V * (~terminal))

                        greedy_mu = Q.argmax(axis=1)
                        policy_change_count = int(np.sum((greedy_mu != baseline_greedy) & reachable_mask))

                        policy = _eval_policy_from_Q(Q, cfg.eval_tiebreak_seed)
                        summary, traj = evaluate(
                            eval_env, policy, n_episodes=cfg.n_episodes, seed=eval_seed, start_states=shared_starts
                        )
                        behavior = behavioral_row(traj, trap_states, grid_shape, n_reachable)

                        row = {
                            "budget": budget,
                            "p_train": p_train,
                            "seed": seed,
                            "mu": mu,
                            # p_effective_equiv: not a genuine p_effective (structure_injection
                            # does not follow the T(p) = (1-p)T_det + p*M geometry, unlike
                            # the lam dial) -- just mu itself, reported to place the axis
                            # on the comparison figures (cf. plot_intervention.py).
                            "p_effective_equiv": mu,
                            "policy_change_count": policy_change_count,
                            "n_reachable_states": n_reachable,
                            "perceived_controllability": perceived_controllability(T_mu, terminal),
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

    dump_config(cfg, resolved_out + ".config.yaml")
    print(f"wrote {len(all_rows)} rows to {resolved_out}")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweeps structure injection over agents trained at high p_train.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--p-train-values", type=float, nargs="+", default=None)
    parser.add_argument("--agents-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(RescueInjectionConfig, args.config)
    if args.budgets is not None:
        cfg.budgets = args.budgets
    if args.p_train_values is not None:
        cfg.p_train_values = args.p_train_values
    if args.agents_dir is not None:
        cfg.agents_dir = args.agents_dir
    if args.out is not None:
        cfg.out = args.out
    if args.n_episodes is not None:
        cfg.n_episodes = args.n_episodes

    run_rescue_injection(cfg)
