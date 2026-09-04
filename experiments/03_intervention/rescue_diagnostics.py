"""Mechanistic diagnostics comparing the two rescue routes (lam dial vs.
structure injection), not captured by intervention.csv alone: number of
states whose greedy action changes (lam route), and the actual severity of
the clip on the learned T_hat (not on T_det).

Reloads saved agents, replans (value_iteration, not a retraining). Does not
import helplessness.agents.collect.

The CLI writes results/lam_policy_change_counts.csv, read by
plot_intervention.py's plot_rescue() and plot_two_route_comparison(). This
CSV previously existed with no runner script that produced it (a
reproducibility gap this restructuring fixes) -- lam_values=(-0.5,-0.25,0.0)
below is chosen to match the pre-restructure file exactly.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import RescueDiagnosticsConfig, dump_config, load_config, resolve_path
from helplessness.envs.gridworld import GridWorld
from helplessness.evaluation.offline_support import agent_path
from helplessness.interventions.controllability_dial import clip_diagnostics, controllability_dial
from helplessness.planning.value_iteration import GAMMA, value_iteration

FIELDNAMES = [
    "budget", "p_train", "seed", "lam",
    "policy_change_count", "n_reachable_states",
    "clipped_fraction", "clipped_mass",
]


def _baseline_greedy(T_hat: np.ndarray, R_hat: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    V0, _ = value_iteration(T_hat, R_hat, terminal, gamma=GAMMA)
    Q0 = np.einsum("sap,p->sa", T_hat, R_hat + GAMMA * V0 * (~terminal))
    return Q0.argmax(axis=1)


def lam_policy_change_counts(
    p_train_values, lam: float, budgets=(5_000, 50_000), seeds=range(10), agents_dir: str = "results/agents"
) -> list[dict]:
    """For each saved agent, the number of reachable states whose greedy
    action (deterministic argmax of Q) differs between lam=0 (intact) and
    the given lam -- never computed in run.py, only inferred indirectly
    from returns identical to 4 decimal places."""
    ref_env = GridWorld(p=0.0, seed=0)
    terminal = ref_env.terminal_mask()
    reachable_mask = ~terminal
    resolved_agents_dir = resolve_path(agents_dir)

    rows = []
    for budget in budgets:
        for p_train in p_train_values:
            for seed in seeds:
                path = agent_path(resolved_agents_dir, budget, p_train, seed)
                agent = TabularModelBasedAgent.load(path)
                T_hat, R_hat = agent.T_hat, agent.R_hat
                baseline = _baseline_greedy(T_hat, R_hat, terminal)

                T_lam, clipped_fraction = controllability_dial(T_hat, lam)
                V, _ = value_iteration(T_lam, R_hat, terminal, gamma=GAMMA)
                Q = np.einsum("sap,p->sa", T_lam, R_hat + GAMMA * V * (~terminal))
                greedy = Q.argmax(axis=1)

                diag = clip_diagnostics(T_hat, lam)
                rows.append({
                    "budget": budget, "p_train": p_train, "seed": seed, "lam": lam,
                    "policy_change_count": int(np.sum((greedy != baseline) & reachable_mask)),
                    "n_reachable_states": int(reachable_mask.sum()),
                    "clipped_fraction": diag["clipped_fraction"],
                    "clipped_mass": diag["clipped_mass"],
                })
    return rows


def write_lam_policy_change_counts(cfg: RescueDiagnosticsConfig) -> list[dict]:
    all_rows = []
    for lam in cfg.lam_values:
        all_rows += lam_policy_change_counts(
            cfg.p_train_values, lam=lam, budgets=cfg.budgets, seeds=cfg.seeds, agents_dir=cfg.agents_dir
        )
    resolved_out = resolve_path(cfg.out)
    os.makedirs(os.path.dirname(resolved_out) or ".", exist_ok=True)
    with open(resolved_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    dump_config(cfg, resolved_out + ".config.yaml")
    print(f"wrote {len(all_rows)} rows to {resolved_out}")
    return all_rows


def _print_table(p_train_values, lam: float, rows: list[dict]) -> None:
    from collections import defaultdict

    by_p = defaultdict(list)
    for r in rows:
        by_p[r["p_train"]].append(r)

    print(f"lam route, lam={lam} (n_reachable_states={rows[0]['n_reachable_states']})")
    print(f"{'p_train':<10}{'mean policy_change':<20}{'max':<6}{'mean clipped_fraction':<24}{'mean clipped_mass':<20}")
    for p_train in sorted(by_p):
        rs = by_p[p_train]
        changes = [r["policy_change_count"] for r in rs]
        cf = [r["clipped_fraction"] for r in rs]
        cm = [r["clipped_mass"] for r in rs]
        print(
            f"{p_train:<10}{np.mean(changes):<20.2f}{max(changes):<6}"
            f"{np.mean(cf):<24.4f}{np.mean(cm):<20.4f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Writes results/lam_policy_change_counts.csv.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--p-train-values", type=float, nargs="+", default=None)
    parser.add_argument("--lam-values", type=float, nargs="+", default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(RescueDiagnosticsConfig, args.config)
    if args.p_train_values is not None:
        cfg.p_train_values = args.p_train_values
    if args.lam_values is not None:
        cfg.lam_values = args.lam_values
    if args.out is not None:
        cfg.out = args.out

    rows = write_lam_policy_change_counts(cfg)
    _print_table(cfg.p_train_values, cfg.lam_values[-1], [r for r in rows if r["lam"] == cfg.lam_values[-1]])
