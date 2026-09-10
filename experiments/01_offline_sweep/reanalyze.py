"""Regenerates results/offline_sweep_<budget>.csv from agents already
trained and saved to disk (by run.py's sweep()), without recollecting any
transition.

This module does NOT import helplessness.agents.collect -- deliberately:
this makes it impossible to (re)train anything by accident, not merely a
docstring promise. The only "learning" operation here is agent.plan(), i.e.
re-running value_iteration on an already-frozen T_hat/R_hat: this is not a
retraining, just replanning on a frozen belief.
"""

from __future__ import annotations

import argparse
import csv
import os

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import OfflineSweepConfig, dump_config, resolve_path
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.evaluation.offline_support import (
    OFFLINE_METRICS_FIELDNAMES,
    agent_metrics_row,
    agent_path,
    shared_context,
    write_sweep_readme,
)
from helplessness.planning.value_iteration import GAMMA


def reanalyze(
    budget: int,
    p_values=(0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0),
    seeds=range(10),
    agents_dir: str = "results/agents",
    out_path: str | None = None,
    alpha: float = 0.1,
) -> None:
    out_path = out_path or f"results/offline_sweep_{budget}.csv"
    resolved_out_path = resolve_path(out_path)
    resolved_agents_dir = resolve_path(agents_dir)
    os.makedirs(os.path.dirname(resolved_out_path) or ".", exist_ok=True)

    ref_env = GridWorld(p=0.0, seed=0)
    terminal, R_true, oracle_return, trap_states, grid_shape, n_reachable = shared_context(ref_env)

    seeds = list(seeds)
    n_rows = 0
    with open(resolved_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OFFLINE_METRICS_FIELDNAMES)
        writer.writeheader()

        for p in p_values:
            T_true_p = GridWorld(p=p, seed=0).transition_tensor()
            for seed in seeds:
                path = agent_path(resolved_agents_dir, budget, p, seed)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"missing agent: {path}. Rerun run.py sweep(budget={budget}) "
                        "to (re)generate the saved agents before reanalyzing."
                    )
                agent = TabularModelBasedAgent.load(path)
                agent.plan(terminal, gamma=GAMMA)  # replans on the frozen T_hat/R_hat, collects nothing

                row = agent_metrics_row(
                    agent, p, seed, budget, alpha, terminal, R_true, T_true_p,
                    oracle_return, trap_states, grid_shape, n_reachable,
                )
                writer.writerow(row)
                n_rows += 1

    write_sweep_readme(resolved_out_path, budget, alpha, p_values, seeds)
    dump_config(
        OfflineSweepConfig(
            budget=budget, p_values=list(p_values), seeds=seeds, alpha=alpha,
            agents_dir=agents_dir, out=out_path,
        ),
        resolved_out_path + ".config.yaml",
    )
    print(f"reanalyzed (no retraining): {n_rows} rows written to {resolved_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reanalyzes the offline sweep from saved agents.")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--agents-dir", type=str, default="results/agents")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    reanalyze(budget=args.budget, agents_dir=args.agents_dir, out_path=args.out)
