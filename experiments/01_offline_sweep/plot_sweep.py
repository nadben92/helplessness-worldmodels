"""Evolution curves of the offline sweep metrics as a function of p.

Reads the CSVs produced by run.py's sweep() -- never plot inside a training
loop (cf. project conventions). One figure per metric, one curve per budget
(mean over seeds, band = 95% CI over seeds, not a plain standard deviation).

Writes directly into reports/: this is one of the final figures used in
week1_offline_report.md, not an intermediate result (cf. target layout,
results/ vs reports/ in CONTRIBUTING.md).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from helplessness.config import resolve_path
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.measures.controllability import (
    controllability_from_tensor,
    value_gap,
    value_gap_second,
)
from helplessness.planning.value_iteration import GAMMA, value_iteration

try:
    from scipy.stats import t as _student_t

    def _t_crit(df: int) -> float:
        return float(_student_t.ppf(0.975, df))
except ImportError:  # fallback without scipy: normal approximation, correct for large n
    def _t_crit(df: int) -> float:
        return 1.96


METRICS = [
    ("perceived_controllability", "I(a;s'|s) under T_hat (bits)"),
    ("tv_visited", "TV(T_hat, T_true) on visited pairs"),
    ("policy_agreement_reachable", "policy agreement vs. oracle(p_train)"),
    ("value_gap_mean", "max_a Q - mean_a Q (averaged over s)"),
    ("value_gap_second_mean", "max_a Q - 2nd max_a Q (averaged over s)"),
]

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _load(path: str):
    rows = list(csv.DictReader(open(path)))
    by_p = defaultdict(list)
    for r in rows:
        by_p[float(r["p"])].append(r)
    return by_p


def _mean_ci95(by_p, field: str):
    """Mean and 95% CI half-width (based on Student's t, df = n_seeds - 1)."""
    p_values = sorted(by_p)
    means, halfwidths = [], []
    for p in p_values:
        vals = np.array([float(r[field]) for r in by_p[p]])
        n = len(vals)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        halfwidth = _t_crit(n - 1) * sem if n > 1 else 0.0
        means.append(mean)
        halfwidths.append(halfwidth)
    return np.array(p_values), np.array(means), np.array(halfwidths)


def _oracle_curve(p_values: np.ndarray):
    """perceived_controllability, value_gap_mean and value_gap_second_mean
    under the TRUE tensor (no seed, no budget -- a deterministic function of
    p alone), to spot which learned model deviates most from the reference.

    Q* is recomputed from V* (same pattern as compute_oracle() in
    offline_support.py: not a second value_iteration, just the final backup
    exposed).
    """
    controllabilities, value_gaps, value_gaps_second = [], [], []
    for p in p_values:
        env = GridWorld(p=float(p), seed=0)
        T_true = env.transition_tensor()
        R_true = env.reward_vector()
        terminal = env.terminal_mask()
        V_star, _ = value_iteration(T_true, R_true, terminal, gamma=GAMMA)
        Q_star = np.einsum("sap,p->sa", T_true, R_true + GAMMA * V_star * (~terminal))
        controllabilities.append(controllability_from_tensor(T_true, terminal))
        value_gaps.append(float(np.mean(value_gap(Q_star))))
        value_gaps_second.append(float(np.mean(value_gap_second(Q_star))))
    return np.array(controllabilities), np.array(value_gaps), np.array(value_gaps_second)


def plot_sweep(paths: dict[int, str], out_path: str = "reports/offline_sweep_curves.png") -> None:
    """paths: {budget: csv_path}. 2x3 figure (one metric per subplot), one
    curve per budget, overlaid on the same axes. For perceived_controllability
    and value_gap_mean, the oracle reference (true tensor) is plotted as a
    dashed line to spot each model's deviation."""
    datasets = {budget: _load(path) for budget, path in paths.items()}
    budgets = sorted(datasets)
    color_by_budget = dict(zip(budgets, COLORS))

    all_p = sorted({float(p) for by_p in datasets.values() for p in by_p})
    dense_p = np.linspace(min(all_p), max(all_p), 41)
    oracle_controllability, oracle_value_gap, oracle_value_gap_second = _oracle_curve(dense_p)
    oracle_by_field = {
        "perceived_controllability": oracle_controllability,
        "value_gap_mean": oracle_value_gap,
        "value_gap_second_mean": oracle_value_gap_second,
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax in axes.flat[len(METRICS):]:
        ax.set_visible(False)
    for ax, (field, label) in zip(axes.flat, METRICS):
        for budget in budgets:
            p_values, means, halfwidths = _mean_ci95(datasets[budget], field)
            color = color_by_budget[budget]
            ax.plot(p_values, means, marker="o", label=f"budget={budget}", color=color)
            ax.fill_between(p_values, means - halfwidths, means + halfwidths, alpha=0.2, color=color)
        if field in oracle_by_field:
            ax.plot(
                dense_p, oracle_by_field[field], linestyle="--", color="black",
                linewidth=1.5, label="oracle (T_true)", zorder=0,
            )
        ax.set_xlabel("p")
        ax.set_ylabel(label)
        ax.set_title(field)
        ax.legend()
        ax.grid(alpha=0.3)

    n_seeds = len(next(iter(datasets[budgets[0]].values())))
    fig.suptitle(f"Offline sweep -- mean +/- 95% CI over {n_seeds} seeds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"figure written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots the offline sweep curves.")
    parser.add_argument(
        "--csv", action="append", required=True,
        help="budget:path (repeatable, e.g. --csv 5000:results/offline_sweep_5000.csv)",
    )
    parser.add_argument("--out", type=str, default="reports/offline_sweep_curves.png")
    args = parser.parse_args()

    paths = {}
    for item in args.csv:
        budget_str, path = item.split(":", 1)
        paths[int(budget_str)] = resolve_path(path)

    plot_sweep(paths, out_path=resolve_path(args.out))
