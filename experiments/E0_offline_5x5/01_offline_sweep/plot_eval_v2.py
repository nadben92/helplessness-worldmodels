"""Comparison figure: outcome_rates by p, old protocol (fixed start +
deterministic argmax) vs. new (random start + random tie-break).

Reads results/offline_eval_v2_<budget>.csv (stratum_type="pooled" rows
only). Never plot inside an evaluation loop. Writes directly into reports/
(final figure for week1_offline_report.md).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from helplessness.config import resolve_path

OUTCOME_ORDER = ["goal", "trap", "truncated"]
OUTCOME_COLORS = {"goal": "#2ca02c", "trap": "#d62728", "truncated": "#7f7f7f"}
PROTOCOL_HATCHES = {"old": "", "new": "//"}
PROTOCOLS = {
    "old": ("fixed", "deterministic"),
    "new": ("random", "random"),
}


def _load(path: str):
    return list(csv.DictReader(open(path)))


def _pooled(rows, p: float, start_mode: str, tiebreak_mode: str):
    return [
        r for r in rows
        if r["stratum_type"] == "pooled" and float(r["p"]) == p
        and r["start_mode"] == start_mode and r["tiebreak_mode"] == tiebreak_mode
    ]


def _outcome_means(rows, p_values, start_mode: str, tiebreak_mode: str):
    means = {k: [] for k in OUTCOME_ORDER}
    for p in p_values:
        sub = _pooled(rows, p, start_mode, tiebreak_mode)
        for k in OUTCOME_ORDER:
            vals = np.array([float(r[f"frac_{k}"]) for r in sub])
            means[k].append(vals.mean())
    return {k: np.array(v) for k, v in means.items()}


def _plot_panel(ax, rows, p_values):
    x = np.arange(len(p_values))
    width = 0.35
    for i, (label, (start_mode, tiebreak_mode)) in enumerate(PROTOCOLS.items()):
        means = _outcome_means(rows, p_values, start_mode, tiebreak_mode)
        offset = (i - 0.5) * width
        bottom = np.zeros(len(p_values))
        for k in OUTCOME_ORDER:
            ax.bar(
                x + offset, means[k], width=width, bottom=bottom,
                color=OUTCOME_COLORS[k], hatch=PROTOCOL_HATCHES[label],
                edgecolor="white", linewidth=0.6,
            )
            bottom += means[k]
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in p_values])
    ax.set_xlabel("p")
    ax.set_ylabel("fraction of episodes")
    ax.grid(axis="y", alpha=0.3)


def plot_comparison(paths: dict[int, str], out_path: str = "reports/eval_v2_comparison.png") -> None:
    budgets = sorted(paths)
    fig, axes = plt.subplots(1, len(budgets), figsize=(7 * len(budgets) + 2, 6), squeeze=False)
    axes = axes[0]

    for ax, budget in zip(axes, budgets):
        rows = _load(paths[budget])
        p_values = sorted({float(r["p"]) for r in rows})
        _plot_panel(ax, rows, p_values)
        ax.set_title(f"budget={budget}")

    outcome_handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[k]) for k in OUTCOME_ORDER]
    protocol_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=PROTOCOL_HATCHES[label])
        for label in PROTOCOLS
    ]
    protocol_desc = {
        "old": "old = fixed start (S) + deterministic argmax",
        "new": "new = random start + random tie-break",
    }
    legend1 = fig.legend(outcome_handles, OUTCOME_ORDER, loc="upper left", bbox_to_anchor=(1.005, 0.95), title="outcome")
    fig.add_artist(legend1)
    fig.legend(
        protocol_handles, [protocol_desc[label] for label in PROTOCOLS],
        loc="upper left", bbox_to_anchor=(1.005, 0.55), title="protocol",
    )

    fig.suptitle("outcome_rates: old protocol vs. new (artefacts fixed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"figure written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares outcome_rates between the old and new evaluation protocols.")
    parser.add_argument(
        "--csv", action="append", required=True,
        help="budget:path (repeatable, e.g. --csv 5000:results/offline_eval_v2_5000.csv)",
    )
    parser.add_argument("--out", type=str, default="reports/eval_v2_comparison.png")
    args = parser.parse_args()

    paths = {}
    for item in args.csv:
        budget_str, path = item.split(":", 1)
        paths[int(budget_str)] = resolve_path(path)

    plot_comparison(paths, out_path=resolve_path(args.out))
