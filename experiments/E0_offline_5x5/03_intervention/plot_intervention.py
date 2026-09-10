"""Analyses of the controllability-dial intervention (results/intervention.csv).

Superposition first (the most important analysis, cf. run.py and project
guidelines): if the phenotypes, plotted against p_effective rather than lam,
align onto a single curve across all (budget, p_train, lam) combinations,
perceived controllability is a sufficient scalar summary of the entire
training effect. Quantified by a local fit (mean per bin of p_effective,
pooling all conditions): R^2 = fraction of variance explained by p_effective
alone, RMSE of the residuals around that fit.

Induction and rescue (produced only after checking the superposition, cf.
project guidelines) plot each phenotype against lam, without pooling: these
are different questions (does PUSHING lam reproduce a real p_train, not does
p_effective summarise everything).

Never plot inside an evaluation loop. Per-seed points shown alongside the
means -- a bimodal distribution would be hidden by a mean alone.

Figures are written directly into reports/ (final figures for
week1_offline_report.md).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from helplessness.config import resolve_path

PHENOTYPES = [
    ("return_mean", "mean return"),
    ("frac_goal", "fraction goal reached"),
    ("frac_trap", "fraction trap"),
    ("frac_truncated", "fraction truncated"),
    ("wall_stuck_rate", "wall-stuck rate"),
    ("cycle_rate", "genuine cycle rate"),
    ("distinct_states_mean", "distinct states / episode"),
    ("trap_distance_mean", "mean distance to trap"),
    ("perceived_controllability", "perceived controllability (bits)"),
    ("value_gap_mean", "value_gap (max-mean)"),
    ("value_gap_second_mean", "value_gap (max-2nd max)"),
    ("success_rate", "success rate"),
]


def _load(paths: list[str]) -> tuple[list[dict], list[dict]]:
    """Returns (learned_rows, oracle_rows) -- kept separate because the
    oracle has neither a real seed nor a real budget, it is not a condition
    to "collapse" with the others, it is the theoretical reference."""
    learned, oracle = [], []
    for path in paths:
        for r in csv.DictReader(open(path)):
            (oracle if r["budget"] == "oracle" else learned).append(r)
    return learned, oracle


def _values(rows: list[dict], field: str) -> tuple[np.ndarray, np.ndarray]:
    pe, y = [], []
    for r in rows:
        v = r[field]
        if v in ("nan", ""):
            continue
        pe.append(float(r["p_effective"]))
        y.append(float(v))
    return np.array(pe), np.array(y)


def collapse_quality(rows: list[dict], field: str, n_bins: int = 20) -> dict:
    """R^2 and RMSE of a local fit (mean per bin of p_effective) pooling all
    conditions -- quantifies the superposition rather than relying on the
    eye. Explicit nan if too few valid points."""
    pe, y = _values(rows, field)
    n = len(y)
    if n < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "n": n}

    edges = np.linspace(pe.min(), pe.max(), n_bins + 1)
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(pe, edges) - 1, 0, n_bins - 1)
    fit = np.zeros_like(y)
    for b in range(n_bins):
        m = bin_idx == b
        if m.any():
            fit[m] = y[m].mean()

    residual = y - fit
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {"r2": r2, "rmse": rmse, "n": n}


def print_collapse_table(rows: list[dict]) -> None:
    print(f"{'phenotype':<28}{'R2':<10}{'RMSE':<10}{'n':<8}")
    for field, _label in PHENOTYPES:
        stats = collapse_quality(rows, field)
        print(f"{field:<28}{stats['r2']:<10.3f}{stats['rmse']:<10.4f}{stats['n']:<8}")


def plot_superposition(
    learned: list[dict], oracle: list[dict], out_path: str = "reports/intervention_superposition.png"
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, (field, label) in zip(axes.flat, PHENOTYPES):
        pe, y = _values(learned, field)
        ax.scatter(pe, y, s=10, alpha=0.25, color="#1f77b4", label="agents (T_hat, all p_train/lam/seed)")

        # local fit (same bins as collapse_quality) to visualise the curve
        if len(y) >= 2:
            order = np.argsort(pe)
            n_bins = 20
            edges = np.linspace(pe.min(), pe.max(), n_bins + 1)
            centers, means = [], []
            for i in range(n_bins):
                m = (pe >= edges[i]) & (pe <= edges[i + 1] if i == n_bins - 1 else pe < edges[i + 1])
                if m.any():
                    centers.append((edges[i] + edges[i + 1]) / 2)
                    means.append(y[m].mean())
            ax.plot(centers, means, color="#1f77b4", linewidth=2)

        pe_o, y_o = _values(oracle, field)
        if len(y_o):
            order = np.argsort(pe_o)
            ax.plot(pe_o[order], y_o[order], color="black", linestyle="--", linewidth=1.5, label="true T(p) reference")

        stats = collapse_quality(learned, field)
        ax.set_title(f"{field}\nR2={stats['r2']:.3f} RMSE={stats['rmse']:.4f} (n={stats['n']})")
        ax.set_xlabel("p_effective")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Superposition: phenotypes vs. p_effective, all (budget, p_train, lam, seed) pooled")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"figure written to {out_path}")


BUDGET_COLORS = {5_000: "#1f77b4", 50_000: "#d62728"}


def _rows_at(rows: list[dict], p_train: float | None = None, lam: float | None = None):
    out = rows
    if p_train is not None:
        out = [r for r in out if float(r["p_train"]) == p_train]
    if lam is not None:
        out = [r for r in out if float(r["lam"]) == lam]
    return out


def _scatter_and_mean(ax, rows: list[dict], field: str, x_field: str, budgets: list[int], jitter: float = 0.008):
    """Per-seed points (slightly offset in x by budget) + mean line per
    budget -- never show only the mean (possible bimodality)."""
    for budget in budgets:
        sub = [r for r in rows if r["budget"] not in ("oracle",) and int(r["budget"]) == budget]
        by_x = defaultdict(list)
        for r in sub:
            v = r[field]
            if v in ("nan", ""):
                continue
            by_x[float(r[x_field])].append(float(v))
        xs = sorted(by_x)
        if not xs:
            continue
        offset = jitter if budget == max(budgets) else -jitter
        for x in xs:
            vals = by_x[x]
            ax.scatter(
                [x + offset] * len(vals), vals, s=14, alpha=0.35,
                color=BUDGET_COLORS.get(budget, "gray"), linewidths=0,
            )
        means = [np.mean(by_x[x]) for x in xs]
        ax.plot(
            [x + offset for x in xs], means, color=BUDGET_COLORS.get(budget, "gray"),
            marker="o", markersize=4, linewidth=1.8, label=f"budget={budget}",
        )


def plot_induction(learned: list[dict], out_path: str = "reports/intervention_induction.png") -> None:
    """Agents trained at p_train=0, lam swept upward: does raising lam
    reproduce the phenotype of agents actually trained at high p? Horizontal
    lines = intact value (lam=0) of agents actually trained at each
    reference p_train, averaged over budgets+seeds.

    lam<0 (clipped zone at ~72%, cf. superposition) is still shown but
    greyed out in the background so as not to suggest a clean extrapolation.
    """
    rows_p0 = _rows_at(learned, p_train=0.0)
    budgets = sorted({int(r["budget"]) for r in rows_p0})
    ref_p_trains = sorted({float(r["p_train"]) for r in learned if float(r["p_train"]) > 0.0})
    ref_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(ref_p_trains)))

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, (field, label) in zip(axes.flat, PHENOTYPES):
        ax.axvspan(-0.55, 0.0, color="gray", alpha=0.08, zorder=0)
        _scatter_and_mean(ax, rows_p0, field, "lam", budgets)

        for p_ref, color in zip(ref_p_trains, ref_colors):
            ref_rows = _rows_at(learned, p_train=p_ref, lam=0.0)
            vals = [float(r[field]) for r in ref_rows if r[field] not in ("nan", "")]
            if vals:
                ax.axhline(np.mean(vals), color=color, linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_title(field)
        ax.set_xlabel("lam (p_train=0)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    handles = [plt.Line2D([0], [0], color=BUDGET_COLORS[b], marker="o", label=f"budget={b}") for b in budgets]
    handles += [
        plt.Line2D([0], [0], color=c, linestyle="--", label=f"p_train={p} intact (reference)")
        for p, c in zip(ref_p_trains, ref_colors)
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 0.98), fontsize=8)
    fig.suptitle(
        "Induction: agents trained at p_train=0, lam swept upward "
        "(greyed zone lam<0 = tensor clipped at ~72%, not a clean extrapolation)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"figure written to {out_path}")


def plot_rescue(
    learned: list[dict], p_trains=(0.7, 0.85, 0.95), out_path: str = "reports/intervention_rescue.png"
) -> None:
    """Agents trained at high p, lam<0: does restoring the belief restore
    normal behaviour?

    The title reports both clipped_fraction AND clipped_mass measured on the
    T_hat ACTUALLY plotted here (not on T_det): on the true deterministic
    tensor, clipped_fraction is ~0.72 for any lam<0 (a threshold effect, cf.
    intervention_induction.png where the p_train=0 agent is nearly identical
    to T_det) -- but the T_hat learned at high p_train are already smoothed
    by training noise, with far fewer exact zeros, so far fewer rows flip
    negative. Reporting the T_det figure here would be wrong precisely in
    the regime this figure documents. clipped_mass (mean TV between the raw
    tensor and the clipped tensor) is reported in addition because
    clipped_fraction remains a threshold effect even here: it says nothing
    about the severity of the clip, only about its occurrence.
    """
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(p_trains)))

    from rescue_diagnostics import lam_policy_change_counts

    clip_rows = []
    for lam in (-0.5, -0.25):
        clip_rows += lam_policy_change_counts(list(p_trains), lam=lam)
    mean_cf = float(np.mean([r["clipped_fraction"] for r in clip_rows]))
    mean_cm = float(np.mean([r["clipped_mass"] for r in clip_rows]))
    per_p = {
        p: (
            np.mean([r["clipped_fraction"] for r in clip_rows if r["p_train"] == p]),
            np.mean([r["clipped_mass"] for r in clip_rows if r["p_train"] == p]),
        )
        for p in p_trains
    }
    per_p_str = ", ".join(f"p_train={p}: frac={cf:.3f} mass={cm:.4f}" for p, (cf, cm) in per_p.items())

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax, (field, label) in zip(axes.flat, PHENOTYPES):
        for p_train, color in zip(p_trains, colors):
            rows = [r for r in _rows_at(learned, p_train=p_train) if float(r["lam"]) <= 0.0]
            by_lam = defaultdict(list)
            for r in rows:
                v = r[field]
                if v in ("nan", ""):
                    continue
                by_lam[float(r["lam"])].append(float(v))
            lams = sorted(by_lam)
            if not lams:
                continue
            for lam in lams:
                vals = by_lam[lam]
                ax.scatter([lam] * len(vals), vals, s=12, alpha=0.3, color=color, linewidths=0)
            ax.plot(lams, [np.mean(by_lam[l]) for l in lams], color=color, marker="o", markersize=4, linewidth=1.8)
        ax.set_title(field)
        ax.set_xlabel("lam (<=0)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    fig.text(
        0.5, 0.955,
        f"clipped_fraction (measured on the T_hat plotted, lam in {{-0.5,-0.25}}) = {mean_cf:.3f} on average "
        f"(NOT 0.72: that figure belongs to T_det/p_train=0, not relevant here) -- "
        f"clipped_mass = {mean_cm:.4f} (raw->clipped TV, actual clip severity).",
        ha="center", fontsize=9, color="#8B0000",
    )
    fig.text(0.5, 0.935, per_p_str, ha="center", fontsize=8, color="#555555")

    handles = [plt.Line2D([0], [0], color=c, marker="o", label=f"p_train={p}") for p, c in zip(p_trains, colors)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 0.92), fontsize=8)
    fig.suptitle("Rescue: agents trained at high p, lam<0 (budgets pooled, points = seeds)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"figure written to {out_path}")


ROUTE_COLORS = {"lam": "#1f77b4", "injection": "#2ca02c"}
COMPARISON_ROWS = [
    ("return_mean", "mean return"),
    ("frac_goal", "fraction goal reached"),
    ("frac_truncated", "fraction truncated"),
    ("policy_change_frac", "fraction of states whose action changes"),
]


def plot_two_route_comparison(
    lam_rows: list[dict],
    lam_policy_rows: list[dict],
    injection_rows: list[dict],
    p_trains=(0.7, 0.85, 0.95, 1.0),
    out_path: str = "reports/rescue_two_routes.png",
) -> None:
    """Both rescue routes on the SAME agents, x = perceived controllability
    (bits) rather than lam/mu -- lam and mu are not on the same scale,
    perceived controllability is the only axis on which to compare them
    directly. Faceted by p_train. Points = seeds, line = mean.

    policy_change_frac = policy_change_count / n_reachable_states: the
    mechanistic evidence for Part 3 -- how much of the policy actually
    changed, not just how much the return moved.
    """
    # key (budget, p_train, seed, lam) -> policy_change_count for the lam
    # route (never stored in intervention.csv, computed separately by
    # rescue_diagnostics)
    lam_policy_lookup = {
        (r["budget"], float(r["p_train"]), r["seed"], float(r["lam"])): r
        for r in lam_policy_rows
    }

    fig, axes = plt.subplots(len(COMPARISON_ROWS), len(p_trains), figsize=(5 * len(p_trains), 4 * len(COMPARISON_ROWS)))

    for col, p_train in enumerate(p_trains):
        lam_sub = [r for r in lam_rows if r["budget"] != "oracle" and float(r["p_train"]) == p_train and float(r["lam"]) <= 0.0]
        inj_sub = [r for r in injection_rows if float(r["p_train"]) == p_train and float(r["mu"]) >= 0.0]

        for row_idx, (field, label) in enumerate(COMPARISON_ROWS):
            ax = axes[row_idx][col]

            for route, sub in (("lam", lam_sub), ("injection", inj_sub)):
                xs, ys = [], []
                for r in sub:
                    x = float(r["perceived_controllability"])
                    if field == "policy_change_frac":
                        if route == "lam":
                            key = (r["budget"], float(r["p_train"]), r["seed"], float(r["lam"]))
                            pr = lam_policy_lookup.get(key)
                            if pr is None:
                                continue
                            y = int(pr["policy_change_count"]) / int(pr["n_reachable_states"])
                        else:
                            y = int(r["policy_change_count"]) / int(r["n_reachable_states"])
                    else:
                        v = r[field]
                        if v in ("nan", ""):
                            continue
                        y = float(v)
                    xs.append(x)
                    ys.append(y)
                if not xs:
                    continue
                xs, ys = np.array(xs), np.array(ys)
                ax.scatter(xs, ys, s=14, alpha=0.3, color=ROUTE_COLORS[route], linewidths=0)
                order = np.argsort(xs)
                # mean per bin of perceived controllability for a readable line
                n_bins = 8
                edges = np.linspace(xs.min(), xs.max(), n_bins + 1) if xs.max() > xs.min() else np.array([xs.min(), xs.min() + 1e-6])
                centers, means = [], []
                for i in range(len(edges) - 1):
                    m = (xs >= edges[i]) & (xs <= edges[i + 1] if i == len(edges) - 2 else xs < edges[i + 1])
                    if m.any():
                        centers.append((edges[i] + edges[i + 1]) / 2)
                        means.append(ys[m].mean())
                ax.plot(centers, means, color=ROUTE_COLORS[route], linewidth=2, marker="o", markersize=4)

            if row_idx == 0:
                ax.set_title(f"p_train={p_train}")
            if col == 0:
                ax.set_ylabel(label)
            if row_idx == len(COMPARISON_ROWS) - 1:
                ax.set_xlabel("perceived controllability (bits)")
            ax.grid(alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=ROUTE_COLORS["lam"], marker="o", label="lam<0 (amplification, M dial)"),
        plt.Line2D([0], [0], color=ROUTE_COLORS["injection"], marker="o", label="mu>0 (structure injection, T_det)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=10)
    fig.suptitle(
        "Comparison of the two rescue routes, same agents, x = perceived controllability (not lam/mu)",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"figure written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intervention analyses (superposition, induction, rescue).")
    parser.add_argument("--csv", type=str, default="results/intervention.csv")
    parser.add_argument(
        "--stage",
        choices=["superposition", "induction", "rescue", "comparison", "all"],
        default="all",
    )
    args = parser.parse_args()

    learned, oracle = _load([resolve_path(args.csv)])

    if args.stage in ("superposition", "all"):
        print_collapse_table(learned)
        plot_superposition(learned, oracle, out_path=resolve_path("reports/intervention_superposition.png"))
    if args.stage in ("induction", "all"):
        plot_induction(learned, out_path=resolve_path("reports/intervention_induction.png"))
    if args.stage in ("rescue", "all"):
        plot_rescue(learned, out_path=resolve_path("reports/intervention_rescue.png"))
    if args.stage in ("comparison", "all"):
        lam_policy_rows = list(csv.DictReader(open(resolve_path("results/lam_policy_change_counts.csv"))))
        injection_rows = list(csv.DictReader(open(resolve_path("results/rescue_injection.csv"))))
        plot_two_route_comparison(
            learned, lam_policy_rows, injection_rows, out_path=resolve_path("reports/rescue_two_routes.png")
        )
