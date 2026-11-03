"""E5f_swap_gradient (docs/e5/DECISIONS.md (38)): b at step 1 against t along E4's three axes in the swapped slot, from
analyse_swap.py's tables (curves.csv, points.csv), read-only. The style of the E5 slide figures (slides/figures_e5/style_e5.py
on branch e6: the v2 chart, axis names 18 pt, ticks 15 pt, no title, legend or annotation, reference lines dashed without
text), its constants copied here: E5 imports no other branch.

Layout: one column per axis, the primary ones first (arc_ctrl, arc_inc, then the secondary segment, (38)), x = t = k / 10. Top row: the shuffle agents (A, E, F); bottom
row: the baselines A and B at --baseline-k (qualitative, (38)). Per agent and class, the mean over the swaps of its kept
zones (the classes of analyse_swap.py):
  moved slot    the condition's colour (shuffle violet, baseline vermilion), solid line
  neutral       grey, solid line
  old UD (nw)   the UD zone colour, dash-dot line (a witness, (38))
Markers: the agent (A circle, E square, F triangle; baselines A circle, B square). A filled marker counts for the reading
rule (kept in at least 2 of the 3 swaps), a hollow one does not; a cross on the bottom edge marks
a t where NO agent of the row has a kept moved slot (invalid or badly localised). Shaded: the near-duplicate points (segment t >=
0.8, arc_inc t <= 0.2). Dashed: 0 (grey) and -log2 5 (black).

Usage: python experiments/E5_belief_training/analysis_swap/fig_swap_gradient.py --tables <analyse_swap tables>
           --out-dir <slides/out_e5/png_white of the e6 worktree> [--baseline-k 300000]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# style_e5.py (branch e6), copied
EXPOSED, SHUFFLE, THEORY, CHANCE = "#E8412C", "#A21CAF", "#000000", "#888888"
ZONE_LINE = {"UD": "#FF5A36", "CD": "#4DA3FF", "corridor": "#6E6E6E"}
LW, MS, SIZE_LABEL, SIZE_TICK = 2.8, 9, 18, 15
FULL = (13.33, 7.5)
LOG2_5 = 2.321928094887362

AXES = ("arc_ctrl", "arc_inc", "segment")  # primary, primary, secondary ((38))
MARKERS = {"A": "o", "E": "s", "F": "^", "B": "s"}
NEAR_DUPLICATE = {"segment": (0.75, 1.05), "arc_inc": (-0.05, 0.25)}
YLIM = (-2.6, 0.6)


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def table(tables: Path, name: str) -> list[dict]:
    p = tables / name
    if not p.is_file():
        raise FileNotFoundError(f"missing data source: {p}")
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def draw(ax, crv: list[dict], pts: list[dict], label: str, axis: str, colour: str) -> None:
    seed = label.split("_")[1]
    mk = MARKERS[seed]
    for cls, col, ls, value, flag in (("moved", colour, "-", "mean", "counts"), ("neutral", ZONE_LINE["corridor"], "-", "mean", "counts"),
                                      ("old_ud", ZONE_LINE["UD"], "-.", "mean", "counts")):
        sel = sorted((float(r["t"]), float(r[value]), truthy(r[flag])) for r in crv
                     if r["label"] == label and r["axis"] == axis and r["class"] == cls and r[value] not in ("", "nan"))
        if not sel:
            continue
        ax.plot([s[0] for s in sel], [s[1] for s in sel], ls, color=col, lw=LW * 0.7, alpha=0.9, zorder=2)
        for t, b, counts in sel:
            ax.plot(t, b, mk, color=col, mfc=col if counts else "white", mec=col, mew=1.8, ms=MS, zorder=3)


def draw_missing(ax, pts: list[dict], labels: list[str], axis: str, colour: str) -> None:
    """A cross on the bottom edge where NO agent of the row has a kept moved slot (one row-level mark: per-agent crosses at
    the same height hid the agents that do have one)."""
    have = {round(float(p["t"]), 2) for p in pts if p["label"] in labels and p["axis"] == axis and p["class"] == "moved"}
    missing = [k / 10 for k in range(11) if round(k / 10, 2) not in have]
    if missing:  # an empty unclipped line would sit at the figure's origin and break the layout
        ax.plot(missing, [YLIM[0] + 0.08] * len(missing), "x", color=colour, mew=2.2, ms=MS, zorder=3, clip_on=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-k", type=int, default=300000)
    parser.add_argument("--name", default="E5f_swap_gradient")
    args = parser.parse_args(argv)
    crv, pts = table(args.tables, "curves.csv"), table(args.tables, "points.csv")
    labels = sorted({r["label"] for r in crv})
    shuffle = [x for x in labels if x.startswith("shuffle_")]
    baseline = [x for x in labels if x.startswith("baseline_") and x.endswith(f"_k{args.baseline_k:06d}")]
    if not shuffle:
        raise SystemExit("no shuffle measurement in the tables")
    plt.style.use("default")
    plt.rcParams.update({"axes.labelsize": SIZE_LABEL, "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_TICK,
                         "lines.linewidth": LW, "lines.markersize": MS})
    fig, axs = plt.subplots(2, 3, figsize=FULL, sharex=True, sharey=True, layout="constrained")
    for row, (group, colour) in enumerate(((shuffle, SHUFFLE), (baseline, EXPOSED))):
        for col, axis in enumerate(AXES):
            ax = axs[row, col]
            if axis in NEAR_DUPLICATE:
                ax.axvspan(*NEAR_DUPLICATE[axis], color="#DDDDDD", zorder=0, lw=0)
            ax.axhline(0.0, color=CHANCE, ls="--", lw=2.2, zorder=1)
            ax.axhline(-LOG2_5, color=THEORY, ls="--", lw=2.2, zorder=1)
            for label in group:
                draw(ax, crv, pts, label, axis, colour)
            draw_missing(ax, pts, group, axis, colour)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(*YLIM)
            if row == 1:
                ax.set_xlabel(axis.replace("_", " "))
        axs[row, 0].set_ylabel(f"{'shuffle' if row == 0 else 'baseline'}\nb (bits)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
