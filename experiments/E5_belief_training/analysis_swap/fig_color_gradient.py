"""E5g_color_gradient (EXPLORATORY, REPORT.md section 10; not planned by docs/e5/DECISIONS.md (38)): b at step 1 against
t along E4's colour axes in the swapped slot m, with the localisation under each curve, from gradient_start.py's
gradient_points.csv, read-only. The style of the E5 slide figures (style_e5.py on branch e6, its constants copied as in
fig_swap_gradient.py: axis names 18 pt, ticks 15 pt, no title, legend or annotation, reference lines dashed without text).

Layout: one column per axis (arc_ctrl, segment, then arc_inc if any kept point remains), x = t; one row per shuffle agent
(A against its own reference, A against B's reference when given ((39)), E, F), each a b panel over a localisation panel.
  b(m)            violet, solid, filled circles with the bootstrap SE (over the walks), mean over the swaps KEPT at that
                  t; hollow circles: the mean over the swaps NOT kept at that t (invalid or badly localised), never removed
  neutral slots   grey, dotted (kept)
  b(m) - neutral  violet, dashed, diamonds (kept)
  baselines       light grey, A and B at --baseline-k, kept zones only: b(nw) dash-dot, b(m) solid
  localisation    slot m's localisation at step 1 (mean over the three swaps; the band down to their minimum); shaded
                  where the minimum falls below 0.90 (also in the b panel). Dashed: 0.90
Dashed lines in the b panels: 0 (grey) and -log2 5 (black).

Usage: python experiments/E5_belief_training/analysis_swap/fig_color_gradient.py --tables <tables>
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
SHUFFLE, THEORY, CHANCE = "#A21CAF", "#000000", "#888888"
ZONE_LINE = {"corridor": "#6E6E6E"}
BASELINE_GREY, SHADE = "#C8C8C8", "#E4E4E4"
LW, MS, SIZE_LABEL, SIZE_TICK = 2.8, 9, 18, 15
LOG2_5 = 2.321928094887362
LOC_MIN = 0.90
AXES = ("arc_ctrl", "segment", "arc_inc")
AGENTS = ("shuffle_A_k650000", "shuffle_A-refB_k650000", "shuffle_E_k650000", "shuffle_F_k650000")
ROW_NAME = {"shuffle_A_k650000": "A (own ref)", "shuffle_A-refB_k650000": "A (ref B)", "shuffle_E_k650000": "E", "shuffle_F_k650000": "F"}
YLIM, LOC_YLIM = (-2.6, 1.0), (0.0, 1.05)  # b(m) - neutral reaches +0.83 (A against B, arc_inc)


def number(v: str) -> float | None:
    return None if v in ("", "nan", None) else float(v)


def table(tables: Path) -> list[dict]:
    p = tables / "gradient_points.csv"
    if not p.is_file():
        raise FileNotFoundError(f"missing data source: {p}")
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def series(pts: list[dict], label: str, axis: str, cls: str) -> list[dict]:
    return sorted((r for r in pts if r["label"] == label and r["axis"] == axis and r["class"] == cls), key=lambda r: float(r["t"]))


def kept_line(ax, rows: list[dict], colour: str, ls: str, marker: str, err: bool = False, lw: float = LW, ms: float = MS, z: int = 3) -> None:
    sel = [(float(r["t"]), number(r["mean_kept"]), number(r["se_kept"])) for r in rows if number(r["mean_kept"]) is not None]
    if not sel:
        return
    t, m, se = zip(*sel)
    ax.plot(t, m, ls, color=colour, lw=lw, zorder=z)
    if marker:
        if err:
            ax.errorbar(t, m, yerr=se, fmt=marker, color=colour, ms=ms, mew=1.8, capsize=3, lw=1.6, zorder=z + 1)
        else:
            ax.plot(t, m, marker, color=colour, ms=ms, mew=1.8, zorder=z + 1)


def shade_low_loc(ax, moved: list[dict]) -> None:
    for r in moved:
        if float(r["loc_m_min"]) < LOC_MIN:
            t = float(r["t"])
            ax.axvspan(t - 0.05, t + 0.05, color=SHADE, lw=0, zorder=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-k", type=int, default=300000)
    parser.add_argument("--name", default="E5g_color_gradient")
    args = parser.parse_args(argv)
    pts = table(args.tables)
    labels = {r["label"] for r in pts}
    agents = [x for x in AGENTS if x in labels]
    if not agents:
        raise SystemExit("no shuffle agent in gradient_points.csv")
    baselines = sorted(x for x in labels if x.startswith("baseline_") and x.endswith(f"_k{args.baseline_k:06d}"))
    axes = [a for a in AXES if a != "arc_inc" or any(number(r["mean_kept"]) is not None for x in agents for r in series(pts, x, a, "moved"))]
    plt.style.use("default")
    plt.rcParams.update({"axes.labelsize": SIZE_LABEL, "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_TICK,
                         "lines.linewidth": LW, "lines.markersize": MS})
    ratios = [r for _ in agents for r in (3, 1)]
    fig, axs = plt.subplots(2 * len(agents), len(axes), figsize=(13.33, 3.4 * len(agents)), sharex=True,
                            gridspec_kw={"height_ratios": ratios}, layout="constrained", squeeze=False)
    for i, label in enumerate(agents):
        for j, axis in enumerate(axes):
            ax, lx = axs[2 * i, j], axs[2 * i + 1, j]
            moved = series(pts, label, axis, "moved")
            shade_low_loc(ax, moved)
            shade_low_loc(lx, moved)
            ax.axhline(0.0, color=CHANCE, ls="--", lw=2.0, zorder=1)
            ax.axhline(-LOG2_5, color=THEORY, ls="--", lw=2.0, zorder=1)
            for b in baselines:
                kept_line(ax, series(pts, b, axis, "old_ud"), BASELINE_GREY, "-.", "", lw=2.0, z=1)
                kept_line(ax, series(pts, b, axis, "moved"), BASELINE_GREY, "-", "", lw=2.0, z=1)
            kept_line(ax, series(pts, label, axis, "neutral"), ZONE_LINE["corridor"], ":", "", lw=2.4, z=2)
            kept_line(ax, series(pts, label, axis, "delta"), SHUFFLE, "--", "D", lw=2.0, ms=6, z=2)
            kept_line(ax, moved, SHUFFLE, "-", "o", err=True)
            lost = [(float(r["t"]), number(r["mean_not_kept"])) for r in moved if number(r["mean_not_kept"]) is not None]
            if lost:
                ax.plot(*zip(*lost), "o", color=SHUFFLE, mfc="white", mew=1.8, ms=MS, zorder=4)
            t = [float(r["t"]) for r in moved]
            lx.fill_between(t, [float(r["loc_m_min"]) for r in moved], [float(r["loc_m_mean"]) for r in moved], color=SHUFFLE, alpha=0.2, lw=0)
            lx.plot(t, [float(r["loc_m_mean"]) for r in moved], "-", color=SHUFFLE, lw=2.0)
            lx.axhline(LOC_MIN, color=CHANCE, ls="--", lw=1.6)
            ax.set_ylim(*YLIM)
            lx.set_ylim(*LOC_YLIM)
            lx.set_yticks([0, 0.9])
            ax.set_xlim(-0.05, 1.05)
            if j:
                ax.tick_params(labelleft=False)
                lx.tick_params(labelleft=False)
        axs[2 * i, 0].set_ylabel(f"{ROW_NAME[label]}\nb (bits)", fontsize=SIZE_LABEL - 3)
        axs[2 * i + 1, 0].set_ylabel("loc", fontsize=SIZE_LABEL - 3)
    for j, axis in enumerate(axes):
        axs[-1, j].set_xlabel(axis.replace("_", " "))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
