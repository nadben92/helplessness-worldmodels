"""The robustness diagnostic's figures (REPORT.md), read-only from analyse.py's tables:
  R1_localisation   localisation at step 1 against t, by model, the three axes (cluster, restricted swaps)
  R2_sensitivity    per-variable TV against t on arc_ctrl (a2: the image's effect, h of the origin), three panels:
                    agent in the recoloured slot / in another slot / in the corridor; rows E, F, A
  R3_step4          a2 (bars) and a1 (dots) of 4a, 4b, 4c and the canonical image, by model
  R4_scatter        area under arc_ctrl against the locality index L, the nine models

Colours: E, F, A in the first three categorical slots, every other model in grey; the families of R3 in four slots
(validated, adjacent pairs; contrast relief: values printed on the bars).

Usage: python experiments/E5_belief_training/analysis_robustness/figures.py [--tables <dir>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
FOCUS = {"E": SLOT[0], "F": SLOT[1], "A": SLOT[2]}
GREY = "#9a9893"
MECHANISTIC = ("baseA_exp_k300k", "baseA_ref_k300k", "baseB_exp_k300k", "baseB_ref_k300k", "A", "ref_A", "E", "F", "ref_B")


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def threshold(ax) -> None:
    ax.axhspan(0.85, 0.95, color=GRID, alpha=0.6, linewidth=0)
    ax.axhline(0.90, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))


def fig_localisation(tables: Path, out: Path) -> None:
    rows = read(tables / "scores_restricted.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    for ax, axis in zip(axes, ("arc_ctrl", "segment", "arc_inc")):
        style(ax)
        threshold(ax)
        for r in rows:
            if r["axis"] != axis or not r["swaps"]:
                continue
            t = [k / 10 for k in range(11)]
            y = [float(r[f"t{k:02d}"]) for k in range(11)]
            focus = r["id"] in FOCUS
            ax.plot(t, y, color=FOCUS.get(r["id"], GREY), linewidth=2 if focus else 1,
                    linestyle="-" if focus or "ref" not in r["id"] else (0, (4, 2)), zorder=3 if focus else 2, label=r["id"])
            if focus:
                ax.annotate(r["id"], (t[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.set_title(axis, fontsize=10, color=INK)
        ax.set_xlabel("t", fontsize=9, color=MUTED)
        ax.set_ylim(-0.02, 1.04)
    axes[0].set_ylabel("localisation at step 1 (cluster)", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=m) for m, c in FOCUS.items()]
    handles += [plt.Line2D([], [], color=GREY, linewidth=1, label="baselines (exposed)"),
                plt.Line2D([], [], color=GREY, linewidth=1, linestyle=(0, (4, 2)), label="references")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Localisation along the swap axes (swaps localised at t = 0; band: 0.85-0.95)", y=1.08, fontsize=10, color=INK)
    fig.savefig(out / "R1_localisation.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_sensitivity(tables: Path, out: Path) -> None:
    rows = [r for r in read(tables / "sensitivity.csv") if r["tv"] == "tv_a2"]
    pos = {r["id"]: set(map(int, r["pos_vars"].split("+"))) for r in read(tables / "latent_summary.csv")}
    models = [m for m in ("E", "F", "A") if m in pos]
    fig, axes = plt.subplots(len(models), 3, figsize=(11, 2.8 * len(models)), sharex=True, sharey=True, facecolor=SURFACE, squeeze=False)
    for i, mid in enumerate(models):
        for j, where in enumerate(("moved", "other", "corridor")):
            ax = axes[i, j]
            style(ax)
            sub = sorted((r for r in rows if r["id"] == mid and r["where"] == where), key=lambda r: float(r["t"]))
            t = [float(r["t"]) for r in sub]
            for v in range(8):
                p = v in pos[mid]
                ax.plot(t, [float(r[f"v{v}"]) for r in sub], color=FOCUS[mid] if p else GREY, linewidth=1.8 if p else 0.9,
                        zorder=3 if p else 2)
            ax.axhline(0.10, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            if i == 0:
                ax.set_title({"moved": "agent in the recoloured slot", "other": "agent in another slot", "corridor": "agent in the corridor"}[where],
                             fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{mid}: TV per variable", fontsize=9, color=MUTED)
            if i == len(models) - 1:
                ax.set_xlabel("t (arc_ctrl)", fontsize=9, color=MUTED)
    fig.legend(handles=[plt.Line2D([], [], color=INK, linewidth=1.8, label="position variables (Pos, coloured by model)"),
                        plt.Line2D([], [], color=GREY, linewidth=0.9, label="other variables"),
                        plt.Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), label="'moves little' (0.10)")],
               loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Posterior sensitivity to the recolouring of m (h of the origin image)", y=0.99, fontsize=10, color=INK)
    fig.subplots_adjust(top=0.9)
    fig.savefig(out / "R2_sensitivity.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_step4(tables: Path, out: Path) -> None:
    rows = read(tables / "step4.csv")
    fams = ("canonical", "4a", "4b", "4c")
    models = [m for m in MECHANISTIC if any(r["id"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURFACE)
    style(ax)
    threshold(ax)
    width = 0.2
    for i, mid in enumerate(models):
        for j, fam in enumerate(fams):
            r = next(r for r in rows if r["id"] == mid and r["family"] == fam)
            x = i + (j - 1.5) * (width + 0.02)
            a2, a1 = float(r["a2"]), float(r["a1"])
            ax.bar(x, a2, width, color=SLOT[j], edgecolor=SURFACE, linewidth=1, yerr=2 * float(r["a2_se"]), ecolor=MUTED, capsize=0,
                   label=fam if i == 0 else None)
            ax.plot(x, a1, marker="o", markersize=4, color=INK, linestyle="none", label="a1 (own h)" if i == 0 and j == 0 else None)
            ax.text(x, a2 + 0.02, f"{a2:.2f}", ha="center", va="bottom", fontsize=6, color=INK, rotation=90)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_k300k", "\n300k") for m in models], fontsize=8, color=INK)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("localisation at step 1 (a2, bars; ±2 SE)", fontsize=9, color=MUTED)
    ax.legend(loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Step 4: new backgrounds (4a uniform, 4b distinct) and a recoloured agent (4c)", y=1.06, fontsize=10, color=INK)
    fig.savefig(out / "R3_step4.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_scatter(tables: Path, out: Path) -> None:
    rows = read(tables / "models_summary.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor=SURFACE)
    style(ax)
    for r in rows:
        L, a, se = float(r["L"]), float(r["area_arc_ctrl"]), float(r["L_se"])
        ref = "ref" in r["id"]
        c = FOCUS.get(r["id"], GREY)
        ax.errorbar(L, a, xerr=2 * se, fmt="o", markersize=8, color=c, mfc=SURFACE if ref else c, mec=c, ecolor=c, elinewidth=1,
                    markeredgewidth=2)
        ax.annotate(r["id"], (L, a), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("locality index L (other / moved, Pos variables; ±2 SE)", fontsize=9, color=MUTED)
    ax.set_ylabel("area under arc_ctrl (restricted swaps)", fontsize=9, color=MUTED)
    ax.set_title("Robustness against locality (filled: exposed, hollow: reference)", fontsize=9, color=INK)
    fig.savefig(out / "R4_scatter.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, default=HERE / "tables")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_localisation(args.tables, args.out)
    fig_sensitivity(args.tables, args.out)
    fig_step4(args.tables, args.out)
    fig_scatter(args.tables, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
