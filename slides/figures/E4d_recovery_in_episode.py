"""E4d -- Recovery within the episode: b against the step in the old UD quadrant (training image, 4 configurations),
with the step-1 -> plateau (steps 6-10) gap over all valid UD probes (recovery_summary_by_position.csv)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E4d_recovery_in_episode"
T = ("results", "E4_generalization_gradient", "final", "tables")


def main():
    curves = [r for r in C.read_csv(C.path(*T, "recovery_curves_extremes.csv"))
              if r["probe"] == "training" and r["zone_role"] == "ud"]
    summ = [r for r in C.read_csv(C.path(*T, "recovery_summary_by_position.csv")) if r["position"] == "ud"]
    gaps = np.array([C.num(r["mean_gap_plateau_minus_step1"]) for r in summ])
    k = np.arange(1, 26)
    fig, ax = C.full()
    fig.subplots_adjust(left=0.09, right=0.8, top=0.78, bottom=0.2)
    C.title(fig, "After one step that works, b recovers a little, then stalls",
            "b in the old UD quadrant by step since entry, training image (frozen models, controllable world)")
    allb = []
    for r in curves:
        b = np.array([C.num(r[f"b_k{i}"]) for i in k])
        allb.append(b)
        ax.plot(k, b, color=P.EXPOSED, lw=1.1, alpha=0.45, zorder=2)
    mean = np.mean(allb, axis=0)
    ax.plot(k, mean, color=P.EXPOSED, lw=3.2, zorder=4)
    ax.axvspan(6, 10, color=P.GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(8, -1.45, "plateau\n(steps 6–10)", ha="center", fontsize=P.SIZE_SMALL, color=P.TEXT_2)
    ax.annotate("", (3.0, mean[5:10].mean()), (3.0, mean[0]),
                arrowprops=dict(arrowstyle="-|>", color=P.TEXT, lw=1.8))
    ax.text(3.4, (mean[0] + mean[5:10].mean()) / 2,
            f"+{gaps.min():.2f} to +{gaps.max():.2f} bit\n(per configuration, all valid UD probes)",
            fontsize=P.SIZE_ANNOT, color=P.TEXT, va="center", fontweight="bold")
    C.theory_line(ax, -np.log2(5))
    ax.set_xlim(0.5, 25.5)
    ax.set_ylim(-2.45, -1.35)
    ax.set_xlabel("step since entering the quadrant")
    ax.set_ylabel("b  (bits)")
    C.hgrid(ax)
    fig.text(0.045, 0.085, "Within-class variation of b; not read by the critic (D7).", fontsize=P.SIZE_ANNOT,
             color=P.TEXT, fontweight="bold")
    C.footnote(fig, "Thin: 4 configurations; thick: mean; dotted: −log₂5. Sources: results/E4_generalization_gradient/"
                    "final/tables/recovery_curves_extremes.csv and recovery_summary_by_position.csv.", y=0.025)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
