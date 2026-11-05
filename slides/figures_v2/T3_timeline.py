"""T3 (v2) -- E1 -> E8 in four parts; status by styling only (filled: done; orange: running; dashed: to come).
Statuses from docs/EXPERIMENTS.md and the experiment docs (2026-10-28); E8 = therapy and relapse (user's plan)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "T3_timeline"
PARTS = [("Measure", ["E1", "E2", "E3"]), ("Locate", ["E4", "E5"]), ("Belief → behaviour", ["E6", "E7"]),
         ("Correct", ["E8"])]
STEPS = {"E1": ("Tabular", "done"), "E2": ("Online", "done"), "E3": ("World\nmodel", "done"),
         "E4": ("Generali-\nsation", "done"), "E5": ("Belief\ntraining", "running"), "E6": ("Behaviour", "done"),
         "E7": ("Dissoci-\nation", "done"), "E8": ("Therapy\n& relapse", "next")}
STYLE = {"done": (P.ZONE_CD, P.REFERENCE, "-"), "running": (P.ANTI_ORACLE, "#C46F00", "-"),
         "next": ("white", P.CHANCE, "--")}


def main():
    fig = C.plt.figure(figsize=C.FULL)
    ax = D.canvas(fig)
    x0, gap, w, h, y = 0.07, 0.123, 0.108, 0.42, 0.45
    xs = {e: x0 + i * gap for i, e in enumerate(STEPS)}
    for part, members in PARTS:
        a, b = xs[members[0]] - w / 2, xs[members[-1]] + w / 2
        ax.plot([a, b], [0.77, 0.77], color="black", lw=2.5)
        ax.text((a + b) / 2, 0.8, part, ha="center", va="bottom", fontsize=P.SIZE_BLOCK, fontweight="bold")
    for e, (name, st) in STEPS.items():
        fc, ec, ls = STYLE[st]
        D.box(ax, xs[e], y, w, h, "", fc=fc, ec=ec, lw=3, ls=ls)
        ax.text(xs[e], y + 0.09, e, ha="center", va="center", fontsize=30, fontweight="bold")
        ax.text(xs[e], y - 0.08, name, ha="center", va="center", fontsize=P.SIZE_BLOCK - 2, linespacing=1.15)
    D.conn(ax, (0.03, 0.14), (0.97, 0.14), color="black", lw=2.2)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
