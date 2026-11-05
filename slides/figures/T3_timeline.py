"""T3 -- The project's steps E1 -> E8 with one-line questions and status (statuses from docs/EXPERIMENTS.md and the
experiment docs as of 2026-10-28; E6/E7 run on E4's weights, waiting for E5's agents)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _schema as S  # noqa: E402

FIG_ID = "T3_timeline"

PARTS = [("I  Measure", ["E1", "E2", "E3"]), ("II  Locate", ["E4", "E5"]),
         ("III  Belief → behaviour", ["E6", "E7"]), ("IV  Correct", ["E8"])]
STEPS = {
    "E1": ("Tabular", "Does the measure recover true control?", "done"),
    "E2": ("Online tabular", "Does the agent's own exploration bias b?", "done"),
    "E3": ("World model", "Does a world model learn the uncontrollable zone?", "done"),
    "E4": ("Generalisation", "Is the belief tied to position or to colour?", "done"),
    "E5": ("Belief training", "Can training move the belief onto colour?", "running"),
    "E6": ("Behaviour", "Does behaviour follow the belief?", "done*"),
    "E7": ("Dissociation", "Belief or critic: which one drives behaviour?", "done*"),
    "E8": ("Immunisation", "Does prior control protect the belief?", "next"),
}
STATUS = {"done": ("done", P.ZONE_CD, P.REFERENCE, "-"), "done*": ("done*", P.ZONE_CD, P.REFERENCE, "-"),
          "running": ("running", "#F1DDB0", P.GOAL, "-"), "next": ("to come", P.BG, P.CHANCE, "--")}


def main():
    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "Two months: from measuring the belief to testing what it does")
    ax = S.canvas(fig, (0.02, 0.06, 0.96, 0.80))
    n, x0, gap = 8, 0.065, 0.124
    xs = {e: x0 + i * gap for i, e in enumerate(STEPS)}
    w, h, y = 0.112, 0.50, 0.46
    for part, members in PARTS:
        a, b = xs[members[0]] - w / 2, xs[members[-1]] + w / 2
        ax.plot([a, b], [0.79, 0.79], color=P.TEXT_2, lw=2)
        ax.text((a + b) / 2, 0.815, part, ha="center", va="bottom", fontsize=P.SIZE_ANNOT + 1, color=P.TEXT,
                fontweight="bold")
    for e, (name, q, st) in STEPS.items():
        label, fc, ec, ls = STATUS[st]
        x = xs[e]
        S.box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=2.2, ls=ls)
        ax.text(x, y + 0.19, e, ha="center", va="center", fontsize=P.SIZE_TITLE + 2, fontweight="bold", color=P.TEXT)
        ax.text(x, y + 0.11, name, ha="center", va="center", fontsize=P.SIZE_SMALL, color=P.TEXT_2)
        ax.text(x, y - 0.04, _wrap(q, 13), ha="center", va="center", fontsize=P.SIZE_SMALL + 0.5, color=P.TEXT,
                linespacing=1.25)
        ax.text(x, y - 0.205, label, ha="center", va="center", fontsize=P.SIZE_ANNOT, color=ec, fontweight="bold")
    S.conn(ax, (0.03, 0.12), (0.97, 0.12), color=P.AXIS, lw=1.2)
    C.footnote(fig, "* E6 and E7 developed and run on E4's agents (CPU); to be rerun on E5's agents. "
                    "E5's figures will slot in at part II.", y=0.03)
    return C.save(fig, FIG_ID)


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    lines.append(cur)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
