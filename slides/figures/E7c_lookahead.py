"""E7c -- Directed choices against the look-ahead depth k (D4): exposed with its own model, exposed + oracle transitions,
exposed + latent correction at the root, reference. Static greedy share on 100 contexts of the zone (mean of 4 agents,
range band); full episodes of the oracle (k = 1, 5, 10) as open markers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E7c_lookahead"
D = ("results", "E7_latent_manipulation", "dev_e4_cpu_diag", "analysis")
KS = [1, 2, 3, 5, 10]
SERIES = [("reference", "model", "reference", P.REFERENCE),
          ("exposed", "oracle", "exposed + oracle transitions", P.ORACLE),
          ("exposed", "model_corrected", "exposed + latent correction", P.CORRECTION),
          ("exposed", "model", "exposed (own model)", P.EXPOSED)]


def main():
    st = C.read_csv(C.path(*D, "d4_static.csv"))
    ep = C.read_csv(C.path(*D, "d4_episodes.csv"))
    fig, ax = C.full()
    fig.subplots_adjust(left=0.09, right=0.72, top=0.8, bottom=0.21)
    C.title(fig, "True transitions fix behaviour; the latent correction plateaus",
            "Share of directed greedy choices in the old UD quadrant vs look-ahead depth k (4 agents)")
    for model, variant, name, col in SERIES:
        m, lo, hi = [], [], []
        for k in KS:
            v = np.array([C.num(r["greedy_directed"]) for r in st
                          if r["model"] == model and r["variant"] == variant and int(r["k"]) == k])
            m.append(v.mean())
            lo.append(v.min())
            hi.append(v.max())
        ax.fill_between(KS, lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(KS, m, "-o", color=col, lw=3, ms=7, zorder=4)
        dy = {"reference": 8, "exposed + oracle transitions": -14}.get(name, 0)
        C.end_label(ax, KS[-1], m[-1], f"{name}  {m[-1]:.2f}", col, dx=12, dy=dy)
    ranges = []
    for k in (1, 5, 10):
        v = np.array([C.num(r["directed_share"]) for r in ep if r["model"] == "exposed" and r["k"] == str(k)])
        ax.scatter([k], [v.mean()], s=110, marker="D", color=P.BG, edgecolor=P.ORACLE, lw=2.2, zorder=5)
        ranges.append(f"k = {k}: {v.min():.2f}–{v.max():.2f}")
    C.chance_line(ax, 0.4, label="chance", where=6.2)
    ax.set_xscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()
    ax.set_xlim(0.9, 11)
    ax.set_ylim(0.25, 1.04)
    ax.set_xlabel("look-ahead depth k (imagined steps)")
    ax.set_ylabel("directed share")
    C.hgrid(ax)
    ax.text(1.55, 0.585, "◇  exposed + oracle, full episodes (ε-greedy)\n    range over agents: " + ", ".join(ranges),
            color=P.ORACLE, fontsize=P.SIZE_SMALL, va="center")
    C.footnote(fig, "Lines: static greedy choices on 100 zone contexts (mean, band = range over 4 agents). Oracle: true "
                    "next state, value from the exposed critic. Correction: b corrected in h at the real step only "
                    "(the imagination falls back onto the conveyor). Source: results/E7_latent_manipulation/"
                    "dev_e4_cpu_diag/analysis/d4_static.csv, d4_episodes.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
