"""E3b -- World-model b (I_s, exposed minus reference) per zone over training: 4 seeds (thin) and their mean (thick,
on the checkpoints measured for all four), with the analytic values (UD -log2 5, UN -1.65, 0 elsewhere)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E3b_b_over_training"
SRC = ("results", "E3_worldmodel_4zones", "campaign", "tables", "b_alpha_by_checkpoint.csv")
ZONES = [("uncontrollable_deterministic", "UD"), ("uncontrollable_noisy", "UN"), ("controllable_noisy", "CN"),
         ("controllable_deterministic", "CD"), ("corridor", "corridor")]
# Analytic b of the 4-zone environment, as measured exactly by E1 (measurements.csv, I*(exp) - I*(ctrl)).
THEORY = {"UD": -np.log2(5), "UN": -1.6529325012980811}


def main():
    rows = C.read_csv(C.path(*SRC))
    seeds = sorted({int(r["seed"]) for r in rows})
    fig, ax = C.full()
    fig.subplots_adjust(left=0.09, right=0.8, top=0.8, bottom=0.16)
    C.title(fig, "The world model learns the conveyor; controllable zones stay at 0",
            "b = I(exposed) − I(reference) per zone, 4 seeds (thin) and mean (thick)")
    ends = {}
    for zone, code in ZONES:
        col = P.ZONE_LINE[code]
        per = {}
        for s in seeds:
            sel = sorted(((int(r["env_step"]), C.num(r["b_I_s"])) for r in rows
                          if int(r["seed"]) == s and r["zone"] == zone))
            per[s] = dict(sel)
            ax.plot(*zip(*sel), color=col, lw=1.0, alpha=0.4, zorder=2)
        common = sorted(set.intersection(*(set(d) for d in per.values())))
        mean = np.array([np.mean([per[s][x] for s in seeds]) for x in common])
        ax.plot(common, mean, "-o", color=col, lw=3.0, ms=5, zorder=4)
        ends[code] = (common[-1], mean[-1])
    for code, th in THEORY.items():
        ax.hlines(th, 2e4, 3.3e5, color=P.THEORY, ls=P.THEORY_LS, lw=1.6, zorder=1)
    ax.axhline(0, color=P.THEORY, ls=P.THEORY_LS, lw=1.6, zorder=1)
    offsets = {"UD": 0, "UN": 0, "CN": 14, "CD": -2, "corridor": -18}
    for code, (x, y) in ends.items():
        txt = f"{code}  {C.fmt(y)}" + (f"  (theory {C.fmt(THEORY[code])})" if code in THEORY else "")
        C.end_label(ax, x, y, txt, P.ZONE_LINE[code], dx=10, dy=offsets[code])
    ax.set_xscale("log")
    ax.set_xlim(1.4e4, 3.4e5)
    ax.set_xticks([2e4, 5e4, 1e5, 2e5, 3e5], ["20k", "50k", "100k", "200k", "300k"])
    ax.minorticks_off()
    ax.set_ylim(-2.6, 0.75)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("b  (bits)")
    C.hgrid(ax)
    C.footnote(fig, "Dotted: analytic b (−log₂5 for UD, −1.65 for UN, 0 for controllable zones). Mean over seeds 0, 1, "
                    "5, 6 on checkpoints common to all four; pilot seeds 0 and 5 end at their 290k replay (flagged in "
                    "the E3 report). Source: results/E3_worldmodel_4zones/campaign/tables/b_alpha_by_checkpoint.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
