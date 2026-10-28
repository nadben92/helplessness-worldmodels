"""E3b (v2) -- World-model b per zone over training: 4 seeds (thin) and mean on common checkpoints (thick), zone
colours as in T1; analytic values dotted (UD -log2 5, UN -1.65, 0)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E3b_b_over_training"
SRC = ("results", "E3_worldmodel_4zones", "campaign", "tables", "b_alpha_by_checkpoint.csv")
ZONES = [("uncontrollable_deterministic", "UD"), ("uncontrollable_noisy", "UN"), ("controllable_noisy", "CN"),
         ("controllable_deterministic", "CD"), ("corridor", "corridor")]
THEORY = (-np.log2(5), -1.6529325012980811, 0.0)  # analytic b, as measured exactly by E1


def main():
    rows = C.read_csv(C.path(*SRC))
    seeds = sorted({int(r["seed"]) for r in rows})
    fig, ax = C.full()
    for y in THEORY:
        C.theory(ax, y)
    for zone, code in ZONES:
        col = P.ZONE_LINE[code]
        per = {}
        for s in seeds:
            sel = sorted((int(r["env_step"]), C.num(r["b_I_s"])) for r in rows if int(r["seed"]) == s and r["zone"] == zone)
            per[s] = dict(sel)
            ax.plot(*zip(*sel), color=col, lw=1.2, alpha=0.45)
        common = sorted(set.intersection(*(set(d) for d in per.values())))
        ax.plot(common, [np.mean([per[s][x] for s in seeds]) for x in common], "o-", color=col, lw=3.2)
    ax.set_xscale("log")
    ax.set_xticks([2e4, 5e4, 1e5, 2e5, 3e5], ["20k", "50k", "100k", "200k", "300k"])
    ax.minorticks_off()
    ax.set_xlabel("environment steps")
    ax.set_ylabel("b (bits)")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
