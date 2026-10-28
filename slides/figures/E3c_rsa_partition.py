"""E3c -- RSA of the GRU state h at 300,000 steps: unique share of explained variance of each theoretical matrix
(position, theme, controllability, stochasticity) and the shared share, exposed model vs reference, mean of 4 seeds
(analysis.json, rsa_over_seeds)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E3c_rsa_partition"
SRC = ("results", "E3_worldmodel_4zones", "campaign", "analysis.json")
CHECKPOINT = "300000"
PREDICTORS = ["position", "theme", "controllability", "stochasticity", "shared"]


def main():
    d = json.load(open(C.path(*SRC), encoding="utf-8"))["rsa_over_seeds"][CHECKPOINT]["h"]
    part = d["partition"]
    vals = {}
    for run in ("experimental", "reference"):
        u = part[run]["unique"]
        vals[run] = [u.get(p, np.nan) if p != "shared" else part[run]["shared"] for p in PREDICTORS]
    r2 = {"experimental": d["r2"]["env"]["mean"], "reference": d["r2"]["ref"]["mean"]}

    fig, ax = C.full()
    fig.subplots_adjust(left=0.2, right=0.93, top=0.78, bottom=0.17)
    C.title(fig, "The world model's state is organised by position, not by controllability",
            "Share of the latent h's representational geometry explained uniquely by each factor (300k steps, 4 seeds)")
    y = np.arange(len(PREDICTORS))[::-1]
    hgt = 0.36
    for k, (run, col, name) in enumerate((("experimental", P.EXPOSED, "exposed"),
                                          ("reference", P.REFERENCE, "reference"))):
        v = np.array(vals[run], float)
        yy = y + (hgt / 2 if k == 0 else -hgt / 2)
        ax.barh(yy, np.nan_to_num(v), height=hgt * 0.92, color=col, zorder=2)
        for yi, vi, p in zip(yy, v, PREDICTORS):
            if np.isnan(vi):
                txt = "n/a (constant)"
            else:
                txt = f"{vi:.3f}"
            ax.text((0 if np.isnan(vi) else vi) + 0.006, yi, txt, va="center", fontsize=P.SIZE_SMALL,
                    color=col if not np.isnan(vi) else P.TEXT_2)
        ax.text(0.6, 0.72 - 0.08 * k, f"{name}  (total R² {r2[run]:.2f})", transform=ax.transAxes, color=col,
                fontsize=P.SIZE_ANNOT + 1, fontweight="bold", va="top")
    ax.set_yticks(y, ["position", "theme", "controllability", "stochasticity", "shared"])
    ax.tick_params(axis="y", length=0, labelsize=P.SIZE_AXIS)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("unique share of explained variance")
    ax.xaxis.grid(True, color=P.GRID)
    C.footnote(fig, "State-level RSA of h; in E3 theme, position and zone are confounded (each theme sits in one "
                    "region), which E4 separates. Controllability is constant in the reference (removed). Source: "
                    "results/E3_worldmodel_4zones/campaign/analysis.json (rsa_over_seeds, 300000, h, partition).")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
