"""E3c (v2) -- RSA of h at 300,000 steps: unique share of explained variance per factor and shared share, exposed vs
reference (mean of 4 seeds, analysis.json). Controllability is constant in the reference (no bar)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E3c_rsa_partition"
SRC = ("results", "E3_worldmodel_4zones", "campaign", "analysis.json")
PREDICTORS = ["position", "theme", "controllability", "stochasticity", "shared"]


def main():
    part = json.load(open(C.path(*SRC), encoding="utf-8"))["rsa_over_seeds"]["300000"]["h"]["partition"]
    fig, ax = C.full()
    y = np.arange(len(PREDICTORS))[::-1]
    h = 0.38
    for k, (run, col) in enumerate((("experimental", P.EXPOSED), ("reference", P.REFERENCE))):
        u = part[run]["unique"]
        v = [part[run]["shared"] if p == "shared" else u.get(p, np.nan) for p in PREDICTORS]
        ax.barh(y + (h / 2 if k == 0 else -h / 2), np.nan_to_num(v), height=h * 0.95, color=col)
    ax.set_yticks(y, PREDICTORS)
    ax.set_xlabel("unique share of explained variance")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
