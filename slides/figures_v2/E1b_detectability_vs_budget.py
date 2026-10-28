"""E1b (v2) -- Smallest detectable |b| (bootstrap null p95, zone CN) vs budget n, log-log, 20 seeds; vertical lines at
the retained n = 100 (solid) and the 3-sigma margin n = 300 (dashed)."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E1b_detectability_vs_budget"
SRC = ("experiments", "E1_offline_4zones", "artifacts", "permutation_null.csv")


def main():
    rows = [r for r in C.read_csv(C.path(*SRC))
            if r["zone"] == "controllable_noisy" and r["quantity"] == "b" and r["method"] == "bootstrap"]
    ns = sorted({int(r["n_budget"]) for r in rows})
    by_n = {n: np.array([C.num(r["p95"]) for r in rows if int(r["n_budget"]) == n]) for n in ns}
    mean = np.array([by_n[n].mean() for n in ns])
    half = np.array([stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) for v in by_n.values()])
    fig, ax = C.full()
    ax.fill_between(ns, mean - half, mean + half, color=P.REFERENCE, alpha=0.25, lw=0)
    for n in ns:
        ax.scatter(np.full(len(by_n[n]), n), by_n[n], s=18, color=P.REFERENCE, alpha=0.5)
    ax.plot(ns, mean, "o-", color=P.REFERENCE)
    ax.axvline(100, color="black", lw=2)
    ax.axvline(300, color=P.CHANCE, lw=2, ls=P.CHANCE_LS)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_yticks([0.01, 0.02, 0.05, 0.1], ["0.01", "0.02", "0.05", "0.1"])
    ax.set_xlabel("budget n")
    ax.set_ylabel("detectable |b| (bits)")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
