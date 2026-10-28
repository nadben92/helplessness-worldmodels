"""E1b -- Smallest detectable |b| (bootstrap null p95, zone CN where b = 0 is true) against the sample budget n,
log-log, mean over 20 seeds; E1's retained n = 100 (2-sigma margin) and n = 300 (3-sigma)."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E1b_detectability_vs_budget"
SRC = ("experiments", "E1_offline_4zones", "artifacts", "permutation_null.csv")
TOLERANCE = 0.03  # bits, E1's precision tolerance (etape_b_report.txt)


def main():
    rows = [r for r in C.read_csv(C.path(*SRC))
            if r["zone"] == "controllable_noisy" and r["quantity"] == "b" and r["method"] == "bootstrap"]
    ns = sorted({int(r["n_budget"]) for r in rows})
    by_n = {n: np.array([C.num(r["p95"]) for r in rows if int(r["n_budget"]) == n]) for n in ns}
    mean = np.array([by_n[n].mean() for n in ns])
    half = np.array([stats.t.ppf(0.975, len(by_n[n]) - 1) * by_n[n].std(ddof=1) / np.sqrt(len(by_n[n])) for n in ns])
    slope, intercept, *_ = stats.linregress(np.log10(ns), np.log10(mean))

    fig, ax = C.full()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
    C.title(fig, "100 samples per state-action resolve b to a few hundredths of a bit",
            "Smallest detectable |b| (95th percentile of the bootstrap null), zone CN, 20 seeds")
    x = np.array(ns, float)
    ax.fill_between(x, mean - half, mean + half, color=P.REFERENCE, alpha=P.ALPHA_CI, lw=0)
    for n in ns:
        ax.scatter(np.full(len(by_n[n]), n), by_n[n], s=12, color=P.REFERENCE, alpha=0.35, lw=0, zorder=3)
    ax.plot(x, mean, "o-", color=P.REFERENCE, zorder=4)
    xf = np.geomspace(8, 1300, 50)
    ax.plot(xf, 10 ** intercept * xf ** slope, color=P.REFERENCE, lw=1.0, ls=(0, (4, 3)), alpha=0.8)
    ax.annotate(f"slope {C.fmt(slope)}", (60, 10 ** intercept * 60 ** slope), xytext=(10, 14),
                textcoords="offset points", fontsize=P.SIZE_ANNOT, color=P.REFERENCE)
    for n, lab in ((100, "retained\nn = 100"), (300, "margin\nn = 300")):
        ax.axvline(n, color=P.TEXT_2 if n == 100 else P.CHANCE, lw=1.6 if n == 100 else 1.2,
                   ls="-" if n == 100 else P.CHANCE_LS, zorder=1)
        ax.text(n * 1.06, 0.15, lab, fontsize=P.SIZE_ANNOT, color=P.TEXT if n == 100 else P.TEXT_2, va="top")
    i100 = ns.index(100)
    ax.annotate(f"{mean[i100]:.3f} bit", (100, mean[i100]), xytext=(-14, -26), textcoords="offset points",
                ha="right", fontsize=P.SIZE_ANNOT, color=P.TEXT, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7, 1500)
    ax.set_ylim(0.008, 0.16)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_yticks([0.01, 0.02, 0.05, 0.1], ["0.01", "0.02", "0.05", "0.1"])
    ax.minorticks_off()
    ax.set_xlabel("samples per (state, action), n")
    ax.set_ylabel("detectable |b|  (bits)")
    C.hgrid(ax)
    C.footnote(fig, "Band: 95% t-interval over seeds; dots: seeds; dashed: log-log fit. n = 100 is E1's "
                    "recommendation (2σ margin on the 0.03-bit tolerance for UN), n = 300 the 3σ margin. "
                    "Source: experiments/E1_offline_4zones/artifacts/permutation_null.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
