"""E7_bars (v2) -- The latent correction in bars: b at step 1, localisation in the zone, directed share (episodes,
k = 1), for exposed, latent fix (unconstrained stepwise, alpha = 1), constrained fix (D7's recorded setting per agent),
oracle (true transitions in the planner, k = 1, D4) and reference. Bar = mean of E4's four agents (training image),
black dots = the four agents. Values from E7h's collection, plus D4 for the oracle:
  oracle b            the true belief: the reference's value (0 by construction), not a measurement
  oracle localisation the exposed model's zone localisation from E6 (h is not modified by the oracle; the D4 runs
                      themselves measured step 1 only, measure = "first", so they hold no zone localisation)
  oracle directed     d4_episodes.csv, exposed, k = 1
Grey dashed: chance (2/5) on the directed share."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import E7h_correction_summary as H  # noqa: E402

FIG_ID = "E7_bars"
D4 = ("results", "E7_latent_manipulation", "dev_e4_cpu_diag", "analysis", "d4_episodes.csv")
CATEGORIES = [("exposed", "exposed", P.EXPOSED), ("latent fix", "unconstrained", "#B98CE8"),
              ("constrained fix", "constrained", P.CORRECTION), ("oracle", "oracle", P.ORACLE),
              ("reference", "reference", P.REFERENCE)]
PANELS = [("b", "b (bits)"), ("loc", "localisation"), ("dir", "directed share")]


def values() -> dict:
    v, _ = H.collect()
    oracle_dir = {r["agent"]: C.num(r["directed_share"]) for r in C.read_csv(C.path(*D4))
                  if r["model"] == "exposed" and r["k"] == "1"}
    v["oracle"] = {"b": dict(v["reference"]["b"]), "loc": dict(v["exposed"]["loc"]),
                   "dir": {a: oracle_dir[a] for a in H.AGENTS}}
    return v


def main():
    v = values()
    fig, axes = C.full(1, 3)
    for ax, (m, label) in zip(axes, PANELS):
        for i, (_, cond, col) in enumerate(CATEGORIES):
            vals = np.array([v[cond][m][a] for a in H.AGENTS], float)
            ax.bar(i, vals.mean(), width=0.75, color=col, edgecolor="black", lw=1)
            ax.scatter(i + np.linspace(-0.18, 0.18, len(vals)), vals, s=28, color="black", zorder=4)
        if m == "dir":
            C.chance(ax, 0.4)
        ax.set_xticks(range(len(CATEGORIES)), [c for c, _, _ in CATEGORIES], rotation=35, ha="right",
                      rotation_mode="anchor")
        ax.set_ylabel(label)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylim(-2.35, 0.15)  # headroom so the bars at 0 (oracle, reference) and their dots stay visible
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    for name, cond, _ in CATEGORIES:
        print(name, {m: round(float(np.mean([v[cond][m][a] for a in H.AGENTS])), 3) for m, _ in PANELS},
              {m: [round(float(v[cond][m][a]), 2) for a in H.AGENTS] for m, _ in PANELS})
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
