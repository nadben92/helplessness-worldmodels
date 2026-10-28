"""E1a (v2) -- Tabular b per zone (n = 1000, 20 seeds: mean bar, 95% t-interval, seeds) and analytic b (dotted)."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E1a_b_vs_theory"
SRC = ("experiments", "E1_offline_4zones", "artifacts", "measurements.csv")
N_BUDGET = 1000
ZONES = [("controllable_deterministic", "CD"), ("controllable_noisy", "CN"),
         ("uncontrollable_noisy", "UN"), ("uncontrollable_deterministic", "UD")]


def main():
    rows = C.read_csv(C.path(*SRC))
    fig, ax = C.full()
    rng = np.random.default_rng(0)  # jitter of the seed dots only
    for i, (zone, code) in enumerate(ZONES):
        sel = [r for r in rows if r["zone"] == zone and r["region"] == "all" and int(r["n_budget"]) == N_BUDGET]
        exp = {int(r["seed"]): r for r in sel if r["env"] == "exp"}
        ctrl = {int(r["seed"]): r for r in sel if r["env"] == "ctrl"}
        b = np.array([C.num(exp[s]["b"]) for s in sorted(exp)])
        th = C.num(exp[0]["I_star"]) - C.num(ctrl[0]["I_star"])
        half = stats.t.ppf(0.975, len(b) - 1) * b.std(ddof=1) / np.sqrt(len(b))
        ax.bar(i, b.mean(), width=0.6, color=P.ZONE_COLORS[code], edgecolor="black", lw=1.2)
        ax.errorbar(i, b.mean(), yerr=half, color="black", capsize=8, lw=2)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(b)), b, s=30, color="black", zorder=4)
        ax.hlines(th, i - 0.42, i + 0.42, color=P.THEORY, ls=P.THEORY_LS, lw=2.6, zorder=5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(ZONES)), [c for _, c in ZONES])
    ax.set_ylabel("b (bits)")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
