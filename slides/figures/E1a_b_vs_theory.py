"""E1a -- Tabular b per zone against its analytic value, 20 seeds, budget n = 1000 (E1's headline budget), all states
of the zone. Theory = I*(exp) - I*(ctrl), read from the same rows (true kernels)."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E1a_b_vs_theory"
SRC = ("experiments", "E1_offline_4zones", "artifacts", "measurements.csv")
N_BUDGET = 1000
ZONES = [("controllable_deterministic", "CD"), ("controllable_noisy", "CN"),
         ("uncontrollable_noisy", "UN"), ("uncontrollable_deterministic", "UD")]


def load():
    rows = C.read_csv(C.path(*SRC))
    out = {}
    for zone, code in ZONES:
        sel = [r for r in rows if r["zone"] == zone and r["region"] == "all" and int(r["n_budget"]) == N_BUDGET]
        exp = {int(r["seed"]): r for r in sel if r["env"] == "exp"}
        ctrl = {int(r["seed"]): r for r in sel if r["env"] == "ctrl"}
        b = np.array([C.num(exp[s]["b"]) for s in sorted(exp)])
        theory = {round(C.num(exp[s]["I_star"]) - C.num(ctrl[s]["I_star"]), 9) for s in exp}
        assert len(theory) == 1, theory
        out[code] = {"b": b, "theory": theory.pop()}
    return out


def main():
    d = load()
    fig, ax = C.full()
    fig.subplots_adjust(left=0.1, right=0.86, top=0.8, bottom=0.14)
    C.title(fig, "The tabular measure recovers the true controllability",
            f"b per zone, 20 seeds, n = {N_BUDGET} samples per (s, a)")
    rng = np.random.default_rng(0)  # jitter of the seed dots only
    for i, (_, code) in enumerate(ZONES):
        b, th = d[code]["b"], d[code]["theory"]
        m = b.mean()
        half = stats.t.ppf(0.975, len(b) - 1) * b.std(ddof=1) / np.sqrt(len(b))
        ax.bar(i, m, width=0.6, color=P.ZONE_COLORS[code], edgecolor=P.ZONE_LINE[code], lw=1.4, zorder=2)
        ax.errorbar(i, m, yerr=half, color=P.ZONE_LINE[code], capsize=6, lw=1.6, alpha=0.9, zorder=3)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(b)), b, s=14, color=P.ZONE_LINE[code], alpha=0.55, zorder=4,
                   lw=0)
        ax.hlines(th, i - 0.42, i + 0.42, color=P.THEORY, ls=P.THEORY_LS, lw=2.2, zorder=5)
        lab = C.fmt(m) if abs(m) > 0.005 else "0.00"
        ax.text(i, m - 0.12 if m < -0.1 else 0.1, f"{lab}\n(theory {C.fmt(th)})" if abs(th) > 0 else f"{lab}",
                ha="center", va="top" if m < -0.1 else "bottom", fontsize=P.SIZE_ANNOT, color=P.TEXT)
    ax.axhline(0, color=P.AXIS, lw=0.9)
    ax.set_xticks(range(len(ZONES)), [c for _, c in ZONES])
    ax.tick_params(axis="x", length=0, labelsize=P.SIZE_AXIS)
    ax.set_ylabel("b  (bits)")
    ax.set_ylim(-2.95, 0.45)
    C.hgrid(ax)
    ax.text(3.55, -np.log2(5), "theory\n−log₂5", color=P.THEORY, fontsize=P.SIZE_SMALL, va="center", ha="left")
    C.footnote(fig, "Bars: mean over 20 seeds, 95% t-interval (narrower than the bar edge); dots: seeds. Dotted: "
                    "analytic b = I*(exp) − I*(ctrl). Source: experiments/E1_offline_4zones/artifacts/measurements.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
