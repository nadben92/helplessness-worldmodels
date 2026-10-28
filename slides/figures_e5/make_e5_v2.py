"""E5 slide figures for the talk (MANIFEST_e5.md): E5a_v2 position vs theme in two panels, E5e theme only (neutral
slot) vs distance to T_inc. Does not regenerate E5a-E5d (make_all_e5.py). Reads the analysis tables, read-only.

    .venv/Scripts/python.exe slides/figures_e5/make_e5_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_e5 as S  # noqa: E402

SEEDS = {"baseline": ("2", "102"), "shuffle": ("402", "502")}
LABELS = {"2": "baseline_A_k300000", "102": "baseline_B_k300000", "402": "shuffle_E_k475000", "502": "shuffle_F_k475000"}
YLIM = (-2.4, 0.2)


def e5a_v2_position_vs_theme():
    rows = [r for r in S.table("extremes.csv") if r["group"] in SEEDS and r["probe"] in ("position_only", "theme_only")]
    fig, axs = S.full(1, 2, sharey=True)
    for ax, (cond, col) in zip(axs, S.CONDITIONS):
        ax.axhline(0.0, color=S.CHANCE, ls="--", lw=2.2, zorder=1)
        for j, probe in enumerate(("position_only", "theme_only")):
            b = np.array([float(r["b_step1"]) for r in rows if r["group"] == cond and r["probe"] == probe and S.truthy(r["valid"])])
            ax.bar(j, b.mean(), 0.6, facecolor=col, alpha=0.4, edgecolor=col, lw=3, zorder=2)
            ax.plot(j + np.linspace(-0.08, 0.08, len(b)), b, "o", color=col, mec="black", mew=1.2, ms=14, zorder=3)
        ax.set_xticks([0, 1], ["position only", "theme only"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel(cond)
    axs[0].set_ylim(*YLIM)
    axs[0].set_ylabel("b (bits)")
    return S.save(fig, "E5a_v2_position_vs_theme")


def e5e_theme_only():
    """Neutral slot only: per seed, the mean b of its valid probes per 0.1 D bin (gradient_binned.csv); thin = seed,
    thick = mean of the seeds over the bins every seed of the condition has."""
    rows = [r for r in S.table("gradient_binned.csv") if r["unit"] == "label" and r["position"] == "neutral"]
    fig, ax = S.full()
    ax.axhline(0.0, color=S.CHANCE, ls="--", lw=2.2, zorder=1)
    for cond, col in S.CONDITIONS:
        per_seed = {}
        for s in SEEDS[cond]:
            sel = sorted((float(r["bin_from"]) + 0.05, float(r["b_mean"])) for r in rows if r["key"] == LABELS[s])
            per_seed[s] = dict((round(x, 2), b) for x, b in sel)
            ax.plot(*zip(*sel), "-", color=col, lw=1.4, alpha=0.55, zorder=2)
        common = sorted(set.intersection(*(set(v) for v in per_seed.values())))
        ax.plot(common, [np.mean([per_seed[s][x] for s in per_seed]) for x in common], "o-", color=col, lw=3.4, ms=10, zorder=3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.2, 0.2)
    ax.set_xlabel("theme distance")
    ax.set_ylabel("b at step 1 (bits)")
    return S.save(fig, "E5e_theme_only")


def main() -> int:
    for f in (e5a_v2_position_vs_theme, e5e_theme_only):
        for p in f():
            print(p.relative_to(S.REPO), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
