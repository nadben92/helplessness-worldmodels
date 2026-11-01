"""E5 slide figures for the talk (MANIFEST_e5.md): E5a_v2 position vs theme in two panels, E5e theme only (neutral
slot) vs distance to T_inc; with --loc, the same on the probes kept by the localisation criterion
(analysis_450k/LOCALISATION_CRITERION.md, apply_localisation.py): E5e_loc, and E5a_v3 if an extreme probe is removed.
Does not regenerate E5a-E5d (make_all_e5.py). Reads the analysis tables, read-only.

--set picks the measurements and the figures' suffix (SETS): 475k (default, the preliminary figures, unchanged), 650k
(shuffle A, E, F at 650,000 against the baselines at 300,000, docs/e5/DECISIONS.md (37)) and 650k_matched (the same
shuffle measurements against the baselines at the matched checkpoint, 100,000). --tables: the analysis tables of the set
(default: style_e5.TABLES, analysis_450k's); a measurement absent from a table (no localisation) is left out.

    .venv/Scripts/python.exe slides/figures_e5/make_e5_v2.py [--loc] [--set 650k --tables <analysis_650k/tables>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_e5 as S  # noqa: E402

# per set: the measurements of each condition (tables' `label`, one per seed) and the suffix of the figure ids
SETS = {
    "475k": ({"baseline": ("baseline_A_k300000", "baseline_B_k300000"), "shuffle": ("shuffle_E_k475000", "shuffle_F_k475000")}, ""),
    "650k": ({"baseline": ("baseline_A_k300000", "baseline_B_k300000"),
              "shuffle": ("shuffle_A_k650000", "shuffle_E_k650000", "shuffle_F_k650000")}, "_650k"),
    "650k_matched": ({"baseline": ("baseline_A_k100000", "baseline_B_k100000"),
                      "shuffle": ("shuffle_A_k650000", "shuffle_E_k650000", "shuffle_F_k650000")}, "_650k_matched"),
}
LABELS, SUFFIX = SETS["475k"]
YLIM = (-2.4, 0.2)


def e5a_v2_position_vs_theme(source: str = "extremes.csv", keep: str = "valid", fig_id: str = "E5a_v2_position_vs_theme"):
    rows = [r for r in S.table(source) if r["probe"] in ("position_only", "theme_only")]
    fig, axs = S.full(1, 2, sharey=True)
    for ax, (cond, col) in zip(axs, S.CONDITIONS):
        ax.axhline(0.0, color=S.CHANCE, ls="--", lw=2.2, zorder=1)
        for j, probe in enumerate(("position_only", "theme_only")):
            b = np.array([float(r["b_step1"]) for r in rows if r["label"] in LABELS[cond] and r["probe"] == probe and S.truthy(r[keep])])
            if not len(b):
                continue
            ax.bar(j, b.mean(), 0.6, facecolor=col, alpha=0.4, edgecolor=col, lw=3, zorder=2)
            ax.plot(j + np.linspace(-0.08, 0.08, len(b)), b, "o", color=col, mec="black", mew=1.2, ms=14, zorder=3)
        ax.set_xticks([0, 1], ["position only", "theme only"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel(cond)
    axs[0].set_ylim(*YLIM)
    axs[0].set_ylabel("b (bits)")
    return S.save(fig, fig_id)


def e5e_theme_only(source: str = "gradient_binned.csv", fig_id: str = "E5e_theme_only"):
    """Neutral slot only: per seed, the mean b of its valid probes per 0.1 D bin (gradient_binned.csv); thin = seed,
    thick = mean of the seeds over the bins every seed of the condition has."""
    rows = [r for r in S.table(source) if r["unit"] == "label" and r["position"] == "neutral"]
    fig, ax = S.full()
    ax.axhline(0.0, color=S.CHANCE, ls="--", lw=2.2, zorder=1)
    for cond, col in S.CONDITIONS:
        per_seed = {}
        for s in LABELS[cond]:
            sel = sorted((float(r["bin_from"]) + 0.05, float(r["b_mean"])) for r in rows if r["key"] == s)
            if sel:  # a seed may have no probe left under a filter
                per_seed[s] = dict((round(x, 2), b) for x, b in sel)
                ax.plot(*zip(*sel), "-", color=col, lw=1.4, alpha=0.55, zorder=2)
        if not per_seed:
            continue
        common = sorted(set.intersection(*(set(v) for v in per_seed.values())))
        ax.plot(common, [np.mean([per_seed[s][x] for s in per_seed]) for x in common], "o-", color=col, lw=3.4, ms=10, zorder=3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.2, 0.2)
    ax.set_xlabel("theme distance")
    ax.set_ylabel("b at step 1 (bits)")
    return S.save(fig, fig_id)


def main(argv: list[str] | None = None) -> int:
    global LABELS, SUFFIX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loc", action="store_true")
    parser.add_argument("--set", choices=list(SETS), default="475k")
    parser.add_argument("--tables", type=Path, default=None)
    args = parser.parse_args(argv)
    LABELS, SUFFIX = SETS[args.set]
    if args.tables is not None:
        S.TABLES = args.tables
    shown = {x for v in LABELS.values() for x in v}
    if args.loc:
        figs = [lambda: e5e_theme_only("gradient_binned_loc.csv", "E5e_loc" + SUFFIX)]
        if not all(S.truthy(r["kept"]) for r in S.table("extremes_loc.csv") if r["label"] in shown and S.truthy(r["valid"])):
            figs.append(lambda: e5a_v2_position_vs_theme("extremes_loc.csv", "kept", "E5a_v3_position_vs_theme" + SUFFIX))
    else:
        figs = [lambda: e5a_v2_position_vs_theme(fig_id="E5a_v2_position_vs_theme" + SUFFIX),
                lambda: e5e_theme_only(fig_id="E5e_theme_only" + SUFFIX)]
    for f in figs:
        for p in f():
            print(p.relative_to(S.REPO), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
