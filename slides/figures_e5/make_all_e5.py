"""E5 slide figures (MANIFEST_e5.md): E5a position vs theme, E5b theme gradient, E5c distribution of beliefs, E5d
recovery. Reads experiments/E5_belief_training/analysis_450k/tables (analyse_450k.py), read-only.

    .venv/Scripts/python.exe slides/figures_e5/make_all_e5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_e5 as S  # noqa: E402

GROUP = {"baseline": "baseline", "shuffle": "shuffle"}  # analyse_450k groups shown (baseline_matched: tables only)
BINS_MIN_N = 2  # a binned mean needs >= 2 valid probes (E4b's rule)


def e5a_position_vs_theme():
    rows = [r for r in S.table("extremes.csv") if r["group"] in GROUP and r["probe"] in ("position_only", "theme_only")]
    fig, ax = S.full()
    S.ref_lines(ax)
    width = 0.34
    for j, probe in enumerate(("position_only", "theme_only")):
        for i, (cond, col) in enumerate(S.CONDITIONS):
            b = np.array([float(r["b_step1"]) for r in rows if r["group"] == cond and r["probe"] == probe and S.truthy(r["valid"])])
            x = j + (i - 0.5) * width
            ax.bar(x, b.mean(), width * 0.9, facecolor=col, alpha=0.35, edgecolor=col, lw=3, zorder=2)
            ax.plot(x + np.linspace(-0.06, 0.06, len(b)), b, "o", color=col, mec="black", mew=1.2, ms=13, zorder=3)
    ax.set_xticks([0, 1], ["position only", "theme only"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-2.6, 0.3)
    ax.set_ylabel("b at step 1 (bits)")
    return S.save(fig, "E5a_position_vs_theme")


def e5b_theme_gradient():
    rows = [r for r in S.table("gradient_probes.csv") if r["group"] in GROUP and S.truthy(r["valid"])]
    fig, axs = S.full(1, 2, sharey=True)
    rng = np.random.default_rng(1)  # jitter of the probe dots only
    edges = np.round(np.arange(-0.05, 1.1501, 0.1), 2)  # centred on multiples of 0.1 D (analyse_450k.BINS)
    for ax, (cond, ccol) in zip(axs, S.CONDITIONS):
        S.ref_lines(ax)
        for pos, col in S.POSITIONS:
            sel = [r for r in rows if r["group"] == cond and r["position"] == pos]
            x = np.array([float(r["dist_to_inc_over_D"]) for r in sel])
            b = np.array([float(r["b_step1"]) for r in sel])
            ax.scatter(x + rng.uniform(-0.008, 0.008, len(x)), b, s=30, color=col, alpha=0.4, zorder=2)
            idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
            pts = [(x[idx == i].mean(), b[idx == i].mean()) for i in range(len(edges) - 1) if (idx == i).sum() >= BINS_MIN_N]
            if pts:
                ax.plot(*zip(*pts), "o-", color=col, lw=3.2, zorder=3)
        for side in ax.spines.values():  # the panel's condition: its frame in the condition's colour
            side.set_color(ccol)
            side.set_linewidth(2.5)
        ax.set_xlim(-0.05, 1.08)
        ax.set_xlabel("distance to T_inc (/ D)")
    axs[0].set_ylabel("b at step 1 (bits)")
    axs[0].set_ylim(-2.6, 0.3)
    return S.save(fig, "E5b_theme_gradient")


def e5c_distribution():
    rows = [r for r in S.table("gradient_probes.csv") if r["group"] in GROUP and S.truthy(r["valid"])]
    seeds = {"baseline": ("2", "102"), "shuffle": ("402", "502")}
    fig, ax = S.full()
    S.ref_lines(ax)
    rng = np.random.default_rng(2)  # horizontal spread only
    for i, (cond, col) in enumerate(S.CONDITIONS):
        for s, marker in zip(seeds[cond], ("o", "s")):
            b = np.array([float(r["b_step1"]) for r in rows if r["group"] == cond and r["seed"] == s])
            ax.plot(i + rng.uniform(-0.22, 0.22, len(b)), b, marker, ls="none", color=col, alpha=0.75, ms=10, zorder=2)
    ax.set_xticks([0, 1], [c for c, _ in S.CONDITIONS])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-2.6, 0.3)
    ax.set_ylabel("b at step 1 (bits)")
    return S.save(fig, "E5c_distribution_of_beliefs")


def e5d_recovery():
    rows = [r for r in S.table("recovery.csv") if r["group"] in GROUP]
    k = np.arange(1, 26)
    fig, ax = S.full()
    S.ref_lines(ax)
    for cond, col in S.CONDITIONS:
        B = np.array([[float(r[f"b_k{i}"]) for i in k] for r in rows if r["group"] == cond])
        for b in B:
            ax.plot(k, b, color=col, lw=1.3, alpha=0.5)
        ax.plot(k, B.mean(axis=0), color=col, lw=3.4)
    ax.set_xlim(0.5, 25.5)
    ax.set_ylim(-2.6, 0.3)
    ax.set_xlabel("step in quadrant")
    ax.set_ylabel("b (bits)")
    return S.save(fig, "E5d_recovery_in_episode")


def main() -> int:
    for f in (e5a_position_vs_theme, e5b_theme_gradient, e5c_distribution, e5d_recovery):
        for p in f():
            print(p.relative_to(S.REPO), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
