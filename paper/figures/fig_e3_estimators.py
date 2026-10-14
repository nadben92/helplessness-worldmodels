"""E3: the three posterior estimators of the autoencoder isolation test (no KL, k = 6, lr 1e-3, 3 seeds; criterion
fixed before the runs: hard-sample template accuracy >= 0.95 on all 3 seeds). Hard-sample template accuracy against
the training step, one line per seed. Writes paper/figures/out/fig_e3_estimators.pdf.

Source: results/E3_worldmodel_4zones/estimator_comparison/{straight_through,gumbel_st,warmup_probs}_seedNN.json
(experiments/E3_worldmodel_4zones/compare_estimators.py; docs/e3/ESTIMATOR_COMPARISON.md). The curves are plotted as
recorded.
"""

from __future__ import annotations

import common
import e3_sources as src

NAME = "fig_e3_estimators"
ROOT = src.E3_RESULTS / "estimator_comparison"
ESTIMATORS = (("straight_through", "(a) straight-through (ST)"), ("gumbel_st", "(b) Gumbel-softmax"),
              ("warmup_probs", "(c) probabilities, then ST"))
SEEDS = (0, 1, 2)
CRITERION = 0.95  # fixed before the runs (DECISIONS 2026-09-24, "Categorical-bottleneck fix", point 2)


def main() -> None:
    common.apply_style()
    fig, axes = common.figure(aspect=0.36, ncols=3, sharey=True)
    colors = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"], common.TOL_BRIGHT["green"])
    for ax, (estimator, title) in zip(axes, ESTIMATORS):
        for seed, color in zip(SEEDS, colors):
            run = src.read_json(ROOT / f"{estimator}_seed{seed:02d}.json")
            if run["estimator"] != estimator or run["seed"] != seed:
                raise SystemExit(f"{estimator}_seed{seed:02d}.json records {run['estimator']}, seed {run['seed']}")
            steps = [p["step"] for p in run["curve"]]
            ax.plot(steps, [p["acc_hard"] for p in run["curve"]], color=color, lw=1.0, label=f"seed {seed}")
            switch = run["schedule"].get("warmup_steps") or 0
        if switch:
            ax.axvline(switch, color="black", lw=0.6, ls=":")
        ax.axhline(CRITERION, color="gray", lw=0.6, ls="--")
        ax.set_title(title, fontsize=common.SMALL_FONT_SIZE)
        ax.set_xlabel("training step")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("template accuracy (hard sample)")
    axes[0].legend(loc="center right")
    fig.tight_layout(pad=0.2)
    path = common.save_pdf(fig, NAME)
    print(f"wrote {path.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
