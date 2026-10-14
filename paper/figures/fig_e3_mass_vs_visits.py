"""E3: why the agent_mini prior never imagined the goal. (a) median mass of the prior on the right successor of the
deterministic pairs off the goal, against the pair's visits n (agent_mini, environment, seed 0; weights at 18,000 and
20,000 environment steps; K = 1,358 draws per pair, real contexts), with the four pairs into the goal (mass on the
goal). (b) the same median over the pairs visited 2 to 8 times after 0 to 20,000 more gradient steps on the frozen
replay of 18,000. Writes paper/figures/out/fig_e3_mass_vs_visits.pdf.

Sources (plotted as recorded):
  reports/diagnostics_wm_goal_<agent_mini run>/diagnostics.json (experiments/E3_worldmodel_4zones/diagnose_world_model_goal.py):
    by_n (median per visit count) and goal_pairs, for each weight set;
  reports/gradients_vs_data_<agent_mini run>/result.json (gradients_vs_data.py): points.
"""

from __future__ import annotations

import common
import e3_sources as src

NAME = "fig_e3_mass_vs_visits"
RUN = "grid12x12__4zones__pair-experimental__worldmodel-agent-agent_mini__seed00__20260930T193212Z"
DIAG = common.REPO_ROOT / "reports" / f"diagnostics_wm_goal_{RUN}" / "diagnostics.json"
GRADS = common.REPO_ROOT / "reports" / f"gradients_vs_data_{RUN}" / "result.json"
WEIGHT_SETS = (("18000", "kernel_on_replay_at_18000", "18,000 steps"),
               ("20000", "kernel_final_measures_replay_at_20000", "20,000 steps"))


def main() -> None:
    diag, grads = src.read_json(DIAG), src.read_json(GRADS)
    common.apply_style()
    fig, (ax_a, ax_b) = common.figure(aspect=0.42, ncols=2, gridspec_kw={"width_ratios": [2, 1]})
    colors = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"])
    for (key, sub, label), color in zip(WEIGHT_SETS, colors):
        kernel = diag[key][sub]
        by_n = sorted((int(n), v) for n, v in kernel["by_n"].items())
        ax_a.plot([n + 1 for n, _ in by_n], [v["median"] for _, v in by_n], ".", ms=3, color=color, label=label)
        goal = [(p["n"], p["mass_on_goal"]) for p in kernel["goal_pairs"]]
        ax_a.plot([n + 1 for n, _ in goal], [m for _, m in goal], "x", ms=5, mew=1.0, color=color)
    ax_a.plot([], [], "x", color="black", label="pairs into the goal")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("visits of the pair, $n + 1$")
    ax_a.set_ylabel("prior mass on the right successor")
    ax_a.set_title("(a)", fontsize=common.SMALL_FONT_SIZE)
    ax_a.legend(loc="center right", fontsize=common.SMALL_FONT_SIZE - 1)

    points = sorted((int(k), v) for k, v in grads["points"].items())
    ax_b.plot([k for k, _ in points], [v["median_mass_right_successor_n2_8"] for _, v in points], "o-", ms=3,
              color=common.TOL_BRIGHT["green"])
    ax_b.set_xlabel("extra gradient steps")
    ax_b.set_ylabel("median mass, $2 \\leq n \\leq 8$")
    ax_b.set_ylim(0, 1)
    ax_b.set_title("(b)", fontsize=common.SMALL_FONT_SIZE)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
