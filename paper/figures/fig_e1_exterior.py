"""E1 Step A: V* outside the uncontrollable blocks against the distance to each of them (NW: the uncontrollable-
deterministic block; SE: the uncontrollable-noisy one), in the environment and in its control, at the frozen gamma,
R and drift direction. The four curves coincide. Number of states per distance bin annotated. Writes
paper/figures/out/fig_e1_exterior.pdf.

Source: experiments/E1_offline_4zones/artifacts/gradient_curves.csv (calibration.py), plotted as recorded; gamma, R and
the direction from experiments/E1_offline_4zones/config.yaml.
"""

from __future__ import annotations

import common
import e1_sources as src

NAME = "fig_e1_exterior"
SERIES = (("exp", "nw_uncontrollable_det", "o", "-", "env, distance to NW"),
          ("ctrl", "nw_uncontrollable_det", "s", "--", "control, distance to NW"),
          ("exp", "se_uncontrollable_noisy", "^", ":", "env, distance to SE"),
          ("ctrl", "se_uncontrollable_noisy", "v", "-.", "control, distance to SE"))


def main() -> None:
    cfg = src.e1_config()
    rows = [r for r in src.gradient_curves()
            if r["gamma"] == cfg["gamma"] and r["R"] == cfg["goal_reward"] and r["direction"] == cfg["direction"]]
    if not rows:
        raise SystemExit("gradient_curves.csv: no row at the frozen gamma, R and direction")
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.6, aspect=0.75)
    colors = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"], common.TOL_BRIGHT["green"], common.TOL_BRIGHT["purple"])
    for (env, zone, marker, ls, label), color in zip(SERIES, colors):
        sub = sorted((r for r in rows if r["env"] == env and r["zone"] == zone), key=lambda r: r["distance"])
        ax.plot([r["distance"] for r in sub], [r["V_mean"] for r in sub], marker=marker, ls=ls, ms=4, lw=0.9,
                color=color, label=label, mfc="none")
    ref = sorted((r for r in rows if r["env"] == "exp" and r["zone"] == SERIES[0][1]), key=lambda r: r["distance"])
    for r in ref:
        ax.annotate(f"{r['n']} states", (r["distance"], r["V_mean"]), textcoords="offset points", xytext=(6, 4),
                    fontsize=common.SMALL_FONT_SIZE - 1)
    ax.set_xticks([r["distance"] for r in ref])
    ax.set_xlabel("distance to the block (cells)")
    ax.set_ylabel(r"mean exterior $V^*$")
    ax.legend(fontsize=common.SMALL_FONT_SIZE - 1, loc="upper right")
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
