"""E3, closed-loop pilot (pilot data: seeds 0 and 5, environment and reference; measurement protocol x1, 50 contexts per
state): I_s by zone at the re-measured checkpoints 63,096 to 300,000 environment steps, one line per run, with its
bootstrap standard error. The 300,000 point is measured on the replay of 290,000 (the last one saved). Writes
paper/figures/out/fig_e3_pilot_is.pdf.

Source: results/E3_worldmodel_4zones/gpu_pilot_remeasure/<run>/env<NNNNNN>/results.json (extracted from
handoff/gpu_pilot_remeasure/; experiments/E3_worldmodel_4zones/remeasure_checkpoints.py): real_context.I_s_by_zone.
"""

from __future__ import annotations

import common
import e3_sources as src

NAME = "fig_e3_pilot_is"
STYLE = {0: "o", 5: "s"}
COLOR = {"experimental": common.TOL_BRIGHT["red"], "reference": common.TOL_BRIGHT["blue"]}


def main() -> None:
    common.apply_style()
    fig, axes = common.figure(aspect=0.62, nrows=2, ncols=2, sharex=True)
    for run in src.pilot_runs():
        seed, condition = src.run_identity(run)
        base = src.REMEASURE_ROOT / run.name
        points = sorted(int(d[3:]) for d in src.listdir(base) if d.startswith("env"))
        measures = [src.read_json(base / f"env{c:06d}" / "results.json")["real_context"]["I_s_by_zone"] for c in points]
        for ax, zone in zip(axes.flat, src.ZONES):
            mean = [m[zone]["mean"] for m in measures]
            se = [m[zone]["se"] for m in measures]
            ax.errorbar(points, mean, yerr=se, marker=STYLE[seed], ms=2.5, lw=0.9, capsize=1.5,
                        ls="-" if condition == "experimental" else "--", color=COLOR[condition],
                        label=f"{src.CONDITION_ABBR[condition]} {seed}")
    for ax, zone in zip(axes.flat, src.ZONES):
        ax.set_title(src.ZONE_ABBR[zone], fontsize=common.SMALL_FONT_SIZE)
        ax.set_xscale("log")
    for ax in axes[1]:
        ax.set_xlabel("environment steps")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\hat I_s$ (bits)")
    axes[0, 0].legend(loc="center right", fontsize=common.SMALL_FONT_SIZE - 1, ncols=2)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
