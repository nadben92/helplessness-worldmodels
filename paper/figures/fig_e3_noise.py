"""E3, measurement noise (pilot data: final weights of the 4 pilot runs, replay of 290,000 environment steps): standard
deviation of I_s by zone over 8 independent draws of the real contexts (and of the kernel's prior draws), against the
number of contexts per state (x1 = 50, x2 = 100, x4 = 200). Writes paper/figures/out/fig_e3_noise.pdf.

Source: results/E3_worldmodel_4zones/gpu_pilot_noise/<run>/noise__<run>.json (extracted from handoff/gpu_pilot_noise/;
experiments/E3_worldmodel_4zones/measure_noise.py): by_multiplier.<m>.sd_by_zone and max_per_state, plotted as recorded.
"""

from __future__ import annotations

import common
import e3_sources as src

NAME = "fig_e3_noise"
STYLE = {0: "o", 5: "s"}
COLOR = {"experimental": common.TOL_BRIGHT["red"], "reference": common.TOL_BRIGHT["blue"]}


def main() -> None:
    common.apply_style()
    fig, axes = common.figure(aspect=0.55, ncols=4, sharey=True)
    for run in src.pilot_runs():
        seed, condition = src.run_identity(run)
        noise = src.read_json(src.NOISE_ROOT / run.name / f"noise__{run.name}.json")
        if noise["run"] != run.name:
            raise SystemExit(f"noise file of {run.name} records run {noise['run']}")
        mults = sorted(noise["by_multiplier"], key=int)
        contexts = [noise["by_multiplier"][m]["max_per_state"] for m in mults]
        for ax, zone in zip(axes, src.ZONES):
            ax.plot(contexts, [noise["by_multiplier"][m]["sd_by_zone"][zone] for m in mults], marker=STYLE[seed], ms=3,
                    lw=0.9, ls="-" if condition == "experimental" else "--", color=COLOR[condition],
                    label=f"{src.CONDITION_ABBR[condition]} {seed}")
    for ax, zone in zip(axes, src.ZONES):
        ax.set_title(src.ZONE_ABBR[zone], fontsize=common.SMALL_FONT_SIZE)
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts, [str(c) for c in contexts])
        ax.minorticks_off()
        ax.set_xlabel("contexts / state")
    axes[0].set_ylabel(r"sd of $\hat I_s$ over 8 draws (bits)")
    axes[1].legend(loc="upper right", fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
