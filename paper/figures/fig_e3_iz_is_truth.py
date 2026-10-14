"""E3: controllability by zone measured in z-space (I_z) and in s-space (I_s), against the true kernel's, on the bridge
run (E3-0, seed 0, environment, final weights, K_s = 1,358 draws per (s, a), real contexts). Writes
paper/figures/out/fig_e3_iz_is_truth.pdf.

Source: results/E3_worldmodel_4zones/*gpu_e30_bridge*/recomputed/final__ks1358__20260930T124347Z/ (results.json:
I_by_category_real = I_z, I_s_by_category_real = I_s; arrays.npz: P_true). The truth is the zone mean of
measures.controllability.controllability_per_state on the run's true kernel (the one implementation of I).
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

from helplessness.measures.controllability import controllability_per_state

NAME = "fig_e3_iz_is_truth"


def main() -> None:
    path = src.bridge_recomputation()
    res = src.read_json(path / "results.json")
    arrays = np.load(common.require_file(path / "arrays.npz"))
    zone = np.array(res["zone_of_state"])
    truth_per_state = controllability_per_state(arrays["P_true"])
    series = {
        r"$\hat I_z$": [res["I_by_category_real"][z] for z in src.ZONES],
        r"$\hat I_s$": [res["I_s_by_category_real"][z] for z in src.ZONES],
        "true kernel": [float(truth_per_state[zone == z].mean()) for z in src.ZONES],
    }
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.75, aspect=0.55)
    width = 0.26
    colors = (common.TOL_BRIGHT["red"], common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["grey"])
    x = np.arange(len(src.ZONES))
    for i, ((label, values), color) in enumerate(zip(series.items(), colors)):
        ax.bar(x + (i - 1) * width, values, width, color=color, label=label)
    ax.set_xticks(x, [src.ZONE_ABBR[z] for z in src.ZONES])
    ax.set_ylabel(r"$I(a; s' \mid s)$ (bits)")
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 2.75)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
