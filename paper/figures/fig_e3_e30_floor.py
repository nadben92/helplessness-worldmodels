"""E3: first floor of b = I_s(environment) - I_s(reference) by zone, E3-0 (gpu_e30b: two window shapes x seeds 0 and 5,
final weights), against the true kernels' difference. One marker per (window shape, seed) pair. Writes
paper/figures/out/fig_e3_e30_floor.pdf.

Source: the 8 gpu_e30b run directories under results/E3_worldmodel_4zones/ (results.json, arrays.npz), paired by
experiments/E3_worldmodel_4zones/aggregate_b.py (load_run, aggregate: b on I_s, real contexts). The true difference is
the zone mean of measures.controllability.controllability_per_state on each run's P_true, environment minus reference.
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

from helplessness.measures.controllability import controllability_per_state

NAME = "fig_e3_e30_floor"
SHAPES = (("win64", "o"), ("win16", "s"))
SEED_COLORS = {0: common.TOL_BRIGHT["blue"], 5: common.TOL_BRIGHT["red"]}


def true_zone_means(run_dir) -> dict:
    res = src.read_json(run_dir / "results.json")
    arrays = np.load(common.require_file(run_dir / "arrays.npz"))
    zone = np.array(res["zone_of_state"])
    per_state = controllability_per_state(arrays["P_true"])
    return {z: float(per_state[zone == z].mean()) for z in src.ZONES}


def main() -> None:
    aggregate_b = src.e3_module("aggregate_b.py")
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.75, aspect=0.6)
    x = np.arange(len(src.ZONES))
    truth = None
    for k, (shape, marker) in enumerate(SHAPES):
        runs = src.e30b_runs(shape)
        agg = aggregate_b.aggregate([aggregate_b.load_run(r) for r in runs])
        if agg["unpaired"] or len(agg["pairs"]) != 2:
            raise SystemExit(f"gpu_e30b {shape}: expected 2 complete pairs, got {len(agg['pairs'])} (unpaired {agg['unpaired']})")
        for pair in agg["pairs"]:
            offset = (k - 0.5) * 0.3 + (0.06 if pair["seed"] == 5 else -0.06)
            ax.plot(x + offset, [pair["I_s"][z]["b"] for z in src.ZONES], marker, ls="none", ms=4,
                    color=SEED_COLORS[pair["seed"]], label=f"{shape}, seed {pair['seed']}")
        by_condition = {src.run_identity(r)[1]: true_zone_means(r) for r in runs if src.run_identity(r)[0] == 0}
        t = [by_condition["experimental"][z] - by_condition["reference"][z] for z in src.ZONES]
        if truth is not None and not np.allclose(truth, t):
            raise SystemExit("the true b differs between the window shapes' runs (same seed, same worlds expected)")
        truth = t
    for xi, t in zip(x, truth):
        ax.hlines(t, xi - 0.35, xi + 0.35, color="black", lw=0.8, label="true kernels" if xi == 0 else None)
    ax.axhline(0.0, color="gray", lw=0.5, zorder=0)
    ax.set_xticks(x, [src.ZONE_ABBR[z] for z in src.ZONES])
    ax.set_ylabel(r"$\hat b$ (bits)")
    ax.legend(loc="lower right", fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
