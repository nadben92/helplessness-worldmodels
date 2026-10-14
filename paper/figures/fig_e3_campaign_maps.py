"""E3 final campaign: I_s by state at the budget (300,000 environment steps), mean over the 4 seeds, environment,
reference and their difference b(s), drawn on the CANONICAL grid (rotation 0's placement of the zone types).

The campaign's own maps (docs/e3/campaign_figures/maps_<checkpoint>.png) average the seeds cell by cell in the grid's
coordinates; seeds 0 and 1 run rotation 0 and seeds 5 and 6 rotation 1 (E2's counterbalancing), so each quadrant there
averages one controllable and one uncontrollable zone type. Here each run's value at a state is placed at its zone
type's canonical position before the mean, with E2's own remapping (analysis_data.canonical_rc_for_state: the quadrant
blocks keep their geometry, only the type occupying them moves; the corridor never moves). Display only: the values are
the runs' measured real_I_s, nothing is recomputed. Writes paper/figures/out/fig_e3_campaign_maps.pdf.

Source: results/E3_worldmodel_4zones/campaign/runs/<run>/measures/env300000/ (results.json: seed, condition,
zone_of_state; arrays.npz: real_I_s), the 8 runs the campaign pipeline analysed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common
import e3_sources as src

from helplessness.experiment_loader import load_experiment_module

NAME = "fig_e3_campaign_maps"
CHECKPOINT = 300000
RUNS_ROOT = src.E3_RESULTS / "campaign" / "runs"
N_RUNS = 8  # seeds 0, 1, 5, 6; environment and reference (DECISIONS 2026-10-07 (30))


def load_runs() -> list[dict]:
    runs = []
    for name in src.listdir(RUNS_ROOT):
        d = RUNS_ROOT / name / "measures" / f"env{CHECKPOINT:06d}"
        res = src.read_json(d / "results.json")
        common.require_file(d / "arrays.npz")
        with np.load(common.long_path(d / "arrays.npz")) as arr:
            values = np.asarray(arr["real_I_s"], dtype=float)
        runs.append({"seed": int(res["seed"]), "condition": res["condition"],
                     "zone_of_state": np.asarray(res["zone_of_state"]), "I_s": values})
    if len(runs) != N_RUNS:
        raise SystemExit(f"Expected {N_RUNS} runs in {RUNS_ROOT}, found {len(runs)}")
    return runs


def canonical_mean(runs: list[dict], ad, e3_env) -> np.ndarray:
    acc, count = np.zeros(ad.GRID_SHAPE), np.zeros(ad.GRID_SHAPE)
    for run in runs:
        rotation = e3_env.rotation_for_seed_index(run["seed"])
        if not np.array_equal(run["zone_of_state"], ad.zone_of_state_for_rotation(rotation)):
            raise SystemExit(f"seed {run['seed']} ({run['condition']}): zone_of_state differs from rotation {rotation}'s layout")
        for s, zone_type in enumerate(run["zone_of_state"]):
            canon = ad.canonical_rc_for_state(str(zone_type), s, rotation) or divmod(s, ad.N_COLS)
            acc[canon] += run["I_s"][s]
            count[canon] += 1
    if (count == 0).any():
        raise SystemExit("some canonical cell received no value")
    mean = acc / count
    # The goal is terminal (no transition out of it): the measure still returns a value there, which the campaign's
    # analyses leave out (campaign_analysis.zone_mask); left out here too, drawn as an empty cell. It never moves.
    mean[divmod(int(e3_env.goal_state()), ad.N_COLS)] = np.nan
    return mean


def outline(ax, zone_grid: np.ndarray) -> None:
    rows, cols = zone_grid.shape
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and zone_grid[r, c] != zone_grid[r, c + 1]:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color="black", lw=0.5)
            if r + 1 < rows and zone_grid[r, c] != zone_grid[r + 1, c]:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color="black", lw=0.5)


def main() -> None:
    ad = load_experiment_module("E2_online_4zones", "analysis_data.py", "paper_e2_analysis_data")
    e3_env = src.e3_module("env.py")
    runs = load_runs()
    env = canonical_mean([r for r in runs if r["condition"] == "experimental"], ad, e3_env)
    ref = canonical_mean([r for r in runs if r["condition"] == "reference"], ad, e3_env)
    zone_grid = ad.zone_of_state_for_rotation(0).reshape(ad.GRID_SHAPE)

    common.apply_style()
    fig, axes = common.figure(width_fraction=1.0, aspect=0.36, ncols=3)
    vmax = float(max(np.nanmax(env), np.nanmax(ref)))
    dmax = float(np.nanmax(np.abs(env - ref)))
    panels = ((env, "environment", "Blues", 0.0, vmax), (ref, "reference", "Blues", 0.0, vmax),
              (env - ref, r"$\hat b(s)$ = env $-$ ref", "RdBu_r", -dmax, dmax))
    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad("white")  # the goal
        im = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
        ax.text(*divmod(int(e3_env.goal_state()), ad.N_COLS)[::-1], "G", ha="center", va="center",
                fontsize=common.SMALL_FONT_SIZE - 2)
        outline(ax, zone_grid)
        ax.set_title(title, fontsize=common.SMALL_FONT_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=common.SMALL_FONT_SIZE - 1)
        cb.set_label("bit", fontsize=common.SMALL_FONT_SIZE - 1)
    for zone_type, abbr in src.ZONE_ABBR.items():  # zone labels at each type's canonical block centre
        rr, cc = np.nonzero(zone_grid == zone_type)
        axes[0].text(cc.mean(), rr.mean(), abbr, ha="center", va="center", fontsize=common.SMALL_FONT_SIZE,
                     color="white" if zone_type.startswith("controllable") else "black")
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
