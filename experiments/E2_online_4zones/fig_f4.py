"""F4 -- spatial maps of the null model at the final checkpoint, averaged
over the reference condition's 20 seeds. Every map is built on the
CANONICAL (rotation=0) grid: a state's value is read from whichever
seed/rotation actually produced it, then placed at that zone type's
canonical position (analysis_data.py::canonical_rc_for_state) -- "agrege
par type de zone, jamais par quadrant" (corridor cells never move, so
they need no remapping).

Confirmed against the sweep's own summary.json (2026 post-pilot review):
the reference condition's 20 seeds DO span all 4 counterbalancing
rotations (5 seeds each, seed_index // 5 == assignment, per
env.py::rotation_for_seed_index) -- this is not a single-rotation view
and not a protocol deviation. Every map below therefore shows the FOUR
DISTINCT ZONE TYPES occupying their four fixed canonical quadrants
(TYPE_TO_CANONICAL_QUADRANT), each value averaged over all 20 seeds
after remapping that seed's own physical quadrant back onto its type's
canonical one -- four distinct regions are the EXPECTED appearance of
this transform (one region per zone type), not four raw, un-remapped
physical quadrants that counterbalancing should have made identical.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import figures as fg

ALPHA_REF, EPSILON_REF = 1.0, 0.1
FINAL_T = ad.FINAL_CHECKPOINT


def _canonical_grid_average(per_state_values_by_run: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """per_state_values_by_run: list of (values[state], rotation) pairs,
    one per seed. Returns a (12,12) grid of the mean over seeds, at each
    canonical cell (NaN where no data was ever remapped there -- should
    not happen for a full 144-state layout, kept as a safety net)."""
    acc = np.zeros(ad.GRID_SHAPE)
    count = np.zeros(ad.GRID_SHAPE)
    zone_of_state_by_rotation = {r: ad.zone_of_state_for_rotation(r) for r in range(4)}
    for values, rotation in per_state_values_by_run:
        zone_of_state = zone_of_state_by_rotation[rotation]
        for s in range(ad.N_STATES):
            if s == ad.GOAL_S:
                continue
            zone_type = zone_of_state[s]
            canon = ad.canonical_rc_for_state(zone_type, s, rotation)
            if canon is None:  # corridor: fixed position
                canon = divmod(s, ad.N_COLS)
            acc[canon] += values[s]
            count[canon] += 1
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, acc / np.maximum(count, 1), np.nan)


def _load_reference_runs() -> list[dict]:
    run_dirs = [d for d in ad.all_run_dirs() if d.name.startswith(f"{ad.REFERENCE_CONDITION}_seed")]
    out = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        with np.load(run_dir / "arrays.npz") as arrays:
            N_final = arrays[f"N_checkpoint_{FINAL_T}"]
            N_start_final = arrays[f"N_start_checkpoint_{FINAL_T}"]
        out.append({"rotation": summary["assignment"], "N": N_final, "N_start": N_start_final, "seed": summary["seed"]})
    return out


def _zone_contours(ax) -> None:
    """Draws the canonical zone boundaries (block edges + corridor) for
    orientation on every map."""
    for key, (rows, cols) in ad._BLOCK_RANGES.items():
        ax.add_patch(plt.Rectangle(
            (cols[0] - 0.5, rows[0] - 0.5), cols[1] - cols[0] + 1, rows[1] - rows[0] + 1,
            fill=False, edgecolor="black", linewidth=1.2,
        ))
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axhline(6, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(6, color="black", linewidth=0.6, alpha=0.5)


def _plot_map(grid: np.ndarray, title: str, name: str, cmap: str = "gray", vmin=None, vmax=None, center_zero=False) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if center_zero:
        m = np.nanmax(np.abs(grid)) if vmax is None else vmax
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-m, vmax=m)
    else:
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    _zone_contours(ax)
    ax.set_title(f"{title}\n(canonical zone-type frame, N=20 seeds, 4 rotations)", fontsize=10)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fg.save_figure(fig, name)

    rows = []
    for r in range(ad.N_ROWS):
        for c in range(ad.N_COLS):
            rows.append({"row": r, "col": c, "value": grid[r, c]})
    fg.save_csv(rows, ["row", "col", "value"], name)


def make_f4() -> None:
    runs = _load_reference_runs()

    per_state = ad.load_per_state([ad.SWEEP_ROOT / f"{ad.REFERENCE_CONDITION}_seed{r['seed']:02d}" for r in runs])

    # N(s), N_start share, min count per state -- from arrays.npz directly
    n_total_by_run = [(r["N"].sum(axis=(1, 2)).astype(float), r["rotation"]) for r in runs]
    n_start_share_by_run = []
    n_min_by_run = []
    for r in runs:
        N_pair = r["N"].sum(axis=2)  # (S, A)
        N_state = N_pair.sum(axis=1)  # (S,)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(N_state > 0, r["N_start"].sum(axis=1) / np.maximum(N_state, 1), np.nan)
        n_start_share_by_run.append((share, r["rotation"]))
        n_min_by_run.append((N_pair.min(axis=1).astype(float), r["rotation"]))

    _plot_map(_canonical_grid_average(n_total_by_run), "N(s), seed mean (final checkpoint)", "F4_N_total", cmap="gray_r")
    _plot_map(_canonical_grid_average(n_start_share_by_run), "Share of N(s) from episode starts", "F4_N_start_share", cmap="gray_r", vmin=0, vmax=1)
    _plot_map(_canonical_grid_average(n_min_by_run), "Minimum count per state (over actions)", "F4_N_min_per_state", cmap="gray_r")

    # b_hat online, Delta online, and the online-matched diffs, at the
    # final checkpoint, per state -- from per_state.csv (already
    # rotation-correct per row; remap for canonical display).
    def per_state_values(condition: str, field: str) -> list[tuple[np.ndarray, int]]:
        out = []
        for r in runs:
            mask = (
                (per_state["condition"] == condition) & (per_state["checkpoint"] == FINAL_T)
                & (per_state["seed"] == r["seed"])
            )
            values = np.full(ad.N_STATES, np.nan)
            values[per_state["state"][mask]] = per_state[field][mask]
            values[ad.GOAL_S] = 0.0  # excluded from measures, but needed as a placeholder for the grid
            out.append((values, r["rotation"]))
        return out

    b_online = per_state_values("online", "b")
    delta_online = per_state_values("online", "delta")
    b_matched = per_state_values("matched", "b")
    delta_matched = per_state_values("matched", "delta")

    _plot_map(_canonical_grid_average(b_online), "b̂ online (final checkpoint)", "F4_b_online", center_zero=True)
    _plot_map(_canonical_grid_average(delta_online), "Δ online (final checkpoint)", "F4_delta_online", cmap="gray_r")

    b_diff = [(vo[0] - vm[0], vo[1]) for vo, vm in zip(b_online, b_matched)]
    delta_diff = [(vo[0] - vm[0], vo[1]) for vo, vm in zip(delta_online, delta_matched)]
    _plot_map(_canonical_grid_average(b_diff), "b̂ (online − matched, final checkpoint)", "F4_b_diff", center_zero=True)
    _plot_map(_canonical_grid_average(delta_diff), "Δ (online − matched, final checkpoint)", "F4_delta_diff", center_zero=True)

    # Delta exterior only (corridor -- the only region outside the 4
    # blocks), tightened color scale -- "c'est la que se lit le plancher
    # de propagation."
    delta_diff_grid = _canonical_grid_average(delta_diff)
    exterior_mask = np.zeros(ad.GRID_SHAPE, dtype=bool)
    exterior_mask[0, :] = True
    exterior_mask[6, :] = True
    exterior_mask[:, 0] = True
    exterior_mask[:, 6] = True
    delta_ext = np.where(exterior_mask, delta_diff_grid, np.nan)
    ext_vals = delta_ext[exterior_mask & ~np.isnan(delta_ext)]
    ext_scale = float(np.nanmax(np.abs(ext_vals))) if len(ext_vals) else 1.0
    _plot_map(delta_ext, "Δ (online − matched), exterior only (tightened scale)", "F4_delta_diff_exterior", center_zero=True, vmax=ext_scale)
