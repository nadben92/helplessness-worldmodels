"""Whether a zero-data probe is interpretable (E4's final analysis, results/E4_generalization_gradient/final/REPORT.md;
part of E5's protocol, docs/e5/DECISIONS.md (22)), and why not. Moved from
experiments/E4_generalization_gradient/final_analysis.py on 2026-10-21 (same computations) so that E4's analysis and E5's
probes share them.

  validity     a probe's b is interpretable only if the REFERENCE model predicts the transitions at the first step in
               the zone of interest: its prior's mass on the true successor >= MASS_MIN and |I_ref - log2 5| <= I_TOL, both
               at step 1 (by state, then by zone; the mass and the ghosts averaged over the actions). step1_summary reads
               them from one model's probing.fixed_trajectory.measure_walks result.
  diagnostic   along the fixed walks, the model's reconstruction of every observed frame (its posterior decoded) against
               the true frame: the error (in [0, 1] pixel units) overall and outside the zone of interest; the agent's
               decoded position, by the kernel's rule (the state whose image is nearest to the reconstruction -- biased
               towards the states inside a quadrant whose colour the reconstruction does not render) and directly (the cell
               holding the pixel nearest to the agent's colour); the mean colour the reconstruction gives the zone of
               interest (the agent's cell left out) and its distance to the probe's colour and to named reference colours
               (E4: the slot's training colour; E5: every colour the slot takes in training -- under shuffle a slot is not
               tied to one colour). In the corridor, in the zone, at the zone's first step.

Arrays out; the forward passes run on the model's device (probing/ is where torch code measures).
"""

from __future__ import annotations

import numpy as np
import torch

from helplessness.envs.layouts import two_zones
from helplessness.probing import walks as walks_mod
from helplessness.probing.fixed_trajectory import fixed_trajectory_context, template_distances_torch

LOG2_5 = float(np.log2(5))
MASS_MIN = 0.90
I_TOL = 0.15
PARTS = ("corridor", "zone", "zone_step1")


def by_state_then_zone(values, states) -> float:
    """The mean over states of each state's mean (NaN values left out)."""
    values, states = np.asarray(values, dtype=np.float64), np.asarray(states)
    ok = np.isfinite(values)
    if not ok.any():
        return float("nan")
    uniq, inv = np.unique(states[ok], return_inverse=True)
    return float((np.bincount(inv, weights=values[ok]) / np.bincount(inv)).mean())


def step1_summary(res: dict) -> dict:
    """mass, ghost (both averaged over the actions) and I at the first step in the zone, by state then by zone, and the
    states behind them, from a measure_walks result."""
    k = np.asarray(res["steps_since_entry"])
    sel = k == 1
    states = np.asarray(res["states"])[sel]
    out = {}
    for field, key in (("mass_true", "mass"), ("ghost_fraction", "ghost"), ("I", "I")):
        values = np.asarray(res[field], dtype=np.float64)
        if values.ndim == 2:  # per action: the mean over the actions, as the training diagnostic does
            values = values.mean(axis=1)
        out[f"{key}_step1"] = by_state_then_zone(values[sel], states)
    out["n_states_step1"] = int(np.unique(states).size)
    return out


def is_valid(mass_ref_step1: float, I_ref_step1: float, mass_min: float = MASS_MIN, i_tol: float = I_TOL) -> bool:
    return bool(mass_ref_step1 >= mass_min and abs(I_ref_step1 - LOG2_5) <= i_tol)


def toric_cells(a, b) -> np.ndarray:
    """Toric Manhattan distance, in cells, between states."""
    R, C = two_zones.GRID_SHAPE
    ra, ca = np.divmod(np.asarray(a), C)
    rb, cb = np.divmod(np.asarray(b), C)
    dr, dc = np.abs(ra - rb), np.abs(ca - cb)
    return np.minimum(dr, R - dr) + np.minimum(dc, C - dc)


def decoded_agent_cell(dec: torch.Tensor, agent: torch.Tensor, kpx: int) -> np.ndarray:
    """The agent's decoded position, directly: the cell holding the pixel nearest to the agent's colour. dec: (n, 3, H, W)
    reconstructions clamped to [0, 1]; agent: (3, 1, 1) the agent's colour in [0, 1]; kpx: pixels per cell. (n,) state
    indices. (Split out of reconstruction_diagnostic on 2026-10-23 for E6's per-step localisation, same operations.)"""
    R, C = two_zones.GRID_SHAPE
    d_agent = ((dec - agent) ** 2).sum(dim=1)
    per_cell = d_agent.reshape(-1, R, kpx, C, kpx).amin(dim=(2, 4)).reshape(-1, R * C)
    return per_cell.argmin(dim=1).cpu().numpy()


def reconstruction_diagnostic(model, world, walks: dict, slot: str, probe_rgb, reference_colors: dict, agent_color,
                              generator: torch.Generator, batch: int = 1024) -> list[dict]:
    """One row per part (PARTS) of the walks: n_frames, recon_mse, recon_mse_outside_zone, nearest_template_accuracy /
    _error_cells, agent_cell_accuracy / _error_cells, zone_rgb_reconstructed, zone_rgb_dist_to_probe and, for every
    name of reference_colors (name -> RGB), zone_rgb_dist_to_<name>, and zone_rgb_nearest (the nearest of the probe's
    colour and the reference colours). world: the probe environment (templates_np, templates_t, layout)."""
    device = next(model.parameters()).device
    tf = world.templates_t.reshape(world.templates_t.shape[0], -1).double().to(device)
    tsq = (tf ** 2).sum(1)
    R, C = two_zones.GRID_SHAPE
    kpx = world.templates_t.shape[-1] // C
    agent = torch.tensor(agent_color, dtype=torch.float32, device=device)[:, None, None] / 255.0
    cells = np.zeros((R, C), dtype=bool)
    for st in two_zones.region_states(world.layout)[slot]:
        cells[divmod(int(st), C)] = True
    zone_px = torch.as_tensor(np.kron(cells, np.ones((kpx, kpx), dtype=bool)), device=device)
    valid = walks["phase"] != walks_mod.PHASE_PAD
    states = walks["states"][valid]
    k = walks["steps_since_entry"][valid]
    with torch.no_grad():
        ctx = fixed_trajectory_context(model, world.templates_np, walks, generator=generator)
        mask_t = torch.as_tensor(valid, device=ctx["h"].device)
        h, z = ctx["h"][mask_t], ctx["z"][mask_t]
        templates = world.templates_t.to(device)
        nearest, mse, mse_out, agent_cell, zone_rgb = [], [], [], [], []
        for i in range(0, h.shape[0], batch):
            dec = model.decode(h[i:i + batch], z[i:i + batch])
            nearest.append(template_distances_torch(dec, tf, tsq).argmin(axis=1))
            dec = dec.clamp(0, 1)
            st = torch.as_tensor(states[i:i + batch].astype(np.int64), device=device)
            err = ((dec - templates[st]) ** 2).mean(dim=1)
            mse.append(err.mean(dim=(1, 2)).cpu().numpy())
            mse_out.append(err[:, ~zone_px].mean(dim=1).cpu().numpy())
            agent_cell.append(decoded_agent_cell(dec, agent, kpx))
            n = dec.shape[0]
            own = torch.zeros(n, R * C, dtype=torch.bool, device=device)
            own[torch.arange(n, device=device), st] = True
            own_px = own.reshape(n, R, 1, C, 1).expand(n, R, kpx, C, kpx).reshape(n, R * kpx, C * kpx)
            pix = zone_px[None] & ~own_px
            rgb = (dec * pix[:, None]).sum(dim=(2, 3)) / pix.sum(dim=(1, 2))[:, None]
            zone_rgb.append((rgb * 255.0).cpu().numpy())
    nearest, mse, mse_out = np.concatenate(nearest), np.concatenate(mse), np.concatenate(mse_out)
    agent_cell, zone_rgb = np.concatenate(agent_cell), np.concatenate(zone_rgb)
    dist, dist_agent = toric_cells(nearest, states), toric_cells(agent_cell, states)
    named = {"probe": np.asarray(probe_rgb, dtype=float), **{n: np.asarray(c, dtype=float) for n, c in reference_colors.items()}}
    rows = []
    for part, sel in (("corridor", k == 0), ("zone", k >= 1), ("zone_step1", k == 1)):
        rgb = zone_rgb[sel].mean(axis=0)
        d = {n: float(np.linalg.norm(rgb - c)) for n, c in named.items()}
        rows.append({"part": part, "n_frames": int(sel.sum()), "recon_mse": float(mse[sel].mean()),
                     "recon_mse_outside_zone": float(mse_out[sel].mean()),
                     "nearest_template_accuracy": float((nearest[sel] == states[sel]).mean()),
                     "nearest_template_error_cells": float(dist[sel].mean()),
                     "agent_cell_accuracy": float((agent_cell[sel] == states[sel]).mean()),
                     "agent_cell_error_cells": float(dist_agent[sel].mean()),
                     "zone_rgb_reconstructed": "-".join(f"{v:.0f}" for v in rgb),
                     **{f"zone_rgb_dist_to_{n}": v for n, v in d.items()}, "zone_rgb_nearest": min(d, key=d.get)})
    return rows
