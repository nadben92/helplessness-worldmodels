"""E5's checkpoint measures of a run trained under shuffled zone assignments (docs/E5_belief_training.md), the same two
measures as E4's (pipeline.two_zones_records), taken assignment by assignment:

  real_context  E3's real contexts (probe.real_context_probe), drawn from the replay steps of ONE assignment (the replay
                rows of that assignment's block: AssignmentReplayView), their one-step predictions matched against THAT
                assignment's state images and scored against its successors: the prior's mass on the true successor
                and I_s, by region; up to max_per_state contexts per (assignment, state);
  walks         E4's fixed walks into each slot on the assignment's image in its probe environment (the reference's
                dynamics, every slot controllable): I at the first step in the zone and the curve, by slot.

Both by slot and by zone type (the slot's role in that assignment: "ud", "cd", "neutral"; the corridor). The canonical
assignment's walks use E4's own measurement seed (image index 0), every other assignment its own (image index
ASSIGNMENT_IMAGE_BASE + k), the same for a run and its reference. Also: the share of the stored steps by zone type
(visits), and the counts of episodes by assignment so far. Own measurement streams; nothing touches the training.
"""

from __future__ import annotations

import numpy as np
import torch

from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.episode_sources import ASSIGNMENTS, assignment_roles
from helplessness.pipeline.measure import controllability_of_kernel, real_context_settings
from helplessness.pipeline.runtime import max_images_per_forward
from helplessness.pipeline.two_zones_records import region_means
from helplessness.probing import probe as e3_probe
from helplessness.probing.fixed_trajectory import measure_walks
from helplessness.streams import worldmodel as wm_streams

ZONE_TYPES = ("corridor", "ud", "cd", "neutral")
ASSIGNMENT_IMAGE_BASE = 1000  # measurement-seed image index of assignment k (k != canonical): 1000 + k
N_STATES = two_zones.N_STATES


def _nanmean(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


class AssignmentReplayView:
    """The replay as real_context_probe reads it, labelled by table row (assignment k, state s -> k * n_states + s)
    instead of by state, the rows outside `keep` relabelled n_rows (never selected). Measurements only."""

    def __init__(self, replay, keep_assignments, n_states: int = N_STATES):
        self.replay, self.n_states = replay, int(n_states)
        self.n_rows = len(ASSIGNMENTS) * self.n_states
        self.keep = np.zeros(self.n_rows, dtype=bool)
        for k in keep_assignments:
            self.keep[k * self.n_states:(k + 1) * self.n_states] = True

    def labels_and_flags(self) -> tuple[np.ndarray, np.ndarray]:
        rows, first = self.replay.obs_rows_and_flags()
        return np.where(self.keep[rows], rows, self.n_rows), first

    def windows(self, logical_starts, seq_len):
        return self.replay.windows(logical_starts, seq_len)


def _region_to_type(roles: dict[str, str]) -> dict[str, str]:
    return {"corridor": "corridor", **{q: roles[q] for q in QUADRANT_ORDER}}


def by_type(values_by_region: dict[str, float], roles: dict[str, str]) -> dict[str, float]:
    """Region -> value into zone type -> value (the two neutral slots averaged)."""
    out: dict[str, list] = {t: [] for t in ZONE_TYPES}
    for region, t in _region_to_type(roles).items():
        out[t].append(values_by_region[region])
    return {t: _nanmean(v) for t, v in out.items()}


def real_context_by_assignment(rssm, replay, worlds: dict[int, object], cfg: dict, seed: int, condition: str, env_step: int,
                               max_per_state: int, draws: int, slot: int) -> dict[int, dict]:
    """worlds: assignment -> its World (the run's dynamics, templates on the model's device)."""
    seq = wm_streams.spawn_sequences(seed, condition)["measure"].spawn(wm_streams.N_MEASURE_SLOTS)[slot]
    child = np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(env_step)))
    device = next(rssm.parameters()).device
    rng, gen = np.random.default_rng(child), wm_streams.torch_generator(child, device)
    rc = real_context_settings(cfg)
    view = AssignmentReplayView(replay, sorted(worlds))
    ctx = e3_probe.real_context_probe(rssm, view, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=max_per_state,
                                      n_states=view.n_rows, rng=rng, generator=gen, max_images=max_images_per_forward(cfg))
    out = {}
    for k in sorted(worlds):
        world = worlds[k]
        idx = np.flatnonzero(ctx["states"] // N_STATES == k)
        idx_t = torch.as_tensor(idx, device=device)
        sub = {"states": ctx["states"][idx] - k * N_STATES, "n_states": N_STATES, "prior_probs_t": ctx["prior_probs_t"][idx_t],
               "h_next": ctx["h_next"][idx_t]}
        kernel = e3_probe.real_context_kernel(rssm, sub, world.templates_t, draws=draws, margin_threshold=cfg["measure"]["kernel_margin_threshold"],
                                              generator=gen)
        p_hat = kernel["p_hat"]
        succ = two_zones_setup.successors(world)
        mass = np.take_along_axis(p_hat, succ[:, :, None], axis=2)[:, :, 0].mean(axis=1)
        I_s = controllability_of_kernel(p_hat)
        roles = assignment_roles(k)
        mass_r, I_r = region_means(mass, world.layout), region_means(I_s, world.layout)
        out[k] = {"mass_true_by_region": mass_r, "I_s_by_region": I_r, "mass_true_by_type": by_type(mass_r, roles),
                  "I_s_by_type": by_type(I_r, roles), "n_contexts": int(idx.size), "n_states_measured": int(np.isfinite(I_s).sum()),
                  "ghost_fraction_mean": float(np.nanmean(kernel["ghost_fraction"])) if np.isfinite(kernel["ghost_fraction"]).any() else float("nan")}
    return out


def walks_by_assignment(rssm, cfg: dict, conf: dict, probe_worlds: dict[int, object], canonical: int, n_walks: int, purpose: str) -> dict[int, dict]:
    """probe_worlds: assignment -> its probe World (the reference's dynamics, the assignment's image)."""
    w, draws = cfg["e4"]["walks"], cfg["e4"]["probe"]["draws"]
    out = {}
    for k in sorted(probe_worlds):
        pw = probe_worlds[k]
        succ = two_zones_setup.successors(pw)
        image_index = 0 if k == canonical else ASSIGNMENT_IMAGE_BASE + k
        slots = {}
        for slot in QUADRANT_ORDER:
            res = measure_walks(rssm, pw.templates_np, pw.templates_t, two_zones_setup.walks_for_slot(cfg, slot, n_walks), succ, draws,
                                cfg["measure"]["kernel_margin_threshold"], two_zones_setup.measurement_seed(cfg, conf, image_index, slot, purpose),
                                w["m"], w["n_in"], chunk=w["chunk"], max_images=max_images_per_forward(cfg))
            slots[slot] = {"I_entry": res["entry"], "n_states_entry": res["n_states_entry"], "I_curve": res["curve"].tolist(),
                           "n_states_curve": res["n_states_curve"].tolist(), "mass_true_mean": float(np.nanmean(res["mass_true"])),
                           "ghost_fraction_mean": float(np.nanmean(res["ghost_fraction"])), "n_walks": n_walks}
        roles = assignment_roles(k)
        step1 = {q: slots[q]["I_curve"][0] for q in QUADRANT_ORDER}
        out[k] = {"slots": slots, "I_step1_by_type": {t: _nanmean([step1[q] for q in QUADRANT_ORDER if roles[q] == t])
                                                     for t in ("ud", "cd", "neutral")}}
    return out


def type_of_row(n_states: int = N_STATES, layout=None) -> np.ndarray:
    """(12 * n_states,) the zone type of every replay row (assignment k, state s); the goal is in the corridor."""
    layout = two_zones.build_layout(1, "reference") if layout is None else layout
    region = np.array([two_zones.region_of_state(layout, s) for s in range(n_states)])
    out = np.empty(len(ASSIGNMENTS) * n_states, dtype=object)
    for k in range(len(ASSIGNMENTS)):
        roles = assignment_roles(k)
        out[k * n_states:(k + 1) * n_states] = [roles.get(r, "corridor") for r in region]
    return out


def visit_shares(rows: np.ndarray, types: np.ndarray) -> dict[str, float]:
    """The share of the stored steps by zone type. rows: replay table rows (obs_rows_and_flags)."""
    t = types[rows]
    return {z: float(np.mean(t == z)) if t.size else float("nan") for z in ZONE_TYPES}


def mean_over_assignments(per_assignment: dict[int, dict], key: str) -> dict[str, float]:
    return {t: _nanmean([v[key][t] for v in per_assignment.values()]) for t in next(iter(per_assignment.values()))[key]}
