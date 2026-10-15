"""Controllability of a frozen world model along fixed walks (E4's zero-data probes, docs/E4_generalization_gradient.md).
New module (2026-10-15): nothing of probe.py is changed; the matching rule and the controllability are E3's own
(measures.kernel_probe, pipeline.measure.controllability_of_kernel).

1. fixed_trajectory_context: the model filters each walk (probing/walks.py) -- the image of every visited state and
   the action taken -- with its frozen weights (encoder, posterior, recurrent model), from its learned initial state
   (is_first at the walk's first step) or from a given state (`initial`). Every context (h_t, z_t) comes from the walk
   itself, never from a training buffer. The posterior draws are plain straight-through samples from `generator`
   (as probe.real_context_probe).
2. step_controllability: at a context, for every action a, h' = GRU(h_t, z_t, a) and the prior p(z' | h') (the
   model's one-step prediction, as E3's main measure); `draws` samples z' per action, decoded as decode(h', z') and
   matched to the templates of the environment's states (measures.kernel_probe: nearest template, ghost when the margin
   is below margin_threshold); the kernel p_hat(s' | context, a) over the non-ghost draws; I_s of the context =
   controllability_of_kernel (NaN when an action has only ghosts). Identical (context, action, code) draws are decoded
   once and counted with their multiplicity: the decoder is deterministic, so this is the same estimate as decoding
   every draw, much cheaper when the prior is peaked.
3. summarize_zone: per-step values -> by state, then by zone: the mean over the first m steps in the zone (the entry
   measure), and the curve over steps since entry 1..n_in, with the number of distinct states behind every value.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.loop import encode_windows
from helplessness.agents.worldmodel.rssm import RSSM, sample_categorical_index
from helplessness.measures.kernel_probe import match_from_distances


def template_distances_torch(decoded: torch.Tensor, templates_flat64: torch.Tensor, templates_sq: torch.Tensor) -> np.ndarray:
    """measures.kernel_probe.template_distances on the model's device: the same formula, |a|^2 + |b|^2 - 2 a.b in
    float64, clipped at 0 -- the distance computation dominates the kernel's cost and runs on the GPU this way."""
    flat = decoded.reshape(decoded.shape[0], -1).to(torch.float64)
    d = (flat**2).sum(dim=1)[:, None] + templates_sq[None, :] - 2.0 * (flat @ templates_flat64.T)
    return torch.clamp(d, min=0.0).cpu().numpy()


def fixed_trajectory_context(
    rssm: RSSM, images: np.ndarray, walks: dict, generator: torch.Generator | None = None, chunk: int = 64,
    initial: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None, max_images: int | None = None,
) -> dict[str, torch.Tensor]:
    """The model's contexts along the walks: h (n_walks, T, h_dim) and z (n_walks, T, n_vars, n_classes), on the
    model's device. images: (n_states, H, W, C) uint8, the image of every state in the probed environment (the pixel
    table); walks: probing.walks.generate_walks output. Steps past a walk's end are computed on padding and must be
    ignored (walks["phase"] == PHASE_PAD). initial: None = the learned initial state at step 0 (is_first); else
    (h, z, a_prev_onehot), each (n_walks, ...), the state the filter starts from (no is_first)."""
    device = next(rssm.parameters()).device
    states, actions = walks["states"], walks["actions"]
    n_walks, T = states.shape
    hs = torch.zeros(n_walks, T, rssm.cfg.h_dim, device=device)
    zs = torch.zeros(n_walks, T, rssm.cfg.n_vars, rssm.cfg.n_classes, device=device)
    with torch.no_grad():
        for first in range(0, n_walks, chunk):
            sl = slice(first, min(first + chunk, n_walks))
            x = images[np.clip(states[sl], 0, None)]
            embeds = encode_windows(rssm, x, device, max_images)
            a_onehot = F.one_hot(torch.as_tensor(np.clip(actions[sl], 0, None), device=device), num_classes=rssm.cfg.n_actions)
            a_onehot = a_onehot.to(embeds.dtype)
            batch = sl.stop - sl.start
            if initial is None:
                h, z = rssm.initial_state(batch, device)
                a_prev = torch.zeros(batch, rssm.cfg.n_actions, device=device, dtype=embeds.dtype)
                first_flag = torch.ones(batch, dtype=torch.bool, device=device)
            else:
                h, z, a_prev = (v[sl].to(device) for v in initial)
                first_flag = torch.zeros(batch, dtype=torch.bool, device=device)
            not_first = torch.zeros(batch, dtype=torch.bool, device=device)
            for t in range(T):
                out = rssm.observation_step(h, z, a_prev, None, first_flag if t == 0 else not_first, generator=generator,
                                            embed=embeds[:, t])
                h, z = out["h"], out["z"]
                hs[sl, t], zs[sl, t] = h, z
                a_prev = a_onehot[:, t]
    return {"h": hs, "z": zs}


def step_controllability(
    rssm: RSSM, h: torch.Tensor, z: torch.Tensor, templates_t: torch.Tensor, draws: int, margin_threshold: float,
    generator: torch.Generator | None = None, true_successor: np.ndarray | None = None, chunk: int = 512,
    decode_chunk: int = 1024,
) -> dict[str, np.ndarray]:
    """I_s at each context (h (n, h_dim), z (n, n_vars, n_classes)): module docstring, step 2. templates_t:
    (n_states, C, H, W) float, the probed environment's state images. true_successor (n, n_actions) int, optional: the
    environment's successor of each (context's state, action), for the mass on it. Returns I (n,), ghost_fraction
    (n, n_actions), mass_true (n, n_actions) (NaN without true_successor), n_unique_decodes (int)."""
    device = next(rssm.parameters()).device
    from helplessness.pipeline.measure import controllability_of_kernel  # pipeline imports probing, not the reverse

    n, A = h.shape[0], rssm.cfg.n_actions
    V, C = rssm.cfg.n_vars, rssm.cfg.n_classes
    n_states = templates_t.shape[0]
    templates_flat64 = templates_t.reshape(n_states, -1).to(device=device, dtype=torch.float64)
    templates_sq = (templates_flat64**2).sum(dim=1)
    powers = torch.as_tensor(C ** np.arange(V), device=device, dtype=torch.long)
    p_hat = np.full((n, A, n_states), np.nan)
    ghost = np.full((n, A), np.nan)
    n_unique = 0
    with torch.no_grad():
        for first in range(0, n, chunk):
            sl = slice(first, min(first + chunk, n))
            m = sl.stop - sl.start
            h_next = torch.zeros(m, A, rssm.cfg.h_dim, device=device)
            prior = torch.zeros(m, A, V, C, device=device)
            for a in range(A):
                a_onehot = F.one_hot(torch.full((m,), a, device=device, dtype=torch.long), num_classes=A).to(h.dtype)
                out = rssm.imagination_step(h[sl], z[sl], a_onehot, generator=generator)
                h_next[:, a], prior[:, a] = out["h"], out["prior_probs"]
            rows = m * A
            idx = sample_categorical_index(prior.reshape(rows, 1, V, C).expand(rows, draws, V, C).reshape(rows * draws, V, C), generator)
            codes = (idx * powers).sum(-1).reshape(rows, draws)                      # one integer per drawn z'
            keys = torch.arange(rows, device=device)[:, None] * (C ** V) + codes    # (row, code)
            uniq, inverse = torch.unique(keys.reshape(-1), return_inverse=True)
            n_unique += int(uniq.numel())
            u_row = (uniq // (C ** V)).to(torch.long)
            u_code = uniq % (C ** V)
            u_idx = (u_code[:, None] // powers[None, :]) % C                        # (U, V) class of each variable
            nearest = np.empty(uniq.numel(), dtype=np.int64)
            is_ghost = np.empty(uniq.numel(), dtype=bool)
            h_rows = h_next.reshape(rows, -1)
            for d0 in range(0, uniq.numel(), decode_chunk):
                ds = slice(d0, min(d0 + decode_chunk, uniq.numel()))
                z_u = F.one_hot(u_idx[ds], num_classes=C).to(h.dtype)
                decoded = rssm.decode(h_rows[u_row[ds]], z_u)
                matched = match_from_distances(template_distances_torch(decoded, templates_flat64, templates_sq), margin_threshold)
                nearest[ds], is_ghost[ds] = matched["nearest_state"], matched["is_ghost"]
            inv = inverse.cpu().numpy().reshape(rows, draws)
            near_d, ghost_d = nearest[inv], is_ghost[inv]                            # (rows, draws) per draw
            ok = ~ghost_d
            counts = np.zeros((rows, n_states))
            np.add.at(counts, (np.repeat(np.arange(rows), draws)[ok.reshape(-1)], near_d.reshape(-1)[ok.reshape(-1)]), 1.0)
            n_ok = ok.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                block = np.where(n_ok[:, None] > 0, counts / n_ok[:, None], np.nan)
            p_hat[sl] = block.reshape(m, A, n_states)
            ghost[sl] = (1.0 - n_ok / draws).reshape(m, A)
    I = controllability_of_kernel(p_hat)
    mass = np.full((n, A), np.nan)
    if true_successor is not None:
        mass = np.take_along_axis(p_hat, np.asarray(true_successor)[:, :, None], axis=2)[:, :, 0]
    return {"I": I, "ghost_fraction": ghost, "mass_true": mass, "n_unique_decodes": n_unique}


def summarize_zone(values: np.ndarray, states: np.ndarray, steps_since_entry: np.ndarray, m: int, n_in: int) -> dict:
    """values, states, steps_since_entry: one entry per measured step (steps in the zone of interest only). Returns
    entry: mean over states of each state's mean over the steps with 1 <= steps_since_entry <= m; curve (n_in,): the
    same at each step since entry k = 1..n_in; n_states_entry and n_states_curve (n_in,): distinct states behind them
    (states whose value is NaN, an all-ghost action, are left out and not counted)."""
    values, states, k = np.asarray(values, dtype=np.float64), np.asarray(states), np.asarray(steps_since_entry)
    finite = np.isfinite(values)

    def by_state_then_zone(mask: np.ndarray) -> tuple[float, int]:
        mask = mask & finite
        if not mask.any():
            return float("nan"), 0
        uniq, inv = np.unique(states[mask], return_inverse=True)
        per_state = np.bincount(inv, weights=values[mask]) / np.bincount(inv)
        return float(per_state.mean()), int(uniq.size)

    entry, n_entry = by_state_then_zone((k >= 1) & (k <= m))
    curve, n_curve = np.full(n_in, np.nan), np.zeros(n_in, dtype=np.int64)
    for step in range(1, n_in + 1):
        curve[step - 1], n_curve[step - 1] = by_state_then_zone(k == step)
    return {"entry": entry, "n_states_entry": n_entry, "curve": curve, "n_states_curve": n_curve}


def measure_walks(rssm: RSSM, images: np.ndarray, templates_t: torch.Tensor, walks: dict, successor: np.ndarray, draws: int,
                  margin_threshold: float, seed: int, m: int, n_in: int, chunk: int = 64, max_images: int | None = None) -> dict:
    """The whole measure of one (image, zone of interest, model): the model filters the walks (fixed_trajectory_context),
    I_s at every step in the zone of interest (step_controllability), and its summary (summarize_zone). images:
    (n_states, H, W, C) uint8 and templates_t (n_states, C, H, W) float, the probed environment's state images;
    successor (n_states, n_actions): its true successors (the mass on them); seed: the measurement's torch generator
    (the same for the two models of a pair, so they see the same draws wherever their probabilities agree).
    Returns the per-step arrays (I, states, steps_since_entry, steps_since_start, walk, ghost_fraction, mass_true) and
    the summary."""
    from helplessness.probing.walks import PHASE_ZONE

    device = next(rssm.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    ctx = fixed_trajectory_context(rssm, images, walks, generator=generator, chunk=chunk, max_images=max_images)
    mask = walks["phase"] == PHASE_ZONE
    mask_t = torch.as_tensor(mask, device=device)
    states = walks["states"][mask]
    res = step_controllability(rssm, ctx["h"][mask_t], ctx["z"][mask_t], templates_t.to(device), draws, margin_threshold,
                               generator=generator, true_successor=successor[states])
    summary = summarize_zone(res["I"], states, walks["steps_since_entry"][mask], m, n_in)
    return {"I": res["I"], "states": states, "steps_since_entry": walks["steps_since_entry"][mask],
            "steps_since_start": walks["steps_since_start"][mask], "walk": np.nonzero(mask)[0],
            "ghost_fraction": res["ghost_fraction"], "mass_true": res["mass_true"], "n_unique_decodes": res["n_unique_decodes"],
            **summary}
