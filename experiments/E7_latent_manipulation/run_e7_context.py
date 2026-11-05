"""E7 on all agents, report section 22: does h carry the CONTEXT (position, colour drawn), from which the weights
recompute the belief at every step, or a transported "level of control"? Frozen weights, training image, the test
environment (everything controllable), E6's protocol and seeds; every model's weights hashed before and after. The
correction is the constrained one of each agent's (48) setting (run_e7_constrained.py's sweep, the kept training-image
setting with the largest specific gain, as run_all_cpu.best_setting). Nothing of the measures is reimplemented: D7's
blocks (run_e7_apply.stepwise_hook / fixed_hook, run_e7_diagnostics.correction, lookahead.channel_q, analyse_e7_d7's
c_measures) and the episodes' own calls.

Phases (--phase), one output directory per agent (and part), <out-dir>/E7_context/e7ctx__<agent>__<phase>[__<part>]__<UTC>/:
  persist   A. E6's episodes of the exposed model with the correction applied ONCE, at t = 0; then the paired replay:
            on each episode's played path (its observations and actions), two streams of h with the same draws (cloned
            generators), corrected at t = 0 or not; per step b, C1, C3 (V and Q of channel_q, the D7 counterfactual) and
            the mass on the true successor of the played action, for both streams; and |dh_t| = |h_corr - h_raw|, its
            cos with dh_0; the policy's greedy directed share on its own Q (Q_normal of channel_q) for both streams. Also
            E6's exposed episodes (no correction, same seeds) for the directed share by index. episodes.npz, summary.json.
  natpersist  D. persist with a NATURAL direction applied once in place of the correction (--direction col: -d_col,
            a = 1; rec: +alpha * d_rec, alpha calibrated so that b's gain at t = 0 equals -d_col's), d_col and d_rec of
            the training slot from the agent's four collect runs (--collect). episodes.npz, summary.json.
  transfer  E, shuffle agents. Episodes starting in --slot in the test world drawn with each of the 3 training
            assignments where --slot carries T_inc (collect's images and seeds): exposed, and h <- h - d_col(Q1) at every
            zone step (a = 1) for each of the 4 quadrants Q1 (d_col from --collect); b at step 1, C1, C3 (localised zone
            steps), zone localisation, directed share. episodes.npz, summary.json.
  decoder   B. The first-step contexts of episodes started on the zone's interior cells (20 per cell), before and after
            the correction: the PREDICTED images (for each action, h'_a = GRU(h, z, a), 4 prior draws z', decoded) and the
            reconstruction (h, z): the zone's mean colour off the agent's cells, its RGB distance to T_inc, T_ctrl and
            the neutral (corridor) colour, the decoded agent cell. Saves the corrections d (C's cosine). decoder.npz.
  collect   C, shuffle agents. Real episodes (E6's protocol) of the exposed model in the test world drawn with each of
            the 12 training assignments (episode_sources.assignment_colors), starting in --slot: per (assignment,
            half (episode parity), cell, step class (t = 0, t >= 4, every zone step)) the sum of h and the count.
            collect.npz.
  imagine   A in imagination, paired (B's contexts; h and h + d rolled 5 imagined steps with the same actions and draws):
            per step |dh|, |dh| / |dh_0|, cos(dh, dh_0), b of both streams on zone cells. imagine.npz.
  coldir    C, shuffle agents. From the agent's four collect runs (--collect): d_col, d_ctrl, d_rec per slot and the
            split-half reliability; then, on the training image, h <- h + a * v at every zone step (run_e7_apply.fixed_hook)
            for v = -d_col (a in --amplitudes) and 5 random directions of d_col's norm (a in --random-amplitudes), with
            D7's q_fn: b at step 1, C1, C3 (localised zone steps), zone localisation, directed share. --conditions keeps
            a subset (to split an agent over processes). episodes.npz, summary.json.

Usage, from the repo root (src on PYTHONPATH):
  python experiments/E7_latent_manipulation/run_e7_context.py --phase persist --agents configs/e6/agents_e5.yaml --runs-root <root>
      --agent-ids f1_A --config configs/e7/e7_all.yaml --constrained <e7con dir> --device cpu --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import analyse_e7_d7 as d7a
import analyse_e7_diagnostics as ad
import run_e7_apply as ra
import run_e7_d7 as rd7
import run_e7_diagnostics as rdiag
import run_e7_dissociation as rdis
from helplessness.agents.worldmodel.policy import q_values
from helplessness.agents.worldmodel.rssm import sample_categorical_index
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.measures import decision_rule as dr
from helplessness.measures import zone_phenotypes as zp
from helplessness.pipeline import episode_sources as es
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.fixed_trajectory import step_controllability
from helplessness.probing.frozen_episodes import fresh_start_contexts, localise
from helplessness.probing.lookahead import channel_q
from helplessness.probing.probe_validity import decoded_agent_cell
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_context"
PHASES = ("persist", "decoder", "collect", "coldir", "imagine", "natpersist", "transfer")
NPZ = {"persist": "episodes.npz", "decoder": "decoder.npz", "collect": "collect.npz", "coldir": "episodes.npz", "imagine": "imagine.npz",
       "natpersist": "episodes.npz", "transfer": "episodes.npz"}
IMAGINE_STEPS = 5
N_COLS = two_zones.GRID_SHAPE[1]
DECODER_PER_CELL = 20
PRED_DRAWS = 4
N_RANDOM = 5
AMPLITUDES = (0.25, 0.5, 0.75, 1.0, 1.5)
RANDOM_AMPLITUDES = (1.0, 1.5)
REC_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)  # D: the grid of d_rec's calibration
STEP_CLASSES = ("first", "late", "all")  # t = 0, t >= 4 (step >= 5), every zone step
SEED = 2207  # this section's seeds (decoder contexts, replay, random directions); E6's episodes keep E6's


def gen(seed_parts: list[int], device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([SEED, *seed_parts]).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1)))


def best_setting(con_dir: Path) -> dict:
    """(48): the kept training-image setting with the largest specific gain (run_all_cpu.best_setting's rule)."""
    rows = [r for r in json.loads((con_dir / "settings.json").read_text(encoding="utf-8"))["settings"] if r["image"] == "training" and r["kept"]]
    if not rows:
        raise SystemExit(f"{con_dir}: no kept setting")
    return max(rows, key=lambda r: r["specific_gain"])


class W:
    """What analyse_e7_d7.c_measures reads of a world."""

    def __init__(self, st):
        self.directed, self.dist, self.succ = st.directed, st.dist, st.tw.succ


def c13(st, V: np.ndarray, Qh: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = d7a.c_measures(V, Qh, s, W(st))
    return m["C1"], m["C3"]


# ------------------------------------------------------------------------------------------------------------- A
def persist(st, setting: dict | None, vector: np.ndarray | None = None) -> tuple[dict, dict]:
    """A (setting: the constrained correction) or D (vector: a fixed perturbation of h on zone rows, setting None)."""
    model, e6, dev = st.models["experimental"], st.e6, st.device
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, e6)
    zone = np.zeros(two_zones.N_STATES, bool)
    zone[st.zone] = True
    if vector is None:
        inner = ra.stepwise_hook(model, st.tw, st.e7, 1.0, st.e7["apply_in"], setting["lam"], setting["mu_pos"], setting["mu_goal"],
                                 float(st.e7["constrained"]["tau"]))

        def perturb(h, z, s):
            return rdiag.correction(st, model, h, z, s, setting)
    else:
        inner = ra.fixed_hook(vector, 1.0, "zone")

        def perturb(h, z, s):
            on = torch.as_tensor(zone[s], device=h.device)[:, None].to(h.dtype)
            return torch.as_tensor(vector, dtype=h.dtype, device=h.device)[None, :] * on

    def once(t, h, z, states, in_zone, rows):
        if t == 0:
            return inner(t, h, z, states, in_zone, rows)
        return h, {"intervened": np.zeros(len(rows)), "delta_norm": np.zeros(len(rows))}
    ep = fs.run_model(model, st.tw, st.image, e6, st.eps["experimental"], hook=once)
    ep0 = fs.run_model(model, st.tw, st.image, e6, st.eps["experimental"])
    states, actions, length = ep["states"], ep["actions"], ep["length"]
    n, H = actions.shape
    A = model.rssm.cfg.n_actions
    out = {f"{k}_{f}": np.full((n, H), np.nan) for k in ("b", "C1", "C3", "mass", "dir") for f in ("corr", "raw")}
    out["delta_norm"] = np.full(n, np.nan)
    out["dh_norm"], out["dh_cos0"] = np.full((n, H), np.nan), np.full((n, H), np.nan)  # |h_corr - h_raw| and its cos with dh_0
    dh0 = None
    g_post, g_meas, g_act, g_z = (gen([1, i], dev) for i in range(4))
    streams = {f: list(model.rssm.initial_state(n, dev)) for f in ("corr", "raw")}
    for t in range(H):
        rows = np.flatnonzero(length > t)
        if rows.size == 0:
            break
        s = states[rows, t]
        x = st.tw.templates_t[torch.as_tensor(s, device=dev)]
        a_prev = torch.zeros(len(rows), A, device=dev) if t == 0 else F.one_hot(torch.as_tensor(actions[rows, t - 1], device=dev), A).to(x.dtype)
        first = torch.full((len(rows),), t == 0, dtype=torch.bool, device=dev)
        r_t = torch.as_tensor(rows, device=dev)
        # both streams start the step from the same generator states (common draws); they consume the same draws, so
        # the raw stream leaves the generators where the corrected one would have
        post_state, meas_state, act_state, z_state = g_post.get_state(), g_meas.get_state(), g_act.get_state(), g_z.get_state()
        for f in ("corr", "raw"):
            for g_, s_ in ((g_post, post_state), (g_meas, meas_state), (g_act, act_state), (g_z, z_state)):
                g_.set_state(s_)
            h_all, z_all = streams[f]
            with torch.no_grad():
                obs = model.rssm.observation_step(h_all[r_t], z_all[r_t], a_prev, x, first, generator=g_post)
            h, z = obs["h"], obs["z"]
            if t == 0 and f == "corr":
                d = perturb(h, z, s)
                h = h + d
                out["delta_norm"][rows] = d.norm(dim=1).cpu().numpy()
            h_all, z_all = h_all.clone(), z_all.clone()
            h_all[r_t], z_all[r_t] = h, z
            streams[f] = [h_all, z_all]
            inz = zone[s]
            if not inz.any():
                continue
            sel = torch.as_tensor(np.flatnonzero(inz), device=dev)
            sc = step_controllability(model.rssm, h[sel], z[sel], st.tw.templates_t, int(e6["draws"]), float(e6["margin"]),
                                      generator=g_meas, true_successor=st.tw.succ[s[inz]])
            rr = rows[inz]
            out[f"b_{f}"][rr, t] = sc["I"] - ref_fresh[s[inz]]
            out[f"mass_{f}"][rr, t] = sc["mass_true"][np.arange(inz.sum()), actions[rr, t]]
            ch = channel_q(model, h[sel], z[sel], g_act, g_z, mean_zbar=True)
            out[f"C1_{f}"][rr, t], out[f"C3_{f}"][rr, t] = c13(st, ch["V_zbar"].cpu().numpy(), ch["Q_hbar"].cpu().numpy(), s[inz])
            out[f"dir_{f}"][rr, t] = dr.greedy_directed_share(ch["Q_normal"].cpu().numpy(), st.directed[s[inz]], ad.TIE_TOL)
        dh = (streams["corr"][0][r_t] - streams["raw"][0][r_t]).cpu().numpy()  # no draw: the streams' draws are untouched
        if t == 0:
            dh0 = np.zeros((n, dh.shape[1]))
            dh0[rows] = dh
        out["dh_norm"][rows, t] = np.linalg.norm(dh, axis=1)
        out["dh_cos0"][rows, t] = (dh * dh0[rows]).sum(axis=1) / (np.linalg.norm(dh, axis=1) * np.linalg.norm(dh0[rows], axis=1))
    exit_ = zp.first_exit(states, length, zone)["time"]
    first_stay = np.arange(H)[None, :] < exit_[:, None]
    by_t = []
    for t in range(H):
        m = first_stay[:, t] & np.isfinite(out["b_corr"][:, t]) & np.isfinite(out["b_raw"][:, t])
        if m.sum() == 0:
            break
        by_t.append({"t": t, "n_in_zone": int(m.sum()), **{f"{k}_{f}": float(np.nanmean(out[f"{k}_{f}"][m, t])) for k in ("b", "C1", "C3", "mass", "dir") for f in ("corr", "raw")},
                     "dh_norm": float(np.nanmean(out["dh_norm"][m, t])), "dh_ratio": float(np.nanmean(out["dh_norm"][m, t] / out["dh_norm"][m, 0])),
                     "dh_cos0": float(np.nanmean(out["dh_cos0"][m, t]))})
    dist = st.dist
    dir_once = zp.directed_by_index(states, length, zone, dist, H)
    dir_exp = zp.directed_by_index(ep0["states"], ep0["length"], zone, dist, H)
    summary = {"by_t": by_t, "directed_by_index": {"once": dir_once["share"].tolist(), "once_n": dir_once["n"].tolist(),
                                                   "exposed": dir_exp["share"].tolist(), "exposed_n": dir_exp["n"].tolist()}}
    arrays = {**out, "states": states, "actions": actions, "length": length, "first_stay": first_stay, "exposed_states": ep0["states"],
              "exposed_length": ep0["length"]}
    return arrays, summary


def b_gain_t0(st, vectors: list[np.ndarray]) -> list[float]:
    """D's calibration: b's mean gain at t = 0 over E6's starts for h + v, with the replay's own t = 0 draws (persist's
    posterior and measure generators, fresh), so that the gain is the replay's at t = 0 (mean over the starts where both b are
    finite, as the replay's by_t)."""
    model, dev = st.models["experimental"], st.device
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
    s = fs.starts_for(st.image["slot"], st.e6["n_reps"])
    h0, z0 = model.rssm.initial_state(len(s), dev)
    x = st.tw.templates_t[torch.as_tensor(s, device=dev)]
    with torch.no_grad():
        obs = model.rssm.observation_step(h0, z0, torch.zeros(len(s), model.rssm.cfg.n_actions, device=dev), x,
                                          torch.ones(len(s), dtype=torch.bool, device=dev), generator=gen([1, 0], dev))

    def b(h):
        return step_controllability(model.rssm, h, obs["z"], st.tw.templates_t, int(st.e6["draws"]), float(st.e6["margin"]),
                                    generator=gen([1, 1], dev), true_successor=st.tw.succ[s])["I"] - ref_fresh[s]
    b_raw = b(obs["h"])
    return [float(np.nanmean(b(obs["h"] + torch.as_tensor(v, dtype=obs["h"].dtype, device=dev)) - b_raw)) for v in vectors]  # finite pairs, as the replay


def natural_vector(st, collect_dirs: list[Path], which: str) -> tuple[np.ndarray | None, dict]:
    """D: -d_col (a = 1), or +alpha * d_rec with alpha calibrated on b's gain at t = 0 (None: not calibratable)."""
    dq = directions(collect_dirs)[st.image["slot"]]
    v_col = -dq["d_col"]
    target = b_gain_t0(st, [v_col])[0]
    if which == "col":
        return v_col, {"direction": "-d_col", "alpha": 1.0, "target_gain": target, "gain": target, "norm": float(np.linalg.norm(v_col))}
    grid = [(0.0, 0.0), *zip(REC_GRID, b_gain_t0(st, [a * dq["d_rec"] for a in REC_GRID]))]
    info = {"direction": "+d_rec", "target_gain": target, "grid": [list(g) for g in grid], "reached": False,
            "cos_d_col_d_rec": cos(dq["d_col"], dq["d_rec"])}
    for (a0, g0), (a1, g1) in zip(grid, grid[1:]):
        if g0 < target <= g1:
            alpha = a0 + (target - g0) * (a1 - a0) / (g1 - g0)
            info["reached"] = True
            break
    else:
        alpha, gmax = max(grid[1:], key=lambda g: g[1])
        if gmax < 0.10:
            return None, {**info, "alpha": alpha, "gain": gmax, "calibratable": False}
    v = alpha * dq["d_rec"]
    return v, {**info, "alpha": float(alpha), "gain": b_gain_t0(st, [v])[0], "calibratable": True, "norm": float(np.linalg.norm(v))}


# ------------------------------------------------------------------------------------------------------------- B
def zone_pixels(slot: str, k: int) -> dict[int, tuple[slice, slice]]:
    return {int(c): (slice((c // N_COLS) * k, (c // N_COLS + 1) * k), slice((c % N_COLS) * k, (c % N_COLS + 1) * k)) for c in fs.zone_states(slot)}


def zone_color(img: np.ndarray, cells: dict, exclude: list[np.ndarray]) -> np.ndarray:
    """img (m, C, H, W) in [0, 1]: (m, 3) the mean RGB (0-255) of the zone's cells but the excluded ones (per row)."""
    out = np.zeros((img.shape[0], 3))
    for i in range(img.shape[0]):
        px = [img[i, :, ys, xs].reshape(3, -1) for c, (ys, xs) in cells.items() if all(c != int(e[i]) for e in exclude)]
        out[i] = np.concatenate(px, axis=1).mean(axis=1) * 255.0
    return out


def decoder(st, setting: dict) -> tuple[dict, dict]:
    model, dev, k = st.models["experimental"], st.device, st.tw.kpx
    rssm, A, C = model.rssm, model.rssm.cfg.n_actions, model.rssm.cfg.n_classes
    cells_rep = np.repeat(fs.interior_states(st.image["slot"]), DECODER_PER_CELL)
    h, z = fresh_start_contexts(rssm, st.tw, cells_rep, gen([2, 0], dev))
    d = rdiag.correction(st, model, h, z, cells_rep, setting)
    zone_px = zone_pixels(st.image["slot"], k)
    refs = {"inc": np.asarray(st.conf["t_inc"], float), "ctrl": np.asarray(st.conf["t_ctrl"], float), "neu": np.asarray(two_zones.CORRIDOR_COLOR, float)}
    out: dict = {"cells": cells_rep, "delta": d.cpu().numpy(), "h": h.cpu().numpy()}
    mean_img = {}
    for tag, hr in (("before", h), ("after", h + d)):
        with torch.no_grad():
            rec = rssm.decode(hr, z).clamp(0, 1).cpu().numpy()
        col = zone_color(rec, zone_px, [cells_rep])
        for r, ref in refs.items():
            out[f"rec_D_{r}_{tag}"] = np.linalg.norm(col - ref, axis=1)
        out[f"rec_loc_{tag}"] = (localise(rssm, st.tw, hr, z)["agent_cell"] == cells_rep).astype(float)
        g = gen([2, 1], dev)  # the same prior draws before and after
        D = {r: np.zeros((len(cells_rep), A, PRED_DRAWS)) for r in refs}
        loc = np.zeros((len(cells_rep), A, PRED_DRAWS))
        imgs = []
        for a in range(A):
            onehot = F.one_hot(torch.full((len(cells_rep),), a, device=dev, dtype=torch.long), A).to(hr.dtype)
            with torch.no_grad():
                im = rssm.imagination_step(hr, z, onehot, generator=g)
                prior = im["prior_probs"].unsqueeze(1).expand(-1, PRED_DRAWS, -1, -1)
                zz = F.one_hot(sample_categorical_index(prior, g), C).to(hr.dtype)
                hh = im["h"].unsqueeze(1).expand(-1, PRED_DRAWS, -1).reshape(-1, hr.shape[-1])
                pred = rssm.decode(hh, zz.reshape(-1, *zz.shape[2:])).clamp(0, 1)
            succ = st.tw.succ[cells_rep, a]
            pred_np = pred.cpu().numpy()
            col = zone_color(pred_np, zone_px, [np.repeat(cells_rep, PRED_DRAWS), np.repeat(succ, PRED_DRAWS)])
            for r, ref in refs.items():
                D[r][:, a] = np.linalg.norm(col - ref, axis=1).reshape(len(cells_rep), PRED_DRAWS)
            loc[:, a] = (decoded_agent_cell(pred, st.tw.agent, k) == np.repeat(succ, PRED_DRAWS)).reshape(len(cells_rep), PRED_DRAWS)
            imgs.append(pred_np.mean(axis=0))
        for r in refs:
            out[f"pred_D_{r}_{tag}"] = D[r].mean(axis=(1, 2))
        out[f"pred_loc_{tag}"] = loc.mean(axis=(1, 2))
        mean_img[tag] = np.mean(imgs, axis=0)
    out["pred_mean_before"], out["pred_mean_after"] = mean_img["before"], mean_img["after"]
    norm = float(np.linalg.norm(refs["inc"] - refs["neu"]))
    s = (out["pred_D_inc_after"] - out["pred_D_inc_before"]) / norm
    summary = {"s_mean": float(s.mean()), "inc_minus_neutral": norm, "delta_norm_mean": float(np.linalg.norm(out["delta"], axis=1).mean()),
               **{f"{p}_{m}_{tag}": float(np.mean(out[f"{p}_{m}_{tag}"])) for p in ("pred", "rec") for m in ("D_inc", "D_ctrl", "D_neu", "loc")
                  for tag in ("before", "after")}}
    return out, summary


def imagine(st, setting: dict) -> tuple[dict, dict]:
    """A in imagination, paired: B's contexts, h and h + d rolled IMAGINE_STEPS imagined steps with the same actions (the
    first: each action; then uniform from a fixed seed) and the same prior draws; the particle's cell follows the true
    transitions; per step |dh|, |dh| / |dh_0|, cos(dh, dh_0) and b (zone cells) of both streams."""
    model, dev = st.models["experimental"], st.device
    rssm, A, C = model.rssm, model.rssm.cfg.n_actions, model.rssm.cfg.n_classes
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
    cells = np.repeat(fs.interior_states(st.image["slot"]), DECODER_PER_CELL)
    h, z = fresh_start_contexts(rssm, st.tw, cells, gen([2, 0], dev))  # B's contexts
    d = rdiag.correction(st, model, h, z, cells, setting)
    zone = np.zeros(two_zones.N_STATES, bool)
    zone[st.zone] = True
    m, draws, margin = len(cells), int(st.e6["draws"]), float(st.e6["margin"])
    shape = (A, IMAGINE_STEPS + 1, m)
    out = {k: np.full(shape, np.nan) for k in ("dh_norm", "dh_cos0", "b_corr", "b_raw", "in_zone")}
    for a0 in range(A):
        g, gm = gen([4, a0], dev), gen([5, a0], dev)
        acts = np.random.default_rng(np.random.SeedSequence([SEED, 4, a0])).integers(A, size=(m, IMAGINE_STEPS - 1))
        hs, zs, cell = {"corr": h + d, "raw": h}, {"corr": z, "raw": z}, cells.copy()
        for i in range(IMAGINE_STEPS + 1):
            if i > 0:
                a = np.full(m, a0) if i == 1 else acts[:, i - 2]
                onehot = F.one_hot(torch.as_tensor(a, device=dev), A).to(h.dtype)
                g_state = g.get_state()
                for f in ("corr", "raw"):
                    g.set_state(g_state)
                    with torch.no_grad():
                        im = rssm.imagination_step(hs[f], zs[f], onehot, generator=g)
                        zs[f] = F.one_hot(sample_categorical_index(im["prior_probs"], g), C).to(h.dtype)
                    hs[f] = im["h"]
                cell = st.tw.succ[cell, a]
            dh = (hs["corr"] - hs["raw"]).cpu().numpy()
            out["dh_norm"][a0, i] = np.linalg.norm(dh, axis=1)
            out["dh_cos0"][a0, i] = (dh * d.cpu().numpy()).sum(axis=1) / (np.linalg.norm(dh, axis=1) * d.norm(dim=1).cpu().numpy())
            inz = zone[cell]
            out["in_zone"][a0, i] = inz
            if inz.any():
                sel = torch.as_tensor(np.flatnonzero(inz), device=dev)
                gm_state = gm.get_state()
                for f in ("corr", "raw"):
                    gm.set_state(gm_state)
                    I = step_controllability(rssm, hs[f][sel], zs[f][sel], st.tw.templates_t, draws, margin, generator=gm)["I"]
                    out[f"b_{f}"][a0, i, inz] = I - ref_fresh[cell[inz]]
    by_i = []
    gap0 = float(np.nanmean(out["b_corr"][:, 0] - out["b_raw"][:, 0]))
    for i in range(IMAGINE_STEPS + 1):
        inz = out["in_zone"][:, i] == 1
        gap = float(np.nanmean((out["b_corr"][:, i] - out["b_raw"][:, i])[inz])) if inz.any() else np.nan
        by_i.append({"step": i, "share_in_zone": float(inz.mean()), "b_gap": gap, "r_b": gap / gap0,
                     "dh_ratio_all": float(np.nanmean(out["dh_norm"][:, i] / out["dh_norm"][:, 0])),
                     "dh_ratio_in_zone": float(np.nanmean((out["dh_norm"][:, i] / out["dh_norm"][:, 0])[inz])) if inz.any() else np.nan,
                     "dh_cos0_all": float(np.nanmean(out["dh_cos0"][:, i]))})
    return out, {"by_step": by_i}


# ------------------------------------------------------------------------------------------------------------- C
def collect(st, slot: str) -> tuple[dict, dict]:
    model, dev = st.models["experimental"], st.device
    hd = model.rssm.cfg.h_dim
    sums = np.zeros((len(es.ASSIGNMENTS), 2, two_zones.N_STATES, len(STEP_CLASSES), hd))
    counts = np.zeros((len(es.ASSIGNMENTS), 2, two_zones.N_STATES, len(STEP_CLASSES)))
    zone = np.zeros(two_zones.N_STATES, bool)
    zone[fs.zone_states(slot)] = True
    e6 = {**st.e6, "measure": "first"}
    for k in range(len(es.ASSIGNMENTS)):
        colors = es.assignment_colors(k, st.conf["diagonal"], st.conf["t_inc"], st.conf["t_ctrl"])
        _, tw = fs.build_test_world(st.cfg, st.conf, colors, dev)
        image = {"name": f"assignment{k}_{slot}", "index": 3000 + k, "slot": slot}

        def hook(t, h, z, states, in_zone, rows, k=k):
            inz = zone[states]
            if inz.any():
                hn = h.cpu().numpy()[inz]
                half, cell = rows[inz] % 2, states[inz]
                for cls, on in ((0, t == 0), (1, t >= 4), (2, True)):
                    if on:
                        np.add.at(sums[k, :, :, cls], (half, cell), hn)
                        np.add.at(counts[k, :, :, cls], (half, cell), 1)
            return h, {}
        fs.run_model(model, tw, image, e6, st.eps["experimental"], hook=hook)
    return {"sums": sums, "counts": counts}, {"slot": slot, "roles": [es.assignment_roles(k)[slot] for k in range(len(es.ASSIGNMENTS))]}


def directions(collect_dirs: list[Path]) -> dict:
    """Per slot: d_col (T_inc - neutral), d_ctrl (T_ctrl - neutral), d_rec (t >= 4 - t = 0 in T_inc), cell-matched, weighted
    by the steps; d_col of each half; and the weights."""
    out = {}
    for d in collect_dirs:
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        arr = np.load(d / "collect.npz")
        sums, counts, slot = arr["sums"], arr["counts"], meta["slot"]
        roles = [es.assignment_roles(k)[slot] for k in range(len(es.ASSIGNMENTS))]
        cells = fs.zone_states(slot)

        def mean_h(role, halves, cls):
            ks = [k for k, r in enumerate(roles) if r == role]
            s = sums[np.ix_(ks, halves)][..., cls, :].sum(axis=(0, 1))
            c = counts[np.ix_(ks, halves)][..., cls].sum(axis=(0, 1))
            return s, c

        def diff(role_a, cls_a, role_b, cls_b, halves=(0, 1)):
            sa, ca = mean_h(role_a, list(halves), cls_a)
            sb, cb = mean_h(role_b, list(halves), cls_b)
            ok = [c for c in cells if ca[c] > 0 and cb[c] > 0]
            w = np.array([ca[c] + cb[c] for c in ok], float)
            v = np.array([sa[c] / ca[c] - sb[c] / cb[c] for c in ok])
            return (w[:, None] * v).sum(axis=0) / w.sum(), int(len(ok))
        all_cls = STEP_CLASSES.index("all")
        d_col, n_col = diff("ud", all_cls, "neutral", all_cls)
        out[slot] = {"d_col": d_col, "n_cells_col": n_col, "d_ctrl": diff("cd", all_cls, "neutral", all_cls)[0],
                     "d_rec": diff("ud", STEP_CLASSES.index("late"), "ud", STEP_CLASSES.index("first"))[0],
                     "d_col_half0": diff("ud", all_cls, "neutral", all_cls, (0,))[0], "d_col_half1": diff("ud", all_cls, "neutral", all_cls, (1,))[0]}
    return out


def random_directions(norm: float, dim: int) -> list[np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3]))
    out = []
    for _ in range(N_RANDOM):
        u = rng.standard_normal(dim)
        out.append(u / np.linalg.norm(u) * norm)
    return out


def coldir_conditions(dirs: dict, slot: str, dim: int, amplitudes, random_amplitudes) -> list[dict]:
    v = -dirs[slot]["d_col"]
    items = [{"name": "exposed", "vector": None, "alpha": 0.0}]
    items += [{"name": f"col_a{a:g}", "vector": v, "alpha": float(a)} for a in amplitudes]
    for j, u in enumerate(random_directions(float(np.linalg.norm(v)), dim)):
        items += [{"name": f"rand{j}_a{a:g}", "vector": u, "alpha": float(a)} for a in random_amplitudes]
    return items


def measure_condition(st, tw, image: dict, hook, ref_plain: dict, ref_fresh: np.ndarray) -> tuple[dict, dict]:
    """One condition of C or E on (tw, image), with D7's q_fn: b at step 1, localisation, directed share, C1 and C3 on
    the localised zone steps (the zone of image's slot)."""
    model, e6 = st.models["experimental"], {**st.e6, "measure": "zone"}
    if not np.array_equal(tw.succ, st.tw.succ):
        raise RuntimeError("the test world's transitions differ from the training image's (c13 reads st's)")
    zgen = torch.Generator(device=st.device).manual_seed(rdis.seed_of(st.d["seed"], "rollout", 0.0, 7))  # the same for every condition
    buffer: list[dict] = []

    def q_fn(h, z, s, g):
        ch = channel_q(model, h, z, rd7.clone(g, st.device), zgen, mean_zbar=True)
        q = q_values(model.rssm, model.goal_head, model.critic, h, z, model.cfg.k_act, model.cfg, g)
        if not torch.equal(q, ch["Q_normal"]):
            raise RuntimeError("the counterfactual's Q differs from the policy's")
        buffer.append({k: ch[k].cpu().numpy() for k in rd7.KEYS})
        return q
    ep = fs.run_model(model, tw, image, e6, st.eps["experimental"], hook=hook, q_fn=q_fn)
    full = rd7.assemble(buffer, ep["length"], ep["Q"].shape, "")
    belief = fs.belief_row(ep, ref_plain, ref_fresh, image)
    ph = fs.phenotype_row(ep, image, st.eps["experimental"], st.e6, with_bootstrap=False)
    st_, ln = ep["states"], ep["length"]
    zone = np.zeros(two_zones.N_STATES, bool)
    zone[fs.zone_states(image["slot"])] = True
    keep = zp.in_zone_steps(st_, ln, zone) & (ep["loc_template"] == 1)
    i, t = np.nonzero(keep)
    C1, C3 = c13(st, full["V_zbar"][i, t], full["Q_hbar"][i, t], st_[i, t]) if len(i) else (np.array([np.nan]), np.array([np.nan]))
    return {"b_step1": belief["b_step1"], "loc_template_zone": belief["loc_template_zone"], "loc_template_step1": belief["loc_template_step1"],
            "directed_share": ph["directed_share"], "C1": float(np.nanmean(C1)), "C3": float(np.nanmean(C3)), "n_C_steps": int(len(i))}, ep


def transfer(st, collect_dirs: list[Path], slot: str) -> tuple[dict, dict]:
    """E: episodes starting in slot (the zone) in the test world drawn with each of the 3 training assignments where slot
    carries T_inc (collect's images and seeds); exposed, and h <- h - d_col(Q1) at every zone step for each Q1."""
    dirs = directions(collect_dirs)
    rows, arrays = [], {}
    e6 = {**st.e6, "measure": "zone"}
    for k in range(len(es.ASSIGNMENTS)):
        if es.assignment_roles(k)[slot] != "ud":
            continue
        colors = es.assignment_colors(k, st.conf["diagonal"], st.conf["t_inc"], st.conf["t_ctrl"])
        _, tw = fs.build_test_world(st.cfg, st.conf, colors, st.device)
        image = {"name": f"assignment{k}_{slot}", "index": 3000 + k, "slot": slot}  # collect's image
        ref_fresh = fs.reference_fresh(st.models["reference"], tw, image, st.e6)
        ref_plain = fs.run_model(st.models["reference"], tw, image, e6, st.eps["reference"])
        for q1 in [None, *QUADRANT_ORDER]:
            t0 = time.perf_counter()
            hook = None if q1 is None else ra.fixed_hook(-dirs[q1]["d_col"], 1.0, "zone")
            row, ep = measure_condition(st, tw, image, hook, ref_plain, ref_fresh)
            name = "exposed" if q1 is None else f"dcol_{q1}"
            rows.append({"assignment": k, "slot": slot, "condition": name, "q1": q1, **row, "seconds": time.perf_counter() - t0})
            arrays[f"a{k}__{name}__states"], arrays[f"a{k}__{name}__loc_template"] = ep["states"], ep["loc_template"]
            print(f"[ctx transfer] {st.agent['id']} {slot} a{k} {name}: b_step1 {row['b_step1']:+.3f}, C3 {row['C3']:.2f}, "
                  f"directed {row['directed_share']:.3f}, loc zone {row['loc_template_zone']:.2f}; {rows[-1]['seconds']:.0f} s", flush=True)
    return arrays, {"rows": rows, "slot": slot, "norms": {q: float(np.linalg.norm(dq["d_col"])) for q, dq in dirs.items()}}


def coldir(st, collect_dirs: list[Path], only: list[str] | None, amplitudes, random_amplitudes) -> tuple[dict, dict]:
    model, e6 = st.models["experimental"], {**st.e6, "measure": "zone"}
    dirs = directions(collect_dirs)
    slot = st.image["slot"]
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
    ref_plain = fs.run_model(st.models["reference"], st.tw, st.image, e6, st.eps["reference"])
    arrays = {f"{q}__{k}": v for q, dq in dirs.items() for k, v in dq.items() if isinstance(v, np.ndarray)}
    rows = []
    for it in coldir_conditions(dirs, slot, model.rssm.cfg.h_dim, amplitudes, random_amplitudes):
        if only is not None and it["name"] not in only:
            continue
        t0 = time.perf_counter()
        hook = None if it["vector"] is None else ra.fixed_hook(it["vector"], it["alpha"], "zone")
        row, ep = measure_condition(st, st.tw, st.image, hook, ref_plain, ref_fresh)
        rows.append({"condition": it["name"], "alpha": it["alpha"], "norm": 0.0 if it["vector"] is None else float(it["alpha"] * np.linalg.norm(it["vector"])),
                     **row, "seconds": time.perf_counter() - t0})
        arrays[f"{it['name']}__states"], arrays[f"{it['name']}__loc_template"] = ep["states"], ep["loc_template"]
        print(f"[ctx coldir] {st.agent['id']} {it['name']}: b_step1 {rows[-1]['b_step1']:+.3f}, C3 {rows[-1]['C3']:.2f}, "
              f"directed {rows[-1]['directed_share']:.3f}, loc zone {rows[-1]['loc_template_zone']:.2f}; {rows[-1]['seconds']:.0f} s", flush=True)
    summary = {"rows": rows, "slot": slot, "directions": {q: {"norm_d_col": float(np.linalg.norm(dq["d_col"])), "n_cells_col": dq["n_cells_col"],
                                                              "split_half_cos": cos(dq["d_col_half0"], dq["d_col_half1"])} for q, dq in dirs.items()}}
    return arrays, summary


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", required=True)
    parser.add_argument("--constrained", type=Path, default=None, help="persist, decoder: the agent's run_e7_constrained.py output")
    parser.add_argument("--slot", choices=QUADRANT_ORDER, default=None, help="collect: the quadrant the episodes start in")
    parser.add_argument("--collect", nargs="+", type=Path, default=[], help="coldir: the agent's four collect runs")
    parser.add_argument("--direction", choices=("col", "rec"), default=None, help="natpersist: -d_col or +alpha * d_rec")
    parser.add_argument("--conditions", nargs="+", default=None, help="coldir: a subset of the conditions (default: all)")
    parser.add_argument("--amplitudes", nargs="+", type=float, default=list(AMPLITUDES))
    parser.add_argument("--random-amplitudes", nargs="+", type=float, default=list(RANDOM_AMPLITUDES))
    parser.add_argument("--n-reps", type=int, default=None, help="quick runs only")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    if args.n_reps is not None:
        cfg["e6"] = {**cfg["e6"], "n_reps": args.n_reps}
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        setting = best_setting(args.constrained) if args.phase in ("persist", "decoder", "imagine") else None
        part = {"collect": args.slot, "coldir": None if args.conditions is None else "-".join(args.conditions),
                "natpersist": args.direction, "transfer": args.slot}.get(args.phase)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / (f"e7ctx__{agent['id']}__{args.phase}" + (f"__{part}" if part else "") + f"__{stamp}")
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            st = rdiag.Setting(cfg, agent, device)
            before = rdiag.weights_hash(st.models)
            if args.phase == "persist":
                arrays, summary = persist(st, setting)
            elif args.phase == "decoder":
                arrays, summary = decoder(st, setting)
            elif args.phase == "imagine":
                arrays, summary = imagine(st, setting)
            elif args.phase == "natpersist":
                vector, calib = natural_vector(st, args.collect, args.direction)
                if vector is None:
                    raise RuntimeError(f"d_rec not calibratable: {calib}")
                arrays, summary = persist(st, None, vector)
                arrays["vector"], summary["calibration"] = vector, calib
            elif args.phase == "transfer":
                arrays, summary = transfer(st, args.collect, args.slot)
            elif args.phase == "collect":
                arrays, summary = collect(st, args.slot)
            else:
                arrays, summary = coldir(st, args.collect, args.conditions, args.amplitudes, args.random_amplitudes)
            after = rdiag.weights_hash(st.models)
            if after != before:
                raise RuntimeError(f"weights moved: {before} -> {after}")
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / NPZ[args.phase], **arrays)
        (out / "summary.json").write_text(json.dumps({**summary, "setting": setting}, indent=1, default=float), encoding="utf-8")
        meta = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "phase": args.phase, "part": part,
                "slot": args.slot if args.phase in ("collect", "transfer") else st.image["slot"], "image": st.image["name"],
                "constrained": None if args.constrained is None else str(args.constrained), "collect": [str(p) for p in args.collect],
                "weights": {r: str(p_) for r, p_ in agent["weights"].items()}, "weights_hash_before": before, "weights_hash_after": after,
                "agents_file": str(args.agents), "config": str(args.config), "config_bases": cfg["_bases"], "n_reps": int(cfg["e6"]["n_reps"]),
                "device": str(device), "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                "seconds": time.perf_counter() - t0, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[ctx {args.phase}] {agent['id']} {part or ''}: done in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
