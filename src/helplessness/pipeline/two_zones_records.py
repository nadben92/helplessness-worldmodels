"""E4's checkpoint records of a two-zone run (docs/E4_generalization_gradient.md) and the hooks that plug them into E3's
closed loop (pipeline.agent_run.run_one, ClosedLoopHooks). Moved verbatim from
experiments/E4_generalization_gradient/run_e4.py on 2026-10-19 so that E5 records the same measures without importing
E4; run_e4.py imports them from here. At every checkpoint, without touching the training streams (own measurement
streams, torch.random.fork_rng):
  real_context  from real contexts of the run's own replay (E3's probe.real_context_probe / real_context_kernel, reduced
                settings e4.checkpoint): by region (corridor and the four slots), the prior's mass on the true successor
                in the run's own environment, and I_s (E3's main measure);
  walks         on the training image in its probe environment (reference dynamics), the fixed walks into each slot
                (e4.checkpoint.n_walks of them): I_s at entry (mean over the first m steps in the slot, by state then by
                slot), the curve over steps since entry, the states covered.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones_setup as e4_setup
from helplessness.pipeline.agent_run import ClosedLoopHooks
from helplessness.pipeline.measure import controllability_of_kernel, real_context_settings
from helplessness.pipeline.runtime import max_images_per_forward
from helplessness.probing import probe as e3_probe
from helplessness.probing.fixed_trajectory import measure_walks
from helplessness.streams import worldmodel as wm_streams

REGIONS = ("corridor", *QUADRANT_ORDER)


def region_means(values: np.ndarray, layout) -> dict[str, float]:
    regions = two_zones.region_states(layout)
    out = {}
    for name in REGIONS:
        v = values[regions[name]]
        out[name] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


def real_context_record(rssm, world, replay, cfg: dict, seed: int, condition: str, env_step: int, device, max_per_state: int,
                        draws: int, slot: int) -> dict:
    """Mass on the true successor and I_s by region, from real replay contexts (E3's functions)."""
    seq = wm_streams.spawn_sequences(seed, condition)["measure"].spawn(wm_streams.N_MEASURE_SLOTS)[slot]
    child = np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(env_step)))
    rng, gen = np.random.default_rng(child), wm_streams.torch_generator(child, device)
    rc = real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=max_per_state,
                                      n_states=two_zones.N_STATES, rng=rng, generator=gen, max_images=max_images_per_forward(cfg))
    kernel = e3_probe.real_context_kernel(rssm, ctx, world.templates_t, draws=draws,
                                          margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=gen)
    p_hat = kernel["p_hat"]
    succ = e4_setup.successors(world)
    mass = np.take_along_axis(p_hat, succ[:, :, None], axis=2)[:, :, 0].mean(axis=1)  # NaN for unmeasured states
    I_s = controllability_of_kernel(p_hat)
    return {"mass_true_by_region": region_means(mass, world.layout), "I_s_by_region": region_means(I_s, world.layout),
            "ghost_fraction_mean": float(np.nanmean(kernel["ghost_fraction"])) if np.isfinite(kernel["ghost_fraction"]).any() else float("nan"),
            "n_states_measured": int(np.isfinite(I_s).sum()), "max_per_state": max_per_state, "draws": draws}


def walks_record(rssm, cfg: dict, conf: dict, probe_world, n_walks: int, purpose: str) -> dict:
    """I_s at entry, the curve and the states covered, by slot, on the training image (fixed walks)."""
    w, draws = cfg["e4"]["walks"], cfg["e4"]["probe"]["draws"]
    succ = e4_setup.successors(probe_world)
    out = {}
    for slot in QUADRANT_ORDER:
        res = measure_walks(rssm, probe_world.templates_np, probe_world.templates_t, e4_setup.walks_for_slot(cfg, slot, n_walks), succ,
                            draws, cfg["measure"]["kernel_margin_threshold"], e4_setup.measurement_seed(cfg, conf, 0, slot, purpose),
                            w["m"], w["n_in"], chunk=w["chunk"], max_images=max_images_per_forward(cfg))
        out[slot] = {"I_entry": res["entry"], "n_states_entry": res["n_states_entry"], "I_curve": res["curve"].tolist(),
                     "n_states_curve": res["n_states_curve"].tolist(), "mass_true_mean": float(np.nanmean(res["mass_true"])),
                     "ghost_fraction_mean": float(np.nanmean(res["ghost_fraction"])), "n_walks": n_walks}
    return out


def make_hooks(cfg: dict, conf: dict) -> ClosedLoopHooks:
    probe_world = e4_setup.probe_world(cfg, conf)
    ck = cfg["e4"]["checkpoint"]

    def build_world(cfg_, seed, condition, themes=None):
        return e4_setup.world(cfg_, conf, condition)

    def checkpoint_record(state: dict) -> dict:
        rssm, device, env_step = state["rssm"], state["device"], state["env_step"]
        only_at = ck.get("at")  # None: every checkpoint (wave 1); else these steps and the run's last checkpoint
        if only_at is not None and env_step not in only_at and env_step != state["n_env_steps"]:
            return {"e4": None}
        t0 = time.perf_counter()
        was_training = rssm.training
        rssm.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            probe_world.templates_t = probe_world.templates_t.to(device)
            rec = {"real_context": real_context_record(rssm, state["world"], state["replay"], cfg, state["seed"], state["condition"],
                                                       env_step, device, ck["real_context_max_per_state"], ck["real_context_draws"],
                                                       e4_setup.SLOT_CHECKPOINT_REAL_CONTEXT),
                   "walks": walks_record(rssm, cfg, conf, probe_world, ck["n_walks"], "checkpoint")}
            if cfg["e4"].get("final_protocol_control") and env_step == state["n_env_steps"]:
                rc = cfg["measure"]["real_context"]
                rec["final"] = {"real_context": real_context_record(rssm, state["world"], state["replay"], cfg, state["seed"], state["condition"],
                                                                     env_step, device, rc["max_per_state"], rc["kernel_draws"],
                                                                     e4_setup.SLOT_FINAL_REAL_CONTEXT),
                                "walks": walks_record(rssm, cfg, conf, probe_world, cfg["e4"]["walks"]["n_walks"], "final")}
        rssm.train(was_training)
        rec["seconds"] = time.perf_counter() - t0
        state["log"](f"E4 checkpoint {env_step}: I_entry by slot "
                     f"{ {q: round(v['I_entry'], 3) for q, v in rec['walks'].items()} } | mass by region "
                     f"{ {q: round(v, 3) for q, v in rec['real_context']['mass_true_by_region'].items()} } ({rec['seconds']:.0f} s)")
        return {"e4": rec}

    def finalize(records: list, world, run_dir: Path) -> dict:
        kinds = {q: type(z.dynamics).__name__ for z in world.layout.zones for q in [z.appearance.category]}
        with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["env_step", "grad_step", "region", "role", "dynamics", "mass_true", "I_s_real_context", "I_entry_walks",
                             "n_states_entry", "success_rate", "episode_length"])
            role_of = {slot: role for role, slot in conf["roles"].items()}
            for r in records:
                e4 = r["e4"]
                if e4 is None:  # no diagnostics at this checkpoint (--diagnostics-at)
                    continue
                for region in REGIONS:
                    walk = e4["walks"].get(region, {})
                    writer.writerow([r["env_step"], r["grad_step"], region, role_of.get(region, "corridor"), kinds.get(region),
                                     e4["real_context"]["mass_true_by_region"][region], e4["real_context"]["I_s_by_region"][region],
                                     walk.get("I_entry"), walk.get("n_states_entry"), r["success_rate"], r["episode_length"]])
        return {"configuration": conf["id"], "roles": conf["roles"], "zone_kinds": kinds,
                "slot_colors": {q: list(c) for q, c in conf["training_colors"].items()}}

    return ClosedLoopHooks(episode=cfg["episode"], build_world=build_world, checkpoint_record=checkpoint_record, finalize=finalize)
