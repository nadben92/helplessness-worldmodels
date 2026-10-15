"""E4 training run (docs/E4_generalization_gradient.md): one training configuration (colour pair x orientation x
diagonal, configs/e4/e4.yaml e4.configurations), in the experimental environment (UD + CD on the diagonal) or its
reference (everything controllable), with E3's closed-loop procedure unchanged (helplessness.pipeline.agent_run.run_one,
through ClosedLoopHooks: E4's world, episode and checkpoint records; E3's training loop, switch, schedules, agent,
resume and weight saving as they are).

At every checkpoint (checkpoints.grid) the weights are saved (checkpoints/agent_env<NNNNNN>.pt) and E4 records, without
touching the training streams (own measurement streams, torch.random.fork_rng):
  real_context  from real contexts of the run's own replay (E3's probe.real_context_probe / real_context_kernel, reduced
                settings e4.checkpoint): by region (corridor and the four slots), the prior's mass on the true successor
                in the run's own environment, and I_s (E3's main measure);
  walks         on the training image in its probe environment (reference dynamics), the fixed walks into each slot
                (e4.checkpoint.n_walks of them): I_s at entry (mean over the first m steps in the slot, by state then by
                slot), the curve over steps since entry, the states covered.
At the last checkpoint (e4.final_protocol_control): the same two measures with the full settings (E3's
measure.real_context; e4.walks.n_walks) -- the pilot's protocol control.

Outputs (results/E4_generalization_gradient/<run name>/): config.yaml (the resolved config, with e4.run_configuration),
meta.yaml, console.log, checkpoints.jsonl (one record per checkpoint, E4's in "e4"), metrics.csv (one row per checkpoint
and region), results.json, counts.npz, replay_final.npz, checkpoints/ (weights at every checkpoint, agent_final.pt,
resume.pt). With --push-batch, the light files are pushed to results/<batch> at the end, even after a failure.

Usage, from the repo root:
  python experiments/E4_generalization_gradient/run_e4.py --configuration p1-a-d1 --condition experimental [--device cuda]
      [--config configs/e4/e4.yaml] [--out-dir results] [--max-env-steps N]
      [--push-batch e4_wave1 --push-repo https://github.com/<owner>/<repo>.git --push-token-file /etc/e4-git/RESULTS_PUSH_TOKEN]
  python experiments/E4_generalization_gradient/run_e4.py --resume results/E4_generalization_gradient/<run dir> [--push-batch ...]
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

import e4_setup
from helplessness.cluster import results_push
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline.agent_run import ClosedLoopHooks, run_one
from helplessness.pipeline.measure import controllability_of_kernel, real_context_settings
from helplessness.pipeline.runtime import configure_runtime, max_images_per_forward
from helplessness.probing import probe as e3_probe
from helplessness.probing.fixed_trajectory import measure_walks
from helplessness.runs import make_run_dir, make_run_name, write_run_meta
from helplessness.streams import worldmodel as wm_streams

REGIONS = ("corridor", *QUADRANT_ORDER)
PAIR_ROLE = {"experimental": "experimental", "reference": "control"}


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
                for region in REGIONS:
                    walk = e4["walks"].get(region, {})
                    writer.writerow([r["env_step"], r["grad_step"], region, role_of.get(region, "corridor"), kinds.get(region),
                                     e4["real_context"]["mass_true_by_region"][region], e4["real_context"]["I_s_by_region"][region],
                                     walk.get("I_entry"), walk.get("n_states_entry"), r["success_rate"], r["episode_length"]])
        return {"configuration": conf["id"], "roles": conf["roles"], "zone_kinds": kinds,
                "slot_colors": {q: list(c) for q, c in conf["training_colors"].items()}}

    return ClosedLoopHooks(episode=cfg["episode"], build_world=build_world, checkpoint_record=checkpoint_record, finalize=finalize)


def run(cfg: dict, config_id: str, condition: str, run_dir: Path, device, max_env_steps: int | None, resume: bool) -> dict:
    conf = e4_setup.configuration(cfg, config_id)
    return run_one(cfg, conf["seed"], condition, run_dir, device, max_env_steps=max_env_steps, resume=resume, hooks=make_hooks(cfg, conf))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=e4_setup.DEFAULT_CONFIG)
    parser.add_argument("--configuration", help="a key of e4.configurations (e.g. p1-a-d1)")
    parser.add_argument("--condition", choices=two_zones.CONDITIONS, help="experimental (exposed) or reference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=e4_setup.REPO_ROOT / "results")
    parser.add_argument("--max-env-steps", type=int, default=None, help="stops the run early, the planned budget unchanged")
    parser.add_argument("--resume", type=Path, default=None, help="continue the interrupted run in this directory")
    parser.add_argument("--push-batch", default=None, help="push the light results to results/<batch> at the end")
    parser.add_argument("--push-repo", default=None, help="the remote's URL (https)")
    parser.add_argument("--push-token-file", default=None, help=f"file holding the token (else ${results_push.TOKEN_ENV})")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.push_batch and not args.push_repo:
        parser.error("--push-batch needs --push-repo")
    if args.resume:
        run_dir = args.resume
        cfg = e4_setup.load_config(run_dir / "config.yaml")
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
        config_id, condition, max_env_steps, resume = cfg["e4"]["run_configuration"], meta["condition"], meta.get("max_env_steps"), True
    else:
        if not (args.configuration and args.condition):
            parser.error("--configuration and --condition are required (or --resume)")
        cfg = e4_setup.load_config(args.config)
        cfg["e4"]["run_configuration"] = args.configuration
        conf = e4_setup.configuration(cfg, args.configuration)
        name = make_run_name(two_zones.GRID_SHAPE, f"{e4_setup.ZONE_PRESET}-{args.configuration}", PAIR_ROLE[args.condition],
                             e4_setup.AGENT_TYPE, conf["seed"])
        run_dir = make_run_dir(args.out_dir, e4_setup.EXPERIMENT_ID, name)
        config_id, condition, max_env_steps, resume = args.configuration, args.condition, args.max_env_steps, False
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    status, code = {"status": "done", "exit_code": 0}, 0
    try:
        run(cfg, config_id, condition, run_dir, device, max_env_steps, resume)
    except BaseException as err:  # the run's status and logs are pushed below, then the error is raised again
        status, code = {"status": "failed", "exit_code": 1, "error": f"{type(err).__name__}: {err}",
                        "traceback": traceback.format_exc()[-4000:]}, 1
        write_run_meta(run_dir, e4_setup.configuration(cfg, config_id)["seed"], status="failed", error=status["error"])
        raise
    finally:
        if args.push_batch:
            try:
                token = results_push.read_token(args.push_token_file)
                results_push.push_run(run_dir, e4_setup.EXPERIMENT_ID, args.push_batch, args.push_repo,
                                      {**status, "configuration": config_id, "condition": condition, "host": socket.gethostname()}, token)
            except Exception as push_err:  # a failed push never hides the run's own outcome
                print(f"[run_e4] RESULTS PUSH FAILED: {type(push_err).__name__}: {push_err}", flush=True)
                code = code or 2
    return code


if __name__ == "__main__":
    sys.exit(main())
