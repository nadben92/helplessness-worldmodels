"""The pilot's runs as campaign seeds (docs/e3/DECISIONS.md, 2026-10-07 (30)): a finished closed-loop run of the pilot
(gpu_pilot_cl, seeds 0 and 5) measured afterwards with the campaign's protocol and measures, from the weights and the
replay saved on the cluster. Read-only on the run: everything is written into <out dir>/<run name>/, laid out as a
campaign run directory, so campaign_analysis.py and make_handoff.py read it as one.

The measures are run_agent.py's own, with the campaign config's measure, twin and checkpoints blocks
(configs/e3/gpu_campaign.yaml; training blocks: the run's own, identical -- checked): at every checkpoint of the run,
the twin (twin_at_checkpoint, its kernel kept) and the goal diagnostics (goal_kernel_at_checkpoint: 10,000 draws per
pair into the goal, Q gaps next to the goal, V by zone, the zones' separation on h and z); at every checkpoint after
the switch (read from the run's console.log; extra checkpoints would be needed only for a switch after ~39,800, and
there are no weights for them: an error), the full measures (full_measures_at_checkpoint, x4 contexts). Same streams,
keyed by the checkpoint, as in a campaign run.

The replay at checkpoint c is the prefix of the last resume point's replay (remeasure_checkpoints.replay_prefix); the
run's last checkpoint (300,000) is after the last resume point (290,000): it is measured on the replay at 290,000,
FLAGGED (replay_env_steps in its results, replay_final.npz is that replay). The weights at every checkpoint and the
final ones are copied into <out>/<run name>/checkpoints/ for the handoff.

Usage: python experiments/E3_worldmodel_4zones/campaign_measure_pilot.py <pilot run dir> --config configs/e3/gpu_campaign.yaml
           [--device cuda] --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.agents.worldmodel.heads import Critic, GoalHead
from helplessness.experiment_loader import load_experiment_module
from helplessness.runs import write_run_meta

remeasure = load_experiment_module("E3_worldmodel_4zones", "remeasure_checkpoints.py", "campaign_pilot_remeasure")
run_agent, run_mini = remeasure.run_agent, remeasure.run_mini
TRAINING_BLOCKS = ("model", "render", "budget", "collection", "train", "agent", "log")
MEASURE_BLOCKS = ("measure", "twin", "checkpoints")


def campaign_config_for(run_cfg: dict, campaign_cfg: dict) -> dict:
    """The run's config with the campaign's measure, twin and checkpoints blocks; refuses a run whose training blocks
    differ from the campaign's (then it would not be a campaign seed)."""
    differ = [b for b in TRAINING_BLOCKS if run_cfg[b] != campaign_cfg[b]]
    if differ:
        raise ValueError(f"the run's training differs from the campaign's in {differ}: not a campaign seed")
    if run_cfg["checkpoints"]["grid"] != campaign_cfg["checkpoints"]["grid"]:
        raise ValueError("the run's checkpoint grid is not the campaign's")
    return {**run_cfg, **{b: campaign_cfg[b] for b in MEASURE_BLOCKS}}


def switch_env_step(run_dir: Path) -> int:
    line = next(line for line in (run_dir / "console.log").read_text(encoding="utf-8").splitlines() if ">>> SWITCH" in line)
    return int(re.search(r"environment step (\d+)", line).group(1))


def measure_pilot_run(run_dir: Path, campaign_config: Path, device_arg: str | None, out_dir: Path) -> Path:
    run_dir = Path(run_dir)
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    seed, condition, last = int(meta["seed"]), meta["condition"], int(meta["env_steps"])
    cfg = campaign_config_for(run_mini.load_mini_config(run_dir / "config.yaml"), run_mini.load_mini_config(campaign_config))
    if device_arg is not None:
        cfg["runtime"]["device"] = device_arg
    device = run_mini.configure_runtime(cfg)
    out = Path(out_dir) / run_dir.name
    (out / "checkpoints").mkdir(parents=True, exist_ok=False)
    log = run_mini.RunLog(out / "console.log", f"[campaign measures seed={seed} {condition}]")
    env_meta = run_mini.environment_metadata(device)
    write_run_meta(out, seed, status="running", condition=condition, measured_from=str(run_dir), source_run_meta=meta,
                   **{k: env_meta[k] for k in ("device", "source_sha256")})
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    records = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = [r["env_step"] for r in records]
    switch_env = switch_env_step(run_dir)
    if run_agent.extra_checkpoints(switch_env, steps, last, cfg["measure"]["checkpoint_measures"]):
        raise ValueError(f"switch at {switch_env}: the campaign's rule asks for extra checkpoints, which this run has no weights for")
    counts = np.load(run_dir / "counts.npz")
    resume = torch.load(run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    full = pickle.loads(resume["replay"])
    resume_env = int(resume["progress"]["env_step"])
    world = run_mini.build_world(cfg, seed, condition)
    P_ref = (run_mini.build_world(cfg, seed, "reference", themes=world.themes).env.transition_tensor() if condition == "experimental"
             else world.env.transition_tensor())
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    acfg = run_agent.agent_config(cfg)
    log(f"{run_dir.name}: seed {seed} {condition}, switch at environment step {switch_env}, checkpoints {steps}, resume replay at "
        f"{resume_env}; code {env_meta['source_sha256']}")
    twin_arrays, new_records, final = {}, [], None
    t_start = time.perf_counter()
    for rec in records:
        c = rec["env_step"]
        replay, replay_env = ((remeasure.replay_prefix(full, c + int(rec["episodes"])), c) if c <= resume_env else (full, resume_env))
        if device.type == "cuda":
            replay.to_device(device)
        state = torch.load(remeasure.weights_path(run_dir, c, last), map_location="cpu", weights_only=False)
        rssm = run_mini.build_rssm(cfg)
        rssm.load_state_dict(state["rssm"])
        goal_head = GoalHead(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
        goal_head.load_state_dict(state["goal_head"])
        critic = Critic(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
        critic.load_state_dict(state["critic"])
        for m in (rssm, goal_head, critic):
            m.to(device).eval()
        N = counts[f"N_{c}"]
        new = dict(rec)
        new["twin_by_zone"] = run_agent.twin_at_checkpoint(N, world, P_ref, cfg, acfg, seed, condition, c, twin_arrays)
        new["twin_matches_pilot_record"] = new["twin_by_zone"] == rec["twin_by_zone"]  # same counts, same stream: same twin
        new["goal_diagnostics_pilot"] = rec.get("goal_diagnostics")
        new["goal_diagnostics"] = run_agent.goal_kernel_at_checkpoint(rssm, world, replay, N, cfg, seed, condition, c, device,
                                                                      goal_head=goal_head, critic=critic, acfg=acfg)
        new["replay_env_steps"] = replay_env
        if c > switch_env:
            fields, arrays, new["measures"] = run_agent.full_measures_at_checkpoint(
                rssm, world, replay, N, cfg, seed, condition, out, log, int(state["step"]), c, device,
                extra={"replay_env_steps": replay_env, "measured_from": str(run_dir)})
            if c == last:
                final = (fields, arrays, replay_env)
        new_records.append(new)
        with open(out / "checkpoints.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(run_mini._jsonable(new)) + "\n")
        shutil.copy2(remeasure.weights_path(run_dir, c, last), out / "checkpoints" / f"agent_env{c:06d}.pt")
        log(f"checkpoint {c} done (replay at {replay_env}{', FLAGGED' if replay_env < c else ''}); {time.perf_counter() - t_start:.0f} s so far")
    shutil.copy2(run_dir / "checkpoints" / "agent_final.pt", out / "checkpoints" / "agent_final.pt")
    shutil.copy2(run_dir / "counts.npz", out / "counts.npz")
    np.savez_compressed(out / "twin.npz", **twin_arrays)
    np.savez_compressed(out / "replay_final.npz", **full.export_compact())  # the replay at the last resume point (flagged)
    if final is None:
        raise ValueError(f"the last checkpoint {last} was not measured")
    fields, arrays, replay_env = final
    np.savez_compressed(out / "arrays.npz", **arrays)
    report = {"seed": seed, "condition": condition, "zone_class": run_mini.e3_env.block_class_for_seed(seed), "zone_kinds": world.zone_kinds,
              "env_steps": last, "grad_steps": records[-1]["grad_step"], "switch_step": records[-1]["switch_step"],
              "switch_env_step": switch_env, "measured_from": str(run_dir), "replay_env_steps_final": replay_env,
              "replay_final_env_steps": resume_env, "checkpoints": new_records, "reproducibility": env_meta, **fields}
    (out / "results.json").write_text(json.dumps(run_mini._jsonable({k: v for k, v in report.items() if k not in run_mini.ARRAY_FIELDS}),
                                                 indent=1), encoding="utf-8")
    write_run_meta(out, seed, status="done", env_steps=last, switch_step=records[-1]["switch_step"])
    log(f"done in {time.perf_counter() - t_start:.0f} s; outputs in {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True, help="the campaign's config (its measure, twin, checkpoints blocks)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    measure_pilot_run(args.run_dir, args.config, args.device, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
