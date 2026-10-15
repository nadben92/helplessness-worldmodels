"""E3 Phase 5: the agent in closed loop (condition A of spec section 6, or
Ref in the reference environment) with its tabular twin (condition B) at
every checkpoint. CPU test bench: configs/e3/agent_mini.yaml (mini_long's
network and world-model procedure).

One environment step at a time, the agent filters its latent state, acts
epsilon-greedily on a one-step lookahead (agents.worldmodel.policy), and
stores what it sees (agents.worldmodel.closed_loop: the goal's observation
is stored). After a minimal prefill played under the same policy, every
environment step is followed by train.replay_ratio gradient steps:

  world model   the retained procedure of run_mini.py, unchanged: posterior
                probabilities until the label-free batch-pair JS criterion,
                then straight-through; lr and beta_rep scheduled on the
                switch; SeparationNotReachedError / CollapseAfterSwitchError;
  agent heads   from the switch on (before it the latent is not a discrete
                code): the goal head on the batch's posterior states (BCE
                against 1[r > 0]) and the critic in imagination
                (agents.worldmodel.imagination), slow critic by EMA.
                Before the switch the policy is the same one, on untrained
                heads: their output layers start at zero (heads.py), so
                every Q-value ties and the random tie-break explores
                uniformly until the heads have learnt something.

At E2's checkpoints (environment steps): the twin's full chain from the
agent's own counts N[s, a, s'] (experiments/E2_online_4zones/twin.py,
imported: Dirichlet alpha on the 5 successors, VI, b with the matched-count
reference, Delta under P); Delta under P of the WORLD-MODEL agent itself
(spec 5.7: its greedy policy read on its own latent states, real contexts
and canonical probe, evaluated exactly under P; probe.agent_greedy_policy);
the behaviour (success rate, episode length, tie fraction, visitation by
zone) and the training scalars.

At the end (measure.final): the E3 measures of the world model in closed
loop, run_mini.measure_final on the closed-loop replay (I_s, I_s projected,
off-support mass, I_z, kernels, probes, D1 / D2), written as run_mini writes
them, so aggregate_b.py pairs a closed-loop environment run with its
reference (b on I_s and on I_s projected, bias control).

The campaign (configs/e3/gpu_campaign.yaml; docs/e3/DECISIONS.md, 2026-10-07
(29)), each behind its config key: the same measures at every checkpoint
after the switch into measures/env<NNNNNN>/, the last one being the final
measures (measure.checkpoint_measures, extra_checkpoints); the goal pairs
with their own number of draws, the agent's Q gaps next to the goal, V by
zone and the zones' separation in the latent (measure.checkpoint_goal_kernel
goal_draws, agent); the twin's kernels (twin.save_kernels -> twin.npz); the
final replay (checkpoints.save_final_replay -> replay_final.npz); the
measures' arrays without the per-context latent distributions
(measure.light_arrays). Without these keys (the pilot), nothing changes.

Resume (docs/e3/DECISIONS.md, 2026-09-30 (8)): every checkpoints.resume_every
environment steps and at every checkpoint, checkpoints/resume.pt holds the
whole state -- models, optimisers, switch, replay, environment, counts,
behaviour, every random stream -- written atomically;
`run_agent.py --resume <run dir>` continues in the same directory and gives
the same results as an uninterrupted run (CPU; tested).

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/run_agent.py [--config configs/e3/agent_mini.yaml] [--seeds 0] [--conditions experimental]
        [--max-env-steps N] [--device cpu] [--out-dir results]
    python experiments/E3_worldmodel_4zones/run_agent.py --resume results/E3_worldmodel_4zones/<run directory>
Outputs: results/E3_worldmodel_4zones/<run name>/ with config.yaml, meta.yaml, console.log, metrics.csv (one row per
checkpoint and zone), checkpoints.jsonl, results.json, arrays.npz (the final measures), counts.npz (N at every
checkpoint), checkpoints/agent_env<NNNNNN>.pt (the weights at every checkpoint), checkpoints/agent_final.pt and
checkpoints/resume.pt.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.agents.worldmodel.acceleration import compile_from_config
from helplessness.agents.worldmodel.closed_loop import ClosedLoopCollector, EpisodeStats, stored_steps_bound
from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead, make_slow_critic
from helplessness.agents.worldmodel.imagination import train_heads_step
from helplessness.agents.worldmodel.loop import (
    CollapseAfterSwitchError,
    SeparationNotReachedError,
    WorldModelLossConfig,
    phase_learning_rate,
    rep_beta,
    set_learning_rate,
    train_world_model_step,
)
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.envs.zones import geographic_support_mask
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon
from helplessness.runs import require_code_provenance, write_run_meta

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "run_agent_run_mini")
e3_rng = load_experiment_module("E3_worldmodel_4zones", "rng_streams.py", "run_agent_rng_streams")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "run_agent_probe")
e2_twin = load_experiment_module("E2_online_4zones", "twin.py", "run_agent_e2_twin")

DEFAULT_CONFIG = run_mini.REPO_ROOT / "configs" / "e3" / "agent_mini.yaml"

from helplessness.pipeline.agent_run import (  # moved to the library on 2026-10-15, re-exported here
    AGENT_KEYS,
    OPTIONAL_AGENT_KEYS,
    TWIN_KEYS,
    SLOT_AGENT_DELTA,
    SLOT_GOAL_KERNEL,
    NUMPY_STREAMS,
    Interrupted,
    agent_config,
    checkpoint_env_steps,
    by_zone_mean,
    twin_at_checkpoint,
    agent_delta_at_checkpoint,
    goal_kernel_at_checkpoint,
    agent_at_goal_and_latent,
    extra_checkpoints,
    LIGHT_DROPPED,
    light_arrays,
    full_measures_at_checkpoint,
    _stream_states,
    _restore_streams,
    run_one,
    ClosedLoopHooks,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=run_mini.DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", choices=e3_rng.CONDITIONS, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None, help="caps the number of environment steps (smoke tests)")
    parser.add_argument("--resume", type=Path, default=None, help="continue the interrupted run in this directory")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.resume:
        cfg = run_mini.load_mini_config(args.resume / "config.yaml")
        meta = yaml.safe_load((args.resume / "meta.yaml").read_text(encoding="utf-8"))
        if args.device is not None:
            cfg["runtime"]["device"] = args.device
        device = run_mini.configure_runtime(cfg)
        run_one(cfg, int(meta["seed"]), meta["condition"], args.resume, device, max_env_steps=meta.get("max_env_steps"), resume=True)
        return 0
    cfg = run_mini.load_mini_config(args.config)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = run_mini.configure_runtime(cfg)
    failures = 0
    for seed in (args.seeds or cfg["seeds"]):
        for condition in (args.conditions or cfg["conditions"]):
            run_dir = run_mini.make_seed_run_dir(args.out_dir, f"agent-{Path(args.config).stem}", seed, condition)
            try:
                run_one(cfg, seed, condition, run_dir, device, max_env_steps=args.max_env_steps)
            except (SeparationNotReachedError, CollapseAfterSwitchError) as err:
                failures += 1
                print(f"[run_agent] seed {seed} {condition} STOPPED by a training guard ({type(err).__name__}): {err}", flush=True)
            except BaseException as err:
                write_run_meta(run_dir, seed, status="failed", error=f"{type(err).__name__}: {err}")
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
