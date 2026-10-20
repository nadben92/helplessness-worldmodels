"""E5 training run (docs/E5_belief_training.md): one configuration of E4 (fixed for all of E5, --configuration), one
seed (A = E4's seed of the configuration, B = A + e5.seed_b_offset), one training condition, with E4's procedure and
parameters unchanged (E3's closed loop, pipeline.agent_run.run_one, through ClosedLoopHooks):

  baseline      E4's run (E4's checkpoint records, pipeline.two_zones_records), experimental or reference;
                --save-resumable-at keeps the full state at those steps (the immunisation branches from it);
  shuffle       --fraction f: at each episode, with probability f, the zone assignment is redrawn among the 12
                (pipeline.episode_sources.ZoneAssignmentShuffle; the assignment stream is the seed's, shared by episode
                index between the experimental run and its reference); records by assignment
                (pipeline.assignment_records): every assignment at the checkpoints of e5.shuffle_records
                (all_assignments_at; null = every checkpoint), the canonical one at every checkpoint (in "e4", E4's
                format), plus visits by zone type and the episodes by assignment (assignments.npz);
  immunisation  --switch-step S [--post-steps P]: the reference's dynamics until the first episode starting once S
                transitions are done, then the experimental dynamics for P steps (pipeline.episode_sources.
                DynamicsSwitch), on the reference's streams; checkpoints at the switch + E4's grid points below P, and
                at the switch + P; beta_rep's ramp stays E4's planned budget (299,744 + 256). --branch-from <a
                resumable_env<N>.pt of the baseline reference of the same seed, N <= S>: starts from that state instead
                of recomputing phase 1 (identical, tested).

Every run records, at every checkpoint, what E4 records (checkpoints.jsonl "e4") and E5's additions ("e5": visits by
zone type; for the immunisation the effective switch step and the post-switch step). Outputs as run_e4.py's, under
results/E5_belief_training/<run name>/.

Usage, from the repo root:
  python experiments/E5_belief_training/run_e5.py --configuration p1-a-d1 --seed-label B --training baseline \\
      --condition reference [--save-resumable-at 100000 125000 ...] [--config configs/e5/e5.yaml] [--device cuda]
  python experiments/E5_belief_training/run_e5.py --configuration p1-a-d1 --seed-label A --training shuffle --fraction 1.0 \\
      --condition experimental
  python experiments/E5_belief_training/run_e5.py --configuration p1-a-d1 --seed-label B --training immunisation \\
      --switch-step 100000 [--post-steps 175000] [--branch-from <run>/checkpoints/resumable_env100000.pt]
  python experiments/E5_belief_training/run_e5.py --resume results/E5_belief_training/<run dir>
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

import e5_setup
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import two_zones as two_zones_world
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import ClosedLoopHooks, checkpoint_env_steps, run_one
from helplessness.pipeline.assignment_records import (
    decodability,
    mean_over_assignments,
    real_context_by_assignment,
    type_of_row,
    visit_shares,
    walks_by_assignment,
)
from helplessness.pipeline.episode_sources import (
    ASSIGNMENTS,
    DynamicsSwitch,
    ZoneAssignmentShuffle,
    assignment_colors,
    assignment_world,
    canonical_assignment,
)
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import make_hooks as e4_hooks
from helplessness.runs import make_run_dir, make_run_name, write_run_meta
from helplessness.streams.assignment import assignment_generator

N_STATES = two_zones.N_STATES


def _visits(replay, offset: int, types: np.ndarray) -> dict:
    """Visits by zone type of a run with one appearance: its replay rows are states, offset to the canonical block."""
    return visit_shares(replay.obs_rows_and_flags()[0] + offset, types)


def estimator_switch(state: dict) -> dict:
    """E3's estimator switch (probabilities -> straight-through, the batch-pair JS criterion): its gradient and
    environment steps (None before it) and the environment steps elapsed since (docs/e5/DECISIONS.md (16), (20))."""
    env = state.get("switch_env_step")
    return {"grad_step": state.get("switch_step"), "env_step": env, "env_steps_since": None if env is None else state["env_step"] - env}


def apply_e5_procedure(cfg: dict) -> dict:
    """E5's two changes to E4's training procedure, on every E5 run (docs/e5/DECISIONS.md (19), (20)): the switch
    statistic on same-background pairs and the later separation deadline. Both leave a run that switches before E4's
    deadline with one background bit-identical to E4's (tested). Returns cfg, modified in place."""
    est = cfg["train"]["latent_estimator"]
    if cfg["e5"].get("switch_pairs", "any") != "any":
        est["switch_pairs"] = cfg["e5"]["switch_pairs"]
    if cfg["e5"].get("switch_max_steps") is not None:
        est["switch_max_steps"] = int(cfg["e5"]["switch_max_steps"])
    return cfg


def decodability_record(state: dict, cfg: dict, world, canonical: int) -> dict | None:
    """The agent state's decodability on the canonical assignment at the early checkpoints (e5.decodability.until_step),
    for a run with one appearance (the shuffle runs have it by assignment at every checkpoint)."""
    dec = cfg["e5"]["decodability"]
    if state["env_step"] > dec["until_step"]:
        return None
    rssm, device = state["rssm"], state["device"]
    was_training = rssm.training
    rssm.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        acc = decodability(rssm, state["replay"], {canonical: world}, cfg, state["seed"], state["condition"], state["env_step"],
                           dec["max_per_state"], e5_setup.SLOT_DECODABILITY, row_offset=canonical * N_STATES)[canonical]
    rssm.train(was_training)
    return {"decode_accuracy": acc, "max_per_state": dec["max_per_state"]}


def baseline_hooks(cfg: dict, conf: dict) -> ClosedLoopHooks:
    base = e4_hooks(cfg, conf)
    types, canonical = type_of_row(), canonical_assignment(conf["diagonal"])

    def checkpoint_record(state: dict) -> dict:
        rec = base.checkpoint_record(state)
        if rec.get("e4") is not None:
            rec["e5"] = {"visits_by_type": _visits(state["replay"], canonical * N_STATES, types), "estimator_switch": estimator_switch(state),
                         "decodability": decodability_record(state, cfg, state["world"], canonical)}
        return rec

    return ClosedLoopHooks(episode=base.episode, build_world=base.build_world, checkpoint_record=checkpoint_record, finalize=base.finalize)


def shuffle_hooks(cfg: dict, conf: dict, fraction: float) -> ClosedLoopHooks:
    """The run's own condition decides the dynamics of every assignment (the reference: everything controllable,
    T_inc kept on the slot that would hold the conveyor)."""
    base = e4_hooks(cfg, conf)
    canonical = canonical_assignment(conf["diagonal"])
    types = type_of_row()
    sr = cfg["e5"]["shuffle_records"]
    ck = cfg["e4"]["checkpoint"]
    worlds: dict[int, object] = {}
    probe_worlds = {k: two_zones_setup.probe_world(cfg, conf, assignment_colors(k, conf["diagonal"], conf["t_inc"], conf["t_ctrl"]))
                    for k in range(len(ASSIGNMENTS))}

    def episode_source(world, cfg_, seed, condition):
        if assignment_colors(canonical, conf["diagonal"], conf["t_inc"], conf["t_ctrl"]) != conf["training_colors"]:
            raise AssertionError("the canonical assignment must be the configuration's training image")
        device = world.templates_t.device
        worlds.clear()
        for k in range(len(ASSIGNMENTS)):
            w = world if k == canonical else assignment_world(k, conf, condition, cfg_["render"], cfg_["episode"], seed)
            w.templates_t = w.templates_t.to(device)
            worlds[k] = w
        return ZoneAssignmentShuffle([worlds[k] for k in range(len(ASSIGNMENTS))], canonical, fraction, assignment_generator(seed))

    def checkpoint_record(state: dict) -> dict:
        import time

        rssm, device, env_step, source = state["rssm"], state["device"], state["env_step"], state["episode_source"]
        at = sr.get("all_assignments_at")
        final = env_step == state["n_env_steps"]
        ks = list(range(len(ASSIGNMENTS))) if (at is None or env_step in at or final) else [canonical]
        t0 = time.perf_counter()
        was_training = rssm.training
        rssm.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            for k in ks:
                probe_worlds[k].templates_t = probe_worlds[k].templates_t.to(device)
            t_rc = time.perf_counter()
            rc = real_context_by_assignment(rssm, state["replay"], {k: worlds[k] for k in ks}, cfg, state["seed"], state["condition"], env_step,
                                            ck["real_context_max_per_state"], ck["real_context_draws"], e5_setup.SLOT_ASSIGNMENT_REAL_CONTEXT)
            t_walks = time.perf_counter()
            wk = walks_by_assignment(rssm, cfg, conf, {k: probe_worlds[k] for k in ks}, canonical, ck["n_walks"], "checkpoint")
            t_end = time.perf_counter()
            final_rec = None
            if final and cfg["e4"].get("final_protocol_control"):  # the canonical assignment, E4's full settings
                rcf = cfg["measure"]["real_context"]
                final_rec = {"real_context": real_context_by_assignment(rssm, state["replay"], {canonical: worlds[canonical]}, cfg, state["seed"],
                                                                        state["condition"], env_step, rcf["max_per_state"], rcf["kernel_draws"],
                                                                        e5_setup.SLOT_ASSIGNMENT_FINAL_REAL_CONTEXT)[canonical],
                             "walks": walks_by_assignment(rssm, cfg, conf, {canonical: probe_worlds[canonical]}, canonical,
                                                          cfg["e4"]["walks"]["n_walks"], "final")[canonical]["slots"]}
        rssm.train(was_training)
        counts = np.bincount(np.asarray(source.episode_assignment, dtype=np.int64), minlength=len(ASSIGNMENTS))
        e4 = {"real_context": rc[canonical], "walks": wk[canonical]["slots"], "seconds": time.perf_counter() - t0}
        if final_rec is not None:
            e4["final"] = final_rec
        e5 = {"fraction": fraction, "canonical": canonical, "assignments_measured": ks,
              "by_assignment": {str(k): {"real_context": rc[k], "I_step1_by_type": wk[k]["I_step1_by_type"],
                                         "walks": wk[k]["slots"]} for k in ks},
              "mean_over_assignments": ({"mass_true_by_type": mean_over_assignments(rc, "mass_true_by_type"),
                                         "I_s_by_type": mean_over_assignments(rc, "I_s_by_type"),
                                         "I_step1_by_type": mean_over_assignments(wk, "I_step1_by_type")} if len(ks) > 1 else None),
              "visits_by_type": visit_shares(state["replay"].obs_rows_and_flags()[0], types),
              "episodes_by_assignment": counts.tolist(), "episodes_drawn": int(np.sum(source.episode_drawn)),
              "estimator_switch": estimator_switch(state),
              "decode_accuracy_canonical": rc[canonical]["decode_accuracy"],
              "episodes": len(source.episode_assignment),
              "seconds": {"real_context": t_walks - t_rc, "walks": t_end - t_walks, "per_assignment": (t_end - t_rc) / len(ks)}}
        np.savez_compressed(state["run_dir"] / "assignments.npz", **source.log_arrays())
        state["log"](f"E5 checkpoint {env_step}: {len(ks)} assignment(s) in {e4['seconds']:.0f} s; canonical I_step1 by slot "
                     f"{ {q: round(v['I_curve'][0], 3) for q, v in e4['walks'].items()} }; episodes drawn {e5['episodes_drawn']}/{e5['episodes']}")
        return {"e4": e4, "e5": e5}

    return ClosedLoopHooks(episode=base.episode, build_world=base.build_world, checkpoint_record=checkpoint_record, finalize=base.finalize,
                           episode_source=episode_source)


def immunisation_hooks(cfg: dict, conf: dict, switch_step: int, post_steps: int) -> ClosedLoopHooks:
    """The run's condition is "reference" (its streams, its world); the experimental world is built on the same
    streams (env_condition="reference"), with the training image."""
    base = e4_hooks(cfg, conf)
    types, canonical = type_of_row(), canonical_assignment(conf["diagonal"])
    grid = checkpoint_env_steps(cfg, 10**9)[:-1]  # E4's grid points (in post-switch steps)
    after: dict[str, object] = {}

    def episode_source(world, cfg_, seed, condition):
        if condition != "reference":
            raise ValueError("the immunisation runs on the reference's streams")
        w = two_zones_world.build_world(conf["training_colors"], conf["agent_color"], conf["diagonal"], "experimental", cfg_["render"],
                                        cfg_["episode"], seed, env_condition="reference")
        w.templates_t = w.templates_t.to(world.templates_t.device)
        after["world"] = w
        return DynamicsSwitch(world, w, switch_step, post_steps, grid)

    def checkpoint_record(state: dict) -> dict:
        source = state["episode_source"]
        current = after["world"] if source.switched else state["world"]
        rec = base.checkpoint_record({**state, "world": current})
        eff = source.dynamics_switch_step
        rec["e5"] = {"nominal_switch_step": switch_step, "dynamics_switch_step": eff, "post_steps": post_steps,
                     "post_switch_step": None if eff is None else state["env_step"] - eff,
                     "visits_by_type": _visits(state["replay"], canonical * N_STATES, types), "estimator_switch": estimator_switch(state),
                     "decodability": decodability_record(state, cfg, current, canonical)}
        return rec

    return ClosedLoopHooks(episode=base.episode, build_world=base.build_world, checkpoint_record=checkpoint_record, finalize=base.finalize,
                           episode_source=episode_source)


def check_branch(path: Path, seed: int, switch_step: int) -> None:
    """The branch file must be a resumable state of a reference run of the same seed, at or before the switch."""
    run_dir = path.parent.parent
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    if int(meta["seed"]) != seed or meta["condition"] != "reference":
        raise SystemExit(f"{path}: not a reference run of seed {seed} (seed {meta['seed']}, {meta['condition']})")
    step = int(path.stem.rsplit("env", 1)[1])
    if step > switch_step:
        raise SystemExit(f"{path}: saved at {step}, after the switch step {switch_step}")


def run_spec(cfg: dict) -> dict:
    return cfg["e5"]["run"]


def build(cfg: dict) -> tuple[dict, str, ClosedLoopHooks, dict]:
    """(conf, run_one condition, hooks, run_one keyword arguments) of the run described by cfg["e5"]["run"]."""
    spec = run_spec(cfg)
    conf = e5_setup.configuration(cfg, spec["configuration"], spec["seed_label"])
    kwargs = {}
    if spec["training"] == "baseline":
        hooks, condition = baseline_hooks(cfg, conf), spec["condition"]
        if spec.get("save_resumable_at"):
            kwargs["save_resumable_at"] = tuple(int(s) for s in spec["save_resumable_at"])
    elif spec["training"] == "shuffle":
        hooks, condition = shuffle_hooks(cfg, conf, float(spec["fraction"])), spec["condition"]

    else:
        condition = "reference"
        hooks = immunisation_hooks(cfg, conf, int(spec["switch_step"]), int(spec["post_steps"]))
        kwargs["total_env_steps"] = int(spec["switch_step"]) + int(spec["post_steps"]) + int(cfg["episode"]["max_steps"])
    apply_e5_procedure(cfg)  # docs/e5/DECISIONS.md (19), (20): every E5 run, one procedure
    return conf, condition, hooks, kwargs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=e5_setup.DEFAULT_CONFIG)
    parser.add_argument("--configuration", help="the E4 configuration of E5 (a key of e4.configurations)")
    parser.add_argument("--seed-label", choices=e5_setup.SEED_LABELS)
    parser.add_argument("--training", choices=e5_setup.TRAININGS)
    parser.add_argument("--condition", choices=two_zones.CONDITIONS, help="baseline and shuffle: experimental or reference")
    parser.add_argument("--fraction", type=float, default=None, help="shuffle: the probability that an episode's assignment is redrawn")
    parser.add_argument("--switch-step", type=int, default=None, help="immunisation: the nominal switch step")
    parser.add_argument("--post-steps", type=int, default=None, help="immunisation: steps after the switch (default e5.immunisation.post_steps)")
    parser.add_argument("--branch-from", default=None,
                        help="immunisation: a resumable state of the baseline reference of the same seed, or `auto` (the newest "
                             "such run under --out-dir with resumable_env<switch step>.pt)")
    parser.add_argument("--save-resumable-at", nargs="+", default=None,
                        help="baseline: keep the full state at these steps (`candidates`: e5.resumable_candidates, and the steps given)")
    parser.add_argument("--max-env-steps", type=int, default=None, help="baseline / shuffle: stop early, the planned budget unchanged")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=e5_setup.REPO_ROOT / "results")
    parser.add_argument("--resume", type=Path, default=None, help="continue the interrupted run in this directory")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.resume:
        run_dir = args.resume
        cfg = e5_setup.load_config(run_dir / "config.yaml")
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
        max_env_steps, resume, branch = meta.get("max_env_steps"), True, None
    else:
        if not (args.configuration and args.seed_label and args.training):
            parser.error("--configuration, --seed-label and --training are required (or --resume)")
        spec = {"configuration": args.configuration, "seed_label": args.seed_label, "training": args.training}
        if args.training in ("baseline", "shuffle"):
            if args.condition is None:
                parser.error(f"{args.training} needs --condition")
            spec["condition"] = args.condition
        elif args.condition not in (None, "reference"):
            parser.error("the immunisation runs on the reference's streams (no --condition)")
        if args.training == "shuffle":
            if args.fraction is None:
                parser.error("shuffle needs --fraction")
            spec["fraction"] = args.fraction
        if args.training == "immunisation":
            if args.switch_step is None:
                parser.error("immunisation needs --switch-step")
            if args.max_env_steps is not None:
                parser.error("the immunisation's length is switch + post steps (no --max-env-steps)")
        cfg = e5_setup.load_config(args.config)
        if args.training == "immunisation":
            spec["switch_step"] = args.switch_step
            spec["post_steps"] = args.post_steps if args.post_steps is not None else int(cfg["e5"]["immunisation"]["post_steps"])
            spec["branch_from"] = None
        if args.save_resumable_at:
            if args.training != "baseline":
                parser.error("--save-resumable-at is for the baseline runs")
            steps = {int(v) for v in args.save_resumable_at if v != "candidates"}
            if "candidates" in args.save_resumable_at:
                steps |= {int(v) for v in cfg["e5"]["resumable_candidates"]}
            spec["save_resumable_at"] = sorted(steps)
        cfg["e5"]["run"] = spec
        cfg["e4"]["run_configuration"] = args.configuration  # as E4's runs (E4's tools read it)
        conf = e5_setup.configuration(cfg, args.configuration, args.seed_label)
        if args.branch_from is not None and args.training != "immunisation":
            parser.error("--branch-from is for the immunisation")
        max_env_steps, resume, branch = args.max_env_steps, False, None
        if args.branch_from == "auto":
            branch = e5_setup.find_run(args.out_dir, args.configuration, "baseline", "reference", conf["seed"],
                                       need=f"checkpoints/resumable_env{args.switch_step:06d}.pt") / "checkpoints" / f"resumable_env{args.switch_step:06d}.pt"
        elif args.branch_from is not None:
            branch = Path(args.branch_from)
        if branch is not None:
            check_branch(branch, conf["seed"], args.switch_step)
            cfg["e5"]["run"]["branch_from"] = str(branch)
        name = make_run_name(two_zones.GRID_SHAPE, e5_setup.zone_preset(args.configuration, args.training, args.fraction),
                             e5_setup.pair_role(args.training, args.condition or "reference"), e5_setup.AGENT_TYPE, conf["seed"])
        run_dir = make_run_dir(args.out_dir, e5_setup.EXPERIMENT_ID, name)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    conf, condition, hooks, kwargs = build(cfg)
    try:
        run_one(cfg, conf["seed"], condition, run_dir, device, max_env_steps=max_env_steps, resume=resume, hooks=hooks,
                branch_from=branch, **kwargs)
    except BaseException as err:
        write_run_meta(run_dir, conf["seed"], status="failed", error=f"{type(err).__name__}: {err}",
                       traceback=traceback.format_exc()[-4000:])
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
