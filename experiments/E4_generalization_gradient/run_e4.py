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
With --diagnostics-at (later batches), these records are taken only at the given steps and the run's last checkpoint
("e4": null elsewhere); the weights are saved at every checkpoint either way, and the training is the same.
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
import os
import socket
import sys
import traceback
from pathlib import Path

import yaml

import e4_setup
from helplessness.cluster import results_push
from helplessness.envs.layouts import two_zones
from helplessness.pipeline.agent_run import run_one
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import (  # noqa: F401  (moved to the library on 2026-10-19; re-exported)
    REGIONS,
    make_hooks,
    real_context_record,
    region_means,
    walks_record,
)
from helplessness.runs import make_run_dir, make_run_name, write_run_meta

PAIR_ROLE = {"experimental": "experimental", "reference": "control"}


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
    parser.add_argument("--diagnostics-at", type=int, nargs="+", default=None,
                        help="checkpoint diagnostics only at these environment steps (and the last checkpoint)")
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
        if args.diagnostics_at is not None:
            cfg["e4"]["checkpoint"]["at"] = sorted(args.diagnostics_at)
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
