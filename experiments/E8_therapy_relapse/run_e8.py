"""E8's therapy runs (docs/E8_therapy_relapse.md, docs/e8/DECISIONS.md): one start model of configs/e8/agents.yaml
re-trained in a controllable world from its checkpoint's weights, through the library's closed loop
(pipeline.agent_run.run_one with fresh_start; e8_therapy.make_hooks), with E5's procedure
(two_zones_setup.apply_e5_procedure) and E4's parameters unchanged.

Phases (--phase):
  train    a new therapy run of --model: baselines, the full exposure; shuffle models, --treated-quadrant q (nw, ne, se, sw,
           all). The start file's SHA-256 is checked first. Checkpoints e8.checkpoints (0 = the start's weights, before any
           step), each with its weights (agent_env<N>.pt) and full resume state (resumable_env<N>.pt).
  resume   continues the interrupted run in --run-dir from its checkpoints/resume.pt (identical to the uninterrupted run
           on the same device; tested).
  probes   the frozen measures of a pair's therapy checkpoints (probes_e8.py; DECISIONS (10)).

Outputs: results/E8_therapy_relapse/<run name>/ (config.yaml, meta.yaml, checkpoints.jsonl, console.log, steps.npz,
assignments.npz for the shuffle runs, metrics.csv, results.json, checkpoints/, replay_final.npz). Run name:
grid12x12__2zones-p1-b-d1-<base|sh100>-th<full|q>__pair-<experimental|control>__worldmodel-agent-e8__seed<NN>__<UTC>.

Usage, from the repo root:
  python experiments/E8_therapy_relapse/run_e8.py --phase train --model base_A_exp --runs-root <root> [--device cuda]
  python experiments/E8_therapy_relapse/run_e8.py --phase train --model shuffle_E_exp --treated-quadrant nw --runs-root <root>
  python experiments/E8_therapy_relapse/run_e8.py --phase resume --run-dir results/E8_therapy_relapse/<run>
  python experiments/E8_therapy_relapse/run_e8.py --phase probes ...   (see probes_e8.py)
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import e8_setup
import e8_therapy
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import FreshStart, run_one
from helplessness.pipeline.runtime import configure_runtime
from helplessness.runs import make_run_dir, make_run_name, write_run_meta
from helplessness.streams.therapy import therapy_root

PHASES = ("train", "resume", "probes")


def run_config(cfg: dict, model_id: str, model: dict, therapy: str, start: Path, runs_root: Path) -> dict:
    """The run's resolved config: E5's procedure applied, the therapy's checkpoints as the grid, the run's spec."""
    two_zones_setup.apply_e5_procedure(cfg)  # docs/e8/DECISIONS.md (4): E5's procedure (the collapse guard's statistic)
    cfg["checkpoints"]["grid"] = e8_setup.checkpoint_grid(cfg)
    cfg["e4"]["run_configuration"] = model["configuration"]
    cfg["e8"]["run"] = {"model": model_id, **{k: model[k] for k in ("kind", "role", "configuration", "seed", "run", "checkpoint", "sha256")},
                        "therapy": therapy, "start_file": str(start.relative_to(runs_root)).replace("\\", "/")}
    return cfg


def build(cfg: dict) -> tuple[dict, str, object, FreshStart]:
    """(conf, run_one condition, hooks, fresh start) of the run described by cfg["e8"]["run"]; the start file's path is
    resolved under the runs root recorded in cfg["e8"]["runs_root"] (an argument, never in the repository)."""
    spec = cfg["e8"]["run"]
    conf = {**two_zones_setup.configuration(cfg, spec["configuration"]), "seed": int(spec["seed"])}
    treated = e8_setup.treated_slot(spec, conf, spec["therapy"])
    hooks = e8_therapy.make_hooks(cfg, conf, spec["therapy"], treated, int(spec["seed"]))
    start = Path(cfg["e8"]["runs_root"]) / spec["start_file"]
    return conf, spec["role"], hooks, FreshStart(weights=start, stream_root=therapy_root(int(spec["seed"])), sha256=spec["sha256"])


def main(argv: list[str] | None = None) -> int:
    # no abbreviations: --phase probes passes its own options (--run ...) through to probes_e8.py
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--config", type=Path, default=e8_setup.DEFAULT_CONFIG)
    parser.add_argument("--model", help="train: a model id of the agents file (e8.agents)")
    parser.add_argument("--treated-quadrant", default=None, help="train, shuffle models: nw, ne, se, sw or all")
    parser.add_argument("--runs-root", type=Path, default=None, help="train / resume: the root the start runs live under")
    parser.add_argument("--run-dir", type=Path, default=None, help="resume: the interrupted run's directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=e8_setup.REPO_ROOT / "results")
    args, rest = parser.parse_known_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.phase == "probes":  # everything but --phase goes to probes_e8.py (its --config, --device and --out-dir included)
        here = str(Path(__file__).resolve().parent)  # this script's own directory (when loaded by path, e.g. from the tests)
        if here not in sys.path:
            sys.path.insert(0, here)
        import probes_e8

        argv = list(sys.argv[1:] if argv is None else argv)
        i = argv.index("--phase")
        return probes_e8.main(argv[:i] + argv[i + 2:])
    if rest:
        parser.error(f"unrecognised arguments: {rest}")
    if args.phase == "train":
        if not (args.model and args.runs_root):
            parser.error("train needs --model and --runs-root")
        cfg = e8_setup.load_config(args.config)
        models = e8_setup.load_models(cfg["e8"]["agents"])["models"]
        if args.model not in models:
            parser.error(f"unknown model {args.model!r} (agents file: {sorted(models)})")
        model = models[args.model]
        therapy = e8_setup.therapy_of(model, args.treated_quadrant)
        start = e8_setup.start_weights(model, args.runs_root)  # SHA-256 checked
        cfg = run_config(cfg, args.model, model, therapy, start, args.runs_root)
        name = make_run_name(two_zones.GRID_SHAPE, e8_setup.zone_preset(model, therapy), e8_setup.PAIR_ROLE[model["role"]],
                             e8_setup.AGENT_TYPE, int(model["seed"]))
        run_dir = make_run_dir(args.out_dir, e8_setup.EXPERIMENT_ID, name)
        resume = False
    else:
        if args.run_dir is None:
            parser.error("resume needs --run-dir")
        run_dir = args.run_dir
        cfg = e8_setup.load_config(run_dir / "config.yaml")
        resume = True
    # the start file is read on a fresh start only (resume: its state comes from resume.pt)
    cfg["e8"]["runs_root"] = str(args.runs_root) if args.runs_root is not None else "."
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    conf, condition, hooks, fresh = build(cfg)
    cfg["e8"].pop("runs_root")  # not part of the resolved config (machine-specific)
    try:
        run_one(cfg, conf["seed"], condition, run_dir, device, resume=resume, hooks=hooks, total_env_steps=int(cfg["e8"]["max_env_steps"]),
                save_resumable_at=tuple(cfg["checkpoints"]["grid"]), fresh_start=fresh)
    except BaseException as err:
        write_run_meta(run_dir, conf["seed"], status="failed", error=f"{type(err).__name__}: {err}",
                       traceback=traceback.format_exc()[-4000:])
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
