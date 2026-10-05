"""Re-measurement of a closed-loop run's checkpoint weights (docs/e3/DECISIONS.md, 2026-10-05 (22), (23)): the
complete final measures (run_mini.measure_final: I_s, I_s projected, off-support mass, I_z, kernels, probes) of the
weights saved at the given checkpoints (checkpoints/agent_env<NNNNNN>.pt; agent_final.pt for the run's last step), each
on the replay AS IT WAS at that checkpoint. Nothing of the run is modified; the run directory is only read.

The replay: run_agent.py keeps no replay file; the only stored replay is the one pickled in checkpoints/resume.pt, the
last resume point (for the pilot: 290,000 environment steps, the last multiple of resume_every before the end). The
closed-loop replay is strict and chronological (never overwritten), so the replay at checkpoint c is EXACTLY the first
c + episodes(c) stored observations of any later replay (one observation per environment step, plus each finished
episode's arrival observation; episodes(c) from checkpoints.jsonl), and is rebuilt as that prefix. A checkpoint after
the resume point (the pilot's 300,000) has no exact replay: it is measured on the resume point's replay, flagged
"replay_env_steps" < the checkpoint (its run's own final measures, results.json / arrays.npz, are the exact value).
The counts N are the run's own at the checkpoint (counts.npz N_<c>). The measurement streams are the run's own
(run_mini.seed_streams), as in its final measures; the real contexts are drawn from the rebuilt replay.

Usage: python experiments/E3_worldmodel_4zones/remeasure_checkpoints.py <run dir> --env-steps 63096 100000 ... \\
           [--device cuda] --out-dir DIR
Outputs: <out dir>/<run name>/env<NNNNNN>/ with results.json, arrays.npz, SUMMARY.md, console.log, artifacts/.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.experiment_loader import load_experiment_module

run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "remeasure_run_agent")
recompute = load_experiment_module("E3_worldmodel_4zones", "recompute_measures.py", "remeasure_recompute")
run_mini = run_agent.run_mini


def weights_path(run_dir: Path, env_step: int, last_env_step: int) -> Path:
    p = run_dir / "checkpoints" / f"agent_env{env_step:06d}.pt"
    if p.exists():
        return p
    if env_step == last_env_step and (run_dir / "checkpoints" / "agent_final.pt").exists():
        return run_dir / "checkpoints" / "agent_final.pt"
    raise FileNotFoundError(f"no weights for environment step {env_step} in {run_dir / 'checkpoints'}")


def replay_prefix(replay: ReplayBuffer, length: int) -> ReplayBuffer:
    """The first `length` stored observations of a strict chronological replay, as their own buffer."""
    arr = replay.export_compact()
    if length > len(arr["s"]):
        raise ValueError(f"prefix of {length} observations asked from a replay of {len(arr['s'])}")
    return ReplayBuffer.from_compact({k: v[:length] for k, v in arr.items()}, replay._table)


def remeasure(run_dir: Path, env_steps: list[int], device_arg: str | None, out_dir: Path) -> list[Path]:
    import yaml

    run_dir = Path(run_dir)
    cfg = run_mini.load_mini_config(run_dir / "config.yaml")
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    seed, condition = int(meta["seed"]), meta["condition"]
    if device_arg is not None:
        cfg["runtime"]["device"] = device_arg
    device = run_mini.configure_runtime(cfg)
    records = {r["env_step"]: r for r in (json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines())}
    counts = np.load(run_dir / "counts.npz")
    resume = torch.load(run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    full = pickle.loads(resume["replay"])
    resume_env = int(resume["progress"]["env_step"])
    resume_episodes = int(resume["collector"]["stats"]["episodes"])
    if len(full) != resume_env + resume_episodes:
        raise ValueError(f"resume replay holds {len(full)} observations, expected {resume_env} + {resume_episodes}: not a strict closed-loop replay")
    world = run_mini.build_world(cfg, seed, condition)
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    last = int(meta["env_steps"])
    outputs = []
    for c in env_steps:
        if c not in records:
            raise ValueError(f"{c} is not a checkpoint of this run ({sorted(records)})")
        out = out_dir / run_dir.name / f"env{c:06d}"
        (out / "artifacts").mkdir(parents=True, exist_ok=False)
        log = run_mini.RunLog(out / "console.log", f"[remeasure env {c}]")
        if c <= resume_env:
            replay = replay_prefix(full, c + int(records[c]["episodes"]))
            replay_env = c
        else:
            replay, replay_env = full, resume_env
            log(f"FLAG: checkpoint {c} is after the last resume point ({resume_env}): measured on the replay at {resume_env}")
        if device.type == "cuda":
            replay.to_device(device)
        path = weights_path(run_dir, c, last)
        state = torch.load(path, map_location="cpu", weights_only=False)
        rssm = run_mini.build_rssm(cfg)
        rssm.load_state_dict(state["rssm"])
        rssm.to(device)
        N = counts[f"N_{c}"]
        log(f"{run_dir.name}: seed {seed} {condition}, weights {path.name} (gradient step {state['step']}), replay {len(replay)} observations "
            f"(environment step {replay_env}), N {int(N.sum())} transitions, device {device}")
        t0 = time.perf_counter()
        fields, arrays = run_mini.measure_final(rssm, world, replay, N, cfg, run_mini.seed_streams(seed, device, condition), out, log,
                                                int(state["step"]))
        fields.pop("diagnostics_final")
        results = {"run_dir": str(run_dir), "checkpoint_env_step": c, "replay_env_steps": replay_env, "weights": path.name,
                   "weights_sha256": recompute.file_sha256(path), "seed": seed, "condition": condition, "seconds": time.perf_counter() - t0,
                   "reproducibility": run_mini.environment_metadata(device), **fields}
        (out / "results.json").write_text(json.dumps(run_mini._jsonable({k: v for k, v in results.items() if k not in run_mini.ARRAY_FIELDS}),
                                                     indent=1), encoding="utf-8")
        np.savez_compressed(out / "arrays.npz", **arrays)
        (out / "SUMMARY.md").write_text("\n".join(recompute.summary_lines(f"env {c}", int(state["step"]), fields)) + "\n", encoding="utf-8")
        log(f"written to {out} in {results['seconds']:.0f} s")
        outputs.append(out)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--env-steps", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    remeasure(args.run_dir, args.env_steps, args.device, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
