"""The estimator switch of E5's runs (docs/e5/DECISIONS.md (16), (19), (20)): the first hours' check and the final table.

  early  at every checkpoint up to each run's switch: the switch statistic (same-background pairs for E5's runs;
         E3/E4's for E4's), the agent state's decodability and the reconstruction error on the agent's patch; E5's runs
         against E4's runs given with --e4-runs. E4's runs did not record the decodability: --e4-decodability
         recomputes it from their checkpoint weights and replay_final.npz (on the cluster, where the weights are), with
         E5's measure and settings. Then the ALERT rule (user, 2026-10-20), never an automatic stop: a shuffle run that
         has not switched at --alert-fraction (0.5) of its separation deadline (train.latent_estimator.switch_max_steps)
         with a decodability that has not risen over its last 3 checkpoints (last minus third-to-last <= --rise-tol,
         0.01) -> "ALERT", to report to the user.
  final  one line per run: the switch's gradient and environment steps, and the environment steps trained after it
         (the run's last step minus the switch's).

Records: checkpoints.jsonl (E5: e5.estimator_switch, e5.decode_accuracy_canonical or e5.decodability, switch_pairs;
every run: batch_pair_js, switch_step, train.recon_patch). Light files only, except --e4-decodability.

Usage, from the repo root:
  python experiments/E5_belief_training/switch_monitor.py early --runs results/E5_belief_training/<run> ... \\
      [--e4-runs <E4 run dir> ... [--e4-decodability --device cuda]] [--out <file.csv>]
  python experiments/E5_belief_training/switch_monitor.py final --runs results/E5_belief_training/<run> ... [--out <file.csv>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

N_STATES = 144


def run_info(run_dir: Path) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    spec = cfg.get("e5", {}).get("run")
    label = (f"E5 {spec['training']}{'' if spec.get('fraction') is None else ' f=' + str(spec['fraction'])} "
             f"{spec.get('condition') or ''} seed {meta['seed']}" if spec else f"E4 {meta['condition']} seed {meta['seed']}")
    records = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"dir": run_dir, "cfg": cfg, "meta": meta, "spec": spec, "label": label, "records": records,
            "deadline": int(cfg["train"]["latent_estimator"]["switch_max_steps"]),
            "statistic": cfg["train"]["latent_estimator"].get("switch_pairs", "any")}


def decodability_of(rec: dict) -> float | None:
    e5 = rec.get("e5") or {}
    if e5.get("decode_accuracy_canonical") is not None:
        return float(e5["decode_accuracy_canonical"])
    dec = e5.get("decodability")
    return None if not dec else float(dec["decode_accuracy"])


def e4_decodability(info: dict, steps: list[int], device_arg: str, max_per_state: int) -> dict[int, float]:
    """E5's decodability measure on an E4 run's checkpoint weights and replay prefix (pipeline.assignment_records)."""
    import torch

    from helplessness.agents.worldmodel.replay import ReplayBuffer
    from helplessness.pipeline import two_zones_setup
    from helplessness.pipeline.assignment_records import decodability
    from helplessness.pipeline.episode_sources import canonical_assignment
    from helplessness.pipeline.runtime import configure_runtime
    from helplessness.probing.theme_probe_measure import load_rssm

    cfg = info["cfg"]
    cfg["runtime"]["device"] = device_arg
    device = configure_runtime(cfg)
    seed, condition = int(info["meta"]["seed"]), info["meta"]["condition"]
    conf = two_zones_setup.configuration(cfg, cfg["e4"]["run_configuration"])
    world = two_zones_setup.world(cfg, conf, condition)
    world.templates_t = world.templates_t.to(device)
    canonical = canonical_assignment(conf["diagonal"])
    data = dict(np.load(info["dir"] / "replay_final.npz"))
    stored = {int(r["env_step"]): int(r["stored_steps"]) for r in info["records"]}
    out = {}
    for step in steps:
        replay = ReplayBuffer.from_compact({k: v[:stored[step]] for k, v in data.items()}, world.templates_np)
        rssm = load_rssm(info["dir"] / "checkpoints" / f"agent_env{step:06d}.pt", device)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            out[step] = decodability(rssm, replay, {canonical: world}, cfg, seed, condition, step, max_per_state, 25,
                                     row_offset=canonical * N_STATES)[canonical]
    return out


def early_rows(info: dict, e4_dec: dict[int, float] | None = None) -> list[dict]:
    rows = []
    for r in info["records"]:
        dec = decodability_of(r) if e4_dec is None else e4_dec.get(int(r["env_step"]))
        rows.append({"run": info["label"], "env_step": r["env_step"], "grad_step": r["grad_step"], "statistic": info["statistic"],
                     "js": r.get("batch_pair_js"), "switched": r["switch_step"] is not None, "switch_grad_step": r["switch_step"],
                     "decodability": dec, "recon_patch": (r.get("train") or {}).get("recon_patch")})
        if r["switch_step"] is not None:
            break  # up to (and including) the first checkpoint after the switch
    return rows


def alert(info: dict, rows: list[dict], fraction: float, tol: float) -> str | None:
    """The user's rule: a shuffle run not switched at `fraction` of its deadline, its decodability not risen over its
    last 3 checkpoints."""
    if not info["spec"] or info["spec"]["training"] != "shuffle" or not rows or rows[-1]["switched"]:
        return None
    last = rows[-1]
    if last["grad_step"] < fraction * info["deadline"]:
        return None
    dec = [r["decodability"] for r in rows if r["decodability"] is not None]
    if len(dec) >= 3 and dec[-1] - dec[-3] <= tol:
        return (f"ALERT {info['label']}: no switch at gradient step {last['grad_step']} (>= {fraction:.0%} of the deadline "
                f"{info['deadline']}), decodability {dec[-3]:.3f} -> {dec[-1]:.3f} over the last 3 checkpoints")
    return None


def final_row(info: dict) -> dict:
    last = info["records"][-1] if info["records"] else {}
    sw = (last.get("e5") or {}).get("estimator_switch") or {}
    switch_env = sw.get("env_step")
    if switch_env is None and last.get("switch_step") is not None:  # an E4 run: from its log
        for line in (info["dir"] / "console.log").read_text(encoding="utf-8").splitlines():
            if ">>> SWITCH" in line and "environment step" in line:
                switch_env = int(line.split("environment step ")[1].split(")")[0])
                break
    env_end = last.get("env_step")
    return {"run": info["label"], "dir": info["dir"].name, "statistic": info["statistic"], "deadline": info["deadline"],
            "switch_grad_step": last.get("switch_step"), "switch_env_step": switch_env, "last_env_step": env_end,
            "env_steps_after_switch": None if switch_env is None or env_end is None else env_end - switch_env}


def write(rows: list[dict], out: Path | None) -> None:
    if not rows:
        return
    if out is not None:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    keys = list(rows[0])
    print("\t".join(keys))
    for r in rows:
        print("\t".join("" if r[k] is None else (f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])) for k in keys))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("early")
    e.add_argument("--runs", type=Path, nargs="+", required=True)
    e.add_argument("--e4-runs", type=Path, nargs="*", default=[])
    e.add_argument("--e4-decodability", action="store_true")
    e.add_argument("--device", default="cpu")
    e.add_argument("--max-per-state", type=int, default=5, help="E5's e5.decodability.max_per_state")
    e.add_argument("--alert-fraction", type=float, default=0.5)
    e.add_argument("--rise-tol", type=float, default=0.01)
    e.add_argument("--out", type=Path, default=None)
    f = sub.add_parser("final")
    f.add_argument("--runs", type=Path, nargs="+", required=True)
    f.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "final":
        write([final_row(run_info(d)) for d in args.runs], args.out)
        return 0
    rows, alerts = [], []
    for d in args.runs:
        info = run_info(d)
        r = early_rows(info)
        rows += r
        a = alert(info, r, args.alert_fraction, args.rise_tol)
        if a:
            alerts.append(a)
    for d in args.e4_runs:
        info = run_info(d)
        r = early_rows(info)
        dec = e4_decodability(info, [x["env_step"] for x in r], args.device, args.max_per_state) if args.e4_decodability else None
        rows += early_rows(info, dec) if dec is not None else r
    write(rows, args.out)
    for a in alerts:
        print(a, flush=True)
    print(f"{len(alerts)} alert(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
