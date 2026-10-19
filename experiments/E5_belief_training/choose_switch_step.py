"""Proposes the immunisation's switch_step (docs/e5/DECISIONS.md, (6)) from the training records of E4's REFERENCE run of
the configuration retained for E5: the FIRST checkpoint of E4's grid at which the future UD slot (controllable in the
reference) meets E4's convergence thresholds as a reference can meet them --

  mass on the true successor at that slot >= --mass-min (0.97, E4's), from the real-context record;
  |I at the first step in that slot - log2 5| <= --i-tol (0.1 bit, E4's |b| threshold at CD and in the corridor), from
  the fixed walks on the training image (I_curve[0]);

(a reference has no b: its I at the slot is compared with the true value, log2 5). The first checkpoint that meets both,
not two consecutive ones (user, 2026-10-19). switch_step stays a parameter of the batch: this script only proposes it,
and draws the two curves for a visual check (switch_step.png).

Records: the run's checkpoints.jsonl ("e4" at every checkpoint: E4's wave 1). A run with diagnostics at a few
checkpoints only (E4's wave 2, --diagnostics-at) gets the missing ones recomputed with --recompute, on the cluster where
its weights are (checkpoints/agent_env<N>.pt at every grid point) with its replay_final.npz: the replay as of the
checkpoint is its first stored_steps steps (recorded at every checkpoint), and the measures are E4's own functions
(pipeline.two_zones_records) with E4's measurement slots and seeds keyed by the checkpoint, so a recomputed record is the
record the training would have written on the same device (--verify compares them where both exist).

Usage, from the repo root:
  python experiments/E5_belief_training/choose_switch_step.py --run <E4 reference run dir> --out <dir>
  python experiments/E5_belief_training/choose_switch_step.py --run <dir> --recompute [--verify] [--device cuda] --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import real_context_record, walks_record
from helplessness.probing.theme_probe_measure import load_rssm

LOG2_5 = float(np.log2(5))


def read_records(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def recompute(run_dir: Path, cfg: dict, records: list[dict], device, only_missing: bool = True, log=print) -> dict[int, dict]:
    """E4's checkpoint record (real_context + walks) recomputed from the weights and the replay prefix."""
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    seed, condition = int(meta["seed"]), meta["condition"]
    conf = two_zones_setup.configuration(cfg, cfg["e4"]["run_configuration"])
    world = two_zones_setup.world(cfg, conf, condition)
    probe = two_zones_setup.probe_world(cfg, conf)
    world.templates_t, probe.templates_t = world.templates_t.to(device), probe.templates_t.to(device)
    data = dict(np.load(run_dir / "replay_final.npz"))
    ck = cfg["e4"]["checkpoint"]
    out = {}
    for r in records:
        if only_missing and r.get("e4"):
            continue
        step, stored = int(r["env_step"]), int(r["stored_steps"])
        t0 = time.perf_counter()
        replay = ReplayBuffer.from_compact({k: v[:stored] for k, v in data.items()}, world.templates_np)
        rssm = load_rssm(run_dir / "checkpoints" / f"agent_env{step:06d}.pt", device)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            out[step] = {"real_context": real_context_record(rssm, world, replay, cfg, seed, condition, step, device,
                                                             ck["real_context_max_per_state"], ck["real_context_draws"],
                                                             two_zones_setup.SLOT_CHECKPOINT_REAL_CONTEXT),
                         "walks": walks_record(rssm, cfg, conf, probe, ck["n_walks"], "checkpoint"), "recomputed": True}
        log(f"recomputed checkpoint {step} in {time.perf_counter() - t0:.0f} s")
    return out


def choose(records: dict[int, dict], ud_slot: str, mass_min: float, i_tol: float) -> dict:
    rows = []
    for step in sorted(records):
        e4 = records[step]
        mass = float(e4["real_context"]["mass_true_by_region"][ud_slot])
        i1 = float(e4["walks"][ud_slot]["I_curve"][0])
        rows.append({"env_step": step, "mass_ud_slot": mass, "I_step1_ud_slot": i1, "mass_ok": mass >= mass_min,
                     "I_ok": abs(i1 - LOG2_5) <= i_tol, "recomputed": bool(e4.get("recomputed"))})
        rows[-1]["ok"] = rows[-1]["mass_ok"] and rows[-1]["I_ok"]
    first = next((r["env_step"] for r in rows if r["ok"]), None)
    return {"switch_step": first, "rows": rows, "criterion": {"slot": ud_slot, "mass_min": mass_min, "I_target": LOG2_5, "I_tol": i_tol,
                                                               "rule": "first checkpoint meeting both"}}


def plot(choice: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, crit = choice["rows"], choice["criterion"]
    x = [r["env_step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(x, [r["I_step1_ud_slot"] for r in rows], marker="o")
    axes[0].axhspan(crit["I_target"] - crit["I_tol"], crit["I_target"] + crit["I_tol"], color="g", alpha=0.15)
    axes[0].set_title(f"I at the first step in the future UD slot ({crit['slot']}), reference")
    axes[1].plot(x, [r["mass_ud_slot"] for r in rows], marker="o")
    axes[1].axhline(crit["mass_min"], color="k", ls=":")
    axes[1].set_title("mass on the true successor at that slot, reference")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("environment steps")
        if choice["switch_step"] is not None:
            ax.axvline(choice["switch_step"], color="r", ls="--")
        for r in rows:
            if r["recomputed"]:
                ax.axvline(r["env_step"], color="0.8", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="E4's reference run of the configuration (its directory)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mass-min", type=float, default=0.97)
    parser.add_argument("--i-tol", type=float, default=0.1)
    parser.add_argument("--recompute", action="store_true", help="recompute the checkpoints without diagnostics (weights needed)")
    parser.add_argument("--verify", action="store_true", help="with --recompute: also recompute the recorded ones and compare")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    meta = yaml.safe_load((args.run / "meta.yaml").read_text(encoding="utf-8"))
    if meta["condition"] != "reference":
        raise SystemExit(f"{args.run} is a {meta['condition']} run: the switch step is read on the reference")
    conf = two_zones_setup.configuration(cfg, cfg["e4"]["run_configuration"])
    records = read_records(args.run)
    by_step = {int(r["env_step"]): r["e4"] for r in records if r.get("e4")}
    args.out.mkdir(parents=True, exist_ok=True)
    verify = None
    if args.recompute:
        cfg["runtime"]["device"] = args.device
        device = configure_runtime(cfg)
        new = recompute(args.run, cfg, records, device, only_missing=not args.verify)
        if args.verify:
            verify = {}
            for step, rec in new.items():
                if step in by_step:
                    old = by_step[step]
                    verify[step] = {"mass_ud": [old["real_context"]["mass_true_by_region"][conf["roles"]["ud"]],
                                                rec["real_context"]["mass_true_by_region"][conf["roles"]["ud"]]],
                                    "I_step1_ud": [old["walks"][conf["roles"]["ud"]]["I_curve"][0], rec["walks"][conf["roles"]["ud"]]["I_curve"][0]]}
        by_step = {**new, **by_step}
        (args.out / "recomputed_records.json").write_text(json.dumps({str(k): v for k, v in new.items()}, indent=1), encoding="utf-8")
    missing = [int(r["env_step"]) for r in records if int(r["env_step"]) not in by_step]
    choice = choose(by_step, conf["roles"]["ud"], args.mass_min, args.i_tol)
    choice.update(run=str(args.run), configuration=conf["id"], seed=int(meta["seed"]), checkpoints_without_diagnostics=missing,
                  verify=verify)
    (args.out / "switch_step.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    plot(choice, args.out / "switch_step.png")
    print(json.dumps({k: choice[k] for k in ("switch_step", "configuration", "seed", "checkpoints_without_diagnostics")}), flush=True)
    if missing:
        print(f"WARNING: {len(missing)} checkpoint(s) without diagnostics ({missing}); the first may be among them: --recompute", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
