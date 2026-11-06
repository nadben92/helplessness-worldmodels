"""The twin's start counts of a baseline pair, rebuilt from its start runs' replay_final.npz (docs/e8/DECISIONS.md (20)),
with the forms TW0-TW3 of EXPECTED_FORMS checked on them. Local, CPU, numpy only (the run's environment for the
dynamics). Writes counts_<model>.npz (N_<k> at every checkpoint of the start run's checkpoints.jsonl, N_start = N at the
agents file's checkpoint; bulk arrays, outside the repository) and forms_<pair>.json (the summary).

    python experiments/E8_therapy_relapse/start_counts.py --pair base_B --run base_B_exp=<run dir> --run base_B_ref=<run dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import two_zones_setup
from helplessness.twin.replay_counts import counts_at, replay_transitions


def load_run(run_dir: Path) -> dict:
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    records = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    conf = {**two_zones_setup.configuration(cfg, cfg["e4"]["run_configuration"]), "seed": int(meta["seed"])}
    world = two_zones_setup.world(cfg, conf, meta["condition"])
    replay = np.load(run_dir / "replay_final.npz")
    if "obs" in replay.files:
        raise ValueError(f"{run_dir}: per-episode appearances (a shuffle run): not a baseline")
    P = world.env.transition_tensor()
    t = replay_transitions(*(replay[k] for k in ("s", "a", "terminal", "truncated", "is_first")), P=P)
    zone = two_zones.region_states(world.layout)[two_zones.roles(conf["diagonal"])["ud"]]
    return {"meta": meta, "records": records, "P": P, "t": t, "zone": zone, "condition": meta["condition"]}


def tw0(run: dict) -> dict:
    acc, last, env_steps = run["t"]["accounting"], run["records"][-1], int(run["meta"]["env_steps"])
    checks = {
        "transitions == meta env_steps": acc["transitions"] == env_steps,
        "transitions == last checkpoint env_step": acc["transitions"] == int(last["env_step"]),
        "stored == with_stored_successor + pending + end_observations":
            acc["stored"] == acc["with_stored_successor"] + acc["pending"] + acc["end_observations"],
        "stored == last checkpoint stored_steps": acc["stored"] == int(last["stored_steps"]),
        "end_observations == episodes (checkpoints.jsonl)": acc["end_observations"] == int(last["episodes"]),
        "terminal == successes (checkpoints.jsonl)": acc["terminal"] == int(last["successes"]),
        "episodes_started == episodes + pending": acc["episodes_started"] == int(last["episodes"]) + acc["pending"],
    }  # the flags' sequence (ends followed by first observations) is checked by replay_transitions itself
    return {"ok": all(checks.values()), "checks": checks, "accounting": acc, "env_steps": env_steps}


def zone_successors(run: dict, N: np.ndarray) -> dict[int, dict[int, list[int]]]:
    """Zone cell -> action -> its counted successors."""
    return {int(s): {int(a): np.flatnonzero(N[s, a]).tolist() for a in range(N.shape[1]) if N[s, a].sum()} for s in run["zone"]}


def tw1(run: dict, N: np.ndarray) -> dict:
    P, bad, few = run["P"], [], []
    for s, by_a in zone_successors(run, N).items():
        succ = {int(np.argmax(P[s, a])) for a in range(P.shape[1])}
        if len(succ) != 1 or not all(np.count_nonzero(P[s, a]) == 1 for a in range(P.shape[1])):
            raise ValueError(f"cell {s} of the exposed zone is not a deterministic conveyor in the run's environment")
        bad += [(s, a, x) for a, xs in by_a.items() for x in xs if x not in succ]
        few += [s] if 0 < len(by_a) < 2 else []
    counted = sum(int(N[s].sum()) for s in run["zone"])
    return {"ok": not bad and not few, "zone_cells": len(run["zone"]), "zone_transitions": counted,
            "off_conveyor": bad, "cells_counted_under_fewer_than_2_actions": few}


def tw2(run: dict, N: np.ndarray) -> dict:
    nondeterministic, shared, multi, cells_2plus = [], [], 0, 0
    for s, by_a in zone_successors(run, N).items():
        nondeterministic += [(s, a) for a, xs in by_a.items() if len(xs) != 1]
        firsts = [xs[0] for xs in by_a.values()]
        shared += [s] if len(set(firsts)) != len(firsts) else []
        cells_2plus += len(by_a) >= 2
        multi += len(by_a) >= 2 and len(set(firsts)) >= 2
    return {"ok": not nondeterministic and not shared and multi == cells_2plus, "zone_cells": len(run["zone"]),
            "cells_counted_under_2plus_actions": cells_2plus, "of_which_2plus_successors": multi,
            "action_with_several_successors": nondeterministic, "cells_where_two_actions_share_a_successor": shared}


def tw3(run: dict) -> dict:
    t, P = run["t"], run["P"]
    off = int((P[t["s"], t["a"], t["s_next"]] != 1.0).sum())
    return {"ok": off == 0, "transitions_off_the_dynamics": off}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--run", action="append", required=True, help="model=run directory (the pair's exposed and reference start runs)")
    ap.add_argument("--agents", default=str(REPO_ROOT / "configs" / "e8" / "agents.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    agents = yaml.safe_load(Path(args.agents).read_text(encoding="utf-8"))
    pair = agents["pairs"][args.pair]
    dirs = dict(x.split("=", 1) for x in args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pair": args.pair, "decision": "docs/e8/DECISIONS.md (20)", "models": {}}
    runs = {}
    for role in ("exposed", "reference"):
        mid = pair[role]
        model = agents["models"][mid]
        run_dir = Path(dirs[mid])
        if model["kind"] != "baseline" or Path(model["run"]).name != run_dir.name:
            raise ValueError(f"{mid}: {run_dir} is not its start run {model['run']} (or not a baseline)")
        run = runs[role] = load_run(run_dir)
        expected = "experimental" if role == "exposed" else "reference"
        if run["condition"] != expected:
            raise ValueError(f"{mid}: condition {run['condition']}, expected {expected}")
        counts = {f"N_{int(r['env_step'])}": counts_at(run["t"], int(r["env_step"]), *run["P"].shape[:2]) for r in run["records"]}
        N_start = counts[f"N_{int(model['checkpoint'])}"]
        np.savez_compressed(out / f"counts_{mid}.npz", N_start=N_start, **counts)
        pend = run["t"]["accounting"]["pending"]
        summary["models"][mid] = {"run": model["run"], "condition": run["condition"], "start_checkpoint": int(model["checkpoint"]),
                                  "n_start": int(N_start.sum()), "TW0": tw0(run), "TW3": tw3(run),
                                  "pending_transition": None if not pend else
                                  {k: int(run["t"][k][-1]) for k in ("s", "a", "s_next")}}
        print(f"{mid}: {int(N_start.sum())} transitions at {model['checkpoint']}; TW0 {summary['models'][mid]['TW0']['ok']}, "
              f"TW3 {summary['models'][mid]['TW3']['ok']}", flush=True)
    exp_mid, ref_mid = pair["exposed"], pair["reference"]
    if not np.array_equal(runs["exposed"]["zone"], runs["reference"]["zone"]):
        raise ValueError("the exposed and reference runs' UD zones differ")
    N_exp = np.load(out / f"counts_{exp_mid}.npz")["N_start"]
    N_ref = np.load(out / f"counts_{ref_mid}.npz")["N_start"]
    summary["TW1"], summary["TW2"] = tw1(runs["exposed"], N_exp), tw2(runs["reference"], N_ref)
    summary["ok"] = all([summary["TW1"]["ok"], summary["TW2"]["ok"]] + [m["TW0"]["ok"] and m["TW3"]["ok"] for m in summary["models"].values()])
    (out / f"forms_{args.pair}.json").write_text(json.dumps(summary, indent=1, default=int), encoding="utf-8")
    print(f"TW1 {summary['TW1']['ok']}, TW2 {summary['TW2']['ok']}; all forms {summary['ok']}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
