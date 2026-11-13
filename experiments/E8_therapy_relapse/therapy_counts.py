"""The tabular twin of a baseline pair's therapy (docs/e8/DECISIONS.md (20), (22); forms: docs/e8/EXPECTED_FORMS.md
section 4). Local, CPU, numpy only (the therapy world for the dynamics).

  1. TW4: each therapy run's counts.npz (N_<k>, written by run_one) against the counts rebuilt from its replay_final.npz
     (twin.replay_counts, DECISIONS (20)'s rule; the pending transition completed by the THERAPY world's dynamics),
     np.array_equal at every checkpoint. A difference stops here (exit code 2), nothing else is computed.
  2. TW0, TW3 (start_counts.py's checks) and TW2 on both runs: nw is controllable in the therapy, for the exposed run as
     for the reference.
  3. The mixture: N_mix(k) = N_start + N_<k> (N_start from start_counts.py's counts_<model>.npz), checked equal to the
     array the twin is built from, then E2's twin through pipeline.agent_run.twin_at_checkpoint (the start run's seed and
     condition, env_step = n_start + k: DECISIONS (22)); b_tab = the twin's per-state b averaged over nw's cells.

Writes forms_therapy_<pair>.json and twin_<pair>.csv (light: per run and checkpoint, the visits, n_mix, b_tab and I_hat in
nw) into --out.

    python experiments/E8_therapy_relapse/therapy_counts.py --pair base_A --run base_A_exp=<therapy run dir> \\
        --run base_A_ref=<therapy run dir> --start-counts <dir of counts_<model>.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

import start_counts
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import agent_config, twin_at_checkpoint
from helplessness.twin.replay_counts import counts_at, replay_transitions

START_CONDITION = {"exposed": "experimental", "reference": "reference"}  # the start runs' conditions (their twin streams)


def load_therapy_run(run_dir: Path) -> dict:
    """The run's records, its therapy world's dynamics, its counts.npz and the transitions rebuilt from its replay."""
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    records = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    conf = {**two_zones_setup.configuration(cfg, cfg["e4"]["run_configuration"]), "seed": int(meta["seed"])}
    world = two_zones_setup.world(cfg, conf, "reference")  # e8_therapy.make_hooks' world, for both roles
    replay = np.load(run_dir / "replay_final.npz")
    P = world.env.transition_tensor()
    t = replay_transitions(*(replay[k] for k in ("s", "a", "terminal", "truncated", "is_first")), P=P)
    zone = two_zones.region_states(world.layout)[two_zones.roles(conf["diagonal"])["ud"]]
    counts = np.load(run_dir / "counts.npz")
    return {"meta": meta, "cfg": cfg, "records": records, "world": world, "P": P, "t": t, "zone": zone, "counts": counts,
            "seed": int(meta["seed"])}


def tw4(run: dict) -> dict:
    n_s, n_a = run["P"].shape[:2]
    by_k = {}
    for r in run["records"]:
        k = int(r["env_step"])
        rebuilt = counts_at(run["t"], k, n_s, n_a)
        stored = run["counts"][f"N_{k}"]
        by_k[k] = bool(stored.shape == rebuilt.shape and np.array_equal(stored, rebuilt))
    missing = sorted(set(run["counts"].files) - {f"N_{k}" for k in by_k})
    return {"ok": all(by_k.values()) and not missing, "equal_by_checkpoint": by_k, "keys_without_record": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--run", action="append", required=True, help="model=therapy run directory (the pair's exposed and reference runs)")
    ap.add_argument("--start-counts", type=Path, required=True, help="start_counts.py's output directory of the pair")
    ap.add_argument("--agents", default=str(REPO_ROOT / "configs" / "e8" / "agents.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    pair = yaml.safe_load(Path(args.agents).read_text(encoding="utf-8"))["pairs"][args.pair]
    dirs = dict(x.split("=", 1) for x in args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"pair": args.pair, "decisions": "docs/e8/DECISIONS.md (20), (22)", "models": {}}
    runs = {}
    for role in ("exposed", "reference"):
        mid = pair[role]
        run = runs[role] = load_therapy_run(Path(dirs[mid]))
        if run["cfg"]["e8"]["run"]["model"] != mid:
            raise ValueError(f"{dirs[mid]} is the run of {run['cfg']['e8']['run']['model']}, not {mid}")
        summary["models"][mid] = {"run": Path(dirs[mid]).name, "TW4": tw4(run)}
        print(f"{mid}: TW4 {summary['models'][mid]['TW4']['ok']}", flush=True)
    if not all(m["TW4"]["ok"] for m in summary["models"].values()):
        (args.out / f"forms_therapy_{args.pair}.json").write_text(json.dumps(summary, indent=1, default=int), encoding="utf-8")
        print("TW4 FAILED: counts.npz differs from the replay's counts; the twin stops here")
        return 2
    if not np.array_equal(runs["exposed"]["zone"], runs["reference"]["zone"]):
        raise ValueError("the exposed and reference runs' UD zones differ")
    rows = []
    for role, run in runs.items():
        mid = pair[role]
        m = summary["models"][mid]
        N_T = run["counts"][f"N_{int(run['records'][-1]['env_step'])}"]
        m["TW0"], m["TW3"], m["TW2"] = start_counts.tw0(run), start_counts.tw3(run), start_counts.tw2(run, N_T)
        pend = run["t"]["accounting"]["pending"]
        m["pending_transition"] = None if not pend else {k: int(run["t"][k][-1]) for k in ("s", "a", "s_next")}
        N_start = np.load(args.start_counts / f"counts_{mid}.npz")["N_start"]
        n_start = int(N_start.sum())
        acfg = agent_config(run["cfg"], run["cfg"]["episode"])
        m["n_start"], m["mixture_equal"] = n_start, {}
        for r in run["records"]:
            k = int(r["env_step"])
            N_mix = N_start + run["counts"][f"N_{k}"]
            built = {}
            twin_at_checkpoint(N_mix, run["world"], run["P"], run["cfg"], acfg, run["seed"], START_CONDITION[role], n_start + k, built)
            m["mixture_equal"][k] = bool(np.array_equal(N_mix, N_start + counts_at(run["t"], k, *run["P"].shape[:2])))
            zone = run["zone"]
            n_zone = N_mix[zone].sum(axis=2)
            rows.append({"pair": args.pair, "model": mid, "role": role, "checkpoint": k, "grad_step": int(r["grad_step"]),
                         "treated_visits": int(r["e8"]["visits"]["treated"]), "n_mix": int(N_mix.sum()),
                         "n_therapy_nw": int(run["counts"][f"N_{k}"][zone].sum()), "n_start_nw": int(N_start[zone].sum()),
                         "n_mix_nw_min_pair": int(n_zone.min()),
                         "b_tab_nw": float(np.mean(built[f"b_{n_start + k}"][zone])),
                         "I_hat_nw": float(np.mean(built[f"I_hat_{n_start + k}"][zone])),
                         "I_ref_nw": float(np.mean(built[f"I_ref_{n_start + k}"][zone]))})
        print(f"{mid}: TW0 {m['TW0']['ok']}, TW2 {m['TW2']['ok']}, TW3 {m['TW3']['ok']}, mixture {all(m['mixture_equal'].values())}", flush=True)
    summary["ok"] = all(m["TW4"]["ok"] and m["TW0"]["ok"] and m["TW2"]["ok"] and m["TW3"]["ok"] and all(m["mixture_equal"].values())
                        for m in summary["models"].values())
    (args.out / f"forms_therapy_{args.pair}.json").write_text(json.dumps(summary, indent=1, default=int), encoding="utf-8")
    with open(args.out / f"twin_{args.pair}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"all forms {summary['ok']}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
