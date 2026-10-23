"""E6's measurement of one or more agents (docs/E6_phenotypes_latent.md, docs/e6/DECISIONS.md): for every test image of
configs/e6/e6.yaml (e6.images), the episodes of the exposed model and of its reference in the probe environment (the
reference's dynamics, the image), started in the central 3 x 3 of the zone of interest, frozen weights; the belief at
step 1 and the phenotypes; E4's validity probe of the (agent, image) and the coherence of b at step 1 with it. The
library does everything (pipeline.frozen_setup, probing.frozen_episodes, measures.zone_phenotypes); this script only
orchestrates and writes.

Outputs, one directory per agent: <out-dir>/E6_phenotypes_latent/e6__<agent id>__<UTC>/
  rows.csv        one row per (image, role): role exposed (b_step1 = I_exposed - I_reference at step 1) and reference
                  (b_step1 = 0 by construction); validity (E4's criterion, the reference's probe), b_step1_probe (E4's
                  probe: corridor walk, then the zone's border) and coherence_gap = b_step1 - b_step1_probe; localisation;
                  the phenotypes with their bootstrap intervals; by-index curves (JSON lists)
  episodes__<image>.npz   every per-step array of both models' episodes, b_local, the reference's fresh-start I
  meta.yaml, status.json  the agent, its weights (path, SHA-256), the config, the code's provenance, timings

Usage, from the repo root:
  python experiments/E6_phenotypes_latent/run_e6.py --agents configs/e6/agents_dev_e4.yaml --runs-root <root> [--agent-ids ID ...]
      [--images training ...] [--config configs/e6/e6.yaml] [--device cpu|cuda] [--out-dir results] [--n-reps N] [--no-bootstrap]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.config import REPO_ROOT
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.frozen_episodes import load_frozen_agent
from helplessness.probing.theme_probe_measure import sha256
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E6_phenotypes_latent"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e6" / "e6.yaml"
ROLES = {"experimental": "exposed", "reference": "reference"}


def load_config(path: Path) -> dict:
    return fs.load_layered_config(path, ("e6", "e7"))


def measure_agent(cfg: dict, e6: dict, agent: dict, device, images_only: list[str] | None, bootstrap: bool, log=print) -> tuple[list[dict], dict, dict]:
    """Every test image of one agent. Returns the rows, the arrays (name -> array, for the npz of each image) and timings."""
    conf = {**setup.configuration(cfg, agent["configuration"]), "seed": agent["seed"]}
    images = fs.test_images(conf, e6["images"])
    if images_only:
        images = [im for im in images if im["name"] in images_only]
    models = {role: load_frozen_agent(agent["weights"][role], device) for role in ("experimental", "reference")}
    eps = {role: (float(e6["epsilon"]) if e6.get("epsilon") is not None else m.cfg.epsilon) for role, m in models.items()}
    rows, arrays, timings = [], {}, {}
    for im in images:
        t0 = time.perf_counter()
        _, tw = fs.build_test_world(cfg, conf, im["slot_colors"], device)
        ep = {role: fs.run_model(m, tw, im, e6, eps[role]) for role, m in models.items()}
        t1 = time.perf_counter()
        ref_fresh = fs.reference_fresh(models["reference"], tw, im, e6)
        t2 = time.perf_counter()
        val = fs.probe_validity_pair(cfg, conf, {r: m.rssm for r, m in models.items()}, im, int(e6["validity"]["n_walks"]), device)
        t3 = time.perf_counter()
        timings[im["name"]] = {"episodes_s": t1 - t0, "reference_fresh_s": t2 - t1, "validity_s": t3 - t2}
        H = int(e6["horizon"])
        for role, m in models.items():
            belief = fs.belief_row(ep[role], ep["reference"], ref_fresh, im)
            if role == "reference":
                belief["b_step1"] = 0.0  # I_ref - I_ref on the same episodes: zero by construction
            ph = fs.phenotype_row(ep[role], im, eps[role], e6, with_bootstrap=bootstrap)
            bl = fs.b_local_by_index(ep[role], ref_fresh, im, H)
            rows.append({"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "seed": agent["seed"],
                         "image": im["name"], "image_index": im["index"], "slot": im["slot"], "position": im["role"], "role": ROLES[role],
                         "epsilon": eps[role], **{k: v for k, v in val.items()}, "coherence_gap": belief["b_step1"] - val["b_step1_probe"] if role == "experimental" else float("nan"),
                         **belief, **ph, "b_local_by_index": [None if not np.isfinite(v) else float(v) for v in bl["mean"]],
                         "b_local_by_index_n": bl["n"].tolist()})
            for key, value in ep[role].items():
                arrays[f"{im['name']}__{ROLES[role]}__{key}"] = value
            arrays[f"{im['name']}__{ROLES[role]}__b_local"] = fs.b_local(ep[role], ref_fresh)
        arrays[f"{im['name']}__reference_fresh_I"] = ref_fresh
        exp_row = rows[-2]
        log(f"[E6] {agent['id']} {im['name']} ({im['slot']}): valid {val['valid']}, b_step1 {exp_row['b_step1']:+.3f} "
            f"(probe {val['b_step1_probe']:+.3f}); directed exposed {exp_row['directed_share']:.3f} / reference {rows[-1]['directed_share']:.3f}; "
            f"{t3 - t0:.0f} s")
    return rows, arrays, timings


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True, help="the agents file (configs/e6/agents_*.yaml)")
    parser.add_argument("--runs-root", type=Path, required=True, help="the root the agents file's run globs are relative to")
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--images", nargs="+", default=None, help="a subset of e6.images by name")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--n-reps", type=int, default=None, help="override e6.n_reps (quick runs)")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    e6 = fs.e6_protocol(cfg)
    if args.n_reps is not None:
        e6["n_reps"] = args.n_reps
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e6__{agent['id']}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            rows, arrays, timings = measure_agent(cfg, e6, agent, device, args.images, not args.no_bootstrap)
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        write_rows(out / "rows.csv", rows)
        (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
        np.savez_compressed(out / "episodes.npz", **arrays)
        meta = {"agent": {k: (str(v) if k == "weights" else v) for k, v in agent.items() if k not in ("experimental", "reference", "weights")},
                "weights": {r: str(p) for r, p in agent["weights"].items()}, "weights_sha256": {r: sha256(p) for r, p in agent["weights"].items()},
                "agents_file": str(args.agents), "config": str(args.config), "config_bases": cfg["_bases"], "e6": {k: v for k, v in e6.items()},
                "device": str(device), "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                "seconds": time.perf_counter() - t0, "timings": timings, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[E6] {agent['id']}: {len(rows)} rows in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
