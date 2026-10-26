"""E7's D6 (docs/e6/DECISIONS.md (61)): the h'_a channel. Frozen weights, training image, the test environment, k = 1;
every model's weights hashed before and after.

Phases (--phase), one output directory per agent, <out-dir>/E7_dissociation/e7d6__<agent>__<phase>__<UTC>/:
  counterfactual  E6's episodes (both models) and plan B's (reverse, reverse_random, every alpha), rerun with a q_fn that
                  computes, on a clone of the act generator (the policy's own 16 draws per action), the variants of
                  probing.lookahead.channel_q -- (a) normal, (b) common h, (c) common z, test 2's V(h'_a, z_bar) -- and
                  returns the policy's own Q (checked equal to (a), bit for bit): the episodes are E6's and plan B's.
  replay          the references' runs (E6's reference, plan B's) with the policy on (b), epsilon-greedy.
Measured in the zone. episodes.npz, summary.json, meta.yaml, status.json.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_d6.py --phase counterfactual --agents configs/e6/agents_dev_e4.yaml --runs-root <root>
      --directions <e7dir dirs or globs> [--agent-ids ...] [--out-dir results]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

import run_e7_apply as ra
import run_e7_diagnostics as rdiag
import run_e7_dissociation as rdis
from helplessness.agents.worldmodel.policy import q_values
from helplessness.config import REPO_ROOT
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.lookahead import channel_q
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_dissociation"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e7" / "e7.yaml"
VARIANTS = ("Q_normal", "Q_hbar", "Q_zbar", "V_zbar")


def run(st, directions: Path, phase: str) -> tuple[dict, list[dict]]:
    e6 = {**st.e6, "measure": "zone"}
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
    ref_plain = fs.run_model(st.models["reference"], st.tw, st.image, e6, st.eps["reference"])
    items = rdiag.induction_items(st, directions)
    if phase == "replay":
        items = [it for it in items if it["role"] == "reference"]
    arrays, rows = {"reference_fresh_I": ref_fresh}, []
    for it in items:
        t0 = time.perf_counter()
        model = st.models[it["role"]]
        hook = None if it["vector"] is None else ra.fixed_hook(it["vector"], it["alpha"], st.e7["apply_in"])
        zgen = torch.Generator(device=st.device).manual_seed(rdis.seed_of(st.d["seed"], "rollout", it["alpha"], 7 + len(it["mode"])))
        buffer: list[dict] = []

        if phase == "counterfactual":
            def q_fn(h, z, s, g, model=model, zgen=zgen, buffer=buffer):
                clone = torch.Generator(device=st.device)
                clone.set_state(g.get_state())
                ch = channel_q(model, h, z, clone, zgen)
                q = q_values(model.rssm, model.goal_head, model.critic, h, z, model.cfg.k_act, model.cfg, g)
                if not torch.equal(q, ch["Q_normal"]):
                    raise RuntimeError("variant (a) differs from the policy's Q")
                buffer.append({k: ch[k].cpu().numpy() for k in VARIANTS})
                return q
        else:
            def q_fn(h, z, s, g, model=model, zgen=zgen):
                return channel_q(model, h, z, g, zgen)["Q_hbar"]  # consumes g as q_values does
        ep = fs.run_model(model, st.tw, st.image, e6, st.eps[it["role"]], hook=hook, q_fn=q_fn)
        row = rdiag.episode_row(st, ep, ref_plain, ref_fresh, it["role"], mode=it["mode"], alpha=it["alpha"], tau=None, k=1)
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        tag = f"{it['mode']}__{it['alpha']:g}"
        for key in ("states", "actions", "length", "terminated", "Q", "I", "loc_template", "p_goal", "explored", "tied"):
            arrays[f"{tag}__{key}"] = ep[key]
        arrays[f"{tag}__b_local"] = fs.b_local(ep, ref_fresh)
        if buffer:
            n, H, A = ep["Q"].shape
            for k in VARIANTS:
                full = np.full((n, H, A), np.nan)
                for t, b in enumerate(buffer):
                    full[np.flatnonzero(ep["length"] > t), t] = b[k]
                arrays[f"{tag}__{k}"] = full
            assert np.array_equal(arrays[f"{tag}__Q_normal"], ep["Q"], equal_nan=True)
        print(f"[D6 {phase}] {st.agent['id']} {tag}: directed {row['directed_share']:.3f}, loc zone {row['loc_template_zone']:.2f}; "
              f"{row['seconds']:.0f} s", flush=True)
    return arrays, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("counterfactual", "replay"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--n-reps", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    if args.n_reps is not None:
        cfg["e6"] = {**cfg["e6"], "n_reps": args.n_reps}
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e7d6__{agent['id']}__{args.phase}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            st = rdiag.Setting(cfg, agent, device)
            before = rdiag.weights_hash(st.models)
            arrays, rows = run(st, ra.find_directions(args.directions, agent["id"]), args.phase)
            after = rdiag.weights_hash(st.models)
            if after != before:
                raise RuntimeError(f"weights moved: {before} -> {after}")
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / "episodes.npz", **arrays)
        (out / "summary.json").write_text(json.dumps({"rows": rows}, indent=1, default=float), encoding="utf-8")
        meta = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "phase": args.phase,
                "image": st.image["name"], "slot": st.image["slot"], "weights": {r: str(p_) for r, p_ in agent["weights"].items()},
                "weights_hash_before": before, "weights_hash_after": after, "agents_file": str(args.agents), "config": str(args.config),
                "config_bases": cfg["_bases"], "device": str(device), "torch_threads": torch.get_num_threads(),
                "torch_version": str(torch.__version__), "seconds": time.perf_counter() - t0, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[D6 {args.phase}] {agent['id']}: done in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
