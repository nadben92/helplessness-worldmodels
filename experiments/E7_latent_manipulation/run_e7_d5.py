"""E7's D5 (docs/e6/DECISIONS.md (58)-(60)): which action is believed ineffective, and the policy's reading of the prior.
Frozen weights, training image, the test environment; every model's weights hashed before and after.

Phases (--phase), one output directory per agent and phase, <out-dir>/E7_dissociation/e7d5__<agent>__<phase>__<tag>__<UTC>/:
  kernels  task 1: E6's episodes of both models and plan B's induction (reverse and reverse_random, every alpha), rerun with
           run_episodes' action_kernel (per step and action: the mass on the true successor, the kernel's argmax right or
           not); the same seeds as E6 and plan B (their arrays are reproduced). episodes.npz, summary.json.
  sampled  task 2: the references with the anti-oracle by draws at k = 1 (probing.lookahead.sampled_mixed_policy_q), p in
           --p-values; --shared for the one-Bernoulli-per-step variant. episodes.npz, summary.json.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_d5.py --phase kernels --agents configs/e6/agents_dev_e4.yaml --runs-root <root>
      --directions <e7dir dirs or globs> [--agent-ids ...] [--out-dir results]
  python experiments/E7_latent_manipulation/run_e7_d5.py --phase sampled --p-values 0 0.1 ... 1 [--shared] ...
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
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.lookahead import embeddings, sampled_mixed_policy_q
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_dissociation"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e7" / "e7.yaml"
KEEP = ("states", "actions", "length", "terminated", "Q", "I", "loc_template", "p_goal", "explored", "tied", "mass_played", "mass_true_all",
        "argmax_true")


def run_kernels(st, directions: Path) -> tuple[dict, list[dict]]:
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
    ref_plain = fs.run_model(st.models["reference"], st.tw, st.image, st.e6, st.eps["reference"])
    arrays, rows = {"reference_fresh_I": ref_fresh}, []
    for it in rdiag.induction_items(st, directions):
        t0 = time.perf_counter()
        hook = None if it["vector"] is None else ra.fixed_hook(it["vector"], it["alpha"], st.e7["apply_in"])
        ep = fs.run_model(st.models[it["role"]], st.tw, st.image, st.e6, st.eps[it["role"]], hook=hook, action_kernel=True)
        row = rdiag.episode_row(st, ep, ref_plain, ref_fresh, it["role"], mode=it["mode"], alpha=it["alpha"], tau=None, k=None)
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        tag = f"{it['mode']}__{it['alpha']:g}"
        for key in KEEP:
            arrays[f"{tag}__{key}"] = ep[key]
        arrays[f"{tag}__b_local"] = fs.b_local(ep, ref_fresh)
        print(f"[D5 kernels] {st.agent['id']} {tag}: directed {row['directed_share']:.3f}, loc zone {row['loc_template_zone']:.2f}; "
              f"{row['seconds']:.0f} s", flush=True)
    return arrays, rows


def run_sampled(st, p_values: list[float], shared: bool) -> tuple[dict, list[dict]]:
    e6 = {**st.e6, "measure": "zone"}
    model, eps = st.models["reference"], st.eps["reference"]
    emb = embeddings(model, st.tw)
    alt = np.repeat(st.conveyor[:, None], model.rssm.cfg.n_actions, axis=1)
    mix = np.zeros(two_zones.N_STATES, dtype=bool)
    mix[st.zone] = True
    ref_fresh = fs.reference_fresh(model, st.tw, st.image, st.e6)
    ref_plain = fs.run_model(model, st.tw, st.image, e6, eps)
    I_ref_step1 = fs.by_state_then_zone(ref_plain["I"][:, 0], ref_plain["states"][:, 0])
    b_eff = rdis.effective_b(st, model, p_values, I_ref_step1, st.conveyor, mix, rdis.seed_of(st.d["seed"], "kernel"))
    arrays, rows = {"reference_fresh_I": ref_fresh}, []
    for p in p_values:
        t0 = time.perf_counter()
        alt_gen = torch.Generator(device=st.device).manual_seed(rdis.seed_of(st.d["seed"], "alt", p, 1 + 100 * int(shared)))

        def q_fn(h, z, s, g, p=p, alt_gen=alt_gen):
            return sampled_mixed_policy_q(model, st.tw, emb, h, z, s, p, mix, alt, g, alt_gen, shared=shared)
        ep = fs.run_model(model, st.tw, st.image, e6, eps, q_fn=q_fn)
        row = rdiag.episode_row(st, ep, ref_plain, ref_fresh, "reference", mode="sampled_shared" if shared else "sampled", alpha=p, tau=None, k=1)
        row.update({"p": p, "shared": shared, "b_eff": b_eff[f"{p:g}"], "seconds": time.perf_counter() - t0})
        rows.append(row)
        tag = f"p{p:g}__k1"
        for key in ("states", "actions", "length", "terminated", "Q", "I", "loc_template", "p_goal", "explored", "tied"):
            arrays[f"{tag}__{key}"] = ep[key]
        arrays[f"{tag}__b_local"] = fs.b_local(ep, ref_fresh)
        print(f"[D5 sampled{' shared' if shared else ''}] {st.agent['id']} p {p:g}: b_eff {row['b_eff']:+.3f}, directed {row['directed_share']:.3f}, "
              f"loc zone {row['loc_template_zone']:.2f}; {row['seconds']:.0f} s", flush=True)
    return arrays, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("kernels", "sampled"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", default=[])
    parser.add_argument("--p-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--shared", action="store_true")
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
    tag = "shared" if args.shared else "main"
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e7d5__{agent['id']}__{args.phase}__{tag}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            st = rdiag.Setting(cfg, agent, device)
            before = rdiag.weights_hash(st.models)
            if args.phase == "kernels":
                arrays, rows = run_kernels(st, ra.find_directions(args.directions, agent["id"]))
            else:
                arrays, rows = run_sampled(st, args.p_values, args.shared)
            after = rdiag.weights_hash(st.models)
            if after != before:
                raise RuntimeError(f"weights moved: {before} -> {after}")
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / "episodes.npz", **arrays)
        (out / "summary.json").write_text(json.dumps({"rows": rows}, indent=1, default=float), encoding="utf-8")
        meta = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "phase": args.phase, "tag": tag,
                "p_values": args.p_values if args.phase == "sampled" else None, "image": st.image["name"], "slot": st.image["slot"],
                "weights": {r: str(p_) for r, p_ in agent["weights"].items()}, "weights_hash_before": before, "weights_hash_after": after,
                "agents_file": str(args.agents), "config": str(args.config), "config_bases": cfg["_bases"], "diagnostics": st.d,
                "device": str(device), "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                "seconds": time.perf_counter() - t0, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[D5 {args.phase}] {agent['id']} {tag}: done in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
