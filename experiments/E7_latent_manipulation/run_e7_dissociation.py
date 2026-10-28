"""E7 as a belief x critic dissociation (docs/e6/DECISIONS.md (55)): the anti-oracle. The PLANNER of a model sees, from the
zone AI, the conveyor of the paired exposed environment with probability p (k = 1: the policy's Q mixed as
(1 - p) Q_model + p Q_conveyor, probing.lookahead.mixed_policy_q; k > 1: k_step_q's mixed variant). Nothing of h, z, the
decoder or the heads is touched; the real environment stays controllable. E6's episodes (training image, starts, epsilon,
streams), measured in the zone. Every model's weights are hashed before and after (the run fails if they moved).

Per run: the effective b at step 1 for every p of the run (the model's decoded kernel at E6's t = 0 contexts, mixed with
the conveyor's cell, minus the reference's I at step 1), and per (p, k) the episodes.

Outputs: <out-dir>/E7_dissociation/e7dis__<agent>__<role>__<tag>__<UTC>/: episodes.npz, summary.json, meta.yaml, status.json.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_dissociation.py --agents configs/e6/agents_dev_e4.yaml --runs-root <root>
      --role reference --p-values 0 0.1 0.2 --ks 1 5 [--agent-ids ...] [--tag a] [--out-dir results]
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

import run_e7_diagnostics as rdiag
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline.measure import controllability_of_kernel
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.fixed_trajectory import step_controllability
from helplessness.probing.lookahead import embeddings, k_step_q, mixed_policy_q
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_dissociation"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e7" / "e7.yaml"
PURPOSE = {"alt": 10, "kernel": 11, "rollout": 12}


def seed_of(base: int, purpose: str, *extra) -> int:
    return int(np.random.SeedSequence([int(base), PURPOSE[purpose], *[int(round(1000 * x)) for x in extra]])
               .generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def step1_contexts(st, model) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """E6's t = 0 contexts of the image's episodes, rebuilt exactly as run_episodes draws them (the first draw of the act
    generator: the posterior at the start image after the initial state)."""
    s = fs.seeds(st.e6["seed"], st.image)
    starts = fs.starts_for(st.image["slot"], st.e6["n_reps"])
    n, A = len(starts), model.rssm.cfg.n_actions
    h, z = model.rssm.initial_state(n, st.device)
    gen = torch.Generator(device=st.device).manual_seed(int(s["act"]))
    with torch.no_grad():
        out = model.rssm.observation_step(h, z, torch.zeros(n, A, device=st.device), st.tw.templates_t[torch.as_tensor(starts, device=st.device)],
                                          torch.ones(n, dtype=torch.bool, device=st.device), generator=gen)
    return out["h"], out["z"], starts


def effective_b(st, model, p_values: list[float], I_ref_step1: float, alt: np.ndarray, mix: np.ndarray, seed: int) -> dict:
    """b_eff(p): the model's decoded s-space kernel at E6's t = 0 contexts, mixed on the zone's states as
    (1 - p) p_hat + p delta(conveyor[s]); I by controllability_of_kernel; by state then zone minus the reference's I at
    step 1."""
    h, z, starts = step1_contexts(st, model)
    k = step_controllability(model.rssm, h, z, st.tw.templates_t, int(st.e6["draws"]), float(st.e6["margin"]),
                             generator=torch.Generator(device=st.device).manual_seed(seed), return_kernel=True)
    p_hat = k["p_hat"]
    delta = np.zeros_like(p_hat)
    rows, acts = np.meshgrid(np.arange(len(starts)), np.arange(p_hat.shape[1]), indexing="ij")
    delta[rows, acts, alt[starts][:, None].repeat(p_hat.shape[1], 1)] = 1.0
    on = mix[starts][:, None, None]
    out = {"I_kernel_p0": fs.by_state_then_zone(k["I"], starts)}
    for p in p_values:
        I = controllability_of_kernel(np.where(on, (1.0 - p) * p_hat + p * delta, p_hat))
        out[f"{p:g}"] = fs.by_state_then_zone(I, starts) - I_ref_step1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--role", choices=("reference", "experimental"), required=True)
    parser.add_argument("--p-values", nargs="+", type=float, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--tag", default="all")
    parser.add_argument("--h-mode", choices=("mean", "action"), default="mean", help="the conveyor branch's h ((57)): mean (main) or h'_a")
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
        out = args.out_dir / EXPERIMENT_ID / f"e7dis__{agent['id']}__{args.role}__{args.h_mode}__{args.tag}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            st = rdiag.Setting(cfg, agent, device)
            before = rdiag.weights_hash(st.models)
            d = st.d
            e6 = {**st.e6, "measure": "zone"}
            model, eps = st.models[args.role], st.eps[args.role]
            emb = embeddings(model, st.tw)
            alt = np.repeat(st.conveyor[:, None], model.rssm.cfg.n_actions, axis=1)
            mix = np.zeros(two_zones.N_STATES, dtype=bool)
            mix[st.zone] = True
            ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
            ref_plain = fs.run_model(st.models["reference"], st.tw, st.image, e6, st.eps["reference"])  # E6's reference
            I_ref_step1 = fs.by_state_then_zone(ref_plain["I"][:, 0], ref_plain["states"][:, 0])
            b_eff = effective_b(st, model, args.p_values, I_ref_step1, st.conveyor, mix, seed_of(d["seed"], "kernel"))
            arrays, rows = {"reference_fresh_I": ref_fresh}, []
            for k in args.ks:
                for p in args.p_values:
                    t1 = time.perf_counter()
                    alt_gen = torch.Generator(device=device).manual_seed(seed_of(d["seed"], "alt", p, k))
                    if k == 1:
                        def q_fn(h, z, s, g, p=p, alt_gen=alt_gen):
                            return mixed_policy_q(model, st.tw, emb, h, z, s, p, mix, alt, g, alt_gen, h_mode=args.h_mode)
                    else:
                        def q_fn(h, z, s, g, p=p, k=k, alt_gen=alt_gen):
                            return k_step_q(model, st.tw, h, z, s, k, "mixed", int(d["n_particles"]), int(d["k_inner"]), alt_gen, embeds=emb,
                                            p_mix=p, mix=mix, alt_succ=alt, h_mode=args.h_mode)["Q"][..., k - 1]
                    ep = fs.run_model(model, st.tw, st.image, e6, eps, q_fn=q_fn)
                    row = rdiag.episode_row(st, ep, ref_plain, ref_fresh, args.role, mode="anti_oracle", alpha=p, tau=None, k=k)
                    row.update({"p": p, "h_mode": args.h_mode, "b_eff": b_eff[f"{p:g}"], "seconds": time.perf_counter() - t1})
                    rows.append(row)
                    tag = f"p{p:g}__k{k}"
                    for key in ("states", "actions", "length", "terminated", "Q", "I", "loc_template", "p_goal", "explored", "tied"):
                        arrays[f"{tag}__{key}"] = ep[key]
                    arrays[f"{tag}__b_local"] = fs.b_local(ep, ref_fresh)
                    print(f"[E7 dis] {agent['id']} {args.role} p {p:g} k {k}: b_eff {row['b_eff']:+.3f}, directed {row['directed_share']:.3f}, "
                          f"loc zone {row['loc_template_zone']:.2f}, p_goal {row['p_goal_zone']:.3f}; {row['seconds']:.0f} s", flush=True)
            after = rdiag.weights_hash(st.models)
            if after != before:
                raise RuntimeError(f"weights moved: {before} -> {after}")
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / "episodes.npz", **arrays)
        summary = {"rows": rows, "b_eff": b_eff, "I_ref_step1": I_ref_step1}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
        meta = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "role": args.role, "h_mode": args.h_mode,
                "p_values": args.p_values, "ks": args.ks, "tag": args.tag, "image": st.image["name"], "slot": st.image["slot"],
                "weights": {r: str(p_) for r, p_ in agent["weights"].items()}, "weights_hash_before": before, "weights_hash_after": after,
                "agents_file": str(args.agents), "config": str(args.config), "config_bases": cfg["_bases"], "diagnostics": d,
                "device": str(device), "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                "seconds": time.perf_counter() - t0, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[E7 dis] {agent['id']} {args.role} {args.tag}: done in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
