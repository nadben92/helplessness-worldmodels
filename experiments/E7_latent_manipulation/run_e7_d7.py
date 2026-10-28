"""E7's D7 (docs/e6/DECISIONS.md (62)): b against the controllability the critic reads. Frozen weights, training image, the
test environment, k = 1; every model's weights hashed before and after.

Phases (--phase), one output directory per agent (and setting), <out-dir>/E7_dissociation/e7d7__<agent>__<phase>__<tag>__<UTC>/:
  natural    E6's episodes (both models) and plan B's (the overlay), rerun with D6's counterfactual q_fn and the second
             z_bar (channel_q's mean_zbar): per zone step V(h'_a, z_bar) for both z_bar and Q of variant (b). The returned Q
             is the policy's own (checked bit for bit): the episodes are E6's and plan B's.
  corrected  the exposed model with the constrained stepwise correction of plan A at alpha = 1 (run_e7_apply.stepwise_hook,
             unchanged), --lam / --mu-pos (mu_goal from the agent's sweep); the hook is wrapped to keep the uncorrected h and
             measure its I; the q_fn computes the same quantities on the corrected and the uncorrected h from clones of the
             act and z_bar generators, and returns the policy's own Q on the corrected h.
Outputs: episodes.npz, summary.json, meta.yaml, status.json.

Options for E6/E7 on all agents (docs/e6/DECISIONS.md (63); the defaults reproduce D7): --image <name> (natural only) runs
on that test image of the config's e6.images instead of e7.diagnostics.image (training); --no-overlay leaves plan B out
(E6's two models only; no --directions needed).

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_d7.py --phase natural --agents configs/e6/agents_dev_e4.yaml --runs-root <root>
      --directions <e7dir dirs or globs> [--agent-ids ...] [--out-dir results]
  python experiments/E7_latent_manipulation/run_e7_d7.py --phase natural --no-overlay --image <name> --config configs/e7/e7_all.yaml ...
  python experiments/E7_latent_manipulation/run_e7_d7.py --phase corrected --lam 0.01 --mu-pos 1 --constrained <e7con dirs or globs> ...
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
from helplessness.probing.fixed_trajectory import step_controllability
from helplessness.probing.lookahead import channel_q
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_dissociation"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e7" / "e7.yaml"
KEYS = ("V_zbar", "V_zbar_mean", "Q_hbar")
SAVE = ("states", "actions", "length", "terminated", "Q", "I", "loc_template", "p_goal", "explored", "tied")


def clone(g: torch.Generator, device) -> torch.Generator:
    c = torch.Generator(device=device)
    c.set_state(g.get_state())
    return c


def assemble(buffer: list[dict], length: np.ndarray, shape: tuple, prefix: str) -> dict:
    """The per-call buffers (one per step, the active rows in order) as (n, H, A) arrays."""
    out = {}
    for k in buffer[0]:
        full = np.full(shape, np.nan)
        for t, b in enumerate(buffer):
            full[np.flatnonzero(length > t), t] = b[k]
        out[f"{prefix}{k}"] = full
    return out


def natural(st, directions: Path | None) -> tuple[dict, list[dict]]:
    """directions None: E6's two models only (no plan B overlay)."""
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
    e6 = {**st.e6, "measure": "zone"}
    ref_plain = fs.run_model(st.models["reference"], st.tw, st.image, e6, st.eps["reference"])
    arrays, rows = {"reference_fresh_I": ref_fresh}, []
    for it in rdiag.induction_items(st, directions, plan_b=directions is not None):
        t0 = time.perf_counter()
        model = st.models[it["role"]]
        hook = None if it["vector"] is None else ra.fixed_hook(it["vector"], it["alpha"], st.e7["apply_in"])
        zgen = torch.Generator(device=st.device).manual_seed(rdis.seed_of(st.d["seed"], "rollout", it["alpha"], 7 + len(it["mode"])))  # D6's
        buffer: list[dict] = []

        def q_fn(h, z, s, g, model=model, zgen=zgen, buffer=buffer):
            ch = channel_q(model, h, z, clone(g, st.device), zgen, mean_zbar=True)
            q = q_values(model.rssm, model.goal_head, model.critic, h, z, model.cfg.k_act, model.cfg, g)
            if not torch.equal(q, ch["Q_normal"]):
                raise RuntimeError("the counterfactual's Q differs from the policy's")
            buffer.append({k: ch[k].cpu().numpy() for k in KEYS})
            return q
        ep = fs.run_model(model, st.tw, st.image, e6, st.eps[it["role"]], hook=hook, q_fn=q_fn)
        row = rdiag.episode_row(st, ep, ref_plain, ref_fresh, it["role"], mode=it["mode"], alpha=it["alpha"], tau=None, k=1)
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        tag = f"{it['mode']}__{it['alpha']:g}"
        arrays.update({f"{tag}__{k}": ep[k] for k in SAVE})
        arrays[f"{tag}__b_local"] = fs.b_local(ep, ref_fresh)
        arrays.update(assemble(buffer, ep["length"], ep["Q"].shape, f"{tag}__"))
        print(f"[D7 natural] {st.agent['id']} {tag}: directed {row['directed_share']:.3f}; {row['seconds']:.0f} s", flush=True)
    return arrays, rows


def corrected(st, setting: dict) -> tuple[dict, list[dict]]:
    model = st.models["experimental"]
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, st.e6)
    ref_plain = fs.run_model(st.models["reference"], st.tw, st.image, st.e6, st.eps["reference"])
    inner = ra.stepwise_hook(model, st.tw, st.e7, 1.0, st.e7["apply_in"], setting["lam"], setting["mu_pos"], setting["mu_goal"],
                             float(st.e7["constrained"]["tau"]))
    raw_gen = torch.Generator(device=st.device).manual_seed(rdis.seed_of(st.d["seed"], "kernel", 7.0))
    zgen = torch.Generator(device=st.device).manual_seed(rdis.seed_of(st.d["seed"], "rollout", 1.0, 70))
    keep: dict = {}
    buffer: list[dict] = []

    def hook(t, h, z, states, in_zone, rows):
        keep["h_raw"] = h.clone()
        h_new, info = inner(t, h, z, states, in_zone, rows)
        I_raw = np.full(len(rows), np.nan)
        if in_zone.any():
            sel = torch.as_tensor(np.flatnonzero(in_zone), device=h.device)
            I_raw[in_zone] = step_controllability(model.rssm, h[sel], z[sel], st.tw.templates_t, int(st.e6["draws"]), float(st.e6["margin"]),
                                                  generator=raw_gen)["I"]
        return h_new, {**info, "I_raw": I_raw}

    def q_fn(h, z, s, g):
        zstate = zgen.get_state()
        z_raw = torch.Generator(device=st.device)
        z_raw.set_state(zstate)
        ch_c = channel_q(model, h, z, clone(g, st.device), zgen, mean_zbar=True)
        ch_r = channel_q(model, keep["h_raw"], z, clone(g, st.device), z_raw, mean_zbar=True)  # the same draws
        q = q_values(model.rssm, model.goal_head, model.critic, h, z, model.cfg.k_act, model.cfg, g)
        if not torch.equal(q, ch_c["Q_normal"]):
            raise RuntimeError("the counterfactual's Q differs from the policy's")
        buffer.append({**{f"corr_{k}": ch_c[k].cpu().numpy() for k in KEYS}, **{f"raw_{k}": ch_r[k].cpu().numpy() for k in KEYS}})
        return q
    t0 = time.perf_counter()
    ep = fs.run_model(model, st.tw, st.image, st.e6, st.eps["experimental"], hook=hook, q_fn=q_fn)
    belief = fs.belief_row(ep, ref_plain, ref_fresh, st.image)
    ph = fs.phenotype_row(ep, st.image, st.eps["experimental"], st.e6, with_bootstrap=False)
    row = {"agent": st.agent["id"], "setting": setting["setting"], "lam": setting["lam"], "mu_pos": setting["mu_pos"], "mu_goal": setting["mu_goal"],
           **belief, "directed_share": ph["directed_share"], "seconds": time.perf_counter() - t0}
    arrays = {k: ep[k] for k in SAVE}
    arrays.update({"intervened": ep["intervened"], "delta_norm": ep["delta_norm"], "I_raw": ep["I_raw"], "reference_fresh_I": ref_fresh,
                   "b_local": fs.b_local(ep, ref_fresh), "b_raw": ep["I_raw"] - ref_fresh[np.clip(ep["states"][:, :-1], 0, None)]})
    arrays.update(assemble(buffer, ep["length"], ep["Q"].shape, ""))
    print(f"[D7 corrected] {st.agent['id']} {setting['setting']}: b_step1 {row['b_step1']:+.3f}, directed {row['directed_share']:.3f}, "
          f"loc zone {row['loc_template_zone']:.2f}; {row['seconds']:.0f} s", flush=True)
    return arrays, [row]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("natural", "corrected"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", default=[])
    parser.add_argument("--constrained", nargs="+", default=[])
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--mu-pos", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--n-reps", type=int, default=None)
    parser.add_argument("--image", default=None, help="natural: a test image of e6.images (default e7.diagnostics.image)")
    parser.add_argument("--no-overlay", action="store_true", help="natural: E6's two models only, no plan B (no --directions)")
    args = parser.parse_args(argv)
    if args.phase == "corrected" and (args.image is not None or args.no_overlay):
        parser.error("--image and --no-overlay are for --phase natural")
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    if args.image is not None:
        cfg["e7"] = {**cfg["e7"], "diagnostics": {**cfg["e7"]["diagnostics"], "image": args.image}}
    if args.n_reps is not None:
        cfg["e6"] = {**cfg["e6"], "n_reps": args.n_reps}
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        setting = None
        if args.phase == "corrected":
            con = json.loads((ra.find_constrained(args.constrained, agent["id"]) / "settings.json").read_text(encoding="utf-8"))
            setting = next(r for r in con["settings"] if r["image"] == "training" and r["lam"] == args.lam and r["mu_pos"] == args.mu_pos)
        tag = ("natural" if args.image is None else f"natural_{args.image}") if setting is None else setting["setting"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e7d7__{agent['id']}__{args.phase}__{tag}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            st = rdiag.Setting(cfg, agent, device)
            before = rdiag.weights_hash(st.models)
            if args.phase == "natural":
                arrays, rows = natural(st, None if args.no_overlay else ra.find_directions(args.directions, agent["id"]))
            else:
                arrays, rows = corrected(st, setting)
            after = rdiag.weights_hash(st.models)
            if after != before:
                raise RuntimeError(f"weights moved: {before} -> {after}")
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / "episodes.npz", **arrays)
        (out / "summary.json").write_text(json.dumps({"rows": rows, "setting": setting}, indent=1, default=float), encoding="utf-8")
        meta = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "phase": args.phase, "tag": tag,
                "image": st.image["name"], "slot": st.image["slot"], "weights": {r: str(p_) for r, p_ in agent["weights"].items()},
                "weights_hash_before": before, "weights_hash_after": after, "agents_file": str(args.agents), "config": str(args.config),
                "config_bases": cfg["_bases"], "device": str(device), "torch_threads": torch.get_num_threads(),
                "torch_version": str(torch.__version__), "seconds": time.perf_counter() - t0, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[D7 {args.phase}] {agent['id']} {tag}: done in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
