"""E7's diagnostics of the value and of the decision rule (docs/e6/DECISIONS.md (51)-(54)): does the exposed agent's
residual deficit under the constrained correction come from a value cached by its critic (H1), and does the references'
insensitivity to a partial induction come from the greedy decision rule (H2)? Frozen weights, training image, the test
environment (everything controllable); every model's state dict is hashed before and after (the run fails if it moved).

Phases (--phase), one output directory per agent and phase, <out-dir>/E7_latent_manipulation/e7diag__<agent>__<phase>__<UTC>/:
  values      diagnostics 1 and 4 (static): V of every cell at E6's step-1 context (both models); at the zone's cells
              (e7.diagnostics.contexts_per_cell contexts each), Q_k for k = 1..k_max in the variants oracle and model (both
              models) and model_corrected (the exposed model, from h_1 + d: the constrained correction of its plan-A
              setting, run_e7_constrained.py), with b and the decoded cell of n_trace particles along the rollout.
              values.npz, summary.json.
  induction   diagnostic 2: E6's training-image episodes (exposed, reference) and plan B's induction (reverse and
              reverse_random at e7.induction.alphas), every per-step array saved (Q, b_local, localisation).
              episodes.npz, rows.json.
  softmax     diagnostic 3: the same episodes with the action drawn from softmax(Q / tau), tau in --taus.
  lookahead   diagnostic 4's episodes: E6's episodes with the policy on the oracle Q_k, k in e7.diagnostics.lookahead_ks.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_diagnostics.py --phase values --agents configs/e6/agents_dev_e4.yaml --runs-root <root>
      --directions <e7dir dirs or globs> --constrained <e7con dirs or globs> [--agent-ids ...] [--taus ...] [--out-dir results]
"""

from __future__ import annotations

import argparse
import hashlib
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
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.measures import decision_rule as dr
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.belief_steering import constraint_terms, optimise_delta, target_posteriors
from helplessness.probing.fixed_trajectory import step_controllability
from helplessness.probing.frozen_episodes import fresh_start_contexts, load_frozen_agent, localise
from helplessness.probing.lookahead import cell_values, embeddings, k_step_q
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_latent_manipulation"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e7" / "e7.yaml"
PHASES = ("values", "induction", "softmax", "lookahead")
PURPOSE = {"values": 0, "contexts": 1, "oracle": 2, "model": 3, "model_corrected": 4, "trace": 5, "lookahead": 6}


def weights_hash(models: dict) -> dict:
    """SHA-256 of every model's parameters and buffers (rssm, goal head, critic), in state-dict order."""
    out = {}
    for role, m in models.items():
        h = hashlib.sha256()
        for module in (m.rssm, m.goal_head, m.critic):
            for k, v in module.state_dict().items():
                h.update(k.encode()), h.update(v.detach().cpu().contiguous().numpy().tobytes())
        out[role] = h.hexdigest()
    return out


def torch_seed(seed: int, purpose: str, extra: int = 0) -> int:
    return int(np.random.SeedSequence([int(seed), PURPOSE[purpose], int(extra)]).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


class Setting:
    """One agent on the diagnostics' image: config blocks, models, the test world, the zone and the directed actions."""

    def __init__(self, cfg: dict, agent: dict, device):
        self.cfg, self.agent, self.device = cfg, agent, device
        self.e6, self.e7 = fs.e6_protocol(cfg), dict(cfg["e7"])
        self.d = dict(self.e7["diagnostics"])
        self.conf = {**setup.configuration(cfg, agent["configuration"]), "seed": agent["seed"]}
        self.image = next(im for im in fs.test_images(self.conf, self.e6["images"]) if im["name"] == self.d["image"])
        _, self.tw = fs.build_test_world(cfg, self.conf, self.image["slot_colors"], device)
        self.models = {role: load_frozen_agent(agent["weights"][role], device) for role in ("experimental", "reference")}
        self.eps = {role: (float(self.e6["epsilon"]) if self.e6.get("epsilon") is not None else m.cfg.epsilon) for role, m in self.models.items()}
        self.zone = fs.zone_states(self.image["slot"])
        self.dist = fs.goal_distances()
        self.directed = dr.directed_action_mask(self.tw.succ, self.dist)
        self.conveyor = setup.successors(setup.world(cfg, self.conf, "experimental"))[:, 4]  # the conveyor ignores the action


# ------------------------------------------------------------------------------------------------------------- values
def correction(st: Setting, model, h, z, states, setting: dict) -> torch.Tensor:
    """The constrained correction d of plan A's setting at the contexts (h, z) (run_e7_apply.stepwise_hook's call)."""
    e7 = st.e7
    target = target_posteriors(model.rssm, h, z, st.tw.succ[states], st.tw.templates_t)
    extra = constraint_terms(model.rssm, model.goal_head, h, z, st.tw.agent, st.tw.kpx, setting["mu_pos"], setting["mu_goal"],
                             float(e7["constrained"]["tau"]))
    return optimise_delta(model.rssm, h, z, target, float(setting["lam"]), float(e7["lr"]), int(e7["stepwise_max_steps"]), float(e7["tol"]),
                          int(e7["patience"]), extra=extra)["delta"]


def trace_measures(st: Setting, model, trace: dict, ref_fresh: np.ndarray, gen) -> dict:
    """b and the decoded cell of the traced particles at every step of the rollout: (k_max, n, A, n_trace) each."""
    th, tz = trace["trace_h"], trace["trace_z"]
    shape = th.shape[:-1]
    flat_h, flat_z = th.reshape(-1, th.shape[-1]).to(st.device), tz.reshape(-1, *tz.shape[-2:]).to(st.device)
    I = step_controllability(model.rssm, flat_h, flat_z, st.tw.templates_t, int(st.e6["draws"]), float(st.e6["margin"]), generator=gen)["I"]
    dec = localise(model.rssm, st.tw, flat_h, flat_z)["nearest_template"]
    ref = ref_fresh[np.asarray(dec)]
    return {"I": I.reshape(shape), "decoded": np.asarray(dec).reshape(shape), "b": (I - ref).reshape(shape)}


def phase_values(st: Setting, constrained: dict | None, log) -> tuple[dict, dict]:
    d, e6 = st.d, st.e6
    arrays, summary = {}, {"zone": st.zone.tolist(), "interior": fs.interior_states(st.image["slot"]).tolist(), "variants": {}}
    states_all = np.array([s for s in range(two_zones.N_STATES) if s != two_zones.GOAL_STATE])
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, e6)
    arrays["reference_fresh_I"] = ref_fresh
    cells = np.repeat(st.zone, int(d["contexts_per_cell"]))
    arrays["cells"] = cells
    for role, model in st.models.items():
        t0 = time.perf_counter()
        V = np.full(two_zones.N_STATES, np.nan)
        V[states_all] = cell_values(model, st.tw, states_all, int(e6["fresh_samples"]),
                                    torch.Generator(device=st.device).manual_seed(torch_seed(d["seed"], "values"))).mean(axis=1)
        arrays[f"{role}__V"] = V
        h, z = fresh_start_contexts(model.rssm, st.tw, cells, torch.Generator(device=st.device).manual_seed(torch_seed(d["seed"], "contexts")))
        variants = ["oracle", "model"] + (["model_corrected"] if role == "experimental" and constrained is not None else [])
        emb = embeddings(model, st.tw)
        for variant in variants:
            t1 = time.perf_counter()
            h_root = h
            if variant == "model_corrected":
                h_root = h + correction(st, model, h, z, cells, constrained)
                arrays[f"{role}__{variant}__delta_norm"] = (h_root - h).norm(dim=1).cpu().numpy()
            gen = torch.Generator(device=st.device).manual_seed(torch_seed(d["seed"], variant))
            Qs, traces = [], []
            for first in range(0, len(cells), int(d["chunk"])):
                sl = slice(first, min(first + int(d["chunk"]), len(cells)))
                out = k_step_q(model, st.tw, h_root[sl], z[sl], cells[sl], int(d["k_max"]), "oracle" if variant == "oracle" else "model",
                               int(d["n_particles"]), int(d["k_inner"]), gen, embeds=emb, n_trace=int(d["n_trace"]) if variant != "oracle" else 0)
                Qs.append(out["Q"].cpu().numpy())
                if variant != "oracle":
                    traces.append(out)
            Q = np.concatenate(Qs)
            arrays[f"{role}__{variant}__Q"] = Q
            gen_m = torch.Generator(device=st.device).manual_seed(torch_seed(d["seed"], "trace"))
            root = step_controllability(model.rssm, h_root, z, st.tw.templates_t, int(e6["draws"]), float(e6["margin"]), generator=gen_m)["I"]
            arrays[f"{role}__{variant}__b_root"] = root - ref_fresh[cells]
            arrays[f"{role}__{variant}__loc_root"] = localise(model.rssm, st.tw, h_root, z)["nearest_template"] == cells
            if traces:
                tr = {k: torch.cat([t[k] for t in traces], dim=1) for k in ("trace_h", "trace_z", "trace_actions")}
                m = trace_measures(st, model, tr, ref_fresh, gen_m)
                for k, v in m.items():
                    arrays[f"{role}__{variant}__trace_{k}"] = v
                arrays[f"{role}__{variant}__trace_actions"] = tr["trace_actions"].numpy()
            dirs = st.directed[cells]
            tol = float(model.cfg.tie_tol_abs)
            rows = []
            for k in range(1, int(d["k_max"]) + 1):
                Qk = Q[..., k - 1]
                rows.append({"k": k, "greedy_directed": float(dr.greedy_directed_share(Qk, dirs, tol).mean()),
                             "argmax_directed": float(dr.argmax_directed(Qk, dirs).mean()),
                             "greedy_directed_interior": float(dr.greedy_directed_share(Qk, dirs, tol)[np.isin(cells, summary["interior"])].mean()),
                             "gap_median": float(np.median(dr.best_second_gap(Qk)))})
            summary["variants"][f"{role}__{variant}"] = rows
            log(f"[diag values] {st.agent['id']} {role} {variant}: greedy directed k=1 {rows[0]['greedy_directed']:.2f}, "
                f"k=5 {rows[4]['greedy_directed']:.2f}, k=10 {rows[9]['greedy_directed']:.2f}; b_root {np.nanmean(arrays[f'{role}__{variant}__b_root']):+.2f}; "
                f"{time.perf_counter() - t1:.0f} s")
        summary[f"{role}__seconds"] = time.perf_counter() - t0
    summary["constrained_setting"] = None if constrained is None else {k: constrained[k] for k in ("setting", "lam", "mu_pos", "mu_goal")}
    return arrays, summary


# ----------------------------------------------------------------------------------------------------------- episodes
def episode_row(st: Setting, ep: dict, ref_plain: dict, ref_fresh: np.ndarray, role: str, **labels) -> dict:
    belief = fs.belief_row(ep, ref_plain, ref_fresh, st.image)
    ph = fs.phenotype_row(ep, st.image, st.eps[role], st.e6, with_bootstrap=False)
    return {"agent": st.agent["id"], "model": "exposed" if role == "experimental" else "reference", **labels, **belief,
            "directed_share": ph["directed_share"], "success_rate": ph["success_rate"], "n_episodes": ph["n_episodes"],
            "p_goal_zone": ra.zone_mean(ep, "p_goal", st.image) if np.isfinite(ep["p_goal"]).any() else float("nan")}


def induction_items(st: Setting, directions: Path) -> list[dict]:
    """E6's two models without intervention, then plan B: the reference with the reverse direction or its random control."""
    dirs = dict(np.load(directions / "directions.npz"))
    items = [{"mode": "none", "alpha": 0.0, "role": "experimental", "vector": None},
             {"mode": "reverse_none", "alpha": 0.0, "role": "reference", "vector": None}]
    for mode, key in (("reverse", "reverse__v_hat"), ("reverse_random", "reverse__random")):
        items += [{"mode": mode, "alpha": float(a), "role": "reference", "vector": dirs[key]} for a in st.e7["induction"]["alphas"] if float(a) != 0.0]
    return items


def phase_episodes(st: Setting, phase: str, directions: Path | None, taus: list[float] | None, log) -> tuple[dict, dict]:
    e6 = dict(st.e6)
    ref_fresh = fs.reference_fresh(st.models["reference"], st.tw, st.image, e6)
    ref_plain = fs.run_model(st.models["reference"], st.tw, st.image, e6, st.eps["reference"])  # b at step 1's subtrahend
    arrays, rows = {"reference_fresh_I": ref_fresh}, []
    if phase == "induction":
        plan = [{**it, "tau": None, "k": None} for it in induction_items(st, directions)]
    elif phase == "softmax":
        e6["measure"] = "zone"
        base = [it for it in induction_items(st, directions) if it["mode"] != "reverse_random"]
        plan = [{**it, "tau": float(t), "k": None} for t in taus for it in base]
    else:
        e6["measure"] = "first"
        plan = [{"mode": "none", "alpha": 0.0, "role": role, "vector": None, "tau": None, "k": int(k)}
                for k in st.d["lookahead_ks"] for role in ("experimental", "reference")]
    for it in plan:
        t0 = time.perf_counter()
        model = st.models[it["role"]]
        hook = None if it["vector"] is None else ra.fixed_hook(it["vector"], it["alpha"], st.e7["apply_in"])
        q_fn = None
        if it["k"] is not None:
            emb = embeddings(model, st.tw)
            gen = torch.Generator(device=st.device).manual_seed(torch_seed(st.d["seed"], "lookahead", it["k"]))

            def q_fn(h, z, s, _act_gen, model=model, emb=emb, gen=gen, k=it["k"]):
                return k_step_q(model, st.tw, h, z, s, k, "oracle", int(st.d["n_particles"]), int(st.d["k_inner"]), gen, embeds=emb)["Q"][..., k - 1]
        ep = fs.run_model(model, st.tw, st.image, e6, st.eps[it["role"]], hook=hook, tau=it["tau"], q_fn=q_fn)
        rows.append({**episode_row(st, ep, ref_plain, ref_fresh, it["role"], mode=it["mode"], alpha=it["alpha"], tau=it["tau"], k=it["k"]),
                     "seconds": time.perf_counter() - t0})
        tag = f"{it['mode']}__{it['alpha']:g}" + ("" if it["tau"] is None else f"__tau{it['tau']:g}") + ("" if it["k"] is None else f"__k{it['k']}")
        for key, value in ep.items():
            arrays[f"{tag}__{key}"] = value
        arrays[f"{tag}__b_local"] = fs.b_local(ep, ref_fresh)
        r = rows[-1]
        log(f"[diag {phase}] {st.agent['id']} {tag}: b_step1 {r['b_step1']:+.3f}, directed {r['directed_share']:.3f}, "
            f"loc {r['loc_template_step1']:.2f}/{r['loc_template_zone']:.2f}; {r['seconds']:.0f} s")
    return arrays, {"rows": rows}


# --------------------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", default=[], help="run_e7_directions.py outputs (induction, softmax)")
    parser.add_argument("--constrained", nargs="+", default=[], help="run_e7_constrained.py outputs (values: model_corrected)")
    parser.add_argument("--taus", nargs="+", type=float, default=None, help="softmax: the temperatures (default e7.diagnostics.taus)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--n-reps", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    if args.n_reps is not None:
        cfg["e6"] = {**cfg["e6"], "n_reps": args.n_reps}
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    taus = args.taus or cfg["e7"]["diagnostics"].get("taus")
    if args.phase == "softmax" and not taus:
        parser.error("softmax needs --taus (or e7.diagnostics.taus)")
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e7diag__{agent['id']}__{args.phase}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            st = Setting(cfg, agent, device)
            before = weights_hash(st.models)
            directions = ra.find_directions(args.directions, agent["id"]) if args.phase in ("induction", "softmax") else None
            constrained = None
            if args.phase == "values":
                con_dir = ra.find_constrained(args.constrained, agent["id"])
                if con_dir is not None:
                    kept = [r for r in json.loads((con_dir / "settings.json").read_text(encoding="utf-8"))["settings"] if r["kept"]]
                    constrained = max(kept, key=lambda r: r["specific_gain"]) if kept else None  # (48): the best kept setting
                arrays, summary = phase_values(st, constrained, print)
            else:
                arrays, summary = phase_episodes(st, args.phase, directions, taus, print)
            after = weights_hash(st.models)
            if after != before:
                raise RuntimeError(f"weights moved: {before} -> {after}")
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / ("values.npz" if args.phase == "values" else "episodes.npz"), **arrays)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
        meta = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "phase": args.phase,
                "image": st.image["name"], "slot": st.image["slot"], "taus": taus if args.phase == "softmax" else None,
                "directions": None if directions is None else str(directions),
                "constrained": None if args.phase != "values" or constrained is None else {k: constrained[k] for k in ("setting", "lam", "mu_pos", "mu_goal")},
                "weights": {r: str(p) for r, p in agent["weights"].items()}, "weights_hash_before": before, "weights_hash_after": after,
                "agents_file": str(args.agents), "config": str(args.config), "config_bases": cfg["_bases"], "diagnostics": st.d,
                "device": str(device), "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                "seconds": time.perf_counter() - t0, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[diag {args.phase}] {agent['id']}: done in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
