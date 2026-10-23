"""E7, phase "apply" (docs/E7_latent_manipulation.md, docs/e6/DECISIONS.md): E6's episodes (the same starts, seeds and
measures: pipeline.frozen_setup.run_model) with an intervention on h, at every step in the zone of interest
(e7.apply_in: zone) or at every step (all), for every alpha of e7.alphas:

  direction  the EXPOSED model, h <- h + alpha * v_hat (v_hat: the forward task's common direction, run_e7_directions.py);
  random     the exposed model, h <- h + alpha * r, r a random direction of v_hat's norm (the control);
  stepwise   the exposed model, at every step d optimised on the current h (target: the test environment's successors),
             h <- h + alpha * d (e7.stepwise_max_steps Adam steps);
  reverse / reverse_random   the REFERENCE model with the reverse task's v_hat (towards the exposed training dynamics)
             or a random direction of its norm, on the images whose zone is the former UD slot.
Plans (e7.apply_plans, --plans): v1 = the modes above (the first report); A = the constrained correction's KEPT settings
(run_e7_constrained.py, --constrained), modes constrained_direction (its v_hat) and constrained_stepwise (the constrained
correction at every step), alphas e7.constrained.alphas; B = the induction in the reference, reverse and reverse_random over
e7.induction.alphas. alpha = 0 is measured once (mode none) and must reproduce E6 exactly (the same function and seeds; checked by
analyse_e7.py against E6's rows). b at step 1 under intervention = I_model(t = 0, intervened) - I_reference(t = 0, no
intervention), the abscissa of E7's points; the phenotypes are E6's; the checks: the localisation, the mass on the true
successor of the played action, p_goal, all measured on the intervened (h, z).

Outputs: <out-dir>/E7_latent_manipulation/e7apply__<agent id>__<UTC>/: rows.json / rows.csv (one row per image x mode x
alpha), episodes.npz (optional, --save-episodes), meta.yaml, status.json.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_apply.py --agents <file> --runs-root <root> --directions <e7dir run dirs or globs>
      [--agent-ids ...] [--images ...] [--modes direction random stepwise reverse reverse_random] [--alphas ...] [--device ...]
"""

from __future__ import annotations

import argparse
import csv
import glob
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
from helplessness.measures import zone_phenotypes as zp
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.belief_steering import constraint_terms, optimise_delta, target_posteriors
from helplessness.probing.frozen_episodes import load_frozen_agent
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_latent_manipulation"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e7" / "e7.yaml"
MODES = ("direction", "random", "stepwise", "reverse", "reverse_random")
MODEL_OF = {"none": "experimental", "direction": "experimental", "random": "experimental", "stepwise": "experimental",
            "reverse": "reference", "reverse_random": "reference"}


def find_directions(patterns: list[str], agent_id: str) -> Path:
    found = sorted(Path(d) for p in patterns for d in glob.glob(p) if Path(d).name.startswith(f"e7dir__{agent_id}__")
                   and (Path(d) / "directions.npz").is_file())
    if not found:
        raise FileNotFoundError(f"no e7dir run of agent {agent_id} among {patterns}")
    return found[-1]


def fixed_hook(vector: np.ndarray, alpha: float, apply_in: str):
    """h <- h + alpha * vector on the rows in the zone (apply_in zone) or every row (all)."""
    def hook(t, h, z, states, in_zone, rows):
        v = torch.as_tensor(vector, dtype=h.dtype, device=h.device)
        on = torch.as_tensor(in_zone if apply_in == "zone" else np.ones(len(rows), bool), device=h.device)
        h_new = h + alpha * v * on[:, None].to(h.dtype)
        return h_new, {"intervened": on.cpu().numpy().astype(float), "delta_norm": (alpha * float(np.linalg.norm(vector))) * on.cpu().numpy()}
    return hook


def stepwise_hook(model, tw, e7: dict, alpha: float, apply_in: str, lam: float | None = None, mu_pos: float = 0.0, mu_goal: float = 0.0,
                  tau: float = 0.02):
    """At every step, d optimised on the current h towards the test environment's successors, h <- h + alpha * d. lam:
    e7.lam unless given; mu_pos, mu_goal: the constrained correction's terms (plan A; 0: the unconstrained one)."""
    lam = float(e7["lam"]) if lam is None else float(lam)

    def hook(t, h, z, states, in_zone, rows):
        on = in_zone if apply_in == "zone" else np.ones(len(rows), bool)
        info = {"intervened": on.astype(float), "delta_norm": np.zeros(len(rows))}
        if not on.any():
            return h, info
        sel = torch.as_tensor(np.flatnonzero(on), device=h.device)
        target = target_posteriors(model.rssm, h[sel], z[sel], tw.succ[states[on]], tw.templates_t)
        extra = constraint_terms(model.rssm, model.goal_head, h[sel], z[sel], tw.agent, tw.kpx, mu_pos, mu_goal, tau)
        d = optimise_delta(model.rssm, h[sel], z[sel], target, lam, float(e7["lr"]), int(e7["stepwise_max_steps"]),
                           float(e7["tol"]), int(e7["patience"]), extra=extra)["delta"]
        h_new = h.clone()
        h_new[sel] = h[sel] + alpha * d
        info["delta_norm"][on] = alpha * d.norm(dim=1).cpu().numpy()
        return h_new, info
    return hook


def find_constrained(patterns: list[str], agent_id: str) -> Path | None:
    found = sorted(Path(d) for p in patterns for d in glob.glob(p) if Path(d).name.startswith(f"e7con__{agent_id}__")
                   and (Path(d) / "settings.json").is_file())
    return found[-1] if found else None


def build_plan(plans, im: dict, vec: dict, modes: list[str], alphas: list[float], e7: dict, constrained: dict | None, log) -> list[dict]:
    """The episodes to run for one image: mode, alpha, role, plan, setting. alpha = 0 is measured once (none; reverse_none
    when the reference is intervened on)."""
    items = [{"mode": "none", "alpha": 0.0, "role": "experimental", "plan": "all", "setting": None}]
    ud = im["role"] == "ud"
    if "v1" in plans:
        for mode in modes:
            if mode.startswith("reverse") and not ud:
                continue
            if mode != "stepwise" and vec.get(mode) is None:
                log(f"[E7 apply] {im['name']}: no {mode} vector, skipped")
                continue
            items += [{"mode": mode, "alpha": float(a), "role": MODEL_OF[mode], "plan": "v1", "setting": None} for a in alphas if float(a) != 0.0]
    if "A" in plans:
        if constrained is None:
            log(f"[E7 apply] {im['name']}: plan A needs a run_e7_constrained.py output, skipped")
        for r in [r for r in (constrained or {}).get("settings", []) if r["kept"]]:
            for mode in ("constrained_direction", "constrained_stepwise"):
                items += [{"mode": mode, "alpha": float(a), "role": "experimental", "plan": "A", "setting": r}
                          for a in e7["constrained"]["alphas"] if float(a) != 0.0]
    if "B" in plans and ud:
        for mode in ("reverse", "reverse_random"):
            if vec.get(mode) is None:
                log(f"[E7 apply] {im['name']}: no {mode} vector, plan B skipped")
                continue
            items += [{"mode": mode, "alpha": float(a), "role": "reference", "plan": "B", "setting": None}
                      for a in e7["induction"]["alphas"] if float(a) != 0.0]
    if any(it["role"] == "reference" for it in items):
        items.append({"mode": "reverse_none", "alpha": 0.0, "role": "reference", "plan": "all", "setting": None})
    return items


def zone_mean(ep: dict, field: str, image: dict) -> float:
    zone = np.zeros(144, dtype=bool)
    zone[fs.zone_states(image["slot"])] = True
    inz = zp.in_zone_steps(ep["states"], ep["length"], zone)
    v = ep[field][inz]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def run_agent(cfg, e6, e7, agent, directions: Path, device, images_only, modes, alphas, bootstrap, plans=("v1",), constrained_dir=None,
              log=print):
    conf = {**setup.configuration(cfg, agent["configuration"]), "seed": agent["seed"]}
    images = fs.test_images(conf, e6["images"])
    if images_only:
        images = [im for im in images if im["name"] in images_only]
    models = {role: load_frozen_agent(agent["weights"][role], device) for role in ("experimental", "reference")}
    eps = {role: (float(e6["epsilon"]) if e6.get("epsilon") is not None else m.cfg.epsilon) for role, m in models.items()}
    dirs = dict(np.load(directions / "directions.npz"))
    vec = {"direction": dirs.get("forward__v_hat"), "random": dirs.get("forward__random"),
           "reverse": dirs.get("reverse__v_hat"), "reverse_random": dirs.get("reverse__random")}
    constrained = json.loads((constrained_dir / "settings.json").read_text(encoding="utf-8")) if constrained_dir else None
    con_vec = dict(np.load(constrained_dir / "directions.npz")) if constrained_dir else {}
    rows, arrays = [], {}
    for im in images:
        _, tw = fs.build_test_world(cfg, conf, im["slot_colors"], device)
        ref_plain = fs.run_model(models["reference"], tw, im, e6, eps["reference"])
        ref_fresh = fs.reference_fresh(models["reference"], tw, im, e6)
        for item in build_plan(plans, im, vec, modes, alphas, e7, constrained, log):
            mode, alpha, role, setting = item["mode"], item["alpha"], item["role"], item["setting"]
            t0 = time.perf_counter()
            model = models[role]
            if mode in ("none", "reverse_none"):
                hook = None
            elif mode == "stepwise":
                hook = stepwise_hook(model, tw, e7, alpha, e7["apply_in"])
            elif mode == "constrained_stepwise":
                hook = stepwise_hook(model, tw, e7, alpha, e7["apply_in"], setting["lam"], setting["mu_pos"], setting["mu_goal"],
                                     float(e7["constrained"]["tau"]))
            elif mode == "constrained_direction":
                hook = fixed_hook(con_vec[f"{setting['image']}__{setting['setting']}__v_hat"], alpha, e7["apply_in"])
            else:
                hook = fixed_hook(vec[mode], alpha, e7["apply_in"])
            ep = fs.run_model(model, tw, im, e6, eps[role], hook=hook)
            belief = fs.belief_row(ep, ref_plain, ref_fresh, im)
            ph = fs.phenotype_row(ep, im, eps[role], e6, with_bootstrap=bootstrap)
            row = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "seed": agent["seed"],
                   "image": im["name"], "image_index": im["index"], "slot": im["slot"], "position": im["role"],
                   "model": "exposed" if role == "experimental" else "reference", "mode": mode, "alpha": alpha, "apply_in": e7["apply_in"],
                   "plan": item["plan"], "setting": None if setting is None else setting["setting"],
                   "epsilon": eps[role], **belief, **ph, "mass_played_zone": zone_mean(ep, "mass_played", im),
                   "p_goal_zone": zone_mean(ep, "p_goal", im), "q_gap_zone": zone_mean(ep, "q_gap", im),
                   "delta_norm_zone": zone_mean(ep, "delta_norm", im) if "delta_norm" in ep else 0.0,
                   "b_local_by_index": [None if not np.isfinite(v) else float(v) for v in fs.b_local_by_index(ep, ref_fresh, im, int(e6["horizon"]))["mean"]],
                   "seconds": time.perf_counter() - t0}
            rows.append(row)
            tag = f"{im['name']}__{mode}__" + ("" if setting is None else setting["setting"] + "__") + f"{alpha:g}"
            arrays[f"{tag}__states"] = ep["states"]
            arrays[f"{tag}__I"] = ep["I"]
            log(f"[E7 apply] {agent['id']} {im['name']} {mode} {row['setting'] or ''} alpha {alpha:g}: b_step1 {row['b_step1']:+.3f}, directed {row['directed_share']:.3f}, "
                f"loc {row['loc_template_step1']:.2f}/{row['loc_template_zone']:.2f}, |d| {row['delta_norm_zone']:.2f}; {row['seconds']:.0f} s")
    return rows, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--directions", nargs="+", required=True, help="run_e7_directions.py output dirs (or globs); the newest per agent")
    parser.add_argument("--constrained", nargs="+", default=[], help="run_e7_constrained.py output dirs (or globs), plan A; the newest per agent")
    parser.add_argument("--plans", nargs="+", choices=("v1", "A", "B"), default=None, help="default: e7.apply_plans")
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=None, help="default: e7.modes, plus reverse and reverse_random if e7.reverse")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--bootstrap", action="store_true", help="bootstrap intervals of the phenotypes (off: E7's points are read on E6's curves)")
    parser.add_argument("--save-episodes", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--n-reps", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    e6, e7 = fs.e6_protocol(cfg), dict(cfg["e7"])
    if args.n_reps is not None:
        e6["n_reps"] = args.n_reps
    modes = args.modes or [*e7["modes"], *(["reverse", "reverse_random"] if e7.get("reverse", True) else [])]
    alphas = args.alphas or e7["alphas"]
    plans = args.plans or list(e7.get("apply_plans", ["v1"]))
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        directions = find_directions(args.directions, agent["id"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e7apply__{agent['id']}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            rows, arrays = run_agent(cfg, e6, e7, agent, directions, device, args.images, modes, alphas, args.bootstrap, plans,
                                     find_constrained(args.constrained, agent["id"]) if "A" in plans else None)
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "rows.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows])
        if args.save_episodes:
            np.savez_compressed(out / "episodes.npz", **arrays)
        meta = {"agent": agent["id"], "condition": agent["condition"], "directions": str(directions), "plans": plans, "modes": modes, "alphas": alphas,
                "constrained": str(find_constrained(args.constrained, agent["id"])) if "A" in plans else None,
                "weights": {r: str(p) for r, p in agent["weights"].items()}, "agents_file": str(args.agents), "config": str(args.config),
                "config_bases": cfg["_bases"], "e6": e6, "e7": e7, "device": str(device), "torch_threads": torch.get_num_threads(),
                "seconds": time.perf_counter() - t0, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
        print(f"[E7 apply] {agent['id']}: {len(rows)} rows in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
