"""E8's frozen measures of the therapy checkpoints (docs/e8/DECISIONS.md (10), (17), (18)); `run_e8.py --phase probes`.
One job = the pairs that share a reference (configs/e8/agents.yaml) at one therapy checkpoint: every model is read from
its therapy run (checkpoints/agent_env<N>.pt), weights frozen (checked unchanged at the end). Nothing here measures by
itself: E5's walks (pipeline.assignment_records.walks_by_assignment), E4's probe (frozen_setup.probe_validity_pair),
E6's episodes (frozen_setup.run_model, belief_row, phenotype_row), D7's readout (probing.critic_readout.readout_q_fn,
measures.decision_rule.critic_readouts), E7's oracle (probing.critic_readout.oracle_q_fn) and the decoder / true-cell
split (measures.zone_phenotypes.localisation_split).

Per pair, image and role (exposed, reference), one row:
  walks_*      b at step 1 by E5's 12-assignment protocol on the image's zone (its UD slot): I at the first step in the
               zone (the curve's first point, E5's records) and its step-1 summary; valid = E4's criterion read from the
               REFERENCE's step-1 mass and I (the pair's seeds); b_walks = I_exposed - I_reference (exposed row);
  probe_*      the training image only (e8.probes.e4_probe): E4's 150-walk probe and its validity, as E6;
  E6           b at step 1 of the episodes, the localisation, the phenotypes (directed share, recovery latency, zone share);
  C1/C2/C3     D7's readout, means over the zone steps measured and well localised (`_loc`) and over all of them (`_all`);
  loc_*        the decoder / true-cell split of the zone steps, and the quadrant the agent is decoded in when
               mislocalised;
  oracle_*     the exposed row at the oracle checkpoints, training image: the oracle at k = 1's directed share and b.

Usage (from the repo root; normally through run_e8.py --phase probes or the local chain):
  python experiments/E8_therapy_relapse/run_e8.py --phase probes --pairs shuffle_A shuffle_E shuffle_F \\
      --run shuffle_A_exp=<run dir> --run shuffle_E_exp=<run dir> --run shuffle_F_exp=<run dir> --run shuffle_B_ref=<run dir> \\
      --checkpoint 2000 [--images all|treated|untreated] [--n-reps N] [--n-walks N] [--device cpu] [--out-dir results]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

import e8_setup
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.measures import decision_rule as dr
from helplessness.measures import zone_phenotypes as zp
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.pipeline.assignment_records import walks_by_assignment
from helplessness.pipeline.episode_sources import ASSIGNMENTS, assignment_colors, canonical_assignment, treated_assignments
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.critic_readout import (READOUT_KEYS, assemble, diagnostics_seed, dissociation_seed, oracle_q_fn,
                                                 readout_q_fn)
from helplessness.probing.frozen_episodes import load_frozen_agent
from helplessness.probing.lookahead import embeddings
from helplessness.probing.probe_validity import is_valid
from helplessness.probing.theme_probe_measure import sha256
from helplessness.runs import code_provenance, require_code_provenance

ROLES = {"experimental": "exposed", "reference": "reference"}
SAVE = ("states", "actions", "length", "terminated", "I", "loc_template", "decoded_cell", "decoded_template", "Q")
ASSIGNMENT_E6_INDEX_BASE = 3000  # E7's 12 assignment images (run_e7_context.py's collect); DECISIONS (18)


def images_for(conf: dict, kind: str, therapy: str, which: str) -> list[dict]:
    """The measured images: the training image for a baseline pair; the 12 assignment images for a shuffle pair (the
    canonical one IS the training image, with its E4 index), split treated / untreated by the therapy."""
    training = fs.test_images(conf, [{"name": "training", "axis": "training", "position": "ud"}])[0]
    canonical = canonical_assignment(conf["diagonal"])
    if kind == "baseline":
        return [{**training, "assignment": canonical, "treated": True}]
    treated = set(treated_assignments(therapy))
    out = []
    for k in range(len(ASSIGNMENTS)):
        if which != "all" and (k in treated) != (which == "treated"):
            continue
        colors = assignment_colors(k, conf["diagonal"], conf["t_inc"], conf["t_ctrl"])
        if k == canonical:
            if colors != training["slot_colors"]:
                raise AssertionError("the canonical assignment must be the training image")
            out.append({**training, "name": f"assignment{k:02d}", "assignment": k, "treated": k in treated})
        else:
            out.append({"name": f"assignment{k:02d}", "index": ASSIGNMENT_E6_INDEX_BASE + k, "slot": ASSIGNMENTS[k][0], "role": "ud",
                        "slot_colors": colors, "is_training": True, "image": None, "assignment": k, "treated": k in treated})
    return out


def put(arrays: dict, key: str, value) -> None:
    """Every array of a job under its own key: a second write to a key stops the job (E7's lookahead once saved two
    models' episodes under one key, the second overwriting the first: docs/e8/DECISIONS.md (19))."""
    if key in arrays:
        raise KeyError(f"episodes.npz key written twice: {key}")
    arrays[key] = value


def frozen_copy(model) -> dict:
    return {name: {k: v.detach().clone() for k, v in getattr(model, name).state_dict().items()} for name in ("rssm", "goal_head", "critic")}


def unchanged(model, copy_: dict) -> bool:
    return all(torch.equal(v, copy_[name][k]) for name in copy_ for k, v in getattr(model, name).state_dict().items())


def readout_summary(ep: dict, arrays: dict, tw, image: dict, tol: float) -> dict:
    """C1, C2, C3 (D7, main z_bar) over the zone steps measured, well localised (_loc) and all (_all)."""
    in_zone = np.zeros(two_zones.N_STATES, dtype=bool)
    in_zone[fs.zone_states(image["slot"])] = True
    st, ln = ep["states"], ep["length"]
    keep = zp.in_zone_steps(st, ln, in_zone) & np.isfinite(ep["I"])
    i, t = np.nonzero(keep)
    s = st[i, t]
    dist = fs.goal_distances()
    directed = dr.directed_action_mask(tw.succ, dist)
    c = dr.critic_readouts(arrays["V_zbar"][i, t], arrays["Q_hbar"][i, t], directed[s], dist[tw.succ[s]], tol)
    loc = ep["loc_template"][i, t] == 1
    out = {"n_zone_steps_measured": int(keep.sum()), "n_zone_steps_localised": int(loc.sum())}
    for name, v in c.items():
        out[f"{name}_loc"] = float(np.nanmean(v[loc])) if loc.any() else float("nan")
        out[f"{name}_all"] = float(np.nanmean(v)) if v.size else float("nan")
    return out


def localisation_summary(ep: dict, tw, image: dict) -> dict:
    """The decoder / true-cell split of the zone steps (E7's F), and where the mislocalised steps are decoded."""
    in_zone = np.zeros(two_zones.N_STATES, dtype=bool)
    in_zone[fs.zone_states(image["slot"])] = True
    dist = fs.goal_distances()
    directed = dr.directed_action_mask(tw.succ, dist)
    split = zp.localisation_split(ep["states"], ep["actions"], ep["decoded_cell"], in_zone, dist, directed, two_zones.GRID_SHAPE[1])
    st, act, dec = ep["states"], ep["actions"], ep["decoded_cell"]
    s, s1 = st[:, :-1], st[:, 1:]
    steps = (s >= 0) & (s1 >= 0) & (act >= 0) & in_zone[np.clip(s, 0, None)] & np.isfinite(dec)
    dv, sv = dec[steps].astype(int), s[steps]
    mis = dv != sv
    layout = two_zones.build_layout(1, "reference")
    region = np.array([two_zones.region_of_state(layout, x) for x in range(two_zones.N_STATES)])
    where = region[dv[mis]]
    shares = {f"mis_decoded_in_{r}": float(np.mean(where == r)) if mis.any() else float("nan") for r in ("corridor", *QUADRANT_ORDER)}
    return {f"loc_{k}": (float(v) if not isinstance(v, int) else v) for k, v in {**split, **shares}.items()}


def measure_job(cfg: dict, pc: dict, e6: dict, pairs: list[str], models_spec: dict, pairs_spec: dict, run_dirs: dict, checkpoint: int,
                which: str, oracle: bool, device, log=print) -> tuple[list[dict], dict, dict]:
    """The pairs sharing one reference, at one checkpoint. Returns rows, arrays and the job's meta fields."""
    ref_ids = {pairs_spec[p]["reference"] for p in pairs}
    if len(ref_ids) != 1:
        raise ValueError(f"the pairs of a job must share their reference, got {sorted(ref_ids)}")
    ref_id = ref_ids.pop()
    files = {m: Path(run_dirs[m]) / "checkpoints" / f"agent_env{checkpoint:06d}.pt" for m in [ref_id, *[pairs_spec[p]["exposed"] for p in pairs]]}
    therapies = {m: yaml.safe_load((Path(run_dirs[m]) / "config.yaml").read_text(encoding="utf-8"))["e8"]["run"]["therapy"] for m in files}
    if len(set(therapies.values())) != 1:
        raise ValueError(f"the job's runs had different therapies: {therapies}")
    therapy = therapies[ref_id]
    models = {m: load_frozen_agent(f, device) for m, f in files.items()}
    grad_step = {m: int(torch.load(f, map_location="cpu", weights_only=False)["step"]) for m, f in files.items()}
    frozen = {m: frozen_copy(models[m]) for m in models}
    ref_spec = models_spec[ref_id]
    base_conf = setup.configuration(pc, ref_spec["configuration"])
    images = images_for({**base_conf, "seed": int(ref_spec["seed"])}, ref_spec["kind"], therapy, which)
    canonical = canonical_assignment(base_conf["diagonal"])
    nw = int(pc["e8_probes"]["n_walks"])
    d7_seed = int(pc["e7"]["diagnostics"]["seed"])
    rows, arrays = [], {}
    for im in images:
        t0 = time.perf_counter()
        ref_conf = {**base_conf, "seed": int(ref_spec["seed"])}  # the seed only sets the (deterministic) dynamics' stream
        _, tw = fs.build_test_world(pc, ref_conf, im["slot_colors"], device)
        pw = setup.probe_world(pc, ref_conf, im["slot_colors"])
        k = im["assignment"]

        def episodes(model, mode: str) -> tuple[dict, dict]:
            eps = float(e6["epsilon"]) if e6.get("epsilon") is not None else model.cfg.epsilon
            zgen = torch.Generator(device=device).manual_seed(dissociation_seed(d7_seed, "rollout", 0.0, 7 + len(mode)))  # D7's natural
            buffer: list[dict] = []
            ep = fs.run_model(model, tw, im, e6, eps, q_fn=readout_q_fn(model, device, zgen, buffer))
            return ep, assemble(buffer, ep["length"], ep["Q"].shape, "")

        ref_model = models[ref_id]
        ep_ref, ro_ref = episodes(ref_model, "reverse_none")
        ref_fresh = fs.reference_fresh(ref_model, tw, im, e6)
        t_ref = time.perf_counter()
        for p in pairs:
            exp_id = pairs_spec[p]["exposed"]
            conf = {**base_conf, "seed": int(models_spec[exp_id]["seed"])}  # the pair's measurement seeds (E4's rule)
            walks = {role: walks_by_assignment(models[mid].rssm, pc, conf, {k: pw}, canonical, nw, pc["e8_probes"]["purpose"],
                                               slots_of={k: (im["slot"],)}, step1=True)[k]["slots"][im["slot"]]
                     for role, mid in (("experimental", exp_id), ("reference", ref_id))}
            valid = is_valid(walks["reference"]["step1"]["mass_step1"], walks["reference"]["step1"]["I_step1"])
            probe = {}
            if pc["e8_probes"].get("e4_probe") and im["image"] is not None:
                probe = fs.probe_validity_pair(pc, conf, {"experimental": models[exp_id].rssm, "reference": ref_model.rssm}, im,
                                               int(e6["validity"]["n_walks"]), device)
            ep_exp, ro_exp = episodes(models[exp_id], "none")
            # the reference's episodes serve every partner (E6's seeds do not read the agent); its row is repeated per pair
            # with that pair's walks and validity
            for role, mid, ep, ro in (("experimental", exp_id, ep_exp, ro_exp), ("reference", ref_id, ep_ref, ro_ref)):
                belief = fs.belief_row(ep, ep_ref, ref_fresh, im)
                if role == "reference":
                    belief["b_step1"] = 0.0  # I_ref - I_ref on the same episodes, as E6
                eps = float(e6["epsilon"]) if e6.get("epsilon") is not None else models[mid].cfg.epsilon
                ph = fs.phenotype_row(ep, im, eps, e6, with_bootstrap=False)
                w = walks[role]
                row = {"pair": p, "therapy": therapy, "checkpoint": int(checkpoint), "grad_step": grad_step[mid], "model": mid,
                       "role": ROLES[role], "seed": int(models_spec[mid]["seed"]), "image": im["name"], "image_index": im["index"],
                       "assignment": int(k), "treated": bool(im["treated"]), "slot": im["slot"],
                       "walks_I_step1": w["I_curve"][0], "walks_mass_step1": w["step1"]["mass_step1"], "walks_I_step1_s1": w["step1"]["I_step1"],
                       "walks_valid": bool(valid), "walks_n": nw,
                       "b_walks": (w["I_curve"][0] - walks["reference"]["I_curve"][0]) if role == "experimental" else 0.0,
                       **{k_: v for k_, v in probe.items()}, **belief,
                       **{k_: ph[k_] for k_ in ("directed_share", "recovery_latency", "recovery_censored", "zone_share", "success_rate", "n_episodes")
                          if k_ in ph},
                       **readout_summary(ep, ro, tw, im, float(models[mid].cfg.tie_tol_abs)), **localisation_summary(ep, tw, im)}
                if oracle and role == "experimental" and k == canonical:
                    ok = int(pc["e8_probes"]["oracle_k"])
                    gen = torch.Generator(device=device).manual_seed(diagnostics_seed(d7_seed, "lookahead", ok))
                    m = models[mid]
                    q_fn = oracle_q_fn(m, tw, ok, int(pc["e7"]["diagnostics"]["n_particles"]), int(pc["e7"]["diagnostics"]["k_inner"]), gen,
                                       embeddings(m, tw))
                    ep_o = fs.run_model(m, tw, im, {**e6, "measure": "first"}, eps, q_fn=q_fn)
                    bo = fs.belief_row(ep_o, ep_ref, ref_fresh, im)
                    row.update({"oracle_k": ok, "oracle_directed_share": fs.phenotype_row(ep_o, im, eps, e6, with_bootstrap=False)["directed_share"],
                                "oracle_b_step1": bo["b_step1"], "oracle_loc_template_step1": bo["loc_template_step1"]})
                    for key in ("states", "actions", "length"):
                        put(arrays, f"{p}__{im['name']}__oracle__{key}", ep_o[key])
                rows.append(row)
                if role == "reference" and p != pairs[0]:
                    continue  # the reference's arrays are the same for every partner: saved once
                tag = f"{p}__{im['name']}__exposed" if role == "experimental" else f"{ref_id}__{im['name']}__reference"
                for key in SAVE:
                    put(arrays, f"{tag}__{key}", ep[key])
                for key in READOUT_KEYS:
                    put(arrays, f"{tag}__{key}", ro[key])
        put(arrays, f"{ref_id}__{im['name']}__reference_fresh_I", ref_fresh)
        log(f"[E8 probes] k={checkpoint} {im['name']} ({im['slot']}, {'treated' if im['treated'] else 'untreated'}): "
            + "; ".join(f"{r['pair']} b_walks {r['b_walks']:+.2f} valid {r['walks_valid']} directed {r['directed_share']:.2f}"
                        for r in rows[-2 * len(pairs)::2]) + f"; reference {time.perf_counter() - t0:.0f} s (its episodes {t_ref - t0:.0f} s)")
    moved = [m for m in models if not unchanged(models[m], frozen[m])]
    if moved:
        raise RuntimeError(f"weights moved during the measures: {moved}")
    meta = {"pairs": pairs, "reference_model": ref_id, "therapy": therapy, "checkpoint": int(checkpoint), "images": [im["name"] for im in images],
            "weights": {m: str(f) for m, f in files.items()}, "weights_sha256": {m: sha256(f) for m, f in files.items()},
            "grad_step": grad_step, "weights_unchanged": True}
    return rows, arrays, meta


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def probe_config(cfg: dict, n_reps: int | None, n_walks: int | None) -> tuple[dict, dict]:
    """The probes' config (e8.probes.config: E4's blocks + e6 + e7) with E8's probe settings under `e8_probes`; E6's
    protocol block (n_reps overridable for quick runs)."""
    pc = fs.load_layered_config(e8_setup.REPO_ROOT / cfg["e8"]["probes"]["config"], ("e6", "e7"))
    pc["e8_probes"] = dict(cfg["e8"]["probes"])
    if n_walks is not None:
        pc["e8_probes"]["n_walks"] = int(n_walks)
    e6 = fs.e6_protocol(pc)
    e6["measure"] = "all"  # DECISIONS (18): E6's own setting (configs/e6/e6.yaml)
    test_walks = pc["e8_probes"].get("test_only_validity_n_walks")
    if test_walks is not None:  # TESTS ONLY (configs/e8/e8_mini.yaml): E4's validity probe on fewer walks; never in a real config
        e6["validity"] = {**e6["validity"], "n_walks": int(test_walks)}
    if n_reps is not None:
        e6["n_reps"] = int(n_reps)
    return pc, e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=e8_setup.DEFAULT_CONFIG)
    parser.add_argument("--pairs", nargs="+", required=True, help="pairs of the agents file sharing one reference")
    parser.add_argument("--run", action="append", default=[], help="model_id=<therapy run directory> (one per model of the job)")
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--images", choices=("all", "treated", "untreated"), default="all")
    parser.add_argument("--no-oracle", action="store_true", help="skip the oracle even at an oracle checkpoint")
    parser.add_argument("--n-reps", type=int, default=None, help="override e6.n_reps (quick runs)")
    parser.add_argument("--n-walks", type=int, default=None, help="override e8.probes.n_walks (quick runs)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=e8_setup.REPO_ROOT / "results")
    args = parser.parse_args(argv)
    cfg = e8_setup.load_config(args.config)
    spec = e8_setup.load_models(cfg["e8"]["agents"])
    run_dirs = dict(r.split("=", 1) for r in args.run)
    pc, e6 = probe_config(cfg, args.n_reps, args.n_walks)
    if args.device is not None:
        pc["runtime"]["device"] = args.device
    device = configure_runtime(pc)
    provenance = code_provenance(e8_setup.REPO_ROOT)
    require_code_provenance(provenance)
    run_cfg = yaml.safe_load((Path(next(iter(run_dirs.values()))) / "config.yaml").read_text(encoding="utf-8"))
    last = max(int(c) for c in run_cfg["checkpoints"]["grid"])
    oracle_at = {int(c) for c in pc["e8_probes"]["oracle_checkpoints"]} | {last}
    ref_id = spec["pairs"][args.pairs[0]]["reference"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / e8_setup.EXPERIMENT_ID / f"probes__{ref_id}__k{args.checkpoint:06d}__{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    status = out / "status.json"
    status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
    t0 = time.perf_counter()
    try:
        rows, arrays, meta = measure_job(cfg, pc, e6, args.pairs, spec["models"], spec["pairs"], run_dirs, args.checkpoint, args.images,
                                         (args.checkpoint in oracle_at) and not args.no_oracle, device)
    except BaseException as err:
        status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                      "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
        raise
    write_rows(out / "rows.csv", rows)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(out / "episodes.npz", **arrays)
    meta.update({"config": str(args.config), "probe_config": cfg["e8"]["probes"]["config"], "e6": e6, "e8_probes": pc["e8_probes"],
                 "runs": run_dirs, "device": str(device), "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                 "seconds": time.perf_counter() - t0, "code": provenance})
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    status.write_text(json.dumps({"status": "done", "seconds": meta["seconds"]}), encoding="utf-8")
    print(f"[E8 probes] {args.pairs} k={args.checkpoint}: {len(rows)} rows in {meta['seconds']:.0f} s -> {out}", flush=True)
    return 0
