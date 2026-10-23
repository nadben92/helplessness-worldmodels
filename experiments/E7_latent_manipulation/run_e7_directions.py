"""E7, phase "directions" (docs/E7_latent_manipulation.md, docs/e6/DECISIONS.md): the corrections of h of one or more
agents, and whether they share a direction. Frozen weights; only h is changed.

Per agent and test image (e6.images), the situations: every state of the zone of interest, reached by
e7.contexts_per_state context walks (probing.context_walks: corridor warm-up, random or heading for the goal, entry by
each side in turn, path to the state), filtered by the model from its initial state; split by context index into the
estimation (even) and test (odd) halves; kept only when the model localises the agent (E4's rule on the decoded (h, z)).
Tasks:
  forward  the EXPOSED model, target = the test environment's (controllable) successors;
  reverse  the REFERENCE model, target = the exposed TRAINING dynamics (the conveyor), on the images whose zone of
           interest is the former UD slot only (elsewhere the training dynamics are the corridor's, controllable).
For each situation the correction d (probing.belief_steering.optimise_delta) and I_s at h and at h + d. Then, per task:
the common direction v = the mean of the estimation half's corrections on e7.direction_images, scaled to their mean norm
(v_hat); alignment (leave-one-out cosines, null of isotropic vectors with the same norms); efficacy on the test half
(the shift of I_s with v_hat against each situation's own optimum; a random direction of the same norm as control);
stability (random restarts on a subset). Nothing here decides whether the direction exists: the numbers are reported.

Outputs: <out-dir>/E7_latent_manipulation/e7dir__<agent id>__<UTC>/: directions.npz (v, v_hat, the random direction,
every correction), situations.csv, summary.json, meta.yaml, status.json.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_directions.py --agents <file> --runs-root <root> [--agent-ids ...]
      [--config configs/e7/e7.yaml] [--images ...] [--device cpu|cuda] [--out-dir results] [--contexts-per-state N]
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
from helplessness.envs.layouts import two_zones
from helplessness.interventions import latent_direction as ld
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.belief_steering import optimise_delta, target_posteriors
from helplessness.probing.context_walks import context_walks
from helplessness.probing.fixed_trajectory import fixed_trajectory_context, step_controllability
from helplessness.probing.frozen_episodes import load_frozen_agent, localise
from helplessness.probing.theme_probe_measure import sha256
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_latent_manipulation"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e7" / "e7.yaml"
TASKS = {"forward": "experimental", "reverse": "reference"}


def seq(e7_seed: int, image: dict, task: str, purpose: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(e7_seed), int(image["index"]), fs.SLOT_INDEX[image["slot"]], list(TASKS).index(task), purpose])


def torch_seed(s: np.random.SeedSequence) -> int:
    return int(s.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def training_successors(cfg: dict, conf: dict) -> np.ndarray:
    """The exposed training environment's successors (the conveyor at the UD slot)."""
    return setup.successors(setup.world(cfg, conf, "experimental"))


def situations(cfg, e6, e7, conf, model, image, tw, task, target_succ, device) -> dict:
    """The situations of one (model, image, task) and their corrections."""
    world = setup.probe_world(cfg, conf, image["slot_colors"])
    zone = fs.zone_states(image["slot"])
    corridor = two_zones.corridor_states(world.layout)
    rng = np.random.default_rng(seq(e7["seed"], image, task, 0))
    walks = context_walks(tw.succ, zone, corridor, fs.goal_distances(), zone, int(e7["contexts_per_state"]), rng, two_zones.GRID_SHAPE[1],
                          *map(int, e7["n_pre"]))
    gen = torch.Generator(device=device).manual_seed(torch_seed(seq(e7["seed"], image, task, 1)))
    ctx = fixed_trajectory_context(model.rssm, world.templates_np, walks, generator=gen)
    last = torch.as_tensor(walks["length"] - 1, device=device)
    rows = torch.arange(len(last), device=device)
    h, z = ctx["h"][rows, last], ctx["z"][rows, last]
    states = walks["target"]
    loc = localise(model.rssm, tw, h, z)["nearest_template"] == states
    context_index = np.tile(np.arange(int(e7["contexts_per_state"])), len(zone))
    half = np.where(context_index % 2 == 0, "estimation", "test")
    sel = torch.as_tensor(np.flatnonzero(loc), device=device)
    target = target_posteriors(model.rssm, h[sel], z[sel], target_succ[states[loc]], tw.templates_t)
    opt = optimise_delta(model.rssm, h[sel], z[sel], target, float(e7["lam"]), float(e7["lr"]), int(e7["max_steps"]), float(e7["tol"]),
                         int(e7["patience"]))
    mgen = torch.Generator(device=device).manual_seed(torch_seed(seq(e7["seed"], image, task, 2)))
    I0 = step_controllability(model.rssm, h[sel], z[sel], tw.templates_t, int(e6["draws"]), float(e6["margin"]), generator=mgen)["I"]
    I1 = step_controllability(model.rssm, h[sel] + opt["delta"], z[sel], tw.templates_t, int(e6["draws"]), float(e6["margin"]), generator=mgen)["I"]
    return {"h": h[sel], "z": z[sel], "target": target, "delta": opt["delta"], "states": states[loc], "half": half[loc],
            "side": walks["side"][loc], "kind": walks["kind"][loc], "I0": I0, "I_opt": I1, "kl0": opt["kl0"].cpu().numpy(),
            "kl": opt["kl"].cpu().numpy(), "steps": opt["steps"], "converged": opt["converged"], "n_contexts": int(len(states)),
            "n_localised": int(loc.sum()), "loc_by_state": {int(s): float(loc[states == s].mean()) for s in np.unique(states)},
            "mgen": mgen}


def run_agent(cfg, e6, e7, agent, device, images_only, log=print) -> tuple[dict, dict, list[dict]]:
    conf = {**setup.configuration(cfg, agent["configuration"]), "seed": agent["seed"]}
    images = fs.test_images(conf, e6["images"])
    if images_only:
        images = [im for im in images if im["name"] in images_only]
    models = {role: load_frozen_agent(agent["weights"][role], device) for role in ("experimental", "reference")}
    train_succ = training_successors(cfg, conf)
    per = {}  # (task, image name) -> situations
    for im in images:
        _, tw = fs.build_test_world(cfg, conf, im["slot_colors"], device)
        for task, role in TASKS.items():
            if task == "reverse" and (not e7.get("reverse", True) or im["role"] != "ud"):
                continue
            t0 = time.perf_counter()
            target_succ = tw.succ if task == "forward" else train_succ
            per[(task, im["name"])] = s = situations(cfg, e6, e7, conf, models[role], im, tw, task, target_succ, device)
            s["tw"] = tw
            log(f"[E7 directions] {agent['id']} {task} {im['name']}: {s['n_localised']}/{s['n_contexts']} localised; "
                f"I {np.nanmean(s['I0']):.3f} -> {np.nanmean(s['I_opt']):.3f}; KL {s['kl0'].mean():.2f} -> {s['kl'].mean():.2f} "
                f"({s['steps']} steps, plateau {s['converged']}); {time.perf_counter() - t0:.0f} s")
    arrays, summary, sit_rows = {}, {"tasks": {}}, []
    for task, role in TASKS.items():
        keys = [k for k in per if k[0] == task]
        if not keys:
            continue
        est = [(k, np.flatnonzero(per[k]["half"] == "estimation")) for k in keys if k[1] in e7["direction_images"]]
        deltas_est = np.concatenate([per[k]["delta"].cpu().numpy()[i] for k, i in est]) if est else np.zeros((0, 1))
        if deltas_est.shape[0] < 2:
            summary["tasks"][task] = {"skipped": "fewer than two estimation corrections on the direction images"}
            continue
        md = ld.mean_direction(deltas_est)
        v_hat = md["v_unit"] * md["mean_norm"]
        rng = np.random.default_rng(np.random.SeedSequence([int(e7["seed"]), list(TASKS).index(task), 99]))
        rand = ld.unit(rng.standard_normal(v_hat.shape)) * md["mean_norm"]
        task_sum = {"model": role, "direction_images": list(e7["direction_images"]), "n_estimation": int(deltas_est.shape[0]),
                    "v_norm": md["v_norm"], "mean_delta_norm": md["mean_norm"],
                    "alignment": {k: v for k, v in ld.alignment(deltas_est, rng).items() if k != "cosines"}, "by_image": {}}
        db_v_all, db_opt_all, db_r_all = [], [], []
        for k in keys:
            s = per[k]
            d = s["delta"].cpu().numpy()
            ok_est, ok_test = s["half"] == "estimation", s["half"] == "test"
            img = {"n_contexts": s["n_contexts"], "n_localised": s["n_localised"], "loc_by_state": s["loc_by_state"],
                   "steps": s["steps"], "converged": s["converged"], "I0_mean": float(np.nanmean(s["I0"])), "I_opt_mean": float(np.nanmean(s["I_opt"])),
                   "kl0_mean": float(s["kl0"].mean()), "kl_mean": float(s["kl"].mean())}
            if ok_est.sum() >= 2:
                img["alignment_estimation"] = {kk: v for kk, v in ld.alignment(d[ok_est], rng).items() if kk != "cosines"}
                img["cos_with_v"] = float(np.mean((ld.unit(d[ok_est]) * md["v_unit"]).sum(axis=1)))
            if ok_test.any():
                sel = torch.as_tensor(np.flatnonzero(ok_test), device=s["h"].device)
                h, z = s["h"][sel], s["z"][sel]
                vt = torch.as_tensor(v_hat, dtype=h.dtype, device=h.device)
                rt = torch.as_tensor(rand, dtype=h.dtype, device=h.device)
                I_v = step_controllability(models[role].rssm, h + vt, z, s["tw"].templates_t, int(e6["draws"]), float(e6["margin"]), generator=s["mgen"])["I"]
                I_r = step_controllability(models[role].rssm, h + rt, z, s["tw"].templates_t, int(e6["draws"]), float(e6["margin"]), generator=s["mgen"])["I"]
                db_v, db_opt, db_r = I_v - s["I0"][ok_test], s["I_opt"][ok_test] - s["I0"][ok_test], I_r - s["I0"][ok_test]
                img["efficacy"] = ld.efficacy(db_v, db_opt)
                img["random_db_mean"] = float(np.nanmean(db_r))
                db_v_all.append(db_v)
                db_opt_all.append(db_opt)
                db_r_all.append(db_r)
            task_sum["by_image"][k[1]] = img
            for i in range(len(s["states"])):
                sit_rows.append({"task": task, "image": k[1], "state": int(s["states"][i]), "half": s["half"][i], "side": s["side"][i], "kind": s["kind"][i],
                                 "delta_norm": float(np.linalg.norm(d[i])), "cos_with_v": float((ld.unit(d[i]) * md["v_unit"]).sum()),
                                 "I0": float(s["I0"][i]), "I_opt": float(s["I_opt"][i]), "kl0": float(s["kl0"][i]), "kl": float(s["kl"][i])})
            arrays[f"{task}__{k[1]}__delta"] = d
            arrays[f"{task}__{k[1]}__states"] = s["states"]
            arrays[f"{task}__{k[1]}__half"] = s["half"]
        if db_v_all:
            task_sum["efficacy_test"] = ld.efficacy(np.concatenate(db_v_all), np.concatenate(db_opt_all))
            task_sum["random_db_mean_test"] = float(np.nanmean(np.concatenate(db_r_all)))
        # stability: random restarts on a subset of the direction images' estimation situations
        rs = e7["restarts"]
        pool = [(k, i) for k, idx in est for i in idx]
        pick = np.random.default_rng(np.random.SeedSequence([int(e7["seed"]), list(TASKS).index(task), 98])).permutation(len(pool))[: int(rs["n_situations"])]
        stab = []
        for j in pick:
            k, i = pool[j]
            s = per[k]
            hi, zi, ti = s["h"][i:i + 1].repeat(int(rs["k"]), 1), s["z"][i:i + 1].repeat(int(rs["k"]), 1, 1), s["target"][i:i + 1].repeat(int(rs["k"]), 1, 1, 1)
            g = torch.Generator(device=hi.device).manual_seed(int(rng.integers(2**31)))
            init = torch.randn(hi.shape, generator=g, device=hi.device)
            init = init / init.norm(dim=1, keepdim=True) * md["mean_norm"]
            sol = optimise_delta(models[role].rssm, hi, zi, ti, float(e7["lam"]), float(e7["lr"]), int(e7["max_steps"]), float(e7["tol"]),
                                 int(e7["patience"]), init=init)["delta"].cpu().numpy()
            st = ld.stability(np.vstack([s["delta"][i:i + 1].cpu().numpy(), sol]))
            stab.append(st)
        if stab:
            task_sum["stability"] = {key: float(np.mean([st[key] for st in stab])) for key in ("norm_mean", "norm_sd", "cos_mean", "cos_min")}
            task_sum["stability"]["n_situations"] = len(stab)
            task_sum["stability"]["k_restarts"] = int(rs["k"])
        arrays[f"{task}__v"] = md["v"]
        arrays[f"{task}__v_hat"] = v_hat
        arrays[f"{task}__random"] = rand
        summary["tasks"][task] = task_sum
        log(f"[E7 directions] {agent['id']} {task}: alignment cos {task_sum['alignment']['cos_mean']:.3f} "
            f"(null {task_sum['alignment']['null_cos_mean']:.3f}, p95 {task_sum['alignment']['null_cos_p95']:.3f}); "
            f"efficacy {task_sum.get('efficacy_test', {}).get('ratio_of_means', float('nan')):.3f}; "
            f"stability cos {task_sum.get('stability', {}).get('cos_mean', float('nan')):.3f}")
    return arrays, summary, sit_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--contexts-per-state", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    e6, e7 = fs.e6_protocol(cfg), dict(cfg["e7"])
    if args.contexts_per_state is not None:
        e7["contexts_per_state"] = args.contexts_per_state
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e7dir__{agent['id']}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            arrays, summary, sit_rows = run_agent(cfg, e6, e7, agent, device, args.images)
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / "directions.npz", **arrays)
        with open(out / "situations.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sit_rows[0]))
            w.writeheader()
            w.writerows(sit_rows)
        summary.update(agent=agent["id"], condition=agent["condition"], seconds=time.perf_counter() - t0)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
        meta = {"agent": agent["id"], "condition": agent["condition"], "configuration": agent["configuration"], "seed": agent["seed"],
                "weights": {r: str(p) for r, p in agent["weights"].items()}, "weights_sha256": {r: sha256(p) for r, p in agent["weights"].items()},
                "agents_file": str(args.agents), "config": str(args.config), "config_bases": cfg["_bases"], "e7": e7, "device": str(device),
                "torch_threads": torch.get_num_threads(), "seconds": summary["seconds"], "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": summary["seconds"]}), encoding="utf-8")
        print(f"[E7 directions] {agent['id']} done in {summary['seconds']:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
