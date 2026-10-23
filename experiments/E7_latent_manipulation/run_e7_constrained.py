"""E7, phase "constrained" (manipulation A, docs/e6/DECISIONS.md (47)-(49)): the correction of the EXPOSED models' h with
what must not move held fixed -- L = sum_a KL(target_a || prior_a(d)) + lam |d|^2 + mu_pos D_pos + mu_goal (dp_goal)^2
(probing.belief_steering.constraint_terms) -- swept over lam and mu_pos, mu_goal calibrated once per agent.

Situations: every context of run_e7_directions.contexts (e7.constrained.contexts_per_state per state: the position term
decodes at every Adam step) on e7.constrained.images, the
forward task (target: the test environment's controllable successors); ALL contexts are corrected (localised or not), so
the localisation before intervention is the exposed model's own (below 1), the comparison the design asks for.
Per setting: the gain of b = the mean shift of I_s (the reference's I is unchanged) with its bootstrap interval; the
localisation under intervention (E4's nearest-template rule on the intervened (h, z)); p_goal's shift; the random control
(each situation's own random direction at the norm of its correction): its gain and localisation; the specificity (the gain
minus the control's, paired, 95% bootstrap interval). Kept (e7.constrained): localisation >= the exposed model's without
intervention - loc_tol, and the gain specific (interval above 0 and at least specific_share of the gain). mu_goal: the
smallest of mu_goal_grid with max |dp_goal| <= goal_tol at the most permissive setting (smallest lam, mu_pos = 0). Lines of
the figure: the exposed model's localisation without intervention, and the REFERENCE's on the same walks.

Outputs: <out-dir>/E7_latent_manipulation/e7con__<agent id>__<UTC>/: settings.csv, settings.json (with the kept flags),
directions.npz (per setting: v_hat = the estimation half's mean correction at their mean norm, for run_e7_apply.py's plan
A), tradeoff.png, meta.yaml, status.json.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/run_e7_constrained.py --agents <file> --runs-root <root> [--agent-ids ...] [--device cpu]
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

import run_e7_directions as rd
from helplessness.config import REPO_ROOT
from helplessness.interventions import latent_direction as ld
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.belief_steering import constraint_terms, optimise_delta, p_goal_with_grad, target_posteriors
from helplessness.probing.fixed_trajectory import step_controllability
from helplessness.probing.frozen_episodes import load_frozen_agent, localise
from helplessness.runs import code_provenance, require_code_provenance

EXPERIMENT_ID = "E7_latent_manipulation"


def boot_ci(values: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), float("nan")
    means = v[rng.integers(v.size, size=(n, v.size))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(model, tw, c: dict, delta: torch.Tensor, rand_unit: torch.Tensor, e6: dict, seed: int) -> dict:
    """I_s, localisation and p_goal at h, h + d and h + |d| r (r: each situation's random unit direction); one measurement
    seed for every call (common random numbers across settings)."""
    h, z, states = c["h"], c["z"], c["states"]
    out = {}
    for name, hh in (("before", h), ("after", h + delta), ("random", h + rand_unit * delta.norm(dim=1, keepdim=True))):
        gen = torch.Generator(device=h.device).manual_seed(seed)
        out[f"I_{name}"] = step_controllability(model.rssm, hh, z, tw.templates_t, int(e6["draws"]), float(e6["margin"]), generator=gen)["I"]
        out[f"loc_{name}"] = localise(model.rssm, tw, hh, z)["nearest_template"] == states
        with torch.no_grad():
            out[f"pgoal_{name}"] = p_goal_with_grad(model.goal_head, hh, z).cpu().numpy()
    return out


def reproduction(cfg, e6, e7, conf, exposed, im, tw, directions: Path | None, device) -> dict:
    """The first report's shape test: lam = e7.lam, mu_pos = mu_goal = 0 on run_e7_directions' situations (e7.contexts_per_state
    contexts, the localised ones) gives its corrections again (the optimisation is per situation, so the batch does not
    change it; only the matrix products' blocking can, at float precision)."""
    s = rd.situations(cfg, e6, e7, conf, exposed, im, tw, "forward", tw.succ, device)
    out = {"n": int(len(s["states"])), "I_opt_mean": float(np.nanmean(s["I_opt"])), "delta_norm": float(s["delta"].norm(dim=1).mean())}
    if directions is not None:
        ref = np.load(directions / "directions.npz")[f"forward__{im['name']}__delta"]
        d = s["delta"].cpu().numpy()
        out.update(directions=str(directions), same_shape=ref.shape == d.shape,
                   max_abs_diff=float(np.max(np.abs(ref - d))) if ref.shape == d.shape else float("nan"))
    return out


def run_agent(cfg, e6, e7, agent, device, directions: Path | None = None, log=print) -> tuple[list[dict], dict, dict]:
    con = e7["constrained"]
    e7_sweep = {**e7, "contexts_per_state": int(con.get("contexts_per_state", e7["contexts_per_state"]))}
    conf = {**setup.configuration(cfg, agent["configuration"]), "seed": agent["seed"]}
    images = [im for im in fs.test_images(conf, e6["images"]) if im["name"] in con["images"]]
    exposed = load_frozen_agent(agent["weights"]["experimental"], device)
    reference = load_frozen_agent(agent["weights"]["reference"], device)
    settings, arrays, extra = [], {}, {}
    for im in images:
        _, tw = fs.build_test_world(cfg, conf, im["slot_colors"], device)
        if con.get("reproduce_first_report", False):
            extra.setdefault(im["name"], {})["reproduction"] = rep_ = reproduction(cfg, e6, e7, conf, exposed, im, tw, directions, device)
            log(f"[E7 constrained] {agent['id']} {im['name']}: the first report's setting -> {rep_}")
        c = rd.contexts(cfg, e7_sweep, conf, exposed, im, tw, "forward", device)
        c_ref = rd.contexts(cfg, e7_sweep, conf, reference, im, tw, "forward", device)  # the same walks (same seeds), the reference's h
        target = target_posteriors(exposed.rssm, c["h"], c["z"], tw.succ[c["states"]], tw.templates_t)
        mseed = rd.torch_seed(rd.seq(e7["seed"], im, "forward", 3))
        rng = np.random.default_rng(rd.seq(e7["seed"], im, "forward", 4))
        rand_unit = torch.as_tensor(ld.unit(rng.standard_normal(tuple(c["h"].shape))), dtype=c["h"].dtype, device=device)
        loc_ref = float(c_ref["loc"].mean())

        def solve(lam, mu_pos, mu_goal):
            ex = constraint_terms(exposed.rssm, exposed.goal_head, c["h"], c["z"], tw.agent, tw.kpx, mu_pos, mu_goal, float(con["tau"]))
            return optimise_delta(exposed.rssm, c["h"], c["z"], target, lam, float(e7["lr"]), int(e7["max_steps"]), float(e7["tol"]),
                                  int(e7["patience"]), extra=ex)

        # mu_goal: the smallest keeping p_goal within goal_tol at the most permissive setting
        lam0 = min(map(float, con["lams"]))
        mu_goal, calib = None, []
        for mg in map(float, con["mu_goal_grid"]):
            ev = evaluate(exposed, tw, c, solve(lam0, 0.0, mg)["delta"], rand_unit, e6, mseed)
            moved = float(np.max(np.abs(ev["pgoal_after"] - ev["pgoal_before"])))
            calib.append({"mu_goal": mg, "max_abs_dp_goal": moved})
            log(f"[E7 constrained] {agent['id']} {im['name']}: mu_goal {mg:g} -> max |dp_goal| {moved:.3f}")
            if moved <= float(con["goal_tol"]):
                mu_goal = mg
                break
        calibrated = mu_goal is not None
        mu_goal = mu_goal if calibrated else float(con["mu_goal_grid"][-1])
        extra.setdefault(im["name"], {}).update({"mu_goal": mu_goal, "mu_goal_calibrated": calibrated, "calibration": calib, "loc_reference": loc_ref,
                             "n_situations": int(len(c["states"])), "n_localised_before": int(c["loc"].sum()),
                             "contexts_per_state": e7_sweep["contexts_per_state"]})
        boot = np.random.default_rng(rd.seq(e7["seed"], im, "forward", 5))
        for lam in map(float, con["lams"]):
            for mu_pos in map(float, con["mu_pos"]):
                t0 = time.perf_counter()
                opt = solve(lam, mu_pos, mu_goal)
                ev = evaluate(exposed, tw, c, opt["delta"], rand_unit, e6, mseed)
                gain = ev["I_after"] - ev["I_before"]
                gain_r = ev["I_random"] - ev["I_before"]
                diff = gain - gain_r
                loc_before, loc_after = float(ev["loc_before"].mean()), float(ev["loc_after"].mean())
                ci_diff = boot_ci(diff, boot, int(con["n_boot"]))
                specific = bool(ci_diff[0] > 0 and np.nanmean(diff) >= float(con["specific_share"]) * np.nanmean(gain))
                kept = bool(loc_after >= loc_before - float(con["loc_tol"]) and specific)
                d = opt["delta"].cpu().numpy()
                est = c["half"] == "estimation"
                md = ld.mean_direction(d[est])
                sid = f"lam{lam:g}_mupos{mu_pos:g}"
                arrays[f"{im['name']}__{sid}__v_hat"] = md["v_unit"] * md["mean_norm"]
                arrays[f"{im['name']}__{sid}__delta"] = d
                was = c["loc"]
                row = {"agent": agent["id"], "condition": agent["condition"], "image": im["name"], "setting": sid, "lam": lam, "mu_pos": mu_pos,
                       "mu_goal": mu_goal, "n": int(len(gain)), "delta_norm": float(np.linalg.norm(d, axis=1).mean()),
                       "gain_b": float(np.nanmean(gain)), "gain_b_ci": boot_ci(gain, boot, int(con["n_boot"])),
                       "gain_b_random": float(np.nanmean(gain_r)), "gain_b_random_ci": boot_ci(gain_r, boot, int(con["n_boot"])),
                       "specific_gain": float(np.nanmean(diff)), "specific_gain_ci": ci_diff, "specific": specific,
                       "loc_before": loc_before, "loc_after": loc_after, "loc_random": float(ev["loc_random"].mean()),
                       "loc_after_among_localised_before": float(ev["loc_after"][was].mean()) if was.any() else float("nan"),
                       "loc_reference": loc_ref, "max_abs_dp_goal": float(np.max(np.abs(ev["pgoal_after"] - ev["pgoal_before"]))),
                       "mean_abs_dp_goal": float(np.mean(np.abs(ev["pgoal_after"] - ev["pgoal_before"]))),
                       "kl0_mean": float(opt["kl0"].mean()), "kl_mean": float(opt["kl"].mean()), "steps": opt["steps"],
                       "v_hat_norm": md["mean_norm"], "kept": kept, "seconds": time.perf_counter() - t0}
                settings.append(row)
                log(f"[E7 constrained] {agent['id']} {sid}: gain {row['gain_b']:+.3f} (random {row['gain_b_random']:+.3f}, specific "
                    f"{row['specific_gain']:+.3f} [{ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}]), loc {loc_before:.2f} -> {loc_after:.2f} "
                    f"(random {row['loc_random']:.2f}, reference {loc_ref:.2f}), |dp_goal| <= {row['max_abs_dp_goal']:.3f}, |d| "
                    f"{row['delta_norm']:.2f}; kept {kept}; {row['seconds']:.0f} s")
    return settings, arrays, extra


def plot(settings: list[dict], extra: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.grid(color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    for r in settings:
        ax.scatter(r["gain_b"], r["loc_after"], s=50, color="#2a78d6" if r["kept"] else "#7a7a75", zorder=3)
        ax.scatter(r["gain_b_random"], r["loc_random"], s=40, facecolors="white", edgecolors="#eb6834", linewidths=1.5, zorder=3)
        ax.plot([r["gain_b"], r["gain_b_random"]], [r["loc_after"], r["loc_random"]], color="#c3c2b7", lw=0.6, zorder=2)
        ax.annotate(f"λ{r['lam']:g} μ{r['mu_pos']:g}", (r["gain_b"], r["loc_after"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    any_extra = next(iter(extra.values()))
    ax.axhline(settings[0]["loc_before"], color="#2b2b29", ls="--", lw=1, label="exposed, no intervention")
    ax.axhline(any_extra["loc_reference"], color="#1baf7a", ls=":", lw=1.5, label="reference")
    ax.scatter([], [], color="#2a78d6", label="setting kept")
    ax.scatter([], [], color="#7a7a75", label="setting not kept")
    ax.scatter([], [], facecolors="white", edgecolors="#eb6834", label="random control (same norm)")
    ax.set_xlabel("gain of b (bit): mean shift of I_s")
    ax.set_ylabel("localisation under intervention")
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e7" / "e7.yaml")
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--contexts-per-state", type=int, default=None, help="override e7.constrained.contexts_per_state")
    parser.add_argument("--directions", nargs="+", default=[], help="run_e7_directions.py outputs (or globs): the reproduction check")
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    e6, e7 = fs.e6_protocol(cfg), dict(cfg["e7"])
    if args.contexts_per_state is not None:
        e7["constrained"] = {**e7["constrained"], "contexts_per_state": args.contexts_per_state}
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    provenance = code_provenance(REPO_ROOT)
    require_code_provenance(provenance)
    for agent in fs.load_agents(args.agents, args.runs_root, args.agent_ids):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out_dir / EXPERIMENT_ID / f"e7con__{agent['id']}__{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        status = out / "status.json"
        status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            import glob
            found = sorted(Path(d) for p in args.directions for d in glob.glob(p) if Path(d).name.startswith(f"e7dir__{agent['id']}__"))
            settings, arrays, extra = run_agent(cfg, e6, e7, agent, device, found[-1] if found else None)
        except BaseException as err:
            status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                          "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
            raise
        np.savez_compressed(out / "directions.npz", **arrays)
        with open(out / "settings.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(settings[0]))
            w.writeheader()
            w.writerows([{k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in r.items()} for r in settings])
        seconds = time.perf_counter() - t0
        (out / "settings.json").write_text(json.dumps({"agent": agent["id"], "condition": agent["condition"], "settings": settings,
                                                       "images": extra, "seconds": seconds}, indent=1, default=float), encoding="utf-8")
        plot(settings, extra, out / "tradeoff.png", f"{agent['id']}: constrained correction (A)")
        meta = {"agent": agent["id"], "condition": agent["condition"], "weights": {r: str(p) for r, p in agent["weights"].items()},
                "agents_file": str(args.agents), "config": str(args.config), "config_bases": cfg["_bases"], "e7": e7, "device": str(device),
                "torch_threads": torch.get_num_threads(), "seconds": seconds, "code": provenance}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "done", "seconds": seconds}), encoding="utf-8")
        print(f"[E7 constrained] {agent['id']}: {len(settings)} settings, {sum(r['kept'] for r in settings)} kept, {seconds:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
