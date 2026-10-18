"""E4 pilot analyses (wave 1), local, CPU (docs/E4_generalization_gradient.md, "pilot"). Every sub-command writes into
--out (a new directory) and prints its summary.

  convergence  from the runs' checkpoints.jsonl alone (what results/<batch> holds): at every checkpoint, the prior's
               mass on the true successor by region (every zone of E4 is deterministic) for each run, and b by region
               for each configuration (experimental - reference): the slots from the fixed walks on the training image
               (--measure: b_step1, I at the first step in the zone, the main measure -- see `probes`; or
               b_entry_mean_m, the mean over the first m steps; both are written), the corridor from E3's real-context
               I_s. Criterion (defaults, all parameters): mass >= 0.97 in the four quadrants of both runs (--mass-regions;
               the corridor is left out by default: its mean over states includes the goal, a terminal state from which
               the prior predicts the next episode's start, not the tensor's successors -- 0.05 to 0.40 at wave 1); |b| <=
               0.1 bit at the CD slot and in the corridor (the corridor region and the two corridor-coloured slots);
               |b_UD - (-log2 5)| <= 0.15 bit; on two consecutive checkpoints. B = the first checkpoint of such a pair of
               consecutive checkpoints for EVERY configuration of the wave; when a configuration never meets it, B = the
               last checkpoint (300,000) and the configuration is flagged.
  protocol     at the last checkpoint (the runs' "final" record): b by slot from the fixed walks (full settings), at the
               first step and as the mean over m steps, next to b from E3's end-of-training measure (real replay
               contexts over every state of the zone, E3's full settings).
  evolution    b by step since entry (1..n_in) on the training image, by slot: from the final records, or from a probe
               measurement (--probes).
  forgetting   needs the weights (--experimental-run/--reference-run, --checkpoint): a sample of walks filtered from
               several initial states -- the model's learned initial state and the states reached after other random
               corridor prefixes; by step, the distance between the h trajectories and the gap between the I_s values;
               the step after which the gap stays below --tol, against n_pre.
  probes       from measure_probes.py outputs (--probes, one directory per configuration): the extreme probes (training
               image: every slot; theme only: T_inc at the neutral slot; position only: the corridor's colour at the UD
               slot) and the full grid (b against the distance to T_inc, by axis and position, with figures). The main
               measure is b at the FIRST step in the zone (b_step1, the first encounter: the walks follow the reference's
               controllable dynamics in every slot, so from step 2 on the model's context shows its actions working);
               b by step since entry 2..n_in is the recovery measure; b at entry (mean over the first m steps,
               measure_probes.py's b_entry) stays reported as b_entry_mean_m.
  sufficiency  from measure_probes.py outputs saved with --save-steps: the standard error of b_step1 and of
               b_entry_mean_m against the number of walks used (bootstrap resampling of the walks, the same walks for
               both models), by (image, zone) and overall; the smallest number of walks whose median standard error of
               b_step1 is below --se-target, to set e4.walks.n_walks.

Inputs: --runs / --probes directories, or --export <dir>, the extracted archive of export_batch.py (its run and probe
directories are found under <dir>/results/E4_generalization_gradient/); forgetting takes --export with --configuration.
Checkpoints without diagnostics (later batches, run_e4.py --diagnostics-at) are skipped.

Usage, from the repo root:
  python experiments/E4_generalization_gradient/pilot_analysis.py convergence --runs <run dir> ... --out <dir>
  python experiments/E4_generalization_gradient/pilot_analysis.py protocol --runs <run dir> ... --out <dir>
  python experiments/E4_generalization_gradient/pilot_analysis.py evolution --runs <run dir> ... [--probes <dir> ...] --out <dir>
  python experiments/E4_generalization_gradient/pilot_analysis.py forgetting --configuration p1-a-d1 --checkpoint B \\
      --experimental-run <dir> --reference-run <dir> --out <dir>
  python experiments/E4_generalization_gradient/pilot_analysis.py probes --probes <dir> ... --out <dir>
  python experiments/E4_generalization_gradient/pilot_analysis.py sufficiency --probes <dir> ... --out <dir>
  (any of them with --export <extracted archive> instead of --runs / --probes / the run directories)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import e4_setup
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.probing import walks as walks_mod
from helplessness.probing.fixed_trajectory import fixed_trajectory_context, step_controllability

LOG2_5 = float(np.log2(5))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def export_dirs(root: Path) -> tuple[list[Path], list[Path]]:
    """The run and probe directories of an extracted export_batch.py archive."""
    base = root / "results" / e4_setup.EXPERIMENT_ID
    runs = sorted(d for d in base.glob("*__pair-*") if (d / "config.yaml").is_file())
    probes = sorted(d for d in base.glob("probes__*") if (d / "images.csv").is_file())
    return runs, probes


def find_in_export(root: Path, config_id: str, condition: str) -> Path:
    role = "experimental" if condition == "experimental" else "control"
    found = [d for d in export_dirs(root)[0] if f"__2zones-{config_id}__pair-{role}__" in d.name]
    if not found:
        raise SystemExit(f"no {condition} run of {config_id} in {root}")
    return found[-1]


def load_runs(run_dirs: list[Path]) -> dict:
    """configuration -> condition -> {"dir", "records" (checkpoints.jsonl, those with diagnostics), "roles"}."""
    runs: dict = {}
    for d in run_dirs:
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (d / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        records = [r for r in records if r.get("e4")]
        cid = cfg["e4"]["run_configuration"]
        conf = e4_setup.configuration(cfg, cid)
        runs.setdefault(cid, {})[meta["condition"]] = {"dir": d, "records": records, "roles": conf["roles"], "cfg": cfg}
    return runs


def _nan(v):
    return float("nan") if v is None else float(v)


WALK_MEASURES = ("b_step1", "b_entry_mean_m")


def walk_I(walk_rec: dict, measure: str) -> float:
    """I of one slot's fixed-walk record: at the first step in the zone (b_step1, the main measure: I_curve[0]) or the
    mean over the first m steps (b_entry_mean_m: I_entry)."""
    if measure == "b_step1":
        return _nan(walk_rec["I_curve"][0])
    if measure == "b_entry_mean_m":
        return _nan(walk_rec["I_entry"])
    raise ValueError(f"unknown measure {measure!r}")


def b_by_region(exp_rec: dict, ref_rec: dict, key: str = "walks", real_key: str = "real_context",
                measure: str = "b_step1") -> dict[str, float]:
    """b by region at one checkpoint: the slots from the fixed walks (`measure`), the corridor from E3's real-context I_s."""
    b = {slot: walk_I(exp_rec[key][slot], measure) - walk_I(ref_rec[key][slot], measure) for slot in QUADRANT_ORDER}
    b["corridor"] = _nan(exp_rec[real_key]["I_s_by_region"]["corridor"]) - _nan(ref_rec[real_key]["I_s_by_region"]["corridor"])
    return b


MASS_REGIONS = {"quadrants": tuple(QUADRANT_ORDER), "all": ("corridor", *QUADRANT_ORDER)}


def criterion(mass_exp: dict, mass_ref: dict, b: dict, roles: dict, args) -> dict:
    regions = MASS_REGIONS[args.mass_regions]
    mass_ok = all(_nan(m[r]) >= args.mass_min for m in (mass_exp, mass_ref) for r in regions)
    corridor_like = ["corridor", roles["neutral"], roles["other"]]
    b_cd_ok = abs(b[roles["cd"]]) <= args.b_cd_max and all(abs(b[r]) <= args.b_cd_max for r in corridor_like)
    b_ud_ok = abs(b[roles["ud"]] + LOG2_5) <= args.b_ud_tol
    return {"mass_ok": mass_ok, "b_cd_corridor_ok": bool(b_cd_ok), "b_ud_ok": bool(b_ud_ok), "all": bool(mass_ok and b_cd_ok and b_ud_ok)}


def cmd_convergence(args) -> dict:
    runs = load_runs(args.runs)
    rows, met = [], {}
    for cid, pair in sorted(runs.items()):
        if set(pair) != {"experimental", "reference"}:
            raise SystemExit(f"{cid}: needs one experimental and one reference run, got {sorted(pair)}")
        exp = {r["env_step"]: r for r in pair["experimental"]["records"]}
        ref = {r["env_step"]: r for r in pair["reference"]["records"]}
        steps = sorted(set(exp) & set(ref))
        flags = []
        for step in steps:
            e, r = exp[step]["e4"], ref[step]["e4"]
            mass_e, mass_r = e["real_context"]["mass_true_by_region"], r["real_context"]["mass_true_by_region"]
            b_all = {measure: b_by_region(e, r, measure=measure) for measure in WALK_MEASURES}
            b = b_all[args.measure]
            c = criterion(mass_e, mass_r, b, pair["experimental"]["roles"], args)
            flags.append(c["all"])
            role_of = {slot: role for role, slot in pair["experimental"]["roles"].items()}
            for region in ("corridor", *QUADRANT_ORDER):
                rows.append({"configuration": cid, "env_step": step, "region": region, "role": role_of.get(region, "corridor"),
                             "mass_exp": _nan(mass_e[region]), "mass_ref": _nan(mass_r[region]), "b": b[region],
                             **{measure: b_all[measure][region] for measure in WALK_MEASURES}, **c})
        first = next((steps[i] for i in range(len(steps) - 1) if flags[i] and flags[i + 1]), None)
        met[cid] = {"first_consecutive": first, "checkpoints": steps}
    last = max(max(m["checkpoints"]) for m in met.values())
    never = [cid for cid, m in met.items() if m["first_consecutive"] is None]
    B = last if never else max(m["first_consecutive"] for m in met.values())
    choice = {"B": int(B), "per_configuration": {cid: m["first_consecutive"] for cid, m in met.items()}, "never_converged": never,
              "criterion": {"measure": args.measure, "mass_regions": list(MASS_REGIONS[args.mass_regions]), "mass_min": args.mass_min,
                            "b_cd_max": args.b_cd_max, "b_ud_target": -LOG2_5, "b_ud_tol": args.b_ud_tol, "consecutive": 2}}
    _write_csv(args.out / "convergence.csv", rows)
    (args.out / "budget_choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for cid in sorted(runs):
        for region in ("corridor", *QUADRANT_ORDER):
            sel = [r for r in rows if r["configuration"] == cid and r["region"] == region]
            x = [r["env_step"] for r in sel]
            axes[0].plot(x, [r["b"] for r in sel], marker=".", label=f"{cid} {sel[0]['role']}")
            axes[1].plot(x, [min(r["mass_exp"], r["mass_ref"]) for r in sel], marker=".", label=f"{cid} {sel[0]['role']}")
    axes[0].axhline(-LOG2_5, color="k", ls=":")
    axes[0].axvline(B, color="r", ls="--")
    axes[1].axhline(args.mass_min, color="k", ls=":")
    for ax, title in zip(axes, (f"b by region (exp - ref, slots: {args.measure})", "mass on the true successor (min of the two runs)")):
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("environment steps")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out / "convergence.png", dpi=120)
    return choice


def cmd_protocol(args) -> dict:
    runs = load_runs(args.runs)
    rows = []
    for cid, pair in sorted(runs.items()):
        e, r = pair["experimental"]["records"][-1]["e4"], pair["reference"]["records"][-1]["e4"]
        if "final" not in e or "final" not in r:
            raise SystemExit(f"{cid}: no final protocol-control record (e4.final_protocol_control)")
        role_of = {slot: role for role, slot in pair["experimental"]["roles"].items()}
        for slot in QUADRANT_ORDER:
            b_walks = {measure: walk_I(e["final"]["walks"][slot], measure) - walk_I(r["final"]["walks"][slot], measure)
                       for measure in WALK_MEASURES}
            b_e3 = _nan(e["final"]["real_context"]["I_s_by_region"][slot]) - _nan(r["final"]["real_context"]["I_s_by_region"][slot])
            rows.append({"configuration": cid, "slot": slot, "role": role_of[slot], "b_fixed_walks_step1": b_walks["b_step1"],
                         "b_fixed_walks_entry_mean_m": b_walks["b_entry_mean_m"], "b_e3_real_context": b_e3,
                         "difference_step1": b_walks["b_step1"] - b_e3, "difference_entry_mean_m": b_walks["b_entry_mean_m"] - b_e3})
    _write_csv(args.out / "protocol_control.csv", rows)
    return {"rows": rows}


def cmd_evolution(args) -> dict:
    out = {}
    plt = _plt()
    if args.runs:
        runs = load_runs(args.runs)
        fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4), squeeze=False)
        for ax, (cid, pair) in zip(axes[0], sorted(runs.items())):
            e, r = pair["experimental"]["records"][-1]["e4"], pair["reference"]["records"][-1]["e4"]
            src = "final" if "final" in e and "final" in r else None
            ew, rw = (e[src]["walks"], r[src]["walks"]) if src else (e["walks"], r["walks"])
            role_of = {slot: role for role, slot in pair["experimental"]["roles"].items()}
            for slot in QUADRANT_ORDER:
                curve = np.array(ew[slot]["I_curve"], dtype=float) - np.array(rw[slot]["I_curve"], dtype=float)
                out[f"{cid}/{slot}"] = curve.tolist()
                ax.plot(np.arange(1, curve.size + 1), curve, marker=".", label=role_of[slot])
            ax.axhline(-LOG2_5, color="k", ls=":")
            ax.set_title(f"{cid}: b by step since entry (training image)")
            ax.set_xlabel("steps since entry")
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "evolution.png", dpi=120)
    (args.out / "evolution.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return {"n_curves": len(out)}


def cmd_forgetting(args) -> dict:
    from measure_probes import load_rssm

    if args.export is not None:
        args.experimental_run = find_in_export(args.export, args.configuration, "experimental")
        args.reference_run = find_in_export(args.export, args.configuration, "reference")
    if not (args.experimental_run and args.reference_run):
        raise SystemExit("forgetting needs --experimental-run and --reference-run, or --export")
    cfg = e4_setup.load_config(args.config)
    conf = e4_setup.configuration(cfg, args.configuration)
    device = torch.device("cpu")
    models = {name: load_rssm(run / "checkpoints" / f"agent_env{args.checkpoint:06d}.pt", device)
              for name, run in (("experimental", args.experimental_run), ("reference", args.reference_run))}
    world = e4_setup.probe_world(cfg, conf)
    succ = e4_setup.successors(world)
    w = cfg["e4"]["walks"]
    slot = conf["roles"][args.slot_role]
    main = e4_setup.walks_for_slot(cfg, slot, args.n_walks)
    valid = main["phase"] != walks_mod.PHASE_PAD
    report = {"slot": slot, "n_pre": w["n_pre"], "n_walks": args.n_walks, "n_prefixes": args.n_prefixes, "tol": args.tol, "models": {}}
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, rssm in models.items():
        seed = e4_setup.measurement_seed(cfg, conf, 0, slot, "forgetting")
        gen = torch.Generator().manual_seed(seed + 1)  # the prefixes' own draws
        starts = [None]
        for k in range(args.n_prefixes):  # other initial states: the end of another random corridor prefix
            prefix = walks_mod.generate_walks(world.env.transition_tensor(), e4_setup.two_zones.corridor_states(world.layout),
                                              e4_setup.two_zones.region_states(world.layout)[slot], args.n_walks, w["n_pre"], 1,
                                              np.random.default_rng(np.random.SeedSequence([w["seed"], 7, k])))
            pre_steps = w["n_pre"] + 1
            pre = {key: (v[:, :pre_steps] if v.ndim == 2 else v) for key, v in prefix.items()}
            ctx = fixed_trajectory_context(rssm, world.templates_np, pre, generator=gen)
            a_last = torch.nn.functional.one_hot(torch.as_tensor(pre["actions"][:, pre_steps - 2]), rssm.cfg.n_actions).float()
            starts.append((ctx["h"][:, -1], ctx["z"][:, -1], a_last))
        hs, Is = [], []
        for start in starts:  # the same draws for every start: the gaps come from h alone
            gen_k = torch.Generator().manual_seed(seed)
            ctx = fixed_trajectory_context(rssm, world.templates_np, main, generator=gen_k, initial=start)
            mask = torch.as_tensor(valid)
            res = step_controllability(rssm, ctx["h"][mask], ctx["z"][mask], world.templates_t, args.draws,
                                       cfg["measure"]["kernel_margin_threshold"], generator=gen_k)
            I = np.full(main["states"].shape, np.nan)
            I[valid] = res["I"]
            hs.append(ctx["h"].numpy())
            Is.append(I)
        T = main["states"].shape[1]
        dist = np.array([[np.nanmean(np.where(valid[:, t], np.linalg.norm(hs[k][:, t] - hs[0][:, t], axis=-1), np.nan)) for t in range(T)]
                         for k in range(1, len(starts))])
        gap = np.array([[np.nanmean(np.where(valid[:, t], np.abs(Is[k][:, t] - Is[0][:, t]), np.nan)) for t in range(T)]
                        for k in range(1, len(starts))])
        worst = np.nanmax(gap, axis=0)
        above = np.flatnonzero(~(worst < args.tol))
        settle = int(above[-1] + 1) if above.size else 0
        settle_corridor = int(np.flatnonzero(~(worst[: w["n_pre"] + 1] < args.tol))[-1] + 1) if (~(worst[: w["n_pre"] + 1] < args.tol)).any() else 0
        report["models"][name] = {"steps_until_gap_below_tol_within_corridor": settle_corridor, "steps_until_gap_below_tol_whole_walk": settle,
                                  "below_n_pre": bool(settle_corridor < w["n_pre"]), "h_distance": dist.tolist(), "I_gap": gap.tolist()}
        axes[0].plot(np.nanmean(dist, axis=0), label=name)
        axes[1].plot(worst, label=name)
    for ax, title in zip(axes, ("mean |h - h_ref_start| by step", "max over starts of mean |I - I_ref_start| by step")):
        ax.axvline(w["n_pre"], color="k", ls=":")
        ax.set_title(title)
        ax.set_xlabel("steps since the walk's start")
        ax.legend()
    axes[1].axhline(args.tol, color="r", ls="--")
    fig.tight_layout()
    fig.savefig(args.out / "forgetting.png", dpi=120)
    (args.out / "forgetting.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return {name: {k: v for k, v in m.items() if not isinstance(v, list)} for name, m in report["models"].items()}


def probe_measures(d: Path) -> tuple[list[dict], list[dict]]:
    """images.csv and probes.csv of one measure_probes.py output, each row with its b at the first step in the zone
    (b_step1, the main measure: the first encounter, from curves.npz), the distinct states behind it (n_states_step1)
    and b by step since entry (b_curve, 1..n_in: steps 2..n_in are the recovery measure); b_entry (the mean over the
    first m steps) is kept as it is."""
    with open(d / "images.csv", encoding="utf-8") as f:
        images = list(csv.DictReader(f))
    with open(d / "probes.csv", encoding="utf-8") as f:
        probes = list(csv.DictReader(f))
    curves = np.load(d / "curves.npz")
    b_curves = curves["curve_exp"] - curves["curve_ref"]
    row_of = {}
    for i, r in enumerate(images):
        r["b_curve"] = b_curves[i]
        r["b_step1"] = float(b_curves[i, 0])
        r["n_states_step1"] = int(curves["n_states"][i, 0])
        row_of[(r["image"], r["zone"])] = r
    for p in probes:
        r = row_of[(p["image"], p["slot"])]
        p["b_step1"], p["n_states_step1"], p["b_curve"] = r["b_step1"], r["n_states_step1"], r["b_curve"]
    return images, probes


def _extreme(r: dict) -> dict:
    return {"b_step1": r["b_step1"], "b_entry_mean_m": float(r["b_entry"]), "n_states_step1": r["n_states_step1"],
            "b_curve": [float(v) for v in r["b_curve"] if np.isfinite(v)]}


def cmd_probes(args) -> dict:
    plt = _plt()
    summary = {}
    for d in args.probes:
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        cid = meta["configuration"]
        images, probes = probe_measures(d)
        train = {r["role"]: _extreme(r) for r in images if r["is_training"] == "True"}
        theme_only = [_extreme(p) for p in probes if p["axis"] == "segment" and int(p["k"]) == 0 and p["position"] == "neutral"]
        position_only = [_extreme(p) for p in probes if p["axis"] == "position_only"]
        b1 = {role: v["b_step1"] for role, v in train.items()}
        summary[cid] = {"main_measure": "b_step1 (b at the first step in the zone); b_entry_mean_m and b_curve kept",
                        "training_image": train,
                        "checks": {"b_ud_negative": b1.get("ud", np.nan) < 0, "b_cd_near_zero": abs(b1.get("cd", np.nan)) <= 0.1},
                        "theme_only_T_inc_at_neutral": theme_only[0] if theme_only else None,
                        "position_only_corridor_colour_at_ud": position_only[0] if position_only else None}
        extremes = [("training", role, v) for role, v in train.items()] + [("theme_only", "neutral", v) for v in theme_only] \
            + [("position_only", "ud", v) for v in position_only]
        _write_csv(args.out / f"extremes_{cid}.csv",
                   [{"probe": probe, "zone_role": role, "b_step1": v["b_step1"], "b_entry_mean_m": v["b_entry_mean_m"],
                     "n_states_step1": v["n_states_step1"], **{f"b_k{k + 1}": b for k, b in enumerate(v["b_curve"])}}
                    for probe, role, v in extremes])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        for ax, axis in zip(axes, ("segment", "arc_ctrl", "arc_inc")):
            for position in ("ud", "cd", "neutral"):
                sel = sorted((float(p["dist_to_inc"]), p["b_step1"]) for p in probes if p["axis"] == axis and p["position"] == position)
                ax.plot([s[0] for s in sel], [s[1] for s in sel], marker="o", label=position)
            ax.axhline(-LOG2_5, color="k", ls=":")
            ax.axhline(0.0, color="grey", lw=0.5)
            ax.set_title(f"{cid} -- {axis}")
            ax.set_xlabel("RGB distance to T_inc")
        axes[0].set_ylabel("b at the first step in the zone (bits)")
        axes[0].legend(title="position")
        fig.tight_layout()
        fig.savefig(args.out / f"grid_{cid}.png", dpi=120)
        _write_csv(args.out / f"grid_{cid}.csv", [{**{k: p[k] for k in ("axis", "k", "t", "position", "dist_to_inc", "dist_to_ctrl")},
                                                    "b_step1": p["b_step1"], "n_states_step1": p["n_states_step1"], "b_entry_mean_m": p["b_entry"]}
                                                   for p in probes])
    (args.out / "probes_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary


MEASURES = {"b_step1": lambda m: (1, 1), "b_entry_mean_m": lambda m: (1, m)}  # steps since entry averaged, from m


def b_resampled(steps: dict, key: str, counts: np.ndarray, k_first: int, k_last: int) -> float:
    """b over the steps since entry k_first..k_last (by state, then by zone) with every walk weighted by its count in a
    resample: (1, 1) is b at the first step in the zone, (1, m) b at entry."""
    k, states, walk = steps[f"{key}_steps_since_entry"], steps[f"{key}_states"].astype(np.int64), steps[f"{key}_walk"].astype(np.int64)
    sel = (k >= k_first) & (k <= k_last)
    out = []
    for name in ("experimental", "reference"):
        I = steps[f"{key}_{name}_I"].astype(np.float64)
        ok = sel & np.isfinite(I)
        w = counts[walk[ok]].astype(np.float64)
        s = states[ok]
        tot = np.bincount(s, weights=w * I[ok], minlength=e4_setup.two_zones.N_STATES)
        n = np.bincount(s, weights=w, minlength=e4_setup.two_zones.N_STATES)
        out.append(float(np.mean(tot[n > 0] / n[n > 0])) if (n > 0).any() else float("nan"))
    return out[0] - out[1]


def cmd_sufficiency(args) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for d in args.probes:
        if not (d / "steps.npz").is_file():
            raise SystemExit(f"{d}: no steps.npz (measure_probes.py --save-steps)")
        cfg = e4_setup.load_config(args.config)
        m = cfg["e4"]["walks"]["m"]
        steps = dict(np.load(d / "steps.npz"))
        keys = sorted({k[: k.index("_", 5)] for k in steps if k.endswith("_steps_since_entry")})
        for key in keys:
            n_total = int(steps[f"{key}_walk"].max()) + 1
            for measure, k_range in MEASURES.items():
                full = b_resampled(steps, key, np.ones(n_total), *k_range(m))
                for n in [n for n in args.n_walks if n <= n_total]:
                    draws = [b_resampled(steps, key, np.bincount(rng.integers(0, n_total, n), minlength=n_total), *k_range(m))
                             for _ in range(args.bootstrap)]
                    rows.append({"probes_dir": d.name, "image_zone": key, "measure": measure, "n_walks": n, "b_all_walks": full,
                                 "se": float(np.nanstd(draws, ddof=1))})
    _write_csv(args.out / "sufficiency.csv", rows)
    summary = {}
    for measure in MEASURES:
        summary[measure] = {}
        for n in sorted({r["n_walks"] for r in rows}):
            se = [r["se"] for r in rows if r["measure"] == measure and r["n_walks"] == n and np.isfinite(r["se"])]
            summary[measure][n] = {"median_se": float(np.median(se)) if se else float("nan"),
                                   "max_se": float(np.max(se)) if se else float("nan"), "pairs": len(se)}
    enough = [n for n, s in summary["b_step1"].items() if s["median_se"] <= args.se_target]  # the main measure decides
    result = {"main_measure": "b_step1", "se_by_measure_and_n_walks": summary, "se_target": args.se_target,
              "recommended_n_walks": min(enough) if enough else None}
    (args.out / "sufficiency.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for measure, style in (("b_step1", "-"), ("b_entry_mean_m", "--")):
        ns = sorted(summary[measure])
        ax.plot(ns, [summary[measure][n]["median_se"] for n in ns], ls=style, marker="o", label=f"{measure}: median over (image, zone)")
        ax.plot(ns, [summary[measure][n]["max_se"] for n in ns], ls=style, marker=".", label=f"{measure}: max")
    ax.axhline(args.se_target, color="r", ls="--")
    ax.set_xlabel("walks")
    ax.set_ylabel("standard error of b (bits)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "sufficiency.png", dpi=120)
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("convergence", "protocol", "evolution", "forgetting", "probes", "sufficiency"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--runs", type=Path, nargs="*", default=[])
        p.add_argument("--probes", type=Path, nargs="*", default=[])
        p.add_argument("--export", type=Path, default=None, help="an extracted export_batch.py archive")
    s = sub.choices["sufficiency"]
    s.add_argument("--config", type=Path, default=e4_setup.DEFAULT_CONFIG)
    s.add_argument("--n-walks", type=int, nargs="+", default=[25, 50, 100, 150, 200, 300])
    s.add_argument("--bootstrap", type=int, default=200)
    s.add_argument("--se-target", type=float, default=0.05)
    c = sub.choices["convergence"]
    c.add_argument("--mass-min", type=float, default=0.97)
    c.add_argument("--b-cd-max", type=float, default=0.1)
    c.add_argument("--b-ud-tol", type=float, default=0.15)
    c.add_argument("--measure", default="b_step1", choices=WALK_MEASURES, help="the slots' b (default: at the first step in the zone)")
    c.add_argument("--mass-regions", default="quadrants", choices=tuple(MASS_REGIONS),
                   help="regions whose mass must reach --mass-min (default: the four quadrants; 'all' adds the corridor, whose "
                        "mean includes the terminal goal state)")
    f = sub.choices["forgetting"]
    f.add_argument("--config", type=Path, default=e4_setup.DEFAULT_CONFIG)
    f.add_argument("--configuration", required=True)
    f.add_argument("--checkpoint", type=int, required=True)
    f.add_argument("--experimental-run", type=Path, default=None)
    f.add_argument("--reference-run", type=Path, default=None)
    f.add_argument("--slot-role", default="ud", choices=("ud", "cd", "neutral", "other"))
    f.add_argument("--n-walks", type=int, default=20)
    f.add_argument("--n-prefixes", type=int, default=3)
    f.add_argument("--draws", type=int, default=8)
    f.add_argument("--tol", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.export is not None and args.command != "forgetting":
        runs, probes = export_dirs(args.export)
        args.runs, args.probes = args.runs or runs, args.probes or probes
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"convergence": cmd_convergence, "protocol": cmd_protocol, "evolution": cmd_evolution, "forgetting": cmd_forgetting,
              "probes": cmd_probes, "sufficiency": cmd_sufficiency}[args.command](args)
    print(json.dumps(result, indent=1, default=float)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
