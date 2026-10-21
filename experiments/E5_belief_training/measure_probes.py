"""E5's zero-data probes at one checkpoint (docs/E5_belief_training.md, "measures"): E4's measure unchanged
(probing.theme_probe_measure: the 93 probe images -- 3 axes x 11 themes x 3 positions + position only --, the same
fixed walks, frozen weights, b = I_experimental - I_reference at the first step in the zone and by step since entry).
The positions are those of the configuration's CANONICAL layout for every condition (under shuffle, "former UD slot" is
the canonical UD slot). The measurement seeds are those of the run's seed (E4's rule: the same for both models).

The pair:
  baseline / shuffle  the experimental run and the reference run of the same training and seed, both at checkpoint k;
  immunisation        the immunisation run at k steps AFTER ITS SWITCH (its weights agent_env<switch + k>.pt, the switch
                      read from its records) against the baseline reference of the same seed at checkpoint k
                      (docs/e5/DECISIONS.md, (7)); for seed A that reference is E4's own (--reference-run in E4's results).

Walks: e5.probes.n_walks of configs/e5/e5.yaml (150; E4's probes used 300 -- docs/e5/DECISIONS.md (25)), the first ones
of E4's fixed walks; --n-walks overrides it.

Validity (docs/e5/DECISIONS.md (22), helplessness.probing.probe_validity, E4's final criterion): a probe's b is
interpretable only if the REFERENCE of the pair (the same condition and seed; the baseline reference for the
immunisation) predicts the transitions at the first step in the zone: mass on the true successor >= 0.90 and
|I_ref - log2 5| <= 0.15, read from the same measurement (its steps in the zone at step 1). images.csv and probes.csv
gain mass/ghost/I at step 1 of both models and `valid`; validity_summary.json gives the share of valid probes, overall and
by axis and position. The reconstruction diagnostic (both models, the first --diagnose-walks walks; 0 = none):
reconstruction.csv, one row per (image, zone, model, part) -- the reconstruction error, the agent's decoded position
(kernel rule and direct), the colour the reconstruction gives the zone of interest and its distance to the probe's colour
and to every colour that slot takes in training (the canonical training colour, T_inc, T_ctrl, the corridor's: under
shuffle a slot is not tied to one colour).

Outputs, as E4's (images.csv, probes.csv, curves.npz, steps.npz, meta.yaml, status.json) plus reconstruction.csv and
validity_summary.json, in a new directory <out-dir>/E5_belief_training/probes__<configuration>__<tag>__seed<NN>__k<NNNNNN>__<UTC>/.

Usage, from the repo root:
  python experiments/E5_belief_training/measure_probes.py --checkpoint <k> --experimental-run <dir> --reference-run <dir>
      [--images all | training | extreme | 0 5 17] [--n-walks N] [--device cuda] [--out-dir results] [--save-steps]
  python experiments/E5_belief_training/measure_probes.py --checkpoint <k> --find-runs results --configuration p1-a-d1       --training shuffle --fraction 1.0 --seed-label A [--e4-results <E4 results root>] [--config configs/e5/e5.yaml] ...
--find-runs <root>: the newest runs of the pair under <root>/E5_belief_training/ (the cluster's probe phase); the
reference of immunisation A is E4's reference run of the configuration, under --e4-results.
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

import e5_setup
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing import probe_validity
from helplessness.probing.theme_probe_measure import is_extreme, load_rssm, measure, probe_rows, sha256, write_csv
from helplessness.probing.theme_probes import probe_images


def run_info(run_dir: Path) -> dict:
    """What a run is: its config, seed, training (E4's runs: baseline), condition, and its effective switch."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    spec = cfg.get("e5", {}).get("run") or {"training": "baseline", "configuration": cfg["e4"]["run_configuration"],
                                            "condition": meta["condition"]}  # an E4 run
    info = {"cfg": cfg, "seed": int(meta["seed"]), "condition": meta["condition"], "training": spec["training"],
            "configuration": spec["configuration"], "fraction": spec.get("fraction"), "switch": None}
    if spec["training"] == "immunisation":
        effs = {r["e5"]["dynamics_switch_step"] for r in e5_setup.records(run_dir) if r.get("e5")}
        effs.discard(None)
        if len(effs) != 1:
            raise SystemExit(f"{run_dir}: no single effective switch step in its records ({effs})")
        info["switch"] = effs.pop()
    return info


def weights_path(run_dir: Path, info: dict, k: int) -> Path:
    step = k if info["switch"] is None else info["switch"] + k
    path = run_dir / "checkpoints" / f"agent_env{step:06d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def find_pair(args) -> tuple[Path, Path]:
    cfg = e5_setup.load_config(args.config)
    seed = e5_setup.configuration(cfg, args.configuration, args.seed_label)["seed"]
    root, cid = args.find_runs, args.configuration
    if args.training == "immunisation":
        exp = e5_setup.find_run(root, cid, "immunisation", None, seed)
        if args.seed_label == "A":
            if args.e4_results is None:
                raise SystemExit("immunisation A: its reference is E4's -- give --e4-results")
            ref = e5_setup.find_e4_run(args.e4_results, cid, "reference", need=f"checkpoints/agent_env{args.checkpoint:06d}.pt")
        else:
            ref = e5_setup.find_run(root, cid, "baseline", "reference", seed, need=f"checkpoints/agent_env{args.checkpoint:06d}.pt")
        return exp, ref
    if args.training == "baseline" and args.seed_label == "A":  # seed A's baseline is E4's own pair (not retrained)
        if args.e4_results is None:
            raise SystemExit("baseline A: E4's own runs -- give --e4-results")
        need = f"checkpoints/agent_env{args.checkpoint:06d}.pt"
        return (e5_setup.find_e4_run(args.e4_results, cid, "experimental", need=need),
                e5_setup.find_e4_run(args.e4_results, cid, "reference", need=need))
    return (e5_setup.find_run(root, cid, args.training, "experimental", seed, args.fraction),
            e5_setup.find_run(root, cid, args.training, "reference", seed, args.fraction, need=f"checkpoints/agent_env{args.checkpoint:06d}.pt"))


def check_pair(exp: dict, ref: dict) -> None:
    if exp["configuration"] != ref["configuration"] or exp["seed"] != ref["seed"]:
        raise SystemExit(f"not a pair: {exp['configuration']} seed {exp['seed']} vs {ref['configuration']} seed {ref['seed']}")
    if ref["condition"] != "reference" or ref["training"] == "immunisation":
        raise SystemExit("the reference run must be a reference (baseline or shuffle)")
    if exp["training"] == "immunisation":
        if ref["training"] != "baseline":
            raise SystemExit("the immunisation is measured against the baseline reference of its seed")
    elif exp["condition"] != "experimental" or (exp["training"], exp["fraction"]) != (ref["training"], ref["fraction"]):
        raise SystemExit("the experimental and reference runs must share their training")


def validity_hook(conf: dict, models: dict, diagnose_walks: int, diagnostics: list[dict]):
    """measure()'s on_pair: mass, ghost and I at step 1 of both models, the probe's validity (the reference's), and the
    reconstruction diagnostic of both models on the first diagnose_walks walks (appended to diagnostics)."""
    references = {"canonical_training": None, "T_inc": conf["t_inc"], "T_ctrl": conf["t_ctrl"], "corridor": tuple(int(v) for v in CORRIDOR_COLOR)}

    def on_pair(image, slot, res, world, walks):
        row = {}
        for name, tag in (("experimental", "exp"), ("reference", "ref")):
            summary = probe_validity.step1_summary(res[name])
            row.update({f"{key[:-len('_step1')]}_{tag}_step1": value for key, value in summary.items() if key != "n_states_step1"})
        row["valid"] = probe_validity.is_valid(row["mass_ref_step1"], row["I_ref_step1"])
        if diagnose_walks:
            sub = {key: (value[:diagnose_walks] if getattr(value, "ndim", 0) >= 1 else value) for key, value in walks.items()}
            colours = {**references, "canonical_training": conf["training_colors"][slot]}
            for name, model in models.items():
                gen = torch.Generator(device=next(model.parameters()).device)
                gen.manual_seed(two_zones_setup.measurement_seed(conf["cfg"], conf, image["index"], slot, "probe"))
                for part in probe_validity.reconstruction_diagnostic(model, world, sub, slot, image["slot_colors"][slot], colours,
                                                                     conf["agent_color"], gen):
                    diagnostics.append({"image": image["index"], "zone": slot, "model": name, "valid": row["valid"], **part})
        return row

    return on_pair


def validity_summary(probes: list[dict]) -> dict:
    """The share of valid probes, overall and by (axis, position)."""
    out = {"criterion": {"mass_ref_step1_min": probe_validity.MASS_MIN, "abs_I_ref_step1_minus_log2_5_max": probe_validity.I_TOL},
           "probes": len(probes), "valid": sum(bool(p["valid"]) for p in probes), "by_axis_position": {}}
    out["share_valid"] = out["valid"] / out["probes"] if probes else float("nan")
    for p in probes:
        cell = out["by_axis_position"].setdefault(f"{p['axis']}/{p['position']}", {"probes": 0, "valid": 0})
        cell["probes"] += 1
        cell["valid"] += bool(p["valid"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=int, required=True, help="k: environment steps (after the switch for the immunisation)")
    parser.add_argument("--experimental-run", type=Path, default=None)
    parser.add_argument("--reference-run", type=Path, default=None)
    parser.add_argument("--find-runs", type=Path, default=None, help="results root where to find the pair (instead of the two above)")
    parser.add_argument("--config", type=Path, default=e5_setup.DEFAULT_CONFIG, help="--find-runs: the E5 config (the seeds)")
    parser.add_argument("--configuration", default=None)
    parser.add_argument("--training", choices=e5_setup.TRAININGS, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--seed-label", choices=e5_setup.SEED_LABELS, default=None)
    parser.add_argument("--e4-results", type=Path, default=None, help="E4's results root (the reference of immunisation A)")
    parser.add_argument("--images", nargs="+", default=["all"], help="all, training, extreme, or image indices")
    parser.add_argument("--n-walks", type=int, default=None, help="default: e5.probes.n_walks of --config (150)")
    parser.add_argument("--diagnose-walks", type=int, default=50, help="walks of the reconstruction diagnostic (0: none)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=e5_setup.REPO_ROOT / "results")
    parser.add_argument("--save-steps", action="store_true")
    args = parser.parse_args(argv)
    if args.find_runs is not None:
        if not (args.configuration and args.training and args.seed_label):
            parser.error("--find-runs needs --configuration, --training and --seed-label")
        args.experimental_run, args.reference_run = find_pair(args)
    elif not (args.experimental_run and args.reference_run):
        parser.error("give --experimental-run and --reference-run, or --find-runs")
    if args.n_walks is None:
        args.n_walks = int(e5_setup.load_config(args.config)["e5"]["probes"]["n_walks"])
    exp, ref = run_info(args.experimental_run), run_info(args.reference_run)
    check_pair(exp, ref)
    cfg = exp["cfg"]
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    conf = {**two_zones_setup.configuration(cfg, exp["configuration"]), "seed": exp["seed"]}  # the run's seed (A or B)
    all_images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    if args.images == ["all"]:
        images = all_images
    elif args.images == ["training"]:
        images = [im for im in all_images if im["is_training"]]
    elif args.images == ["extreme"]:
        images = [im for im in all_images if is_extreme(im)]
    else:
        images = [all_images[int(i)] for i in args.images]
    tag = e5_setup.training_tag(exp["training"], exp["fraction"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / e5_setup.EXPERIMENT_ID / f"probes__{exp['configuration']}__{tag}__seed{exp['seed']:02d}__k{args.checkpoint:06d}__{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    status = out / "status.json"
    status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
    try:
        weights = {"experimental": weights_path(args.experimental_run, exp, args.checkpoint),
                   "reference": weights_path(args.reference_run, ref, args.checkpoint)}
        models = {name: load_rssm(path, device) for name, path in weights.items()}
        print(f"[E5 measure_probes] {exp['configuration']} {tag} seed {exp['seed']} at k = {args.checkpoint}: {len(images)} images; "
              f"weights {weights['experimental'].name} / {weights['reference'].name}", flush=True)
        t0 = time.perf_counter()
        diagnostics: list[dict] = []
        hook = validity_hook({**conf, "cfg": cfg}, models, args.diagnose_walks, diagnostics)
        result = measure(cfg, conf, models, images, device, save_steps=args.save_steps, n_walks=args.n_walks, on_pair=hook)
        seconds = time.perf_counter() - t0
    except BaseException as err:
        status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                      "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
        raise
    write_csv(out / "images.csv", result["rows"])
    by_pair = {(r["image"], r["zone"]): r for r in result["rows"]}
    probes = [{**p, **{key: by_pair[(p["image"], p["slot"])][key] for key in ("mass_ref_step1", "ghost_ref_step1", "I_ref_step1", "valid")}}
              for p in probe_rows(conf, images, result["rows"])]
    write_csv(out / "probes.csv", probes)
    if diagnostics:
        write_csv(out / "reconstruction.csv", diagnostics)
    summary = validity_summary(probes)
    (out / "validity_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    np.savez_compressed(out / "curves.npz", curve_exp=result["curve_exp"], curve_ref=result["curve_ref"], n_states=result["n_states_curve"])
    if args.save_steps:
        np.savez_compressed(out / "steps.npz", **result["steps"])
    meta = {"configuration": exp["configuration"], "training": exp["training"], "fraction": exp["fraction"], "seed": exp["seed"],
            "checkpoint_k": args.checkpoint, "dynamics_switch_step": exp["switch"],
            "weights": {k: str(v) for k, v in weights.items()}, "weights_sha256": {k: sha256(p) for k, p in weights.items()},
            "runs": {"experimental": str(args.experimental_run), "reference": str(args.reference_run)}, "device": str(device),
            "torch_threads": torch.get_num_threads(), "n_images": len(images), "n_image_zones": len(result["rows"]), "seconds": seconds,
            "walks": {**cfg["e4"]["walks"], "n_walks_used": args.n_walks}, "probe": cfg["e4"]["probe"], "diagnose_walks": args.diagnose_walks,
            "validity": {"valid": summary["valid"], "probes": summary["probes"]},
            "t_inc": list(conf["t_inc"]), "t_ctrl": list(conf["t_ctrl"]),
            "positions": "the configuration's canonical layout"}
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    status.write_text(json.dumps({"status": "done", "seconds": seconds, "n_image_zones": len(result["rows"])}), encoding="utf-8")
    print(f"[E5 measure_probes] {len(result['rows'])} (image, zone) measured in {seconds:.0f} s; {summary['valid']}/{summary['probes']} "
          f"probes valid -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
