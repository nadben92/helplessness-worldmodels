"""E4's zero-data probes of one training configuration at one checkpoint (docs/E4_generalization_gradient.md, the
measure): the 93 distinct probe images (helplessness.probing.theme_probes) and, for each image and each of its zones of
interest, the fixed walks into that slot (the same for every image and both models), filtered by the experimental
(exposed) model and by the reference model, frozen at the checkpoint (their weights checkpoints/agent_env<NNNNNN>.pt);
I_s at every step in the zone only (helplessness.probing.fixed_trajectory.measure_walks: the corridor steps are
filtered, never decoded; the same measurement seed for the two models). b = I_experimental - I_reference. The extreme
probes (the training image, theme only = T_inc at the neutral slot, position only) use the walks' full n_in steps in the
zone (e4.walks.n_in, for the evolution in the zone); every other image the same walks cut after n_in_grid steps.

Outputs, in a new directory <out-dir>/E4_generalization_gradient/probes__<configuration>__env<NNNNNN>__<UTC>/:
  images.csv     one row per (image, zone of interest): the slot colours, the role of the zone, I at entry for both
                 models and b at entry (mean over the first m steps in the zone, by state then by zone), the states
                 covered, the mean ghost fraction, the decode count, the seconds taken;
  probes.csv     one row per probe (axis, k, t, position): its image, its zone, b at entry; the distance of the probe's
                 colour to T_inc and to T_ctrl (RGB);
  curves.npz     b and I by step since entry (1..n_in) for every (image, zone): arrays curve_exp, curve_ref (rows = rows of
                 images.csv), n_states;
  steps.npz      every measured step (optional, --save-steps): I, state, steps since entry / start, walk, both models;
  meta.yaml      the configuration, the two runs and their weights' SHA-256, the checkpoint, the settings, the timing;
  status.json    "running", then "done" or "failed" with the error and its traceback.

Usage, from the repo root (CPU or GPU):
  python experiments/E4_generalization_gradient/measure_probes.py --configuration p1-a-d1 --checkpoint 300000 \\
      (--experimental-run <exp run dir> --reference-run <ref run dir> | --find-runs results)
      [--images all | training | extreme | 0 5 17] [--n-walks N] [--device cuda] [--out-dir results] [--save-steps]
--find-runs <root>: the newest experimental and reference runs of the configuration under <root>/E4_generalization_gradient/
whose weights at the checkpoint exist (the cluster's probe phase).
"""

from __future__ import annotations

import argparse
import csv
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

import e4_setup
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig
from helplessness.envs.appearance.theme_axes import rgb_distance
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline.runtime import configure_runtime, max_images_per_forward
from helplessness.probing.fixed_trajectory import measure_walks
from helplessness.probing.theme_probes import probe_images


def load_rssm(path: Path, device: torch.device) -> RSSM:
    state = torch.load(path, map_location="cpu", weights_only=False)
    rssm = RSSM(RSSMConfig(**state["rssm_config"]))
    rssm.load_state_dict(state["rssm"])
    return rssm.to(device).eval()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_extreme(image: dict) -> bool:
    """The training image, theme only (T_inc at the neutral slot) and position only."""
    return image["is_training"] or any((p["axis"], p["k"], p["position"]) in (("segment", 0, "neutral"), ("position_only", 0, "ud"))
                                       for p in image["probes"])


def find_run(root: Path, config_id: str, condition: str, checkpoint: int) -> Path:
    role = "experimental" if condition == "experimental" else "control"
    found = sorted(d for d in (root / e4_setup.EXPERIMENT_ID).glob(f"*__2zones-{config_id}__pair-{role}__*")
                   if (d / "checkpoints" / f"agent_env{checkpoint:06d}.pt").exists())
    if not found:
        raise FileNotFoundError(f"no {condition} run of {config_id} with weights at {checkpoint} under {root}")
    return found[-1]


def measure(cfg: dict, conf: dict, models: dict, images: list[dict], device, log=lambda s: print(s, flush=True), save_steps: bool = False,
            n_walks: int | None = None) -> dict:
    """Every (image, zone) of `images` on both models. Returns rows (images.csv), curves and the optional steps."""
    w, draws = cfg["e4"]["walks"], cfg["e4"]["probe"]["draws"]
    margin = cfg["measure"]["kernel_margin_threshold"]
    role_of = {slot: role for role, slot in conf["roles"].items()}
    rows, curves_exp, curves_ref, n_states_curve, steps = [], [], [], [], {}
    for image in images:
        world = e4_setup.probe_world(cfg, conf, image["slot_colors"])
        succ = e4_setup.successors(world)
        n_in = w["n_in"] if is_extreme(image) else w.get("n_in_grid", w["n_in"])
        for slot in image["zones"]:
            walks = e4_setup.walks_for_slot(cfg, slot, n_walks, n_in)
            seed = e4_setup.measurement_seed(cfg, conf, image["index"], slot, "probe")
            t0 = time.perf_counter()
            res = {name: measure_walks(model, world.templates_np, world.templates_t, walks, succ, draws, margin, seed, w["m"], n_in,
                                       chunk=w["chunk"], max_images=max_images_per_forward(cfg))
                   for name, model in models.items()}
            seconds = time.perf_counter() - t0
            e, r = res["experimental"], res["reference"]
            rows.append({"image": image["index"], "is_training": image["is_training"], "extreme": is_extreme(image), "zone": slot,
                         "role": role_of[slot], "n_in": n_in, "n_walks": int(walks["states"].shape[0]),
                         **{f"color_{q}": "-".join(map(str, image["slot_colors"][q])) for q in QUADRANT_ORDER},
                         "I_entry_exp": e["entry"], "I_entry_ref": r["entry"], "b_entry": e["entry"] - r["entry"],
                         "n_states_entry_exp": e["n_states_entry"], "n_states_entry_ref": r["n_states_entry"],
                         "ghost_exp": float(np.nanmean(e["ghost_fraction"])), "ghost_ref": float(np.nanmean(r["ghost_fraction"])),
                         "mass_true_exp": float(np.nanmean(e["mass_true"])), "mass_true_ref": float(np.nanmean(r["mass_true"])),
                         "decodes_exp": e["n_unique_decodes"], "decodes_ref": r["n_unique_decodes"], "seconds": seconds,
                         "n_probes": len(image["probes"])})
            pad = w["n_in"] - n_in  # every curve on 1..n_in (NaN beyond a grid image's n_in_grid)
            curves_exp.append(np.concatenate([e["curve"], np.full(pad, np.nan)]))
            curves_ref.append(np.concatenate([r["curve"], np.full(pad, np.nan)]))
            n_states_curve.append(np.concatenate([np.minimum(e["n_states_curve"], r["n_states_curve"]), np.zeros(pad, dtype=np.int64)]))
            if save_steps:
                key = f"i{image['index']:03d}_{slot}"
                for field in ("states", "steps_since_entry", "walk"):  # the same walks for both models
                    steps[f"{key}_{field}"] = np.asarray(res["experimental"][field], dtype=np.int16)
                for name, res_m in res.items():
                    steps[f"{key}_{name}_I"] = np.asarray(res_m["I"], dtype=np.float32)
            log(f"image {image['index']:3d} zone {slot} ({role_of[slot]}): b_entry {rows[-1]['b_entry']:+.3f} "
                f"(I exp {e['entry']:.3f}, ref {r['entry']:.3f}; {seconds:.0f} s)")
    return {"rows": rows, "curve_exp": np.array(curves_exp), "curve_ref": np.array(curves_ref),
            "n_states_curve": np.array(n_states_curve), "steps": steps}


def probe_rows(conf: dict, images: list[dict], rows: list[dict]) -> list[dict]:
    by_key = {(r["image"], r["zone"]): r for r in rows}
    out = []
    for image in images:
        for p in image["probes"]:
            r = by_key.get((image["index"], p["slot"]))
            if r is None:
                continue
            color = image["slot_colors"][p["slot"]]
            out.append({"axis": p["axis"], "k": p["k"], "t": p["t"], "position": p["position"], "slot": p["slot"], "image": image["index"],
                        "color": "-".join(map(str, color)), "dist_to_inc": float(rgb_distance(color, conf["t_inc"])),
                        "dist_to_ctrl": float(rgb_distance(color, conf["t_ctrl"])), "b_entry": r["b_entry"],
                        "I_entry_exp": r["I_entry_exp"], "I_entry_ref": r["I_entry_ref"]})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=e4_setup.DEFAULT_CONFIG)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--checkpoint", type=int, required=True, help="environment step of the weights (agent_env<NNNNNN>.pt)")
    parser.add_argument("--experimental-run", type=Path, default=None)
    parser.add_argument("--reference-run", type=Path, default=None)
    parser.add_argument("--find-runs", type=Path, default=None, help="results root where to find both runs (instead of the two above)")
    parser.add_argument("--images", nargs="+", default=["all"], help="all, training, extreme, or image indices")
    parser.add_argument("--n-walks", type=int, default=None, help="the first N walks only (default: e4.walks.n_walks)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=e4_setup.REPO_ROOT / "results")
    parser.add_argument("--save-steps", action="store_true")
    args = parser.parse_args(argv)
    if args.find_runs is None and not (args.experimental_run and args.reference_run):
        parser.error("give --experimental-run and --reference-run, or --find-runs")
    cfg = e4_setup.load_config(args.config)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    conf = e4_setup.configuration(cfg, args.configuration)
    all_images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    if args.images == ["all"]:
        images = all_images
    elif args.images == ["training"]:
        images = [im for im in all_images if im["is_training"]]
    elif args.images == ["extreme"]:
        images = [im for im in all_images if is_extreme(im)]
    else:
        images = [all_images[int(i)] for i in args.images]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / e4_setup.EXPERIMENT_ID / f"probes__{args.configuration}__env{args.checkpoint:06d}__{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    status = out / "status.json"
    status.write_text(json.dumps({"status": "running", "started_at": stamp}), encoding="utf-8")
    try:
        if args.find_runs is not None:
            args.experimental_run = find_run(args.find_runs, args.configuration, "experimental", args.checkpoint)
            args.reference_run = find_run(args.find_runs, args.configuration, "reference", args.checkpoint)
        weights = {name: run / "checkpoints" / f"agent_env{args.checkpoint:06d}.pt"
                   for name, run in (("experimental", args.experimental_run), ("reference", args.reference_run))}
        models = {name: load_rssm(path, device) for name, path in weights.items()}
        print(f"[measure_probes] {args.configuration} at {args.checkpoint}: {len(images)} images; runs {args.experimental_run.name} "
              f"and {args.reference_run.name}", flush=True)
        t0 = time.perf_counter()
        result = measure(cfg, conf, models, images, device, save_steps=args.save_steps, n_walks=args.n_walks)
        seconds = time.perf_counter() - t0
    except BaseException as err:
        status.write_text(json.dumps({"status": "failed", "error": f"{type(err).__name__}: {err}",
                                      "traceback": traceback.format_exc()[-4000:]}, indent=1), encoding="utf-8")
        raise
    write_csv(out / "images.csv", result["rows"])
    write_csv(out / "probes.csv", probe_rows(conf, images, result["rows"]))
    np.savez_compressed(out / "curves.npz", curve_exp=result["curve_exp"], curve_ref=result["curve_ref"], n_states=result["n_states_curve"])
    if args.save_steps:
        np.savez_compressed(out / "steps.npz", **result["steps"])
    meta = {"configuration": args.configuration, "checkpoint_env_step": args.checkpoint, "device": str(device),
            "torch_threads": torch.get_num_threads(), "runs": {k: str(v) for k, v in (("experimental", args.experimental_run),
                                                                                   ("reference", args.reference_run))},
            "weights_sha256": {k: sha256(p) for k, p in weights.items()}, "n_images": len(images),
            "n_image_zones": len(result["rows"]), "seconds": seconds, "walks": cfg["e4"]["walks"], "probe": cfg["e4"]["probe"],
            "kernel_margin_threshold": cfg["measure"]["kernel_margin_threshold"], "t_inc": list(conf["t_inc"]), "t_ctrl": list(conf["t_ctrl"])}
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    status.write_text(json.dumps({"status": "done", "seconds": seconds, "n_image_zones": len(result["rows"])}), encoding="utf-8")
    print(f"[measure_probes] {len(result['rows'])} (image, zone) measured in {seconds:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
