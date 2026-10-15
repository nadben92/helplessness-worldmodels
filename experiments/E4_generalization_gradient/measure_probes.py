"""E4's zero-data probes of one training configuration at one checkpoint (docs/E4_generalization_gradient.md, the
measure): the 93 distinct probe images (helplessness.probing.theme_probes) and, for each image and each of its zones of
interest, the fixed walks into that slot (the same for every image and both models), filtered by the experimental
(exposed) model and by the reference model, frozen at the checkpoint (their weights checkpoints/agent_env<NNNNNN>.pt);
I_s at every step in the zone (helplessness.probing.fixed_trajectory.measure_walks, the same measurement seed for the
two models). b = I_experimental - I_reference.

Outputs, in a new directory <out-dir>/E4_generalization_gradient/probes__<configuration>__env<NNNNNN>__<UTC>/:
  images.csv     one row per (image, zone of interest): the slot colours, the role of the zone, I at entry for both
                 models and b at entry (mean over the first m steps in the zone, by state then by zone), the states
                 covered, the mean ghost fraction, the decode count, the seconds taken;
  probes.csv     one row per probe (axis, k, t, position): its image, its zone, b at entry; the distance of the probe's
                 colour to T_inc and to T_ctrl (RGB);
  curves.npz     b and I by step since entry (1..n_in) for every (image, zone): arrays curve_exp, curve_ref (rows = rows of
                 images.csv), n_states;
  steps.npz      every measured step (optional, --save-steps): I, state, steps since entry / start, walk, both models;
  meta.yaml      the configuration, the two runs and their weights' SHA-256, the checkpoint, the settings, the timing.

Usage, from the repo root (CPU or GPU):
  python experiments/E4_generalization_gradient/measure_probes.py --configuration p1-a-d1 --checkpoint 300000 \\
      --experimental-run results/E4_generalization_gradient/<exp run> --reference-run results/E4_generalization_gradient/<ref run>
      [--images all | training | 0 5 17] [--device cuda] [--out-dir results] [--save-steps]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
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


def measure(cfg: dict, conf: dict, models: dict, images: list[dict], device, log=print, save_steps: bool = False) -> dict:
    """Every (image, zone) of `images` on both models. Returns rows (images.csv), curves and the optional steps."""
    w, draws = cfg["e4"]["walks"], cfg["e4"]["probe"]["draws"]
    margin = cfg["measure"]["kernel_margin_threshold"]
    role_of = {slot: role for role, slot in conf["roles"].items()}
    rows, curves_exp, curves_ref, n_states_curve, steps = [], [], [], [], {}
    for image in images:
        world = e4_setup.probe_world(cfg, conf, image["slot_colors"])
        succ = e4_setup.successors(world)
        for slot in image["zones"]:
            walks = e4_setup.walks_for_slot(cfg, slot)
            seed = e4_setup.measurement_seed(cfg, conf, image["index"], slot, "probe")
            t0 = time.perf_counter()
            res = {name: measure_walks(model, world.templates_np, world.templates_t, walks, succ, draws, margin, seed, w["m"], w["n_in"],
                                       chunk=w["chunk"], max_images=max_images_per_forward(cfg))
                   for name, model in models.items()}
            seconds = time.perf_counter() - t0
            e, r = res["experimental"], res["reference"]
            rows.append({"image": image["index"], "is_training": image["is_training"], "zone": slot, "role": role_of[slot],
                         **{f"color_{q}": "-".join(map(str, image["slot_colors"][q])) for q in QUADRANT_ORDER},
                         "I_entry_exp": e["entry"], "I_entry_ref": r["entry"], "b_entry": e["entry"] - r["entry"],
                         "n_states_entry_exp": e["n_states_entry"], "n_states_entry_ref": r["n_states_entry"],
                         "ghost_exp": float(np.nanmean(e["ghost_fraction"])), "ghost_ref": float(np.nanmean(r["ghost_fraction"])),
                         "mass_true_exp": float(np.nanmean(e["mass_true"])), "mass_true_ref": float(np.nanmean(r["mass_true"])),
                         "decodes_exp": e["n_unique_decodes"], "decodes_ref": r["n_unique_decodes"], "seconds": seconds,
                         "n_probes": len(image["probes"])})
            curves_exp.append(e["curve"])
            curves_ref.append(r["curve"])
            n_states_curve.append(np.minimum(e["n_states_curve"], r["n_states_curve"]))
            if save_steps:
                key = f"i{image['index']:03d}_{slot}"
                for name, res_m in res.items():
                    for field in ("I", "states", "steps_since_entry", "steps_since_start", "walk"):
                        steps[f"{key}_{name}_{field}"] = res_m[field]
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
    parser.add_argument("--experimental-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--images", nargs="+", default=["all"], help="all, training, or image indices")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=e4_setup.REPO_ROOT / "results")
    parser.add_argument("--save-steps", action="store_true")
    args = parser.parse_args(argv)
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
    else:
        images = [all_images[int(i)] for i in args.images]
    weights = {name: run / "checkpoints" / f"agent_env{args.checkpoint:06d}.pt"
               for name, run in (("experimental", args.experimental_run), ("reference", args.reference_run))}
    models = {name: load_rssm(path, device) for name, path in weights.items()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / e4_setup.EXPERIMENT_ID / f"probes__{args.configuration}__env{args.checkpoint:06d}__{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    result = measure(cfg, conf, models, images, device, save_steps=args.save_steps)
    seconds = time.perf_counter() - t0
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
    print(f"[measure_probes] {len(result['rows'])} (image, zone) measured in {seconds:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
