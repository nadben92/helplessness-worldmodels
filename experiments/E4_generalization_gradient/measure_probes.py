"""E4's zero-data probes of one training configuration at one checkpoint (docs/E4_generalization_gradient.md, the
measure): the 93 distinct probe images (helplessness.probing.theme_probes) and, for each image and each of its zones of
interest, the fixed walks into that slot (the same for every image and both models), filtered by the experimental
(exposed) model and by the reference model, frozen at the checkpoint (their weights checkpoints/agent_env<NNNNNN>.pt);
I_s at every step in the zone only (helplessness.probing.fixed_trajectory.measure_walks: the corridor steps are
filtered, never decoded; the same measurement seed for the two models). b = I_experimental - I_reference. The extreme
probes (the training image, theme only = T_inc at the neutral slot, position only) use the walks' full n_in steps in the
zone (e4.walks.n_in, for the evolution in the zone); every other image the same walks cut after n_in_grid steps.

The main measure is b at the FIRST step in the zone (b_step1 = curves.npz's curve_exp[:, 0] - curve_ref[:, 0], over the
zone's 16 border states), the belief at the first encounter; b over steps 2..n_in is the recovery measure. Every probe
environment is fully controllable (the reference's dynamics in every slot; the walks follow them), so from step 2 on
the model has already seen, inside the zone, one transition where its action worked. pilot_analysis.py reads the
main measure from curves.npz; b_entry below (the mean over the first m steps, the first plan's main measure) stays
written as it was.

Outputs, in a new directory <out-dir>/E4_generalization_gradient/probes__<configuration>__env<NNNNNN>__<UTC>/:
  images.csv     one row per (image, zone of interest): the slot colours, the role of the zone, I at entry for both
                 models and b at entry (mean over the first m steps in the zone, by state then by zone), the states
                 covered, the mean ghost fraction, the decode count, the seconds taken;
  probes.csv     one row per probe (axis, k, t, position): its image, its zone, b at entry; the distance of the probe's
                 colour to T_inc and to T_ctrl (RGB);
  curves.npz     b and I by step since entry (1..n_in) for every (image, zone): arrays curve_exp, curve_ref (rows = rows of
                 images.csv; column 0 = the first step in the zone, the main measure), n_states (distinct states behind
                 each step, the smaller of the two models');
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
from helplessness.pipeline.runtime import configure_runtime
from helplessness.probing.theme_probe_measure import (  # noqa: F401  (moved to the library on 2026-10-19; re-exported)
    is_extreme,
    load_rssm,
    measure,
    probe_rows,
    sha256,
    write_csv,
)
from helplessness.probing.theme_probes import probe_images


def find_run(root: Path, config_id: str, condition: str, checkpoint: int) -> Path:
    role = "experimental" if condition == "experimental" else "control"
    found = sorted(d for d in (root / e4_setup.EXPERIMENT_ID).glob(f"*__2zones-{config_id}__pair-{role}__*")
                   if (d / "checkpoints" / f"agent_env{checkpoint:06d}.pt").exists())
    if not found:
        raise FileNotFoundError(f"no {condition} run of {config_id} with weights at {checkpoint} under {root}")
    return found[-1]


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
