"""Step 1's consistency check (README.md): a model's localisation on the swap images, recomputed locally with the code of
the swap probes -- the images of helplessness.probing.swap_probes, the probe world, the fixed walks and the measurement
seed of measure_probes (two_zones_setup), and helplessness.probing.probe_validity.reconstruction_diagnostic, called as
measure_probes.validity_hook calls it (the pair's seed, the run's config) -- then compared with the cluster's
reconstruction.csv. With --n-walks 150 the walks, chunks and generator draws are the cluster's own; with fewer walks,
the first ones (the posterior draws then differ: a sampling check, not an identity).

One output row per (image, zone, part): agent_cell_accuracy and n_frames, local and cluster, and their difference.

Usage, from the repo root:
  python experiments/E5_belief_training/analysis_robustness/local_localisation.py --model F --runs-root <root> [...]
      --probes-root <extracted>/results/E5_belief_training --n-walks 100 --out <csv> [--images 0 5 17] [--threads 6]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

import common
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
from helplessness.pipeline import two_zones_setup
from helplessness.probing import probe_validity
from helplessness.probing.theme_probe_measure import load_rssm


def localise(model_path: Path, config: Path, pair_seed: int, n_walks: int, images_sel: list[int] | None, threads: int,
             log=print) -> list[dict]:
    """reconstruction_diagnostic of one model on the swap images (all, or the positions images_sel), every zone."""
    cfg, conf, images = common.setup(str(config))
    conf = {**conf, "seed": pair_seed}
    torch.set_num_threads(threads)
    model = load_rssm(model_path, torch.device("cpu"))
    references = {"canonical_training": None, "T_inc": conf["t_inc"], "T_ctrl": conf["t_ctrl"], "corridor": tuple(int(v) for v in CORRIDOR_COLOR)}
    rows = []
    for image in (images if images_sel is None else [images[i] for i in images_sel]):
        world = two_zones_setup.probe_world(cfg, conf, image["slot_colors"])
        n_in = cfg["e4"]["walks"].get("n_in_grid", cfg["e4"]["walks"]["n_in"])  # a swap image is never an extreme probe
        t0 = time.perf_counter()
        for slot in image["zones"]:
            walks = two_zones_setup.walks_for_slot(cfg, slot, n_walks, n_in)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(two_zones_setup.measurement_seed(cfg, conf, image["index"], slot, "probe"))
            colours = {**references, "canonical_training": conf["training_colors"][slot]}
            for part in probe_validity.reconstruction_diagnostic(model, world, walks, slot, image["slot_colors"][slot], colours,
                                                                 conf["agent_color"], gen):
                rows.append({"image": image["index"], "zone": slot, "part": part["part"], "n_frames": part["n_frames"],
                             "agent_cell_accuracy": part["agent_cell_accuracy"]})
        log(f"image {image['index']}: {time.perf_counter() - t0:.1f} s")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="an id of models.yaml")
    parser.add_argument("--runs-root", type=Path, nargs="+", required=True)
    parser.add_argument("--probes-root", type=Path, nargs="+", required=True)
    parser.add_argument("--n-walks", type=int, default=100)
    parser.add_argument("--images", type=int, nargs="+", default=None, help="positions in the swap set (default: all)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    model = next(m for m in common.load_models() if m["id"] == args.model)
    weights = common.weights_of(model, args.runs_root)
    pdir = common.probes_dir_of(model, args.probes_root)
    if weights is None or pdir is None:
        raise SystemExit(f"{args.model}: weights {weights}, probes {pdir}")
    meta = common.probe_meta(pdir)
    if common.sha256(weights) != meta["weights_sha256"][model["probes"]["model"]]:
        raise SystemExit(f"{args.model}: the local weights are not the probed ones")
    t0 = time.perf_counter()
    rows = localise(weights, weights.parent.parent / "config.yaml", int(meta["seed"]), args.n_walks, args.images, args.threads)
    seconds = time.perf_counter() - t0
    cluster = common.read_reconstruction(pdir, model["probes"]["model"])
    for r in rows:
        c = cluster[(r["image"], r["zone"], r["part"])]
        r.update({"id": args.model, "n_walks": args.n_walks, "cluster_n_frames": int(c["n_frames"]),
                  "cluster_agent_cell_accuracy": float(c["agent_cell_accuracy"]),
                  "diff": r["agent_cell_accuracy"] - float(c["agent_cell_accuracy"])})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    step1 = [r for r in rows if r["part"] == "zone_step1"]
    worst = max(step1, key=lambda r: abs(r["diff"]))
    print(f"[{args.model}] {len(step1)} (image, zone) in {seconds:.0f} s; zone_step1 max |diff| {abs(worst['diff']):.4f} "
          f"(image {worst['image']} zone {worst['zone']}: local {worst['agent_cell_accuracy']:.3f}, cluster {worst['cluster_agent_cell_accuracy']:.3f}); "
          f"> 0.05: {sum(abs(r['diff']) > 0.05 for r in step1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
