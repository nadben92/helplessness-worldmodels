"""The exposed model's localisation at step 1 on every probe (LOCALISATION_CRITERION.md), read-only, frozen weights.

For one probe measurement (a measure_probes.py output directory) and the exposed run it measured: every (image, slot) of
probes.csv, rebuilt as theme_probe_measure.measure builds it (the probe world, the first n_walks of E4's fixed walks,
n_in by image kind), and probe_validity.reconstruction_diagnostic on the exposed model with measure_probes.py's
diagnostic generator. Writes one row per (image, slot): the localisation at step 1 by the direct rule (agent cell) and
by the kernel rule (nearest template), and the frames counted. The weights' SHA-256 must equal the probe's meta.yaml.

Usage: python experiments/E5_belief_training/analysis_450k/localisation_probes.py --probe-dir <dir> --run-dir <exposed run>
           --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml

from helplessness.pipeline import two_zones_setup as e4_setup
from helplessness.probing import probe_validity
from helplessness.probing.theme_probe_measure import is_extreme, load_rssm, sha256
from helplessness.probing.theme_probes import probe_images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="the exposed run (config.yaml, checkpoints/)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    meta = yaml.safe_load((args.probe_dir / "meta.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    weights = args.run_dir / "checkpoints" / f"agent_env{int(meta['checkpoint_k']):06d}.pt"
    before = sha256(weights)
    if before != meta["weights_sha256"]["experimental"]:
        raise SystemExit(f"{weights}: SHA-256 {before} differs from the probe's {meta['weights_sha256']['experimental']}")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = load_rssm(weights, device)
    conf = {**e4_setup.configuration(cfg, meta["configuration"]), "seed": int(meta["seed"])}
    images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    with open(args.probe_dir / "probes.csv", newline="", encoding="utf-8") as fh:
        wanted = sorted({(int(r["image"]), r["slot"]) for r in csv.DictReader(fh)})
    w, n_walks = cfg["e4"]["walks"], int(meta["walks"]["n_walks_used"])
    colours = {"T_inc": conf["t_inc"], "T_ctrl": conf["t_ctrl"]}
    rows, t0 = [], time.perf_counter()
    for index, slot in wanted:
        image = images[index]
        assert image["index"] == index
        world = e4_setup.probe_world(cfg, conf, image["slot_colors"])
        n_in = w["n_in"] if is_extreme(image) else w.get("n_in_grid", w["n_in"])
        walks = e4_setup.walks_for_slot(cfg, slot, n_walks, n_in)
        gen = torch.Generator(device=device)
        gen.manual_seed(e4_setup.measurement_seed(cfg, conf, index, slot, "probe"))
        parts = {p["part"]: p for p in probe_validity.reconstruction_diagnostic(model, world, walks, slot, image["slot_colors"][slot],
                                                                                 colours, conf["agent_color"], gen)}
        s1 = parts["zone_step1"]
        rows.append({"image": index, "slot": slot, "n_frames_step1": s1["n_frames"], "loc_agent_cell_step1": s1["agent_cell_accuracy"],
                     "loc_nearest_template_step1": s1["nearest_template_accuracy"], "loc_agent_cell_zone": parts["zone"]["agent_cell_accuracy"],
                     "recon_mse_step1": s1["recon_mse"]})
        print(f"image {index:3d} {slot}: loc step 1 {s1['agent_cell_accuracy']:.3f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    if sha256(weights) != before:
        raise SystemExit(f"{weights} changed during the run")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} probes -> {args.out}; weights {before[:12]} unchanged", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
