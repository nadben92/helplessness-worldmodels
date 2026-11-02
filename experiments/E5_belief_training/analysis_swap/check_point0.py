"""The swap probes at t = 0 against the training records (docs/e5/DECISIONS.md (38), tests before launch): the base image
of each swap IS a shuffle training assignment (4, 11, 6 for p1-b-d1), whose four slots the run's checkpoint record
measured with the same function (pipeline.assignment_records.walks_by_assignment: the probe world of the assignment,
E4's fixed walks, measure_walks). Read-only, frozen weights, CPU.

  record  each model measured as its own record measured it: its run's seed, image index ASSIGNMENT_IMAGE_BASE + k,
          purpose "checkpoint", e4.checkpoint.n_walks walks, n_in = e4.walks.n_in. I at step 1 must equal the record's
          I_curve[0] up to the device (the records were measured on the GPU).
  probe   the swap set's own protocol (measure_probes.py --probe-set swap: the exposed run's seed for both models, image
          index SWAP_IMAGE_BASE + i, purpose "probe", e5.probes.n_walks walks, n_in_grid): b at step 1 must agree with the
          record's b within sampling error (other walks, other draws).

One row per (swap, slot): I and b of both modes beside the records'. Usage, from the repo root:
  python experiments/E5_belief_training/analysis_swap/check_point0.py --experimental-run <dir> --reference-run <dir>
      --checkpoint 650000 --out <csv> [--modes record probe]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # E5's own directory (e5_setup)

import e5_setup  # noqa: E402
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER  # noqa: E402
from helplessness.pipeline import two_zones_setup  # noqa: E402
from helplessness.pipeline.assignment_records import ASSIGNMENT_IMAGE_BASE  # noqa: E402
from helplessness.pipeline.episode_sources import assignment_index  # noqa: E402
from helplessness.pipeline.runtime import configure_runtime, max_images_per_forward  # noqa: E402
from helplessness.probing import swap_probes  # noqa: E402
from helplessness.probing.fixed_trajectory import measure_walks  # noqa: E402
from helplessness.probing.theme_probe_measure import load_rssm, write_csv  # noqa: E402


def record_at(run_dir: Path, k_step: int) -> dict:
    for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(r.get("env_step", -1)) == k_step and r.get("e5"):
                return r
    raise SystemExit(f"{run_dir}: no E5 record at {k_step}")


def step1(cfg, conf_seed, conf, model, world, slot, image_index, purpose, n_walks, n_in) -> float:
    w = cfg["e4"]["walks"]
    seed = two_zones_setup.measurement_seed(cfg, {**conf, "seed": conf_seed}, image_index, slot, purpose)
    res = measure_walks(model, world.templates_np, world.templates_t, two_zones_setup.walks_for_slot(cfg, slot, n_walks, n_in),
                        two_zones_setup.successors(world), cfg["e4"]["probe"]["draws"], cfg["measure"]["kernel_margin_threshold"], seed,
                        w["m"], n_in, chunk=w["chunk"], max_images=max_images_per_forward(cfg))
    return float(res["curve"][0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experimental-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--modes", nargs="+", choices=("record", "probe"), default=["record", "probe"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {"experimental": args.experimental_run, "reference": args.reference_run}
    cfgs = {name: yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) for name, d in runs.items()}
    seeds = {name: int(yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))["seed"]) for name, d in runs.items()}
    cfg = cfgs["experimental"]
    cfg["runtime"]["device"] = "cpu"
    device = configure_runtime(cfg)
    configuration = cfg["e5"]["run"]["configuration"]
    conf = two_zones_setup.configuration(cfg, configuration)
    records = {name: record_at(d, args.checkpoint) for name, d in runs.items()}
    models = {name: load_rssm(d / "checkpoints" / f"agent_env{args.checkpoint:06d}.pt", device) for name, d in runs.items()}
    images = swap_probes.swap_probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    bases = [im for im in images if any(p["axis"] == "segment" and p["k"] == 0 for p in im["probes"])]
    w = cfg["e4"]["walks"]
    n_probe = int(e5_setup.load_config(e5_setup.DEFAULT_CONFIG)["e5"]["probes"]["n_walks"])
    rows = []
    for image in bases:
        k = assignment_index(image["ud_slot"], image["cd_slot"])
        world = two_zones_setup.probe_world(cfg, conf, image["slot_colors"])
        for slot in QUADRANT_ORDER:
            t0 = time.perf_counter()
            rec = {name: float(records[name]["e5"]["by_assignment"][str(k)]["walks"][slot]["I_curve"][0]) for name in runs}
            row = {"swap": image["swap"], "assignment": k, "slot": slot, "zone_role": swap_probes.zone_role(image, slot),
                   "I_exp_record": rec["experimental"], "I_ref_record": rec["reference"], "b_record": rec["experimental"] - rec["reference"]}
            if "record" in args.modes:
                got = {name: step1(cfg, seeds[name], conf, models[name], world, slot, ASSIGNMENT_IMAGE_BASE + k, "checkpoint",
                                   int(cfg["e4"]["checkpoint"]["n_walks"]), w["n_in"]) for name in runs}
                row.update({"I_exp_replay": got["experimental"], "I_ref_replay": got["reference"],
                            "diff_I_exp_replay": got["experimental"] - rec["experimental"], "diff_I_ref_replay": got["reference"] - rec["reference"]})
            if "probe" in args.modes:
                got = {name: step1(cfg, seeds["experimental"], conf, models[name], world, slot, image["index"], "probe", n_probe,
                                   w.get("n_in_grid", w["n_in"])) for name in runs}
                row.update({"b_probe": got["experimental"] - got["reference"],
                            "diff_b_probe": got["experimental"] - got["reference"] - row["b_record"]})
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
            print({key: (round(v, 4) if isinstance(v, float) else v) for key, v in row.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
