"""The test images of E6/E7 on all agents (docs/e6/DECISIONS.md (63)), chosen on the probes' results before any run.

Per agent: the training image, plus VALID probe images (E4's criterion on the reference) of the agent's own probe
measurement -- E4's pairs: E4's final table (probes_b_step1.csv, 300 walks); E5's: E5's analysis tables at the probed
checkpoint (gradient_probes.csv and, for position only, extremes.csv; 150 walks). Each (image, slot) once; the training
image (index 0) left out of the pool (it is in every list). Classes of b at step 1: helpless b <= -1.5, intermediate
-1.5 < b <= -0.5, in control b > -0.5. Picked per class (CLASS_QUOTA), spread over the class: sorted by b (ties: image,
slot), the probes at ranks round(q (n - 1)) for q evenly spaced in [0, 1] (two picks: the class's extremes); a class with
fewer probes than its quota gives all of them.

Writes configs/e6/e6_all.yaml (E4's base; the e6 block of configs/e6/e6.yaml unchanged but `images`: the union) and
configs/e6/images_by_agent_all.yaml (agent -> image names), and prints the selection table.

Usage: python experiments/E6_phenotypes_latent/select_images.py --agents configs/e6/agents_e5.yaml --e4-table <probes_b_step1.csv>
           --e5-tables <dir with gradient_probes.csv and extremes.csv> --out-csv <csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

from helplessness.config import REPO_ROOT
from helplessness.pipeline import two_zones_setup as setup
from helplessness.probing.theme_probes import probe_images

CLASSES = (("helpless", -np.inf, -1.5), ("intermediate", -1.5, -0.5), ("in_control", -0.5, np.inf))
CLASS_QUOTA = {"f1": {"helpless": 2, "intermediate": 99, "in_control": 2},        # shuffle: every intermediate probe
               "baseline": {"helpless": 2, "intermediate": 2, "in_control": 2}}
AXIS_ORDER = ("segment", "arc_ctrl", "arc_inc", "position_only")
E5_LABEL = {102: "baseline_B_k300000", 402: "shuffle_E_k475000", 502: "shuffle_F_k475000"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def class_of(b: float) -> str:
    return next(name for name, lo, hi in CLASSES if lo < b <= hi)


def spec_of(images: list[dict], index: int, position: str) -> tuple[str, int]:
    """The (axis, k) naming the probe image `index` at `position` (the first axis in AXIS_ORDER)."""
    keys = [(p["axis"], p["k"]) for p in images[index]["probes"] if p["position"] == position]
    return min(keys, key=lambda ak: (AXIS_ORDER.index(ak[0]), ak[1]))


def pool_of(agent: dict, e4_rows: list[dict], e5_dir: Path) -> list[dict]:
    """Valid probes of the agent's own measurement, (image, slot) once, the training image left out."""
    if agent["origin"] == "e4":
        rows = [{"image": int(r["image"]), "slot": r["slot"], "position": r["position"], "b": float(r["b_step1"]),
                 "valid": r["valid"] == "True"} for r in e4_rows if r["configuration"] == agent["configuration"]]
    else:
        label = E5_LABEL[agent["seed"]]
        rows = [{"image": int(r["image"]), "slot": r["slot"], "position": r["position"], "b": float(r["b_step1"]),
                 "valid": r["valid"] == "True"} for r in read_csv(e5_dir / "gradient_probes.csv") if r["label"] == label]
        rows += [{"image": 92, "slot": r["slot"], "position": "ud", "b": float(r["b_step1"]), "valid": r["valid"] == "True"}
                 for r in read_csv(e5_dir / "extremes.csv") if r["label"] == label and r["probe"] == "position_only"]
    seen = {}
    for r in rows:
        if r["valid"] and r["image"] != 0:
            seen.setdefault((r["image"], r["slot"]), r)
    return list(seen.values())


def pick(pool: list[dict], quota: dict) -> list[dict]:
    out = []
    for name, _, _ in CLASSES:
        members = sorted((r for r in pool if class_of(r["b"]) == name), key=lambda r: (r["b"], r["image"], r["slot"]))
        n = min(quota[name], len(members))
        ranks = sorted({int(round(q * (len(members) - 1))) for q in np.linspace(0, 1, n)}) if n else []
        if len(ranks) < n:  # rounding collided (n close to the class size): take them all
            ranks = list(range(len(members)))[:n]
        out += [{**members[i], "class": name} for i in ranks]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--e4-table", type=Path, required=True)
    parser.add_argument("--e5-tables", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    agents = yaml.safe_load(args.agents.read_text(encoding="utf-8"))["agents"]
    base_cfg = setup.load_config(REPO_ROOT / "configs" / "e4" / "e4.yaml")
    e4_rows = read_csv(args.e4_table)
    e6_raw = yaml.safe_load((REPO_ROOT / "configs" / "e6" / "e6.yaml").read_text(encoding="utf-8"))
    names, specs, table = {}, {"training": {"name": "training", "axis": "training", "position": "ud"}}, []
    for agent in agents:
        conf = setup.configuration(base_cfg, agent["configuration"])
        images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
        chosen = pick(pool_of(agent, e4_rows, args.e5_tables), CLASS_QUOTA[agent["condition"]])
        names[agent["id"]] = ["training"]
        for r in chosen:
            axis, k = spec_of(images, r["image"], r["position"])
            name = f"{axis}_k{k}_{r['position']}"
            specs.setdefault(name, {"name": name, "axis": axis, "k": int(k), "position": r["position"]})
            names[agent["id"]].append(name)
            table.append({"agent": agent["id"], "name": name, "image": r["image"], "slot": r["slot"], "position": r["position"],
                          "b_probe": round(r["b"], 4), "class": r["class"]})
    e6 = {**e6_raw["e6"], "images": list(specs.values())}
    header = ("# E6/E7 on all agents (docs/e6/DECISIONS.md (63)): configs/e6/e6.yaml's e6 block UNCHANGED but `images`, the union of\n"
              "# every agent's test images (written by experiments/E6_phenotypes_latent/select_images.py; each agent runs its own\n"
              "# subset, configs/e6/images_by_agent_all.yaml). tests/e6/test_e6_all_config.py pins the equality of the rest.\n")
    (REPO_ROOT / "configs" / "e6" / "e6_all.yaml").write_text(header + yaml.safe_dump({"base": e6_raw["base"], "e6": e6}, sort_keys=False),
                                                              encoding="utf-8")
    (REPO_ROOT / "configs" / "e6" / "images_by_agent_all.yaml").write_text(
        "# The test images of every agent (select_images.py, docs/e6/DECISIONS.md (63)); names of configs/e6/e6_all.yaml.\n"
        + yaml.safe_dump(names, sort_keys=False), encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for a, ns in names.items():
        print(f"{a}: {len(ns)} images ({', '.join(ns)})")
    print(f"{len(specs)} distinct test images, {sum(len(v) for v in names.values())} (agent, image) pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
