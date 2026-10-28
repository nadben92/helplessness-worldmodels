"""E5's probes across seeds (docs/e5/DECISIONS.md (31), (33)): every measure_probes.py output of a condition (baseline,
shuffle f = 1, ...), one per seed -- as many seeds as were measured -- aggregated probe by probe, EACH PAIR AT ITS OWN
CHECKPOINT (a shuffle pair at its first checkpoint meeting E4's criterion, K_a or K_b; the baselines at 300,000): the two
probe sets (e5_probes_a, e5_probes_b) merge into one table; a pair measured at several checkpoints needs --pair-k.

Per probe (axis, k, position): b at the first step in the zone (the main measure: curves.npz, experimental minus
reference at step 1) for each seed whose probe is VALID (E4's criterion on the reference, docs/e5/DECISIONS.md (22)),
then their mean, standard deviation and number, and the number of seeds measured. Invalid probes are left out, never
imputed. The extreme probes are also tabulated: the training image at the canonical UD slot, theme only (T_inc at the
neutral slot, segment k = 0) and position only (the corridor's colour at the UD slot).

Matched comparisons (docs/e5/DECISIONS.md (36)), --matched <condition>:<seed>@<k>=<condition>:<seed>@<k> ...: a
measurement against another one named explicitly (a shuffle pair against its baseline at the checkpoint matched to its
level of learning, matched_checkpoint.py), taken among every measurement found (not only the one per pair kept above).
Per probe and comparison: both b, their difference (first minus second) when the probe is valid in both, else left out;
then per probe over the comparisons: the mean, standard deviation and number of differences. Outputs matched_by_probe.csv
and matched_aggregated.csv; a measurement named and not found is an error.

Inputs: probe directories (probes__<configuration>__<tag>__seed<NN>__k<NNNNNN>__<UTC>), given or found under --root;
the newest per (condition, seed, k) is kept. Outputs in --out: probes_by_seed.csv (one row per probe and seed),
probes_aggregated.csv (one row per condition, k and probe), extremes.csv, summary.json (the seeds of every condition).

Usage: python experiments/E5_belief_training/analyse_probes.py --root <results root> [--checkpoint k] [--pair-k ...] [--matched ...] --out <dir>
       python experiments/E5_belief_training/analyse_probes.py <probe dir> ... --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

EXTREMES = {"theme_only": ("segment", 0, "neutral"), "position_only": ("position_only", 0, "ud")}


def condition_of(meta: dict) -> str:
    if meta["training"] == "shuffle":
        return f"shuffle_f{float(meta['fraction']):g}"
    return meta["training"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def load_probe_dir(d: Path) -> dict:
    """One measurement: its condition, seed, k, and b at step 1 with the validity of every probe and image zone."""
    meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
    images = read_csv(d / "images.csv")
    curves = np.load(d / "curves.npz")
    b1 = curves["curve_exp"][:, 0] - curves["curve_ref"][:, 0]
    b1_of = {(int(r["image"]), r["zone"]): float(b1[i]) for i, r in enumerate(images)}
    probes = []
    for p in read_csv(d / "probes.csv"):
        key = (int(p["image"]), p["slot"])
        probes.append({"axis": p["axis"], "k": int(p["k"]), "position": p["position"], "image": key[0], "slot": key[1],
                       "b_step1": b1_of[key], "valid": truthy(p.get("valid", "True"))})
    training_ud = [{"b_step1": b1_of[(int(r["image"]), r["zone"])], "role": r["role"]} for r in images if truthy(r["is_training"])]
    return {"dir": d, "condition": condition_of(meta), "seed": int(meta["seed"]), "k": int(meta["checkpoint_k"]),
            "probes": probes, "training": training_ud}


def find_probe_dirs(root: Path) -> list[Path]:
    return sorted(d for d in (Path(root) / "E5_belief_training").glob("probes__*") if (d / "meta.yaml").is_file()
                  and (d / "curves.npz").is_file() and json.loads((d / "status.json").read_text(encoding="utf-8")).get("status") == "done")


def newest_per_seed(measures: list[dict]) -> list[dict]:
    keep: dict = {}
    for m in sorted(measures, key=lambda m: m["dir"].name):  # the UTC stamp ends the name: the newest last
        keep[(m["condition"], m["seed"], m["k"])] = m
    return list(keep.values())


def one_k_per_pair(measures: list[dict], pair_k: dict[tuple[str, int], int]) -> list[dict]:
    """One measurement per (condition, seed): each pair at ITS checkpoint (docs/e5/DECISIONS.md (33): a shuffle pair at
    its first checkpoint meeting E4's criterion, the baselines at 300,000). A pair measured at several checkpoints needs
    --pair-k <condition>:<seed>=<k>."""
    by_pair: dict = {}
    for m in measures:
        by_pair.setdefault((m["condition"], m["seed"]), []).append(m)
    out = []
    for (cond, seed), ms in sorted(by_pair.items()):
        if (cond, seed) in pair_k:
            ms = [m for m in ms if m["k"] == pair_k[(cond, seed)]]
            if not ms:
                raise SystemExit(f"{cond} seed {seed}: no measurement at k = {pair_k[(cond, seed)]}")
        if len(ms) > 1:
            raise SystemExit(f"{cond} seed {seed} measured at several checkpoints {sorted(m['k'] for m in ms)}: give --pair-k {cond}:{seed}=<k>")
        out.append(ms[0])
    return out


def aggregate(measures: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    by_seed, groups, extremes, seeds = [], {}, [], {}
    for m in measures:
        seeds.setdefault(m["condition"], []).append([m["seed"], m["k"]])
        for p in m["probes"]:
            by_seed.append({"condition": m["condition"], "k_checkpoint": m["k"], "seed": m["seed"], **{x: p[x] for x in ("axis", "k", "position", "b_step1", "valid")}})
            groups.setdefault((m["condition"], p["axis"], p["k"], p["position"]), []).append((m["seed"], p["b_step1"], p["valid"]))
        ud = [t["b_step1"] for t in m["training"] if t["role"] == "ud"]
        extremes.append({"condition": m["condition"], "k_checkpoint": m["k"], "seed": m["seed"], "probe": "training_image_ud",
                         "b_step1": ud[0] if ud else float("nan"), "valid": None})
        for name, (axis, kk, pos) in EXTREMES.items():
            for p in m["probes"]:
                if (p["axis"], p["k"], p["position"]) == (axis, kk, pos):
                    extremes.append({"condition": m["condition"], "k_checkpoint": m["k"], "seed": m["seed"], "probe": name,
                                     "b_step1": p["b_step1"], "valid": p["valid"]})
    aggregated = []
    for (cond, axis, kk, pos), vals in sorted(groups.items()):
        valid = [b for _, b, ok in vals if ok]
        aggregated.append({"condition": cond, "axis": axis, "k": kk, "position": pos, "n_seeds": len(vals),
                           "n_valid": len(valid), "b_step1_mean": float(np.mean(valid)) if valid else float("nan"),
                           "b_step1_sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan"),
                           "seeds_valid": " ".join(str(s) for s, _, ok in vals if ok)})
    return by_seed, aggregated, extremes, {k: sorted(v) for k, v in seeds.items()}


def parse_measure(text: str) -> tuple[str, int, int]:
    """<condition>:<seed>@<k> -> (condition, seed, k)."""
    cond_seed, k = text.split("@")
    cond, seed = cond_seed.rsplit(":", 1)
    return cond, int(seed), int(k)


def matched(measures: list[dict], comparisons: list[str]) -> tuple[list[dict], list[dict]]:
    """--matched <a>=<b>: per probe, b(a) - b(b) where the probe is valid in both measurements; then per probe over the
    comparisons (docs/e5/DECISIONS.md (36))."""
    index = {(m["condition"], m["seed"], m["k"]): m for m in measures}
    by_probe, groups = [], {}
    for item in comparisons:
        a_key, b_key = (parse_measure(x) for x in item.split("="))
        for key in (a_key, b_key):
            if key not in index:
                raise SystemExit(f"--matched {item}: no measurement {key[0]}:{key[1]}@{key[2]}")
        a, b = index[a_key], index[b_key]
        b_of = {(p["axis"], p["k"], p["position"]): p for p in b["probes"]}
        for p in a["probes"]:
            probe = (p["axis"], p["k"], p["position"])
            q = b_of[probe]
            ok = p["valid"] and q["valid"]
            diff = p["b_step1"] - q["b_step1"] if ok else float("nan")
            by_probe.append({"comparison": item, "axis": probe[0], "k": probe[1], "position": probe[2], "b_first": p["b_step1"],
                             "valid_first": p["valid"], "b_second": q["b_step1"], "valid_second": q["valid"], "difference": diff})
            groups.setdefault(probe, []).append((item, diff, ok))
    aggregated = []
    for (axis, kk, pos), vals in sorted(groups.items()):
        d = [x for _, x, ok in vals if ok]
        aggregated.append({"axis": axis, "k": kk, "position": pos, "n_comparisons": len(vals), "n_valid": len(d),
                           "difference_mean": float(np.mean(d)) if d else float("nan"),
                           "difference_sd": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
                           "comparisons_valid": " ".join(c for c, _, ok in vals if ok)})
    return by_probe, aggregated


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="*")
    parser.add_argument("--root", type=Path, default=None, help="a results root: every finished probe measurement under it")
    parser.add_argument("--checkpoint", type=int, default=None, help="keep this k only (one common checkpoint)")
    parser.add_argument("--pair-k", nargs="*", default=[], help="<condition>:<seed>=<k>: the checkpoint of a pair measured at several")
    parser.add_argument("--matched", nargs="*", default=[],
                        help="<condition>:<seed>@<k>=<condition>:<seed>@<k>: b of the first minus b of the second, probe by probe")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    dirs = list(args.dirs) + (find_probe_dirs(args.root) if args.root else [])
    if not dirs:
        parser.error("no probe directory")
    measures = newest_per_seed([load_probe_dir(d) for d in dirs])
    every = list(measures)  # --matched names its measurements among all of them
    if args.checkpoint is not None:
        measures = [m for m in measures if m["k"] == args.checkpoint]
    pair_k = {}
    for item in args.pair_k:
        cond_seed, k = item.split("=")
        cond, seed = cond_seed.rsplit(":", 1)
        pair_k[(cond, int(seed))] = int(k)
    measures = one_k_per_pair(measures, pair_k)
    by_seed, aggregated, extremes, seeds = aggregate(measures)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "probes_by_seed.csv", by_seed)
    write_csv(args.out / "probes_aggregated.csv", aggregated)
    write_csv(args.out / "extremes.csv", extremes)
    matched_rows, matched_agg = matched(every, args.matched) if args.matched else ([], [])
    write_csv(args.out / "matched_by_probe.csv", matched_rows)
    write_csv(args.out / "matched_aggregated.csv", matched_agg)
    used = {parse_measure(x): None for item in args.matched for x in item.split("=")}
    (args.out / "summary.json").write_text(json.dumps({"seeds": seeds, "measures": [m["dir"].name for m in measures],
                                                       "matched": args.matched,
                                                       "matched_measures": [m["dir"].name for m in every if (m["condition"], m["seed"], m["k"]) in used]},
                                                      indent=1), encoding="utf-8")
    for key, s in seeds.items():
        print(f"{key}: {len(s)} pair(s), [seed, checkpoint] {s}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
