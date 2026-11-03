"""Step 1 (README.md): the model inventory and the robustness scores read from the cluster's swap probes. No forward pass.

inventory.csv  one row per model of models.yaml: its weights found under --runs-root (or MISSING), their SHA-256 and
               whether it equals the one the cluster's probe measurement recorded (meta.yaml weights_sha256), the probe
               directory and the pair's measurement seed.
localisation.csv  one row per (model, swap, axis, k, zone): the cluster's localisation at step 1 (common.localisation_table).
scores.csv     one row per (model, axis, where): first_t_below and auc (common.scores) of the curve averaged over the
               three swaps; where = all (the four zones), moved (the zone is the recoloured slot m), other.

Usage, from the repo root:
  python experiments/E5_belief_training/analysis_robustness/inventory_scores.py --runs-root <root> [<root> ...]
      --probes-root <extracted>/results/E5_belief_training [...] --out experiments/E5_belief_training/analysis_robustness/tables
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import common


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, nargs="+", required=True)
    parser.add_argument("--probes-root", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=None, help="a run's config.yaml (default: the first probed run's)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    inventory, loc_rows, score_rows = [], [], []
    for model in common.load_models():
        weights = common.weights_of(model, args.runs_root)
        pdir = common.probes_dir_of(model, args.probes_root)
        meta = common.probe_meta(pdir) if pdir else {}
        sha = common.sha256(weights) if weights else ""
        sha_probe = meta.get("weights_sha256", {}).get(model["probes"]["model"], "")
        inventory.append({"id": model["id"], "agent": model["agent"], "role": model["role"], "seed": model["seed"],
                          "checkpoint": model["checkpoint"], "weights": str(weights) if weights else "MISSING",
                          "sha256": sha, "sha256_matches_probe": (sha == sha_probe) if weights and sha_probe else "",
                          "probes_dir": pdir.name if pdir else "MISSING", "pair_seed": meta.get("seed", "")})
        if pdir is None:
            continue
        config = args.config or (weights.parent.parent / "config.yaml" if weights else None)
        if config is None:
            raise SystemExit("give --config (no local run to read it from)")
        args.config = config
        _, _, images = common.setup(str(config))
        rows = common.localisation_table(images, common.read_reconstruction(pdir, model["probes"]["model"]))
        loc_rows += [{"id": model["id"], **r} for r in rows]
        for axis in common.AXES:
            for where in (None, "moved", "other"):
                t, loc = common.curve(rows, axis, where)
                score_rows.append({"id": model["id"], "role": model["role"], "axis": axis, "where": where or "all",
                                   **common.scores(t, loc), **{f"t{round(x * 10):02d}": round(float(v), 4) for x, v in zip(t, loc)}})
    write_csv(args.out / "inventory.csv", inventory)
    write_csv(args.out / "localisation.csv", loc_rows)
    write_csv(args.out / "scores.csv", score_rows)
    for r in inventory:
        print(f"{r['id']:16s} {r['role']:12s} seed {r['seed']:>3} k {r['checkpoint']:>6}  sha ok: {str(r['sha256_matches_probe']):5s}  {r['weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
