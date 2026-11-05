"""Step 5 (H4, PREDICTIONS.md): every model's training history from its records only (checkpoints.jsonl), no computation
on the weights. A continuation (meta.yaml branched_from) is joined to the run it was branched from, found by name under
--history-root; the original's records stop where the continuation's start.

Read with check_wave.py's own functions (read_run, switch_of, collapse_of, lock_in_of): the estimator switch, the
switch statistic after it (its minimum: how close the run came to the lock-in rule's LOCK_BITS), the lock-in rule.
Per checkpoint: e5.decode_accuracy_canonical (shuffle runs only), the min over the four quadrants of
e4.real_context.mass_true_by_region (the canonical image's real-context mass), batch_pair_js, success_rate.

history.csv (one row per model and checkpoint) and history_summary.csv (one row per model).

Usage, from the repo root:
  python experiments/E5_belief_training/analysis_robustness/training_history.py --runs-root <root> [...]
      --history-root <dir holding the originals> [...] --out experiments/E5_belief_training/analysis_robustness/tables
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # this experiment's check_wave.py

import check_wave  # noqa: E402
import common  # noqa: E402

QUADRANTS = ("nw", "ne", "se", "sw")


def find_records(name: str, history_roots: list[Path]) -> Path:
    found = next((d for root in history_roots for d in root.rglob(name) if (d / "checkpoints.jsonl").is_file()), None)
    if found is None:
        raise SystemExit(f"{name}: no run of that name with its records under any --history-root")
    return found


def chain(run_dir: Path, history_roots: list[Path]) -> tuple[list[dict], list[str]]:
    """The records of a run and of the runs it continues, oldest first, and the run names. A run directory without its
    records (a weights-only copy) is read from the run of the same name under a history root."""
    if not (run_dir / "checkpoints.jsonl").is_file():
        run_dir = find_records(run_dir.name, history_roots)
    run = check_wave.read_run(run_dir)
    src = run["meta"].get("branched_from")
    if not src:
        return run["records"], [run["name"]]
    older, names = chain(find_records([p for p in src.replace("\\", "/").split("/") if p][-3], history_roots), history_roots)
    first = min(int(r["env_step"]) for r in run["records"])
    return [r for r in older if int(r["env_step"]) < first] + run["records"], names + [run["name"]]


def mass_min(record: dict) -> float | None:
    by = ((record.get("e4") or {}).get("real_context") or {}).get("mass_true_by_region") or {}
    values = [check_wave.number(by.get(q)) for q in QUADRANTS]
    return min(values) if values and None not in values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, nargs="+", required=True)
    parser.add_argument("--history-root", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, summary = [], []
    for model in common.load_models():
        weights = common.weights_of(model, args.runs_root)
        if weights is None:
            continue
        records, names = chain(weights.parent.parent, args.history_root)
        run = {"dir": weights.parent.parent, "records": records, "meta": {}}
        sw = check_wave.switch_of(run)
        collapse = check_wave.collapse_of(run, sw["grad_step"])
        lock = check_wave.lock_in_of(run, sw["grad_step"] is not None)
        decode_first = None
        for r in records:
            dec = (r.get("e5") or {}).get("decode_accuracy_canonical")
            rows.append({"id": model["id"], "env_step": int(r["env_step"]), "grad_step": r.get("grad_step"), "decode_accuracy_canonical": dec,
                         "mass_min_quadrant": mass_min(r), "batch_pair_js": r.get("batch_pair_js"),
                         "switch_stat_min": (r.get("switch_pairs") or {}).get("stat_min"), "success_rate": r.get("success_rate")})
            if decode_first is None and dec is not None and dec >= 0.9:
                decode_first = int(r["env_step"])
        at = next((r for r in records if int(r["env_step"]) == model["checkpoint"]), None)
        summary.append({"id": model["id"], "role": model["role"], "seed": model["seed"], "checkpoint": model["checkpoint"],
                        "runs": " <- ".join(reversed(names)), "n_records": len(records),
                        "switch_grad_step": sw["grad_step"], "switch_env_step": sw.get("env_step"), "switch_rule": sw.get("rule"),
                        "js_min_after_switch": collapse["min_after_switch"], "js_evals_after_switch": collapse["evaluations_after_switch"],
                        "longest_below_1bit": collapse["longest_below_1bit"], "lock_in": lock is not None,
                        "lock_bits": check_wave.LOCK_BITS, "decode_accuracy_first_ge_0.9": decode_first,
                        "decode_accuracy_at_checkpoint": (at or {}).get("e5", {}).get("decode_accuracy_canonical") if at else None,
                        "mass_min_quadrant_at_checkpoint": mass_min(at) if at else None})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("history.csv", rows), ("history_summary.csv", summary)):
        with open(args.out / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for s in summary:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
