"""One archive per E5 batch, to bring its results back from the cluster's volume, and its check once extracted (E4's
export_batch.py, with E5's runs). Standard library only.

  create  (Kubeflow notebook, from the e5 checkout) --batch <wave> --configuration <id> -> <out>, a .tar.gz holding, with
          their paths in the checkout:
            - every run of the batch (the newest per job): config.yaml, meta.yaml, results.json, metrics.csv,
              checkpoints.jsonl, console.log, assignments.npz (shuffle: the episodes' assignments) and replay_final.npz
              (the compact replay: with the weights, the RSA and every measure can be recomputed off the cluster;
              --no-replay leaves it out);
            - the weights of each run's FINAL checkpoint, and those of the steps given with --weights-checkpoints;
            - every probe measurement of the batch's pairs (results/E5_belief_training/probes__<configuration>__<tag>__seed<NN>__*);
            - the supervisor directories of the batch (logs/kubeflow/e5-<phase>-<batch>-*) and their submission records;
            - MANIFEST.json: every file's path, size and SHA-256, and each run's status.
          Never resume.pt, the resumable states, nor the other checkpoints' weights.
  verify  (your machine) <extracted dir>: every file of MANIFEST.json present with its size and SHA-256.

Usage:
  python experiments/E5_belief_training/export_batch.py create --batch e5_wave1 --configuration p1-a-d1 [--weights-checkpoints 175000]
      [--no-replay] [--repo-dir .] [--out exports/<batch>__<UTC>.tar.gz]
  python experiments/E5_belief_training/export_batch.py verify <extracted dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import batches
import e5_names

RUN_FILES = ("config.yaml", "meta.yaml", "results.json", "metrics.csv", "checkpoints.jsonl", "console.log", "assignments.npz")
PROBE_FILES = ("images.csv", "probes.csv", "curves.npz", "steps.npz", "meta.yaml", "status.json")
SUPERVISOR_FILES = ("status.json", "supervisor.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seeds(repo: Path, configuration: str) -> dict[str, int]:
    """Seeds A to F of the configuration, read from the configs' text (no yaml here): A = its seed in configs/e4/e4.yaml
    (E5's base), B-F = A + 1-5 x e5.seed_b_offset (configs/e5/e5.yaml; e5_setup.configuration's rule)."""
    e5 = (repo / batches.CONFIG).read_text(encoding="utf-8")
    base = re.search(r"^base:\s*(\S+)", e5, re.M).group(1)
    offset = int(re.search(r"^\s*seed_b_offset:\s*(\d+)", e5, re.M).group(1))
    m = re.search(rf"^\s*{re.escape(configuration)}:\s*\{{[^}}]*seed:\s*(\d+)", (repo / base).read_text(encoding="utf-8"), re.M)
    if m is None:
        raise SystemExit(f"configuration {configuration!r} not found in {base}")
    return {label: int(m.group(1)) + i * offset for i, label in enumerate("ABCDEF")}


def meta_fields(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line[0].isspace() and ": " in line:
                key, value = line.split(": ", 1)
                out[key] = value.strip().strip("'\"")
    return out


def final_step(run: Path) -> int | None:
    steps = sorted(int(p.stem[len("agent_env"):]) for p in (run / "checkpoints").glob("agent_env*.pt"))
    return steps[-1] if steps else None


def collect(repo: Path, batch: str, configuration: str, weight_steps: list[int], replay: bool) -> tuple[list[Path], dict]:
    files, runs = [], {}
    seed_of = seeds(repo, configuration)
    root = repo / "results" / e5_names.EXPERIMENT_ID
    for job, (label, training, fraction, condition) in batches.batch_runs(batch).items():
        found = sorted(root.glob(e5_names.run_glob(configuration, training, condition, seed_of[label], fraction)))
        if not found:
            runs[job] = {"run": None, "status": "not found"}
            continue
        run = found[-1]
        meta = meta_fields(run / "meta.yaml")
        files += [run / name for name in RUN_FILES + (("replay_final.npz",) if replay else ()) if (run / name).is_file()]
        last = final_step(run)
        steps = sorted({*([last] if last is not None else []), *weight_steps})
        weights = [run / "checkpoints" / f"agent_env{s:06d}.pt" for s in steps]
        files += [w for w in weights if w.is_file()]
        runs[job] = {"run": run.name, "status": meta.get("status"), "final_checkpoint": last, "weights": [w.name for w in weights if w.is_file()],
                     "weights_missing": [w.name for w in weights if not w.is_file()]}
    for label, training, fraction in batches.PROBE_PAIRS.get(batch, []):  # a training-only batch has no probe pair
        tag = e5_names.training_tag(training, fraction)
        for probes in sorted(root.glob(f"probes__{configuration}__{tag}__seed{seed_of[label]:02d}__*")):
            files += [probes / name for name in PROBE_FILES if (probes / name).is_file()]
    for sup in sorted((repo / "logs" / "kubeflow").glob(f"e5-*-{batch}-*")):
        if sup.is_dir():
            files += [sup / name for name in SUPERVISOR_FILES if (sup / name).is_file()] + sorted(sup.glob("*.log"))
        elif sup.name.endswith(".submission.json"):
            files.append(sup)
    return files, runs


def cmd_create(args) -> int:
    repo = Path(args.repo_dir).resolve()
    files, runs = collect(repo, args.batch, args.configuration, args.weights_checkpoints or [], not args.no_replay)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else repo / "exports" / f"{args.batch}__{stamp}.tar.gz"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"path": f.relative_to(repo).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files]
    manifest = {"batch": args.batch, "configuration": args.configuration, "created_at": stamp,
                "weights_checkpoints": args.weights_checkpoints or [], "runs": runs, "files": entries,
                "total_bytes": sum(e["bytes"] for e in entries)}
    manifest_path = out.parent / f"{out.name}.MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with tarfile.open(out, "w:gz") as tar:
        for f, e in zip(files, entries):
            tar.add(f, arcname=e["path"], recursive=False)
        tar.add(manifest_path, arcname="MANIFEST.json")
    manifest_path.unlink()
    print(json.dumps({"archive": str(out), "archive_bytes": out.stat().st_size, "files": len(entries),
                      "content_bytes": manifest["total_bytes"], "runs": runs}, indent=1))
    problems = [job for job, r in runs.items() if r["status"] != "done" or r.get("weights_missing")]
    if problems:
        print(f"WARNING: runs not done or with missing weights: {problems} (they are in the archive as they are)")
    return 0


def cmd_verify(args) -> int:
    root = Path(args.directory)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = []
    for e in manifest["files"]:
        f = root / e["path"]
        if not f.is_file():
            bad.append(f"missing: {e['path']}")
        elif f.stat().st_size != e["bytes"] or sha256(f) != e["sha256"]:
            bad.append(f"differs: {e['path']}")
    print(f"{len(manifest['files'])} files checked, {len(bad)} problem(s)")
    for line in bad:
        print("  " + line)
    for job, r in manifest["runs"].items():
        print(f"  {job}: {r.get('run')} -- status {r.get('status')}, final checkpoint {r.get('final_checkpoint')}, weights {r.get('weights')}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--batch", choices=list(batches.BATCHES), required=True)
    c.add_argument("--configuration", required=True)
    c.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[2]))
    c.add_argument("--weights-checkpoints", type=int, nargs="+", default=None, help="also these checkpoints' weights")
    c.add_argument("--no-replay", action="store_true", help="leave the runs' replay_final.npz out")
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("directory")
    args = parser.parse_args(argv)
    return cmd_create(args) if args.command == "create" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
