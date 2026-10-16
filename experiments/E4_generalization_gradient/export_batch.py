"""One archive per E4 batch, to bring its results back from the cluster's volume (docs/RESULTS_CONVENTION.md, light
files), and its check once extracted.

  create  (Kubeflow notebook, from the checkout) -> <out>, a .tar.gz holding, with their paths in the checkout:
            - every run of the batch (the newest per job, batches.batch_runs): config.yaml, meta.yaml, results.json,
              metrics.csv, checkpoints.jsonl (the checkpoint diagnostics), console.log;
            - the weights of each run's FINAL checkpoint (checkpoints/agent_env<last step>.pt), and those of the steps
              given with --weights-checkpoints;
            - every probe measurement of the batch's configurations (results/E4_generalization_gradient/probes__*/):
              images.csv, probes.csv, curves.npz, steps.npz, meta.yaml, status.json;
            - the supervisor directories of the batch (logs/kubeflow/e4-<phase>-<batch>-*/: status.json,
              supervisor.json, one log per job) and their submission records;
            - MANIFEST.json: every file's path, size and SHA-256, and each run's status.
          Never the replay, the counts, resume.pt nor the other checkpoints' weights.
  verify  (your machine) <extracted dir>: every file of MANIFEST.json present with its size and SHA-256.

The analyses read the extracted directory directly (pilot_analysis.py --export <dir>; measure_probes.py and
pilot_analysis.py forgetting with the run directories it contains).

Usage:
  python experiments/E4_generalization_gradient/export_batch.py create --batch e4_wave1 [--weights-checkpoints 150000 175000]
      [--repo-dir .] [--out exports/<batch>__<UTC>.tar.gz]
  python experiments/E4_generalization_gradient/export_batch.py verify <extracted dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:  # the notebook's Python runs this script without the package installed
    sys.path.insert(0, str(_SRC))

import batches

EXPERIMENT_ID = "E4_generalization_gradient"
RUN_FILES = ("config.yaml", "meta.yaml", "results.json", "metrics.csv", "checkpoints.jsonl", "console.log")
PROBE_FILES = ("images.csv", "probes.csv", "curves.npz", "steps.npz", "meta.yaml", "status.json")
SUPERVISOR_FILES = ("status.json", "supervisor.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_runs(repo: Path, batch: str) -> dict[str, Path | None]:
    """job -> the newest run directory of that job (None when the run never started)."""
    root = repo / "results" / EXPERIMENT_ID
    out = {}
    for job, (cid, condition) in batches.batch_runs(batch).items():
        role = "experimental" if condition == "experimental" else "control"
        found = sorted(root.glob(f"*__2zones-{cid}__pair-{role}__*"))
        out[job] = found[-1] if found else None
    return out


def meta_fields(path: Path) -> dict:
    """The top-level scalar fields of a run's meta.yaml (standard library only: the notebook may lack pyyaml)."""
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line[0].isspace() and ": " in line:
                key, value = line.split(": ", 1)
                out[key] = value.strip().strip("'\"")
    return out


def final_step(run: Path) -> int | None:
    """The run's last checkpoint (the newest agent_env<NNNNNN>.pt)."""
    steps = sorted(int(p.stem[len("agent_env"):]) for p in (run / "checkpoints").glob("agent_env*.pt"))
    return steps[-1] if steps else None


def collect(repo: Path, batch: str, weight_steps: list[int]) -> tuple[list[Path], dict]:
    files, runs = [], {}
    for job, run in batch_runs(repo, batch).items():
        if run is None:
            runs[job] = {"run": None, "status": "not found"}
            continue
        meta = meta_fields(run / "meta.yaml")
        files += [run / name for name in RUN_FILES if (run / name).is_file()]
        last = final_step(run)
        steps = sorted({*([last] if last is not None else []), *weight_steps})
        weights = [run / "checkpoints" / f"agent_env{s:06d}.pt" for s in steps]
        missing = [w.name for w in weights if not w.is_file()]
        files += [w for w in weights if w.is_file()]
        runs[job] = {"run": run.name, "status": meta.get("status"), "final_checkpoint": last, "weights": [w.name for w in weights if w.is_file()],
                     "weights_missing": missing}
    for cid in batches.BATCHES[batch]["configurations"]:
        for probes in sorted((repo / "results" / EXPERIMENT_ID).glob(f"probes__{cid}__env*")):
            files += [probes / name for name in PROBE_FILES if (probes / name).is_file()]
    for sup in sorted((repo / "logs" / "kubeflow").glob(f"e4-*-{batch}-*")):
        if sup.is_dir():
            files += [sup / name for name in SUPERVISOR_FILES if (sup / name).is_file()] + sorted(sup.glob("*.log"))
        elif sup.name.endswith(".submission.json"):
            files.append(sup)
    return files, runs


def cmd_create(args) -> int:
    repo = Path(args.repo_dir).resolve()
    files, runs = collect(repo, args.batch, args.weights_checkpoints or [])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else repo / "exports" / f"{args.batch}__{stamp}.tar.gz"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"path": f.relative_to(repo).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files]
    manifest = {"batch": args.batch, "created_at": stamp, "weights_checkpoints": args.weights_checkpoints or [], "runs": runs,
                "files": entries, "total_bytes": sum(e["bytes"] for e in entries)}
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
    c.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[2]))
    c.add_argument("--weights-checkpoints", type=int, nargs="+", default=None, help="also these checkpoints' weights")
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("directory")
    args = parser.parse_args(argv)
    return cmd_create(args) if args.command == "create" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
