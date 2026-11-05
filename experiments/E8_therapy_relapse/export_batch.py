"""One archive of an E8 batch, to bring it back from the cluster's volume, and its check once extracted (E6's
export_batch.py for E8's runs; docs/RESULTS_CONVENTION.md). Standard library only.

  create  (Kubeflow notebook, from the e8 checkout) --batch <batch> -> a .tar.gz holding, with their paths in the
          checkout, for every run of the batch (the newest run directory of each model): its light files (config.yaml,
          meta.yaml, checkpoints.jsonl, console.log, metrics.csv, results.json, steps.npz, assignments.npz) and the
          weights of every checkpoint (checkpoints/agent_env<N>.pt: what the local probes read, about 2.7 MB each); the
          probe jobs of the batch already run on the cluster (results/E8_therapy_relapse/probes__<reference model>__*:
          every file), the supervisor directories (logs/kubeflow/e8-<phase>-<batch>-*) with their submission records, and
          MANIFEST.json (path, size, SHA-256 of every file; each run's status). Never: resume states (resume.pt,
          resumable_env*.pt), replay_final.npz, agent_final.pt.
  verify  (your machine) <extracted dir>: every file of MANIFEST.json present with its size and SHA-256.

Usage:
  python experiments/E8_therapy_relapse/export_batch.py create --batch e8_w1_base [--repo-dir .] [--out exports/<batch>__<UTC>.tar.gz]
  python experiments/E8_therapy_relapse/export_batch.py verify <extracted dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import batches

EXPERIMENT_ID = "E8_therapy_relapse"
RUN_FILES = ("config.yaml", "meta.yaml", "checkpoints.jsonl", "console.log", "metrics.csv", "results.json", "steps.npz", "assignments.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def status_of(run: Path) -> str | None:
    meta = run / "meta.yaml"
    if not meta.is_file():
        return None
    for line in meta.read_text(encoding="utf-8").splitlines():  # standard library only: the status line of the YAML
        if line.startswith("status:"):
            return line.split(":", 1)[1].strip()
    return None


def run_dirs(repo: Path, batch: str) -> dict[str, Path | None]:
    """model id -> the newest run directory of the batch's run of that model (by its name's model fields)."""
    out = {}
    root = repo / "results" / EXPERIMENT_ID
    for model, q in batches.BATCHES[batch]["runs"].items():
        kind = "base" if model.startswith("base") else "sh100"
        therapy = "full" if q is None else q
        role = "control" if model.endswith("_ref") else "experimental"
        found = sorted(d for d in root.glob(f"grid12x12__2zones-p1-b-d1-{kind}-th{therapy}__pair-{role}__worldmodel-agent-e8__seed*__*")
                       if d.is_dir() and _model_of(d) == model)
        out[model] = found[-1] if found else None
    return out


def _model_of(run: Path) -> str | None:
    cfg = run / "config.yaml"
    if not cfg.is_file():
        return None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("model:") and line.startswith("    model:"):  # e8.run.model (yaml.safe_dump's indentation)
            return line.split(":", 1)[1].strip()
    return None


def collect(repo: Path, batch: str) -> tuple[list[Path], dict]:
    files, outputs = [], {}
    for model, d in run_dirs(repo, batch).items():
        if d is None:
            outputs[model] = {"dir": None, "status": "not found"}
            continue
        files += [d / name for name in RUN_FILES if (d / name).is_file()]
        files += sorted((d / "checkpoints").glob("agent_env*.pt"))
        outputs[model] = {"dir": d.relative_to(repo).as_posix(), "status": status_of(d)}
    agents_refs = {m for m in batches.BATCHES[batch]["runs"] if m.endswith("_ref")}
    for ref in sorted(agents_refs):
        for job in sorted((repo / "results" / EXPERIMENT_ID).glob(f"probes__{ref}__*")):
            files += sorted(f for f in job.rglob("*") if f.is_file())
    for sup in sorted((repo / "logs" / "kubeflow").glob(f"e8-*-{batch}-*")):
        if sup.is_dir():
            files += sorted(f for f in sup.iterdir() if f.is_file())
        elif sup.name.endswith(".submission.json"):
            files.append(sup)
    return files, outputs


def cmd_create(args) -> int:
    repo = Path(args.repo_dir).resolve()
    files, outputs = collect(repo, args.batch)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else repo / "exports" / f"{args.batch}__{stamp}.tar.gz"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"path": f.relative_to(repo).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files]
    manifest = {"batch": args.batch, "created_at": stamp, "outputs": outputs, "files": entries, "total_bytes": sum(e["bytes"] for e in entries)}
    manifest_path = out.parent / f"{out.name}.MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with tarfile.open(out, "w:gz") as tar:
        for f, e in zip(files, entries):
            tar.add(f, arcname=e["path"], recursive=False)
        tar.add(manifest_path, arcname="MANIFEST.json")
    manifest_path.unlink()
    print(json.dumps({"archive": str(out), "archive_bytes": out.stat().st_size, "files": len(entries), "outputs": outputs}, indent=1))
    problems = [k for k, o in outputs.items() if o["status"] != "done"]
    if problems:
        print(f"WARNING: runs not done or not found: {problems} (they are in the archive as they are)")
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
    for key, o in manifest["outputs"].items():
        print(f"  {key}: {o.get('dir')} -- status {o.get('status')}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--batch", choices=list(batches.BATCHES), required=True)
    c.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[2]))
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("directory")
    args = parser.parse_args(argv)
    return cmd_create(args) if args.command == "create" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
