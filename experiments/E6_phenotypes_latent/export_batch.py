"""One archive of E6 / E7's outputs, to bring them back from the cluster's volume, and its check once extracted (E5's
export_batch.py for E6's outputs). Standard library only.

  create  (Kubeflow notebook, from the e6 checkout) --batch <batch> [--phases e6 e7-directions e7-apply] -> a .tar.gz holding,
          with their paths in the checkout, for every agent of the batch: the newest output directory of each phase
          (results/E6_phenotypes_latent/e6__<id>__*, results/E7_latent_manipulation/e7dir__<id>__* and e7apply__<id>__*: every
          file -- rows, episodes, directions, meta, status), the supervisor directories (logs/kubeflow/e6-<phase>-<batch>-*)
          with their submission records, and MANIFEST.json (path, size, SHA-256 of every file; each output's status).
          No weights: E6 and E7 train nothing.
  verify  (your machine) <extracted dir>: every file of MANIFEST.json present with its size and SHA-256.

Usage:
  python experiments/E6_phenotypes_latent/export_batch.py create --batch e6_e5 [--phases ...] [--repo-dir .] [--out exports/<batch>__<UTC>.tar.gz]
  python experiments/E6_phenotypes_latent/export_batch.py verify <extracted dir>
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

PREFIX = {"e6": ("E6_phenotypes_latent", "e6"), "e7-directions": ("E7_latent_manipulation", "e7dir"), "e7-apply": ("E7_latent_manipulation", "e7apply")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(repo: Path, batch: str, phases: list[str]) -> tuple[list[Path], dict]:
    files, outputs = [], {}
    for phase in phases:
        experiment, prefix = PREFIX[phase]
        for agent in batches.agent_ids(repo, batch):
            found = sorted((repo / "results" / experiment).glob(f"{prefix}__{agent}__*"))
            key = f"{phase}/{agent}"
            if not found:
                outputs[key] = {"dir": None, "status": "not found"}
                continue
            d = found[-1]
            status = json.loads((d / "status.json").read_text(encoding="utf-8")).get("status") if (d / "status.json").is_file() else None
            files += sorted(f for f in d.rglob("*") if f.is_file())
            outputs[key] = {"dir": d.relative_to(repo).as_posix(), "status": status}
        for sup in sorted((repo / "logs" / "kubeflow").glob(f"e6-{phase}-{batch}-*")):
            if sup.is_dir():
                files += sorted(f for f in sup.iterdir() if f.is_file())
            elif sup.name.endswith(".submission.json"):
                files.append(sup)
    return files, outputs


def cmd_create(args) -> int:
    repo = Path(args.repo_dir).resolve()
    files, outputs = collect(repo, args.batch, args.phases)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else repo / "exports" / f"{args.batch}__{stamp}.tar.gz"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"path": f.relative_to(repo).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files]
    manifest = {"batch": args.batch, "phases": args.phases, "created_at": stamp, "outputs": outputs, "files": entries,
                "total_bytes": sum(e["bytes"] for e in entries)}
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
        print(f"WARNING: outputs not done or not found: {problems} (they are in the archive as they are)")
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
    c.add_argument("--phases", nargs="+", choices=list(batches.PHASES), default=list(batches.PHASES))
    c.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[2]))
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("directory")
    args = parser.parse_args(argv)
    return cmd_create(args) if args.command == "create" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
