"""The files of the therapy runs brought back apart from the batch archive (replay_final.npz, counts.npz: export_batch.py
never takes them), put into their run directories of the extracted batch (docs/e8/DECISIONS.md (22)). Standard library
only.

A source file is named <run directory>__<file name> (e.g. grid12x12__...__20261106T151313Z__counts.npz); it is COPIED
(the original kept) to <batch root>/results/E8_therapy_relapse/<run directory>/<file name>, the run being identified by
the directory name inside the file name, never guessed: the directory must exist and hold its meta.yaml. The copy's
SHA-256 is checked against the source's; a file already in place with the same content is left as it is, one with a
different content stops everything before any copy.

Usage:
  python experiments/E8_therapy_relapse/place_run_files.py --src <dir of the files> --batch-root <extracted batch> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

EXPERIMENT_ID = "E8_therapy_relapse"
FILE_NAMES = ("replay_final.npz", "counts.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(src: Path, batch_root: Path) -> list[dict]:
    """One entry per source file: source, destination, SHA-256, and whether it is already in place. Raises on a file
    whose run directory is missing, on an unknown file name, and on a destination with a different content."""
    runs_root = batch_root / "results" / EXPERIMENT_ID
    out = []
    for f in sorted(src.iterdir()):
        if not f.is_file() or "__" not in f.name or not f.name.endswith(".npz"):
            continue
        run, _, name = f.name.rpartition("__")
        if name not in FILE_NAMES:
            raise ValueError(f"{f.name}: unknown file name {name!r} (expected one of {FILE_NAMES})")
        run_dir = runs_root / run
        if not (run_dir / "meta.yaml").is_file():
            raise ValueError(f"{f.name}: no run directory {run_dir} with a meta.yaml")
        dest = run_dir / name
        digest = sha256(f)
        if dest.exists() and sha256(dest) != digest:
            raise ValueError(f"{dest} exists with another content: nothing copied")
        out.append({"source": str(f), "destination": str(dest), "sha256": digest, "in_place": dest.exists()})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    entries = plan(args.src, args.batch_root)
    if not entries:
        raise SystemExit(f"{args.src}: no <run directory>__<file>.npz")
    for e in entries:
        if not e["in_place"] and not args.dry_run:
            shutil.copy2(e["source"], e["destination"])
            if sha256(Path(e["destination"])) != e["sha256"]:
                raise SystemExit(f"{e['destination']}: SHA-256 differs after the copy")
        e["copied"] = not e["in_place"] and not args.dry_run
    print(json.dumps(entries, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
