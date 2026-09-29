"""Results-naming convention for the zone/dynamics/appearance family (see
CLAUDE.md): each run gets its own self-contained directory, named so it can
be identified without opening any file, holding its resolved config,
metadata (seed, git commit, status), metrics, and artefacts.

results/<experiment_id>/<run_name>/
    config.yaml       -- output of helplessness.config.compose(), fully resolved
    meta.yaml          -- seed, git commit, status, timestamps
    metrics.csv        -- (or whatever the experiment calls its metrics file)
    checkpoints/        -- world-model runs: periodic saves + latest.json pointer
    artifacts/          -- figures, rendered samples, etc.

Not used by E0 (offline_sweep etc. keep their existing, already-documented
results/*.csv convention -- see CONTRIBUTING.md) -- this is for E1 onwards.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

_VALID_PAIR_ROLES = ("experimental", "control")


def make_run_name(
    grid_shape: tuple[int, int],
    zone_preset: str,
    pair_role: str,
    agent_type: str,
    seed: int,
    timestamp: str | None = None,
) -> str:
    """grid16x16__4zones-std__pair-experimental__tabular__seed03__20260910T153000Z

    pair_role is "experimental" or "control" -- never implicit (see
    CLAUDE.md: the pairing must be explicit in the run's own name, not
    something you infer from which config file happened to produce it).
    timestamp defaults to now (UTC, disambiguates reruns with identical
    parameters); pass one explicitly to make the name deterministic in a
    test.
    """
    if pair_role not in _VALID_PAIR_ROLES:
        raise ValueError(f"pair_role must be one of {_VALID_PAIR_ROLES}, got {pair_role!r}")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    grid_str = f"grid{int(grid_shape[0])}x{int(grid_shape[1])}"
    return f"{grid_str}__{zone_preset}__pair-{pair_role}__{agent_type}__seed{int(seed):02d}__{timestamp}"


def git_commit_hash(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return "unknown"


# The code and configs a run depends on: what source_tree_hash covers. On a clean checkout these are
# exactly the files git tracks under these paths (caches and build metadata excluded below).
SOURCE_SCOPE = ("src", "experiments", "configs", "pyproject.toml")


def _outside_source(rel: Path) -> bool:
    return any(part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts) or rel.suffix in (".pyc", ".pyo")


def source_tree_hash(root: str | Path, scope: tuple[str, ...] = SOURCE_SCOPE) -> dict | None:
    """sha256 of the contents of the files under `scope` (relative to
    root), for a machine without git (docs/e3/DECISIONS.md, 2026-09-29):
    each file contributes its POSIX relative path, its length and its
    bytes, in sorted path order; CRLF is read as LF, so a Windows checkout
    and a Linux one of the same commit give the same hash. Returns
    {"source_sha256", "source_n_files"}, or None when the scope holds no
    file. To check a run against a commit: check the commit out and call
    this function on the checkout."""
    root = Path(root)
    files = []
    for entry in scope:
        path = root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files += [f for f in path.rglob("*") if f.is_file() and not _outside_source(f.relative_to(root))]
    if not files:
        return None
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        data = f.read_bytes().replace(b"\r\n", b"\n")
        digest.update(f.relative_to(root).as_posix().encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data)
    return {"source_sha256": digest.hexdigest(), "source_n_files": len(files)}


def code_provenance(root: str | Path) -> dict:
    """What identifies the code a run was launched from: git_commit,
    git_dirty and git_n_changed_paths when git works in root ("unknown" /
    None otherwise -- no git executable, or not a checkout), and always the
    content hash of source_tree_hash (source_sha256, source_n_files, and
    source_scope), None when it cannot be computed."""
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    if commit and status is not None:
        changed = [line for line in status.splitlines() if line.strip()]
        out = {"git_commit": commit, "git_dirty": bool(changed), "git_n_changed_paths": len(changed)}
    else:
        out = {"git_commit": "unknown", "git_dirty": None, "git_n_changed_paths": None}
    source = source_tree_hash(root)
    out.update(source or {"source_sha256": None, "source_n_files": None})
    out["source_scope"] = list(SOURCE_SCOPE)
    return out


class UnknownCodeVersionError(RuntimeError):
    """A run would start without a git commit and without a content hash
    of its code: its results could not be tied to any version."""


def require_code_provenance(provenance: dict) -> None:
    """Raises UnknownCodeVersionError unless the provenance has a git
    commit or a source content hash (docs/e3/DECISIONS.md, 2026-09-29:
    the first GPU batch ran with git_commit "unknown")."""
    if provenance.get("git_commit", "unknown") == "unknown" and not provenance.get("source_sha256"):
        raise UnknownCodeVersionError("refusing to launch a run with neither a git commit (git unavailable or not a checkout) "
                                      "nor a content hash of its source files (runs.source_tree_hash found no file)")


def make_run_dir(results_root: str | Path, experiment_id: str, run_name: str) -> Path:
    """Creates results/<experiment_id>/<run_name>/ with its checkpoints/
    and artifacts/ subdirectories. Fails if the directory already exists
    (a rerun must get a new name -- see make_run_name's timestamp -- rather
    than silently overwrite a previous run's config/metrics)."""
    run_dir = Path(results_root) / experiment_id / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "artifacts").mkdir()
    return run_dir


def write_run_meta(run_dir: str | Path, seed: int, status: str = "running", **extra) -> None:
    """Writes/overwrites meta.yaml in run_dir -- call again with a new
    status ("done", "failed") as the run progresses; git commit and
    started_at are only set on first write."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
    else:
        meta = {"seed": seed, "git_commit": git_commit_hash(), "started_at": _now_iso()}
    meta["status"] = status
    meta["updated_at"] = _now_iso()
    meta.update(extra)
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def write_checkpoint_pointer(run_dir: str | Path, checkpoint_filename: str, step: int) -> None:
    """Writes checkpoints/latest.json -- the resume-after-preemption
    pointer. A plain JSON file with the filename, not a symlink (symlinks
    are unreliable across the Windows/cluster filesystems this project
    actually runs on)."""
    checkpoints_dir = Path(run_dir) / "checkpoints"
    with open(checkpoints_dir / "latest.json", "w") as f:
        json.dump({"filename": checkpoint_filename, "step": step}, f)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
