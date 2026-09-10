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
