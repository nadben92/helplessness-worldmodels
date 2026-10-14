"""Where the E3 report's figures and numbers read their data, and how. Data access only: no analysis here.

Every E3 source lives under results/E3_worldmodel_4zones/ (read-only):
  - E3-0 (offline calibration, gpu_e30b): the 8 run directories *gpu_e30b_win{64,16}__seed{00,05}*;
  - the bridge run's recomputation at K_s = 1,358: *gpu_e30_bridge*/recomputed/BRIDGE_RECOMPUTATION;
  - the closed-loop pilot (gpu_pilot_cl): the 4 run directories *gpu_pilot_closed_loop__seed{00,05}*, extracted from
    handoff/gpu_pilot_cl/; its checkpoint re-measurement, gpu_pilot_remeasure/<run>/env<NNNNNN>/, extracted from
    handoff/gpu_pilot_remeasure/; its measurement-noise batch, gpu_pilot_noise/, extracted from handoff/gpu_pilot_noise/;
  - the final campaign (gpu_campaign): the outputs of its own pipeline (campaign_report.py), versioned in
    docs/e3/campaign_figures/ (figures, included as they are) and docs/e3/campaign_tables/analysis.json (CAMPAIGN_JSON:
    the values, read as they are, never recomputed).

A figure script whose sources are pending calls `pending(...)`: it prints why and exits with PENDING_EXIT, which
paper/build.py reports as pending (the report shows \\pending{} in its place) instead of failing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import common

from helplessness.experiment_loader import load_experiment_module

E3_RESULTS = common.REPO_ROOT / "results" / "E3_worldmodel_4zones"
CAMPAIGN_JSON = common.REPO_ROOT / "docs" / "e3" / "campaign_tables" / "analysis.json"
REMEASURE_ROOT = E3_RESULTS / "gpu_pilot_remeasure"
NOISE_ROOT = E3_RESULTS / "gpu_pilot_noise"
BRIDGE_RECOMPUTATION = "final__ks1358__20260930T124347Z"  # the K_s = 1,358 measures of DECISIONS 2026-09-30 (1)

PENDING_EXIT = 3
N_PILOT_RUNS = 4      # seeds 0 and 5, environment and reference (DECISIONS 2026-10-02 (19))

# Display order and short names of the four zone types (the notes' abbreviations).
ZONES = ("uncontrollable_deterministic", "controllable_noisy", "uncontrollable_noisy", "controllable_deterministic")
ZONE_ABBR = {"uncontrollable_deterministic": "UD", "controllable_noisy": "CN",
             "uncontrollable_noisy": "UN", "controllable_deterministic": "CD"}
CONDITION_ABBR = {"experimental": "env", "reference": "ref"}


def pending(reason: str) -> None:
    """Stops a figure script whose sources are not there yet (the final campaign)."""
    print(f"PENDING: {reason}")
    sys.exit(PENDING_EXIT)


def e3_module(filename: str):
    """An E3 analysis module (experiments/E3_worldmodel_4zones/<filename>), imported as the experiment imports it."""
    return load_experiment_module("E3_worldmodel_4zones", filename, f"paper_e3_{Path(filename).stem}")


def read_json(path: Path) -> dict:
    common.require_file(path)
    with open(common.long_path(path), encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    common.require_file(path)
    with open(common.long_path(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def listdir(path: Path) -> list[str]:
    if not os.path.isdir(common.long_path(path)):
        raise SystemExit(f"Missing source directory: {path}")
    return sorted(os.listdir(common.long_path(path)))


def _runs(pattern: str, n_expected: int, root: Path = E3_RESULTS) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"Missing source directory: {root}")
    found = sorted(p for p in root.glob(pattern) if p.is_dir())
    if len(found) != n_expected:
        raise SystemExit(f"Expected {n_expected} run directories {pattern} in {root}, found {len(found)}")
    return found


def e30b_runs(shape: str) -> list[Path]:
    """The 4 gpu_e30b runs of one window shape ("win64" or "win16")."""
    return _runs(f"*worldmodel-gpu_e30b_{shape}__seed*", 4)


def bridge_recomputation() -> Path:
    bridge = _runs("*worldmodel-gpu_e30_bridge__seed*", 1)[0]
    path = bridge / "recomputed" / BRIDGE_RECOMPUTATION
    common.require_file(path / "results.json")
    return path


def pilot_runs() -> list[Path]:
    return _runs("*worldmodel-agent-gpu_pilot_closed_loop__seed*", N_PILOT_RUNS)


def run_identity(run_dir: Path) -> tuple[int, str]:
    """(seed, condition) as the run recorded them."""
    res = read_json(run_dir / "results.json")
    return int(res["seed"]), res["condition"]


def run_label(run_dir: Path) -> str:
    seed, condition = run_identity(run_dir)
    return f"{CONDITION_ABBR[condition]} {seed}"
