"""Where the E3 report's figures and numbers read their data, and how. Data access only: no analysis here.

Every E3 source lives under results/E3_worldmodel_4zones/ (read-only):
  - E3-0 (offline calibration, gpu_e30b): the 8 run directories *gpu_e30b_win{64,16}__seed{00,05}*;
  - the bridge run's recomputation at K_s = 1,358: *gpu_e30_bridge*/recomputed/BRIDGE_RECOMPUTATION;
  - the closed-loop pilot (gpu_pilot_cl): the 4 run directories *gpu_pilot_closed_loop__seed{00,05}*, extracted from
    handoff/gpu_pilot_cl/; its checkpoint re-measurement, gpu_pilot_remeasure/<run>/env<NNNNNN>/, extracted from
    handoff/gpu_pilot_remeasure/; its measurement-noise batch, gpu_pilot_noise/, extracted from handoff/gpu_pilot_noise/;
  - the final campaign (gpu_campaign): CAMPAIGN_ROOT/<run>/, extracted from handoff/gpu_campaign/ (paper/README.md).
    Until it exists, every campaign figure and number is PENDING, never a fallback.

A figure script whose sources are pending calls `pending(...)`: it prints why and exits with PENDING_EXIT, which
paper/build.py reports as pending (the report shows \\pending{} in its place) instead of failing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import common

from helplessness.experiment_loader import load_experiment_module

E3_RESULTS = common.REPO_ROOT / "results" / "E3_worldmodel_4zones"
CAMPAIGN_ROOT = E3_RESULTS / "gpu_campaign"
REMEASURE_ROOT = E3_RESULTS / "gpu_pilot_remeasure"
NOISE_ROOT = E3_RESULTS / "gpu_pilot_noise"
BRIDGE_RECOMPUTATION = "final__ks1358__20260930T124347Z"  # the K_s = 1,358 measures of DECISIONS 2026-09-30 (1)

PENDING_EXIT = 3
N_PILOT_RUNS = 4      # seeds 0 and 5, environment and reference (DECISIONS 2026-10-02 (19))
N_CAMPAIGN_RUNS = 8   # seeds 0, 1, 5, 6, environment and reference (DECISIONS 2026-10-07 (30))

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


def campaign_runs() -> list[Path]:
    """The final campaign's 8 run directories, or PENDING if they have not been extracted yet."""
    if not CAMPAIGN_ROOT.is_dir():
        pending(f"final campaign not extracted: {CAMPAIGN_ROOT} does not exist (paper/README.md, E3 report)")
    found = sorted(p for p in CAMPAIGN_ROOT.iterdir() if p.is_dir())
    if len(found) != N_CAMPAIGN_RUNS:
        raise SystemExit(f"Expected {N_CAMPAIGN_RUNS} run directories in {CAMPAIGN_ROOT}, found {len(found)}")
    return found


def campaign_analysis(description: str, argv: list[str] | None = None) -> tuple[dict, Path]:
    """(experiments/E3_worldmodel_4zones/campaign_analysis.analyse on the final campaign, output directory) for a
    campaign figure script; PENDING while the campaign is not extracted.

    --runs / --measures-root / --out-dir exist to TEST a campaign script on other runs (e.g. the pilot's,
    with --measures-root results/E3_worldmodel_4zones/gpu_pilot_remeasure). Such a test must write elsewhere than
    paper/figures/out/: the report never shows a campaign figure made from other runs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runs", type=Path, nargs="+", default=None, help="TEST ONLY: run directories other than the campaign's")
    parser.add_argument("--measures-root", type=Path, default=None, help="TEST ONLY: checkpoint measures under DIR/<run>/env<NNNNNN>/")
    parser.add_argument("--out-dir", type=Path, default=None, help="TEST ONLY: where to write the PDF (never paper/figures/out)")
    args = parser.parse_args(argv)
    if args.runs is None:
        if args.measures_root is not None or args.out_dir is not None:
            raise SystemExit("--measures-root and --out-dir are for tests with --runs only")
        runs, measures_root, out_dir = campaign_runs(), None, common.OUT_DIR
    else:
        if args.out_dir is None or args.out_dir.resolve() == common.OUT_DIR.resolve():
            raise SystemExit("a test on other runs (--runs) needs an --out-dir other than paper/figures/out")
        runs, measures_root, out_dir = args.runs, args.measures_root, args.out_dir
    analysis = e3_module("campaign_analysis.py")
    aggregate_b = e3_module("aggregate_b.py")
    return analysis.analyse(runs, measures_root, aggregate_b.load_e30_control()), out_dir


def budget_checkpoint(pair: dict, budget: int) -> dict:
    """The secondary analyses of a pair at the budget's checkpoint (an error if it was not measured in both runs)."""
    matches = [c for c in pair["checkpoints"] if c["env_step"] == budget]
    if len(matches) != 1:
        raise SystemExit(f"seed {pair['seed']}: the budget's checkpoint {budget} is not measured in both runs")
    return matches[0]


def run_identity(run_dir: Path) -> tuple[int, str]:
    """(seed, condition) as the run recorded them."""
    res = read_json(run_dir / "results.json")
    return int(res["seed"]), res["condition"]


def run_label(run_dir: Path) -> str:
    seed, condition = run_identity(run_dir)
    return f"{CONDITION_ABBR[condition]} {seed}"
