"""E2: paired online - matched difference on b_hat vs budget, controllable-noisy
zone, reference condition (alpha=1, epsilon=0.1).

Shows each seed's own curve, the mean over seeds and its 95% CI. Checkpoints
below the power-law fit's range are shaded. Writes paper/figures/out/fig_e2_gap.pdf.

No new analysis: the per-seed differences come from E2's own primitives
(analysis_data.load_per_state, figures.paired_diff_by_checkpoint,
stats_utils.mean_ci95), the same ones that produce panel b of F2. Before
plotting, the script checks that its per-checkpoint mean matches
F2_null_model_paired_diff.csv, and stops if it does not.

Sources (read-only):
  results/E2_online_4zones_sweep/a1_e0.1_seed*/per_state.csv
  experiments/E2_online_4zones/artifacts/figures/csv/F2_null_model_paired_diff.csv
  experiments/E2_online_4zones/artifacts/figures/csv/table3_budget_trend_power_law.csv

Usage: python paper/figures/fig_e2_gap.py [--ref-slope]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import common

E2_DIR = common.REPO_ROOT / "experiments" / "E2_online_4zones"
sys.path.insert(0, str(E2_DIR))  # E2's analysis modules use bare imports

import analysis_data as ad  # noqa: E402
import figures as fg  # noqa: E402
import stats_utils as su  # noqa: E402

NAME = "fig_e2_gap"
ZONE = "controllable_noisy"
ALPHA, EPSILON = 1.0, 0.1
CONDITION_PREFIX = "a1_e0.1_seed"  # sweep.py's run-directory name for this condition
# Lowest checkpoint of the power-law fit, as in tables.py::table3_budget_trend_power_law
# (cross-checked below against that table's n_points).
FIT_MIN_CHECKPOINT = 10_000

F2_CSV = E2_DIR / "artifacts" / "figures" / "csv" / "F2_null_model_paired_diff.csv"
TABLE3_CSV = E2_DIR / "artifacts" / "figures" / "csv" / "table3_budget_trend_power_law.csv"

# Agreement with the F2 CSV, which stores Python float reprs (full precision).
RTOL, ATOL = 1e-9, 1e-12


def _read_csv(path: Path) -> list[dict]:
    with open(common.require_file(path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_dirs() -> list[Path]:
    if not ad.SWEEP_ROOT.is_dir():
        raise SystemExit(f"Missing sweep directory: {ad.SWEEP_ROOT}")
    run_dirs = [d for d in ad.all_run_dirs() if d.name.startswith(CONDITION_PREFIX)]
    if len(run_dirs) != ad.N_SEEDS:
        raise SystemExit(
            f"Expected {ad.N_SEEDS} run directories {CONDITION_PREFIX}* in {ad.SWEEP_ROOT}, found {len(run_dirs)}"
        )
    for d in run_dirs:
        common.require_file(d / "per_state.csv")
    return run_dirs


def _check_seeds_aligned(per_state: np.ndarray) -> None:
    """paired_diff_by_checkpoint returns diffs ordered by sorted seed id; a
    per-seed line is only meaningful if the same seeds are present, in both
    conditions, at every checkpoint."""
    expected = np.arange(ad.N_SEEDS)
    for condition in ("online", "matched"):
        mask = fg.zone_condition_mask(per_state, ZONE, condition, ALPHA, EPSILON)
        for cp in ad.CHECKPOINTS:
            seeds = np.unique(per_state["seed"][mask & (per_state["checkpoint"] == cp)])
            if not np.array_equal(seeds, expected):
                raise SystemExit(f"Seeds at checkpoint {cp} ({condition}) are {seeds.tolist()}, expected 0..{ad.N_SEEDS - 1}")


def _check_against_f2(checkpoints: list[int], means: np.ndarray) -> None:
    rows = [r for r in _read_csv(F2_CSV) if r["quantity"] == "b" and r["zone"] == ZONE]
    f2 = {int(r["checkpoint"]): float(r["mean_diff"]) for r in rows}
    if sorted(f2) != sorted(checkpoints):
        raise SystemExit(f"Checkpoints differ from {F2_CSV.name}: {sorted(f2)} vs {sorted(checkpoints)}")
    ref = np.array([f2[cp] for cp in checkpoints])
    if not np.allclose(means, ref, rtol=RTOL, atol=ATOL):
        worst = int(np.argmax(np.abs(means - ref)))
        raise SystemExit(
            f"Mean recomputed from the raw data does not match {F2_CSV.name} "
            f"(rtol={RTOL}, atol={ATOL}): checkpoint {checkpoints[worst]}, "
            f"recomputed {means[worst]!r}, CSV {ref[worst]!r}"
        )


def _check_fit_range(checkpoints: list[int]) -> None:
    rows = [r for r in _read_csv(TABLE3_CSV) if r["quantity"] == "b" and r["zone"] == ZONE]
    if len(rows) != 1:
        raise SystemExit(f"Expected one row quantity=b, zone={ZONE} in {TABLE3_CSV.name}, found {len(rows)}")
    n_fit = sum(cp >= FIT_MIN_CHECKPOINT for cp in checkpoints)
    if n_fit != int(rows[0]["n_points"]):
        raise SystemExit(
            f"{n_fit} checkpoints >= {FIT_MIN_CHECKPOINT}, but {TABLE3_CSV.name} reports n_points={rows[0]['n_points']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ref-slope", action="store_true",
        help="add a reference n^(-1/2) curve anchored at the mean at the first fitted checkpoint",
    )
    args = parser.parse_args()

    per_state = ad.load_per_state(_run_dirs())
    _check_seeds_aligned(per_state)

    checkpoints = list(ad.CHECKPOINTS)
    _check_fit_range(checkpoints)
    diffs_by_cp = fg.paired_diff_by_checkpoint(per_state, ZONE, ALPHA, EPSILON, "b", ad.CHECKPOINTS)
    per_seed = np.column_stack([diffs_by_cp[cp] for cp in checkpoints])  # (seeds, checkpoints)
    stats = np.array([su.mean_ci95(diffs_by_cp[cp]) for cp in checkpoints])  # (checkpoints, 3)
    means, los, his = stats[:, 0], stats[:, 1], stats[:, 2]
    _check_against_f2(checkpoints, means)

    common.apply_style()
    color = common.ZONE_COLOR[ZONE]
    xs = np.asarray(checkpoints, dtype=float)
    fig, ax = common.figure()

    ax.axvspan(xs[0], FIT_MIN_CHECKPOINT, color=common.TOL_BRIGHT["grey"], alpha=0.25, linewidth=0, zorder=0)
    ax.axhline(0.0, color="black", linewidth=0.6, zorder=1)
    for row in per_seed:
        ax.plot(xs, row, color=color, linewidth=0.5, alpha=0.35, zorder=2)
    ax.fill_between(xs, los, his, color=color, alpha=0.25, linewidth=0, zorder=3)
    ax.plot(xs, means, color=color, linewidth=1.4, marker="o", markersize=2.5, zorder=4)

    if args.ref_slope:
        i0 = checkpoints.index(FIT_MIN_CHECKPOINT)
        fit_xs = xs[i0:]
        ax.plot(fit_xs, means[i0] * (fit_xs / fit_xs[0]) ** -0.5,
                color="black", linestyle="--", linewidth=0.8, zorder=5, label=r"$\propto n^{-1/2}$")
        ax.legend(loc="lower right")

    ax.set_xscale("log")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel(r"budget $n$ (environment steps)")
    ax.set_ylabel(r"$\hat{b}^{\mathrm{on}} - \hat{b}^{\mathrm{m}}$ (bits)")

    path = common.save_pdf(fig, NAME)
    print(f"wrote {path.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
