"""E2's closing analysis -- relaunchable, reads only the sweep's saved
artifacts (results/E2_online_4zones_sweep/), never reruns training
(except the jumeau check's small representative sample, explicit about
why -- see phase_a.py). Produces:

  - Phase A: a status table (printed + saved as JSON), stop-and-report if
    anything fails.
  - Phase B: figures F1-F7 (PNG+PDF+CSV each) under artifacts/figures/,
    the synthesis table, and RESULTS.md.

Run: `python analysis.py` from this directory.
"""

from __future__ import annotations

import json
import time

import numpy as np

import analysis_data as ad
import fig_f1
import fig_f2
import fig_f3
import fig_f4
import fig_f5
import fig_f6
import fig_f7
import figures as fg
import phase_a
import synthesis_table
import tables
import verifications


def _json_safe(obj):
    """Recursively converts tuple dict keys (phase_a.py's status dicts
    use them, e.g. (zone, condition)) into JSON-safe strings -- the DATA
    itself is untouched, only key types, so this is purely a
    serialisation adapter, not a second copy of the status logic."""
    if isinstance(obj, dict):
        return {
            (str(k) if not isinstance(k, str) else k): _json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_phase_a(per_state: np.ndarray, per_pair: np.ndarray, run_dirs: list) -> dict:
    print("=== Phase A ===")
    status = {}

    status["b_exact_zero"] = phase_a.b_exact_zero_status(per_state)
    status["t_hat_equal_via_l1"] = phase_a.t_hat_equal_via_l1_status(per_pair)
    status["delta_nonneg"] = dict(phase_a.delta_nonneg_status(per_state))
    status["b_variance"] = phase_a.b_variance_status(per_state)
    status["n_start_leq_n"] = phase_a.n_start_leq_n_status(run_dirs)
    status["epsilon_one_no_diff"] = phase_a.epsilon_one_no_difference_status(per_state, per_pair)
    status["counterbalancing_dispersion"] = phase_a.counterbalancing_dispersion_status(per_state)

    fg.ensure_dirs()
    with open(fg.ARTIFACTS_DIR / "phase_a_status.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(status), f, indent=2, default=str)

    return status


def _all_observed(status: dict) -> bool:
    def _check(d):
        if isinstance(d, dict):
            if "status" in d:
                return d["status"].startswith("observed")
            return all(_check(v) for v in d.values())
        return True
    checks = [
        status["b_exact_zero"], status["t_hat_equal_via_l1"], status["delta_nonneg"],
        status["b_variance"], status["n_start_leq_n"],
    ]
    ok = all(_check(c) for c in checks)
    for zone_result in status["epsilon_one_no_diff"].values():
        for metric_result in zone_result.values():
            if not metric_result["status"].startswith("observed"):
                ok = False
    return ok


def run_phase_b(per_state: np.ndarray, per_pair: np.ndarray) -> None:
    print("=== Phase B ===")

    print("F1...")
    for alpha in (0.5, 1.0, 5.0):
        fig_f1.make_f1(per_state, alpha)

    print("F2...")
    fig_f2.make_f2(per_state, per_pair)

    print("F3...")
    fig_f3.make_f3(per_pair)

    print("F4...")
    fig_f4.make_f4()

    print("F5...")
    fig_f5.make_f5_contrast(per_state)
    fig_f5.make_f5_count_bins(per_state, per_pair)

    print("F6...")
    fig_f6.make_f6_paired_diff_overlay(per_state, per_pair)
    fig_f6.make_f6_visitation_comparison(per_pair)
    fig_f6.make_f6_e1_comparison(per_state, per_pair)

    print("F7...")
    fig_f7.make_f7_counts_profile()
    fig_f7.make_f7_departure_share()
    fig_f7.make_f7_episode_profile()

    print("Synthesis table...")
    synthesis_rows, _exterior_rows = synthesis_table.build_synthesis_table(per_state)

    print("Tables...")
    tables.table1_paired_diff_by_zone(per_state, per_pair)
    tables.table2_controllable_noisy_ventilated(per_pair)
    tables.table3_budget_trend_power_law(per_state, per_pair)
    tables.table4_delta_exterior_max(per_state)
    tables.table5_b_final_checkpoint_vs_analytic(synthesis_rows)
    tables.table6_contrast_budget_to_90pct(per_state)

    print("Verifications...")
    verifications.run_anisotropy_check()
    verifications.undersampled_pairs_at_final_checkpoint(per_pair)


def main() -> None:
    t0 = time.time()
    run_dirs = ad.all_run_dirs()
    print(f"{len(run_dirs)} runs found.")
    per_state = ad.load_per_state(run_dirs)
    per_pair = ad.load_per_pair(run_dirs)
    print(f"Loaded per_state={len(per_state)} rows, per_pair={len(per_pair)} rows in {time.time()-t0:.1f}s")

    status = run_phase_a(per_state, per_pair, run_dirs)
    if not _all_observed(status):
        print("Phase A: at least one status is NOT 'observed'. Stopping before Phase B.")
        return
    print("Phase A: all statuses observed. Proceeding to Phase B.")

    run_phase_b(per_state, per_pair)
    print(f"Done in {time.time()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
