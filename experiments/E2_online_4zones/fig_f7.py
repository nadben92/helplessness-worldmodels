"""F7 -- reference visitation profile: counts by zone/checkpoint
(max/min per state, quantiles), departure share by zone/count-bin, and
episode length/truncation by starting zone (reusing episode_analysis.py,
never recomputed)."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import episode_analysis
import figures as fg

COUNT_BINS = (("N<10", 0, 10), ("10<=N<100", 10, 100), ("N>=100", 100, np.inf))


def _reference_run_dirs():
    return [d for d in ad.all_run_dirs() if d.name.startswith(f"{ad.REFERENCE_CONDITION}_seed")]


def make_f7_counts_profile() -> None:
    run_dirs = _reference_run_dirs()
    fig, axes = plt.subplots(1, len(ad.ZONE_CATEGORIES), figsize=(20, 4.2), sharey=True)
    csv_rows = []

    zone_state_indices = {}
    for zone in ad.ZONE_CATEGORIES:
        zone_state_indices[zone] = {}

    for ax, zone in zip(axes, ad.ZONE_CATEGORIES):
        fg.setup_log_x_axis(ax)
        ax.set_yscale("log")
        ax.set_title(fg.ZONE_LABELS[zone])
        max_by_cp, min_by_cp, q50_by_cp = [], [], []
        for cp in ad.CHECKPOINTS:
            per_run_max, per_run_min, per_run_all = [], [], []
            for run_dir in run_dirs:
                summary = json.loads((run_dir / "summary.json").read_text())
                rotation = summary["assignment"]
                zone_of_state = ad.zone_of_state_for_rotation(rotation)
                idx = np.flatnonzero(zone_of_state == zone)
                idx = idx[idx != ad.GOAL_S]
                with np.load(run_dir / "arrays.npz") as arrays:
                    N_pair = arrays[f"N_checkpoint_{cp}"][idx].sum(axis=2)  # (n_states_in_zone, A)
                per_run_max.append(N_pair.max(axis=1))
                per_run_min.append(N_pair.min(axis=1))
                per_run_all.append(N_pair.ravel())
            all_max = np.concatenate(per_run_max)
            all_min = np.concatenate(per_run_min)
            all_vals = np.concatenate(per_run_all)
            max_by_cp.append(all_max.mean())
            min_by_cp.append(all_min.mean())
            q50_by_cp.append(np.median(all_vals))
            csv_rows.append({
                "zone": zone, "checkpoint": cp,
                "mean_max_per_state": all_max.mean(), "mean_min_per_state": all_min.mean(),
                "median_N": np.median(all_vals), "q10_N": np.percentile(all_vals, 10),
                "q90_N": np.percentile(all_vals, 90),
            })
        ax.plot(ad.CHECKPOINTS, max_by_cp, color="black", linestyle="-", marker="o", markersize=4, label="max/état (moy.)")
        ax.plot(ad.CHECKPOINTS, min_by_cp, color="black", linestyle="--", marker="x", markersize=4, label="min/état (moy.)")
        ax.plot(ad.CHECKPOINTS, q50_by_cp, color="gray", linestyle=":", marker="", label="médiane N")
    axes[0].set_ylabel("N (échelle log)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"F7a — profil de comptage par zone vs checkpoint ({ad.REFERENCE_CONDITION})")
    fig.tight_layout()
    fg.save_figure(fig, "F7a_counts_profile")
    fg.save_csv(
        csv_rows,
        ["zone", "checkpoint", "mean_max_per_state", "mean_min_per_state", "median_N", "q10_N", "q90_N"],
        "F7a_counts_profile",
    )


def make_f7_departure_share() -> None:
    run_dirs = _reference_run_dirs()
    fig, ax = plt.subplots(figsize=(9, 5))
    csv_rows = []
    x = np.arange(len(ad.ZONE_CATEGORIES))
    width = 0.25
    final_t = ad.FINAL_CHECKPOINT

    for i, (bin_label, lo_n, hi_n) in enumerate(COUNT_BINS):
        shares = []
        for zone in ad.ZONE_CATEGORIES:
            all_shares = []
            for run_dir in run_dirs:
                summary = json.loads((run_dir / "summary.json").read_text())
                rotation = summary["assignment"]
                zone_of_state = ad.zone_of_state_for_rotation(rotation)
                idx = np.flatnonzero(zone_of_state == zone)
                idx = idx[idx != ad.GOAL_S]
                with np.load(run_dir / "arrays.npz") as arrays:
                    N_pair = arrays[f"N_checkpoint_{final_t}"][idx].sum(axis=2)
                    N_start = arrays[f"N_start_checkpoint_{final_t}"][idx]
                mask = (N_pair >= lo_n) & (N_pair < hi_n)
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = np.where(mask & (N_pair > 0), N_start / np.maximum(N_pair, 1), np.nan)
                vals = share[mask]
                if len(vals):
                    all_shares.append(np.nanmean(vals))
            shares.append(np.mean(all_shares) if all_shares else np.nan)
            csv_rows.append({"zone": zone, "count_bin": bin_label, "mean_departure_share": shares[-1]})
        ax.bar(x + (i - 1) * width, shares, width, label=bin_label, color=["white", "0.6", "0.2"][i], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([fg.ZONE_LABELS[z] for z in ad.ZONE_CATEGORIES], rotation=20, ha="right")
    ax.set_ylabel("part des comptages venant des départs")
    ax.set_title(f"F7b — part des comptages venant des départs, par zone × tranche (checkpoint final, {ad.REFERENCE_CONDITION})")
    ax.legend()
    fig.tight_layout()
    fg.save_figure(fig, "F7b_departure_share")
    fg.save_csv(csv_rows, ["zone", "count_bin", "mean_departure_share"], "F7b_departure_share")


def make_f7_episode_profile() -> None:
    """Reuses episode_analysis.py::analyze_condition -- never recomputed."""
    report = episode_analysis.analyze_condition(ad.REFERENCE_CONDITION)
    csv_rows = []
    for zone, stats in report.items():
        csv_rows.append({
            "zone": zone, "n_episodes": stats["n_episodes"], "median_length": stats["median_length"],
            "p90_length": stats["p90_length"], "p99_length": stats["p99_length"],
            "truncation_rate": stats["truncation_rate"],
        })
    fg.save_csv(
        csv_rows, ["zone", "n_episodes", "median_length", "p90_length", "p99_length", "truncation_rate"],
        "F7c_episode_profile",
    )
    return report
