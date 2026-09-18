"""F6 -- epsilon=1 bridge: paired diff vs epsilon=0.1 (F2-style overlay),
count/visitation comparison, and a direct comparison to a regenerated
E1-style curve (generative, uniform n per pair, alpha=1 -- NEVER E1's
own alpha=0 artifacts, per the review note)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import env as e2_env
import figures as fg
import matched_control as matched_mod
import stats_utils as su
import twin as twin_mod

from helplessness.config import load_yaml_block
from helplessness.envs.zones import geographic_support_mask

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5
GAMMA = 0.9
GOAL_REWARD = 50.0
ALPHA_REF = 1.0


def make_f6_paired_diff_overlay(per_state: np.ndarray, per_pair: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    csv_rows = []
    panels = [("b", per_state, "b̂ online − apparié"), ("delta", per_state, "Δ online − apparié"), ("l1", per_pair, "L1 online − apparié")]

    for ax, (field, source, ylabel) in zip(axes, panels):
        fg.setup_log_x_axis(ax)
        ax.set_ylabel(ylabel)
        for epsilon, style in ((0.1, {"linestyle": "-"}), (1.0, {"linestyle": "--"})):
            # average across zones (one line per epsilon, since F2 already
            # shows the per-zone breakdown at epsilon=0.1)
            xs, means, los, his = [], [], [], []
            for cp in ad.CHECKPOINTS:
                per_zone_diffs = []
                for zone in ad.ZONE_CATEGORIES:
                    d = fg.paired_diff_by_checkpoint(source, zone, ALPHA_REF, epsilon, field, (cp,))[cp]
                    per_zone_diffs.append(d)
                # concatenate per-seed diffs across zones is wrong (would
                # pool zones); instead average the per-zone means, one
                # value per seed where present -- use zone=all via mean
                # of per-zone seed-means where seeds align (all zones
                # share the same seed set at a given condition)
                max_len = max(len(d) for d in per_zone_diffs)
                stacked = np.array([d for d in per_zone_diffs if len(d) == max_len])
                seed_means = stacked.mean(axis=0)
                mean, lo, hi = su.mean_ci95(seed_means)
                xs.append(cp)
                means.append(mean)
                los.append(lo)
                his.append(hi)
                csv_rows.append({
                    "quantity": field, "epsilon": epsilon, "checkpoint": cp,
                    "n_seeds": len(seed_means), "mean_diff_across_zones": mean, "ci95_low": lo, "ci95_high": hi,
                })
            ax.plot(xs, means, color="black", marker="o" if epsilon == 0.1 else "s",
                     label=f"ε={epsilon}", **style, markersize=5)
            ax.fill_between(xs, los, his, alpha=0.12, color="black")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"F6a — écart apparié online − apparié, ε=0.1 vs ε=1 (α={ALPHA_REF}, moyenne sur zones)")
    fig.tight_layout()
    fg.save_figure(fig, "F6a_epsilon_bridge_paired_diff")
    fg.save_csv(
        csv_rows, ["quantity", "epsilon", "checkpoint", "n_seeds", "mean_diff_across_zones", "ci95_low", "ci95_high"],
        "F6a_epsilon_bridge_paired_diff",
    )


def make_f6_visitation_comparison(per_pair: np.ndarray) -> None:
    """N distribution comparison between epsilon=1 and epsilon=0.1, at
    the final checkpoint, per zone: mean/median/p90 of N per pair."""
    fig, ax = plt.subplots(figsize=(9, 5))
    csv_rows = []
    x = np.arange(len(ad.ZONE_CATEGORIES))
    width = 0.35
    for i, (epsilon, offset) in enumerate(((0.1, -width / 2), (1.0, width / 2))):
        medians, p10s, p90s = [], [], []
        for zone in ad.ZONE_CATEGORIES:
            mask = (
                (per_pair["zone"] == zone) & (per_pair["condition"] == "online")
                & (per_pair["alpha"] == ALPHA_REF) & (per_pair["epsilon"] == epsilon)
                & (per_pair["checkpoint"] == ad.FINAL_CHECKPOINT)
            )
            vals = per_pair["N"][mask].astype(float)
            medians.append(np.median(vals))
            p10s.append(np.percentile(vals, 10))
            p90s.append(np.percentile(vals, 90))
            csv_rows.append({
                "zone": zone, "epsilon": epsilon, "median_N": np.median(vals),
                "p10_N": np.percentile(vals, 10), "p90_N": np.percentile(vals, 90),
                "mean_N": float(vals.mean()),
            })
        medians = np.array(medians)
        yerr = [medians - np.array(p10s), np.array(p90s) - medians]
        ax.bar(x + offset, medians, width, yerr=yerr, capsize=3,
               color="white" if epsilon == 0.1 else "0.6", edgecolor="black",
               label=f"ε={epsilon}")
    ax.set_xticks(x)
    ax.set_xticklabels([fg.ZONE_LABELS[z] for z in ad.ZONE_CATEGORIES], rotation=20, ha="right")
    ax.set_ylabel("N par couple (médiane, p10-p90)")
    ax.set_title(f"F6b — distribution de N par zone, ε=0.1 vs ε=1 (checkpoint final, α={ALPHA_REF})")
    ax.legend()
    fig.tight_layout()
    fg.save_figure(fig, "F6b_visitation_comparison")
    fg.save_csv(csv_rows, ["zone", "epsilon", "median_N", "p10_N", "p90_N", "mean_N"], "F6b_visitation_comparison")


def make_f6_e1_comparison(per_state: np.ndarray, per_pair: np.ndarray) -> None:
    """Regenerates the generative (uniform n per pair) b_hat-vs-n curve
    at alpha=1 (never E1's own alpha=0 artifacts, per the review note),
    then places the epsilon=1 bridge's own (mean N, b_hat) points on it,
    per zone type, and reports the gap between each bridge point and the
    curve (linear interpolation in log(n))."""
    layout = e2_env.build_layout(CFG, DIRECTION)
    ctrl_layout = e2_env.build_control_layout(CFG, DIRECTION)
    P_true = e2_env.build_env(layout, CFG, goal_reward=GOAL_REWARD).transition_tensor()
    P_ref = e2_env.build_env(ctrl_layout, CFG, goal_reward=GOAL_REWARD).transition_tensor()
    R_true, terminal = e2_env.reward_and_terminal(CFG, goal_reward=GOAL_REWARD)
    support = geographic_support_mask(GRID_SHAPE)

    mask_eps1_pair = (
        (per_pair["condition"] == "online") & (per_pair["alpha"] == ALPHA_REF)
        & (per_pair["epsilon"] == 1.0) & (per_pair["checkpoint"] == ad.FINAL_CHECKPOINT)
    )
    mask_eps1_state = (
        (per_state["condition"] == "online") & (per_state["alpha"] == ALPHA_REF)
        & (per_state["epsilon"] == 1.0) & (per_state["checkpoint"] == ad.FINAL_CHECKPOINT)
    )
    max_n_observed = int(per_pair["N"][mask_eps1_pair].max())
    n_grid = sorted(set(int(round(x)) for x in np.logspace(0, np.log10(max(max_n_observed, 10)), 12)))

    rng = np.random.default_rng(12345)
    curve = {zone: {"n": [], "b": []} for zone in ad.ZONE_CATEGORIES}
    zone_of_state = ad.zone_of_state_for_rotation(0)  # identity assignment, matches build_layout()'s own default

    for n in n_grid:
        N_target = np.full((N_STATES, N_ACTIONS), n, dtype=np.int64)
        N = matched_mod.collect_matched(P_true, N_target, rng)
        result = twin_mod.twin_from_counts(
            N=N, alpha=ALPHA_REF, support=support, P_true=P_true, R_true=R_true, terminal=terminal,
            P_ref=P_ref, gamma=GAMMA, reference_rng=rng,
        )
        for zone in ad.ZONE_CATEGORIES:
            idx = np.flatnonzero(zone_of_state == zone)
            idx = idx[idx != ad.GOAL_S]
            curve[zone]["n"].append(n)
            curve[zone]["b"].append(float(result["b"][idx].mean()))

    fig, axes = plt.subplots(1, len(ad.ZONE_CATEGORIES), figsize=(22, 4.2), sharey=False)
    csv_rows = []
    gap_rows = []

    for ax, zone in zip(axes, ad.ZONE_CATEGORIES):
        ax.set_xscale("log")
        ax.set_title(fg.ZONE_LABELS[zone])
        ax.plot(curve[zone]["n"], curve[zone]["b"], color="black", linestyle="-", marker="", label="courbe régénérée (α=1)")
        for n, b in zip(curve[zone]["n"], curve[zone]["b"]):
            csv_rows.append({"zone": zone, "source": "regenerated_curve", "n_or_seed": n, "b": b})

        pair_mask = (per_pair["zone"] == zone) & mask_eps1_pair
        state_mask = (per_state["zone"] == zone) & mask_eps1_state
        seeds = np.unique(per_state["seed"][state_mask])
        bridge_n, bridge_b = [], []
        for s in seeds:
            mean_n = float(per_pair["N"][pair_mask & (per_pair["seed"] == s)].mean())
            mean_b = float(per_state["b"][state_mask & (per_state["seed"] == s)].mean())
            bridge_n.append(mean_n)
            bridge_b.append(mean_b)
            log_n_grid = np.log(np.array(curve[zone]["n"], dtype=float))
            interp_b = float(np.interp(np.log(max(mean_n, 1e-9)), log_n_grid, curve[zone]["b"]))
            gap_rows.append({
                "zone": zone, "seed": int(s), "bridge_mean_N": mean_n, "bridge_b": mean_b,
                "curve_b_at_same_n": interp_b, "gap": mean_b - interp_b,
            })
            csv_rows.append({"zone": zone, "source": f"bridge_seed{s}", "n_or_seed": mean_n, "b": mean_b})
        ax.scatter(bridge_n, bridge_b, color="black", edgecolor="black", zorder=3, label="pont ε=1 (par graine)", s=20)
        ax.set_xlabel("n (comptage moyen par couple)")
    axes[0].set_ylabel("b̂ (bits)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"F6c — pont ε=1 vs courbe régénérée (générative, α={ALPHA_REF})")
    fig.tight_layout()
    fg.save_figure(fig, "F6c_e1_bridge_comparison")
    fg.save_csv(csv_rows, ["zone", "source", "n_or_seed", "b"], "F6c_e1_bridge_comparison_curve")
    fg.save_csv(
        gap_rows, ["zone", "seed", "bridge_mean_N", "bridge_b", "curve_b_at_same_n", "gap"],
        "F6c_e1_bridge_comparison_gap",
    )
