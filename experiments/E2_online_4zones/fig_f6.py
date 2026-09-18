"""F6 -- epsilon=1 bridge: paired diff vs epsilon=0.1, ONE PANEL PER ZONE
TYPE (never averaged across zones -- an earlier version pooled zones and
diluted an effect that only lives in the controllable-noisy zone, per
review), count/visitation comparison, and a direct comparison to a
regenerated E1-style curve (generative, uniform n per pair, alpha=1 --
NEVER E1's own alpha=0 artifacts, per the review note), averaged over 20
independent generative draws per n with its own 95% CI band, plus the
per-zone gap between each bridge seed and the curve with its paired
test."""

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
    """One figure per quantity (b, delta, l1), ONE PANEL PER ZONE TYPE --
    never pooled across zones (pooling washed out an effect that only
    lives in controllable-noisy, per review note)."""
    csv_rows = []
    panels = [("b", per_state, "b̂ (online − matched)"), ("delta", per_state, "Δ (online − matched)"), ("l1", per_pair, "L1 (online − matched)")]

    for field, source, ylabel in panels:
        fig, axes = plt.subplots(1, len(ad.ZONE_CATEGORIES), figsize=(24, 4.2), sharey=True)
        for ax, zone in zip(axes, ad.ZONE_CATEGORIES):
            fg.setup_log_x_axis(ax)
            ax.set_title(fg.ZONE_LABELS[zone])
            for epsilon, style in ((0.1, {"linestyle": "-", "marker": "o"}), (1.0, {"linestyle": "--", "marker": "s"})):
                diffs_by_cp = fg.paired_diff_by_checkpoint(source, zone, ALPHA_REF, epsilon, field, ad.CHECKPOINTS)
                xs, means, los, his = [], [], [], []
                for cp in ad.CHECKPOINTS:
                    d = diffs_by_cp[cp]
                    mean, lo, hi = su.mean_ci95(d)
                    xs.append(cp)
                    means.append(mean)
                    los.append(lo)
                    his.append(hi)
                    csv_rows.append({
                        "quantity": field, "zone": zone, "epsilon": epsilon, "checkpoint": cp,
                        "n_seeds": len(d), "mean_diff": mean, "ci95_low": lo, "ci95_high": hi,
                    })
                ax.plot(xs, means, color="black", label=f"ε={epsilon}", markersize=5, **style)
                ax.fill_between(xs, los, his, alpha=0.12, color="black")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"F6a — paired difference (online − matched), ε=0.1 vs ε=1 (α={ALPHA_REF}), by zone type -- {field}")
        fig.tight_layout()
        fg.save_figure(fig, f"F6a_epsilon_bridge_paired_diff_{field}")
    fg.save_csv(
        csv_rows, ["quantity", "zone", "epsilon", "checkpoint", "n_seeds", "mean_diff", "ci95_low", "ci95_high"],
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
    ax.set_ylabel("N per pair (median, p10-p90)")
    ax.set_title(f"F6b — N distribution by zone type, ε=0.1 vs ε=1 (final checkpoint, α={ALPHA_REF})")
    ax.legend()
    fig.tight_layout()
    fg.save_figure(fig, "F6b_visitation_comparison")
    fg.save_csv(csv_rows, ["zone", "epsilon", "median_N", "p10_N", "p90_N", "mean_N"], "F6b_visitation_comparison")


N_CURVE_DRAWS = 20
CURVE_SEED = 12345


def make_f6_e1_comparison(per_state: np.ndarray, per_pair: np.ndarray) -> None:
    """Regenerates the generative (uniform n per pair) b_hat-vs-n curve
    at alpha=1 (never E1's own alpha=0 artifacts, per the review note),
    from N_CURVE_DRAWS=20 INDEPENDENT generative draws per n (never a
    single realisation -- each draw uses its own spawned RNG stream, per
    project convention), reports the curve's own 95% CI band, then places
    the epsilon=1 bridge's own (mean N, b_hat) points on it, per zone
    type, and reports the per-seed gap between the bridge point and the
    curve's MEAN at that seed's own n (linear interpolation in log n),
    with the paired (one value per bridge seed) mean/CI/Wilcoxon/effect
    size of that gap -- the same paired_comparison primitive stats_utils
    uses everywhere else."""
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

    zone_of_state = ad.zone_of_state_for_rotation(0)  # identity assignment, matches build_layout()'s own default
    zone_state_idx = {
        zone: np.flatnonzero(zone_of_state == zone)[np.flatnonzero(zone_of_state == zone) != ad.GOAL_S]
        for zone in ad.ZONE_CATEGORIES
    }

    draw_seeds = np.random.SeedSequence(CURVE_SEED).spawn(N_CURVE_DRAWS)
    # curve_draws[zone][n] = list of per-draw b means (len N_CURVE_DRAWS)
    curve_draws = {zone: {n: [] for n in n_grid} for zone in ad.ZONE_CATEGORIES}

    for seed_seq in draw_seeds:
        rng = np.random.default_rng(seed_seq)
        for n in n_grid:
            N_target = np.full((N_STATES, N_ACTIONS), n, dtype=np.int64)
            N = matched_mod.collect_matched(P_true, N_target, rng)
            result = twin_mod.twin_from_counts(
                N=N, alpha=ALPHA_REF, support=support, P_true=P_true, R_true=R_true, terminal=terminal,
                P_ref=P_ref, gamma=GAMMA, reference_rng=rng,
            )
            for zone in ad.ZONE_CATEGORIES:
                curve_draws[zone][n].append(float(result["b"][zone_state_idx[zone]].mean()))

    curve_mean = {zone: {} for zone in ad.ZONE_CATEGORIES}
    curve_lo = {zone: {} for zone in ad.ZONE_CATEGORIES}
    curve_hi = {zone: {} for zone in ad.ZONE_CATEGORIES}
    for zone in ad.ZONE_CATEGORIES:
        for n in n_grid:
            mean, lo, hi = su.mean_ci95(np.array(curve_draws[zone][n]))
            curve_mean[zone][n] = mean
            curve_lo[zone][n] = lo
            curve_hi[zone][n] = hi

    fig, axes = plt.subplots(1, len(ad.ZONE_CATEGORIES), figsize=(22, 4.2), sharey=False)
    csv_rows = []
    gap_rows = []
    gap_summary_rows = []

    for ax, zone in zip(axes, ad.ZONE_CATEGORIES):
        ax.set_xscale("log")
        ax.set_title(fg.ZONE_LABELS[zone])
        means = [curve_mean[zone][n] for n in n_grid]
        los = [curve_lo[zone][n] for n in n_grid]
        his = [curve_hi[zone][n] for n in n_grid]
        ax.plot(n_grid, means, color="black", linestyle="-", marker="", label=f"regenerated curve (α=1, {N_CURVE_DRAWS} draws)")
        ax.fill_between(n_grid, los, his, color="black", alpha=0.15)
        for n, mean, lo, hi in zip(n_grid, means, los, his):
            csv_rows.append({
                "zone": zone, "source": "regenerated_curve", "n_or_seed": n,
                "n_draws": N_CURVE_DRAWS, "b": mean, "ci95_low": lo, "ci95_high": hi,
            })

        pair_mask = (per_pair["zone"] == zone) & mask_eps1_pair
        state_mask = (per_state["zone"] == zone) & mask_eps1_state
        seeds = np.unique(per_state["seed"][state_mask])
        bridge_n, bridge_b, gaps = [], [], []
        log_n_grid = np.log(np.array(n_grid, dtype=float))
        curve_mean_arr = np.array(means)
        for s in seeds:
            mean_n = float(per_pair["N"][pair_mask & (per_pair["seed"] == s)].mean())
            mean_b = float(per_state["b"][state_mask & (per_state["seed"] == s)].mean())
            bridge_n.append(mean_n)
            bridge_b.append(mean_b)
            interp_b = float(np.interp(np.log(max(mean_n, 1e-9)), log_n_grid, curve_mean_arr))
            gap = mean_b - interp_b
            gaps.append(gap)
            gap_rows.append({
                "zone": zone, "seed": int(s), "bridge_mean_N": mean_n, "bridge_b": mean_b,
                "curve_mean_b_at_same_n": interp_b, "gap": gap,
            })
            csv_rows.append({
                "zone": zone, "source": f"bridge_seed{s}", "n_or_seed": mean_n,
                "n_draws": 1, "b": mean_b, "ci95_low": float("nan"), "ci95_high": float("nan"),
            })
        gap_stats = su.paired_comparison(np.array(gaps))
        gap_summary_rows.append({
            "zone": zone, "n": gap_stats["n"], "mean_gap": gap_stats["mean"],
            "ci95_low": gap_stats["ci_low"], "ci95_high": gap_stats["ci_high"],
            "wilcoxon_stat": gap_stats["wilcoxon_stat"], "wilcoxon_p": gap_stats["wilcoxon_p"],
            "cohens_d": gap_stats["cohens_d"],
        })
        ax.scatter(bridge_n, bridge_b, color="black", edgecolor="black", zorder=3, label="ε=1 bridge (per seed)", s=20)
        ax.set_xlabel("n (mean count per pair)")
    axes[0].set_ylabel("b̂ (bits)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"F6c — ε=1 bridge vs regenerated curve (generative, α={ALPHA_REF})")
    fig.tight_layout()
    fg.save_figure(fig, "F6c_e1_bridge_comparison")
    fg.save_csv(
        csv_rows, ["zone", "source", "n_or_seed", "n_draws", "b", "ci95_low", "ci95_high"],
        "F6c_e1_bridge_comparison_curve",
    )
    fg.save_csv(
        gap_rows, ["zone", "seed", "bridge_mean_N", "bridge_b", "curve_mean_b_at_same_n", "gap"],
        "F6c_e1_bridge_comparison_gap",
    )
    fg.save_csv(
        gap_summary_rows,
        ["zone", "n", "mean_gap", "ci95_low", "ci95_high", "wilcoxon_stat", "wilcoxon_p", "cohens_d"],
        "F6c_e1_bridge_comparison_gap_summary",
    )
