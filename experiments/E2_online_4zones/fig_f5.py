"""F5 -- effect of the Dirichlet prior alpha: contrast between paired
zones vs budget (one curve per alpha), the final-checkpoint contrast
table, and b_hat by zone x alpha, binned by a state's own minimum pair
count (over actions), pooling every checkpoint and every seed."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import figures as fg
import stats_utils as su

EPSILON_REF = 0.1
ALPHAS = (0.5, 1.0, 5.0)
ALPHA_LINESTYLES = {0.5: ":", 1.0: "-", 5.0: "--"}
ALPHA_MARKERS = {0.5: "v", 1.0: "o", 5.0: "^"}
CONTRASTS = (
    ("controllable_deterministic", "uncontrollable_deterministic", "controllable-det. − conveyor"),
    ("controllable_noisy", "uncontrollable_noisy", "controllable-noisy − uncontrollable-noisy"),
)


def _per_seed_zone_mean(per_state, zone, alpha, checkpoint):
    mask = (
        (per_state["zone"] == zone) & (per_state["condition"] == "online")
        & (per_state["alpha"] == alpha) & (per_state["epsilon"] == EPSILON_REF)
        & (per_state["checkpoint"] == checkpoint)
    )
    seeds = np.unique(per_state["seed"][mask])
    out = {}
    for s in seeds:
        out[int(s)] = float(per_state["b"][mask & (per_state["seed"] == s)].mean())
    return out


def make_f5_contrast(per_state: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    csv_rows = []
    table_rows = []

    for ax, (zone_a, zone_b, label) in zip(axes, CONTRASTS):
        fg.setup_log_x_axis(ax)
        ax.set_title(label)
        ax.set_ylabel("paired-zone contrast, b̂ (bits)")
        for alpha in ALPHAS:
            xs, means, los, his = [], [], [], []
            for cp in ad.CHECKPOINTS:
                means_a = _per_seed_zone_mean(per_state, zone_a, alpha, cp)
                means_b = _per_seed_zone_mean(per_state, zone_b, alpha, cp)
                common_seeds = sorted(set(means_a) & set(means_b))
                contrasts = np.array([means_a[s] - means_b[s] for s in common_seeds])
                mean, lo, hi = su.mean_ci95(contrasts)
                xs.append(cp)
                means.append(mean)
                los.append(lo)
                his.append(hi)
                csv_rows.append({
                    "contrast": label, "alpha": alpha, "checkpoint": cp,
                    "n_seeds": len(contrasts), "mean_contrast": mean, "ci95_low": lo, "ci95_high": hi,
                })
                if cp == ad.CHECKPOINTS[-1]:
                    table_rows.append({
                        "contrast": label, "alpha": alpha, "n_seeds": len(contrasts),
                        "mean_contrast_final": mean, "ci95_low": lo, "ci95_high": hi,
                    })
            ax.plot(xs, means, marker=ALPHA_MARKERS[alpha], linestyle=ALPHA_LINESTYLES[alpha],
                    color="black", label=f"α={alpha}", markersize=5)
            ax.fill_between(xs, los, his, alpha=0.1, color="black")
    axes[0].legend(fontsize=8)
    fig.suptitle("F5 — paired-zone contrast vs budget, by α")
    fig.tight_layout()
    fg.save_figure(fig, "F5a_alpha_contrast_vs_budget")
    fg.save_csv(
        csv_rows, ["contrast", "alpha", "checkpoint", "n_seeds", "mean_contrast", "ci95_low", "ci95_high"],
        "F5a_alpha_contrast_vs_budget",
    )
    fg.save_csv(
        table_rows, ["contrast", "alpha", "n_seeds", "mean_contrast_final", "ci95_low", "ci95_high"],
        "F5b_alpha_contrast_final_checkpoint_table",
    )


COUNT_BIN_EDGES = (1, 3, 10, 30, 100, 300, np.inf)
COUNT_BIN_LABELS = ("1-3", "3-10", "10-30", "30-100", "100-300", ">300")


def _min_n_per_state(states: np.ndarray, Ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-state minimum of N over actions, from a (state,
    action, N) triple list already restricted to one (condition, alpha,
    epsilon, seed, checkpoint) -- no per-row Python loop over the full
    per_pair array (that pattern made F3/F6a slow, see module history)."""
    order = np.argsort(states, kind="stable")
    states_sorted = states[order]
    Ns_sorted = Ns[order]
    uniq_states, start_idx = np.unique(states_sorted, return_index=True)
    mins = np.minimum.reduceat(Ns_sorted, start_idx)
    return uniq_states, mins


def make_f5_count_bins(per_state: np.ndarray, per_pair: np.ndarray) -> None:
    """b_hat by zone x alpha, binned by a state's OWN minimum pair count
    (over actions) -- pooling ALL checkpoints and ALL seeds, since at the
    final checkpoint alone almost no pair still has N<10 (the original,
    final-checkpoint-only version was empty for half its bins)."""
    online_mask_pair = (per_pair["condition"] == "online") & (per_pair["epsilon"] == EPSILON_REF)
    online_mask_state = (per_state["condition"] == "online") & (per_state["epsilon"] == EPSILON_REF)
    per_pair_o = per_pair[online_mask_pair]
    per_state_o = per_state[online_mask_state]

    # seed_values[(zone, alpha, bin_label)][seed] = [b, b, ...]
    seed_values: dict[tuple, dict[int, list]] = {}

    for alpha in ALPHAS:
        pair_alpha_mask = per_pair_o["alpha"] == alpha
        state_alpha_mask = per_state_o["alpha"] == alpha
        for seed in range(ad.N_SEEDS):
            pair_seed_mask = pair_alpha_mask & (per_pair_o["seed"] == seed)
            state_seed_mask = state_alpha_mask & (per_state_o["seed"] == seed)
            for cp in ad.CHECKPOINTS:
                pair_mask = pair_seed_mask & (per_pair_o["checkpoint"] == cp)
                if not pair_mask.any():
                    continue
                uniq_states, min_n = _min_n_per_state(per_pair_o["state"][pair_mask], per_pair_o["N"][pair_mask])

                state_mask = state_seed_mask & (per_state_o["checkpoint"] == cp)
                b_by_state = dict(zip(per_state_o["state"][state_mask].tolist(), per_state_o["b"][state_mask].tolist()))
                zone_by_state = dict(zip(per_state_o["state"][state_mask].tolist(), per_state_o["zone"][state_mask].tolist()))

                bin_idx = np.searchsorted(COUNT_BIN_EDGES, min_n, side="right") - 1
                for st, mn, bi in zip(uniq_states.tolist(), min_n.tolist(), bin_idx.tolist()):
                    if mn < 1 or bi >= len(COUNT_BIN_LABELS):
                        continue
                    zone = zone_by_state.get(st)
                    b_val = b_by_state.get(st)
                    if zone is None or b_val is None:
                        continue
                    key = (zone, alpha, COUNT_BIN_LABELS[bi])
                    seed_values.setdefault(key, {}).setdefault(seed, []).append(b_val)

    fig, axes = plt.subplots(1, len(ad.ZONE_CATEGORIES), figsize=(24, 4.4), sharey=True)
    csv_rows = []

    for ax, zone in zip(axes, ad.ZONE_CATEGORIES):
        ax.set_title(fg.ZONE_LABELS[zone])
        ax.set_xticks(range(len(COUNT_BIN_LABELS)))
        ax.set_xticklabels(COUNT_BIN_LABELS, rotation=30, ha="right")
        ax.set_xlabel("count bin (min N per state)")
        for alpha in ALPHAS:
            xs, means, los, his, n_obs_list = [], [], [], [], []
            for i, bin_label in enumerate(COUNT_BIN_LABELS):
                per_seed = seed_values.get((zone, alpha, bin_label), {})
                if not per_seed:
                    continue
                seed_means = np.array([float(np.mean(v)) for v in per_seed.values()])
                n_obs = sum(len(v) for v in per_seed.values())
                mean, lo, hi = su.mean_ci95(seed_means)
                xs.append(i)
                means.append(mean)
                los.append(lo)
                his.append(hi)
                n_obs_list.append(n_obs)
                csv_rows.append({
                    "zone": zone, "alpha": alpha, "count_bin": bin_label,
                    "n_seeds": len(seed_means), "n_obs": n_obs,
                    "mean_b": mean, "ci95_low": lo, "ci95_high": hi,
                })
            if xs:
                ax.errorbar(
                    xs, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                    color="black", capsize=3, marker=ALPHA_MARKERS[alpha], linestyle=ALPHA_LINESTYLES[alpha],
                    markersize=5, label=f"α={alpha}",
                )
                for x, m, n_obs in zip(xs, means, n_obs_list):
                    ax.annotate(str(n_obs), (x, m), textcoords="offset points", xytext=(0, 6),
                                fontsize=6, ha="center")
    axes[0].set_ylabel("b̂ (bits)")
    axes[0].legend(fontsize=7)
    fig.suptitle("F5c — b̂ by zone type × α, by min-pair-count bin (all checkpoints, all seeds pooled)")
    fig.tight_layout()
    fg.save_figure(fig, "F5c_b_vs_alpha_by_count_bin")
    fg.save_csv(
        csv_rows, ["zone", "alpha", "count_bin", "n_seeds", "n_obs", "mean_b", "ci95_low", "ci95_high"],
        "F5c_b_vs_alpha_by_count_bin",
    )
