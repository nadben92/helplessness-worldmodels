"""F5 -- effect of the Dirichlet prior alpha: contrast between paired
zones vs budget (one curve per alpha), the final-checkpoint contrast
table, and b_hat by zone x alpha at low vs high pair counts."""

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
    ("controllable_deterministic", "uncontrollable_deterministic", "contrôlable-dét. − tapis roulant"),
    ("controllable_noisy", "uncontrollable_noisy", "contrôlable-bruitée − incontrôlable-bruitée"),
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
        ax.set_ylabel("contraste b̂ (bits)")
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
    fig.suptitle("F5 — contraste entre zones appariées vs budget, par α")
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


def make_f5_count_bins(per_state: np.ndarray, per_pair: np.ndarray) -> None:
    """b_hat by zone x alpha, at low (min pair count < 10) vs high
    (>= 10) count states, final checkpoint -- a state's "count level" is
    its own MIN count over actions (same convention as F4/F7's
    visitation-floor analysis)."""
    fig, axes = plt.subplots(1, len(ad.ZONE_CATEGORIES), figsize=(20, 4), sharey=True)
    csv_rows = []
    final_t = ad.FINAL_CHECKPOINT

    for ax, zone in zip(axes, ad.ZONE_CATEGORIES):
        ax.set_title(fg.ZONE_LABELS[zone])
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([str(a) for a in ALPHAS])
        ax.set_xlabel("α")
        for bin_label, lo_n, hi_n, style in (
            ("min N < 10", 0, 10, {"marker": "o", "linestyle": "-"}),
            ("min N >= 10", 10, np.inf, {"marker": "x", "linestyle": "--"}),
        ):
            means, los, his, xs = [], [], [], []
            for i, alpha in enumerate(ALPHAS):
                pair_mask = (
                    (per_pair["zone"] == zone) & (per_pair["condition"] == "online")
                    & (per_pair["alpha"] == alpha) & (per_pair["epsilon"] == EPSILON_REF)
                    & (per_pair["checkpoint"] == final_t)
                )
                state_mask = (
                    (per_state["zone"] == zone) & (per_state["condition"] == "online")
                    & (per_state["alpha"] == alpha) & (per_state["epsilon"] == EPSILON_REF)
                    & (per_state["checkpoint"] == final_t)
                )
                seeds = np.unique(per_state["seed"][state_mask])
                seed_means = []
                for s in seeds:
                    pm = pair_mask & (per_pair["seed"] == s)
                    sm = state_mask & (per_state["seed"] == s)
                    # min count per state, this seed
                    states = per_pair["state"][pm]
                    Ns = per_pair["N"][pm]
                    min_by_state = {}
                    for st, n in zip(states, Ns):
                        min_by_state[st] = min(min_by_state.get(st, n), n)
                    keep_states = {st for st, n in min_by_state.items() if lo_n <= n < hi_n}
                    if not keep_states:
                        continue
                    b_states = per_state["state"][sm]
                    b_vals = per_state["b"][sm]
                    sel = np.array([st in keep_states for st in b_states])
                    if sel.any():
                        seed_means.append(float(b_vals[sel].mean()))
                if seed_means:
                    mean, lo, hi = su.mean_ci95(np.array(seed_means))
                else:
                    mean, lo, hi = float("nan"), float("nan"), float("nan")
                xs.append(i)
                means.append(mean)
                los.append(lo)
                his.append(hi)
                csv_rows.append({
                    "zone": zone, "alpha": alpha, "count_bin": bin_label,
                    "n_seeds": len(seed_means), "mean_b": mean, "ci95_low": lo, "ci95_high": hi,
                })
            ax.errorbar(
                xs, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                color="black", capsize=3, label=bin_label, **style,
            )
    axes[0].set_ylabel("b̂ (bits)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"F5c — b̂ par zone × α, comptages faibles vs élevés (checkpoint {final_t})")
    fig.tight_layout()
    fg.save_figure(fig, "F5c_b_vs_alpha_by_count_bin")
    fg.save_csv(
        csv_rows, ["zone", "alpha", "count_bin", "n_seeds", "mean_b", "ci95_low", "ci95_high"],
        "F5c_b_vs_alpha_by_count_bin",
    )
