"""Etape B figures (spec 3.8). Reads the measurement/permutation
artifacts written by run_etape_b.py (never recomputes what they already
hold), except the heatmaps, which need per-state arrays the zone-
aggregated CSV doesn't carry -- those are recomputed once, at one
representative (budget, seed), by re-running the same collection/
estimation/measurement functions the main pipeline uses (not a second,
hand-written approximation).

CI bands on the n-curves are mean +/- 1.96*SEM across seeds (a
simple normal-approximation CI, not a bootstrap -- documented here since
the spec says "CI bands" without prescribing a method; the tolerance
check itself, spec 3.7's "I* inside the bootstrap CI", is verified
separately in tests, not by this figure).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact import load_measurement_artifact, load_permutation_artifact
from env import build_control_layout, build_env, build_layout, reward_and_terminal
from estimation import agent_from_counts, collect_generative, spawn_exp_ctrl_rngs
from measurement import compute_state_and_pair_arrays

from helplessness.config import load_yaml_block

CFG = load_yaml_block("zones/e1_4zones.yaml")
GRID_SHAPE = tuple(CFG["grid_shape"])
DIRECTION = "away_from_goal"
GAMMA = 0.9
GOAL_REWARD = 50.0
BUDGETS = (10, 30, 100, 300, 1000)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

ZONE_ORDER = ("uncontrollable_deterministic", "controllable_noisy", "controllable_deterministic", "uncontrollable_noisy")
ZONE_COLORS = {
    "uncontrollable_deterministic": "tab:red",
    "controllable_noisy": "tab:orange",
    "controllable_deterministic": "tab:blue",
    "uncontrollable_noisy": "tab:purple",
}


def _mean_sem_by_budget(rows, zone, column, env="exp", region="all"):
    means, sems = [], []
    for n in BUDGETS:
        vals = [r[column] for r in rows if r["zone"] == zone and r["n_budget"] == n and r["env"] == env and r["region"] == region]
        vals = np.array(vals, dtype=float)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(sems)


def plot_i_hat_vs_asymptote(rows, out_dir: Path = ARTIFACTS_DIR) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for zone in ZONE_ORDER:
        means, sems = _mean_sem_by_budget(rows, zone, "I_hat")
        i_star = next(r["I_star"] for r in rows if r["zone"] == zone and r["env"] == "exp" and r["region"] == "all" and r["n_budget"] == 1000)
        ax.errorbar(BUDGETS, means, yerr=1.96 * sems, marker="o", label=zone, color=ZONE_COLORS[zone])
        ax.axhline(i_star, color=ZONE_COLORS[zone], linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("n (budget per (s,a) pair)")
    ax.set_ylabel("Ihat (bits), mean +/- 1.96*SEM over seeds")
    ax.set_title("Ihat_zone(n) vs analytic asymptote (dotted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "i_hat_vs_n.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_b_vs_prediction(rows, out_dir: Path = ARTIFACTS_DIR) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for zone in ZONE_ORDER:
        means, sems = _mean_sem_by_budget(rows, zone, "b")
        ax.errorbar(BUDGETS, means, yerr=1.96 * sems, marker="o", label=zone, color=ZONE_COLORS[zone])
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axhline(-np.log2(5), color="tab:red", linestyle=":", linewidth=1, alpha=0.6, label="predicted: -log2(5)")
    ax.set_xscale("log")
    ax.set_xlabel("n (budget per (s,a) pair)")
    ax.set_ylabel("b = Ihat_exp - Ihat_ctrl (bits)")
    ax.set_title("b_zone(n) vs prediction")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "b_vs_n.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_delta_and_l1(rows, out_dir: Path = ARTIFACTS_DIR) -> Path:
    """Delta and L1 are in different units (Delta: value, up to ~50; L1:
    probability mass, bounded by 2) and are plotted on SEPARATE axes
    deliberately -- any apparent crossing between their raw curves would
    be a function of R's scale, not a real relationship, so this figure
    does not claim or annotate one (see report.py::delta_l1_reading for
    the correction and the unit-free reading it replaces the crossing
    claim with: Delta reaches its numerical floor while L1 is still
    substantially non-zero). Both deterministic zones have Delta==0 at
    every budget trivially (exact transitions from n=1).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for zone in ZONE_ORDER:
        means_d, sems_d = _mean_sem_by_budget(rows, zone, "delta", region="interior")
        axes[0].errorbar(BUDGETS, means_d, yerr=1.96 * sems_d, marker="o", label=zone, color=ZONE_COLORS[zone])
        means_l1, sems_l1 = _mean_sem_by_budget(rows, zone, "l1")
        axes[1].errorbar(BUDGETS, means_l1, yerr=1.96 * sems_l1, marker="o", label=zone, color=ZONE_COLORS[zone])
    for ax, title, ylabel in (
        (axes[0], "Delta_zone(n) (interior) -- deterministic zones exact at every n, see report.py", "Delta"),
        (axes[1], "L1_zone(n)", "||That - P||_1"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("n (budget per (s,a) pair)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].legend(fontsize=7)
    fig.suptitle("Separate axes, different units -- no crossing point is claimed between them (see report.py)")
    fig.tight_layout()
    path = out_dir / "delta_and_l1_vs_n.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_permutation_histogram(zone: str = "uncontrollable_noisy", n_budget: int = 100, seed: int = 0, out_dir: Path = ARTIFACTS_DIR) -> Path:
    """Recomputes ONE (shuffle, bootstrap) null pair (not stored in the
    summary CSV, which only keeps p95/observed/detectable) to show both
    histograms overlaid -- same RNG construction and category ordering as
    run_etape_b.py, same seed, so this reproduces exactly the nulls that
    produced those rows' p95. Shows both methods explicitly since they
    answer different questions (shuffle: exact conditional test of
    H0: I=0; bootstrap: unconditional fresh-dataset precision -- see
    permutation_null.py's module docstring), not because one corrects
    the other."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from permutation_null import bootstrap_null_i_zone, permutation_null_i_zone
    from run_etape_b import ALL_ZONE_CATEGORIES

    exp_layout = build_layout(CFG, DIRECTION)
    P_exp = build_env(exp_layout).transition_tensor()
    zone_obj = next(z for z in exp_layout.zones if z.appearance.category == zone)
    idx = np.flatnonzero(zone_obj.mask)

    N_exp = collect_generative(P_exp, n_budget, np.random.default_rng(seed))
    category_index = ALL_ZONE_CATEGORIES.index(zone)
    null_shuffle = permutation_null_i_zone(N_exp, idx, 500, np.random.default_rng((seed, n_budget, category_index, 1, 0)))
    null_boot = bootstrap_null_i_zone(N_exp, idx, 500, np.random.default_rng((seed, n_budget, category_index, 1, 1)))

    R, terminal = reward_and_terminal(CFG, goal_reward=GOAL_REWARD)
    agent = agent_from_counts(N_exp, R, alpha=0.0)
    arrays = compute_state_and_pair_arrays(P_exp, R, terminal, GAMMA, agent.T_hat, agent.R_hat)
    observed = float(arrays["I_hat"][idx].mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_shuffle, bins=30, color="lightgray", edgecolor="black", alpha=0.6, label="shuffle null (hypothesis test)")
    ax.hist(null_boot, bins=30, color="tab:blue", edgecolor="black", alpha=0.4, label="bootstrap null (detectability)")
    ax.axvline(observed, color="tab:red", linewidth=2, label=f"observed Ihat = {observed:.4f}")
    ax.axvline(np.percentile(np.abs(null_shuffle), 95), color="gray", linestyle="--", label="shuffle 95th pct")
    ax.axvline(np.percentile(np.abs(null_boot), 95), color="tab:blue", linestyle="--", label="bootstrap 95th pct")
    ax.set_xlabel("Ihat under the null (bits)")
    ax.set_ylabel("count (500 permutations/bootstraps)")
    ax.set_title(f"Permutation vs bootstrap null: {zone}, n={n_budget}, seed={seed}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "permutation_null_histogram.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_heatmaps(n_budget: int = 1000, seed: int = 0, out_dir: Path = ARTIFACTS_DIR) -> Path:
    """12x12 heatmaps of Ihat, b, Delta, V* at one representative
    (budget, seed) -- per-state arrays the zone-aggregated CSV doesn't
    carry, recomputed here via the same functions the main pipeline uses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp_layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(exp_layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    R, terminal = reward_and_terminal(CFG, goal_reward=GOAL_REWARD)

    rng_exp, rng_ctrl = spawn_exp_ctrl_rngs(seed)
    N_exp = collect_generative(P_exp, n_budget, rng_exp)
    N_ctrl = collect_generative(P_ctrl, n_budget, rng_ctrl)
    agent_exp = agent_from_counts(N_exp, R, alpha=0.0)
    agent_ctrl = agent_from_counts(N_ctrl, R, alpha=0.0)
    arrays_exp = compute_state_and_pair_arrays(P_exp, R, terminal, GAMMA, agent_exp.T_hat, agent_exp.R_hat)
    arrays_ctrl = compute_state_and_pair_arrays(P_ctrl, R, terminal, GAMMA, agent_ctrl.T_hat, agent_ctrl.R_hat)

    from helplessness.planning.value_iteration import value_iteration

    V_star, _ = value_iteration(P_exp, R, terminal, GAMMA)

    b_state = arrays_exp["I_hat"] - arrays_ctrl["I_hat"]

    fields = {
        "Ihat (exp)": arrays_exp["I_hat"].reshape(GRID_SHAPE),
        "b = Ihat_exp - Ihat_ctrl": b_state.reshape(GRID_SHAPE),
        "Delta": arrays_exp["delta"].reshape(GRID_SHAPE),
        "V*": V_star.reshape(GRID_SHAPE),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    zone_edge_masks = [z.mask for z in exp_layout.zones if z.appearance.category != "corridor"]
    for ax, (title, field) in zip(axes, fields.items()):
        im = ax.imshow(field, cmap="viridis")
        for mask in zone_edge_masks:
            ax.contour(mask.astype(int), levels=[0.5], colors="white", linewidths=1.2)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"n={n_budget}, seed={seed} (zone boundaries overlaid)")
    fig.tight_layout()
    path = out_dir / "heatmaps.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    rows = load_measurement_artifact(ARTIFACTS_DIR / "measurements.csv")
    p1 = plot_i_hat_vs_asymptote(rows)
    p2 = plot_b_vs_prediction(rows)
    p3 = plot_delta_and_l1(rows)
    p4 = plot_permutation_histogram()
    p5 = plot_heatmaps()
    for p in (p1, p2, p3, p4, p5):
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
