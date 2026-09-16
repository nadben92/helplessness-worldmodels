"""Etape B -- generative collection, estimation, measurement, and the
permutation/bootstrap nulls, at the frozen config (direction=away_from_goal,
gamma=0.9, R=50 -- see config.yaml). Writes the shared output artifact
(spec 3.6) plus figures (spec 3.8).

RNG discipline: independent streams per (environment, budget, seed) for
collection, derived from the same integer seed via
estimation.spawn_exp_ctrl_rngs() -- NOT two separately-seeded Generators
with the same seed value, which turned out to stay in lockstep across the
entire state loop for zones where P_exp == P_ctrl bit-for-bit (already-
controllable zones), making b measure exactly 0 with zero variance there
by construction, not by measurement (see spawn_exp_ctrl_rngs' docstring
for the full diagnosis). A separate Generator seeds the permutation/
bootstrap nulls, never reused across budgets/seeds/zones/methods.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from artifact import write_measurement_artifact, write_permutation_artifact
from env import build_control_layout, build_env, build_layout, reward_and_terminal
from estimation import agent_from_counts, collect_generative, spawn_exp_ctrl_rngs
from measurement import aggregate_zone_region, compute_state_and_pair_arrays
from permutation_null import (
    bootstrap_null_b_zone,
    bootstrap_null_i_zone,
    detectable_effect,
    permutation_null_b_zone,
    permutation_null_i_zone,
)

from helplessness.config import load_yaml_block

CFG = load_yaml_block("zones/e1_4zones.yaml")
GRID_SHAPE = tuple(CFG["grid_shape"])
DIRECTION = "away_from_goal"
GAMMA = 0.9
GOAL_REWARD = 50.0

BUDGETS = (10, 30, 100, 300, 1000)
# 20, not 5: relative standard error of a sample sd from k seeds is
# ~1/sqrt(2*(k-1)) -- ~35% at k=5 (too noisy for the across-seed-sd
# precision analysis in report.py), ~16% at k=20. See
# docs/E1_offline_4zones.md and report.py::fit_inverse_sqrt_n.
SEEDS = tuple(range(20))
N_PERMUTATIONS = 500
REGIONS = ("interior", "boundary", "all")

# I_hat null (I*=0 predicted): the two uncontrollable zones only, per spec.
ZERO_I_STAR_ZONES = ("uncontrollable_deterministic", "uncontrollable_noisy")
# b null where H0: b=0 is actually TRUE (already-controllable zones,
# derive_control leaves dynamics unchanged) -- the real detectability
# headline. The uncontrollable zones' own b-null is also computed (kept,
# labelled) but its H0 is FALSE there (b is genuinely -log2(5) or -1.65
# bits), so its null is inflated by that real effect and is the wrong
# number to cite for detectability -- see report.py.
ALL_ZONE_CATEGORIES = ("uncontrollable_deterministic", "controllable_noisy", "controllable_deterministic", "uncontrollable_noisy")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _zone_by_category(layout, category: str):
    return next(z for z in layout.zones if z.appearance.category == category)


def _add_null_rows(permutation_rows, zone_category, quantity, n_budget, seed, category_index, null_shuffle, null_bootstrap, observed):
    for method, null in (("shuffle", null_shuffle), ("bootstrap", null_bootstrap)):
        p95 = detectable_effect(null)
        permutation_rows.append(
            {
                "zone": zone_category,
                "quantity": quantity,
                "method": method,
                "n_budget": n_budget,
                "seed": seed,
                "p95": p95,
                "observed": observed,
                "detectable": abs(observed) > p95,
            }
        )


def run() -> tuple[list[dict], list[dict]]:
    exp_layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(exp_layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    R, terminal = reward_and_terminal(CFG, goal_reward=GOAL_REWARD)

    measurement_rows: list[dict] = []
    permutation_rows: list[dict] = []

    for n_budget in BUDGETS:
        for seed in SEEDS:
            rng_exp, rng_ctrl = spawn_exp_ctrl_rngs(seed)
            N_exp = collect_generative(P_exp, n_budget, rng_exp)
            N_ctrl = collect_generative(P_ctrl, n_budget, rng_ctrl)

            agent_exp = agent_from_counts(N_exp, R, alpha=0.0)
            agent_ctrl = agent_from_counts(N_ctrl, R, alpha=0.0)
            arrays_exp = compute_state_and_pair_arrays(P_exp, R, terminal, GAMMA, agent_exp.T_hat, agent_exp.R_hat)
            arrays_ctrl = compute_state_and_pair_arrays(P_ctrl, R, terminal, GAMMA, agent_ctrl.T_hat, agent_ctrl.R_hat)

            for category in ALL_ZONE_CATEGORIES:
                zone_exp = _zone_by_category(exp_layout, category)
                zone_ctrl = _zone_by_category(ctrl_layout, category)
                zone_type = type(zone_exp.dynamics).__name__

                i_hat_exp_all = None
                b_all = None
                for region in REGIONS:
                    row_exp = aggregate_zone_region(arrays_exp, zone_exp, GRID_SHAPE, region)
                    row_ctrl = aggregate_zone_region(arrays_ctrl, zone_ctrl, GRID_SHAPE, region)

                    b = row_exp["I_hat"] - row_ctrl["I_hat"] if region == "all" else float("nan")
                    if region == "all":
                        i_hat_exp_all = row_exp["I_hat"]
                        b_all = b

                    for env_label, row in (("exp", row_exp), ("ctrl", row_ctrl)):
                        measurement_rows.append(
                            {
                                "exp": "E1",
                                "env": env_label,
                                "zone": category,
                                "zone_type": zone_type,
                                "n_budget": n_budget,
                                "seed": seed,
                                "region": region,
                                "I_hat": row["I_hat"],
                                "I_star": row["I_star"],
                                "b": b if env_label == "exp" else float("nan"),
                                "delta": row["delta"],
                                "l1": row["l1"],
                                "v_pi_true": row["v_pi_true"],
                                "v_pi_believed": row["v_pi_believed"],
                            }
                        )

                # category_index, not hash(category): Python's str hash is
                # randomised per-process (PYTHONHASHSEED) unless disabled,
                # which would make these null RNG streams non-reproducible
                # across runs -- a fixed list index is deterministic and stable.
                category_index = ALL_ZONE_CATEGORIES.index(category)
                idx = np.flatnonzero(zone_exp.mask)

                if category in ZERO_I_STAR_ZONES:
                    rng_shuffle_i = np.random.default_rng((seed, n_budget, category_index, 1, 0))
                    rng_boot_i = np.random.default_rng((seed, n_budget, category_index, 1, 1))
                    null_shuffle_i = permutation_null_i_zone(N_exp, idx, N_PERMUTATIONS, rng_shuffle_i)
                    null_boot_i = bootstrap_null_i_zone(N_exp, idx, N_PERMUTATIONS, rng_boot_i)
                    _add_null_rows(permutation_rows, category, "I_hat", n_budget, seed, category_index, null_shuffle_i, null_boot_i, i_hat_exp_all)

                # b-null for every zone: TRUE H0 (b=0) in the two
                # controllable zones (the headline), FALSE H0 (b is the
                # real, large effect) in the two uncontrollable zones
                # (kept, labelled as inflated -- see report.py).
                rng_shuffle_b = np.random.default_rng((seed, n_budget, category_index, 2, 0))
                rng_boot_b = np.random.default_rng((seed, n_budget, category_index, 2, 1))
                null_shuffle_b = permutation_null_b_zone(N_exp, N_ctrl, idx, N_PERMUTATIONS, rng_shuffle_b)
                null_boot_b = bootstrap_null_b_zone(N_exp, N_ctrl, idx, N_PERMUTATIONS, rng_boot_b)
                _add_null_rows(permutation_rows, category, "b", n_budget, seed, category_index, null_shuffle_b, null_boot_b, b_all)

    return measurement_rows, permutation_rows


def main() -> None:
    t0 = time.time()
    measurement_rows, permutation_rows = run()
    m_path = write_measurement_artifact(measurement_rows, ARTIFACTS_DIR / "measurements.csv")
    p_path = write_permutation_artifact(permutation_rows, ARTIFACTS_DIR / "permutation_null.csv")
    print(f"Wrote {m_path} ({len(measurement_rows)} rows)")
    print(f"Wrote {p_path} ({len(permutation_rows)} rows)")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
