"""Temps 2's report, computed from pilot.py's saved raw.npz per seed.
Facts only, no interpretation: N[s,a] distribution by zone/checkpoint,
fraction of pairs below count thresholds, the corridor's convergence
checkpoint, episode success rate over time, value-iteration cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import env as e2_env

from helplessness.config import load_yaml_block

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5

ZONE_CATEGORIES = (
    "corridor", "uncontrollable_deterministic", "controllable_noisy",
    "controllable_deterministic", "uncontrollable_noisy",
)


def _state_to_zone_category() -> np.ndarray:
    """(S,) array of zone category strings, one per state -- identity
    assignment (the pilot's own layout)."""
    layout = e2_env.build_layout(CFG, DIRECTION)
    out = np.empty(N_STATES, dtype=object)
    n_rows, n_cols = GRID_SHAPE
    for zone in layout.zones:
        for r in range(n_rows):
            for c in range(n_cols):
                if zone.mask[r, c]:
                    out[r * n_cols + c] = zone.appearance.category
    return out


STATE_ZONE = _state_to_zone_category()


def n_pair_by_checkpoint(raw: dict, checkpoint_steps) -> dict[int, np.ndarray]:
    """t -> (S, A) counts N[s,a] (summed over successors) at that checkpoint."""
    return {int(t): raw[f"N_checkpoint_{t}"].sum(axis=2) for t in checkpoint_steps}


def max_min_per_state_by_zone(N_pair: np.ndarray) -> dict[str, dict[str, float]]:
    """Per zone category: mean over states of (this state's most-visited
    action's count) and mean over states of (this state's least-visited
    action's count)."""
    out = {}
    for category in ZONE_CATEGORIES:
        idx = np.flatnonzero(STATE_ZONE == category)
        if idx.size == 0:
            continue
        per_state_max = N_pair[idx].max(axis=1)
        per_state_min = N_pair[idx].min(axis=1)
        out[category] = {
            "max_action_mean": float(per_state_max.mean()),
            "max_action_min": float(per_state_max.min()),
            "max_action_max": float(per_state_max.max()),
            "min_action_mean": float(per_state_min.mean()),
            "min_action_min": float(per_state_min.min()),
            "min_action_max": float(per_state_min.max()),
        }
    return out


def fraction_below_thresholds(N_pair: np.ndarray, thresholds=(1, 10, 100)) -> dict[str, dict[int, float]]:
    out = {}
    for category in ZONE_CATEGORIES:
        idx = np.flatnonzero(STATE_ZONE == category)
        if idx.size == 0:
            continue
        pairs = N_pair[idx]  # (n_states_in_zone, A)
        out[category] = {t: float((pairs < t).mean()) for t in thresholds}
    return out


def corridor_min_reaches_checkpoint(raw: dict, checkpoint_steps, target: float = 100.0) -> int | None:
    idx = np.flatnonzero(STATE_ZONE == "corridor")
    for t in sorted(checkpoint_steps):
        N_pair = raw[f"N_checkpoint_{t}"].sum(axis=2)
        min_count = N_pair[idx].min()
        if min_count >= target:
            return int(t)
    return None


def episode_success_rate_over_time(episode_t: np.ndarray, episode_success: np.ndarray, n_buckets: int = 10) -> list[dict]:
    if len(episode_t) == 0:
        return []
    edges = np.linspace(episode_t.min(), episode_t.max(), n_buckets + 1)
    out = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (episode_t >= lo) & (episode_t <= hi if i == n_buckets - 1 else episode_t < hi)
        n = int(mask.sum())
        rate = float(episode_success[mask].mean()) if n > 0 else float("nan")
        out.append({"t_range": (float(lo), float(hi)), "n_episodes": n, "success_rate": rate})
    return out


def value_iteration_cost(plan_n_iters: np.ndarray, elapsed_seconds: float) -> dict:
    return {
        "n_replans": int(len(plan_n_iters)),
        "mean_n_iter": float(plan_n_iters.mean()) if len(plan_n_iters) else float("nan"),
        "median_n_iter": float(np.median(plan_n_iters)) if len(plan_n_iters) else float("nan"),
        "max_n_iter": int(plan_n_iters.max()) if len(plan_n_iters) else 0,
        "min_n_iter": int(plan_n_iters.min()) if len(plan_n_iters) else 0,
        "elapsed_seconds": float(elapsed_seconds),
    }


def report_for_seed(run_dir: Path) -> dict:
    raw = np.load(run_dir / "raw.npz")
    checkpoint_steps = raw["checkpoint_steps"].tolist()

    n_pair_by_cp = n_pair_by_checkpoint(raw, checkpoint_steps)
    per_checkpoint = {}
    for t, N_pair in sorted(n_pair_by_cp.items()):
        per_checkpoint[t] = {
            "max_min_by_zone": max_min_per_state_by_zone(N_pair),
            "fraction_below": fraction_below_thresholds(N_pair),
        }

    return {
        "checkpoint_steps": sorted(checkpoint_steps),
        "per_checkpoint": per_checkpoint,
        "corridor_reaches_100_at_checkpoint": corridor_min_reaches_checkpoint(raw, checkpoint_steps, target=100.0),
        "episode_success_rate_over_time": episode_success_rate_over_time(raw["episode_t"], raw["episode_success"]),
        "value_iteration_cost": value_iteration_cost(raw["plan_n_iters"], float(raw["elapsed_seconds"])),
        "n_episodes_total": int(len(raw["episode_t"])),
        "n_episodes_success": int(raw["episode_success"].sum()),
    }


if __name__ == "__main__":
    import json
    import sys

    results_root = Path(__file__).resolve().parents[2] / "results" / "E2_online_4zones"
    seed_dirs = sorted(results_root.glob("pilot_seed*"))
    for run_dir in seed_dirs:
        report = report_for_seed(run_dir)
        print(f"=== {run_dir.name} ===")
        print(json.dumps(report, indent=2, default=str))
