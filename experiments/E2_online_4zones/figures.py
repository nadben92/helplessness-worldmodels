"""Generic plotting/aggregation primitives shared by every F1-F7 figure
in analysis.py -- one implementation of "group by checkpoint x seed and
average", one of "save PNG+PDF+CSV", one styling scheme, reused
everywhere rather than rewritten per figure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import stats_utils as su

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Redundant coding (marker + linestyle), never color alone, per "palette
# lisible en noir et blanc". Colors are still assigned (readable on
# screen) but every series remains distinguishable in pure grayscale.
ZONE_STYLE = {
    "corridor": {"marker": "o", "linestyle": "-", "color": "#000000"},
    "uncontrollable_deterministic": {"marker": "s", "linestyle": "--", "color": "#4477AA"},
    "controllable_noisy": {"marker": "^", "linestyle": "-.", "color": "#EE6677"},
    "controllable_deterministic": {"marker": "D", "linestyle": ":", "color": "#228833"},
    "uncontrollable_noisy": {"marker": "v", "linestyle": (0, (3, 1, 1, 1)), "color": "#CCBB44"},
}
ZONE_LABELS = {
    "corridor": "couloir",
    "uncontrollable_deterministic": "tapis roulant",
    "controllable_noisy": "contrôlable-bruitée",
    "controllable_deterministic": "contrôlable-déterministe",
    "uncontrollable_noisy": "incontrôlable-bruitée",
}
CONDITION_STYLE = {
    "online": {"linestyle": "-", "marker": "o", "fillstyle": "full"},
    "matched": {"linestyle": "--", "marker": "o", "fillstyle": "none"},
}
ANALYTIC_B = {
    "uncontrollable_deterministic": -np.log2(5),
    "uncontrollable_noisy": -1.6529,  # E1's own analytic prediction, docs/E1_offline_4zones.md
    "controllable_noisy": 0.0,
    "controllable_deterministic": 0.0,
}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name: str) -> None:
    ensure_dirs()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_csv(rows: list[dict], columns: list[str], name: str) -> None:
    ensure_dirs()
    path = FIGURES_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def per_seed_values_by_checkpoint(
    arr: np.ndarray, mask: np.ndarray, value_field: str, checkpoints=ad.CHECKPOINTS,
) -> dict[int, np.ndarray]:
    """{checkpoint: array of per-seed means of value_field}, restricted
    to `mask` -- for each seed present, averages value_field over every
    matching row at that checkpoint (e.g. every state in a zone) FIRST,
    then the caller aggregates across seeds -- never flattening
    seeds x states into one undifferentiated pool (CLAUDE.md review note:
    "moyenne sur les 20 graines... graines individuelles visibles")."""
    out = {}
    for cp in checkpoints:
        cp_mask = mask & (arr["checkpoint"] == cp)
        seeds_present = np.unique(arr["seed"][cp_mask])
        vals = np.empty(len(seeds_present))
        for i, s in enumerate(seeds_present):
            seed_mask = cp_mask & (arr["seed"] == s)
            vals[i] = arr[value_field][seed_mask].mean()
        out[cp] = vals
    return out


def zone_condition_mask(arr: np.ndarray, zone: str, condition: str, alpha: float, epsilon: float) -> np.ndarray:
    return (
        (arr["zone"] == zone) & (arr["condition"] == condition)
        & (arr["alpha"] == alpha) & (arr["epsilon"] == epsilon)
    )


def paired_diff_by_checkpoint(
    per_state_or_pair: np.ndarray, zone: str, alpha: float, epsilon: float, value_field: str,
    checkpoints=ad.CHECKPOINTS,
) -> dict[int, np.ndarray]:
    """{checkpoint: array of per-seed (online - matched) differences},
    intra-seed (never two independent clouds compared) -- the primitive
    F2/F3/F5/F6 all reuse."""
    online_mask = zone_condition_mask(per_state_or_pair, zone, "online", alpha, epsilon)
    matched_mask = zone_condition_mask(per_state_or_pair, zone, "matched", alpha, epsilon)
    out = {}
    for cp in checkpoints:
        o_mask = online_mask & (per_state_or_pair["checkpoint"] == cp)
        m_mask = matched_mask & (per_state_or_pair["checkpoint"] == cp)
        seeds = np.unique(per_state_or_pair["seed"][o_mask])
        diffs = np.empty(len(seeds))
        for i, s in enumerate(seeds):
            o_val = per_state_or_pair[value_field][o_mask & (per_state_or_pair["seed"] == s)].mean()
            m_val = per_state_or_pair[value_field][m_mask & (per_state_or_pair["seed"] == s)].mean()
            diffs[i] = o_val - m_val
        out[cp] = diffs
    return out


def setup_log_x_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("checkpoint (pas)")
    ax.axhline(0.0, color="gray", linewidth=0.8, zorder=0)
