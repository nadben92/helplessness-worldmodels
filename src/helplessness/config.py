"""Typed configs for experiment runners.

One dataclass per experiment family, with defaults equal to the values that
were previously hard-coded in each script's argparse defaults. A runner
loads a config.yaml (falling back to these defaults when no file is given),
runs with it, and writes the resolved config back out next to its results so
that any CSV can be traced to exactly the parameters that produced it (see
CONTRIBUTING.md).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> str:
    """Resolves a possibly-relative path (typically cfg.out or
    cfg.agents_dir) against the repo root, not the process's current
    working directory.

    Every config.yaml in experiments/ writes paths like "results/foo.csv"
    on the understanding that "results/" is the one top-level directory
    (see CONTRIBUTING.md), not a path relative to wherever the script
    happens to be invoked from. Without this, running a script from inside
    its experiment directory (as every experiment README instructs) would
    silently create a second, wrong results/ next to the script.
    """
    p = Path(path)
    return str(p) if p.is_absolute() else str(REPO_ROOT / p)


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(cls, path: str | Path | None):
    """Builds a config dataclass from a YAML file, falling back to defaults
    for any field the file omits (and to the class's own defaults if path is
    None)."""
    data = _load_yaml(path) if path else {}
    return cls(**data)


def dump_config(cfg, path: str | Path) -> None:
    """Writes the resolved config next to a result file (adjacent
    config.yaml), so the CSV it produced is traceable in four months."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)


@dataclasses.dataclass
class OfflineSweepConfig:
    budget: int = 20_000
    p_values: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    )
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(10)))
    alpha: float = 0.1
    gamma: float = 0.95
    agents_dir: str = "results/agents"
    out: str = "results/offline_sweep.csv"


@dataclasses.dataclass
class EvalV2Config:
    budgets: list[int] = dataclasses.field(default_factory=lambda: [5_000, 50_000])
    p_values: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    )
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(10)))
    agents_dir: str = "results/agents"
    out: str = "results/offline_eval_v2.csv"
    n_episodes: int = 2000
    eval_start_seed: int = 777_001
    eval_tiebreak_seed: int = 777_002


@dataclasses.dataclass
class SwapsConfig:
    low_p: float = 0.0
    high_ps: list[float] = dataclasses.field(default_factory=lambda: [0.7, 0.85, 0.95])
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(10)))
    budget: int = 50_000
    agents_dir: str = "results/agents"
    out: str = "results/swaps.csv"
    n_episodes: int = 2000
    gamma: float = 0.95


@dataclasses.dataclass
class InterventionConfig:
    budgets: list[int] = dataclasses.field(default_factory=lambda: [5_000, 50_000])
    p_train_values: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    )
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(10)))
    lam_values: list[float] = dataclasses.field(
        default_factory=lambda: [-0.5, -0.25, 0.0, 0.15, 0.3, 0.45, 0.5, 0.6, 0.7, 0.85, 1.0]
    )
    agents_dir: str = "results/agents"
    out: str = "results/intervention.csv"
    n_episodes: int = 300
    eval_start_seed: int = 777_001
    eval_tiebreak_seed: int = 777_002


@dataclasses.dataclass
class RescueInjectionConfig:
    budgets: list[int] = dataclasses.field(default_factory=lambda: [5_000, 50_000])
    p_train_values: list[float] = dataclasses.field(
        default_factory=lambda: [0.7, 0.85, 0.95, 1.0]
    )
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(10)))
    mu_values: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    )
    agents_dir: str = "results/agents"
    out: str = "results/rescue_injection.csv"
    n_episodes: int = 300
    eval_start_seed: int = 777_001
    eval_tiebreak_seed: int = 777_002


@dataclasses.dataclass
class RescueDiagnosticsConfig:
    p_train_values: list[float] = dataclasses.field(
        default_factory=lambda: [0.7, 0.85, 0.95, 1.0]
    )
    lam_values: list[float] = dataclasses.field(default_factory=lambda: [-0.5, -0.25, 0.0])
    budgets: list[int] = dataclasses.field(default_factory=lambda: [5_000, 50_000])
    seeds: list[int] = dataclasses.field(default_factory=lambda: list(range(10)))
    agents_dir: str = "results/agents"
    out: str = "results/lam_policy_change_counts.csv"
