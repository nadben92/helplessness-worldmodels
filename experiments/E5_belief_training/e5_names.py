"""E5's run names and how to find a run, with the standard library only (the cluster notebook's Python runs the batch
and export scripts without the package's dependencies). e5_setup.py re-exports every name.

A run's name is runs.make_run_name's: grid12x12__2zones-<configuration>-<tag>__pair-<role>__worldmodel-agent-e5__seed<NN>__<UTC>,
tag = base | sh<f in hundredths> | imm, role = experimental | control (the immunisation run: experimental).
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_ID = "E5_belief_training"
E4_EXPERIMENT_ID = "E4_generalization_gradient"
AGENT_TYPE = "worldmodel-agent-e5"
TRAININGS = ("baseline", "shuffle", "immunisation")


def training_tag(training: str, fraction: float | None = None) -> str:
    if training == "baseline":
        return "base"
    if training == "shuffle":
        return f"sh{int(round(float(fraction) * 100)):03d}"
    if training == "immunisation":
        return "imm"
    raise ValueError(f"unknown training {training!r}")


def pair_role(training: str, condition: str | None) -> str:
    """runs.make_run_name's vocabulary: the immunisation run is the exposed member of its pair."""
    return "experimental" if training == "immunisation" or condition == "experimental" else "control"


def zone_preset(config_id: str, training: str, fraction: float | None = None) -> str:
    return f"2zones-{config_id}-{training_tag(training, fraction)}"


def run_glob(config_id: str, training: str, condition: str | None, seed: int, fraction: float | None = None) -> str:
    return f"*__{zone_preset(config_id, training, fraction)}__pair-{pair_role(training, condition)}__{AGENT_TYPE}__seed{int(seed):02d}__*"


def find_run(root: Path, config_id: str, training: str, condition: str | None, seed: int, fraction: float | None = None,
             need: str | None = None) -> Path:
    """The newest E5 run of (configuration, training, condition, seed) under <root>/E5_belief_training/; need: a file
    that must exist in it (relative path)."""
    pattern = run_glob(config_id, training, condition, seed, fraction)
    found = sorted(d for d in (Path(root) / EXPERIMENT_ID).glob(pattern) if need is None or (d / need).exists())
    if not found:
        raise FileNotFoundError(f"no {training} {condition or ''} run of {config_id} seed {seed} under {root}" + (f" with {need}" if need else ""))
    return found[-1]


def find_e4_run(root: Path, config_id: str, condition: str, need: str | None = None) -> Path:
    """The newest E4 run of a configuration under <root>/E4_generalization_gradient/ (E4's own results: its seed-A runs)."""
    role = "experimental" if condition == "experimental" else "control"
    found = sorted(d for d in (Path(root) / E4_EXPERIMENT_ID).glob(f"*__2zones-{config_id}__pair-{role}__*")
                   if need is None or (d / need).exists())
    if not found:
        raise FileNotFoundError(f"no E4 {condition} run of {config_id} under {root}" + (f" with {need}" if need else ""))
    return found[-1]
