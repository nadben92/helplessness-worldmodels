"""E4's batch table (docs/E4_generalization_gradient.md): four waves of four runs. A wave trains two configurations
(configs/e4/e4.yaml e4.configurations), each in its experimental (exposed) and its reference environment; the two
configurations of a wave are the two orientations of one pair on one diagonal, so every wave is balanced in
orientation and usable on its own.

  e4_wave1  pair 1, diagonal 1 -- the pilot and the campaign's first wave; the full planned budget (300,000 steps)
  e4_wave2  pair 1, diagonal 2
  e4_wave3  pair 2, diagonal 1
  e4_wave4  pair 2, diagonal 2

max_env_steps: None runs the whole planned budget (configs/e4/e4.yaml budget); waves 2-4 stop at the budget B chosen
after the pilot (pilot_analysis.py convergence), with the planned budget unchanged so that their run equals the
checkpoint B of a full run (README.md, "budget"). Until B is written here, submit_kubeflow.py refuses to train them.
"""

from __future__ import annotations

CONDITIONS = ("experimental", "reference")
CONFIG = "configs/e4/e4.yaml"
B_AFTER_PILOT = None  # environment steps, set from pilot_analysis.py convergence (budget_choice.json "B")

WAVES = {
    "e4_wave1": {"configurations": ("p1-a-d1", "p1-b-d1"), "max_env_steps": None, "full_budget": True},
    "e4_wave2": {"configurations": ("p1-a-d2", "p1-b-d2"), "max_env_steps": B_AFTER_PILOT, "full_budget": False},
    "e4_wave3": {"configurations": ("p2-a-d1", "p2-b-d1"), "max_env_steps": B_AFTER_PILOT, "full_budget": False},
    "e4_wave4": {"configurations": ("p2-a-d2", "p2-b-d2"), "max_env_steps": B_AFTER_PILOT, "full_budget": False},
}
BATCH_CPUS = {name: 3 for name in WAVES}  # as E3's campaign: 4 processes, one thread each, on the job's 3 CPUs


def batch_runs(batch: str) -> dict[str, tuple[str, str]]:
    """job name -> (configuration, condition), the experimental runs first."""
    if batch not in WAVES:
        raise ValueError(f"unknown batch {batch!r}; choose among {list(WAVES)}")
    return {f"{cid}_{condition}": (cid, condition) for condition in CONDITIONS for cid in WAVES[batch]["configurations"]}


def max_env_steps(batch: str) -> int | None:
    """The wave's stopping point (None = the full planned budget); raises for a wave whose B is not set yet."""
    wave = WAVES[batch]
    if wave["full_budget"]:
        return None
    if wave["max_env_steps"] is None:
        raise ValueError(f"{batch}: its budget B is set after the pilot (batches.B_AFTER_PILOT), not yet")
    return int(wave["max_env_steps"])
