"""E4's batch table (docs/E4_generalization_gradient.md). The plan is 2 colour pairs x 2 orientations = 4 configurations
on diagonal 1, i.e. 8 runs in 2 waves; each wave trains the two orientations of one pair, each in its experimental
(exposed) and its reference environment, so every wave is balanced in orientation and usable on its own.

  e4_wave1  pair 1 (p1-a-d1, p1-b-d1) -- the pilot and the campaign's first wave; the full planned budget (300,000 steps)
  e4_wave2  pair 2 (p2-a-d1, p2-b-d1) -- stopped at the budget B chosen after the pilot

The diagonal is no longer a factor: with two zones, the column reflection about the goal maps diagonal 1 onto
diagonal 2 exactly (tests/e4/test_e4_environments.py), so diagonal 2 changes the image only. Its configurations stay
defined (configs/e4/e4.yaml, own seeds) as optional REPLICAS -- the same configuration with the reflected image and a
new seed -- never part of the default waves.

max_env_steps: None runs the whole planned budget; the other batches stop at B with the planned budget unchanged, so
that their run equals the checkpoint B of a full run (README.md, "budget"). Until B is written here,
submit_kubeflow.py refuses to train them.

diagnostics_at: the environment steps where run_e4.py takes its checkpoint diagnostics (None = every checkpoint, which
the choice of B needs in wave 1). Elsewhere, only these control checkpoints and the run's last one; the weights are
still saved at every checkpoint.
"""

from __future__ import annotations

CONDITIONS = ("experimental", "reference")
CONFIG = "configs/e4/e4.yaml"
B_AFTER_PILOT = 300_000  # environment steps, set from pilot_analysis.py convergence (budget_choice.json "B")
CONTROL_CHECKPOINTS = (10000, 100000)  # diagnostics of the later batches, besides their last checkpoint

WAVES = {
    "e4_wave1": {"configurations": ("p1-a-d1", "p1-b-d1"), "full_budget": True, "diagnostics_at": None},
    "e4_wave2": {"configurations": ("p2-a-d1", "p2-b-d1"), "full_budget": False, "diagnostics_at": CONTROL_CHECKPOINTS},
}
REPLICAS = {  # optional, launched only on purpose: diagonal 2, the reflected image, new seeds
    "e4_replica_d2_pair1": {"configurations": ("p1-a-d2", "p1-b-d2"), "full_budget": False, "diagnostics_at": CONTROL_CHECKPOINTS},
    "e4_replica_d2_pair2": {"configurations": ("p2-a-d2", "p2-b-d2"), "full_budget": False, "diagnostics_at": CONTROL_CHECKPOINTS},
}
BATCHES = {**WAVES, **REPLICAS}
BATCH_CPUS = {name: 3 for name in BATCHES}  # as E3's campaign: 4 processes, one thread each, on the job's 3 CPUs
PROBES_MAX_PARALLEL = 2  # probe measurements sharing the GPU at once (phase probes)


def batch_runs(batch: str) -> dict[str, tuple[str, str]]:
    """job name -> (configuration, condition), the experimental runs first."""
    if batch not in BATCHES:
        raise ValueError(f"unknown batch {batch!r}; choose among {list(BATCHES)}")
    return {f"{cid}_{condition}": (cid, condition) for condition in CONDITIONS for cid in BATCHES[batch]["configurations"]}


def max_env_steps(batch: str) -> int | None:
    """The batch's stopping point (None = the full planned budget); raises for a batch whose B is not set yet."""
    spec = BATCHES[batch]
    if spec["full_budget"]:
        return None
    if B_AFTER_PILOT is None:
        raise ValueError(f"{batch}: its budget B is set after the pilot (batches.B_AFTER_PILOT), not yet")
    return int(B_AFTER_PILOT)


def diagnostics_at(batch: str) -> tuple[int, ...] | None:
    return BATCHES[batch]["diagnostics_at"]
