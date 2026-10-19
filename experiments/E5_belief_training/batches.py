"""E5's batch table (docs/E5_belief_training.md): 12 runs on ONE configuration of E4 (a batch parameter, --configuration,
set after E4's analysis), in three waves of four, each four runs in parallel on one GPU. Standard library only.

  e5_wave1  seed A: shuffle f = 1 (experimental, reference), shuffle f = 0.5 (experimental, reference)
  e5_wave2  seed B: baseline (experimental, reference -- the reference keeps its full state at e5.resumable_candidates and
            at the switch step when given), shuffle f = 1 (experimental, reference)
  e5_wave3  seed B: shuffle f = 0.5 (experimental, reference); immunisation A (phase 1 recomputed: E4 kept no full state)
            and immunisation B (branched from wave 2's baseline reference at the switch step)

The baseline of seed A is E4's own pair of the configuration (reused, not retrained). Every run trains E4's full planned
budget (300,000 steps); the immunisation trains switch_step (+ < one episode) + post_steps. Batch parameters, set at
launch: --configuration (every phase), --switch-step (wave 3; wave 2: optional, adds a resumable state there), --post-steps
(wave 3, default e5.immunisation.post_steps), --e4-results (probes of wave 3: E4's results root, for the reference of
immunisation A).

A later condition (the themes, after E4's analysis) is a new wave here: its runs as RUNS entries, its pairs as PROBE_PAIRS.
"""

from __future__ import annotations

CONFIG = "configs/e5/e5.yaml"

# job -> (seed label, training, fraction, condition); the immunisation has no condition (it runs on the reference's streams)
WAVES: dict[str, dict[str, tuple]] = {
    "e5_wave1": {"A_sh100_experimental": ("A", "shuffle", 1.0, "experimental"), "A_sh100_reference": ("A", "shuffle", 1.0, "reference"),
                 "A_sh050_experimental": ("A", "shuffle", 0.5, "experimental"), "A_sh050_reference": ("A", "shuffle", 0.5, "reference")},
    "e5_wave2": {"B_base_experimental": ("B", "baseline", None, "experimental"), "B_base_reference": ("B", "baseline", None, "reference"),
                 "B_sh100_experimental": ("B", "shuffle", 1.0, "experimental"), "B_sh100_reference": ("B", "shuffle", 1.0, "reference")},
    "e5_wave3": {"B_sh050_experimental": ("B", "shuffle", 0.5, "experimental"), "B_sh050_reference": ("B", "shuffle", 0.5, "reference"),
                 "A_imm": ("A", "immunisation", None, None), "B_imm": ("B", "immunisation", None, None)},
}
# the probe pairs of each wave: (seed label, training, fraction)
PROBE_PAIRS: dict[str, list[tuple]] = {
    "e5_wave1": [("A", "shuffle", 1.0), ("A", "shuffle", 0.5)],
    "e5_wave2": [("B", "baseline", None), ("B", "shuffle", 1.0)],
    "e5_wave3": [("B", "shuffle", 0.5), ("A", "immunisation", None), ("B", "immunisation", None)],
}
BATCHES = WAVES
BATCH_CPUS = {name: 3 for name in BATCHES}  # as E4: 4 processes, one thread each, on the job's 3 CPUs
PROBES_MAX_PARALLEL = 2


def batch_runs(batch: str) -> dict[str, tuple]:
    if batch not in BATCHES:
        raise ValueError(f"unknown batch {batch!r}; choose among {list(BATCHES)}")
    return dict(BATCHES[batch])


def train_args(batch: str, job: str, configuration: str, switch_step: int | None = None, post_steps: int | None = None) -> list[str]:
    """run_e5.py's arguments of one job."""
    seed_label, training, fraction, condition = batch_runs(batch)[job]
    args = ["--config", CONFIG, "--configuration", configuration, "--seed-label", seed_label, "--training", training]
    if condition is not None:
        args += ["--condition", condition]
    if training == "shuffle":
        args += ["--fraction", str(fraction)]
    if training == "baseline" and condition == "reference":
        args += ["--save-resumable-at", "candidates", *([str(int(switch_step))] if switch_step is not None else [])]
    if training == "immunisation":
        if switch_step is None:
            raise ValueError(f"{batch}: the immunisation needs --switch-step")
        args += ["--switch-step", str(int(switch_step))]
        if post_steps is not None:
            args += ["--post-steps", str(int(post_steps))]
        if seed_label == "B":  # wave 2's baseline reference of seed B kept its state at the switch step
            args += ["--branch-from", "auto"]
    return args


def probe_args(batch: str, configuration: str, checkpoint: int, e4_results: str | None = None) -> list[tuple[str, list[str]]]:
    """(job name, measure_probes.py's arguments) of every pair of the batch, at checkpoint k (post-switch for the
    immunisation)."""
    out = []
    for seed_label, training, fraction in PROBE_PAIRS[batch]:
        args = ["--config", CONFIG, "--configuration", configuration, "--checkpoint", str(int(checkpoint)), "--find-runs", "results",
                "--training", training, "--seed-label", seed_label]
        if fraction is not None:
            args += ["--fraction", str(fraction)]
        if training == "immunisation" and seed_label == "A":
            if not e4_results:
                raise ValueError("the immunisation of seed A is measured against E4's reference: --e4-results <E4 results root>")
            args += ["--e4-results", e4_results]
        tag = {"baseline": "base", "immunisation": "imm"}.get(training) or f"sh{int(round(fraction * 100)):03d}"
        out.append((f"probes_{seed_label}_{tag}_k{int(checkpoint):06d}", args))
    return out
