"""E5's batch table (docs/E5_belief_training.md): runs on ONE configuration of E4 (p1-b-d1), each wave on one GPU.
Standard library only.

  e5_wave1  shuffle f = 1, seeds A and B, experimental and reference (4 runs; docs/e5/DECISIONS.md (29): the relaunch
            after the first wave 1 failed at the separation deadline)
  e5_wave2  seed B's baseline, experimental and reference (2 runs; the reference keeps its full state at 100,000 steps,
            BASELINE_REFERENCE_RESUMABLE_AT, from which E8's immunisation will branch)

DEFERRED (defined, never submitted): shuffle f = 0.5 (out of the campaign until its diagnosis) and the immunisation
(moved to E8). The baseline of seed A is E4's own pair of the configuration (reused, not retrained). Every run trains E4's
full planned budget (300,000 steps). Batch parameters, set at launch: --configuration (every phase; p1-b-d1 only),
--e4-results (probes of the baseline A).

  e5_probes_baseline_A  no training: the probes of seed A's baseline, i.e. E4's own pair of the configuration (under
            --e4-results), re-measured with E5's probe script (150 walks, validity, reconstruction), so that every
            condition is probed by one protocol (docs/e5/DECISIONS.md (25)).

The configuration is p1-b-d1 (CONFIGURATION, docs/e5/DECISIONS.md (21)); the submission refuses another one.

A later condition (the themes, after E4's analysis) is a new wave here: its runs as RUNS entries, its pairs as PROBE_PAIRS.
"""

from __future__ import annotations

CONFIG = "configs/e5/e5.yaml"
CONFIGURATION = "p1-b-d1"  # E5's configuration of E4 (docs/e5/DECISIONS.md (21)): seed A = 2, seed B = 102

# job -> (seed label, training, fraction, condition); the immunisation has no condition (it runs on the reference's streams)
# 2026-10-22 (docs/e5/DECISIONS.md (29)): the first wave 1 (f = 1 and f = 0.5, seed A) failed at the separation deadline;
# the campaign is now f = 1 on seeds A and B (wave 1) and seed B's baseline (wave 2); f = 0.5 and the immunisation (moved
# to E8) are DEFERRED: defined, never submitted (the submission only knows WAVES).
WAVES: dict[str, dict[str, tuple]] = {
    "e5_wave1": {"A_sh100_experimental": ("A", "shuffle", 1.0, "experimental"), "A_sh100_reference": ("A", "shuffle", 1.0, "reference"),
                 "B_sh100_experimental": ("B", "shuffle", 1.0, "experimental"), "B_sh100_reference": ("B", "shuffle", 1.0, "reference")},
    "e5_wave2": {"B_base_experimental": ("B", "baseline", None, "experimental"), "B_base_reference": ("B", "baseline", None, "reference")},
    "e5_probes_baseline_A": {},  # probes only: seed A's baseline is E4's own pair, re-probed with E5's protocol
}
DEFERRED: dict[str, dict[str, tuple]] = {
    "f050": {"A_sh050_experimental": ("A", "shuffle", 0.5, "experimental"), "A_sh050_reference": ("A", "shuffle", 0.5, "reference"),
             "B_sh050_experimental": ("B", "shuffle", 0.5, "experimental"), "B_sh050_reference": ("B", "shuffle", 0.5, "reference")},
    "e8_immunisation": {"A_imm": ("A", "immunisation", None, None), "B_imm": ("B", "immunisation", None, None)},
}
# the steps where the baseline reference of seed B keeps its full state: 100,000, the immunisation's switch step (E8)
BASELINE_REFERENCE_RESUMABLE_AT = (100000,)
# the probe pairs of each wave: (seed label, training, fraction)
PROBE_PAIRS: dict[str, list[tuple]] = {
    "e5_wave1": [("A", "shuffle", 1.0), ("B", "shuffle", 1.0)],
    "e5_wave2": [("B", "baseline", None)],
    "e5_probes_baseline_A": [("A", "baseline", None)],
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
        args += ["--save-resumable-at", *map(str, BASELINE_REFERENCE_RESUMABLE_AT)]
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
        if training in ("immunisation", "baseline") and seed_label == "A":
            if not e4_results:
                raise ValueError(f"the {training} of seed A is (or is measured against) E4's own runs: --e4-results <E4 results root>")
            args += ["--e4-results", e4_results]
        tag = {"baseline": "base", "immunisation": "imm"}.get(training) or f"sh{int(round(fraction * 100)):03d}"
        out.append((f"probes_{seed_label}_{tag}_k{int(checkpoint):06d}", args))
    return out
