"""E5's batch table (docs/E5_belief_training.md): runs on ONE configuration of E4 (p1-b-d1), each wave on one GPU (one
compute pod at a time). Standard library only. Seeds: A = 2 (E4's), B = 102, E = 402, F = 502 (e5_setup.configuration;
C = 202 and D = 302 are defined but unused since the plan of 2026-10-23).

  e5_wave1  shuffle f = 1, seeds A and B, experimental and reference (docs/e5/DECISIONS.md (29))
  e5_wave2  seed B's baseline, experimental and reference (the reference keeps its full state at 100,000 steps,
            BASELINE_REFERENCE_RESUMABLE_AT, for E8), and two shuffle f = 1 EXPERIMENTAL runs, seeds E and F, replacing
            seed B's experimental run (locked in): both measured against seed B's f = 1 reference (REFERENCE_SEED_OF,
            docs/e5/DECISIONS.md (32))
  e5_probes_all  no training: the probes of every healthy pair of waves 1 and 2 and of seed A's baseline (E4's own pair),
            in ONE batch after wave 2, at K = 300,000 (PROBES_CHECKPOINT); a pair whose run failed, never switched or
            locked in is left out at submission (submit_kubeflow.py, phase probes)
  e5_probes_baselines  no training (docs/e5/DECISIONS.md (36)): the baselines A (E4's own pair) and B at 300,000
            (PROBES_CHECKPOINT), and baseline B again at each checkpoint given (--checkpoint k ...: its checkpoints matched
            to the shuffle pairs' level of learning, matched_checkpoint.py)

Two waves only (2026-10-23): no wave 3. DEFERRED (defined, never submitted): shuffle f = 0.5 (abandoned for E5, (31))
and the immunisation (moved to E8). Every run trains E4's full planned budget (300,000 steps) with E5's procedure
(same-background switch statistic, deadline 75,000, late threshold 1.5 bits from 6,100). Batch parameters, set at
launch: --configuration (every phase; p1-b-d1 only), --e4-results (probes of the baseline A).
"""

from __future__ import annotations

CONFIG = "configs/e5/e5.yaml"
CONFIGURATION = "p1-b-d1"  # E5's configuration of E4 (docs/e5/DECISIONS.md (21)): seeds A = 2, B = 102, C = 202, D = 302

# job -> (seed label, training, fraction, condition); the immunisation has no condition (it runs on the reference's streams)
# docs/e5/DECISIONS.md (29), (31): f = 1 on seeds A-D and the baseline on B and C (A's is E4's pair); f = 0.5 (abandoned)
# and the immunisation (E8) are DEFERRED: defined, never submitted (the submission only knows WAVES).
WAVES: dict[str, dict[str, tuple]] = {
    "e5_wave1": {"A_sh100_experimental": ("A", "shuffle", 1.0, "experimental"), "A_sh100_reference": ("A", "shuffle", 1.0, "reference"),
                 "B_sh100_experimental": ("B", "shuffle", 1.0, "experimental"), "B_sh100_reference": ("B", "shuffle", 1.0, "reference")},
    "e5_wave2": {"B_base_experimental": ("B", "baseline", None, "experimental"), "B_base_reference": ("B", "baseline", None, "reference"),
                 "E_sh100_experimental": ("E", "shuffle", 1.0, "experimental"), "F_sh100_experimental": ("F", "shuffle", 1.0, "experimental")},
    "e5_probes_baseline_A": {},  # probes only: seed A's baseline is E4's own pair, re-probed with E5's protocol
    # probes only, ONE batch after wave 2 at K = PROBES_CHECKPOINT (docs/e5/DECISIONS.md (31), (32)): every healthy pair
    "e5_probes_all": {},  # NOT USED since (33)'s two sets: e5_probes_a, e5_probes_b
    "e5_probes_a": {},
    "e5_probes_b": {},
    "e5_probes_baselines": {},  # docs/e5/DECISIONS.md (36)
    # docs/e5/DECISIONS.md (33): the healthy shuffle runs continued from their resume point to EXTEND_TO steps, in two
    # batches (one pod at a time): seed A's pair first, then seed B's reference with the replacements E and F
    "e5_extend_a": {"A_sh100_experimental": ("A", "shuffle", 1.0, "experimental"), "A_sh100_reference": ("A", "shuffle", 1.0, "reference")},
    "e5_extend_b": {"B_sh100_reference": ("B", "shuffle", 1.0, "reference"), "E_sh100_experimental": ("E", "shuffle", 1.0, "experimental"),
                    "F_sh100_experimental": ("F", "shuffle", 1.0, "experimental")},
    # NOT USED (replaced by e5_extend_a and e5_extend_b, (33)): the five runs in one pod
    "e5_extend": {"A_sh100_experimental": ("A", "shuffle", 1.0, "experimental"), "A_sh100_reference": ("A", "shuffle", 1.0, "reference"),
                  "B_sh100_reference": ("B", "shuffle", 1.0, "reference"), "E_sh100_experimental": ("E", "shuffle", 1.0, "experimental"),
                  "F_sh100_experimental": ("F", "shuffle", 1.0, "experimental")},
}
EXTEND_BATCHES = ("e5_extend_a", "e5_extend_b", "e5_extend")
EXTEND_TO = 650000  # environment steps, a ceiling: the user stops the pod once every pair meets E4's criterion (docs/e5/DECISIONS.md (33))
# an experimental run measured against the reference of ANOTHER seed (docs/e5/DECISIONS.md (32)): the replacements of seed
# B's experimental run (locked in) are paired with seed B's f = 1 reference. Absent: the run's own seed.
REFERENCE_SEED_OF: dict[tuple[str, str], str] = {("E", "shuffle"): "B", ("F", "shuffle"): "B"}
PROBES_CHECKPOINT = 300000  # the baselines' probe checkpoint (converged from 100,000 on); the shuffle pairs' is --checkpoint
DEFERRED: dict[str, dict[str, tuple]] = {
    # f = 0.5: abandoned for E5 (docs/e5/DECISIONS.md (31)), kept defined
    "f050": {"A_sh050_experimental": ("A", "shuffle", 0.5, "experimental"), "A_sh050_reference": ("A", "shuffle", 0.5, "reference"),
             "B_sh050_experimental": ("B", "shuffle", 0.5, "experimental"), "B_sh050_reference": ("B", "shuffle", 0.5, "reference")},
    "e8_immunisation": {"A_imm": ("A", "immunisation", None, None), "B_imm": ("B", "immunisation", None, None)},
}
# the steps where the baseline reference of seed B keeps its full state: 100,000, the immunisation's switch step (E8)
BASELINE_REFERENCE_RESUMABLE_AT = (100000,)
# the probe pairs of each wave: (seed label, training, fraction)
PROBE_PAIRS: dict[str, list[tuple]] = {
    "e5_wave1": [("A", "shuffle", 1.0), ("B", "shuffle", 1.0)],
    "e5_wave2": [("B", "baseline", None), ("E", "shuffle", 1.0), ("F", "shuffle", 1.0)],
    "e5_probes_baseline_A": [("A", "baseline", None)],
    # every pair of the campaign; at submission, those with a failed, never-switched or locked-in run are left out (seed B's
    # f = 1 pair, whose experimental run locked in, is replaced by E and F against B's reference)
    "e5_probes_all": [("A", "baseline", None), ("B", "baseline", None), ("A", "shuffle", 1.0), ("B", "shuffle", 1.0),
                      ("E", "shuffle", 1.0), ("F", "shuffle", 1.0)],
    # docs/e5/DECISIONS.md (33): the probes in two sets, each shuffle pair at ITS first checkpoint meeting E4's criterion:
    # the baselines A and B at PROBES_CHECKPOINT and seed A's f = 1 pair at K_a (--checkpoint); then E and F at K_b
    "e5_probes_a": [("A", "baseline", None), ("B", "baseline", None), ("A", "shuffle", 1.0)],
    "e5_probes_b": [("E", "shuffle", 1.0), ("F", "shuffle", 1.0)],
    # docs/e5/DECISIONS.md (36): the baselines at PROBES_CHECKPOINT (the secondary comparison) and baseline B at its
    # checkpoints matched to the level of the preliminary shuffle pairs E and F (the primary comparison)
    "e5_probes_baselines": [("A", "baseline", None), ("B", "baseline", None)],
}
# the baselines probed at PROBES_CHECKPOINT whatever --checkpoint is; in e5_probes_baselines the seeds whose baseline is
# ALSO probed at every --checkpoint (the level-matched checkpoints, docs/e5/DECISIONS.md (36))
BASELINES_AT_PROBES_CHECKPOINT = ("e5_probes_all", "e5_probes_a", "e5_probes_baselines")
MATCHED_BASELINE_SEEDS: dict[str, tuple[str, ...]] = {"e5_probes_baselines": ("B",)}
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
    if batch in EXTEND_BATCHES:  # the continuation of the finished run of this job (run_e5.py --extend auto)
        return ["--extend", "auto", "--to", str(EXTEND_TO), "--configuration", configuration, "--seed-label", seed_label,
                "--training", training, "--fraction", str(fraction), "--condition", condition]
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


def probe_specs(batch: str, checkpoints) -> list[tuple]:
    """(seed label, training, fraction, k) of every probe job of the batch: each pair at each checkpoint given (one int or
    several), except the baselines of BASELINES_AT_PROBES_CHECKPOINT, at PROBES_CHECKPOINT -- and, in
    MATCHED_BASELINE_SEEDS' batches, ALSO at each checkpoint given (docs/e5/DECISIONS.md (33), (36)). No duplicate."""
    ks = [int(checkpoints)] if isinstance(checkpoints, int) else [int(k) for k in checkpoints]
    out = []
    for seed_label, training, fraction in PROBE_PAIRS[batch]:
        if batch in BASELINES_AT_PROBES_CHECKPOINT and training == "baseline":
            own = [PROBES_CHECKPOINT] + (ks if seed_label in MATCHED_BASELINE_SEEDS.get(batch, ()) else [])
        else:
            own = ks
        for k in own:
            if (seed_label, training, fraction, k) not in out:
                out.append((seed_label, training, fraction, k))
    return out


def probe_args(batch: str, configuration: str, checkpoint, e4_results: str | None = None) -> list[tuple[str, list[str]]]:
    """(job name, measure_probes.py's arguments) of every probe job of the batch (probe_specs), at checkpoint k: one int or
    several (post-switch for the immunisation)."""
    out = []
    for seed_label, training, fraction, k in probe_specs(batch, checkpoint):
        args = ["--config", CONFIG, "--configuration", configuration, "--checkpoint", str(k), "--find-runs", "results",
                "--training", training, "--seed-label", seed_label]
        ref_label = REFERENCE_SEED_OF.get((seed_label, training))
        if ref_label is not None:
            args += ["--reference-seed-label", ref_label]
        if fraction is not None:
            args += ["--fraction", str(fraction)]
        if training in ("immunisation", "baseline") and seed_label == "A":
            if not e4_results:
                raise ValueError(f"the {training} of seed A is (or is measured against) E4's own runs: --e4-results <E4 results root>")
            args += ["--e4-results", e4_results]
        tag = {"baseline": "base", "immunisation": "imm"}.get(training) or f"sh{int(round(fraction * 100)):03d}"
        out.append((f"probes_{seed_label}_{tag}_k{k:06d}", args))
    return out
