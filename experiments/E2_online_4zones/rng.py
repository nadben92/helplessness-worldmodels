"""One integer seed -> every independent RNG stream E2's protocol needs,
via np.random.SeedSequence(seed).spawn(k) -- never
np.random.default_rng(seed) called twice with the same integer value.

That naive pattern was a real, found (not anticipated) bug in E1 (see
experiments/E1_offline_4zones/estimation.py::spawn_exp_ctrl_rngs's
docstring): two separately-constructed Generators seeded with the same
integer value stayed in lockstep across an entire draw sequence whenever
the underlying distributions were identical, silently erasing the
sampling variance a later analysis depended on. SeedSequence.spawn()
breaks that correlation by construction, regardless of the RNG
algorithm's internal stream-consumption pattern -- the fix generalises
here to every stream E2 needs, including the environment's own dynamics
generator (E1's collection was pure DP/generative and never touched this;
E2's online loop does).
"""

from __future__ import annotations

import numpy as np

STREAM_NAMES = (
    "env_dynamics",  # the online training env's own transition noise
    "explore",  # epsilon-greedy action choice
    "tiebreak",  # argmax_random_tie departage at each replan
    "start_state",  # uniform draw over non-terminal start states at each reset
    "matched_control",  # the matched-counts control's multinomial draws from P
    "reference",  # the matched-counts reference's multinomial draws from P_ref
)


def spawn_streams(seed: int) -> dict[str, np.random.Generator]:
    """One np.random.Generator per name in STREAM_NAMES, all independently
    derived from the single integer `seed`. Deterministic: the same seed
    always produces the same 6 streams, in the same order (spawn() is
    itself deterministic given the parent SeedSequence)."""
    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(len(STREAM_NAMES))
    return {name: np.random.default_rng(child) for name, child in zip(STREAM_NAMES, children)}


def spawn_checkpoint_streams(seed: int, checkpoint_step: int) -> dict[str, np.random.Generator]:
    """matched_control + reference streams for ONE checkpoint of ONE seed
    -- the matched-counts control and the matched-counts reference must
    be RECOMPUTED at every checkpoint from N(t), each checkpoint getting
    its own independent draw (spec: "tirages independants par checkpoint
    derives par spawn").

    Entropy is the (seed, checkpoint_step) PAIR itself, not
    spawn_streams(seed)'s own matched_control/reference streams advanced
    checkpoint-by-checkpoint -- advancing a single shared stream would
    make checkpoint k's draw depend on every checkpoint before it having
    been visited in order, which breaks two things this design needs:
    (1) a checkpoint's result must be independent of which OTHER
    checkpoints were computed and in what order (needed to parallelise
    the sweep per (seed, checkpoint) without the result depending on
    execution order), and (2) rerunning a single seed alone must
    reproduce byte-identical per-checkpoint artifacts to the same seed
    computed inside a larger parallel sweep. Keying directly on
    checkpoint_step (not a position/index into some checkpoint list)
    means the checkpoint schedule itself can change without perturbing
    already-computed checkpoints' streams.
    """
    seed_seq = np.random.SeedSequence([int(seed), int(checkpoint_step)])
    child_matched, child_reference = seed_seq.spawn(2)
    return {
        "matched_control": np.random.default_rng(child_matched),
        "reference": np.random.default_rng(child_reference),
    }
