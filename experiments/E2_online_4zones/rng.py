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
