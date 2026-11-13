"""The random streams of E8's therapy runs (docs/e8/DECISIONS.md (5)): a run re-trained from a checkpoint draws from
roots of its own, so that none of its streams repeats or correlates with a training stream.

  therapy_root     SeedSequence([seed, THERAPY_ENTROPY]): its children, in streams.worldmodel.STREAM_NAMES order, are the
                   therapy's training streams (explore, tiebreak, start_state, replay_sample, switch_pairs, the torch
                   generators, the measurement slots). It depends on the seed only: an exposed run and its reference
                   (same seed) draw the same starts and exploration uniforms step for step (common random numbers).
  assignment_root  SeedSequence([e8_seed, ASSIGNMENT_ENTROPY]): the shuffle therapy's per-episode draw of the treated
                   assignment, the same for every shuffle therapy run of a campaign (the k-th episode of every run gets
                   the same assignment).

The entropy words extend streams.worldmodel's and streams.assignment's: the experimental root is SeedSequence(seed),
the reference root [seed, 1], E5's assignment root [seed, 2].
"""

from __future__ import annotations

import numpy as np

from helplessness.streams import worldmodel as wm_streams

THERAPY_ENTROPY = 3
ASSIGNMENT_ENTROPY = 4
START_IN_ZONE_ENTROPY = 5


def therapy_root(seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed), THERAPY_ENTROPY])


def sequences(root: np.random.SeedSequence) -> dict[str, np.random.SeedSequence]:
    """The children of a root, named as streams.worldmodel.spawn_sequences names those of a training root."""
    return dict(zip(wm_streams.STREAM_NAMES, root.spawn(len(wm_streams.STREAM_NAMES))))


def measure_streams(root: np.random.SeedSequence, slot: int, key: int, device) -> tuple[np.random.Generator, object]:
    """The (numpy, torch) pair of measurement `slot` of a therapy run at `key` (a checkpoint): a child of the root's
    measure stream keyed by the checkpoint, as E4's checkpoint records key theirs."""
    seq = sequences(root)["measure"].spawn(wm_streams.N_MEASURE_SLOTS)[slot]
    child = np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(key)))
    return np.random.default_rng(child), wm_streams.torch_generator(child, device)


def assignment_generator(e8_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(e8_seed), ASSIGNMENT_ENTROPY]))


def start_in_zone_generator(seed: int) -> np.random.Generator:
    """The draws of start_in_zone_frac (only used when it is > 0)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), THERAPY_ENTROPY, START_IN_ZONE_ENTROPY]))
