"""The zone-assignment stream of E5's shuffled training (docs/e5/DECISIONS.md, (2)): one generator per seed, the SAME
for a run and its reference (it never depends on the condition), independent of every training stream.

Its root is SeedSequence([seed, ASSIGNMENT_ENTROPY]): a root of its own, distinct from the experimental streams' root
SeedSequence(seed) and the reference streams' root SeedSequence([seed, 1]) (streams.worldmodel), so drawing from it never
consumes, shifts or correlates with any other stream -- a run with fraction 0 draws from it and still equals the run
without it (tested). The draws are per EPISODE: the k-th episode of a run and the k-th episode of its reference get the
same assignment, whatever their lengths.
"""

from __future__ import annotations

import numpy as np

ASSIGNMENT_ENTROPY = 2


def assignment_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), ASSIGNMENT_ENTROPY]))
