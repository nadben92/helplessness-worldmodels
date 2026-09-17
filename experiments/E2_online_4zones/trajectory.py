"""Reconstructs N[s,a,s'] from a logged trajectory alone -- the second,
genuinely independent path the twin's non-circularity check needs (see
docs/E2_online_4zones.md): N_agent is read directly off the live agent
object during collection (pilot.py); N_log is rebuilt here from nothing
but the (s, a, s') record of every transition. The two must agree
bit-for-bit, or something in the collection loop, the agent's own
counting, or this reconstruction disagrees with the others.

Reset transitions were never logged as transitions in the first place
(pilot.py's on_transition hook fires once per real env.step(), never on
env.reset() -- same invariant loop.py already guarantees for N itself),
so there is nothing to exclude here beyond slicing by t.
"""

from __future__ import annotations

import numpy as np


def reconstruct_N_from_trajectory(
    traj_t: np.ndarray, traj_s: np.ndarray, traj_a: np.ndarray, traj_s_next: np.ndarray,
    up_to_t: int, n_states: int, n_actions: int,
) -> np.ndarray:
    """N[s,a,s'] built from every logged transition with t <= up_to_t.
    traj_t is assumed to be the transition index (1-based, as produced by
    train_online_e2's on_transition), one entry per logged transition, in
    the SAME order as traj_s/traj_a/traj_s_next -- no assumption of
    sortedness beyond that (a mask, not a slice, is used)."""
    mask = traj_t <= up_to_t
    N = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
    np.add.at(N, (traj_s[mask].astype(np.int64), traj_a[mask].astype(np.int64), traj_s_next[mask].astype(np.int64)), 1)
    return N
