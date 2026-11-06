"""The twin's counts N[s,a,s'] rebuilt from a closed-loop run's compact replay (`replay_final.npz`,
ReplayBuffer.export_compact), when the run's own `counts.npz` is missing (docs/e8/DECISIONS.md (20)).

The replay convention is closed_loop.py's: a stored step is an observation with the action taken FROM it; an
episode's last observation (goal: terminal; H reached: truncated) is stored too, with a = 0, and nothing is taken from
it. So a transition is every stored step that is not an episode end, its successor the next stored step of the same
episode. When the run stops mid-episode, the last stored step's successor is not stored: `P` (the run's own
transition_tensor, deterministic) completes it, after checking its row is one-hot. Arrays in, arrays out; numpy only.
"""

from __future__ import annotations

import numpy as np


def replay_transitions(s: np.ndarray, a: np.ndarray, terminal: np.ndarray, truncated: np.ndarray, is_first: np.ndarray,
                       P: np.ndarray | None = None) -> dict:
    """The run's transitions in order: s, a, s_next (int64 arrays), and the accounting of the stored steps (counts of
    each kind; `pending`: 1 if the last transition's s_next came from P). Raises ValueError if the flags break the
    convention, or if a pending transition needs P and none (or a non-deterministic row) is given."""
    s, a = np.asarray(s, dtype=np.int64), np.asarray(a, dtype=np.int64)
    terminal, truncated, is_first = (np.asarray(x, dtype=bool) for x in (terminal, truncated, is_first))
    n = len(s)
    end = terminal | truncated
    if n == 0 or not is_first[0]:
        raise ValueError("a replay starts with an episode's first observation")
    if (terminal & truncated).any():
        raise ValueError("a step is either terminal or truncated, never both")
    if (end & is_first).any():
        raise ValueError("an episode's first observation is never its end (every episode has at least one transition)")
    if not (end[:-1] == is_first[1:]).all():
        raise ValueError("an episode end must be followed by a first observation, and a first observation must follow an end")
    if (a[end] != 0).any():
        raise ValueError("an episode's last observation carries a = 0")
    source = np.flatnonzero(~end)
    stored_next = source[source + 1 < n]
    s_next = s[stored_next + 1]
    pending = int(not end[-1])  # the run stopped mid-episode: the last step's successor was never stored
    if pending:
        if P is None:
            raise ValueError("the last transition's successor is not stored: pass the run's transition tensor P")
        row = np.asarray(P)[s[-1], a[-1]]
        if not (np.count_nonzero(row) == 1 and row.max() == 1.0):
            raise ValueError(f"P[{s[-1]}, {a[-1]}] is not one-hot: the pending successor cannot be completed")
        s_next = np.r_[s_next, int(np.argmax(row))]
    return {"s": s[source], "a": a[source], "s_next": s_next,
            "accounting": {"stored": n, "transitions": len(source), "with_stored_successor": len(stored_next), "pending": pending,
                           "end_observations": int(end.sum()), "terminal": int(terminal.sum()), "truncated": int(truncated.sum()),
                           "episodes_started": int(is_first.sum())}}


def counts_at(transitions: dict, k: int, n_states: int, n_actions: int) -> np.ndarray:
    """N[s,a,s'] of the first k transitions (run_one's N_<k>, recorded after k environment steps)."""
    if not 0 <= k <= len(transitions["s"]):
        raise ValueError(f"k = {k} outside [0, {len(transitions['s'])}]")
    flat = (transitions["s"][:k] * n_actions + transitions["a"][:k]) * n_states + transitions["s_next"][:k]
    return np.bincount(flat, minlength=n_states * n_actions * n_states).reshape(n_states, n_actions, n_states).astype(np.int64)
