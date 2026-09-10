"""Generic pieces of the Phase-2 recovery protocol (CLAUDE.md online task,
steps 3-4): the periodic out-of-loop evaluation + sliding-window
mechanistic-metrics tracker, and the recovery_time metric itself. Neither
depends on GridWorld or any concrete Env -- RecoveryTracker takes an
eval_env_factory callable rather than building its own environment, so this
module transfers unchanged to whatever environment E2 actually runs on.

Env-specific orchestration (constructing that factory, loading snapshots,
the CLI, writing CSVs) is deliberately NOT here -- see
experiments/E0_offline_5x5/05_online_prototype/recovery.py, which is
GridWorld-coupled orchestration built on top of these two pieces.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from helplessness.evaluation.rollout import evaluate


class RecoveryTracker:
    """train_online's on_transition callback for one recovery run: records
    the periodic out-of-loop evaluation curve, plus sliding-window
    mechanistic metrics (zero-sample (s,a) pairs, state-visitation entropy)
    over the last `mechanistic_window` transitions.

    eval_env_factory() must return a fresh environment instance at whatever
    "healthy" condition recovery is measured against (p=0 for GridWorld) --
    called once per evaluation checkpoint, never the training env, so
    evaluation never touches the training trajectory (out of loop by
    construction).
    """

    def __init__(
        self,
        agent,
        eval_env_factory: Callable[[], object],
        eval_every: int,
        n_eval_episodes: int,
        eval_seed: int,
        non_terminal_pairs: list[tuple[int, int]],
        mechanistic_window: int,
    ):
        self.agent = agent
        self.eval_env_factory = eval_env_factory
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.non_terminal_pairs = non_terminal_pairs
        self.window: deque = deque(maxlen=mechanistic_window)
        self.rows: list[dict] = []

    def __call__(self, t, s, a, r, s_next, terminated, truncated) -> None:
        self.window.append((s, a))
        if t % self.eval_every != 0:
            return

        eval_env = self.eval_env_factory()
        summary, _ = evaluate(eval_env, self.agent.policy, n_episodes=self.n_eval_episodes, seed=self.eval_seed)

        visited = set(self.window)
        zero_sample_pairs = sum(1 for pair in self.non_terminal_pairs if pair not in visited)

        state_counts: dict[int, int] = {}
        for s_win, _a_win in self.window:
            state_counts[s_win] = state_counts.get(s_win, 0) + 1
        counts = np.array(list(state_counts.values()), dtype=np.float64)
        probs = counts / counts.sum()
        state_entropy = float(-np.sum(probs * np.log2(probs)))

        self.rows.append({
            "t": t,
            "return_mean": summary["return_mean"],
            "success_rate": summary["success_rate"],
            "zero_sample_pairs": zero_sample_pairs,
            "state_entropy": state_entropy,
        })


def recovery_time(rows: list[dict], oracle_return: float, threshold: float = 0.95) -> float | None:
    """First t at which mean return reaches threshold * oracle_return. None
    (not a sentinel number) if the run never recovers within budget --
    "never recovered" must not be silently coded as a large-but-finite time."""
    target = threshold * oracle_return
    for row in sorted(rows, key=lambda r: r["t"]):
        if row["return_mean"] >= target:
            return row["t"]
    return None
