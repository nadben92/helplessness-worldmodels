"""Episode replay buffer (spec 3.4): capacity >= the run's total step
budget (so nothing is ever forgotten for a real run -- wraparound is
still handled correctly below for smaller test buffers), uniform
64-step window sampling that may cross episode boundaries (is_first
marks where to reset), storing x, a, r, terminal, truncated, is_first,
and the true label s.

sample() returns TWO SEPARATE objects (docs/e3/DECISIONS.md, adjustment
D): ModelBatch (x, a, r, terminal, truncated, is_first) and LabelBatch
(s). World-model / goal-head / critic training functions accept only a
ModelBatch -- there is no field on that type through which a label could
reach them (see tests/agents/worldmodel/test_replay.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ReplayCapacityError(ValueError):
    """Raised by ReplayBuffer.for_run() when a run's declared capacity
    cannot hold its own budget + prefill_steps -- see
    docs/e3/DECISIONS.md, point 1: caught at run startup, not discovered
    mid-run."""


class ReplayOverwriteError(RuntimeError):
    """Raised by ReplayBuffer.add() (only in strict/raise_on_overwrite
    mode, used for real runs) if the buffer is asked to overwrite its
    oldest surviving transition -- a real run must never forget data
    (condition B counts every transition); if capacity was sized
    correctly at ReplayBuffer.for_run() time, this can only fire from a
    bug in that accounting, never from ordinary operation."""


@dataclass(frozen=True)
class ModelBatch:
    """(B, T, ...) arrays. No state label anywhere on this type --
    world-model/goal-head/critic training functions take ONLY this."""

    x: np.ndarray
    a: np.ndarray
    r: np.ndarray
    terminal: np.ndarray
    truncated: np.ndarray
    is_first: np.ndarray


@dataclass(frozen=True)
class LabelBatch:
    """(B, T) true state indices -- measures/logging only, never passed
    to a training function."""

    s: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, image_shape: tuple[int, int, int], raise_on_overwrite: bool = False):
        self.capacity = int(capacity)
        self.raise_on_overwrite = raise_on_overwrite
        self._x = np.zeros((self.capacity, *image_shape), dtype=np.uint8)
        self._a = np.zeros(self.capacity, dtype=np.int64)
        self._r = np.zeros(self.capacity, dtype=np.float32)
        self._terminal = np.zeros(self.capacity, dtype=bool)
        self._truncated = np.zeros(self.capacity, dtype=bool)
        self._is_first = np.zeros(self.capacity, dtype=bool)
        self._s = np.zeros(self.capacity, dtype=np.int64)
        self._size = 0
        self._next = 0

    @classmethod
    def for_run(
        cls,
        budget: int,
        prefill_steps: int,
        image_shape: tuple[int, int, int],
        capacity: int | None = None,
    ) -> "ReplayBuffer":
        """The constructor a real run must use (spec 3.4 + prefill,
        docs/e3/DECISIONS.md point 1): validates capacity >= budget +
        prefill_steps UP FRONT, raising ReplayCapacityError immediately
        rather than discovering the shortfall mid-run, and always builds
        the buffer in raise_on_overwrite=True (strict) mode -- a real run
        must never silently forget a transition (condition B counts all
        of them). capacity defaults to exactly budget + prefill_steps
        (the minimum that satisfies the invariant); pass a larger value
        explicitly if you want slack."""
        min_capacity = int(budget) + int(prefill_steps)
        if capacity is None:
            capacity = min_capacity
        if capacity < min_capacity:
            raise ReplayCapacityError(
                f"capacity ({capacity}) must be >= budget + prefill_steps ({min_capacity}) "
                "-- a real run's replay buffer must be able to hold every transition it will "
                "ever collect (spec section 3.4); this is checked here, at run startup, so a "
                "misconfigured run fails immediately instead of silently dropping data later"
            )
        return cls(capacity=capacity, image_shape=image_shape, raise_on_overwrite=True)

    def __len__(self) -> int:
        return self._size

    def add(self, x: np.ndarray, a: int, r: float, terminal: bool, truncated: bool, is_first: bool, s: int) -> None:
        if self.raise_on_overwrite and self._size == self.capacity:
            raise ReplayOverwriteError(
                f"replay buffer (capacity={self.capacity}) is full and about to overwrite its "
                "oldest surviving transition -- this must never happen during a real run (see "
                "ReplayBuffer.for_run(), which sizes capacity >= budget + prefill_steps up "
                "front); this is a bug in that accounting, not something to silently paper over"
            )
        idx = self._next
        self._x[idx] = x
        self._a[idx] = a
        self._r[idx] = r
        self._terminal[idx] = terminal
        self._truncated[idx] = truncated
        self._is_first[idx] = is_first
        self._s[idx] = s
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _oldest_physical_index(self) -> int:
        return 0 if self._size < self.capacity else self._next

    def chronological_windows(self, seq_len: int) -> tuple[ModelBatch, LabelBatch]:
        """EVERY stored step, cut into consecutive, non-overlapping
        seq_len windows in chronological order (a trailing remainder
        shorter than seq_len is dropped) -- for measures that must see
        all available posteriors (e.g. the z-only probe) rather than a
        random subsample. Same model/label split as sample(); never
        crosses the oldest/newest seam (same logical ordering)."""
        n_windows = self._size // seq_len
        if n_windows == 0:
            raise ValueError(f"replay buffer has only {self._size} steps, need >= seq_len={seq_len}")
        oldest = self._oldest_physical_index()
        physical = (oldest + np.arange(n_windows * seq_len)) % self.capacity
        physical = physical.reshape(n_windows, seq_len)
        model_batch = ModelBatch(
            x=self._x[physical], a=self._a[physical], r=self._r[physical], terminal=self._terminal[physical],
            truncated=self._truncated[physical], is_first=self._is_first[physical],
        )
        return model_batch, LabelBatch(s=self._s[physical])

    def sample(self, batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[ModelBatch, LabelBatch]:
        """Uniform draw of batch_size windows of seq_len consecutive
        steps, in true chronological (oldest-to-newest) order -- correct
        under wraparound: once the buffer is full, the oldest surviving
        step is physically at self._next (about to be overwritten next),
        not at physical index 0."""
        if self._size < seq_len:
            raise ValueError(f"replay buffer has only {self._size} steps, need >= seq_len={seq_len}")

        n_valid_starts = self._size - seq_len + 1
        logical_starts = rng.integers(0, n_valid_starts, size=batch_size)
        oldest = self._oldest_physical_index()

        x = np.empty((batch_size, seq_len, *self._x.shape[1:]), dtype=self._x.dtype)
        a = np.empty((batch_size, seq_len), dtype=self._a.dtype)
        r = np.empty((batch_size, seq_len), dtype=self._r.dtype)
        terminal = np.empty((batch_size, seq_len), dtype=bool)
        truncated = np.empty((batch_size, seq_len), dtype=bool)
        is_first = np.empty((batch_size, seq_len), dtype=bool)
        s = np.empty((batch_size, seq_len), dtype=self._s.dtype)

        for b, logical_start in enumerate(logical_starts):
            physical = (oldest + logical_start + np.arange(seq_len)) % self.capacity
            x[b] = self._x[physical]
            a[b] = self._a[physical]
            r[b] = self._r[physical]
            terminal[b] = self._terminal[physical]
            truncated[b] = self._truncated[physical]
            is_first[b] = self._is_first[physical]
            s[b] = self._s[physical]

        model_batch = ModelBatch(x=x, a=a, r=r, terminal=terminal, truncated=truncated, is_first=is_first)
        label_batch = LabelBatch(s=s)
        return model_batch, label_batch
