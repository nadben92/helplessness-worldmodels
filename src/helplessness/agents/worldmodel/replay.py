"""Episode replay buffer (spec 3.4): capacity >= the run's total step
budget (so nothing is ever forgotten for a real run -- wraparound is
still handled correctly below for smaller test buffers), uniform
64-step window sampling that may cross episode boundaries (is_first
marks where to reset), storing x, a, r, terminal, truncated, is_first,
and the true label s.

Two storage modes (docs/e3/DECISIONS.md, 2026-09-28, audit B6):
  - table mode (ReplayBuffer(capacity, table=...), what every real run
    uses): the observation is a pure function of the state for a whole
    run (fixed themes, no pixel noise), so the buffer keeps a PRIVATE
    index into the run's pre-rendered table of n_states images instead
    of the pixels (~10 bytes per step instead of 15,552 at 72x72). add()
    still receives x and asserts x == table[s] bit for bit: the day an
    observation stops being a pure function of s (pixel noise, per-episode
    themes, rendering perturbations in E4-E6), the run fails loudly
    instead of silently storing wrong data. to_device() puts a copy of
    the table on the training device, so a minibatch is gathered there
    and no image crosses the host-device link.
  - pixel mode (ReplayBuffer(capacity, image_shape=...)): the images
    themselves, for tests and tools whose images are not renders.
The index is private: it never appears on ModelBatch, only the images
gathered from it.

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


class ObservationTableMismatch(ValueError):
    """Raised by ReplayBuffer.add() in table mode when the observation is
    not bit-identical to the pre-rendered image of its state: the table
    representation would store wrong data (docs/e3/DECISIONS.md,
    2026-09-28)."""


@dataclass(frozen=True)
class ModelBatch:
    """(B, T, ...) arrays. No state label anywhere on this type --
    world-model/goal-head/critic training functions take ONLY this. x is
    uint8 (B, T, H, W, C): a numpy array, or a torch tensor on the
    training device when the buffer gathers from a device table
    (ReplayBuffer.to_device)."""

    x: object
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
    def __init__(
        self, capacity: int, image_shape: tuple[int, int, int] | None = None, raise_on_overwrite: bool = False,
        table: np.ndarray | None = None,
    ):
        if (image_shape is None) == (table is None):
            raise ValueError("pass exactly one of image_shape (pixel mode) or table (table mode)")
        self.capacity = int(capacity)
        self.raise_on_overwrite = raise_on_overwrite
        if table is not None:
            table = np.ascontiguousarray(table)
            if table.dtype != np.uint8 or table.ndim != 4:
                raise ValueError(f"table must be uint8 (n_states, H, W, C), got {table.dtype} {table.shape}")
            if table.shape[0] > np.iinfo(np.int16).max:
                raise ValueError(f"table has {table.shape[0]} images; the private index is int16")
            self._table = table
            self._obs_index = np.zeros(self.capacity, dtype=np.int16)
            self._x = None
        else:
            self._table = None
            self._obs_index = None
            self._x = np.zeros((self.capacity, *image_shape), dtype=np.uint8)
        self._device_table = None
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
        image_shape: tuple[int, int, int] | None = None,
        capacity: int | None = None,
        table: np.ndarray | None = None,
    ) -> "ReplayBuffer":
        """The constructor a real run must use (spec 3.4 + prefill,
        docs/e3/DECISIONS.md point 1): validates capacity >= budget +
        prefill_steps UP FRONT, raising ReplayCapacityError immediately
        rather than discovering the shortfall mid-run, and always builds
        the buffer in raise_on_overwrite=True (strict) mode -- a real run
        must never silently forget a transition (condition B counts all
        of them). capacity defaults to exactly budget + prefill_steps
        (the minimum that satisfies the invariant); pass a larger value
        explicitly if you want slack. Pass the run's pre-rendered table
        (table mode) or an image_shape (pixel mode)."""
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
        return cls(capacity=capacity, image_shape=image_shape, raise_on_overwrite=True, table=table)

    @property
    def table_mode(self) -> bool:
        return self._table is not None

    @property
    def image_shape(self) -> tuple[int, ...]:
        return tuple((self._table if self.table_mode else self._x).shape[1:])

    def to_device(self, device) -> "ReplayBuffer":
        """Table mode only: gathers every later minibatch from a copy of
        the table on `device` (a torch device), so ModelBatch.x is a uint8
        torch tensor already there. The host table stays the reference of
        add()'s assertion. Returns self."""
        if not self.table_mode:
            raise ValueError("to_device() needs table mode")
        import torch

        self._device_table = torch.as_tensor(self._table).to(device)
        return self

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
        if self.table_mode:
            if not np.array_equal(np.asarray(x), self._table[int(s)]):
                raise ObservationTableMismatch(
                    f"observation of state {int(s)} differs from the pre-rendered table image: the observation is no "
                    "longer a pure function of the state, so the table replay would store wrong data"
                )
            self._obs_index[idx] = s
        else:
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

    def _physical(self, logical_starts: np.ndarray, seq_len: int) -> np.ndarray:
        """(n, seq_len) physical indices of the windows starting at the
        given logical positions (0 = oldest stored step) -- correct under
        wraparound: once the buffer is full, the oldest surviving step is
        physically at self._next, not at physical index 0."""
        oldest = self._oldest_physical_index()
        return (oldest + np.asarray(logical_starts, dtype=np.int64)[:, None] + np.arange(seq_len)[None, :]) % self.capacity

    def _images(self, physical: np.ndarray):
        if not self.table_mode:
            return self._x[physical]
        index = self._obs_index[physical].astype(np.int64)
        if self._device_table is None:
            return self._table[index]
        import torch

        return self._device_table[torch.as_tensor(index, device=self._device_table.device)]

    def windows(self, logical_starts: np.ndarray, seq_len: int) -> tuple[ModelBatch, LabelBatch]:
        """The windows of seq_len consecutive steps starting at the given
        logical positions (0 = oldest stored step), in chronological
        order. Every window must fit in the stored steps."""
        logical_starts = np.asarray(logical_starts, dtype=np.int64)
        if logical_starts.size and (logical_starts.min() < 0 or logical_starts.max() + seq_len > self._size):
            raise ValueError(f"window starts must lie in [0, {self._size - seq_len}] for seq_len={seq_len}")
        physical = self._physical(logical_starts, seq_len)
        model_batch = ModelBatch(
            x=self._images(physical), a=self._a[physical], r=self._r[physical], terminal=self._terminal[physical],
            truncated=self._truncated[physical], is_first=self._is_first[physical],
        )
        return model_batch, LabelBatch(s=self._s[physical])

    def chronological_windows(self, seq_len: int) -> tuple[ModelBatch, LabelBatch]:
        """EVERY stored step, cut into consecutive, non-overlapping
        seq_len windows in chronological order (a trailing remainder
        shorter than seq_len is dropped) -- for measures that must see
        all available posteriors (e.g. the z-only probe) rather than a
        random subsample. Same model/label split as sample(); never
        crosses the oldest/newest seam (same logical ordering). For a
        large buffer, iter_chronological_windows gives the same windows
        in chunks."""
        n_windows = self._size // seq_len
        if n_windows == 0:
            raise ValueError(f"replay buffer has only {self._size} steps, need >= seq_len={seq_len}")
        return self.windows(np.arange(n_windows) * seq_len, seq_len)

    def iter_chronological_windows(self, seq_len: int, chunk: int):
        """chronological_windows(seq_len), yielded `chunk` windows at a
        time (audit B4: never the whole buffer in one array)."""
        n_windows = self._size // seq_len
        if n_windows == 0:
            raise ValueError(f"replay buffer has only {self._size} steps, need >= seq_len={seq_len}")
        for first in range(0, n_windows, chunk):
            yield self.windows(np.arange(first, min(first + chunk, n_windows)) * seq_len, seq_len)

    def labels_and_flags(self) -> tuple[np.ndarray, np.ndarray]:
        """(s, is_first) of every stored step in logical (oldest-first)
        order -- for MEASUREMENTS that select steps by their true state
        (probe.real_context_probe), never for training."""
        physical = (self._oldest_physical_index() + np.arange(self._size)) % self.capacity
        return self._s[physical].copy(), self._is_first[physical].copy()

    def sample(self, batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[ModelBatch, LabelBatch]:
        """Uniform draw of batch_size windows of seq_len consecutive
        steps, in true chronological (oldest-to-newest) order (see
        _physical for the wraparound)."""
        if self._size < seq_len:
            raise ValueError(f"replay buffer has only {self._size} steps, need >= seq_len={seq_len}")
        n_valid_starts = self._size - seq_len + 1
        logical_starts = rng.integers(0, n_valid_starts, size=batch_size)
        return self.windows(logical_starts, seq_len)

    # ------------------------------------------------------------------
    # compact export / import (table mode): what a run saves so every
    # measure can be recomputed later (docs/e3/DECISIONS.md, 2026-09-28)
    # ------------------------------------------------------------------

    def export_compact(self) -> dict[str, np.ndarray]:
        """Every stored step in logical order, compact dtypes: s int16
        (the label, also the table index), a int8, r float32 and the
        three flags."""
        if not self.table_mode:
            raise ValueError("export_compact() needs table mode")
        physical = (self._oldest_physical_index() + np.arange(self._size)) % self.capacity
        return {
            "s": self._s[physical].astype(np.int16), "a": self._a[physical].astype(np.int8), "r": self._r[physical].copy(),
            "terminal": self._terminal[physical].copy(), "truncated": self._truncated[physical].copy(),
            "is_first": self._is_first[physical].copy(),
        }

    @classmethod
    def from_compact(cls, arrays: dict[str, np.ndarray], table: np.ndarray) -> "ReplayBuffer":
        """The buffer export_compact() was taken from, rebuilt from its
        arrays and the run's table (strict mode, capacity = its size)."""
        s = np.asarray(arrays["s"]).astype(np.int64)
        if s.size and (s.min() < 0 or s.max() >= table.shape[0]):
            raise ValueError("state index outside the table")
        n = len(s)
        buf = cls(capacity=n, table=table, raise_on_overwrite=True)
        buf._obs_index[:] = s
        buf._s[:] = s
        buf._a[:] = arrays["a"]
        buf._r[:] = arrays["r"]
        buf._terminal[:] = arrays["terminal"]
        buf._truncated[:] = arrays["truncated"]
        buf._is_first[:] = arrays["is_first"]
        buf._size = n
        buf._next = 0
        return buf
