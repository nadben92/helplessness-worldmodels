from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from helplessness.agents.worldmodel.replay import (
    LabelBatch,
    ModelBatch,
    ReplayBuffer,
    ReplayCapacityError,
    ReplayOverwriteError,
)


def _fill(buf: ReplayBuffer, n: int, start: int = 0) -> None:
    image_shape = buf._x.shape[1:]
    for i in range(start, start + n):
        buf.add(
            x=np.full(image_shape, i % 256, dtype=np.uint8),
            a=i % 5,
            r=float(i),
            terminal=(i % 7 == 0),
            truncated=(i % 11 == 0),
            is_first=(i % 3 == 0),
            s=i,
        )


def test_model_batch_has_no_state_field():
    field_names = {f.name for f in dataclasses.fields(ModelBatch)}
    assert "s" not in field_names
    assert "state" not in field_names
    # background (E5, docs/e5/DECISIONS.md (4)): the agent-free background IMAGE of each step, for the loss's mask, None
    # unless the buffer holds per-episode appearances -- an image, never a state label
    assert field_names == {"x", "a", "r", "terminal", "truncated", "is_first", "background"}


def test_label_batch_only_has_state():
    field_names = {f.name for f in dataclasses.fields(LabelBatch)}
    assert field_names == {"s"}


def test_a_training_function_typed_on_modelbatch_cannot_read_labels():
    """A function that only accepts a ModelBatch parameter has no
    attribute path to reach the label -- this is the structural
    guarantee behind docs/e3/DECISIONS.md's adjustment D."""

    def fake_training_function(batch: ModelBatch) -> None:
        assert not hasattr(batch, "s")
        assert not hasattr(batch, "state")

    buf = ReplayBuffer(capacity=16, image_shape=(4, 4, 3))
    _fill(buf, 10)
    model_batch, label_batch = buf.sample(batch_size=2, seq_len=4, rng=np.random.default_rng(0))
    fake_training_function(model_batch)
    assert isinstance(label_batch, LabelBatch)  # the label lives in a SEPARATE object only


def test_sample_shapes():
    buf = ReplayBuffer(capacity=32, image_shape=(4, 4, 3))
    _fill(buf, 20)
    model_batch, label_batch = buf.sample(batch_size=5, seq_len=6, rng=np.random.default_rng(1))
    assert model_batch.x.shape == (5, 6, 4, 4, 3)
    assert model_batch.a.shape == (5, 6)
    assert model_batch.r.shape == (5, 6)
    assert model_batch.terminal.shape == (5, 6)
    assert model_batch.truncated.shape == (5, 6)
    assert model_batch.is_first.shape == (5, 6)
    assert label_batch.s.shape == (5, 6)


def test_sample_windows_are_chronologically_contiguous_before_wraparound():
    buf = ReplayBuffer(capacity=100, image_shape=(2, 2, 3))
    _fill(buf, 50)
    model_batch, label_batch = buf.sample(batch_size=3, seq_len=10, rng=np.random.default_rng(2))
    for b in range(3):
        s = label_batch.s[b]
        assert list(s) == list(range(s[0], s[0] + 10))  # consecutive true labels


def test_sample_raises_if_fewer_steps_than_seq_len():
    buf = ReplayBuffer(capacity=10, image_shape=(2, 2, 3))
    _fill(buf, 3)
    with pytest.raises(ValueError):
        buf.sample(batch_size=1, seq_len=5, rng=np.random.default_rng(0))


def test_wraparound_preserves_chronological_order_and_drops_the_oldest():
    capacity = 10
    buf = ReplayBuffer(capacity=capacity, image_shape=(2, 2, 3))
    _fill(buf, 15)  # writes s=0..14, overwriting s=0..4 with s=10..14
    assert len(buf) == capacity

    model_batch, label_batch = buf.sample(batch_size=4, seq_len=capacity, rng=np.random.default_rng(3))
    for b in range(4):
        s = label_batch.s[b]
        # the only surviving 10-step chronological window is exactly [5..14]
        assert list(s) == list(range(5, 15))


def test_add_beyond_capacity_does_not_grow_the_buffer():
    buf = ReplayBuffer(capacity=5, image_shape=(2, 2, 3))
    _fill(buf, 12)
    assert len(buf) == 5


def test_is_first_and_true_label_survive_a_sample_round_trip():
    buf = ReplayBuffer(capacity=20, image_shape=(2, 2, 3))
    _fill(buf, 20)
    model_batch, label_batch = buf.sample(batch_size=1, seq_len=20, rng=np.random.default_rng(4))
    expected_is_first = [(i % 3 == 0) for i in range(20)]
    assert list(model_batch.is_first[0]) == expected_is_first
    assert list(label_batch.s[0]) == list(range(20))


# --- point 1 (docs/e3/DECISIONS.md, 2026-09-23): capacity validation and
# strict no-overwrite mode for real runs ------------------------------


def test_for_run_accepts_exactly_budget_plus_prefill_as_capacity():
    buf = ReplayBuffer.for_run(budget=100, prefill_steps=20, image_shape=(2, 2, 3))
    assert buf.capacity == 120
    assert buf.raise_on_overwrite is True


def test_for_run_accepts_a_larger_explicit_capacity():
    buf = ReplayBuffer.for_run(budget=100, prefill_steps=20, image_shape=(2, 2, 3), capacity=1000)
    assert buf.capacity == 1000


def test_for_run_raises_explicitly_when_capacity_is_too_small():
    with pytest.raises(ReplayCapacityError):
        ReplayBuffer.for_run(budget=100, prefill_steps=20, image_shape=(2, 2, 3), capacity=50)


def test_for_run_raises_when_no_capacity_given_and_default_would_be_computed_wrong():
    # sanity: default capacity (budget+prefill) must itself never trigger
    # the error it exists to prevent
    buf = ReplayBuffer.for_run(budget=1, prefill_steps=1, image_shape=(2, 2, 3))
    _fill(buf, 2)  # exactly fills it -- must NOT raise
    assert len(buf) == 2


def test_strict_buffer_raises_on_the_first_overwrite_attempt():
    buf = ReplayBuffer.for_run(budget=5, prefill_steps=0, image_shape=(2, 2, 3))
    _fill(buf, 5)  # exactly fills capacity -- fine
    with pytest.raises(ReplayOverwriteError):
        _fill(buf, 1, start=5)  # the 6th add() must raise, not silently overwrite


def test_non_strict_buffer_still_overwrites_silently_by_default():
    """Default (non-real-run) behaviour is unchanged -- existing tests
    and ad hoc/dev usage must keep working exactly as before."""
    buf = ReplayBuffer(capacity=5, image_shape=(2, 2, 3))
    _fill(buf, 8)  # would raise under raise_on_overwrite=True; must not raise here
    assert len(buf) == 5


def test_no_sampled_window_ever_crosses_the_oldest_newest_join():
    """Across MANY wrap cycles, every sampled window's true labels must
    be a strictly consecutive run -- never jumping from near the newest
    step back to the oldest surviving one (the physical circular-buffer
    seam)."""
    capacity = 15
    buf = ReplayBuffer(capacity=capacity, image_shape=(2, 2, 3))
    total_written = capacity * 7 + 3  # several full wraps plus a partial one
    _fill(buf, total_written)

    rng = np.random.default_rng(123)
    for _ in range(500):
        seq_len = int(rng.integers(2, capacity + 1))
        _, label_batch = buf.sample(batch_size=4, seq_len=seq_len, rng=rng)
        for row in label_batch.s:
            diffs = np.diff(row)
            assert np.all(diffs == 1), f"non-consecutive window sampled: {row}"


def test_chronological_windows_cover_every_step_in_order_without_crossing_the_seam():
    capacity = 12
    buf = ReplayBuffer(capacity=capacity, image_shape=(2, 2, 3))
    _fill(buf, 30)  # wrapped: surviving labels are 18..29
    model_batch, label_batch = buf.chronological_windows(seq_len=4)
    assert label_batch.s.shape == (3, 4)
    assert list(label_batch.s.reshape(-1)) == list(range(18, 30))
    assert model_batch.x.shape == (3, 4, 2, 2, 3)
    assert not hasattr(model_batch, "s")


def test_chronological_windows_drop_the_trailing_remainder():
    buf = ReplayBuffer(capacity=20, image_shape=(2, 2, 3))
    _fill(buf, 10)
    _, label_batch = buf.chronological_windows(seq_len=4)
    assert list(label_batch.s.reshape(-1)) == list(range(8))
