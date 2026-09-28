"""Table mode of the replay buffer (docs/e3/DECISIONS.md, 2026-09-28, audit
B6): a private state index into the pre-rendered table instead of pixels,
the x == table[s] assertion, the same windows as pixel mode, the compact
export, chunked chronological windows."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.agents.worldmodel.replay import ObservationTableMismatch, ReplayBuffer

N_STATES = 7


def _table() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (N_STATES, 4, 4, 3), dtype=np.uint8)


def _fill_both(n: int, capacity: int):
    table = _table()
    pix = ReplayBuffer(capacity=capacity, image_shape=(4, 4, 3))
    tab = ReplayBuffer(capacity=capacity, table=table)
    rng = np.random.default_rng(0)
    for i in range(n):
        s = int(rng.integers(N_STATES))
        kwargs = dict(a=i % 5, r=float(i), terminal=(i % 7 == 0), truncated=(i % 11 == 0), is_first=(i % 3 == 0), s=s)
        pix.add(x=table[s], **kwargs)
        tab.add(x=table[s], **kwargs)
    return pix, tab, table


@pytest.mark.parametrize("n, capacity", [(40, 64), (90, 64)])  # without and with wraparound
def test_table_mode_gives_the_same_windows_as_pixel_mode(n, capacity):
    pix, tab, _ = _fill_both(n, capacity)
    for seq_len in (1, 5, 16):
        mb_p, lb_p = pix.sample(8, seq_len, np.random.default_rng(1))
        mb_t, lb_t = tab.sample(8, seq_len, np.random.default_rng(1))
        for field in ("x", "a", "r", "terminal", "truncated", "is_first"):
            assert np.array_equal(getattr(mb_p, field), getattr(mb_t, field)), field
        assert np.array_equal(lb_p.s, lb_t.s)
    cw_p, cw_t = pix.chronological_windows(8), tab.chronological_windows(8)
    assert np.array_equal(cw_p[0].x, cw_t[0].x) and np.array_equal(cw_p[1].s, cw_t[1].s)


def test_add_refuses_an_observation_that_is_not_the_table_image():
    table = _table()
    buf = ReplayBuffer(capacity=10, table=table)
    buf.add(x=table[2], a=0, r=0.0, terminal=False, truncated=False, is_first=True, s=2)
    wrong = table[2].copy()
    wrong[0, 0, 0] ^= 1
    with pytest.raises(ObservationTableMismatch):
        buf.add(x=wrong, a=0, r=0.0, terminal=False, truncated=False, is_first=False, s=2)
    assert len(buf) == 1


def test_the_state_index_is_private_to_the_buffer():
    _, tab, _ = _fill_both(20, 32)
    model_batch, _ = tab.sample(2, 4, np.random.default_rng(0))
    assert not hasattr(model_batch, "s") and model_batch.x.dtype == np.uint8 and model_batch.x.shape == (2, 4, 4, 4, 3)
    assert tab._x is None  # no pixel storage at all


def test_compact_export_round_trip():
    _, tab, table = _fill_both(90, 64)
    arrays = tab.export_compact()
    assert arrays["s"].dtype == np.int16 and arrays["a"].dtype == np.int8 and len(arrays["s"]) == 64
    rebuilt = ReplayBuffer.from_compact(arrays, table)
    for seq_len in (3, 10):
        a, la = tab.sample(6, seq_len, np.random.default_rng(5))
        b, lb = rebuilt.sample(6, seq_len, np.random.default_rng(5))
        assert np.array_equal(a.x, b.x) and np.array_equal(a.a, b.a) and np.array_equal(a.is_first, b.is_first)
        assert np.array_equal(la.s, lb.s)
    assert np.array_equal(tab.labels_and_flags()[0], rebuilt.labels_and_flags()[0])


def test_iter_chronological_windows_matches_the_single_call():
    _, tab, _ = _fill_both(90, 128)
    whole_m, whole_l = tab.chronological_windows(8)
    chunks = list(tab.iter_chronological_windows(8, chunk=3))
    assert np.array_equal(np.concatenate([c[0].x for c in chunks]), whole_m.x)
    assert np.array_equal(np.concatenate([c[1].s for c in chunks]), whole_l.s)


def test_exactly_one_storage_mode():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=4)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=4, image_shape=(2, 2, 3), table=_table())


def test_device_gather_returns_the_same_images_as_a_torch_tensor():
    torch = pytest.importorskip("torch")
    _, tab, _ = _fill_both(40, 64)
    host, _ = tab.sample(4, 6, np.random.default_rng(2))
    tab.to_device(torch.device("cpu"))
    dev, _ = tab.sample(4, 6, np.random.default_rng(2))
    assert isinstance(dev.x, torch.Tensor) and np.array_equal(dev.x.numpy(), host.x)
