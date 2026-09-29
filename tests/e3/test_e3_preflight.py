"""The preflight's memory test (docs/e3/DECISIONS.md, 2026-09-29), without a
GPU: the verdict adds up each process's worst peak, and the barrier makes
the side-by-side processes measure together."""

from __future__ import annotations

import itertools
import threading

import pytest

from _experiment_loader import load_experiment_module

preflight = load_experiment_module("E3_worldmodel_4zones", "preflight.py", "test_preflight")


def _peaks(training, diagnostic, final):
    return {"training": {"peak_reserved_mib": training}, "diagnostic_point": {"peak_reserved_mib": diagnostic},
            "final_measures": {"peak_reserved_mib": final}}


def test_memory_verdict_sums_each_process_s_worst_peak():
    results = [_peaks(10_000, 29_000, 31_000), _peaks(3_000, 3_500, 4_000)]
    verdict = preflight.memory_verdict(results, total_mib=40_000, headroom=0.1)
    assert verdict["per_process_worst_peak_reserved_mib"] == [31_000, 4_000]
    assert verdict["sum_mib"] == 35_000 and verdict["budget_mib"] == pytest.approx(36_000)
    assert verdict["fits"]
    assert not preflight.memory_verdict(results * 2, total_mib=40_000, headroom=0.1)["fits"]


def test_barrier_waits_for_every_process(tmp_path, monkeypatch):
    released = []
    pids = itertools.count(1000)  # a distinct ready file per caller
    monkeypatch.setattr(preflight.os, "getpid", lambda: next(pids))
    threads = [threading.Thread(target=lambda: (preflight._wait_barrier(tmp_path / "b", 3), released.append(1))) for _ in range(2)]
    for t in threads:
        t.start()
    threads[0].join(timeout=0.5)
    assert not released  # 2 of 3 ready: nobody passes
    preflight._wait_barrier(tmp_path / "b", 3)
    for t in threads:
        t.join(timeout=5)
    assert len(released) == 2
    with pytest.raises(TimeoutError):
        preflight._wait_barrier(tmp_path / "c", 2, timeout_s=0.3)
