"""The preflight's memory test (docs/e3/DECISIONS.md, 2026-09-29), without a
GPU: the verdict adds up each process's worst peak, and the barrier makes
the side-by-side processes measure together."""

from __future__ import annotations

import itertools
import json
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


def test_batch_hours_counts_the_queue():
    assert preflight.batch_hours([10, 10, 10, 10, 6, 6, 6, 6], 8) == 10
    assert preflight.batch_hours([10] * 4 + [6] * 4, 4) == 16
    assert preflight.batch_hours([5, 3, 3], 2) == 6  # longest first, into the slot that frees first


def test_effective_cpus_reports_a_usable_count():
    cpus = preflight.effective_cpus()
    assert cpus["usable"] >= 1 and cpus["cpu_count"] >= cpus["usable"]


def _verdict(problems=(), memory=(), warnings=(), parallel=8, hours=12.0):
    return {"passed": not (list(problems) + list(memory)), "problems": list(problems) + list(memory), "memory_problems": list(memory),
            "warnings": list(warnings), "parallel": parallel, "projected_batch_hours": hours}


def test_launch_rule_of_run_gpu_step_3():
    """docs/e3/DECISIONS.md, 2026-09-30 (3): launch only on a clean pass within 20 h; retry at 4 on a
    memory-only failure at a higher parallelism; stop otherwise."""
    assert preflight.decide(_verdict(), 20, 4) == "launch"
    assert preflight.decide(_verdict(hours=20.0), 20, 4) == "launch"
    assert preflight.decide(_verdict(hours=20.5), 20, 4) == "stop"
    assert preflight.decide(_verdict(hours=float("nan")), 20, 4) == "stop"
    assert preflight.decide(_verdict(hours=None), 20, 4) == "stop"
    assert preflight.decide(_verdict(warnings=["cpu"]), 20, 4) == "stop"
    assert preflight.decide(_verdict(memory=["oom"]), 20, 4) == "retry-max-parallel-4"
    assert preflight.decide(_verdict(memory=["oom"], parallel=4), 20, 4) == "stop"
    assert preflight.decide(_verdict(problems=["determinism"], memory=["oom"]), 20, 4) == "stop"
    assert preflight.decide(_verdict(problems=["determinism"]), 20, 4) == "stop"


def test_verdict_file(tmp_path):
    decision = preflight.write_verdict(tmp_path, [], [], [], 8, 11.5, 20.0, 4)
    verdict = json.loads((tmp_path / "preflight_verdict.json").read_text(encoding="utf-8"))
    assert decision == verdict["decision"] == "launch" and verdict["projected_batch_hours"] == 11.5


def test_acceleration_bench_measures_every_variant_without_failing(tmp_path):
    """docs/e3/DECISIONS.md, 2026-09-30 (8): the bench times the current step, the compiled one and the stacked
    seeds, each with its loss difference from the current step; a variant that cannot run here (inductor without
    a C++ compiler) is reported, never fatal. Tiny model, 2 steps per phase: the mechanics only."""
    import copy

    import torch

    cfg = copy.deepcopy(preflight.run_mini.load_mini_config())
    cfg["render"]["k"] = 2
    cfg["model"].update(n_vars=3, n_classes=4, h_dim=16, embed_dim=16, base_depth=4)
    cfg["train"].update(batch_size=2, seq_len=8)
    out = preflight.bench_accelerations(cfg, torch.device("cpu"), n_steps=2, n_seeds=2)
    assert set(out) >= {"current", "compile_default", "compile_reduce_overhead", "stacked_2_seeds"}
    assert all(r > 0 for r in out["current"]["seed_steps_per_s"].values())
    stacked = out["stacked_2_seeds"]
    assert "error" not in stacked and stacked["max_rel_loss_diff"] < 1e-3 and set(stacked["speedup"]) == {"probabilities", "straight-through"}
    for name in ("compile_default", "compile_reduce_overhead"):
        assert "error" in out[name] or out[name]["max_rel_loss_diff"] < 1e-3
    json.dumps(preflight.run_mini._jsonable(out))

def test_execution_mode_refusals(monkeypatch):
    """DECISIONS 2026-09-30 (9), C: mixed modes in a campaign, compile outside Linux / CUDA or with a failed gate, and
    stacked seeds are all refused -- no silent fallback."""
    import types

    from helplessness.agents.worldmodel import acceleration as acc

    cpu, cuda = types.SimpleNamespace(type="cpu"), types.SimpleNamespace(type="cuda")
    eager = {"runtime": {"device": "cuda"}, "train": {"batch_size": 2, "seq_len": 8}, "render": {"k": 2}}
    compiled = {**eager, "runtime": {"device": "cuda", "compile": {"backend": "inductor", "mode": "reduce-overhead"}}}
    assert preflight.execution_mode_problems(eager, cuda, {"a": eager, "b": eager}, "linux") == ([], {})
    problems, _ = preflight.execution_mode_problems(eager, cuda, {"env": eager, "e30": compiled}, "linux")
    assert len(problems) == 1 and "mixes execution modes" in problems[0]
    problems, _ = preflight.execution_mode_problems(compiled, cuda, {"a": compiled}, "win32")
    assert len(problems) == 1 and "refused" in problems[0] and "Linux / CUDA" in problems[0]
    problems, _ = preflight.execution_mode_problems(compiled, cpu, {"a": compiled}, "linux")
    assert len(problems) == 1 and "refused" in problems[0]
    stacked = {**eager, "runtime": {"device": "cuda", "stacked_seeds": 2}}
    assert "no runner trains stacked seeds" in preflight.execution_mode_problems(stacked, cuda, {"a": stacked}, "linux")[0][0]
    # on Linux / CUDA the gate decides
    monkeypatch.setattr(preflight, "run_mini", types.SimpleNamespace(build_rssm=lambda cfg: None))
    calls = []

    def gate(make_model, shape, device, backend, mode, k):
        calls.append((shape, backend, mode, k))
        return {"passed": False, "error": "compiled losses differ from eager by 3.0e-02 (> 0.0001)"}

    monkeypatch.setattr(acc, "check_compile_equivalence", gate)
    problems, result = preflight.execution_mode_problems(compiled, cuda, {"a": compiled}, "linux")
    assert calls == [((2, 8), "inductor", "reduce-overhead", 2)] and len(problems) == 1 and "gate failed" in problems[0] and not result["passed"]
    monkeypatch.setattr(acc, "check_compile_equivalence", lambda *a, **k: {"passed": True, "error": None})
    assert preflight.execution_mode_problems(compiled, cuda, {"a": compiled}, "linux")[0] == []
