"""helplessness.cluster.batch_status: the state of a submitted batch from its supervisor directory, with the STALE flag
(a running job whose log has not changed for more than an hour), and --status in E4's and E5's submit scripts."""

from __future__ import annotations

import json
import os
import time

from _experiment_loader import load_experiment_module
from helplessness.cluster import batch_status

submit_e4 = load_experiment_module("E4_generalization_gradient", "submit_kubeflow.py", "test_status_submit_e4")
submit_e5 = load_experiment_module("E5_belief_training", "submit_kubeflow.py", "test_status_submit_e5")


def supervisor(tmp_path, now: float):
    out = tmp_path / "e5-train-e5_wave1-x"
    out.mkdir()
    status = {"fresh": {"pid": 11, "exit_code": None, "started_at": "t"}, "frozen": {"pid": 12, "exit_code": None, "started_at": "t"},
              "ok": {"pid": 13, "exit_code": 0}, "crashed": {"pid": 14, "exit_code": 1}, "queued": {"exit_code": None}}
    (out / "status.json").write_text(json.dumps(status), encoding="utf-8")
    for name, age in (("fresh", 60), ("frozen", 2 * 3600), ("ok", 5 * 3600), ("crashed", 5 * 3600)):
        (out / f"{name}.log").write_text("line 1\nlast line\n", encoding="utf-8")
        os.utime(out / f"{name}.log", (now - age, now - age))
    return out


def test_states_and_stale_flag(tmp_path):
    now = time.time()
    out = supervisor(tmp_path, now)
    rows = {r["job"]: r for r in batch_status.batch_status(out, now=now)}
    assert {k: r["state"] for k, r in rows.items()} == {"fresh": "running", "frozen": "STALE", "ok": "done", "crashed": "failed",
                                                         "queued": "not started"}
    assert rows["frozen"]["last_line"] == "last line" and 7190 <= rows["frozen"]["log_age_seconds"] <= 7210
    assert [r["job"] for r in batch_status.batch_status(out, stale_after=3 * 3600, now=now) if r["state"] == "STALE"] == []
    lines = batch_status.format_status(list(rows.values()))
    assert sum(line.startswith("STALE frozen") for line in lines) == 1  # done / failed jobs are never stale


def test_status_option_of_both_submit_scripts(tmp_path, capsys):
    out = supervisor(tmp_path, time.time())
    for submit in (submit_e4, submit_e5):
        assert submit.main(["--status", str(out)]) == 2  # a STALE job
        assert "STALE frozen" in capsys.readouterr().out
    (out / "status.json").write_text(json.dumps({"ok": {"pid": 1, "exit_code": 0}}), encoding="utf-8")
    assert submit_e5.main(["--status", str(out)]) == 0
