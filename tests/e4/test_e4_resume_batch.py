"""submit_kubeflow.py --phase resume without --resume: every run of the batch not done, in one pod, each with the single
run's command; the read-only check of the resume points before anything is submitted (no cluster involved)."""

from __future__ import annotations

import json
import pickle

import pytest
import torch

from _experiment_loader import load_experiment_module

submit = load_experiment_module("E4_generalization_gradient", "submit_kubeflow.py", "test_e4_submit_resume")
batches = submit.batches
STAMP = "20261018T000000Z"


def make_run(repo, cid: str, condition: str, status: str = "running", step: int | None = 170000, latest_step: int | None = None,
             final: bool = False):
    role = "experimental" if condition == "experimental" else "control"
    run = repo / "results" / "E4_generalization_gradient" / f"grid12x12__2zones-{cid}__pair-{role}__worldmodel-agent-e4__seed03__{STAMP}"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (run / "meta.yaml").write_text(f"status: {status}\nseed: 3\n", encoding="utf-8")
    (ck / "agent_env150000.pt").write_bytes(b"w")
    if final:
        (ck / "agent_final.pt").write_bytes(b"w")
    if step is not None:
        state = {k: {} for k in submit.RESUME_KEYS}
        state.update(progress={"env_step": step, "schedule": [300000, []]}, n_env_steps=300000, planned_env_steps=300000,
                     replay=pickle.dumps(list(range(10))), env=pickle.dumps({"env": 1}))
        torch.save(state, ck / "resume.pt")
        (ck / "latest.json").write_text(json.dumps({"filename": "resume.pt", "step": step if latest_step is None else latest_step}),
                                        encoding="utf-8")
    return run


@pytest.fixture
def repo(tmp_path):
    names = list(batches.batch_runs("e4_wave2").items())
    for i, (_, (cid, condition)) in enumerate(names):
        make_run(tmp_path, cid, condition, step=163000 + 2000 * i if i < 3 else None, status="done" if i == 3 else "running",
                 final=i == 3)
    return tmp_path


def test_batch_resume_skips_done_runs_and_uses_the_single_run_command(repo, capsys):
    runs, checks = submit.resume_plan(repo, "e4_wave2", "")
    jobs = list(batches.batch_runs("e4_wave2"))
    assert list(runs) == jobs[:3] and all(not c["problems"] for c in checks)
    assert [c["resume_from_env_step"] for c in checks[:3]] == [163000, 165000, 167000] and checks[3]["done"]
    batch_jobs = submit.commands("resume", "PY", "e4_wave2", resume_runs=runs)
    assert [n for n, _ in batch_jobs] == [f"resume_{n}" for n in jobs[:3]]
    for (_, cmd), d in zip(batch_jobs, runs.values()):
        assert cmd == submit.commands("resume", "PY", "e4_wave2", resume=d)[0][1]
    assert submit.main(["--batch", "e4_wave2", "--phase", "resume", "--repo-dir", str(repo), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "resumes from environment step 163000" in out and "done, not relaunched" in out
    plan = json.loads(out[out.index('{\n "batch": "e4_wave2",\n "phase": "resume",\n "cpus"'):])
    assert [n for n, _ in plan["jobs"]] == [f"resume_{n}" for n in jobs[:3]]
    assert submit.decode_resume_runs(submit.encode_resume_runs(runs)) == runs


def test_single_run_resume_unchanged_and_never_a_done_run(repo, capsys):
    runs = submit.batch_run_dirs_by_job(repo, "e4_wave2")
    first, done = (str(d.relative_to(repo).as_posix()) for d in list(runs.values())[0::3])
    assert submit.commands("resume", "PY", "e4_wave2", resume=first) == [
        ("resume", ["PY", "-u", "experiments/E4_generalization_gradient/run_e4.py", "--resume", first, "--device", "cuda"])]
    assert submit.main(["--batch", "e4_wave2", "--phase", "resume", "--resume", first, "--repo-dir", str(repo), "--dry-run"]) == 0
    assert submit.main(["--batch", "e4_wave2", "--phase", "resume", "--resume", done, "--repo-dir", str(repo), "--dry-run"]) == 1
    assert "never relaunched" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["truncated", "latest_mismatch", "no_latest", "no_resume"])
def test_a_bad_resume_point_stops_everything(repo, capsys, damage):
    run = list(submit.batch_run_dirs_by_job(repo, "e4_wave2").values())[1]
    ck = run / "checkpoints"
    if damage == "truncated":
        data = (ck / "resume.pt").read_bytes()
        (ck / "resume.pt").write_bytes(data[: len(data) // 2])
    elif damage == "latest_mismatch":
        (ck / "latest.json").write_text(json.dumps({"filename": "resume.pt", "step": 160000}), encoding="utf-8")
    elif damage == "no_latest":
        (ck / "latest.json").unlink()
    else:
        (ck / "resume.pt").unlink()
    _, checks = submit.resume_plan(repo, "e4_wave2", "")
    assert [bool(c["problems"]) for c in checks] == [False, True, False, False]
    assert submit.main(["--batch", "e4_wave2", "--phase", "resume", "--repo-dir", str(repo), "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "RESUME CHECK FAILED -- nothing submitted" in out and '"jobs"' not in out


def test_the_check_loads_on_the_cpu_a_resume_point_saved_from_cuda(tmp_path, monkeypatch):
    """The notebook has no GPU: tensors saved from CUDA -- in resume.pt itself and inside its pickled replay (the
    replay's device copy of its table) -- are restored on the CPU by the check (a plain pickle.loads fails there)."""
    import torch.serialization

    run = make_run(tmp_path, "p2-a-d1", "experimental", step=None)
    with monkeypatch.context() as m:  # every storage written as if it lived on cuda:0
        m.setattr(torch.serialization, "location_tag", lambda storage: "cuda:0")
        replay = pickle.dumps({"device_table": torch.arange(6, dtype=torch.uint8)})
        state = {k: {} for k in submit.RESUME_KEYS}
        state.update(progress={"env_step": 170000, "schedule": [300000, []]}, n_env_steps=300000, planned_env_steps=300000,
                     replay=replay, env=pickle.dumps({"env": 1}), collector={"filter": {"h": torch.zeros(3)}})
        torch.save(state, run / "checkpoints" / "resume.pt")
    (run / "checkpoints" / "latest.json").write_text(json.dumps({"filename": "resume.pt", "step": 170000}), encoding="utf-8")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            pickle.loads(replay)
        with pytest.raises(RuntimeError, match="CUDA"):
            torch.load(run / "checkpoints" / "resume.pt", weights_only=False)
    check = submit.resume_check(run)
    assert check["problems"] == [] and check["resume_from_env_step"] == 170000
    loaded, replay_loaded, _ = submit.load_for_check(run / "checkpoints" / "resume.pt")
    assert loaded["collector"]["filter"]["h"].device.type == "cpu" and replay_loaded["device_table"].device.type == "cpu"
