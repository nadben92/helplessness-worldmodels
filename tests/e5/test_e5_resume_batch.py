"""E5's submit_kubeflow.py --phase resume without --resume (helplessness.cluster.resume, shared with E4): every run of the
batch not done, in one pod, each with the single run's command; the read-only check of the resume points, loaded on the
CPU, before anything is submitted (no cluster involved)."""

from __future__ import annotations

import json
import pickle
import shutil

import pytest
import torch

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT

submit = load_experiment_module("E5_belief_training", "submit_kubeflow.py", "test_e5_submit_resume")
batches = submit.batches
STAMP = "20261022T000000Z"
CONF = "p1-b-d1"


def make_run(repo, job: str, step: int | None, status: str = "running", final: bool = False):
    label, training, fraction, condition = batches.batch_runs("e5_wave1")[job]
    seed = submit.seeds(repo, CONF)[label]
    name = submit.e5_names.run_glob(CONF, training, condition, seed, fraction).replace("*", "grid12x12", 1).replace("*", STAMP)
    run = repo / "results" / "E5_belief_training" / name
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (run / "meta.yaml").write_text(f"status: {status}\nseed: {seed}\n", encoding="utf-8")
    if final:
        (ck / "agent_final.pt").write_bytes(b"w")
    if step is not None:
        state = {k: {} for k in submit.cluster_resume.RESUME_KEYS}
        state.update(progress={"env_step": step}, n_env_steps=300000, planned_env_steps=300000,
                     replay=pickle.dumps(list(range(5))), env=pickle.dumps({"env": 1}))
        torch.save(state, ck / "resume.pt")
        (ck / "latest.json").write_text(json.dumps({"filename": "resume.pt", "step": step}), encoding="utf-8")
    return run


@pytest.fixture
def repo(tmp_path):
    for rel in ("configs/e5/e5.yaml", "configs/e4/e4.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    jobs = list(batches.batch_runs("e5_wave1"))
    for i, job in enumerate(jobs):
        make_run(tmp_path, job, None if i == 3 else 160000 + 10000 * i, status="done" if i == 3 else "running", final=i == 3)
    return tmp_path


def test_batch_resume_of_e5(repo, capsys):
    jobs = list(batches.batch_runs("e5_wave1"))
    runs, checks = submit.resume_plan(repo, "e5_wave1", CONF, "")
    assert list(runs) == jobs[:3] and [c["resume_from_env_step"] for c in checks[:3]] == [160000, 170000, 180000] and checks[3]["done"]
    seed_of = {"A": "seed02", "B": "seed102"}  # p1-b-d1's seeds A and B; wave 1 holds both (docs/e5/DECISIONS.md (29))
    assert all(f"__{seed_of[batches.batch_runs('e5_wave1')[job][0]]}__" in str(d) for job, d in runs.items())
    batch_jobs = submit.commands("resume", "PY", "e5_wave1", CONF, resume_runs=runs)
    assert [n for n, _ in batch_jobs] == [f"resume_{n}" for n in jobs[:3]]
    for (_, cmd), d in zip(batch_jobs, runs.values()):
        assert cmd == submit.commands("resume", "PY", "e5_wave1", CONF, resume=d)[0][1]
    assert submit.main(["--batch", "e5_wave1", "--configuration", CONF, "--phase", "resume", "--repo-dir", str(repo), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "resumes from environment step 170000" in out and "done, not relaunched" in out and '"resume_A_sh100_experimental"' in out


def test_a_bad_resume_point_stops_everything_and_done_runs_are_refused(repo, capsys):
    jobs = list(batches.batch_runs("e5_wave1"))
    runs = submit.batch_run_dirs_by_job(repo, "e5_wave1", CONF)
    resume = runs[jobs[1]] / "checkpoints" / "resume.pt"
    resume.write_bytes(resume.read_bytes()[:200])
    assert submit.main(["--batch", "e5_wave1", "--configuration", CONF, "--phase", "resume", "--repo-dir", str(repo), "--dry-run"]) == 1
    assert "RESUME CHECK FAILED -- nothing submitted" in capsys.readouterr().out
    done = runs[jobs[3]].relative_to(repo).as_posix()
    assert submit.main(["--batch", "e5_wave1", "--phase", "resume", "--resume", done, "--repo-dir", str(repo), "--dry-run"]) == 1
    assert "never relaunched" in capsys.readouterr().out
