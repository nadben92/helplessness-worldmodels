"""The continuation of a finished run (run_e5.py --extend, docs/e5/DECISIONS.md (33)), on the CPU mini configuration: a
shuffle run of 600 steps, continued to 800 from its last resume point (500). The continuation recomputes 500-600 and
gives, at 600, exactly the original's weights (bit for bit, the same device); it then trains to 800 with checkpoints past
the original grid, in a new run directory, the original untouched."""

from __future__ import annotations

import hashlib

import torch
import yaml

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT

run_e5 = load_experiment_module("E5_belief_training", "run_e5.py", "e5_run_extension_under_test")


def digest(path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for module in ("rssm", "goal_head", "critic"):
        for key in sorted(state[module]):
            h.update(key.encode())
            h.update(state[module][key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_continuation_reproduces_then_extends(tmp_path):
    common = ["--config", str(REPO_ROOT / "configs" / "e5" / "e5_mini.yaml"), "--configuration", "p1-b-d1", "--seed-label", "A",
              "--training", "shuffle", "--fraction", "1.0", "--condition", "reference", "--device", "cpu", "--out-dir", str(tmp_path)]
    assert run_e5.main(common) == 0
    root = tmp_path / "E5_belief_training"
    (original,) = list(root.glob("grid*"))
    before = sorted(p.name for p in original.rglob("*") if p.is_file())
    assert run_e5.main(["--extend", "auto", "--to", "800", "--extend-every", "100", "--extend-all-every", "200",
                        "--configuration", "p1-b-d1", "--seed-label", "A", "--training", "shuffle", "--fraction", "1.0",
                        "--condition", "reference", "--device", "cpu", "--out-dir", str(tmp_path)]) == 0
    (ext,) = [d for d in root.glob("grid*") if d != original]
    assert sorted(p.name for p in original.rglob("*") if p.is_file()) == before  # the original untouched
    assert digest(ext / "checkpoints" / "agent_env000600.pt") == digest(original / "checkpoints" / "agent_env000600.pt")
    assert (ext / "checkpoints" / "agent_env000700.pt").is_file() and (ext / "checkpoints" / "agent_env000800.pt").is_file()
    cfg = yaml.safe_load((ext / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["e5"]["run"]["extend_to"] == 800 and cfg["e5"]["run"]["extended_from"] == str(original)
    assert cfg["budget"] == yaml.safe_load((original / "config.yaml").read_text(encoding="utf-8"))["budget"]  # the same planned budget
    meta = yaml.safe_load((ext / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta["branched_from"].endswith("resume.pt")
    # the next continuation starts from the latest finished one (its own resume point), not from the original again
    assert run_e5.original_run(tmp_path, "p1-b-d1", "A", "shuffle", "reference", 1.0) == ext
    assert cfg["e5"]["run"]["extension_chain"] == [str(original)]


def test_continuation_of_a_resume_state_without_the_row_offset_equals_the_continuous_run(tmp_path):
    """docs/e5/DECISIONS.md (34): the resume states of waves 1 and 2 were written before the current episode's row offset
    was saved (cf9ef06); e5_extend_a, branched from them, stopped at its first step on ObservationTableMismatch (the
    episode in progress, of a non-canonical assignment, looked up in block 0). A shuffle run whose last resume point (500)
    falls in such an episode, its resume.pt stripped of the offset as theirs, is continued past its planned length (to
    800); the continuation equals a run trained to 800 in one go: every stored step (observation row, state, action,
    reward, flags), the episodes' assignments, the weights and the training losses at 600, 700 and 800."""
    import numpy as np

    from helplessness.pipeline.agent_run import run_one
    from helplessness.pipeline.runtime import configure_runtime

    common = ["--configuration", "p1-b-d1", "--seed-label", "A", "--training", "shuffle", "--fraction", "1.0",
              "--condition", "experimental", "--device", "cpu", "--out-dir", str(tmp_path)]
    assert run_e5.main(["--config", str(REPO_ROOT / "configs" / "e5" / "e5_mini.yaml"), *common]) == 0
    root = tmp_path / "E5_belief_training"
    (original,) = list(root.glob("grid*"))
    state_path = original / "checkpoints" / "resume.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["progress"]["env_step"] == 500 and state["collector"]["obs"] is not None  # in the middle of an episode
    assert state["collector"]["obs_offset"] != 0  # of an assignment whose block is not the table's first
    del state["collector"]["obs_offset"]  # the resume state as waves 1 and 2 wrote it
    torch.save(state, state_path)
    assert run_e5.main(["--extend", str(original), "--to", "800", "--extend-every", "100", "--extend-all-every", "200", *common]) == 0
    (ext,) = [d for d in root.glob("grid*") if d != original]

    cfg = run_e5.e5_setup.load_config(ext / "config.yaml")
    for key in ("extended_from", "extension_chain"):
        cfg["e5"]["run"].pop(key)
    continuous = tmp_path / "continuous"
    (continuous / "checkpoints").mkdir(parents=True)
    conf, condition, hooks, kwargs = run_e5.build(cfg)
    assert kwargs == {"total_env_steps": 800}
    run_one(cfg, conf["seed"], condition, continuous, configure_runtime(cfg), hooks=hooks, **kwargs)

    a, b = dict(np.load(ext / "replay_final.npz")), dict(np.load(continuous / "replay_final.npz"))
    assert sorted(a) == sorted(b) and "obs" in a and len(a["s"]) > 800
    assert all(np.array_equal(a[k], b[k]) for k in a)
    a, b = dict(np.load(ext / "assignments.npz")), dict(np.load(continuous / "assignments.npz"))
    assert all(np.array_equal(a[k], b[k]) for k in a) and len(set(a["episode_assignment"].tolist())) > 1
    for step in (600, 700, 800):
        assert digest(ext / "checkpoints" / f"agent_env{step:06d}.pt") == digest(continuous / "checkpoints" / f"agent_env{step:06d}.pt"), step
    assert digest(ext / "checkpoints" / "agent_env000600.pt") == digest(original / "checkpoints" / "agent_env000600.pt")
    losses = {d: {r["env_step"]: r["train"] for r in run_e5.e5_setup.records(d)} for d in (ext, continuous)}
    assert [600, 700, 800] == sorted(losses[ext]) and all(losses[ext][s] == losses[continuous][s] for s in (600, 700, 800))
