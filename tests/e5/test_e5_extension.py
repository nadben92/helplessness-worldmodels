"""The continuation of a finished run (run_e5.py --extend, docs/e5/DECISIONS.md (33)), on the CPU mini configuration: a
shuffle run of 600 steps, continued to 800 from its last resume point (500). The continuation recomputes 500-600 and
gives, at 600, exactly the original's weights (bit for bit, the same device); it then trains to 800 with checkpoints past
the original grid, in a new run directory, the original untouched. And (34), (35): a continuation of a resume state
written before the row offset was saved, and a resumed run, equal the run trained in one go, checkpoint records
included; recompute_checkpoints.py restores the canonical records the bug of (35) left wrong."""

from __future__ import annotations

import hashlib

import pytest
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


MINI = ["--configuration", "p1-b-d1", "--seed-label", "A", "--training", "shuffle", "--fraction", "1.0", "--condition", "experimental",
        "--device", "cpu"]


def without_seconds(x):
    """A record without its durations (they differ between two runs of the same computation)."""
    if isinstance(x, dict):
        return {k: without_seconds(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, list):
        return [without_seconds(v) for v in x]
    return x


def records(run_dir) -> dict:
    return {r["env_step"]: without_seconds(r) for r in run_e5.e5_setup.records(run_dir)}


@pytest.fixture(scope="module")
def shuffle_runs(tmp_path_factory):
    """An experimental f = 1 mini run (600 steps) whose last resume point (500) falls in an episode of an assignment whose
    UD zone is not the canonical one's (nw), its resume.pt stripped of the row offset as waves 1 and 2's; its
    continuation to 800 (--extend); the same run trained to 800 in one go ("continuous", the continuation's config); and
    that run interrupted after its resume point at 500 and resumed (--resume)."""
    from helplessness.pipeline.agent_run import Interrupted, run_one
    from helplessness.pipeline.episode_sources import ASSIGNMENTS
    from helplessness.pipeline.runtime import configure_runtime

    tmp = tmp_path_factory.mktemp("e5_ext")
    assert run_e5.main(["--config", str(REPO_ROOT / "configs" / "e5" / "e5_mini.yaml"), *MINI, "--out-dir", str(tmp)]) == 0
    root = tmp / "E5_belief_training"
    (original,) = list(root.glob("grid*"))
    state_path = original / "checkpoints" / "resume.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["progress"]["env_step"] == 500 and state["collector"]["obs"] is not None  # in the middle of an episode
    current = state["collector"]["obs_offset"] // run_e5.N_STATES
    assert current != 0 and ASSIGNMENTS[current][0] != "nw"  # not the table's first block; its UD zone not the canonical one's
    del state["collector"]["obs_offset"]  # the resume state as waves 1 and 2 wrote it
    torch.save(state, state_path)
    assert run_e5.main(["--extend", str(original), "--to", "800", "--extend-every", "100", "--extend-all-every", "200", *MINI,
                        "--out-dir", str(tmp)]) == 0
    (ext,) = [d for d in root.glob("grid*") if d != original]

    def fresh(name: str):
        cfg = run_e5.e5_setup.load_config(ext / "config.yaml")
        for key in ("extended_from", "extension_chain"):
            cfg["e5"]["run"].pop(key)
        run_dir = tmp / name
        (run_dir / "checkpoints").mkdir(parents=True)
        conf, condition, hooks, kwargs = run_e5.build(cfg)
        assert kwargs == {"total_env_steps": 800}
        return cfg, conf, condition, hooks, kwargs, run_dir

    cfg, conf, condition, hooks, kwargs, continuous = fresh("continuous")
    run_one(cfg, conf["seed"], condition, continuous, configure_runtime(cfg), hooks=hooks, **kwargs)
    cfg, conf, condition, hooks, kwargs, resumed = fresh("resumed")
    with pytest.raises(Interrupted):
        run_one(cfg, conf["seed"], condition, resumed, configure_runtime(cfg), hooks=hooks, interrupt_after=500, **kwargs)
    assert run_e5.main(["--resume", str(resumed), "--device", "cpu"]) == 0
    return {"original": original, "extension": ext, "continuous": continuous, "resumed": resumed}


def assert_same_run(a, b, steps):
    """The same training (every stored step, the episodes' assignments, the weights and the losses) and the same
    checkpoint records (e4 and e5, every assignment measured, durations aside) at `steps`."""
    import numpy as np

    x, y = dict(np.load(a / "replay_final.npz")), dict(np.load(b / "replay_final.npz"))
    assert sorted(x) == sorted(y) and "obs" in x and all(np.array_equal(x[k], y[k]) for k in x)
    x, y = dict(np.load(a / "assignments.npz")), dict(np.load(b / "assignments.npz"))
    assert all(np.array_equal(x[k], y[k]) for k in x) and len(set(x["episode_assignment"].tolist())) > 1
    ra, rb = records(a), records(b)
    for step in steps:
        assert digest(a / "checkpoints" / f"agent_env{step:06d}.pt") == digest(b / "checkpoints" / f"agent_env{step:06d}.pt"), step
        assert ra[step]["train"] == rb[step]["train"], step
        assert ra[step]["e4"] == rb[step]["e4"], step
        assert ra[step]["e5"] == rb[step]["e5"], step
        assert ra[step] == rb[step], step


def test_continuation_of_an_old_resume_state_equals_the_continuous_run(shuffle_runs):
    """docs/e5/DECISIONS.md (34), (35): the continuation of a resume state without the row offset (waves 1-2's) equals the
    run trained to 800 in one go -- its training AND its checkpoint records: before (35) the canonical assignment's real
    context was measured against the dynamics of the episode in progress at the restore (the run's world's environment
    replaced by the collector's)."""
    assert sorted(records(shuffle_runs["extension"])) == [600, 700, 800]
    assert_same_run(shuffle_runs["extension"], shuffle_runs["continuous"], (600, 700, 800))
    assert (digest(shuffle_runs["extension"] / "checkpoints" / "agent_env000600.pt")
            == digest(shuffle_runs["original"] / "checkpoints" / "agent_env000600.pt"))


def test_resumed_run_equals_the_continuous_run(shuffle_runs):
    """docs/e5/DECISIONS.md (35): a run interrupted after its resume point at 500 and resumed (--resume) equals the
    uninterrupted run, checkpoint records included."""
    steps = sorted(records(shuffle_runs["continuous"]))
    assert sorted(records(shuffle_runs["resumed"])) == steps
    assert_same_run(shuffle_runs["resumed"], shuffle_runs["continuous"], steps)


def test_recompute_restores_the_canonical_record_and_check_wave_reads_it(shuffle_runs, tmp_path):
    """recompute_checkpoints.py (docs/e5/DECISIONS.md (35)): on a copy of the continuation whose canonical masses are
    overwritten (as the bug left them), the recomputation from the weights and the replay truncated at each checkpoint
    (resume.pt at 700 for 600 and 700, replay_final.npz for 800) reproduces every field independent of the world exactly
    and gives back the right canonical records, the continuous run's; check_wave.py reads them, checkpoints.jsonl
    untouched."""
    import json
    import shutil

    from helplessness.pipeline.episode_sources import canonical_assignment

    recompute = load_experiment_module("E5_belief_training", "recompute_checkpoints.py", "e5_recompute_under_test")
    cw = load_experiment_module("E5_belief_training", "check_wave.py", "e5_check_wave_recompute")
    run = tmp_path / shuffle_runs["extension"].name
    shutil.copytree(shuffle_runs["extension"], run)
    canonical = canonical_assignment(1)
    lines = []
    for r in run_e5.e5_setup.records(run):
        for rc in (r["e4"]["real_context"], r["e5"]["by_assignment"][str(canonical)]["real_context"]):
            rc["mass_true_by_region"] = {q: 0.2 for q in rc["mass_true_by_region"]}
        lines.append(json.dumps(r) + "\n")
    (run / "checkpoints.jsonl").write_text("".join(lines), encoding="utf-8")
    corrupted = (run / "checkpoints.jsonl").read_text(encoding="utf-8")
    out = recompute.recompute(run, None, "cpu")
    assert [x["env_step"] for x in out] == [600, 700, 800] and all(x["reproduced"] for x in out)
    assert all(x["largest_difference_unaffected"] == 0.0 for x in out)
    assert [x["replay_source"] for x in out] == ["checkpoints/resume.pt", "checkpoints/resume.pt", "replay_final.npz"]
    right = records(shuffle_runs["continuous"])
    for x in out:
        assert x["e4"]["real_context"] == right[x["env_step"]]["e4"]["real_context"]
    assert (run / "checkpoints.jsonl").read_text(encoding="utf-8") == corrupted
    read = cw.read_run(run)
    assert read["recomputed"] == [600, 700, 800]
    got = {r["env_step"]: r for r in read["records"]}
    assert got[800]["e4"]["real_context"] == right[800]["e4"]["real_context"]
    assert got[800]["e5"]["by_assignment"][str(canonical)] == right[800]["e5"]["by_assignment"][str(canonical)]
    assert got[800]["e5"]["mean_over_assignments"] == right[800]["e5"]["mean_over_assignments"]
