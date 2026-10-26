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
