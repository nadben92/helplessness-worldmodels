"""run_e7_diagnostics.phase_episodes' array keys (fixed 2026-11-06, docs/e8/DECISIONS.md (19)): the lookahead's episodes of
the exposed model and of its reference go under distinct keys (before, one key: the reference's overwrote the exposed
model's); the induction's and the softmax's keys are unchanged."""

from __future__ import annotations

from _experiment_loader import load_experiment_module

rdiag = load_experiment_module("E7_latent_manipulation", "run_e7_diagnostics.py", "test_e7_lookahead_keys_rdiag")


def test_lookahead_keys_differ_by_model():
    plan = [{"mode": "none", "alpha": 0.0, "role": role, "vector": None, "tau": None, "k": k}
            for k in (1, 5, 10) for role in ("experimental", "reference")]  # phase_episodes' lookahead plan
    tags = [rdiag.episode_tag(it) for it in plan]
    assert len(set(tags)) == len(tags)
    assert rdiag.episode_tag(plan[0]) == "none__0__k1__exposed" and rdiag.episode_tag(plan[1]) == "none__0__k1__reference"


def test_other_phases_keep_their_keys():
    assert rdiag.episode_tag({"mode": "reverse", "alpha": 0.5, "role": "reference", "tau": None, "k": None}) == "reverse__0.5"
    assert rdiag.episode_tag({"mode": "none", "alpha": 0.0, "role": "experimental", "tau": 1.57, "k": None}) == "none__0__tau1.57"
