"""action_in_latent.analyse (docs/e3/DECISIONS.md, 2026-09-29) on synthetic
draws: a code that encodes the action is detected beyond what the states
tell; a code that does not, is not."""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

ail = load_experiment_module("E3_worldmodel_4zones", "action_in_latent.py", "test_action_in_latent")


def _samples(z_encodes_action: bool, n: int = 4000, seed: int = 0) -> dict:
    """An uncontrollable zone: states 0-1, the arrival (2 or 3) independent of the action."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    a = rng.integers(0, 5, n)
    s_next = 2 + rng.integers(0, 2, n)
    z = rng.integers(0, 4, (n, 3))
    if z_encodes_action:
        z[:, 0] = a % 4
        z[:, 1] = np.where(a == 4, 3, rng.integers(0, 3, n))
    return {"s": s, "a": a, "s_next": s_next, "z_index": z, "is_ghost": np.zeros(n, dtype=bool)}


def test_a_code_that_encodes_the_action_is_detected_beyond_the_states():
    zones = ["zone", "zone", "other", "other"]
    enc = ail.analyse(_samples(True), zones, ["zone"], n_vars=3, n_classes=4, n_actions=5, rng=np.random.default_rng(1))["zone"]
    assert enc["n_groups"] == 2
    assert abs(enc["states_only"] - 0.2) < 0.05  # the arrival does not depend on the action
    assert enc["z_given_s_next"] > 0.9 and enc["z_and_s_given_s_next"] > 0.9
    rand = ail.analyse(_samples(False), zones, ["zone"], n_vars=3, n_classes=4, n_actions=5, rng=np.random.default_rng(1))["zone"]
    assert abs(rand["z_given_s_next"] - 0.2) < 0.06 and abs(rand["z_and_s_given_s_next"] - 0.2) < 0.06


def test_small_groups_and_ghosts_are_left_out():
    samples = _samples(True, n=200)
    samples["is_ghost"][:150] = True  # 50 draws left, split over 2 arrival states: both groups below MIN_GROUP
    res = ail.analyse(samples, ["zone", "zone", "o", "o"], ["zone"], 3, 4, 5, np.random.default_rng(0))["zone"]
    assert res["n_groups"] == 0 and np.isnan(res["z_given_s_next"])
