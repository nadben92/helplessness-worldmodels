"""Tests for the production experiment-loader
(helplessness.experiment_loader), the src/-side counterpart to
tests/_experiment_loader.py (see docs/e3/DECISIONS.md, point C) --
verifies it behaves identically for E3's purposes: a clean import of
experiments/E2_online_4zones/env.py, with no sys.modules leakage.
"""

from __future__ import annotations

import sys

from helplessness.experiment_loader import load_experiment_module


def test_loads_e2_env_module_with_expected_public_names():
    mod = load_experiment_module("E2_online_4zones", "env.py", "test_loader_e2_env")
    assert hasattr(mod, "build_layout")
    assert hasattr(mod, "build_control_layout")
    assert hasattr(mod, "quadrant_assignment_for_rotation")
    assert hasattr(mod, "rotation_for_seed_index")
    assert hasattr(mod, "QUADRANT_ORDER")


def test_two_loads_under_different_names_do_not_collide():
    mod1 = load_experiment_module("E2_online_4zones", "env.py", "test_loader_collision_a")
    mod2 = load_experiment_module("E2_online_4zones", "env.py", "test_loader_collision_b")
    assert mod1 is not mod2
    assert mod1.QUADRANT_ORDER == mod2.QUADRANT_ORDER


def test_does_not_leak_sibling_module_names_into_sys_modules():
    # Other test files in this suite legitimately cache sys.modules["env"]
    # via their own raw sys.path.insert + import pattern (see
    # tests/_experiment_loader.py's own docstring) -- so this test cannot
    # assume "env"/"twin" are globally absent, only that THIS loader call
    # leaves whatever was already there exactly as it found it.
    before_env = sys.modules.get("env")
    before_twin = sys.modules.get("twin")
    load_experiment_module("E2_online_4zones", "env.py", "test_loader_no_leak")
    assert sys.modules.get("env") is before_env
    assert sys.modules.get("twin") is before_twin


def test_loading_twin_py_resolves_its_own_sibling_import_of_matched_control():
    # twin.py does `from matched_control import collect_matched` at its
    # top level -- this only works if the loader leaves E2's own
    # directory on sys.path for the duration of exec_module().
    mod = load_experiment_module("E2_online_4zones", "twin.py", "test_loader_twin")
    assert hasattr(mod, "twin_from_counts")


def test_repo_root_points_at_the_actual_repository_root():
    from helplessness.experiment_loader import EXPERIMENTS_DIR, REPO_ROOT

    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (EXPERIMENTS_DIR / "E2_online_4zones" / "env.py").is_file()
