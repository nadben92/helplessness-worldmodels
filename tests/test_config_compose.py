"""Tests for helplessness.config.compose(): full resolution (no dangling
block references in the output) and the hard frozen-block guarantee (an
override under FROZEN_BLOCKS must raise, never merge silently)."""

from __future__ import annotations

import pytest

from helplessness.config import FrozenConfigError, compose, load_yaml_block
from helplessness.envs.dynamics.factory import dynamics_from_config
from helplessness.envs.dynamics.controllable import ControllableDeterministic
from helplessness.envs.appearance.themes import TEST_THEME_SEED_START


def test_compose_merges_overrides_into_blocks():
    blocks = {"zones": {"a": {"dynamics": {"type": "ControllableDeterministic"}}}}
    overrides = {"grid_shape": [8, 8], "zones": {"a": {"appearance": {"category": "A"}}}}
    resolved = compose(blocks, overrides)
    assert resolved["grid_shape"] == [8, 8]
    assert resolved["zones"]["a"]["dynamics"]["type"] == "ControllableDeterministic"
    assert resolved["zones"]["a"]["appearance"]["category"] == "A"


def test_compose_result_is_a_fully_independent_copy():
    """The result must not alias the input blocks -- mutating it afterwards
    must never change the shared block dict a second compose() call reads."""
    blocks = {"zones": {"reference": {"dynamics": {"type": "ControllableDeterministic"}}}}
    resolved = compose(blocks, {})
    resolved["zones"]["reference"]["dynamics"]["type"] = "MUTATED"
    assert blocks["zones"]["reference"]["dynamics"]["type"] == "ControllableDeterministic"


def test_compose_rejects_override_of_a_frozen_block_exactly():
    blocks = {"zones": {"reference": {"dynamics": {"type": "ControllableDeterministic"}}}}
    with pytest.raises(FrozenConfigError, match="zones.reference"):
        compose(blocks, {"zones": {"reference": {"dynamics": {"p": 0.5}}}})


def test_compose_rejects_override_nested_under_a_frozen_block():
    blocks = {"themes": {"test_pool": {"seed_range": [900000000, 2147483646]}}}
    with pytest.raises(FrozenConfigError, match="themes.test_pool"):
        compose(blocks, {"themes": {"test_pool": {"seed_range": [0, 100]}}})


def test_compose_allows_overriding_non_frozen_siblings_of_a_frozen_block():
    blocks = {
        "zones": {
            "reference": {"dynamics": {"type": "ControllableDeterministic"}},
            "a": {"dynamics": {"type": "ControllableDeterministic"}},
        }
    }
    resolved = compose(blocks, {"zones": {"a": {"dynamics": {"type": "ControllableNoisy", "p": 0.5}}}})
    assert resolved["zones"]["a"]["dynamics"]["p"] == 0.5
    assert resolved["zones"]["reference"]["dynamics"]["type"] == "ControllableDeterministic"


def test_reference_yaml_matches_the_frozen_python_definition():
    """The checked-in configs/zones/reference.yaml must describe exactly
    the same zone as envs.zones.reference_zone() -- round-tripped through
    dynamics_from_config() rather than compared field-by-field, so this
    test would catch a YAML that no longer parses into a real spec too."""
    block = load_yaml_block("zones/reference.yaml")
    spec = dynamics_from_config(dict(block["dynamics"]))
    assert isinstance(spec, ControllableDeterministic)
    assert block["appearance"]["category"] == "reference"
    assert block["pattern_shape"] == [2, 2]


def test_test_pool_yaml_matches_the_python_constant():
    block = load_yaml_block("themes/test_pool.yaml")
    assert block["seed_range"][0] == TEST_THEME_SEED_START
