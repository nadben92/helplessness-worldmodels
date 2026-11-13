"""The batch table after DECISIONS (23): wave 2 is launchable with its own config, which differs from wave 1's only by the
budget, the checkpoints and the oracle checkpoints; wave 1's command lines carry no --config (unchanged); the later
quadrants stay blocked."""

from __future__ import annotations

import copy

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT

batches = load_experiment_module("E8_therapy_relapse", "batches.py", "test_e8_batches_table")
e8_setup = load_experiment_module("E8_therapy_relapse", "e8_setup.py", "test_e8_batches_setup")

W2_GRID = [0, 500, 1000, 2000, 3500, 5000, 7500, 10000, 15000, 25000, 40000, 60000, 100000, 150000, 200000, 250000]


def test_launchable_and_command_lines():
    assert batches.LAUNCHABLE == ("e8_w1_base", "e8_w2_shuf_nw")
    assert all(b not in batches.LAUNCHABLE for b in ("e8_w3_shuf_ne", "e8_w4_shuf_sw", "e8_w5_shuf_se"))
    assert batches.train_args("e8_w1_base")["base_A_exp"] == ["--phase", "train", "--model", "base_A_exp"]
    assert batches.train_args("e8_w2_shuf_nw")["shuffle_B_ref"] == ["--phase", "train", "--model", "shuffle_B_ref", "--treated-quadrant", "nw",
                                                                    "--config", "configs/e8/e8_w2.yaml"]
    assert batches.config_args("e8_w3_shuf_ne") == []


def test_wave2_config_differs_from_wave1_by_three_values_only():
    w1 = e8_setup.load_config(REPO_ROOT / "configs" / "e8" / "e8.yaml")
    w2 = e8_setup.load_config(REPO_ROOT / batches.BATCH_CONFIG["e8_w2_shuf_nw"])
    assert e8_setup.checkpoint_grid(w2) == W2_GRID
    assert w2["e8"]["max_env_steps"] == 250000
    assert w2["e8"]["probes"]["oracle_checkpoints"] == [0, 2000, 5000, 10000, 25000, 60000, 150000, 250000]
    patched = copy.deepcopy(w1)
    patched["e8"].update(max_env_steps=250000, checkpoints=W2_GRID)
    patched["e8"]["probes"]["oracle_checkpoints"] = w2["e8"]["probes"]["oracle_checkpoints"]
    assert patched == w2
