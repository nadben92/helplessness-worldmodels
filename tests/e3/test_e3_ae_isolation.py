"""Autoencoder isolation test, short version (docs/e3/DECISIONS.md,
2026-09-24): encoder -> categorical posterior -> decoder on the 144
isolated state images of the mini config (k=6), same weighted
reconstruction loss as the world model, no KL / prior / GRU / dynamics,
trained with the RETAINED procedure ("warmup_probs_until_separated": the
decoder receives the posterior probabilities until the label-free
batch-pair JS criterion fires -- 3 bits, 5 consecutive evaluations every
50 steps, not before step 500 -- then plain straight-through samples).
Criterion: hard-sample template accuracy >= 95%.

Short version of the comparison (experiments/E3_worldmodel_4zones/compare_estimators.py,
docs/e3/ESTIMATOR_COMPARISON.md): same rule and the same deadline as
there (3600 steps), but training stops 500 steps after the EFFECTIVE
switch instead of at 6000 steps. In the comparison, seed 0 switched at
step 2301 and ended at 0.999. lr=1e-3: the isolation test's own optimizer
setting (the comparison's), not a calibration hyperparameter.

Plain straight-through and Gumbel-softmax ST fail this test at chance
(0.007 on all 3 seeds of the comparison); they are not re-run here.
"""

from __future__ import annotations

from _experiment_loader import load_experiment_module

ae_isolation = load_experiment_module("E3_worldmodel_4zones", "ae_isolation.py", "test_ae_isolation_module")

N_STEPS_CAP = 6000
SWITCH_DEADLINE = 3600  # the comparison's deadline (60% of 6000)
STOP_AFTER_SWITCH = 500
LR = 1e-3
BATCH = 32


def test_categorical_autoencoder_with_the_retained_procedure_represents_all_144_states():
    result = ae_isolation.run(
        N_STEPS_CAP, LR, BATCH, seed=0, estimator="warmup_probs_until_separated",
        switch_max_steps=SWITCH_DEADLINE, stop_after_switch=STOP_AFTER_SWITCH, eval_every=N_STEPS_CAP,
    )
    assert result["k"] == 6
    assert result["switch_step"] is not None
    assert result["curve"][-1]["step"] == result["switch_step"] + STOP_AFTER_SWITCH
    assert result["accuracy"] >= ae_isolation.CRITERION, (result["switch_step"], result["final"])
