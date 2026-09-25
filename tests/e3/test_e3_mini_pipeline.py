"""Fast smoke test for the FULL Phase-3 pipeline (canonical probe, I_z,
kernel-in-s-space, z-only probe, context dispersion, D1/D2) on a TINY
model (k=2, 3x4 latent, 40 gradient steps) trained with the retained
PROCEDURE: posterior probabilities, the label-free batch-pair JS
criterion (measures.diagnostics.mean_random_pair_jensen_shannon on the
posteriors the training step returns), the switch to straight-through
(loop.SeparationSwitch), beta_dyn from step 0 and the beta_rep ramp from
the switch (loop.rep_beta).

What it checks -- STRUCTURE ONLY: every stage runs without error; shapes;
probability simplices; 0 <= I_z <= log2(5) within MC tolerance; ghost
fractions in [0, 1]; alpha_app definedness logic; and that both training
phases and the beta_rep ramp were actually exercised.

What it does NOT check -- SEPARATION: 40 steps cannot separate 144
states, so the switch threshold is scaled down to a value that is always
met (the switch then fires at the earliest step the scaled rule allows).
Whether the states separate is checked by the categorical isolation test
(tests/e3/test_e3_ae_isolation.py, no dynamics) and, for the full RSSM,
only by run_mini.py (separation curve, SeparationNotReachedError past the
deadline). Comparative claims (conveyor vs controllable-deterministic)
are likewise run_mini.py's.
"""

from __future__ import annotations

import numpy as np
import torch

from _experiment_loader import load_experiment_module

e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "test_mini_pipeline_env")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "test_mini_pipeline_probe")
run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "test_mini_pipeline_run_mini")

import dataclasses  # noqa: E402

from helplessness.agents.worldmodel.loop import (  # noqa: E402
    SeparationSwitch,
    WorldModelLossConfig,
    collect_fixed_policy_data,
    phase_learning_rate,
    rep_beta,
    set_learning_rate,
    train_world_model_step,
)
from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon  # noqa: E402
from helplessness.agents.worldmodel.replay import ReplayBuffer  # noqa: E402
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig  # noqa: E402
from helplessness.envs.appearance.theme_sampling import build_themes  # noqa: E402
from helplessness.envs.observation.allocentric_pixel_backend import render, render_background  # noqa: E402

K = 2
N_TRAIN_STEPS = 40


def tiny_separation_switch() -> SeparationSwitch:
    """The retained switch rule, scaled to this test's 40 steps: evaluated
    every 5 steps, 2 consecutive evaluations, not before step 10, with a
    threshold that is always met (-1 bit) -- see the module docstring: this
    test exercises the procedure, it does not measure separation. For the
    same reason the post-switch collapse guard gets a threshold that is
    never met (-inf): an undertrained toy model has no separation to lose."""
    assert run_mini.load_mini_config()["train"]["latent_estimator"]["kind"] == "warmup_probs_until_separated"
    return SeparationSwitch(max_steps=N_TRAIN_STEPS, threshold_bits=-1.0, consecutive=2, eval_every=5, min_steps=10,
                            collapse_threshold_bits=float("-inf"))


LAST_PROCEDURE: dict = {}


def _tiny_setup(seed: int = 0):
    rotation = e3_env.rotation_for_seed_index(seed)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    layout = e3_env.build_layout(assignment)
    themes = build_themes(e3_env.ZONE_CATEGORIES, goal_state=e3_env.goal_state(), rng=np.random.default_rng(seed), min_dist=15.0)
    env = e3_env.build_pixel_env(layout, themes, goal_reward=50.0, seed=seed, k=K)
    background = render_background(layout, themes, k=K)
    background_t = torch.as_tensor(background, dtype=torch.float32).permute(2, 0, 1) / 255.0
    templates_np = np.stack([render(s, layout, themes, k=K) for s in range(144)])
    templates_t = torch.as_tensor(templates_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    rssm_cfg = RSSMConfig(image_size=(24, 24), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4, n_actions=5)
    rssm = RSSM(rssm_cfg)

    replay = ReplayBuffer(capacity=500, image_shape=(24, 24, 3))
    rng = np.random.default_rng(seed)
    collect_fixed_policy_data(env, replay, n_steps=200, rng=rng, epsilon=1.0)

    train_cfg = run_mini.load_mini_config()["train"]
    lr, lr_after_switch = train_cfg["lr"], train_cfg["lr_after_switch"]
    optimizer = torch.optim.Adam(rssm.parameters(), lr=lr, eps=1e-8)
    loss_cfg = WorldModelLossConfig(k=K, w_patch=10.0)
    switch = tiny_separation_switch()
    pair_rng = np.random.default_rng([seed, 4])
    rep_warmup_steps = 0
    phases, betas, lrs = [], [], []
    for step in range(N_TRAIN_STEPS):
        set_learning_rate(optimizer, phase_learning_rate(step, switch.switch_step, lr, lr_after_switch))
        lrs.append(optimizer.param_groups[0]["lr"])
        model_batch, _ = replay.sample(batch_size=4, seq_len=8, rng=rng)
        beta_rep = rep_beta(step, switch.switch_step, rep_warmup_steps, loss_cfg.beta_rep)
        step_cfg = dataclasses.replace(loss_cfg, beta_rep=beta_rep)
        phases.append(switch.estimator().kind)
        betas.append(beta_rep)
        eval_now = switch.is_eval_step(step)
        out = train_world_model_step(
            rssm, optimizer, model_batch, background_t, step_cfg, clip_norm=train_cfg["clip_norm"], estimator=switch.estimator(),
            return_posteriors=eval_now,
        )
        if eval_now:
            was_switched = switch.switched
            switch.update(step, mean_random_pair_jensen_shannon(out["post_probs"], pair_rng, n_pairs=16))
            if switch.switched and not was_switched:
                rep_warmup_steps = int(round(0.2 * (N_TRAIN_STEPS - switch.switch_step)))
        switch.check_deadline(step + 1)
    rssm.eval()
    LAST_PROCEDURE.update(switch_step=switch.switch_step, phases=phases, betas=betas, rep_warmup_steps=rep_warmup_steps,
                          lrs=lrs, lr=lr, lr_after_switch=lr_after_switch)

    return layout, themes, replay, rng, rssm, templates_t, background_t


def test_canonical_probe_shapes():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    probe = e3_probe.canonical_probe(rssm, templates_t)
    assert probe["z0_mode"].shape == (144, 3, 4)
    assert probe["prior_probs"].shape == (144, 5, 3, 4)
    assert probe["posterior_entropy"].shape == (144,)
    assert np.allclose(probe["prior_probs"].sum(axis=-1), 1.0, atol=1e-5)


def test_canonical_iz_bounds_hold_even_on_an_undertrained_model():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    iz_rng = np.random.default_rng(1)
    result = e3_probe.canonical_iz(rssm, templates_t, M=500, rng=iz_rng)
    I_z = result["I_z"]
    se = result["se"]
    assert I_z.shape == (144,)
    # generous, MC-noise-scaled tolerance (matches
    # latent_controllability()'s own internal bounds check, which
    # already passed for every state here or this would have raised) --
    # an undertrained model's per-state se can be non-negligible at
    # M=500, so a fixed 1e-6 absolute tolerance is inappropriate.
    tol = 6 * se + 1e-6
    assert np.all(I_z >= -tol)
    assert np.all(I_z <= np.log2(5) + tol)


def test_canonical_kernel_probe_produces_valid_kernels_or_all_ghost_rows():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    probe = e3_probe.canonical_probe(rssm, templates_t)
    result = e3_probe.canonical_kernel_probe(rssm, templates_t, K_s=4, margin_threshold=1e-3, probe=probe)
    assert result["p_hat"].shape == (144, 5, 144)
    assert result["ghost_fraction"].shape == (144, 5)
    row_sums = result["p_hat"].sum(axis=-1)
    all_ghost = result["ghost_fraction"] == 1.0
    assert np.all(np.isclose(row_sums, 1.0) | all_ghost)
    assert np.all((result["ghost_fraction"] >= 0.0) & (result["ghost_fraction"] <= 1.0))


def test_template_match_accuracy_is_a_valid_fraction():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    model_batch, label_batch = replay.sample(batch_size=8, seq_len=8, rng=rng)
    acc = e3_probe.template_match_accuracy(rssm, model_batch, label_batch, templates_t, margin_threshold=1e-3)
    assert 0.0 <= acc <= 1.0


def test_z_only_probe_returns_valid_accuracies():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    model_batch, label_batch = replay.sample(batch_size=4, seq_len=16, rng=rng)
    result = e3_probe.z_only_probe(rssm, model_batch, label_batch, n_states=144, test_frac=0.25, rng=rng, epochs=20)
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["n_train"] + result["n_test"] == 4 * 16


def test_context_dispersion_returns_finite_nonnegative_discrepancies():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    iz_rng = np.random.default_rng(2)
    iz_result = e3_probe.canonical_iz(rssm, templates_t, M=300, rng=iz_rng)
    model_batch, label_batch = replay.sample(batch_size=4, seq_len=16, rng=rng)
    result = e3_probe.context_dispersion(rssm, model_batch, label_batch, iz_result["I_z"], M=300, rng=rng, n_samples=10)
    assert np.all(np.isfinite(result["context_I_z"]))
    assert np.all(result["abs_discrepancy"] >= 0.0)
    assert len(result["true_states"]) == 10


def test_d1_alpha_app_defined_only_with_contrast_and_none_otherwise():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    iz_rng = np.random.default_rng(3)
    iz_result = e3_probe.canonical_iz(rssm, templates_t, M=300, rng=iz_rng)

    def zone_of(s):
        r, c = divmod(s, layout.grid_shape[1])
        return layout.zone_at((r, c)).appearance.category

    categories = list(e3_env.ZONE_CATEGORIES)
    I_by_category = {cat: float(iz_result["I_z"][[s for s in range(144) if zone_of(s) == cat]].mean()) for cat in categories}

    s0 = next(s for s in range(144) if zone_of(s) == categories[0])
    target = categories[1]

    d1 = e3_probe.d1_alpha_app_for_state(
        rssm, layout, themes, s0, target, I_by_category, M=300, rng=rng, k=K, contrast_threshold=1e9, global_swap=False,
    )
    assert d1["alpha_app"] is None  # impossible threshold -> always undefined

    d1b = e3_probe.d1_alpha_app_for_state(
        rssm, layout, themes, s0, target, I_by_category, M=300, rng=rng, k=K, contrast_threshold=-1.0, global_swap=False,
    )
    # threshold below zero -> defined unless I_L == I_Lprime exactly
    if I_by_category[categories[0]] != I_by_category[target]:
        assert d1b["alpha_app"] is not None


def test_d1_global_swap_also_runs_and_returns_a_valid_structure():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    iz_rng = np.random.default_rng(4)
    iz_result = e3_probe.canonical_iz(rssm, templates_t, M=300, rng=iz_rng)
    categories = list(e3_env.ZONE_CATEGORIES)
    I_by_category = {cat: float(iz_result["I_z"].mean()) for cat in categories}  # placeholder, structure-only test

    def zone_of(s):
        r, c = divmod(s, layout.grid_shape[1])
        return layout.zone_at((r, c)).appearance.category

    s0 = next(s for s in range(144) if zone_of(s) == categories[0])  # a real zone state, not the corridor/goal
    d1 = e3_probe.d1_alpha_app_for_state(
        rssm, layout, themes, s0, categories[1], I_by_category, M=300, rng=rng, k=K, contrast_threshold=0.0, global_swap=True,
    )
    assert "I_swap" in d1 and np.isfinite(d1["I_swap"])


def test_d2_neighbor_distances_returns_nonnegative_arrays():
    layout, themes, replay, rng, rssm, templates_t, _ = _tiny_setup()
    probe = e3_probe.canonical_probe(rssm, templates_t)
    result = e3_probe.d2_neighbor_distances(probe["posterior_probs"], layout)
    assert result["boundary"].shape[0] > 0
    assert result["within_zone"].shape[0] > 0
    assert np.all(result["boundary"] >= -1e-9)
    assert np.all(result["within_zone"] >= -1e-9)


def test_training_procedure_exercised_both_phases_and_the_beta_rep_ramp():
    _tiny_setup()
    proc = LAST_PROCEDURE
    assert proc["switch_step"] == 16  # evaluations at 10 and 15 pass -> effective from step 16
    assert set(proc["phases"][:16]) == {"probs"}
    assert set(proc["phases"][16:]) == {"straight_through"}
    assert all(b == 0.0 for b in proc["betas"][:17])  # beta_rep 0 until (and at) the switch
    assert proc["betas"][-1] == 0.1  # ramp completed (0.2 x 24 remaining steps = 5 steps)
    assert set(proc["lrs"][:16]) == {proc["lr"]}  # lr until the effective switch ...
    assert set(proc["lrs"][16:]) == {proc["lr_after_switch"]}  # ... lr_after_switch from it on
