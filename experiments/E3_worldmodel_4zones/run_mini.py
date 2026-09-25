"""Phase 3, point 3b (docs/e3/DECISIONS.md): the real end-to-end run on
configs/e3/mini.yaml. Runs TWO seeds by default -- one class-A rotation
(seed 0), one class-B rotation (seed 5, see env.py::ROTATION_CLASS and
tests/e3/test_e3_symmetry.py's T2 reformulation) -- collects fixed
epsilon=1 data (plus N[s,a,s'] counts for the tabular-twin comparison),
trains the RSSM for the full mini budget while logging per-term loss
curves, then runs every Phase-3 measure/diagnostic (canonical probe,
I_z, kernel-in-s-space, z-only probe, context dispersion, D1/D2) and
writes docs/e3/MINI_REPORT.md (+ figures under
docs/e3/mini_report_figures/), broken down by zone type and by
class A/B.

Takes roughly 2.5 hours for both seeds on this CPU-only machine at k=6
(see mini.yaml's own profiling notes) -- not part of the fast
pytest suite (see tests/e3/test_e3_mini_pipeline.py for the fast
smoke-test version of the same code paths on a much smaller budget).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\run_mini.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

import env as e3_env
import probe as e3_probe

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.agents.worldmodel.agent import categorical_mode
from helplessness.agents.worldmodel.losses import reconstruction_mask
from helplessness.agents.worldmodel.loop import (
    CollapseAfterSwitchError,
    SeparationNotReachedError,
    SeparationSwitch,
    WorldModelLossConfig,
    phase_learning_rate,
    rep_beta,
    set_learning_rate,
    train_world_model_step,
)
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, categorical_kl
from helplessness.envs.appearance.theme_sampling import build_themes
from helplessness.envs.observation.allocentric_pixel_backend import render, render_background
from helplessness.envs.zones import geographic_support_mask
from helplessness.measures.diagnostics import jensen_shannon_categorical, mean_random_pair_jensen_shannon
from helplessness.measures.latent_controllability import latent_state_information

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs" / "e3"
FIGURES_DIR = DOCS_DIR / "mini_report_figures"
FAILURES_DIR = REPO_ROOT / "results" / "E3_worldmodel_4zones" / "mini_failures"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def reproducibility_metadata() -> dict:
    """Recorded with every run (report + failure diagnostics). Bit-exact
    reproducibility holds only on the same machine with the same number of
    threads (docs/e3/DECISIONS.md, 2026-09-24)."""
    return {"num_threads": torch.get_num_threads(), "torch_version": torch.__version__, "git_commit": git_commit()}


def save_failure_diagnostic(cfg: dict, seed: int, rssm: RSSM, optimizer: torch.optim.Optimizer, templates_t: torch.Tensor,
                            step: int, switch: SeparationSwitch,
                            error: SeparationNotReachedError | CollapseAfterSwitchError) -> Path:
    """Called when SeparationNotReachedError or CollapseAfterSwitchError is
    raised (docs/e3/DECISIONS.md, 2026-09-24): saves the checkpoint (model + optimizer state) and the
    latent-usage diagnostic (probe.latent_usage_diagnostic: template
    accuracy with the decoder fed probabilities / mode / hard sample,
    posterior entropy, I(z;s), logit norms) into a fresh, timestamped
    directory -- never overwrites an earlier failure."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "collapse_after_switch" if isinstance(error, CollapseAfterSwitchError) else "separation_not_reached"
    out = FAILURES_DIR / f"seed{seed:02d}__{kind}__step{step}__{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    torch.save({"rssm": rssm.state_dict(), "optimizer": optimizer.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg),
                "step": step}, out / "checkpoint.pt")
    rssm.eval()
    diag = e3_probe.latent_usage_diagnostic(rssm, templates_t, M=cfg["measure"]["iz_mc_samples"],
                                            rng=np.random.default_rng(seed + 3000))
    record = {"seed": seed, "steps_done": step, "kind": kind, "error": str(error), **reproducibility_metadata(), "config": cfg,
              "switch_history": [list(map(float, h)) for h in switch.history], "diagnostic": diag}
    (out / "diagnostic.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"[run_mini seed={seed}] FAILURE DIAGNOSTIC (saved to {out}):")
    for key, value in diag.items():
        print(f"    {key:22s} {value:.4f}")
    return out

NOISY_ZONES = ("controllable_noisy", "uncontrollable_noisy")
LOSS_TERMS = ("total", "recon", "recon_patch", "kl_dyn", "kl_rep")
N_SE = 3.0  # every MC-based verdict in the report uses a 3-standard-error margin
MECHANICS_FLOOR = 0.5  # below this template-match accuracy, downstream indicators are not evaluable
NOT_EVALUABLE = "non évaluable (mécanique en échec)"


# ---------------------------------------------------------------------
# verdict rules (pure functions -- tested in tests/e3/test_e3_mini_report.py)
# ---------------------------------------------------------------------


def zone_mean_mc_se(se_per_state: np.ndarray) -> float:
    """MC standard error of a zone MEAN of per-state I_z estimates:
    sqrt(sum se_s^2) / n. Only Monte-Carlo noise counts -- the states are
    a fixed, complete set, not a sample, so their spread is signal."""
    se = np.asarray(se_per_state, dtype=np.float64)
    return float(np.sqrt(np.sum(se**2)) / len(se))


def bounds_verdict(values: np.ndarray, se: np.ndarray, ceiling: float, n_se: float = N_SE) -> dict:
    """0 <= value <= ceiling for every entry, each within n_se of its
    own MC standard error."""
    values, se = np.asarray(values), np.asarray(se)
    tol = n_se * se
    violations = int(np.sum((values < -tol) | (values > ceiling + tol)))
    return {"ok": violations == 0, "n_violations": violations}


def ordering_verdict(mean_low: float, se_low: float, mean_high: float, se_high: float, n_se: float = N_SE) -> str:
    """Expected: mean_low < mean_high. "OK" / "FAIL" only when the gap
    exceeds n_se combined standard errors (sqrt(se_low^2 + se_high^2));
    otherwise "non concluant" -- never "OK" on a difference MC noise
    could have produced."""
    gap = mean_high - mean_low
    combined = float(np.sqrt(se_low**2 + se_high**2))
    if abs(gap) <= n_se * combined:
        return "non concluant"
    return "OK" if gap > 0 else "FAIL"


def mechanics_ok(match_accuracy: float) -> bool:
    """Template-match accuracy gate: below MECHANICS_FLOOR, the s-space
    kernel (and everything built on decoded images or on a collapsed
    posterior) measures the failure of the mechanics, not the belief."""
    return match_accuracy >= MECHANICS_FLOOR


def load_mini_config() -> dict:
    with open(REPO_ROOT / "configs" / "e3" / "mini.yaml") as f:
        return yaml.safe_load(f)


def zone_of_state(layout, s: int) -> str:
    r, c = divmod(s, layout.grid_shape[1])
    return layout.zone_at((r, c)).appearance.category


def labelled_between_state_js(post_probs: np.ndarray, labels: np.ndarray) -> float:
    """Console monitoring only -- never an input to the switch or to
    training. Mean JS (bits, summed over latent variables) between the
    posteriors of every pair of batch samples whose true states DIFFER;
    post_probs (n, n_vars, n_classes) and labels (n,) in the same order.
    NaN if every sample shows the same state."""
    pair_js = jensen_shannon_categorical(post_probs[:, None], post_probs[None, :]).sum(axis=-1)
    different = labels[:, None] != labels[None, :]
    return float(pair_js[different].mean()) if different.any() else float("nan")


def build_separation_switch(cfg: dict, n_grad_steps: int) -> tuple[SeparationSwitch, int]:
    """The retained estimator procedure from the config (docs/e3/DECISIONS.md,
    2026-09-24): posterior probabilities until the label-free batch-pair JS
    criterion fires, then straight-through. switch_max_steps null = 60% of
    the training budget. After the switch, the same statistic guards
    against a collapse (CollapseAfterSwitchError)."""
    est = cfg["train"]["latent_estimator"]
    if est["kind"] != "warmup_probs_until_separated":
        raise ValueError(f"run_mini.py implements the retained procedure only; got latent_estimator.kind={est['kind']!r}")
    max_steps = est["switch_max_steps"] if est["switch_max_steps"] is not None else int(round(0.6 * n_grad_steps))
    switch = SeparationSwitch(
        max_steps=max_steps, threshold_bits=est["switch_js_threshold"], consecutive=est["switch_consecutive"],
        eval_every=est["switch_eval_every"], min_steps=est["switch_min_steps"],
        collapse_threshold_bits=est["collapse_js_threshold"], collapse_consecutive=est["collapse_consecutive"],
    )
    return switch, est["switch_n_pairs"]


def build_run(cfg: dict, seed: int):
    rotation = e3_env.rotation_for_seed_index(seed)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    layout = e3_env.build_layout(assignment)
    k = cfg["render"]["k"]
    themes = build_themes(
        e3_env.ZONE_CATEGORIES, goal_state=e3_env.goal_state(), rng=np.random.default_rng(seed),
        min_dist=cfg["render"]["cielab_min_dist"], marker_min_dist=cfg["render"]["cielab_marker_min_dist"],
    )
    env = e3_env.build_pixel_env(layout, themes, goal_reward=50.0, seed=seed, k=k, grid_lines=cfg["render"]["grid_lines"])
    background = render_background(layout, themes, k=k)
    background_t = torch.as_tensor(background, dtype=torch.float32).permute(2, 0, 1) / 255.0
    templates_np = np.stack([render(s, layout, themes, k=k) for s in range(144)])
    templates_t = torch.as_tensor(templates_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return layout, themes, env, background_t, templates_t


def collect_and_count(env, replay: ReplayBuffer, n_steps: int, rng: np.random.Generator, n_states: int, n_actions: int) -> np.ndarray:
    """Local to run_mini.py (not a modification of Phase 2's
    collect_fixed_policy_data): fills replay AND accumulates
    N[s, a, s'] counts in the same pass, so the report's tabular-twin
    comparison uses the EXACT SAME transitions the world model trained
    on, without a second collection pass."""
    N = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
    obs, info = env.reset()
    s = info["state"]
    is_first = True
    for _ in range(n_steps):
        a = int(rng.integers(n_actions))
        obs_next, r, terminated, truncated, info = env.step(a)
        s_next = info["state"]
        replay.add(x=obs, a=a, r=r, terminal=terminated, truncated=truncated, is_first=is_first, s=s)
        N[s, a, s_next] += 1
        is_first = terminated or truncated
        if is_first:
            obs_next, info = env.reset()
        s = info["state"]
        obs = obs_next
    return N


def kl_post_prior_by_zone(probe: dict, layout) -> dict[str, float]:
    """Post-hoc "KL by zone type", in BITS: mean_a KL(post_probs[s] ||
    prior_probs[s,a]), averaged over states of each zone category --
    computed from the canonical probe (spec 5.1), not from a per-batch
    training-time breakdown (which would need per-sample zone
    attribution inside random training windows). rssm.categorical_kl
    works in nats; converted here (the first mini report labelled nats
    as bits)."""
    post = torch.as_tensor(probe["posterior_probs"])  # (144, n_vars, n_classes)
    prior = torch.as_tensor(probe["prior_probs"])  # (144, n_actions, n_vars, n_classes)
    n_states, n_actions = prior.shape[0], prior.shape[1]
    post_expanded = post.unsqueeze(1).expand(-1, n_actions, -1, -1)
    kl = categorical_kl(post_expanded.reshape(-1, *post.shape[1:]), prior.reshape(-1, *prior.shape[2:]))
    kl = kl.reshape(n_states, n_actions).mean(dim=1).numpy() / np.log(2.0)  # nats -> bits, (144,)

    by_zone: dict[str, list[float]] = {}
    for s in range(n_states):
        by_zone.setdefault(zone_of_state(layout, s), []).append(float(kl[s]))
    return {zone: float(np.mean(vals)) for zone, vals in by_zone.items()}


def build_tabular_twin(N: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
    """T_hat_B from the SAME counts the world model trained on -- reuses
    helplessness.agents.tabular.TabularModelBasedAgent directly (core
    library, alpha=1 matching E2's own reference condition, restricted
    to the geographic support E2 also uses), never a reimplementation."""
    support = geographic_support_mask((12, 12))
    agent = TabularModelBasedAgent(n_states, n_actions, alpha=1.0, alpha_support=support)
    agent.N = N
    return agent.T_hat


def frequent_outcome_preview(p_hat: np.ndarray, layout) -> dict[str, float]:
    """T4 preview (spec section 9): mean, over (s, a) pairs in each
    noisy zone, of the world model's own s-space kernel's max
    probability (the "frequent outcome"). Not the full T4 statistical
    test (that is Phase 4/E3-0 scope) -- a shape preview only."""
    out = {}
    for zone in NOISY_ZONES:
        states = [s for s in range(144) if zone_of_state(layout, s) == zone]
        max_probs = p_hat[states].max(axis=-1)  # (n_states_in_zone, n_actions)
        out[zone] = float(np.nanmean(max_probs))
    return out


def save_collapse_grid(rssm: RSSM, templates: torch.Tensor, background: torch.Tensor, k: int, states: list[int], path: Path) -> None:
    """Collapse diagnostic figure: for each state, the input image, the
    reconstruction-loss mask computed from it, and the POSTERIOR
    reconstruction (canonical path: fresh h0, z = posterior mode). A
    collapsed posterior shows the same reconstruction in every column,
    with no agent marker."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = templates[states]
    with torch.no_grad():
        h0, _ = rssm.initial_state(len(states), x.device)
        z0 = categorical_mode(rssm.posterior_probs(h0, rssm.encoder(x)))
        recon = rssm.decode(h0, z0).clamp(0.0, 1.0)
        mask = reconstruction_mask(x, background.unsqueeze(0).expand_as(x), k)

    fig, axes = plt.subplots(3, len(states), figsize=(1.6 * len(states), 5))
    for j, s in enumerate(states):
        for row, (img, title) in enumerate(((x[j], "input"), (mask[j].expand(3, -1, -1), "mask"), (recon[j], "posterior recon"))):
            ax = axes[row, j]
            ax.imshow(img.permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(title, fontsize=8)
        axes[0, j].set_title(f"s={s}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_one_seed(cfg: dict, seed: int) -> dict:
    k = cfg["render"]["k"]
    rng = np.random.default_rng(seed)
    # The mini runs so far never seeded torch, so network init and
    # latent sampling were not reproducible. This fixes that; the full
    # per-stream seed layout (spec section 6) comes with Phase 4a.
    torch.manual_seed(seed)
    layout, themes, env, background_t, templates_t = build_run(cfg, seed)
    zone_class = e3_env.block_class_for_seed(seed)

    image_side = 12 * k
    budget, prefill = cfg["budget"]["total_steps"], cfg["budget"]["prefill_steps"]
    replay = ReplayBuffer.for_run(budget=budget, prefill_steps=prefill, image_shape=(image_side, image_side, 3))

    print(f"[run_mini seed={seed} class={zone_class}] collecting {prefill + budget} steps under epsilon=1 ...")
    t0 = time.perf_counter()
    N = collect_and_count(env, replay, n_steps=prefill + budget, rng=rng, n_states=144, n_actions=5)
    print(f"[run_mini seed={seed}] collection done in {time.perf_counter() - t0:.1f}s")

    rssm_cfg = RSSMConfig(
        image_size=(image_side, image_side), base_depth=cfg["model"]["base_depth"], embed_dim=cfg["model"]["embed_dim"],
        h_dim=cfg["model"]["h_dim"], n_vars=cfg["model"]["n_vars"], n_classes=cfg["model"]["n_classes"],
        n_actions=5, unimix=cfg["model"]["unimix"],
    )
    rssm = RSSM(rssm_cfg)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=cfg["train"]["lr"], eps=cfg["train"]["adam_eps"])
    loss_cfg = WorldModelLossConfig(
        k=k, w_patch=cfg["train"]["w_patch"], beta_dyn=cfg["train"]["beta_dyn"], beta_rep=cfg["train"]["beta_rep"],
        free_bits=cfg["train"]["free_bits"], free_bits_mode=cfg["train"]["free_bits_mode"],
    )

    n_grad_steps = (prefill + budget) * cfg["train"]["replay_ratio"]
    switch, n_pairs = build_separation_switch(cfg, n_grad_steps)
    rep_warmup_cfg = cfg["train"]["rep_warmup_steps"]
    rep_warmup_steps: int | None = None  # resolved at the switch (default: 20% of the budget left after it)
    pair_rng = np.random.default_rng([seed, 4])  # dedicated stream for the label-free pair sampling
    monitor_every = cfg["measure"]["monitor_every"]
    lr, lr_after_switch = cfg["train"]["lr"], cfg["train"]["lr_after_switch"]
    print(f"[run_mini seed={seed}] lr {lr:g} until the switch, then {lr_after_switch}; clip_norm {cfg['train']['clip_norm']}; "
          f"collapse guard: batch-pair JS < {switch.collapse_threshold_bits} bits on {switch.collapse_consecutive} consecutive "
          f"evaluations after the switch; torch threads {torch.get_num_threads()}")
    lr_switch_step = None
    print(f"[run_mini seed={seed}] training for {n_grad_steps} gradient steps: posterior probabilities until the "
          f"batch-pair JS exceeds {switch.threshold_bits} bits on {switch.consecutive} consecutive evaluations (every "
          f"{switch.eval_every} steps, from step {switch.min_steps}, deadline {switch.max_steps}), then straight-through; "
          f"beta_dyn={loss_cfg.beta_dyn} from step 0, beta_rep 0 until the switch then ramped to {loss_cfg.beta_rep} ...")
    history = {"step": [], "total": [], "recon": [], "recon_patch": [], "kl_dyn": [], "kl_rep": [], "beta_dyn": [], "beta_rep": []}
    separation_every = cfg["measure"]["separation_every"]
    separation_curve = []
    last_stat = float("nan")
    t0 = time.perf_counter()
    for step in range(n_grad_steps):
        if set_learning_rate(optimizer, phase_learning_rate(step, switch.switch_step, lr, lr_after_switch)):
            lr_switch_step = step
            print(f"[run_mini seed={seed}] >>> LR changed to {optimizer.param_groups[0]['lr']:g} at step {step} (effective switch)",
                  flush=True)
        if step % separation_every == 0:
            separation_curve.append({"step": step, **e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=1)})
        model_batch, label_batch = replay.sample(cfg["train"]["batch_size"], cfg["train"]["seq_len"], rng)
        beta_rep = rep_beta(step, switch.switch_step, rep_warmup_steps or 0, loss_cfg.beta_rep)
        step_cfg = dataclasses.replace(loss_cfg, beta_rep=beta_rep)
        phase = "straight-through" if switch.switched else "probabilities"
        eval_now = switch.is_eval_step(step)
        monitor_now = step % monitor_every == 0
        losses = train_world_model_step(
            rssm, optimizer, model_batch, background_t, step_cfg, clip_norm=cfg["train"]["clip_norm"],
            estimator=switch.estimator(), return_posteriors=eval_now or monitor_now,
        )
        post_probs = losses.pop("post_probs", None)
        if eval_now:
            last_stat = mean_random_pair_jensen_shannon(post_probs, pair_rng, n_pairs)
            was_switched = switch.switched
            try:
                switch.update(step, last_stat)
            except CollapseAfterSwitchError as err:
                save_failure_diagnostic(cfg, seed, rssm, optimizer, templates_t, step + 1, switch, err)
                raise
            if switch.switched and not was_switched:
                rep_warmup_steps = rep_warmup_cfg if rep_warmup_cfg is not None else int(round(0.2 * (n_grad_steps - switch.switch_step)))
                print(f"[run_mini seed={seed}] >>> SWITCH to straight-through effective at step {switch.switch_step} "
                      f"(batch-pair JS {last_stat:.3f} bits); beta_rep ramps 0 -> {loss_cfg.beta_rep} over {rep_warmup_steps} steps")
        try:
            switch.check_deadline(step + 1)
        except SeparationNotReachedError as err:
            save_failure_diagnostic(cfg, seed, rssm, optimizer, templates_t, step + 1, switch, err)
            raise
        history["beta_dyn"].append(loss_cfg.beta_dyn)
        history["beta_rep"].append(beta_rep)
        history["step"].append(step)
        for key in ("total", "recon", "recon_patch", "kl_dyn", "kl_rep"):
            history[key].append(losses[key])
        if monitor_now:
            # MONITORING ONLY: the labelled JS uses this batch's state labels, which reach
            # neither the training step nor the switch rule (the switch is driven by the
            # label-free batch-pair JS above, exclusively).
            labelled_js = labelled_between_state_js(post_probs, label_batch.s.reshape(-1))
            print(f"[run_mini seed={seed}] step {step:6d}/{n_grad_steps} | batch-pair JS {last_stat:6.3f} bits | "
                  f"labelled JS {labelled_js:6.3f} bits | phase {phase:16s} | beta_rep {beta_rep:.4f} | "
                  f"recon_patch {losses['recon_patch']:.4f} | grad_norm {losses['grad_norm']:.3g} | "
                  f"lr {optimizer.param_groups[0]['lr']:g}", flush=True)
    separation_curve.append({"step": n_grad_steps, **e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=1)})
    print(f"[run_mini seed={seed}] training done in {time.perf_counter() - t0:.1f}s")

    rssm.eval()

    print(f"[run_mini seed={seed}] canonical probe + I_z ...")
    iz_rng = np.random.default_rng(seed + 1000)
    iz_result = e3_probe.canonical_iz(rssm, templates_t, M=cfg["measure"]["iz_mc_samples"], rng=iz_rng)
    probe = iz_result["probe"]

    print(f"[run_mini seed={seed}] collapse diagnostics (I(z;s), posterior entropy, grid) ...")
    izs = latent_state_information(probe["posterior_probs"], M=cfg["measure"]["iz_mc_samples"], rng=np.random.default_rng(seed + 2000))
    zone_categories = list(e3_env.ZONE_CATEGORIES)
    grid_states = [s for cat in zone_categories for s in [t for t in range(144) if zone_of_state(layout, t) == cat][:2]]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grid_name = f"collapse_grid_seed{seed:02d}.png"
    save_collapse_grid(rssm, templates_t, background_t, k, grid_states, FIGURES_DIR / grid_name)

    print(f"[run_mini seed={seed}] template-match accuracy ...")
    eval_batch, eval_labels = replay.sample(32, 16, rng)
    match_acc = e3_probe.template_match_accuracy(rssm, eval_batch, eval_labels, templates_t, margin_threshold=cfg["measure"]["kernel_margin_threshold"])

    print(f"[run_mini seed={seed}] kernel-in-s-space (slow step) ...")
    kernel_result = e3_probe.canonical_kernel_probe(rssm, templates_t, K_s=cfg["measure"]["kernel_k_s"], margin_threshold=cfg["measure"]["kernel_margin_threshold"], probe=probe)

    print(f"[run_mini seed={seed}] z-only probe (every replay posterior) ...")
    z_batch, z_labels = replay.chronological_windows(seq_len=64)
    z_probe_result = e3_probe.z_only_probe(rssm, z_batch, z_labels, n_states=144, test_frac=cfg["measure"]["z_only_test_frac"], rng=rng)

    print(f"[run_mini seed={seed}] context dispersion ...")
    ctx_batch, ctx_labels = replay.sample(16, 32, rng)
    ctx_result = e3_probe.context_dispersion(rssm, ctx_batch, ctx_labels, iz_result["I_z"], M=cfg["measure"]["iz_mc_samples"], rng=rng, n_samples=cfg["measure"]["context_dispersion_n_samples"])

    print(f"[run_mini seed={seed}] D1/D2 diagnostics ...")
    I_by_category = {cat: float(iz_result["I_z"][[s for s in range(144) if zone_of_state(layout, s) == cat]].mean()) for cat in zone_categories}
    probe_states = {cat: [s for s in range(144) if zone_of_state(layout, s) == cat][:2] for cat in zone_categories}

    d1_results = []
    for own_cat in zone_categories:
        for target_cat in zone_categories:
            if own_cat == target_cat:
                continue
            for s in probe_states[own_cat]:
                for global_swap in (False, True):
                    d1 = e3_probe.d1_alpha_app_for_state(
                        rssm, layout, themes, s, target_cat, I_by_category, M=cfg["measure"]["iz_mc_samples"], rng=rng,
                        k=k, contrast_threshold=cfg["measure"]["contrast_threshold"], global_swap=global_swap,
                    )
                    d1_results.append({"s": s, "own_category": own_cat, "target_category": target_cat, "global_swap": global_swap, **d1})

    d2_result = e3_probe.d2_neighbor_distances(probe["posterior_probs"], layout)
    kl_by_zone = kl_post_prior_by_zone(probe, layout)

    print(f"[run_mini seed={seed}] tabular twin + L1 comparisons ...")
    T_hat_B = build_tabular_twin(N, 144, 5)
    P_true = env.transition_tensor()
    l1_vs_p = np.abs(kernel_result["p_hat"] - P_true).sum(axis=-1)  # (144, 5)
    l1_vs_b = np.abs(kernel_result["p_hat"] - T_hat_B).sum(axis=-1)  # (144, 5)

    freq_outcome = frequent_outcome_preview(kernel_result["p_hat"], layout)

    def zone_mean(arr_per_state_action: np.ndarray, cat: str) -> float:
        states = [s for s in range(144) if zone_of_state(layout, s) == cat]
        return float(np.nanmean(arr_per_state_action[states]))

    hist = {key: np.asarray(history[key]) for key in LOSS_TERMS}
    loss_table = {
        key: {"start": float(v[:50].mean()), "end": float(v[-50:].mean())} for key, v in hist.items()
    }

    return {
        "seed": seed,
        "zone_class": zone_class,
        "history": history,
        "loss_table": loss_table,
        "training_procedure": {
            "switch_rule": {"threshold_bits": switch.threshold_bits, "consecutive": switch.consecutive,
                            "eval_every": switch.eval_every, "min_steps": switch.min_steps, "max_steps": switch.max_steps,
                            "n_pairs": n_pairs},
            "switch_step": switch.switch_step,
            "collapse_guard": {"threshold_bits": switch.collapse_threshold_bits, "consecutive": switch.collapse_consecutive},
            "lr": lr, "lr_after_switch": lr_after_switch, "lr_switch_step": lr_switch_step,
            "clip_norm": cfg["train"]["clip_norm"],
            "rep_warmup_steps": rep_warmup_steps,
            "batch_pair_js": switch.history,
            "n_grad_steps": n_grad_steps,
        },
        "separation_curve": separation_curve,
        "reproducibility": reproducibility_metadata(),
        "loss_weights": {"recon": 1.0, "recon_patch": 0.0, "kl_dyn": loss_cfg.beta_dyn, "kl_rep": loss_cfg.beta_rep},
        "w_patch": loss_cfg.w_patch,
        "I_zs": izs["I_zs"],
        "I_zs_se": izs["se"],
        "canonical_posterior_entropy_mean": float(probe["posterior_entropy"].mean()),
        "posterior_entropy_ceiling": float(rssm_cfg.n_vars * np.log2(rssm_cfg.n_classes)),
        "collapse_grid": grid_name,
        "I_z": iz_result["I_z"],
        "I_z_se": iz_result["se"],
        "zone_of_state": [zone_of_state(layout, s) for s in range(144)],
        "I_by_category": I_by_category,
        "match_accuracy": match_acc,
        "ghost_fraction_mean": float(np.nanmean(kernel_result["ghost_fraction"])),
        "z_only_probe": z_probe_result,
        "context_dispersion_mean_abs": float(ctx_result["abs_discrepancy"].mean()),
        "d1_results": d1_results,
        "d2_boundary_mean": float(d2_result["boundary"].mean()),
        "d2_within_zone_mean": float(d2_result["within_zone"].mean()),
        "kl_by_zone": kl_by_zone,
        "l1_vs_p_by_zone": {cat: zone_mean(l1_vs_p, cat) for cat in zone_categories},
        "l1_vs_b_by_zone": {cat: zone_mean(l1_vs_b, cat) for cat in zone_categories},
        "frequent_outcome_preview": freq_outcome,
        "zone_categories": zone_categories,
    }


def _save_loss_curve_figure(seed_reports: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    terms = ["total", "recon", "recon_patch", "kl_dyn", "kl_rep"]
    for ax, term in zip(axes, terms):
        for report in seed_reports:
            ax.plot(report["history"]["step"], report["history"][term], label=f"seed={report['seed']} (class {report['zone_class']})")
        ax.set_title(term)
        ax.set_xlabel("gradient step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_separation_figure(seed_reports: list[dict], path: Path) -> None:
    """Separation between states along training: canonical (labelled)
    hard-sample template accuracy, between-state JS and posterior
    entropy, plus -- on the JS panel -- the label-free batch-pair JS that
    drives the switch. Per seed, the EFFECTIVE switch step (dashed) and
    the end of the beta_rep ramp (dotted) are marked, in the seed's colour
    (docs/e3/DECISIONS.md, 2026-09-24)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    panels = (("acc_hard", "canonical hard-sample template accuracy"), ("mean_js", "mean JS between states (bits)"),
              ("posterior_entropy", "mean posterior entropy (bits)"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (key, title) in zip(axes, panels):
        for i, report in enumerate(seed_reports):
            color = colors[i % len(colors)]
            tag = f"seed={report['seed']} (class {report['zone_class']})"
            curve = report["separation_curve"]
            ax.plot([c["step"] for c in curve], [c[key] for c in curve], color=color, label=f"{tag}, canonical (labelled)")
            proc = report["training_procedure"]
            if key == "mean_js" and proc["batch_pair_js"]:
                steps, values = zip(*proc["batch_pair_js"])
                ax.plot(steps, values, color=color, alpha=0.4, linewidth=0.8, label=f"{tag}, batch-pair (label-free)")
                ax.axhline(proc["switch_rule"]["threshold_bits"], color="grey", linewidth=0.8, linestyle="-.")
            if proc["switch_step"] is not None:
                ax.axvline(proc["switch_step"], color=color, linestyle="--", linewidth=1)
                ax.axvline(proc["switch_step"] + proc["rep_warmup_steps"], color=color, linestyle=":", linewidth=1)
        ax.set_title(title + "\n(dashed: switch to straight-through; dotted: end of beta_rep ramp)", fontsize=9)
        ax.set_xlabel("gradient step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_iz_by_zone_figure(seed_reports: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = seed_reports[0]["zone_categories"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(categories))
    for i, report in enumerate(seed_reports):
        values = [report["I_by_category"][c] for c in categories]
        ax.bar(x + i * width, values, width, label=f"seed={report['seed']} (class {report['zone_class']})")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("mean canonical I_z (bits)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _fmt(x: float) -> str:
    return f"{x:.4f}" if np.isfinite(x) else "NaN"


def _zone_iz_stats(report: dict, cat: str) -> tuple[float, float, float]:
    """(mean I_z over the zone's states, std over states, MC se of the mean)."""
    idx = [s for s in range(144) if report["zone_of_state"][s] == cat]
    I_z = np.asarray(report["I_z"])[idx]
    return float(I_z.mean()), float(I_z.std()), zone_mean_mc_se(np.asarray(report["I_z_se"])[idx])


def write_report(seed_reports: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_loss_curve_figure(seed_reports, FIGURES_DIR / "loss_curves.png")
    _save_iz_by_zone_figure(seed_reports, FIGURES_DIR / "iz_by_zone.png")
    _save_separation_figure(seed_reports, FIGURES_DIR / "separation_curves.png")

    categories = seed_reports[0]["zone_categories"]
    lines: list[str] = []
    add = lines.append
    add("# E3 Phase 3 -- mini configuration report")
    add("")
    add(f"Generated by `experiments/E3_worldmodel_4zones/run_mini.py` from `configs/e3/mini.yaml`, "
        f"seeds {[r['seed'] for r in seed_reports]} (classes {[r['zone_class'] for r in seed_reports]}).")
    add("")
    add(f"Verdict rules: MC-based checks use a {N_SE:g}-standard-error margin; an ordering whose gap is within "
        f"{N_SE:g} combined standard errors is \"non concluant\", never OK. If template-match accuracy is below "
        f"{MECHANICS_FLOOR}, every indicator built on decoded images or on the posterior's discrimination (s-space "
        f"kernel, L1, T4 preview, ghosts, D1, D2) is reported as \"{NOT_EVALUABLE}\" instead of a value.")
    add("")
    add("## Loss curves (per term)")
    add("")
    add("![loss curves](mini_report_figures/loss_curves.png)")
    add("")
    add("Reconstruction is summed over channels and pixels of each image and averaged over batch x time; KL terms "
        "are exact, summed over latent variables, in nats. `recon_patch` is diagnostic only (unweighted error on the "
        "masked patch, same aggregation), not part of the loss.")
    add("")
    add("## Separation between states along training")
    add("")
    add("![separation curves](mini_report_figures/separation_curves.png)")
    add("")
    add("Canonical path (fresh h0, posterior of render(s)) for the 144 states, every "
        "`measure.separation_every` steps: if the switch to straight-through happens before the states have "
        "separated (JS ~ 0), the categorical bottleneck is expected to collapse again (docs/e3/ESTIMATOR_COMPARISON.md).")
    add("")
    add("## Canonical I_z by zone type")
    add("")
    add("![I_z by zone](mini_report_figures/iz_by_zone.png)")
    add("")

    for report in seed_reports:
        seed, cls = report["seed"], report["zone_class"]
        ok_mech = mechanics_ok(report["match_accuracy"])
        add(f"## Seed {seed} (class {cls})")
        add("")

        add("### Loss terms (mean of the first / last 50 gradient steps)")
        add("")
        proc = report["training_procedure"]
        add(f"Weights: recon x1 (patch weight w_patch={report['w_patch']:g} inside the mask); "
            f"kl_dyn x{report['loss_weights']['kl_dyn']:g} from step 0; kl_rep 0 until the switch, then ramped "
            f"linearly to x{report['loss_weights']['kl_rep']:g} over {proc['rep_warmup_steps']} steps; "
            f"recon_patch x0 (diagnostic). {proc['n_grad_steps']} gradient steps.")
        add("")
        add(f"Posterior estimator: probabilities until the label-free batch-pair JS exceeded "
            f"{proc['switch_rule']['threshold_bits']} bits on {proc['switch_rule']['consecutive']} consecutive "
            f"evaluations (every {proc['switch_rule']['eval_every']} steps, from step {proc['switch_rule']['min_steps']}, "
            f"deadline {proc['switch_rule']['max_steps']}), then straight-through. "
            f"**Effective switch step: {proc['switch_step']}.** Learning rate {proc['lr']:g} until the switch, then "
            f"{proc['lr_after_switch']} (changed at step {proc['lr_switch_step']}); global gradient-norm clipping: "
            f"{'none' if proc['clip_norm'] is None else format(proc['clip_norm'], 'g')}; collapse guard after the switch: batch-pair JS < {proc['collapse_guard']['threshold_bits']} "
            f"bits on {proc['collapse_guard']['consecutive']} consecutive evaluations.")
        repro = report["reproducibility"]
        add("")
        add(f"Reproducibility: torch {repro['torch_version']}, {repro['num_threads']} threads, commit "
            f"`{repro['git_commit'] or 'unknown'}` -- bit-exact only on the same machine with the same number of threads.")
        add("")
        add("| term | start | end |")
        add("|---|---|---|")
        for term in LOSS_TERMS:
            add(f"| {term} | {_fmt(report['loss_table'][term]['start'])} | {_fmt(report['loss_table'][term]['end'])} |")
        add("")

        add("### Collapse diagnostics")
        add("")
        n_replay = report["z_only_probe"]["n_train"] + report["z_only_probe"]["n_test"]
        add(f"- I(z; s) over the canonical posteriors (uniform over 144 states): **{_fmt(report['I_zs'])} bits** "
            f"(MC se {_fmt(report['I_zs_se'])}; ceiling log2(144) = {np.log2(144):.4f}; ~0 = collapse)")
        add(f"- Mean posterior entropy, canonical probe: **{_fmt(report['canonical_posterior_entropy_mean'])} bits**; "
            f"replay (all {n_replay} posteriors): **{_fmt(report['z_only_probe']['mean_posterior_entropy'])} bits** "
            f"(ceiling n_vars x log2(n_classes) = {report['posterior_entropy_ceiling']:.1f})")
        add("")
        add(f"![collapse grid, seed {seed}](mini_report_figures/{report['collapse_grid']})")
        add("")

        add("### I_z canonical (mean +/- std over states, per zone; MC se of the zone mean)")
        add("")
        add("| zone | mean I_z | std I_z | MC se of the mean |")
        add("|---|---|---|---|")
        for cat in categories:
            m, sd, se = _zone_iz_stats(report, cat)
            add(f"| {cat} | {_fmt(m)} | {_fmt(sd)} | {_fmt(se)} |")
        add("")

        add("### KL(posterior || prior) by zone (canonical probe, mean over actions, bits)")
        add("")
        add("| zone | mean KL (bits) |")
        add("|---|---|")
        for cat, val in report["kl_by_zone"].items():
            add(f"| {cat} | {_fmt(val)} |")
        add("")

        add("### Template-match accuracy / z-only probe")
        add("")
        add(f"- Template-match accuracy on posterior reconstructions: **{_fmt(report['match_accuracy'])}**"
            + ("" if ok_mech else f" -- below {MECHANICS_FLOOR}: mechanics failing, downstream indicators not evaluable"))
        add(f"- z-only linear probe (every replay posterior): train accuracy "
            f"**{_fmt(report['z_only_probe']['train_accuracy'])}**, test accuracy "
            f"**{_fmt(report['z_only_probe']['test_accuracy'])}** "
            f"(n_train={report['z_only_probe']['n_train']}, n_test={report['z_only_probe']['n_test']})")
        add("")

        add("### L1 of the s-space kernel vs P (true) and vs the tabular twin B, per zone")
        add("")
        if ok_mech:
            add("| zone | L1 vs P | L1 vs B |")
            add("|---|---|---|")
            for cat in categories:
                add(f"| {cat} | {_fmt(report['l1_vs_p_by_zone'][cat])} | {_fmt(report['l1_vs_b_by_zone'][cat])} |")
        else:
            add(f"{NOT_EVALUABLE}.")
        add("")

        add("### p_hat(frequent outcome) preview (T4 preview) -- noisy zones")
        add("")
        if ok_mech:
            add("| zone | mean p_hat(frequent outcome) |")
            add("|---|---|")
            for cat, val in report["frequent_outcome_preview"].items():
                add(f"| {cat} | {_fmt(val)} |")
        else:
            add(f"{NOT_EVALUABLE}.")
        add("")

        add("### Ghost rate (mean over all (s, a) pairs)")
        add("")
        add(f"**{_fmt(report['ghost_fraction_mean'])}**" if ok_mech else f"{NOT_EVALUABLE}.")
        add("")

        add(f"### Context dispersion (mean abs discrepancy vs canonical I_z): "
            f"**{_fmt(report['context_dispersion_mean_abs'])}**")
        add("")

        add("### D1 (alpha_app, local and global)")
        add("")
        if ok_mech:
            defined_vals = [d["alpha_app"] for d in report["d1_results"] if d["alpha_app"] is not None]
            add(f"- Defined for {len(defined_vals)}/{len(report['d1_results'])} (own, target, local/global) probes.")
            if defined_vals:
                add(f"- alpha_app over defined probes: mean={_fmt(float(np.mean(defined_vals)))}, "
                    f"min={_fmt(float(np.min(defined_vals)))}, max={_fmt(float(np.max(defined_vals)))}")
            add("")
            add("| own zone | target zone | local/global | alpha_app |")
            add("|---|---|---|---|")
            for d in report["d1_results"]:
                kind = "global" if d["global_swap"] else "local"
                val = "undefined (no contrast)" if d["alpha_app"] is None else _fmt(d["alpha_app"])
                add(f"| {d['own_category']} | {d['target_category']} | {kind} | {val} |")
        else:
            add(f"{NOT_EVALUABLE}.")
        add("")

        add("### D2 -- neighbour JS distance, boundary vs within-zone")
        add("")
        if ok_mech:
            add(f"- boundary (different category) mean JS: **{_fmt(report['d2_boundary_mean'])}**")
            add(f"- within-zone (same category) mean JS: **{_fmt(report['d2_within_zone_mean'])}**")
        else:
            add(f"{NOT_EVALUABLE}.")
        add("")

    add('## Expected-shape checklist (spec: "formes attendues seulement")')
    add("")
    add("| # | check | seed (class) | expected | observed | verdict |")
    add("|---|---|---|---|---|---|")
    row = 1
    for report in seed_reports:
        tag = f"{report['seed']} ({report['zone_class']})"
        ok_mech = mechanics_ok(report["match_accuracy"])

        m_conv, _, se_conv = _zone_iz_stats(report, "uncontrollable_deterministic")
        m_ctrl, _, se_ctrl = _zone_iz_stats(report, "controllable_deterministic")
        verdict = ordering_verdict(m_conv, se_conv, m_ctrl, se_ctrl)
        combined = float(np.sqrt(se_conv**2 + se_ctrl**2))
        add(f"| {row} | I_z(conveyor) < I_z(controllable-det) | {tag} | gap > {N_SE:g} x combined se | "
            f"{_fmt(m_conv)} vs {_fmt(m_ctrl)} (gap {_fmt(m_ctrl - m_conv)}, combined se {_fmt(combined)}) | {verdict} |")
        row += 1

        b = bounds_verdict(report["I_z"], report["I_z_se"], ceiling=float(np.log2(5)))
        add(f"| {row} | 0 <= I_z <= log2(5), per state within {N_SE:g} MC se | {tag} | 0 violations | "
            f"{b['n_violations']} violation(s) / 144 | {'OK' if b['ok'] else 'FAIL'} |")
        row += 1

        add(f"| {row} | template-match accuracy high | {tag} | >= {MECHANICS_FLOOR} (mini; calibration's C-a is 0.99) | "
            f"{_fmt(report['match_accuracy'])} | {'OK' if ok_mech else 'FAIL'} |")
        row += 1

        if ok_mech:
            n_defined = sum(1 for d in report["d1_results"] if d["alpha_app"] is not None)
            add(f"| {row} | alpha_app defined only with contrast | {tag} | per contrast_threshold | "
                f"{n_defined}/{len(report['d1_results'])} defined | OK |")
        else:
            add(f"| {row} | alpha_app defined only with contrast | {tag} | per contrast_threshold | "
                f"{NOT_EVALUABLE} | {NOT_EVALUABLE} |")
        row += 1

    report_path = DOCS_DIR / "MINI_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 mini run (configs/e3/mini.yaml).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="override the config's seeds (e.g. to re-run one seed); the report covers only these")
    args = parser.parse_args()
    cfg = load_mini_config()
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    seed_reports = [run_one_seed(cfg, seed) for seed in cfg["seeds"]]
    write_report(seed_reports)


if __name__ == "__main__":
    main()
