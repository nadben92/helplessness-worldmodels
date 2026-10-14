"""E3-0 world-model run on one configuration (configs/e3/mini_long.yaml is
the CPU reference, configs/e3/gpu_e30.yaml the first GPU run; mini.yaml is
retired from current use). For each (seed, condition): collects fixed
epsilon=1 data in the experimental environment or in its reference (plus
N[s,a,s'] counts for the tabular-twin comparison), trains the RSSM with the
retained procedure (docs/e3/DECISIONS.md), measures along training (at the
checkpoints of E2's grid and/or every `measure.diagnostics_every` steps)
and at the end, and writes a report.

The main measure is taken in REAL contexts (docs/e3/DECISIONS.md,
2026-09-28): the contexts (h_t, z_t) of replay windows unrolled with their
true history, after an 8-step burn-in and never at an is_first step, up to
50 per state; I_z, KL(posterior || prior) by zone, the s-space kernel (L1
vs P and vs B, p_hat of the frequent outcome), alpha_app (D1) and D2 are
all computed there. The canonical probe (learned initial state h0) is
reported as secondary.

Outputs -- one directory per (seed, condition), never overwritten
(helplessness.runs.make_run_dir):
    <out-dir>/E3_worldmodel_4zones/grid12x12__4zones__pair-<experimental|control>__worldmodel-<config>__seedNN__<UTC>/
        config.yaml         the resolved config (after CLI overrides)
        meta.yaml           status, seed, condition, commit + dirty flag, python/torch/CUDA/cuDNN/GPU, device,
                            TF32 and determinism flags, threads, permutation, themes
        world.json          what rebuilds the run's world: seed, condition, rotation, themes, render, the
                            dynamics seed, the hash of the 144 pre-rendered images
        replay.npz          the compact replay (s, a, r, flags, s_next), N[s,a,s'] and the episode starts
        console.log         everything printed to the console
        diagnostics.jsonl   training diagnostics, one JSON line per point, appended as they are computed
        checkpoints/        weights_stepNNNNNNNN.pt at every checkpoint of the grid, weights_final.pt,
                            resume.pt (model + optimizer + random states + progress) and latest.json
        results.json        scalars and curves (written at the end)
        arrays.npz          N, T_hat_B, P, the real-context and canonical arrays
        REPORT.md           the run's report; figures in artifacts/
        failure/            checkpoint.pt + diagnostic.json, only if a training guard fired
recompute_measures.py recomputes every measure from such a directory.

Usage, from the repo root (Linux / macOS):
    python experiments/E3_worldmodel_4zones/run_mini.py --config configs/e3/gpu_e30.yaml --seeds 0 --conditions experimental --device cuda
    python experiments/E3_worldmodel_4zones/run_mini.py --resume results/E3_worldmodel_4zones/<run directory>
    python experiments/E3_worldmodel_4zones/run_mini.py --combine <run dir> <run dir> ... --out-dir results
PowerShell:
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\run_mini.py --config configs\\e3\\mini_long.yaml --device cpu
Options: --config, --device {auto,cpu,cuda}, --out-dir (default: results/), --seeds, --conditions,
--max-steps (caps the number of gradient steps, for smoke tests), --resume, --combine.
Exit code 1 if a run stopped on a training guard.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

import env as e3_env
import probe as e3_probe
import rng_streams as e3_rng
from wandb_logger import WandbLogger

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.agents.worldmodel.acceleration import compile_from_config
from helplessness.agents.worldmodel.agent import categorical_mode
from helplessness.agents.worldmodel.losses import reconstruction_mask
from helplessness.agents.worldmodel.loop import (
    CollapseAfterSwitchError,
    draw_start_state,
    non_terminal_states,
    SeparationNotReachedError,
    SeparationSwitch,
    WorldModelLossConfig,
    phase_learning_rate,
    rep_beta,
    set_learning_rate,
    train_world_model_step,
)
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, categorical_kl, resolve_device
from helplessness.envs.appearance.theme_sampling import Themes, build_themes
from helplessness.envs.observation.allocentric_pixel_backend import render, render_background
from helplessness.envs.zones import geographic_support_mask
from helplessness.measures.context_summary import per_state_context_summary, zone_mean_with_se
from helplessness.measures.diagnostics import (
    alpha_app_with_se,
    d1_contrast,
    jensen_shannon_categorical,
    mean_random_pair_jensen_shannon,
    summarize_alpha_app,
)
from helplessness.measures.controllability import controllability_bootstrap, controllability_per_state
from helplessness.measures.kernel_probe import expected_l1_sampling_floor, off_support_mass
from helplessness.measures.latent_controllability import latent_state_information
from helplessness.runs import (
    code_provenance,
    make_run_dir,
    make_run_name,
    require_code_provenance,
    write_checkpoint_pointer,
    write_run_meta,
)
from helplessness.pipeline.runtime import (  # moved to the library on 2026-10-14, re-exported here
    configure_runtime,
    git_state,
    max_images_per_forward,
    gpu_peak_start,
    gpu_peak_end,
    environment_metadata,
    RunLog,
    seed_streams,
    measure_streams,
    _jsonable,
    _nan,
    build_separation_switch,
    switch_thresholds_bits,
)

START_DISTRIBUTIONS = ("uniform_non_terminal",)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e3" / "mini.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results"
# Episode parameters (reward, step cost, horizon, gamma) come from the experiment's own
# config, checked against E2's values by tests/e3/test_e3_protocol_conformance.py.
EXPERIMENT_CONFIG_PATH = REPO_ROOT / "experiments" / "E3_worldmodel_4zones" / "config.yaml"
EXPERIMENT_ID = "E3_worldmodel_4zones"
ZONE_PRESET = "4zones"
GRID_SHAPE = (12, 12)
N_STATES, N_ACTIONS = 144, 5
PAIR_ROLE = {"experimental": "experimental", "reference": "control"}  # runs.make_run_name's vocabulary

NOISY_ZONES = ("controllable_noisy", "uncontrollable_noisy")
DETERMINISTIC_ZONES = ("controllable_deterministic", "uncontrollable_deterministic")
DETERMINISTIC_KINDS = DETERMINISTIC_ZONES
LOSS_TERMS = ("total", "recon", "recon_patch", "kl_dyn", "kl_rep")
N_SE = 3.0  # every MC-based verdict in the report uses a 3-standard-error margin
MECHANICS_FLOOR = 0.5  # below this template-match accuracy, downstream indicators are not evaluable
NOT_EVALUABLE = "non évaluable (mécanique en échec)"
NOT_MEASURED = "non mesuré"
ACC_LABELS = {
    "acc_replay_hard": "replay windows (real contexts h_t), hard sample",
    "acc_real_context": "real contexts of the main measure (after the burn-in, no is_first step), hard sample",
    "acc_canonical_hard": "canonical probe (fresh h0), hard sample, mean of 5 draws",
    "acc_canonical_mode": "canonical probe (fresh h0), posterior mode",
}
# Measurement slots (rng_streams.measure_streams): each measurement owns one.
SLOT_SEPARATION, SLOT_DIAG, SLOT_IZS, SLOT_ZONLY, SLOT_D1_REAL, SLOT_FAILURE, SLOT_FINAL, SLOT_D1_CANONICAL = 1, 2, 3, 4, 6, 7, 8, 9
SLOT_HONLY = 11  # the h-only probe (2026-09-30); slot 10 is action_in_latent.py's
SLOT_PER_CONTEXT = 12  # the per-context I_s check (2026-09-30, also is_pooling_check.py)


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
    own standard error. NaN entries (states not measured) are skipped and
    counted apart."""
    values, se = np.asarray(values, dtype=np.float64), np.asarray(se, dtype=np.float64)
    measured = ~np.isnan(values)
    tol = n_se * np.where(np.isnan(se), 0.0, se)
    violations = int(np.sum(measured & ((values < -tol) | (values > ceiling + tol))))
    return {"ok": violations == 0, "n_violations": violations, "n_measured": int(measured.sum())}


def ordering_verdict(mean_low: float, se_low: float, mean_high: float, se_high: float, n_se: float = N_SE) -> str:
    """Expected: mean_low < mean_high. "OK" / "FAIL" only when the gap
    exceeds n_se combined standard errors (sqrt(se_low^2 + se_high^2));
    otherwise "non concluant" -- never "OK" on a difference MC noise
    could have produced."""
    if not (np.isfinite(mean_low) and np.isfinite(mean_high)):
        return NOT_MEASURED
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


def shape_verdicts(records: list[dict], switch_step: int | None, n_se: float = N_SE, regime: str | None = None,
                   zone_kinds: dict[str, str] | None = None) -> list[dict]:
    """The expected shapes of the long diagnostic run, fixed in advance
    (docs/e3/DECISIONS.md, 2026-09-25), judged between the FIRST training
    diagnostic at or after the effective switch and the LAST one:
      1. KL(posterior || prior) in each deterministic zone decreases and
         ends below 1 bit;
      2. L1(world model, P) decreases in each zone (its distance to
         L1(B, P), what the data allow, is reported);
      3. I_z of the conveyor (uncontrollable_deterministic) decreases;
      4. I_z of the controllable-deterministic zone increases.
    3 and 4 use the standard error of the zone means: a change within n_se
    combined se is "non concluant". They are judged on I_s (the main measure
    since 2026-09-30) when the records have it, then on I_z (secondary). regime: the key of the record holding
    the measures (e.g. "real_context", the main measure since
    2026-09-28); None = the record's top level (the canonical probe).
    zone_kinds: appearance category -> dynamics type in the run's
    condition (default: each zone's own name, the experimental
    environment). The rules bear on the DYNAMICS: "deterministic zone" and
    rules 3-4 select zones by their dynamics type, so in the reference
    environment the conveyor-looking zone (controllable-deterministic
    there) falls under rule 4 and rule 3 has no zone (2026-09-28).
    Returns one row per check."""
    zone_kinds = zone_kinds or {z: z for z in e3_env.ZONE_CATEGORIES}
    post = [r for r in records if switch_step is not None and r["step"] >= switch_step]
    if len(post) < 2:
        return [{"check": "all", "verdict": "non évaluable (moins de 2 points de diagnostic après la bascule)"}]
    first, last = post[0], post[-1]
    a_rec = first if regime is None else first[regime]
    b_rec = last if regime is None else last[regime]
    rows = []
    for zone in e3_env.ZONE_CATEGORIES:
        if zone_kinds[zone] not in DETERMINISTIC_KINDS:
            continue
        a, b = _nan(a_rec["kl_by_zone"][zone]), _nan(b_rec["kl_by_zone"][zone])
        ok = b < a and b < 1.0
        rows.append({"check": f"KL decreasing to < 1 bit ({zone})", "first": a, "last": b,
                     "verdict": "conforme" if ok else ("non mesuré" if not np.isfinite(a + b) else "non conforme")})
    for zone in e3_env.ZONE_CATEGORIES:
        a, b = _nan(a_rec["l1_vs_p_by_zone"][zone]), _nan(b_rec["l1_vs_p_by_zone"][zone])
        ref = _nan(b_rec["l1_b_vs_p_by_zone"][zone])
        rows.append({"check": f"L1(WM, P) decreasing toward L1(B, P) = {ref:.3f} ({zone})", "first": a, "last": b,
                     "verdict": "conforme" if b < a else ("non mesuré" if not np.isfinite(a + b) else "non conforme")})
    measures = [key for key in ("I_s", "I_z") if f"{key}_by_zone" in a_rec and f"{key}_by_zone" in b_rec]
    for key, kind, direction in [(k, kind, d) for k in measures
                                 for kind, d in (("uncontrollable_deterministic", -1), ("controllable_deterministic", +1))]:
        for zone in [z for z in e3_env.ZONE_CATEGORIES if zone_kinds[z] == kind]:
            a, b = a_rec[f"{key}_by_zone"][zone], b_rec[f"{key}_by_zone"][zone]
            se = float(np.sqrt(_nan(a.get("se", a.get("mc_se"))) ** 2 + _nan(b.get("se", b.get("mc_se"))) ** 2))
            change = _nan(b["mean"]) - _nan(a["mean"])
            if not np.isfinite(change):
                verdict = NOT_MEASURED
            elif abs(change) <= n_se * se:
                verdict = "non concluant"
            else:
                verdict = "conforme" if np.sign(change) == direction else "non conforme"
            label = key if key == "I_s" or "I_s" not in measures else "I_z (secondary)"
            rows.append({"check": f"{label} {'decreasing' if direction < 0 else 'increasing'} ({zone})", "first": _nan(a["mean"]),
                         "last": _nan(b["mean"]), "verdict": verdict})
    for row in rows:
        row["first_step"], row["last_step"] = first["step"], last["step"]
    return rows


# ---------------------------------------------------------------------
# config, runtime, metadata, logging
# ---------------------------------------------------------------------


def load_mini_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path is not None else DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# environment, data, twin
# ---------------------------------------------------------------------


def zone_of_state(layout, s: int) -> str:
    r, c = divmod(s, layout.grid_shape[1])
    return layout.zone_at((r, c)).appearance.category


def zone_states(layout, category: str) -> list[int]:
    return [s for s in range(N_STATES) if zone_of_state(layout, s) == category]


def _dynamics_kind(spec) -> str:
    """ControllableDeterministic -> "controllable_deterministic", etc."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type(spec).__name__).lower()


def zone_kinds(layout) -> dict[str, str]:
    """Appearance category -> dynamics type of that zone in this layout
    (identity in the experimental environment; in the reference, the
    uncontrollable types become their controllable counterparts)."""
    kinds = {}
    for zone in layout.zones:
        category = zone.appearance.category
        if category in e3_env.ZONE_CATEGORIES:
            kinds[category] = _dynamics_kind(zone.dynamics)
    return kinds


def labelled_between_state_js(post_probs: np.ndarray, labels: np.ndarray) -> float:
    """Console monitoring only -- never an input to the switch or to
    training. Mean JS (bits, summed over latent variables) between the
    posteriors of every pair of batch samples whose true states DIFFER;
    post_probs (n, n_vars, n_classes) and labels (n,) in the same order.
    NaN if every sample shows the same state."""
    pair_js = jensen_shannon_categorical(post_probs[:, None], post_probs[None, :]).sum(axis=-1)
    different = labels[:, None] != labels[None, :]
    return float(pair_js[different].mean()) if different.any() else float("nan")


def load_episode_config() -> dict:
    """experiments/E3_worldmodel_4zones/config.yaml's episode block:
    goal_reward, step_reward, max_steps (H), gamma."""
    with open(EXPERIMENT_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["episode"]


def run_identity(seed: int, themes, condition: str = "experimental") -> dict:
    """What defines a run's world, for meta.yaml (spec section 10): the
    condition, the rotation and quadrant assignment (the permutation), the
    block class A/B and the theme colours."""
    rotation = e3_env.rotation_for_seed_index(seed)
    return {
        "condition": condition,
        "rotation": int(rotation),
        "quadrant_assignment": {str(k): str(v) for k, v in e3_env.quadrant_assignment_for_rotation(rotation).items()},
        "block_class": e3_env.block_class_for_seed(seed),
        "themes": themes_to_dict(themes),
    }


def themes_to_dict(themes) -> dict:
    return {
        "zone_colors": {cat: list(map(int, rgb)) for cat, rgb in themes.zone_colors.items()},
        "agent_marker_color": list(map(int, themes.agent_marker_color)),
        "corridor_color": list(map(int, themes.corridor_color)),
        "goal_color": list(map(int, themes.goal_color)),
        "goal_state": int(themes.goal_state),
    }


def themes_from_dict(d: dict) -> Themes:
    return Themes(
        zone_colors={cat: tuple(int(v) for v in rgb) for cat, rgb in d["zone_colors"].items()},
        agent_marker_color=tuple(int(v) for v in d["agent_marker_color"]), goal_state=int(d["goal_state"]),
        corridor_color=tuple(int(v) for v in d["corridor_color"]), goal_color=tuple(int(v) for v in d["goal_color"]),
    )


@dataclasses.dataclass
class World:
    """Everything fixed for one (seed, condition): layout, themes,
    interactive environment, background and the pre-rendered images."""

    seed: int
    condition: str
    layout: object
    themes: Themes
    env: object
    env_seed: int
    background_t: torch.Tensor
    templates_t: torch.Tensor
    templates_np: np.ndarray

    @property
    def zone_kinds(self) -> dict[str, str]:
        return zone_kinds(self.layout)

    def templates_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.templates_np).tobytes()).hexdigest()


def build_world(cfg: dict, seed: int, condition: str = "experimental", themes=None) -> World:
    """Layout, themes, environment, background and the 144 templates of a
    (seed, condition). The reference environment (spec 2.3) is the
    experimental layout with derive_control applied per zone
    (env.build_control_layout): same permutation, same themes -- always
    the EXPERIMENTAL seed's theme stream -- same goal. The dynamics seed
    comes from the condition's own env_dynamics stream (rng_streams.py);
    the episode parameters from experiments/E3_worldmodel_4zones/config.yaml.
    Tensors are returned on the CPU; the caller moves them. themes (a
    Themes object) replaces the seed's theme draw -- for tests that must
    not depend on the theme stream."""
    if condition not in e3_rng.CONDITIONS:
        raise ValueError(f"condition must be one of {e3_rng.CONDITIONS}, got {condition!r}")
    rotation = e3_env.rotation_for_seed_index(seed)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    layout = e3_env.build_layout(assignment) if condition == "experimental" else e3_env.build_control_layout(assignment)
    k = cfg["render"]["k"]
    if themes is None:
        themes = build_themes(
            e3_env.ZONE_CATEGORIES, goal_state=e3_env.goal_state(), rng=e3_rng.spawn_streams(seed)["theme"],
            min_dist=cfg["render"]["cielab_min_dist"], marker_min_dist=cfg["render"]["cielab_marker_min_dist"],
        )
    episode = load_episode_config()
    env_seed = e3_rng.env_seed(e3_rng.spawn_streams(seed, condition))
    env = e3_env.build_pixel_env(
        layout, themes, goal_reward=episode["goal_reward"], step_reward=episode["step_reward"], max_steps=episode["max_steps"],
        seed=env_seed, k=k, grid_lines=cfg["render"]["grid_lines"],
    )
    background = render_background(layout, themes, k=k, grid_lines=cfg["render"]["grid_lines"])
    background_t = torch.as_tensor(background, dtype=torch.float32).permute(2, 0, 1) / 255.0
    templates_np = np.stack([render(s, layout, themes, k=k, grid_lines=cfg["render"]["grid_lines"]) for s in range(N_STATES)])
    templates_t = torch.as_tensor(templates_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return World(seed, condition, layout, themes, env, env_seed, background_t, templates_t, templates_np)


def build_run(cfg: dict, seed: int, themes=None, condition: str = "experimental"):
    """(layout, themes, env, background_t, templates_t) of build_world."""
    w = build_world(cfg, seed, condition, themes)
    return w.layout, w.themes, w.env, w.background_t, w.templates_t


def collect_and_count(
    env, replay: ReplayBuffer, n_steps: int, rng: np.random.Generator, n_states: int, n_actions: int,
    start_rng: np.random.Generator, starts_out: list | None = None, next_states_out: list | None = None,
) -> np.ndarray:
    """Local to run_mini.py (not a modification of Phase 2's
    collect_fixed_policy_data): fills replay AND accumulates
    N[s, a, s'] counts in the same pass, so the report's tabular-twin
    comparison uses the EXACT SAME transitions the world model trained
    on, without a second collection pass. Every episode starts in a state
    drawn uniformly among the non-terminal states from start_rng (E2's
    protocol, docs/e3/DECISIONS.md 2026-09-25); starts_out, if given,
    receives every start state in order (including one for a final,
    unfinished episode); next_states_out, the true s' of every stored step
    (the replay's next entry is a reset state at an episode's end)."""
    N = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
    start_states = non_terminal_states(env)

    def reset():
        obs0, info0 = env.reset(start_state=draw_start_state(start_states, start_rng))
        if starts_out is not None:
            starts_out.append(int(info0["state"]))
        return obs0, info0

    obs, info = reset()
    s = info["state"]
    is_first = True
    for _ in range(n_steps):
        a = int(rng.integers(n_actions))
        obs_next, r, terminated, truncated, info = env.step(a)
        s_next = info["state"]
        replay.add(x=obs, a=a, r=r, terminal=terminated, truncated=truncated, is_first=is_first, s=s)
        N[s, a, s_next] += 1
        if next_states_out is not None:
            next_states_out.append(int(s_next))
        is_first = terminated or truncated
        if is_first:
            obs_next, info = reset()
        s = info["state"]
        obs = obs_next
    return N


def kl_post_prior_by_zone(probe: dict, layout) -> dict[str, float]:
    """Canonical-probe "KL by zone type", in BITS: mean_a KL(post_probs[s]
    || prior_probs[s,a]), averaged over states of each zone category.
    CAUTION (docs/e3/DECISIONS.md, 2026-09-28): post_probs[s] is the
    posterior of the CURRENT state s (at h0) and prior_probs[s,a] the
    prediction of the NEXT state after a -- the two describe different
    states whenever s' != s, so this quantity is large even for a perfect
    model and is not the training loss's KL. Kept, labelled, as the
    secondary record; the model-fit KL is the real-context one
    (probe.real_context_probe's kl_next_bits). rssm.categorical_kl works in
    nats; converted here."""
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
    support = geographic_support_mask(GRID_SHAPE)
    agent = TabularModelBasedAgent(n_states, n_actions, alpha=1.0, alpha_support=support)
    agent.N = N
    return agent.T_hat


def frequent_outcome_preview(p_hat: np.ndarray, layout) -> dict[str, float]:
    """T4 preview (spec section 9): mean, over (s, a) pairs in each
    noisy zone, of the world model's own s-space kernel's max
    probability (the "frequent outcome"); states not measured (NaN rows)
    are left out. Not the full T4 statistical test -- a shape preview
    only."""
    out = {}
    for zone in NOISY_ZONES:
        rows = p_hat[zone_states(layout, zone)]  # (n_states_in_zone, n_actions, n_states)
        max_probs = np.where(np.isnan(rows).any(axis=-1), np.nan, np.nan_to_num(rows).max(axis=-1))
        out[zone] = float(np.nanmean(max_probs)) if np.isfinite(max_probs).any() else float("nan")
    return out


def save_collapse_grid(rssm: RSSM, templates: torch.Tensor, background: torch.Tensor, k: int, states: list[int], path: Path) -> None:
    """Collapse diagnostic figure: for each state, the input image, the
    reconstruction-loss mask computed from it, and the POSTERIOR
    reconstruction (canonical path: fresh h0, z = posterior mode). A
    collapsed posterior shows the same reconstruction in every column,
    with no agent marker."""
    plt = _plt()
    device = next(rssm.parameters()).device
    x = templates[states].to(device)
    with torch.no_grad():
        h0, _ = rssm.initial_state(len(states), device)
        z0 = categorical_mode(rssm.posterior_probs(h0, rssm.encoder(x)))
        recon = rssm.decode(h0, z0).clamp(0.0, 1.0)
        mask = reconstruction_mask(x, background.to(device).unsqueeze(0).expand_as(x), k)

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


# ---------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------


def real_context_settings(cfg: dict) -> dict:
    """measure.real_context, with window_len null = the training seq_len."""
    rc = dict(cfg["measure"]["real_context"])
    if rc.get("window_len") is None:
        rc["window_len"] = cfg["train"]["seq_len"]
    if rc["burn_in"] >= rc["window_len"]:
        raise ValueError(f"measure.real_context.burn_in ({rc['burn_in']}) must be < window_len ({rc['window_len']})")
    return rc


def kernel_draws(cfg: dict, final: bool) -> tuple[int, int]:
    """(K for the real-context kernel, K_s for the canonical kernel): the
    final values (measure.real_context.kernel_draws, measure.kernel_k_s),
    or during training the reduced *_training values when the config sets
    them (docs/e3/DECISIONS.md, 2026-09-30)."""
    m, rc = cfg["measure"], cfg["measure"]["real_context"]
    if final:
        return rc["kernel_draws"], m["kernel_k_s"]
    return rc.get("kernel_draws_training", rc["kernel_draws"]), m.get("kernel_k_s_training", m["kernel_k_s"])


def controllability_of_kernel(p_hat: np.ndarray) -> np.ndarray:
    """I_s per state: measures.controllability.controllability_per_state of
    the world model's s-space kernel p_hat (n_states, n_actions, n_states);
    NaN for a state with a row that is not a law (not measured, or every
    draw of an action a ghost)."""
    valid = np.isfinite(p_hat).all(axis=(1, 2)) & np.isclose(np.nan_to_num(p_hat).sum(axis=-1), 1.0).all(axis=1)
    out = np.full(p_hat.shape[0], np.nan)
    if valid.any():
        out[valid] = controllability_per_state(p_hat[valid])
    return out


def per_context_is_check(rssm: RSSM, world: World, contexts: dict, cfg: dict, rng: np.random.Generator,
                         generator: torch.Generator, states_per_zone: int, n_contexts: int, draws: int) -> dict:
    """I_s per context vs pooled over the same contexts (docs/e3/DECISIONS.md,
    2026-09-30): the SECONDARY diagnostic reported with every final
    measurement. The main I_s pools the decoded draws of a state's contexts
    into one kernel (as E2's per-state kernel and the tabular twin B: one
    kernel per state, histories merged); here, for states_per_zone states per
    zone (drawn from `rng` among those with >= n_contexts contexts) and
    n_contexts of their contexts, `draws` prior draws per context and action
    (probe.real_context_action_samples) give each context its own kernel:
      per_context   mean over the contexts of I_s(context's kernel);
      pooled        I_s of the kernel pooling the same contexts' draws;
      gap           pooled - per_context (<= 0 up to sampling: mutual
                    information is convex in the channel);
    with the bootstrap bias of each, per state and by zone."""
    layout = world.layout
    chosen_states, chosen_ctx = [], []
    for cat in e3_env.ZONE_CATEGORIES:
        candidates = [s for s in zone_states(layout, cat) if (contexts["states"] == s).sum() >= n_contexts]
        for s in sorted(rng.choice(candidates, size=min(states_per_zone, len(candidates)), replace=False)) if candidates else []:
            chosen_states.append(int(s))
            chosen_ctx.append(np.sort(rng.choice(np.flatnonzero(contexts["states"] == s), size=n_contexts, replace=False)))
    if not chosen_states:
        return {"states": [], "by_zone": {}, "settings": {"states_per_zone": states_per_zone, "contexts": n_contexts, "draws": draws}}
    idx = np.concatenate(chosen_ctx)
    idx_t = torch.as_tensor(idx, device=contexts["h"].device)
    sub = {"positions": contexts["positions"][idx], "states": contexts["states"][idx],
           "prior_probs_t": contexts["prior_probs_t"][idx_t], "h_next": contexts["h_next"][idx_t]}
    samples = e3_probe.real_context_action_samples(rssm, sub, world.templates_t, draws=draws,
                                                   margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=generator)
    keep = ~samples["is_ghost"]
    counts = np.zeros((idx.size, N_ACTIONS, N_STATES))
    np.add.at(counts, (samples["context"][keep], samples["a"][keep], samples["s_next"][keep]), 1.0)
    with np.errstate(invalid="ignore"):
        kernels = counts / counts.sum(axis=-1, keepdims=True)
    rows = []
    for k, s in enumerate(chosen_states):
        block = kernels[k * n_contexts:(k + 1) * n_contexts]
        valid = np.isfinite(block).all(axis=(1, 2))
        I_ctx = controllability_per_state(block[valid])
        pooled = block[valid].mean(axis=0, keepdims=True)  # the same number of draws per context
        I_pool = float(controllability_per_state(pooled)[0])
        rows.append({"s": s, "zone": zone_of_state(layout, s), "n_contexts": int(valid.sum()),
                     "per_context": float(I_ctx.mean()), "per_context_sd": float(I_ctx.std(ddof=1)) if I_ctx.size > 1 else float("nan"),
                     "per_context_bias": float(np.mean(controllability_bootstrap(block[valid], draws, 200, rng)["bias"])),
                     "pooled": I_pool, "pooled_bias": float(controllability_bootstrap(pooled, draws * int(valid.sum()), 200, rng)["bias"][0]),
                     "gap": I_pool - float(I_ctx.mean())})
    by_zone = {}
    for cat in e3_env.ZONE_CATEGORIES:
        zr = [r for r in rows if r["zone"] == cat]
        if not zr:
            continue
        gaps = np.array([r["gap"] for r in zr])
        by_zone[cat] = {**{key: float(np.mean([r[key] for r in zr]))
                           for key in ("per_context", "per_context_bias", "pooled", "pooled_bias", "gap")},
                        "gap_se_over_states": float(gaps.std(ddof=1) / np.sqrt(gaps.size)) if gaps.size > 1 else float("nan"),
                        "n_states": len(zr)}
    return {"states": rows, "by_zone": by_zone, "ghost_fraction": float(samples["is_ghost"].mean()),
            "settings": {"states_per_zone": states_per_zone, "contexts": n_contexts, "draws": draws}}


def is_by_zone(I_s: np.ndarray, boot: dict, layout) -> dict[str, dict]:
    """Zone summaries of I_s: mean over the measured states with the se of
    the mean (zone_mean_with_se, per-state bootstrap se), and the mean
    bootstrap estimate of the plug-in bias."""
    out = {}
    for cat in e3_env.ZONE_CATEGORIES:
        states = zone_states(layout, cat)
        bias = boot["bias"][states]
        out[cat] = {**zone_mean_with_se(I_s, boot["se"], states),
                    "bias": float(np.nanmean(bias)) if np.isfinite(bias).any() else float("nan")}
    return out


def real_context_measures(
    rssm: RSSM, world: World, replay: ReplayBuffer, P_true: np.ndarray, T_hat_B: np.ndarray, cfg: dict,
    rng: np.random.Generator, generator: torch.Generator, M: int, final: bool = True,
) -> tuple[dict, dict]:
    """The main measure, in real contexts: per state, the s-space kernel
    decoded from prior draws of the state's contexts and its
    controllability I_s = I(a ; s' | s) (THE MAIN MEASURE since
    docs/e3/DECISIONS.md 2026-09-30: measures.controllability on the
    decoded arrival states), with a parametric-bootstrap se and bias; I_z
    (mean over the contexts, SECONDARY since 2026-09-30: an upper bound),
    the one-step KL, the linear probes h -> s and z -> s on the contexts;
    aggregated by zone over the MEASURED states (a state without a context
    is "non mesuré"). final: the final kernel sizes, else the reduced
    training ones (kernel_draws). Returns a JSON-able record and the arrays
    (contexts included)."""
    rc = real_context_settings(cfg)
    draws, _ = kernel_draws(cfg, final)
    layout = world.layout
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"],
                                      max_per_state=rc["max_per_state"], n_states=N_STATES, rng=rng, generator=generator,
                                      max_images=max_images_per_forward(cfg))
    iz = e3_probe.real_context_iz(ctx, M=M, rng=rng)
    kl = per_state_context_summary(ctx["kl_next_bits"], ctx["states"], N_STATES)
    # The reachable successors of each state: the support of the tabular twin's prior (build_tabular_twin).
    support = geographic_support_mask(GRID_SHAPE)
    kernel = e3_probe.real_context_kernel(rssm, ctx, world.templates_t, draws=draws,
                                          margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=generator,
                                          support=support)
    acc = e3_probe.real_context_template_accuracy(rssm, ctx, world.templates_t)
    l1_vs_p = np.abs(kernel["p_hat"] - P_true).sum(axis=-1)  # (144, 5), NaN for unmeasured states
    l1_vs_b = np.abs(kernel["p_hat"] - T_hat_B).sum(axis=-1)
    l1_b_vs_p = np.abs(T_hat_B - P_true).sum(axis=-1)
    # C-b's sampling floor (docs/e3/DECISIONS.md, 2026-09-29): the L1 an exact kernel scores with this many draws
    l1_floor = np.where(np.isnan(l1_vs_p), np.nan, expected_l1_sampling_floor(P_true, draws))
    I_s = controllability_of_kernel(kernel["p_hat"])
    boot = controllability_bootstrap(kernel["p_hat"], draws, cfg["measure"].get("is_bootstrap", 200), rng)
    probes = e3_probe.context_state_probes(ctx, cfg["measure"]["z_only_test_frac"], rng) if ctx["positions"].size else None
    # docs/e3/DECISIONS.md, 2026-09-30 (5): I_s above is computed on the kernel as it is (the model's actual belief,
    # off-support mass included). Reported next to it: the share of that mass outside the reachable successors, and
    # I_s of the kernel projected on the support (SECONDARY: for the comparison with the tabular twin, whose prior
    # is restricted to the same support). The projected bootstrap draws last, so everything above is unchanged.
    off_support = off_support_mass(kernel["p_hat"], support)
    I_s_projected = controllability_of_kernel(kernel["p_hat_projected"])
    boot_projected = controllability_bootstrap(kernel["p_hat_projected"], draws, cfg["measure"].get("is_bootstrap", 200), rng)
    cats = list(e3_env.ZONE_CATEGORIES)

    def by_zone(arr: np.ndarray) -> dict[str, float]:
        out = {}
        for cat in cats:
            vals = arr[zone_states(layout, cat)]
            out[cat] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        return out

    n = iz["n_contexts"]
    record = {
        "kl_by_zone": {cat: zone_mean_with_se(kl["mean"], kl["se"], zone_states(layout, cat))["mean"] for cat in cats},
        "l1_vs_p_by_zone": by_zone(l1_vs_p),
        "l1_vs_b_by_zone": by_zone(l1_vs_b),
        "l1_b_vs_p_by_zone": by_zone(l1_b_vs_p),
        "l1_floor_by_zone": by_zone(l1_floor),
        "kernel_draws": draws,
        "I_s_by_zone": is_by_zone(I_s, boot, layout),
        "I_s_projected_by_zone": is_by_zone(I_s_projected, boot_projected, layout),
        "off_support_mass_by_zone": by_zone(off_support),
        "ghost_fraction_projected_mean": (float(np.nanmean(kernel["ghost_fraction_projected"]))
                                          if np.isfinite(kernel["ghost_fraction_projected"]).any() else float("nan")),
        "I_z_by_zone": {cat: zone_mean_with_se(iz["I_z"], iz["se"], zone_states(layout, cat)) for cat in cats},
        "state_probe_h": None if probes is None else probes["h"]["test_accuracy"],
        "state_probe_z": None if probes is None else probes["z"]["test_accuracy"],
        "frequent_outcome": frequent_outcome_preview(kernel["p_hat"], layout),
        "posterior_entropy": float(e3_probe.per_variable_entropy(ctx["post_probs"]).mean()) if n.sum() else float("nan"),
        "acc_real_context": acc,
        "ghost_fraction_mean": float(np.nanmean(kernel["ghost_fraction"])) if np.isfinite(kernel["ghost_fraction"]).any() else float("nan"),
        "n_contexts_total": int(n.sum()),
        "n_states_measured": int((n > 0).sum()),
        "contexts_per_state": {"min": int(n.min()), "median": float(np.median(n)), "max": int(n.max())},
        "not_measured_states": [int(s) for s in np.flatnonzero(n == 0)],
        "settings": rc,
    }
    arrays = {"contexts": ctx, "iz": iz, "kl": kl, "kernel": kernel, "l1_vs_p": l1_vs_p, "l1_vs_b": l1_vs_b,
              "l1_b_vs_p": l1_b_vs_p, "I_s": I_s, "I_s_se": boot["se"], "I_s_bias": boot["bias"],
              "I_s_projected": I_s_projected, "I_s_projected_se": boot_projected["se"], "I_s_projected_bias": boot_projected["bias"],
              "off_support": off_support}
    return record, arrays


def training_diagnostics(
    rssm: RSSM, world: World, P_true: np.ndarray, T_hat_B: np.ndarray, replay: ReplayBuffer, cfg: dict,
    rng: np.random.Generator, generator: torch.Generator, M: int, final: bool = True,
) -> tuple[dict, dict]:
    """One point of the training diagnostics (docs/e3/DECISIONS.md,
    2026-09-25 and 2026-09-28), also used for the final measurement.
    Top-level keys: the canonical probe (secondary since 2026-09-28) and the
    template accuracies; "real_context": the main measure. Returns a
    JSON-able record and the full arrays it was computed from (reused by
    the final report). All draws come from (rng, generator), owned by the
    caller. M: Monte-Carlo samples per I_z (measure.iz_mc_samples_training
    during training, measure.iz_mc_samples_final at the end). final: the
    final kernel sizes, else the reduced training ones (kernel_draws)."""
    m = cfg["measure"]
    _, k_s = kernel_draws(cfg, final)
    layout, templates_t = world.layout, world.templates_t
    was_training = rssm.training
    rssm.eval()
    t0 = time.perf_counter()
    iz = e3_probe.canonical_iz(rssm, templates_t, M=M, rng=rng, generator=generator)
    probe = iz["probe"]
    kernel = e3_probe.canonical_kernel_probe(rssm, templates_t, K_s=k_s, margin_threshold=m["kernel_margin_threshold"],
                                             probe=probe, generator=generator)
    separation = e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=5, generator=generator)
    batch, labels = replay.sample(32, 16, rng)
    acc_replay = e3_probe.template_match_accuracy(rssm, batch, labels, templates_t, margin_threshold=m["kernel_margin_threshold"],
                                                  generator=generator)
    real_record, real_arrays = real_context_measures(rssm, world, replay, P_true, T_hat_B, cfg, rng, generator, M, final=final)
    rssm.train(was_training)

    l1_vs_p = np.abs(kernel["p_hat"] - P_true).sum(axis=-1)  # (144, 5)
    l1_vs_b = np.abs(kernel["p_hat"] - T_hat_B).sum(axis=-1)
    l1_b_vs_p = np.abs(T_hat_B - P_true).sum(axis=-1)
    l1_floor = expected_l1_sampling_floor(P_true, k_s)
    I_s_canonical = controllability_of_kernel(kernel["p_hat"])  # secondary: the canonical kernel (h0 contexts)
    boot_canonical = controllability_bootstrap(kernel["p_hat"], k_s, m.get("is_bootstrap", 200), rng)
    cats = list(e3_env.ZONE_CATEGORIES)

    def by_zone(arr: np.ndarray) -> dict[str, float]:
        return {cat: float(np.nanmean(arr[zone_states(layout, cat)])) for cat in cats}

    record = {
        "kl_by_zone": kl_post_prior_by_zone(probe, layout),
        "l1_vs_p_by_zone": by_zone(l1_vs_p),
        "l1_vs_b_by_zone": by_zone(l1_vs_b),
        "l1_b_vs_p_by_zone": by_zone(l1_b_vs_p),
        "l1_floor_by_zone": by_zone(l1_floor),
        "kernel_k_s": k_s,
        "I_s_by_zone": is_by_zone(I_s_canonical, boot_canonical, layout),
        "I_z_by_zone": {cat: {"mean": float(iz["I_z"][zone_states(layout, cat)].mean()),
                              "mc_se": zone_mean_mc_se(iz["se"][zone_states(layout, cat)])} for cat in cats},
        "frequent_outcome": frequent_outcome_preview(kernel["p_hat"], layout),
        "posterior_entropy_canonical": float(probe["posterior_entropy"].mean()),
        "acc_replay_hard": acc_replay,
        "acc_real_context": real_record["acc_real_context"],
        "acc_canonical_hard": separation["acc_hard"],
        "acc_canonical_mode": separation["acc_mode"],
        "ghost_fraction_mean": float(np.nanmean(kernel["ghost_fraction"])),
        "real_context": real_record,
        "iz_mc_samples": M,
        "seconds": time.perf_counter() - t0,
    }
    return record, {"iz": iz, "probe": probe, "kernel": kernel, "l1_vs_p": l1_vs_p, "l1_vs_b": l1_vs_b, "l1_b_vs_p": l1_b_vs_p,
                    "real": real_arrays}


def _d1_pairs(cats: list[str], stats: dict, cfg: dict, per_state) -> list[dict]:
    """The D1 table (spec section 8, criteria of docs/e3/DECISIONS.md
    2026-09-25): for every ordered pair of zones (own L, target L'), the
    contrast is accepted only if |I_L' - I_L| > max(d1_contrast_min_bits,
    3 x combined se). For an accepted pair, alpha_app is computed for every
    (measured) state of L, local and global, with its standard error;
    per-state values outside [0, 1] beyond 3 se are anomalies (listed,
    never averaged); the others are summarised by their mean over the
    zone's states +/- 3 sd/sqrt(n). per_state(own, target, global_swap) ->
    (states, I_swap, I_swap_se)."""
    results = []
    for own in cats:
        for target in cats:
            if own == target:
                continue
            (I_L, se_L), (I_Lp, se_Lp) = stats[own], stats[target]
            if not (np.isfinite(I_L) and np.isfinite(I_Lp)):
                results.append({"own_category": own, "target_category": target, "I_L": I_L, "se_L": se_L, "I_Lprime": I_Lp,
                                "se_Lprime": se_Lp, "contrast": float("nan"), "threshold": float("nan"), "combined_se": float("nan"),
                                "accepted": False, "not_measured": True})
                continue
            entry = {"own_category": own, "target_category": target, "I_L": I_L, "se_L": se_L, "I_Lprime": I_Lp,
                     "se_Lprime": se_Lp, **d1_contrast(I_L, se_L, I_Lp, se_Lp, min_bits=cfg["measure"]["d1_contrast_min_bits"], n_se=N_SE)}
            if entry["accepted"]:
                for kind, global_swap in (("local", False), ("global", True)):
                    states, I_swap, I_swap_se = per_state(own, target, global_swap)
                    alpha, se = alpha_app_with_se(I_swap, I_swap_se, I_L, se_L, I_Lp, se_Lp)
                    summary = summarize_alpha_app(alpha, se, n_se=N_SE)
                    for anomaly in summary["anomalies"]:
                        anomaly["s"] = states[anomaly.pop("index")]
                    summary["per_state"] = [{"s": s, "alpha": float(a), "se": float(e)} for s, a, e in zip(states, alpha, se)]
                    entry[kind] = summary
            results.append(entry)
    return results


def d1_analysis(rssm: RSSM, layout, themes, I_z: np.ndarray, I_z_se: np.ndarray, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """D1 from the canonical probe (secondary since 2026-09-28): I_swap from
    a fresh h0 on the counterfactual render of each state of L."""
    cats = list(e3_env.ZONE_CATEGORIES)
    stats = {cat: (float(I_z[zone_states(layout, cat)].mean()), zone_mean_mc_se(I_z_se[zone_states(layout, cat)])) for cat in cats}
    I_by_category = {cat: v[0] for cat, v in stats.items()}

    def per_state(own, target, global_swap):
        states = zone_states(layout, own)
        swaps = [e3_probe.d1_alpha_app_for_state(rssm, layout, themes, s, target, I_by_category, M=cfg["measure"]["iz_mc_samples_final"],
                                                 rng=rng, k=cfg["render"]["k"], contrast_threshold=0.0, global_swap=global_swap)
                 for s in states]
        return states, np.array([d["I_swap"] for d in swaps]), np.array([d["I_swap_se"] for d in swaps])

    return _d1_pairs(cats, stats, cfg, per_state)


def d1_analysis_real(rssm: RSSM, world: World, real_iz: dict, contexts: dict, cfg: dict, rng: np.random.Generator,
                     generator: torch.Generator) -> list[dict]:
    """D1 in real contexts (the main D1 since 2026-09-28): zone means and
    their standard errors from the real-context I_z of the measured states;
    I_swap(s) = mean over the contexts of s of I_z with x_t replaced by the
    counterfactual render (probe.real_context_d1_iswap). Only measured
    states of L enter."""
    layout, cats = world.layout, list(e3_env.ZONE_CATEGORIES)
    zone_stats = {cat: zone_mean_with_se(real_iz["I_z"], real_iz["se"], zone_states(layout, cat)) for cat in cats}
    stats = {cat: (v["mean"], v["se"]) for cat, v in zone_stats.items()}

    def per_state(own, target, global_swap):
        states = [s for s in zone_states(layout, own) if real_iz["n_contexts"][s] > 0]
        swaps = [e3_probe.real_context_d1_iswap(rssm, contexts, layout, world.themes, s, target, k=cfg["render"]["k"],
                                                global_swap=global_swap, M=cfg["measure"]["iz_mc_samples_final"], rng=rng,
                                                generator=generator) for s in states]
        return states, np.array([d["I_swap"] for d in swaps]), np.array([d["I_swap_se"] for d in swaps])

    return _d1_pairs(cats, stats, cfg, per_state)


def measure_final(rssm: RSSM, world: World, replay: ReplayBuffer, N: np.ndarray, cfg: dict, streams: dict, run_dir: Path,
                  log: RunLog, step: int) -> tuple[dict, dict]:
    """Every final measure of a trained model, in real contexts (main) and
    from the canonical probe (secondary). Used by run_one_seed at the end
    of training and by recompute_measures.py on any saved checkpoint:
    same measurement streams, so the same weights and data give the same
    numbers on the same device. Returns (report fields, arrays)."""
    seed = streams["seed"]
    layout, themes, k = world.layout, world.themes, cfg["render"]["k"]
    T_hat_B = build_tabular_twin(N, N_STATES, N_ACTIONS)
    P_true = world.env.transition_tensor()
    rssm.eval()
    device = next(rssm.parameters()).device
    gpu_peak_start(device)
    log("final measurements (real contexts, canonical probe, s-space kernels, template accuracies) ...")
    rng, gen = measure_streams(seed, SLOT_FINAL, streams)
    final, extras = training_diagnostics(rssm, world, P_true, T_hat_B, replay, cfg, rng, gen, cfg["measure"]["iz_mc_samples_final"])
    final = {"step": step, **final}
    iz_result, probe, kernel_result, real = extras["iz"], extras["probe"], extras["kernel"], extras["real"]

    log("collapse diagnostics (I(z;s), posterior entropy, grid) ...")
    rng_izs, _ = measure_streams(seed, SLOT_IZS, streams)
    izs = latent_state_information(probe["posterior_probs"], M=cfg["measure"]["iz_mc_samples_final"], rng=rng_izs)
    grid_states = [s for cat in e3_env.ZONE_CATEGORIES for s in zone_states(layout, cat)[:2]]
    grid_name = f"collapse_grid_seed{seed:02d}.png"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    save_collapse_grid(rssm, world.templates_t, world.background_t, k, grid_states, run_dir / "artifacts" / grid_name)

    log("z-only probe (every replay posterior, in chunks) ...")
    rng_z, gen_z = measure_streams(seed, SLOT_ZONLY, streams)
    z_seq_len = cfg["measure"].get("z_only_seq_len", 64)
    z_probe_result = e3_probe.z_only_probe_chunked(rssm, replay.iter_chronological_windows(z_seq_len, chunk=256), n_states=N_STATES,
                                                   test_frac=cfg["measure"]["z_only_test_frac"], rng=rng_z, generator=gen_z,
                                                   max_images=max_images_per_forward(cfg))

    log("h-only probe (every replay deterministic state h_t, in chunks) ...")
    rng_h, gen_h = measure_streams(seed, SLOT_HONLY, streams)
    h_probe_result = e3_probe.z_only_probe_chunked(rssm, replay.iter_chronological_windows(z_seq_len, chunk=256), n_states=N_STATES,
                                                   test_frac=cfg["measure"]["z_only_test_frac"], rng=rng_h, generator=gen_h,
                                                   max_images=max_images_per_forward(cfg), latent="h")

    pc = cfg["measure"].get("is_per_context")
    per_context = None
    if pc:
        log(f"I_s per context vs pooled (secondary; {pc['states_per_zone']} states per zone, {pc['contexts']} contexts, "
            f"{pc['draws']} draws) ...")
        rng_pc, gen_pc = measure_streams(seed, SLOT_PER_CONTEXT, streams)
        per_context = per_context_is_check(rssm, world, real["contexts"], cfg, rng_pc, gen_pc, pc["states_per_zone"], pc["contexts"],
                                           pc["draws"])

    log("D1 in real contexts (every measured state of every zone with an accepted contrast) ...")
    rng_d1, gen_d1 = measure_streams(seed, SLOT_D1_REAL, streams)
    d1_real = d1_analysis_real(rssm, world, real["iz"], real["contexts"], cfg, rng_d1, gen_d1)
    d2_real = e3_probe.d2_neighbor_distances_contexts(real["contexts"], layout)
    log("D1 / D2 from the canonical probe (secondary) ...")
    rng_d1c, _ = measure_streams(seed, SLOT_D1_CANONICAL, streams)
    d1_canonical = d1_analysis(rssm, layout, themes, iz_result["I_z"], iz_result["se"], cfg, rng_d1c)
    d2_canonical = e3_probe.d2_neighbor_distances(probe["posterior_probs"], layout)

    # Context dispersion (spec 5.5), from the real contexts: per state, the gap between the real-context
    # mean and the canonical value, and the spread of I_z over the state's own contexts.
    gap = np.abs(real["iz"]["I_z"] - iz_result["I_z"])
    dispersion = {cat: {"mean_abs_gap_real_vs_canonical": float(np.nanmean(gap[zone_states(layout, cat)]))
                        if np.isfinite(gap[zone_states(layout, cat)]).any() else float("nan"),
                        "mean_sd_over_contexts": float(np.nanmean(real["iz"]["sd"][zone_states(layout, cat)]))
                        if np.isfinite(real["iz"]["sd"][zone_states(layout, cat)]).any() else float("nan")}
                  for cat in e3_env.ZONE_CATEGORIES}

    fields = {
        "final_step": step,
        "diagnostics_final": final,
        "zone_categories": list(e3_env.ZONE_CATEGORIES),
        "zone_kinds": world.zone_kinds,
        "zone_of_state": [zone_of_state(layout, s) for s in range(N_STATES)],
        "real_context": final["real_context"],
        "I_s_real": real["I_s"], "I_s_real_se": real["I_s_se"], "I_s_real_bias": real["I_s_bias"],
        "I_z_real": real["iz"]["I_z"], "I_z_real_se": real["iz"]["se"], "I_z_real_sd": real["iz"]["sd"],
        "n_contexts": real["iz"]["n_contexts"],
        "I_z": iz_result["I_z"], "I_z_se": iz_result["se"],
        "I_zs": izs["I_zs"], "I_zs_se": izs["se"],
        "canonical_posterior_entropy_mean": float(probe["posterior_entropy"].mean()),
        "posterior_entropy_ceiling": float(rssm.cfg.n_vars * np.log2(rssm.cfg.n_classes)),
        "collapse_grid": grid_name,
        "I_by_category": {cat: float(iz_result["I_z"][zone_states(layout, cat)].mean()) for cat in e3_env.ZONE_CATEGORIES},
        "I_by_category_real": {cat: final["real_context"]["I_z_by_zone"][cat]["mean"] for cat in e3_env.ZONE_CATEGORIES},
        "I_s_by_category_real": {cat: final["real_context"]["I_s_by_zone"][cat]["mean"] for cat in e3_env.ZONE_CATEGORIES},
        "I_s_projected_by_category_real": {cat: final["real_context"]["I_s_projected_by_zone"][cat]["mean"]
                                           for cat in e3_env.ZONE_CATEGORIES},
        "accuracies": {key: final[key] for key in ACC_LABELS},
        "ghost_fraction_mean": final["ghost_fraction_mean"],
        "z_only_probe": z_probe_result,
        "h_only_probe": h_probe_result,
        "is_per_context": per_context,
        "context_dispersion": dispersion,
        "d1": d1_real,
        "d1_canonical": d1_canonical,
        "d2_boundary_mean": float(d2_real["boundary"].mean()) if d2_real["boundary"].size else float("nan"),
        "d2_within_zone_mean": float(d2_real["within_zone"].mean()) if d2_real["within_zone"].size else float("nan"),
        "d2_canonical_boundary_mean": float(d2_canonical["boundary"].mean()),
        "d2_canonical_within_zone_mean": float(d2_canonical["within_zone"].mean()),
        "kl_by_zone": final["kl_by_zone"],
        "l1_vs_p_by_zone": final["l1_vs_p_by_zone"],
        "l1_vs_b_by_zone": final["l1_vs_b_by_zone"],
        "l1_b_vs_p_by_zone": final["l1_b_vs_p_by_zone"],
        "l1_floor_by_zone": final["l1_floor_by_zone"],
        "frequent_outcome_preview": final["frequent_outcome"],
        "gpu_memory_final": gpu_peak_end(device),
    }
    if fields["gpu_memory_final"] is not None:
        log(f"final measures: GPU peak {fields['gpu_memory_final']['peak_allocated_mib']:.0f} MiB allocated, "
            f"{fields['gpu_memory_final']['peak_reserved_mib']:.0f} MiB reserved")
    ctx = real["contexts"]
    arrays = {
        "N": N, "T_hat_B": T_hat_B, "P_true": P_true,
        "p_hat": kernel_result["p_hat"], "ghost_fraction": kernel_result["ghost_fraction"],
        "I_z": iz_result["I_z"], "I_z_se": iz_result["se"],
        "posterior_probs": probe["posterior_probs"], "prior_probs": probe["prior_probs"],
        "l1_vs_p": extras["l1_vs_p"], "l1_vs_b": extras["l1_vs_b"], "l1_b_vs_p": extras["l1_b_vs_p"],
        "real_I_s": real["I_s"], "real_I_s_se": real["I_s_se"], "real_I_s_bias": real["I_s_bias"],
        "real_I_s_projected": real["I_s_projected"], "real_I_s_projected_se": real["I_s_projected_se"],
        "real_I_s_projected_bias": real["I_s_projected_bias"], "real_p_hat_projected": real["kernel"]["p_hat_projected"],
        "real_ghost_fraction_projected": real["kernel"]["ghost_fraction_projected"], "real_off_support": real["off_support"],
        "real_I_z": real["iz"]["I_z"], "real_I_z_se": real["iz"]["se"], "real_I_z_sd": real["iz"]["sd"],
        "real_n_contexts": real["iz"]["n_contexts"], "real_I_z_context": real["iz"]["I_z_context"],
        "real_I_z_context_se": real["iz"]["se_context"], "real_kl_per_state": real["kl"]["mean"],
        "real_kl_next_bits": ctx["kl_next_bits"], "real_p_hat": real["kernel"]["p_hat"],
        "real_ghost_fraction": real["kernel"]["ghost_fraction"], "real_l1_vs_p": real["l1_vs_p"], "real_l1_vs_b": real["l1_vs_b"],
        "real_context_positions": ctx["positions"], "real_context_states": ctx["states"], "real_n_eligible": ctx["n_eligible"],
        "real_post_probs": ctx["post_probs"], "real_prior_probs": ctx["prior_probs"],
    }
    return fields, arrays


# ---------------------------------------------------------------------
# checkpoints, resume
# ---------------------------------------------------------------------


def e2_grid_env_steps() -> list[int]:
    """E2's 16-point checkpoint grid in environment steps
    (experiments/E3_worldmodel_4zones/config.yaml, = E2's sweep.CHECKPOINTS)."""
    with open(EXPERIMENT_CONFIG_PATH, encoding="utf-8") as f:
        return [int(c) for c in yaml.safe_load(f)["budget"]["checkpoints"]]


def checkpoint_steps(cfg: dict, planned_steps: int) -> list[int]:
    """Gradient steps at which the weights are saved (and, with
    measure.diagnostics_at_checkpoints, measured). checkpoints.grid "e2":
    E2's grid points c <= the run's environment steps, mapped to gradient
    step round(c * replay_ratio) -- in E3-0 all the data is collected
    first, so checkpoint c is "the training a run at replay_ratio would
    have done after c environment steps" (docs/e3/DECISIONS.md,
    2026-09-28). A list: explicit gradient steps. null: none. The final
    step is always saved (weights_final.pt), not listed here."""
    grid = (cfg.get("checkpoints") or {}).get("grid")
    if grid is None:
        return []
    if grid == "e2":
        total_env = cfg["budget"]["total_steps"] + cfg["budget"]["prefill_steps"]
        steps = {int(round(c * cfg["train"]["replay_ratio"])) for c in e2_grid_env_steps() if c <= total_env}
    else:
        steps = {int(g) for g in grid}
    return sorted(g for g in steps if 0 < g < planned_steps)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_weights(run_dir: Path, rssm: RSSM, step: int, name: str | None = None, **extra) -> Path:
    path = run_dir / "checkpoints" / (name or f"weights_step{step:08d}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save({"rssm": rssm.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg), "step": step, **extra}, path)
    return path


def _generator_state(g) -> object:
    return g.bit_generator.state if isinstance(g, np.random.Generator) else g.get_state()


def _set_generator_state(g, state) -> None:
    if isinstance(g, np.random.Generator):
        g.bit_generator.state = state
    else:
        g.set_state(state)


# ---------------------------------------------------------------------
# one (seed, condition)
# ---------------------------------------------------------------------


def save_failure_diagnostic(fail_dir: Path, cfg: dict, seed: int, rssm: RSSM, optimizer: torch.optim.Optimizer,
                            templates_t: torch.Tensor, step: int, switch: SeparationSwitch,
                            error: SeparationNotReachedError | CollapseAfterSwitchError, meta: dict, log: RunLog,
                            condition: str = "experimental") -> Path:
    """Called when SeparationNotReachedError or CollapseAfterSwitchError is
    raised (docs/e3/DECISIONS.md, 2026-09-24): saves the checkpoint (model +
    optimizer state) and the latent-usage diagnostic (probe.latent_usage_diagnostic:
    template accuracy with the decoder fed probabilities / mode / hard
    sample, posterior entropy, I(z;s), logit norms) into the run's failure/."""
    kind = "collapse_after_switch" if isinstance(error, CollapseAfterSwitchError) else "separation_not_reached"
    fail_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"rssm": rssm.state_dict(), "optimizer": optimizer.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg),
                "step": step}, fail_dir / "checkpoint.pt")
    rssm.eval()
    diag = e3_probe.latent_usage_diagnostic(rssm, templates_t, M=cfg["measure"]["iz_mc_samples_final"],
                                            rng=e3_rng.measure_streams(seed, SLOT_FAILURE, next(rssm.parameters()).device, condition)[0])
    record = {"seed": seed, "condition": condition, "steps_done": step, "kind": kind, "error": str(error), **meta, "config": cfg,
              "switch_history": [list(map(float, h)) for h in switch.history], "diagnostic": diag}
    (fail_dir / "diagnostic.json").write_text(json.dumps(_jsonable(record), indent=1), encoding="utf-8")
    log(f"FAILURE DIAGNOSTIC (saved to {fail_dir}):")
    for key, value in diag.items():
        log(f"    {key:22s} {value:.4f}")
    return fail_dir


def make_seed_run_dir(out_root: Path, config_name: str, seed: int, condition: str = "experimental") -> Path:
    name = make_run_name(GRID_SHAPE, ZONE_PRESET, PAIR_ROLE[condition], f"worldmodel-{config_name}", seed)
    return make_run_dir(out_root, EXPERIMENT_ID, name)


def build_rssm(cfg: dict) -> RSSM:
    image_side = 12 * cfg["render"]["k"]
    return RSSM(RSSMConfig(
        image_size=(image_side, image_side), base_depth=cfg["model"]["base_depth"], embed_dim=cfg["model"]["embed_dim"],
        h_dim=cfg["model"]["h_dim"], n_vars=cfg["model"]["n_vars"], n_classes=cfg["model"]["n_classes"],
        n_actions=N_ACTIONS, unimix=cfg["model"]["unimix"],
    ))


def world_record(world: World, cfg: dict) -> dict:
    return {"seed": world.seed, "condition": world.condition, **run_identity(world.seed, world.themes, world.condition),
            "zone_kinds": world.zone_kinds, "env_dynamics_seed": world.env_seed, "render": cfg["render"],
            "templates_sha256": world.templates_sha256(), "n_states": N_STATES, "n_actions": N_ACTIONS}


def load_run_data(run_dir: Path, cfg: dict, seed: int, condition: str) -> tuple[World, ReplayBuffer, np.ndarray, np.ndarray]:
    """The world, replay, N and episode starts of an existing run directory
    (resume, recompute): the world is rebuilt from the seed and condition
    with the SAVED themes, and must reproduce the saved hash of the
    pre-rendered images; the replay is rebuilt from replay.npz."""
    saved = json.loads((run_dir / "world.json").read_text(encoding="utf-8"))
    if (saved["seed"], saved["condition"]) != (seed, condition):
        raise ValueError(f"world.json is for seed {saved['seed']} / {saved['condition']}, not {seed} / {condition}")
    world = build_world(cfg, seed, condition, themes=themes_from_dict(saved["themes"]))
    if world.templates_sha256() != saved["templates_sha256"]:
        raise ValueError("the rebuilt pre-rendered images differ from the run's (templates_sha256): rendering changed since the run")
    data = np.load(run_dir / "replay.npz")
    replay = ReplayBuffer.from_compact({key: data[key] for key in ("s", "a", "r", "terminal", "truncated", "is_first")},
                                       world.templates_np)
    return world, replay, data["N"], data["episode_starts"]


def run_one_seed(cfg: dict, seed: int, run_dir: Path, device: torch.device, max_steps: int | None = None,
                 condition: str = "experimental", resume: bool = False, wandb: bool = False,
                 wandb_artifact: bool = False) -> dict:
    """Trains and measures one (seed, condition), writing everything into
    run_dir. resume=True continues an interrupted run in run_dir from its
    last resume point (checkpoints/resume.pt; from step 0 on the saved
    replay if there is none). Raises SeparationNotReachedError /
    CollapseAfterSwitchError after saving the failure diagnostic
    (run_dir/failure/). wandb / wandb_artifact: optional W&B tracking
    (wandb_logger.py; never stops the run)."""
    log = RunLog(run_dir / "console.log", f"[seed={seed} {condition}]")
    meta = environment_metadata(device)
    require_code_provenance(meta)  # neither a git commit nor a source hash: the run does not start
    log(f"code: git commit {meta['git_commit']}, source sha256 {meta['source_sha256']} ({meta['source_n_files']} files)")
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    k = cfg["render"]["k"]
    streams = seed_streams(seed, device, condition)
    collection = cfg["collection"]
    if collection["start_distribution"] not in START_DISTRIBUTIONS or collection["epsilon"] != 1.0:
        raise ValueError(f"collection must be start_distribution in {START_DISTRIBUTIONS} and epsilon 1.0 (no policy before "
                         f"Phase 5), got {collection}")
    budget, prefill = cfg["budget"]["total_steps"], cfg["budget"]["prefill_steps"]

    if resume:
        world, replay, N, episode_starts_arr = load_run_data(run_dir, cfg, seed, condition)
        episode_starts = episode_starts_arr.tolist()
        write_run_meta(run_dir, seed, status="running", resumed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       **{f"resume_{key}": value for key, value in meta.items()
                          if key in ("device", "gpu_name", "git_commit", "git_dirty", "source_sha256")})
        log(f"RESUMING in {run_dir} (device {device})")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        write_run_meta(run_dir, seed, status="running", start_distribution=collection["start_distribution"],
                       collection_epsilon=collection["epsilon"], **meta)
        world = build_world(cfg, seed, condition)
        write_run_meta(run_dir, seed, status="running", **run_identity(seed, world.themes, condition), zone_kinds=world.zone_kinds,
                       rng_streams=list(e3_rng.STREAM_NAMES), episode=load_episode_config())
        (run_dir / "world.json").write_text(json.dumps(world_record(world, cfg), indent=1), encoding="utf-8")
        replay = ReplayBuffer.for_run(budget=budget, prefill_steps=prefill, table=world.templates_np)
        log(f"collecting {prefill + budget} steps under epsilon={collection['epsilon']:g}, episode starts "
            f"{collection['start_distribution']} ...")
        t0 = time.perf_counter()
        episode_starts: list[int] = []
        next_states: list[int] = []
        N = collect_and_count(world.env, replay, n_steps=prefill + budget, rng=streams["collect"], n_states=N_STATES,
                              n_actions=N_ACTIONS, start_rng=streams["start_state"], starts_out=episode_starts,
                              next_states_out=next_states)
        np.savez_compressed(run_dir / "replay.npz", **replay.export_compact(), s_next=np.asarray(next_states, dtype=np.int16), N=N,
                            episode_starts=np.asarray(episode_starts, dtype=np.int64))
        log(f"collection done in {time.perf_counter() - t0:.1f}s; {len(episode_starts)} episode starts over "
            f"{len(set(episode_starts))} distinct states; compact replay saved")
    tracker = WandbLogger(log)
    saved_meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    wandb_id = tracker.start(cfg, run_dir, saved_meta, force=wandb, resume_id=saved_meta.get("wandb_run_id") if resume else None)
    if wandb_id is not None:
        write_run_meta(run_dir, seed, status="running", wandb_run_id=wandb_id)
    layout, themes, templates_t = world.layout, world.themes, world.templates_t.to(device)
    world.templates_t, world.background_t = templates_t, world.background_t.to(device)
    background_t = world.background_t
    if device.type == "cuda":
        replay.to_device(device)
    zone_class = e3_env.block_class_for_seed(seed)
    log(f"condition {condition}; class {zone_class}; device {device}; torch {torch.__version__}; threads {torch.get_num_threads()}; "
        f"TF32 matmul/cudnn {meta['allow_tf32_matmul']}/{meta['allow_tf32_cudnn']}; deterministic {meta['deterministic_algorithms']}; "
        f"run dir {run_dir}")
    T_hat_B = build_tabular_twin(N, N_STATES, N_ACTIONS)
    P_true = world.env.transition_tensor()

    torch.manual_seed(streams["network_init_seed"])  # initialisation only: built on the CPU, then moved (same weights on every device)
    rssm = compile_from_config(build_rssm(cfg).to(device), cfg)  # runtime.compile: off unless set (acceleration.py)
    rssm_cfg = rssm.cfg
    optimizer = torch.optim.Adam(rssm.parameters(), lr=cfg["train"]["lr"], eps=cfg["train"]["adam_eps"])
    loss_cfg = WorldModelLossConfig(
        k=k, w_patch=cfg["train"]["w_patch"], beta_dyn=cfg["train"]["beta_dyn"], beta_rep=cfg["train"]["beta_rep"],
        free_bits=cfg["train"]["free_bits"], free_bits_mode=cfg["train"]["free_bits_mode"],
    )

    planned_steps = int(round((prefill + budget) * cfg["train"]["replay_ratio"]))
    n_grad_steps = planned_steps if max_steps is None else min(planned_steps, max_steps)
    switch, n_pairs = build_separation_switch(cfg)
    rep_warmup_cfg = cfg["train"]["rep_warmup_steps"]
    monitor_every = cfg["measure"]["monitor_every"]
    separation_every = cfg["measure"]["separation_every"]
    diagnostics_every = cfg["measure"]["diagnostics_every"]
    ckpt_steps = checkpoint_steps(cfg, planned_steps)
    diag_at_ckpt = cfg["measure"].get("diagnostics_at_checkpoints", False)
    resume_every = (cfg.get("checkpoints") or {}).get("resume_every")
    lr, lr_after_switch = cfg["train"]["lr"], cfg["train"]["lr_after_switch"]
    _, sep_gen = measure_streams(seed, SLOT_SEPARATION, streams)
    diag_rng, diag_gen = measure_streams(seed, SLOT_DIAG, streams)

    history = {key: [] for key in ("step", *LOSS_TERMS, "grad_norm", "lr", "beta_dyn", "beta_rep")}
    separation_curve, diagnostics = [], []
    progress = {"step": 0, "rep_warmup_steps": None, "lr_switch_step": None, "last_stat": float("nan"), "training_seconds": 0.0}
    rng_owners = {"replay": streams["replay"], "switch_pairs": streams["switch_pairs"], "latent": streams["latent"],
                  "sep_gen": sep_gen, "diag_rng": diag_rng, "diag_gen": diag_gen}
    resume_path = run_dir / "checkpoints" / "resume.pt"

    def save_resume(step: int) -> None:
        flush()
        _atomic_torch_save({
            "step": step, "rssm": rssm.state_dict(), "optimizer": optimizer.state_dict(),
            "switch": {f.name: getattr(switch, f.name) for f in dataclasses.fields(switch)},
            "progress": dict(progress, step=step), "history": history, "separation_curve": separation_curve,
            "diagnostics": diagnostics, "rng": {name: _generator_state(g) for name, g in rng_owners.items()},
            "torch_rng": torch.get_rng_state(), "n_grad_steps": n_grad_steps, "planned_steps": planned_steps,
        }, resume_path)
        write_checkpoint_pointer(run_dir, resume_path.name, step)

    start_step = 0
    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["planned_steps"] != planned_steps:
            raise ValueError("resume.pt was written for another budget")
        rssm.load_state_dict(state["rssm"])
        optimizer.load_state_dict(state["optimizer"])
        for name, value in state["switch"].items():
            setattr(switch, name, value)
        progress.update(state["progress"])
        history, separation_curve, diagnostics = state["history"], state["separation_curve"], state["diagnostics"]
        for name, g in rng_owners.items():
            _set_generator_state(g, state["rng"][name])
        torch.set_rng_state(state["torch_rng"])
        start_step = state["step"]
        (run_dir / "diagnostics.jsonl").write_text("".join(json.dumps(_jsonable(r)) + "\n" for r in diagnostics), encoding="utf-8")
        log(f"resumed from step {start_step} (switch step {switch.switch_step})")
    elif resume:
        log("no resume.pt: training from step 0 on the saved replay")

    log(f"lr {lr:g} until the switch, then {lr_after_switch}; clip_norm {cfg['train']['clip_norm']}; collapse guard: "
        f"batch-pair JS < {switch.collapse_threshold_bits} bits on {switch.collapse_consecutive} consecutive evaluations after "
        f"the switch")
    log(f"training for {n_grad_steps} gradient steps (planned {planned_steps}): posterior probabilities until the batch-pair JS "
        f"exceeds {switch.threshold_bits} bits on {switch.consecutive} consecutive evaluations (every {switch.eval_every} steps, "
        f"from step {switch.min_steps}, deadline {switch.max_steps} absolute), then straight-through; beta_dyn={loss_cfg.beta_dyn} "
        f"from step 0, beta_rep 0 until the switch then ramped to {loss_cfg.beta_rep}; checkpoints at {ckpt_steps}; training "
        f"diagnostics every {diagnostics_every} steps{' and at the checkpoints' if diag_at_ckpt else ''}")

    pending: list[dict] = []

    def flush() -> None:
        """Deferred device synchronisation (audit 7.2): the per-step losses
        are kept as device tensors and converted in one transfer."""
        if not pending:
            return
        for key in (*LOSS_TERMS, "grad_norm"):
            history[key].extend(torch.stack([p[key] for p in pending]).cpu().tolist())
        pending.clear()

    def record_diagnostics(step: int) -> dict:
        gpu_peak_start(device)
        record, _ = training_diagnostics(rssm, world, P_true, T_hat_B, replay, cfg, diag_rng, diag_gen,
                                         cfg["measure"]["iz_mc_samples_training"], final=False)
        record = {"step": step, "switch_step": switch.switch_step, **record, "gpu_memory": gpu_peak_end(device)}
        diagnostics.append(record)
        with open(run_dir / "diagnostics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")
        save_training_diagnostics_figure(diagnostics, switch.switch_step, artifacts / f"training_diagnostics_seed{seed:02d}.png")
        tracker.log(step, record, prefix="diagnostics")
        rc = record["real_context"]
        log(f"diagnostics @ {step}: real contexts: KL det {[round(rc['kl_by_zone'][z], 2) for z in DETERMINISTIC_ZONES]} bits, I_s "
            f"{[round(rc['I_s_by_zone'][z]['mean'], 2) for z in e3_env.ZONE_CATEGORIES]} (I_z "
            f"{[round(rc['I_z_by_zone'][z]['mean'], 2) for z in e3_env.ZONE_CATEGORIES]}) | probes h/z -> s "
            f"{_nan(rc['state_probe_h']):.3f}/{_nan(rc['state_probe_z']):.3f} | acc replay/real/canonical-hard "
            f"{record['acc_replay_hard']:.3f}/{record['acc_real_context']:.3f}/{record['acc_canonical_hard']:.3f} | "
            f"{record['seconds']:.0f}s" + (f" | GPU peak {record['gpu_memory']['peak_reserved_mib']:.0f} MiB reserved"
                                           if record["gpu_memory"] else ""))
        return record

    def fail(err: Exception, steps_done: int) -> None:
        flush()
        save_failure_diagnostic(run_dir / "failure", cfg, seed, rssm, optimizer, templates_t, steps_done, switch, err, meta, log,
                                condition)
        write_run_meta(run_dir, seed, status="failed", error=str(err), steps_done=steps_done, switch_step=switch.switch_step,
                       switch_deadline=switch.max_steps)
        tracker.log(steps_done, {"failed": 1, "switch_step": switch.switch_step})
        tracker.finish({"status": "failed", "steps_done": steps_done}, run_dir, artifact=wandb_artifact)

    t0 = time.perf_counter() - progress["training_seconds"]
    for step in range(start_step, n_grad_steps):
        if set_learning_rate(optimizer, phase_learning_rate(step, switch.switch_step, lr, lr_after_switch)):
            progress["lr_switch_step"] = step
            log(f">>> LR changed to {optimizer.param_groups[0]['lr']:g} at step {step} (effective switch)")
        if not (resume and step == start_step):  # measured and saved before the interruption
            is_ckpt = step in ckpt_steps
            if step % separation_every == 0:
                separation_curve.append({"step": step, **e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=1,
                                                                                      generator=sep_gen)})
            if is_ckpt:
                save_weights(run_dir, rssm, step, switch_step=switch.switch_step)
            if (diagnostics_every and step % diagnostics_every == 0) or (diag_at_ckpt and is_ckpt):
                record_diagnostics(step)
            if step > 0 and (is_ckpt or (resume_every and step % resume_every == 0)):
                progress["training_seconds"] = time.perf_counter() - t0
                save_resume(step)
        model_batch, label_batch = replay.sample(cfg["train"]["batch_size"], cfg["train"]["seq_len"], streams["replay"])
        beta_rep = rep_beta(step, switch.switch_step, progress["rep_warmup_steps"] or 0, loss_cfg.beta_rep)
        step_cfg = dataclasses.replace(loss_cfg, beta_rep=beta_rep)
        phase = "straight-through" if switch.switched else "probabilities"
        eval_now = switch.is_eval_step(step)
        monitor_now = step % monitor_every == 0
        losses = train_world_model_step(
            rssm, optimizer, model_batch, background_t, step_cfg, clip_norm=cfg["train"]["clip_norm"],
            estimator=switch.estimator(), return_posteriors=eval_now or monitor_now, generator=streams["latent"], sync=False,
        )
        post_probs = losses.pop("post_probs", None)
        if eval_now:
            progress["last_stat"] = mean_random_pair_jensen_shannon(post_probs, streams["switch_pairs"], n_pairs)
            was_switched = switch.switched
            try:
                switch.update(step, progress["last_stat"])
            except CollapseAfterSwitchError as err:
                fail(err, step + 1)
                raise
            if switch.switched and not was_switched:
                progress["rep_warmup_steps"] = (rep_warmup_cfg if rep_warmup_cfg is not None
                                                else int(round(0.2 * (planned_steps - switch.switch_step))))
                log(f">>> SWITCH to straight-through effective at step {switch.switch_step} (batch-pair JS "
                    f"{progress['last_stat']:.3f} bits); deadline {switch.max_steps}, margin {switch.max_steps - switch.switch_step} "
                    f"steps; beta_rep ramps 0 -> {loss_cfg.beta_rep} over {progress['rep_warmup_steps']} steps")
                tracker.log(step, {"switch_step": switch.switch_step, "switch_margin": switch.max_steps - switch.switch_step})
        try:
            switch.check_deadline(step + 1)
        except SeparationNotReachedError as err:
            pending.append(losses)
            fail(err, step + 1)
            raise
        history["step"].append(step)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["beta_dyn"].append(loss_cfg.beta_dyn)
        history["beta_rep"].append(beta_rep)
        pending.append(losses)
        if eval_now or len(pending) >= 256:
            flush()
        if monitor_now:
            flush()
            # MONITORING ONLY: the labelled JS uses this batch's state labels, which reach
            # neither the training step nor the switch rule (the switch is driven by the
            # label-free batch-pair JS above, exclusively).
            labelled_js = labelled_between_state_js(post_probs, label_batch.s.reshape(-1))
            log(f"step {step:6d}/{n_grad_steps} | batch-pair JS {progress['last_stat']:6.3f} bits | labelled JS {labelled_js:6.3f} "
                f"bits | phase {phase:16s} | beta_rep {beta_rep:.4f} | recon_patch {history['recon_patch'][-1]:.4f} | "
                f"grad_norm {history['grad_norm'][-1]:.3g} | lr {optimizer.param_groups[0]['lr']:g} | "
                f"{time.perf_counter() - t0:.0f}s")
            tracker.log(step, {"train": {**{key: history[key][-1] for key in (*LOSS_TERMS, "grad_norm")},
                                         "lr": optimizer.param_groups[0]["lr"], "beta_rep": beta_rep,
                                         "phase_straight_through": int(switch.switched), "batch_pair_js": progress["last_stat"],
                                         "labelled_js": labelled_js, "seconds": time.perf_counter() - t0}})
    flush()
    training_done_now = start_step < n_grad_steps or not resume
    if training_done_now:
        separation_curve.append({"step": n_grad_steps, **e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=1,
                                                                                      generator=sep_gen)})
        progress["training_seconds"] = time.perf_counter() - t0
        save_weights(run_dir, rssm, n_grad_steps, name="weights_final.pt", switch_step=switch.switch_step)
        save_resume(n_grad_steps)
    log(f"training done in {progress['training_seconds']:.1f}s")

    fields, arrays = measure_final(rssm, world, replay, N, cfg, streams, run_dir, log, n_grad_steps)
    final = {**fields.pop("diagnostics_final"), "switch_step": switch.switch_step}
    diagnostics.append(final)
    with open(run_dir / "diagnostics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(final)) + "\n")
    save_training_diagnostics_figure(diagnostics, switch.switch_step, artifacts / f"training_diagnostics_seed{seed:02d}.png")

    hist = {key: np.asarray(history[key]) for key in LOSS_TERMS}
    loss_table = {key: {"start": float(v[:50].mean()), "end": float(v[-50:].mean())} for key, v in hist.items()}
    margin = None if switch.switch_step is None else switch.max_steps - switch.switch_step
    report = {
        "seed": seed,
        "condition": condition,
        "zone_class": zone_class,
        "run_dir": str(run_dir),
        "history": history,
        "loss_table": loss_table,
        "training_procedure": {
            "switch_rule": {"threshold_bits": switch.threshold_bits, "consecutive": switch.consecutive,
                            "eval_every": switch.eval_every, "min_steps": switch.min_steps, "max_steps": switch.max_steps,
                            "n_pairs": n_pairs},
            "switch_step": switch.switch_step,
            "switch_deadline": switch.max_steps,
            "switch_margin": margin,
            "collapse_guard": {"threshold_bits": switch.collapse_threshold_bits, "consecutive": switch.collapse_consecutive},
            "lr": lr, "lr_after_switch": lr_after_switch, "lr_switch_step": progress["lr_switch_step"],
            "clip_norm": cfg["train"]["clip_norm"],
            "rep_warmup_steps": progress["rep_warmup_steps"],
            "batch_pair_js": switch.history,
            "n_grad_steps": n_grad_steps,
            "planned_grad_steps": planned_steps,
            "checkpoint_steps": ckpt_steps,
            "training_seconds": progress["training_seconds"],
        },
        "separation_curve": separation_curve,
        "diagnostics": diagnostics,
        "reproducibility": meta,
        "loss_weights": {"recon": 1.0, "recon_patch": 0.0, "kl_dyn": loss_cfg.beta_dyn, "kl_rep": loss_cfg.beta_rep},
        "w_patch": loss_cfg.w_patch,
        "truncated_by_max_steps": n_grad_steps < planned_steps,
        **fields,
    }
    report["model"] = {"n_parameters": int(sum(p.numel() for p in rssm.parameters())), **dataclasses.asdict(rssm_cfg)}

    np.savez_compressed(run_dir / "arrays.npz", episode_starts=np.asarray(episode_starts, dtype=np.int64), **arrays)
    (run_dir / "results.json").write_text(json.dumps(_jsonable({k: v for k, v in report.items() if k not in ARRAY_FIELDS}),
                                                     indent=1), encoding="utf-8")
    write_report([report], run_dir, cfg_name=run_dir.name)
    write_run_meta(run_dir, seed, status="done", switch_step=switch.switch_step, switch_deadline=switch.max_steps,
                   switch_margin=margin, n_grad_steps=n_grad_steps, truncated_by_max_steps=n_grad_steps < planned_steps,
                   checkpoint_steps=ckpt_steps, training_seconds=round(progress["training_seconds"], 1))
    tracker.finish(json.loads((run_dir / "results.json").read_text(encoding="utf-8")) | {"history": None, "diagnostics": None,
                                                                                         "separation_curve": None},
                   run_dir, artifact=wandb_artifact)
    log(f"run done; results in {run_dir}")
    return report


# results.json leaves out the per-state arrays (they are in arrays.npz); load_run_report puts them back.
ARRAY_FIELDS = {"I_z": "I_z", "I_z_se": "I_z_se", "I_z_real": "real_I_z", "I_z_real_se": "real_I_z_se",
                "I_z_real_sd": "real_I_z_sd", "n_contexts": "real_n_contexts",
                "I_s_real": "real_I_s", "I_s_real_se": "real_I_s_se", "I_s_real_bias": "real_I_s_bias"}


def load_run_report(run_dir: Path) -> dict:
    """A finished run's report dict (results.json + the per-state arrays of
    arrays.npz), as run_one_seed returns it -- for combined reports."""
    report = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    arrays = np.load(run_dir / "arrays.npz")
    for field, key in ARRAY_FIELDS.items():
        if key in arrays:  # the I_s arrays exist from 2026-09-30 on
            report[field] = arrays[key]
    report["run_dir"] = str(run_dir)
    return report


# ---------------------------------------------------------------------
# figures and report
# ---------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_training_diagnostics_figure(records: list[dict], switch_step: int | None, path: Path) -> None:
    """Solid lines: real contexts (the main measure); thin dotted lines:
    canonical probe (secondary)."""
    plt = _plt()
    cats = list(e3_env.ZONE_CATEGORIES)
    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    colors = dict(zip(cats, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    rc = [r["real_context"] for r in records]

    def val(x):
        return float("nan") if x is None else x

    ax = axes[0, 0]
    for cat in cats:
        ax.plot(steps, [val(r["kl_by_zone"][cat]) for r in rc], marker="o", color=colors[cat], label=cat)
        ax.plot(steps, [r["kl_by_zone"][cat] for r in records], linestyle=":", linewidth=0.8, color=colors[cat])
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.set_title("KL(posterior || prior) by zone, bits (solid: real contexts, one-step;\ndotted: canonical, post(s) vs prior(s'))")
    ax = axes[0, 1]
    for cat in cats:
        ax.plot(steps, [val(r["l1_vs_p_by_zone"][cat]) for r in rc], marker="o", color=colors[cat], label=f"{cat}: WM vs P")
        ax.plot(steps, [r["l1_vs_p_by_zone"][cat] for r in records], linestyle=":", linewidth=0.8, color=colors[cat])
        ax.axhline(records[-1]["l1_b_vs_p_by_zone"][cat], color=colors[cat], linestyle="--", linewidth=1)
    ax.set_title("L1 of the s-space kernel vs P (dashed: L1(B, P), what the data allow)")
    ax = axes[0, 2]
    for cat in cats:
        if all("I_s_by_zone" in r for r in rc):
            ax.errorbar(steps, [val(r["I_s_by_zone"][cat]["mean"]) for r in rc],
                        yerr=[N_SE * val(r["I_s_by_zone"][cat]["se"]) for r in rc], marker="o", color=colors[cat], label=cat, capsize=2)
        ax.plot(steps, [val(r["I_z_by_zone"][cat]["mean"]) for r in rc], linestyle=":", linewidth=1, color=colors[cat])
    ax.set_title("real contexts, bits: I_s (MAIN, solid, +/- 3 se) and I_z (secondary, dotted)")
    ax = axes[1, 0]
    for zone in NOISY_ZONES:
        ax.plot(steps, [val(r["frequent_outcome"][zone]) for r in rc], marker="o", color=colors[zone], label=zone)
        ax.plot(steps, [r["frequent_outcome"][zone] for r in records], linestyle=":", linewidth=0.8, color=colors[zone])
    ax.set_title("p_hat(frequent outcome), noisy zones")
    ax = axes[1, 1]
    ax.plot(steps, [val(r["posterior_entropy"]) for r in rc], marker="o", color="black", label="real contexts")
    ax.plot(steps, [r["posterior_entropy_canonical"] for r in records], linestyle=":", color="black", label="canonical")
    ax.set_title("posterior entropy (bits)")
    ax = axes[1, 2]
    for key, label in ACC_LABELS.items():
        ax.plot(steps, [r[key] for r in records], marker="o", label=label)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("template accuracy")
    for ax in axes.flat:
        if switch_step is not None:
            ax.axvline(switch_step, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("gradient step (red dashed: switch)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _tag(report: dict) -> str:
    return f"seed={report['seed']} {report.get('condition', 'experimental')} (class {report['zone_class']})"


def _save_loss_curve_figure(seed_reports: list[dict], path: Path) -> None:
    plt = _plt()
    terms = [*LOSS_TERMS, "grad_norm"]
    fig, axes = plt.subplots(1, len(terms), figsize=(4.4 * len(terms), 4))
    for ax, term in zip(axes, terms):
        for report in seed_reports:
            ax.plot(report["history"]["step"], report["history"][term], label=_tag(report))
        ax.set_title(term)
        ax.set_xlabel("gradient step")
        if term == "grad_norm":
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_separation_figure(seed_reports: list[dict], path: Path) -> None:
    """Separation between states along training: canonical-probe
    hard-sample template accuracy, between-state JS and posterior entropy,
    plus -- on the JS panel -- the label-free batch-pair JS that drives
    the switch. Per run, the EFFECTIVE switch step (dashed), the end of
    the beta_rep ramp (dotted) and the switch deadline (dash-dot)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    panels = (("acc_hard", "template accuracy -- CANONICAL PROBE (fresh h0), HARD SAMPLE, 1 draw"),
              ("mean_js", "mean JS between states (bits)"), ("posterior_entropy", "mean posterior entropy (bits)"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (key, title) in zip(axes, panels):
        for i, report in enumerate(seed_reports):
            color = colors[i % len(colors)]
            tag = _tag(report)
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
            ax.axvline(proc["switch_deadline"], color=color, linestyle="-.", linewidth=0.8)
        ax.set_title(title + "\n(dashed: switch; dotted: end of beta_rep ramp; dash-dot: switch deadline)", fontsize=9)
        ax.set_xlabel("gradient step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_iz_by_zone_figure(seed_reports: list[dict], path: Path) -> None:
    """Bars: real-context I_s by zone (the main measure since 2026-09-30;
    I_z for reports written before); black ticks: real-context I_z
    (secondary, an upper bound)."""
    plt = _plt()
    categories = seed_reports[0]["zone_categories"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(seed_reports)
    x = np.arange(len(categories))
    for i, report in enumerate(seed_reports):
        main = report.get("I_s_by_category_real", report["I_by_category_real"])
        ax.bar(x + i * width, [_nan(main[c]) for c in categories], width,
               label=f"{_tag(report)}, {'I_s' if 'I_s_by_category_real' in report else 'I_z'} real contexts")
        ax.scatter(x + i * width, [_nan(report["I_by_category_real"][c]) for c in categories], marker="_", color="black", s=200,
                   zorder=3, label="I_z, real contexts (secondary)" if i == 0 else None)
    ax.set_xticks(x + width * (len(seed_reports) - 1) / 2)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("bits")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _fmt(x) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}" if np.isfinite(x) else "NaN"


def _zone_iz_stats(report: dict, cat: str) -> tuple[float, float, float]:
    """Canonical probe: (mean I_z over the zone's states, std over states, MC se of the mean)."""
    idx = [s for s in range(N_STATES) if report["zone_of_state"][s] == cat]
    I_z = np.asarray(report["I_z"])[idx]
    return float(I_z.mean()), float(I_z.std()), zone_mean_mc_se(np.asarray(report["I_z_se"])[idx])


def _zone_iz_stats_real(report: dict, cat: str) -> dict:
    """Real contexts: mean over the zone's measured states, std over them, se of the mean, n measured / n."""
    idx = [s for s in range(N_STATES) if report["zone_of_state"][s] == cat]
    values = np.asarray(report["I_z_real"], dtype=np.float64)
    out = zone_mean_with_se(values, np.asarray(report["I_z_real_se"], dtype=np.float64), idx)
    measured = values[idx][~np.isnan(values[idx])]
    out["sd"] = float(measured.std()) if measured.size else float("nan")
    return out


def _d1_table(add, d1: list[dict]) -> None:
    add("| own zone L | target zone L' | I_L' - I_L | threshold | accepted | local: mean [interval] (n kept / n) | "
        "global: mean [interval] (n kept / n) | anomalies (local; global) |")
    add("|---|---|---|---|---|---|---|---|")
    for d in d1:
        if d.get("not_measured"):
            add(f"| {d['own_category']} | {d['target_category']} | - | - | {NOT_MEASURED} | - | - | - |")
            continue
        if not d["accepted"]:
            add(f"| {d['own_category']} | {d['target_category']} | {_fmt(d['contrast'])} | {_fmt(d['threshold'])} | no | "
                f"undefined | undefined | - |")
            continue

        def cell(summary: dict) -> str:
            return (f"{_fmt(summary['mean'])} [{_fmt(summary['ci_low'])}, {_fmt(summary['ci_high'])}] "
                    f"({summary['n_kept']}/{summary['n_states']})")

        def anomalies(summary: dict) -> str:
            return ", ".join(f"s={a['s']}: {a['alpha']:.2f}+/-{a['se']:.2f}" for a in summary["anomalies"]) or "none"

        add(f"| {d['own_category']} | {d['target_category']} | {_fmt(d['contrast'])} | {_fmt(d['threshold'])} | yes | "
            f"{cell(d['local'])} | {cell(d['global'])} | {anomalies(d['local'])}; {anomalies(d['global'])} |")


def write_report(seed_reports: list[dict], out_dir: Path, cfg_name: str) -> Path:
    """REPORT.md in out_dir, figures in out_dir/artifacts/."""
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    _save_loss_curve_figure(seed_reports, artifacts / "loss_curves.png")
    _save_iz_by_zone_figure(seed_reports, artifacts / "iz_by_zone.png")
    _save_separation_figure(seed_reports, artifacts / "separation_curves.png")

    categories = seed_reports[0]["zone_categories"]
    lines: list[str] = []
    add = lines.append
    add(f"# E3 -- world-model run report ({cfg_name})")
    add("")
    add(f"Generated by `experiments/E3_worldmodel_4zones/run_mini.py`, runs {[_tag(r) for r in seed_reports]}. Run directories: "
        + ", ".join(f"`{Path(r['run_dir']).name}`" for r in seed_reports) + ".")
    add("")
    add("**Main measure: real contexts** (docs/e3/DECISIONS.md, 2026-09-28). For each state s, the contexts (h_t, z_t) of "
        "replay windows unrolled with their true history where s_t = s, excluding the first steps of each window (burn-in) "
        "and the is_first steps (the h0 regime); up to a fixed number of contexts per state, drawn from the measurement "
        "stream. **The main measure is I_s** (docs/e3/DECISIONS.md, 2026-09-30): I(a ; s' | s) of the s-space kernel -- "
        "prior draws from the state's contexts decoded to their arrival state and matched to the templates, pooled over "
        "the contexts of the state -- computed by measures.controllability, the same function as for the true P; its "
        "standard error and plug-in bias come from a parametric bootstrap. I_z (the action dependence of the predicted "
        "latent code, computed in every context with exact densities and averaged per state; the standard error of a "
        "state's mean is sd / sqrt(n) over its contexts) is secondary: an upper bound, inflated by action information the "
        "code carries beyond the arrival state. A state with no context is \"non mesuré\". KL(posterior || prior) is the "
        "one-step KL of the transition the data took from the context (the training loss's pairing). The canonical probe "
        "(h0) is reported as secondary.")
    add("")
    add(f"Verdict rules: every check uses a {N_SE:g}-standard-error margin; an ordering whose gap is within "
        f"{N_SE:g} combined standard errors is \"non concluant\", never OK. If the template accuracy "
        f"({ACC_LABELS['acc_replay_hard']}) is below {MECHANICS_FLOOR}, every indicator built on decoded images or on the "
        f"posterior's discrimination (s-space kernel, L1, T4 preview, ghosts, D1, D2) is reported as \"{NOT_EVALUABLE}\".")
    add("")
    add("## Loss curves (per term)")
    add("")
    add("![loss curves](artifacts/loss_curves.png)")
    add("")
    add("Reconstruction is summed over channels and pixels of each image and averaged over batch x time; KL terms "
        "are exact, summed over latent variables, in nats. `recon_patch` is diagnostic only (unweighted error on the "
        "masked patch, same aggregation), not part of the loss. `grad_norm`: global gradient norm, before any clipping.")
    add("")
    add("## Separation between states along training (canonical probe)")
    add("")
    add("![separation curves](artifacts/separation_curves.png)")
    add("")
    add("## Controllability by zone type (bars: I_s, real contexts, MAIN; ticks: I_z, real contexts, secondary)")
    add("")
    add("![I_s by zone](artifacts/iz_by_zone.png)")
    add("")

    for report in seed_reports:
        seed, cls, condition = report["seed"], report["zone_class"], report.get("condition", "experimental")
        kinds = report.get("zone_kinds") or {z: z for z in categories}
        acc = report["accuracies"]
        ok_mech = mechanics_ok(acc["acc_replay_hard"])
        proc = report["training_procedure"]
        rc = report["real_context"]
        add(f"## Seed {seed}, {condition} environment (class {cls})")
        add("")
        if condition != "experimental":
            add("Reference environment: every zone keeps its appearance (and its name below); its dynamics: "
                + ", ".join(f"{z} -> {kinds[z]}" for z in categories) + ".")
            add("")
        if report["truncated_by_max_steps"]:
            add(f"**Truncated by --max-steps: {proc['n_grad_steps']} of {proc['planned_grad_steps']} planned gradient steps.**")
            add("")
        add("### Training procedure")
        add("")
        add(f"Weights: recon x1 (patch weight w_patch={report['w_patch']:g} inside the mask); "
            f"kl_dyn x{report['loss_weights']['kl_dyn']:g} from step 0; kl_rep 0 until the switch, then ramped "
            f"linearly to x{report['loss_weights']['kl_rep']:g} over {proc['rep_warmup_steps']} steps; "
            f"recon_patch x0 (diagnostic). {proc['n_grad_steps']} gradient steps"
            + (f" in {proc['training_seconds'] / 3600:.2f} h" if proc.get("training_seconds") else "") + ".")
        add("")
        add(f"Posterior estimator: probabilities until the label-free batch-pair JS exceeded "
            f"{proc['switch_rule']['threshold_bits']} bits on {proc['switch_rule']['consecutive']} consecutive "
            f"evaluations (every {proc['switch_rule']['eval_every']} steps, from step {proc['switch_rule']['min_steps']}), then "
            f"straight-through. **Effective switch step: {proc['switch_step']}; deadline {proc['switch_deadline']} (absolute); "
            f"margin (deadline - switch step): {proc['switch_margin']} steps.** Learning rate {proc['lr']:g} until the switch, "
            f"then {proc['lr_after_switch']} (changed at step {proc['lr_switch_step']}); global gradient-norm clipping: "
            f"{'none' if proc['clip_norm'] is None else format(proc['clip_norm'], 'g')}; collapse guard after the switch: "
            f"batch-pair JS < {proc['collapse_guard']['threshold_bits']} bits on {proc['collapse_guard']['consecutive']} "
            f"consecutive evaluations. Weights saved at gradient steps {proc.get('checkpoint_steps', [])} and at the end.")
        repro = report["reproducibility"]
        add("")
        add(f"Environment: device `{repro['device']}`, torch {repro['torch_version']}, CUDA {repro['cuda_version']}, "
            f"{repro['num_threads']} threads, TF32 matmul/cuDNN {repro['allow_tf32_matmul']}/{repro['allow_tf32_cudnn']}, "
            f"deterministic algorithms {repro['deterministic_algorithms']}, commit "
            f"`{repro['git_commit']}`{' (working tree modified)' if repro['git_dirty'] else ''}, source sha256 "
            f"`{repro.get('source_sha256') or 'not recorded'}` -- bit-exact only on the same "
            f"machine, device and number of threads.")
        add("")
        add("| term | start (first 50 steps) | end (last 50 steps) |")
        add("|---|---|---|")
        for term in LOSS_TERMS:
            add(f"| {term} | {_fmt(report['loss_table'][term]['start'])} | {_fmt(report['loss_table'][term]['end'])} |")
        add("")

        if report["diagnostics"] and len(report["diagnostics"]) > 1:
            add("### Training diagnostics")
            add("")
            add(f"![training diagnostics, seed {seed}](artifacts/training_diagnostics_seed{seed:02d}.png)" if Path(report["run_dir"]) == out_dir
                else f"Figure: `artifacts/training_diagnostics_seed{seed:02d}.png` in the run directory.")
            add("")
            add("Raw values: `diagnostics.jsonl` in the run directory (one line per point).")
            add("")
            add("Expected shapes, fixed in advance (between the first diagnostic at or after the switch and the last one), "
                "**in real contexts** (main):")
            add("")
            add("| check | first | last | verdict |")
            add("|---|---|---|---|")
            for row in shape_verdicts(report["diagnostics"], proc["switch_step"], regime="real_context", zone_kinds=kinds):
                add(f"| {row['check']} | {_fmt(_nan(row.get('first')))} | {_fmt(_nan(row.get('last')))} | {row['verdict']} |")
            add("")
            add("Same checks from the canonical probe (secondary; its KL pairs the posterior of s with the prior of s', see below):")
            add("")
            add("| check | first | last | verdict |")
            add("|---|---|---|---|")
            for row in shape_verdicts(report["diagnostics"], proc["switch_step"], zone_kinds=kinds):
                add(f"| {row['check']} | {_fmt(row.get('first'))} | {_fmt(row.get('last'))} | {row['verdict']} |")
            add("")

        add("### Template accuracy (final)")
        add("")
        add("| what is decoded | accuracy |")
        add("|---|---|")
        for key, label in ACC_LABELS.items():
            add(f"| {label} | **{_fmt(_nan(acc.get(key)))}** |")
        add("")
        if not ok_mech:
            add(f"Replay accuracy below {MECHANICS_FLOOR}: mechanics failing, downstream indicators not evaluable.")
            add("")
        add(f"z-only linear probe (every replay posterior): train accuracy **{_fmt(report['z_only_probe']['train_accuracy'])}**, "
            f"test accuracy **{_fmt(report['z_only_probe']['test_accuracy'])}** (n_train={report['z_only_probe']['n_train']}, "
            f"n_test={report['z_only_probe']['n_test']})")
        add("")
        if report.get("h_only_probe"):
            add(f"h-only linear probe (every replay deterministic state h_t -> s_t, same windows): train accuracy "
                f"**{_fmt(report['h_only_probe']['train_accuracy'])}**, test accuracy "
                f"**{_fmt(report['h_only_probe']['test_accuracy'])}**. On the real contexts (after the burn-in): h -> s "
                f"**{_fmt(_nan(rc.get('state_probe_h')))}**, z -> s **{_fmt(_nan(rc.get('state_probe_z')))}** (test accuracy; the "
                "state information carried by h alone, docs/e3/DECISIONS.md 2026-09-30).")
            add("")

        add("### Real contexts: coverage")
        add("")
        cps = rc["contexts_per_state"]
        settings = rc["settings"]
        add(f"Windows of {settings['window_len']} steps, burn-in {settings['burn_in']} steps, up to {settings['max_per_state']} "
            f"contexts per state: **{rc['n_contexts_total']} contexts; {rc['n_states_measured']} / {N_STATES} states measured**; "
            f"contexts per state min {cps['min']}, median {cps['median']:g}, max {cps['max']}. States not measured: "
            f"{rc['not_measured_states'] or 'none'}.")
        add("")

        if "I_s_by_zone" in rc:
            add(f"### I_s in real contexts (MAIN; kernel of {rc['kernel_draws']} decoded draws per (s, a), pooled over the "
                "state's contexts; per zone: mean over measured states, se of the mean and mean plug-in bias from the bootstrap)")
            add("")
            if ok_mech:
                proj, off = rc.get("I_s_projected_by_zone"), rc.get("off_support_mass_by_zone")  # from 2026-09-30 (5) on
                add("| zone | dynamics here | mean I_s | se of the mean | plug-in bias (bootstrap) | states measured | "
                    "off-support mass | I_s projected on the support (SECONDARY) [se] |")
                add("|---|---|---|---|---|---|---|---|")
                for cat in categories:
                    st = rc["I_s_by_zone"][cat]
                    add(f"| {cat} | {kinds[cat]} | {_fmt(_nan(st['mean']))} | {_fmt(_nan(st['se']))} | {_fmt(_nan(st['bias']))} | "
                        f"{st['n_measured']}/{st['n_states']} | {_fmt(_nan(off[cat])) if off else '-'} | "
                        + (f"{_fmt(_nan(proj[cat]['mean']))} [{_fmt(_nan(proj[cat]['se']))}] |" if proj else "- |"))
                if proj:
                    add("")
                    add("I_s is computed on the kernel as it is: the model's actual belief, off-support mass included (the mass of "
                        "p_hat(s' | s, a) on states that are not one of the 5 reachable successors of s). The projected version "
                        "matches the same draws among the successors' templates only (same ghost rule) and renormalises; it uses "
                        "the environment's truth, so it is a secondary measure, for the comparison with the tabular twin whose prior "
                        f"has the same support (ghost rate of the projection: {_fmt(_nan(rc.get('ghost_fraction_projected_mean')))}).")
            else:
                add(f"{NOT_EVALUABLE}.")
            add("")
            pcc = report.get("is_per_context")
            if pcc and pcc["by_zone"] and ok_mech:
                st = pcc["settings"]
                add(f"I_s per context vs pooled (SECONDARY diagnostic, docs/e3/DECISIONS.md 2026-09-30): {st['states_per_zone']} states "
                    f"per zone, {st['contexts']} contexts each, {st['draws']} draws per context and action; bootstrap bias in brackets.")
                add("")
                add("| zone | per-context mean I_s | pooled I_s (same contexts) | gap pooled - per-context [se over states] |")
                add("|---|---|---|---|")
                for cat, z in pcc["by_zone"].items():
                    add(f"| {cat} | {_fmt(z['per_context'])} ({_fmt(z['per_context_bias'])}) | {_fmt(z['pooled'])} "
                        f"({_fmt(z['pooled_bias'])}) | {_fmt(z['gap'])} [{_fmt(_nan(z['gap_se_over_states']))}] |")
                add("")
        add("### I_z in real contexts (SECONDARY, upper bound; per zone: mean over measured states, std over states, se of the mean)")
        add("")
        add("| zone | dynamics here | mean I_z | std I_z | se of the mean | states measured |")
        add("|---|---|---|---|---|---|")
        for cat in categories:
            st = _zone_iz_stats_real(report, cat)
            add(f"| {cat} | {kinds[cat]} | {_fmt(st['mean'])} | {_fmt(st['sd'])} | {_fmt(st['se'])} | {st['n_measured']}/{st['n_states']} |")
        add("")
        add("### KL(posterior || prior) by zone, real contexts (MAIN; one-step KL of the transition taken, mean over contexts, bits)")
        add("")
        add("| zone | mean KL (bits) |")
        add("|---|---|")
        for cat in categories:
            add(f"| {cat} | {_fmt(_nan(rc['kl_by_zone'][cat]))} |")
        add("")
        add("### s-space kernel from real contexts (MAIN): L1 vs P and vs B; L1(B, P) = what the data allow")
        add("")
        if ok_mech:
            add("| zone | L1(WM, P) | L1 floor (exact kernel, same draws) | L1(WM, B) | L1(B, P) | p_hat(frequent outcome) |")
            add("|---|---|---|---|---|---|")
            for cat in categories:
                fo = rc["frequent_outcome"].get(cat)
                floor = rc.get("l1_floor_by_zone", {}).get(cat)  # absent from reports written before 2026-09-29
                add(f"| {cat} | {_fmt(_nan(rc['l1_vs_p_by_zone'][cat]))} | {'-' if floor is None else _fmt(_nan(floor))} | "
                    f"{_fmt(_nan(rc['l1_vs_b_by_zone'][cat]))} | "
                    f"{_fmt(_nan(rc['l1_b_vs_p_by_zone'][cat]))} | {_fmt(_nan(fo)) if cat in NOISY_ZONES else '-'} |")
            add("")
            add(f"Ghost rate (mean over measured (s, a) pairs): **{_fmt(_nan(rc['ghost_fraction_mean']))}**")
        else:
            add(f"{NOT_EVALUABLE}.")
        add("")
        add("### D1 in real contexts (MAIN: alpha_app, local and global)")
        add("")
        if ok_mech:
            add(f"Contrast accepted only if |I_L' - I_L| > max(0.25 bit, {N_SE:g} x combined se), zone means over measured states. "
                f"alpha_app per measured state of the own zone (I_swap: mean over the state's contexts with x_t replaced by the "
                f"counterfactual render, history kept) with its standard error; a value outside [0, 1] by more than {N_SE:g} se is "
                f"an ANOMALY, listed and never averaged; the others: mean over states +/- {N_SE:g} sd/sqrt(n).")
            add("")
            _d1_table(add, report["d1"])
        else:
            add(f"{NOT_EVALUABLE}.")
        add("")
        add("### D2 in real contexts (MAIN: neighbour JS distance, boundary vs within-zone)")
        add("")
        if ok_mech:
            add(f"- boundary (different category) mean JS: **{_fmt(_nan(report['d2_boundary_mean']))}**")
            add(f"- within-zone (same category) mean JS: **{_fmt(_nan(report['d2_within_zone_mean']))}**")
        else:
            add(f"{NOT_EVALUABLE}.")
        add("")
        add("### Context dispersion (spec 5.5): real contexts vs canonical probe")
        add("")
        add("| zone | mean over states of abs(I_real(s) - I_canonical(s)) | mean sd of I_z over a state's contexts |")
        add("|---|---|---|")
        for cat in categories:
            d = report["context_dispersion"][cat]
            add(f"| {cat} | {_fmt(_nan(d['mean_abs_gap_real_vs_canonical']))} | {_fmt(_nan(d['mean_sd_over_contexts']))} |")
        add("")

        add("### Canonical probe (SECONDARY: fresh h0)")
        add("")
        add(f"- I(z; s) over the canonical posteriors (uniform over 144 states): **{_fmt(report['I_zs'])} bits** "
            f"(MC se {_fmt(report['I_zs_se'])}; ceiling log2(144) = {np.log2(144):.4f}; ~0 = collapse)")
        add(f"- Mean posterior entropy, canonical probe: **{_fmt(report['canonical_posterior_entropy_mean'])} bits**; "
            f"replay (every posterior): **{_fmt(report['z_only_probe']['mean_posterior_entropy'])} bits** "
            f"(ceiling n_vars x log2(n_classes) = {report['posterior_entropy_ceiling']:.1f})")
        add("")
        add(f"![collapse grid, seed {seed}](artifacts/{report['collapse_grid']})" if Path(report["run_dir"]) == out_dir else
            f"Collapse grid: `artifacts/{report['collapse_grid']}` in the run directory.")
        add("")
        add("| zone | mean I_z (canonical) | std I_z | MC se of the mean | KL post(s) vs prior(s') (bits) | L1(WM, P) | L1(WM, B) |")
        add("|---|---|---|---|---|---|---|")
        for cat in categories:
            m, sd, se = _zone_iz_stats(report, cat)
            add(f"| {cat} | {_fmt(m)} | {_fmt(sd)} | {_fmt(se)} | {_fmt(report['kl_by_zone'][cat])} | "
                f"{_fmt(report['l1_vs_p_by_zone'][cat])} | {_fmt(report['l1_vs_b_by_zone'][cat])} |")
        add("")
        add("The canonical KL compares the posterior of the current state s (at h0) with the prior predicted for the next "
            "state s' after each action: two different states whenever s' != s, so it is large even for a perfect model "
            "(docs/e3/DECISIONS.md, 2026-09-28). It is not a measure of the fit of the prior.")
        add("")
        add(f"Canonical p_hat(frequent outcome): {', '.join(f'{z} {_fmt(v)}' for z, v in report['frequent_outcome_preview'].items())}; "
            f"ghost rate {_fmt(report['ghost_fraction_mean'])}; D2 boundary / within-zone "
            f"{_fmt(report['d2_canonical_boundary_mean'])} / {_fmt(report['d2_canonical_within_zone_mean'])}.")
        add("")
        add("D1 from the canonical probe:")
        add("")
        _d1_table(add, report["d1_canonical"])
        add("")

    add('## Expected-shape checklist (spec: "formes attendues seulement"), real contexts')
    add("")
    add("| # | check | run | expected | observed | verdict |")
    add("|---|---|---|---|---|---|")
    row = 1
    for report in seed_reports:
        tag = _tag(report)
        kinds = report.get("zone_kinds") or {z: z for z in categories}
        acc_replay = report["accuracies"]["acc_replay_hard"]
        ok_mech = mechanics_ok(acc_replay)
        is_zones = report["real_context"].get("I_s_by_zone")
        if is_zones is not None:  # the main measure since 2026-09-30
            if kinds.get("uncontrollable_deterministic") == "uncontrollable_deterministic":
                conv, ctrl = is_zones["uncontrollable_deterministic"], is_zones["controllable_deterministic"]
                conv_se, ctrl_se = _nan(conv["se"]), _nan(ctrl["se"])
                verdict = ordering_verdict(_nan(conv["mean"]), conv_se, _nan(ctrl["mean"]), ctrl_se)
                add(f"| {row} | I_s(conveyor) < I_s(controllable-det) (MAIN) | {tag} | gap > {N_SE:g} x combined se | "
                    f"{_fmt(_nan(conv['mean']))} vs {_fmt(_nan(ctrl['mean']))} (combined se "
                    f"{_fmt(float(np.sqrt(conv_se ** 2 + ctrl_se ** 2)))}) | {verdict} |")
            else:
                add(f"| {row} | I_s(conveyor) < I_s(controllable-det) (MAIN) | {tag} | - | no conveyor in this environment | "
                    "non applicable |")
            row += 1
            b = bounds_verdict(report["I_s_real"], report["I_s_real_se"], ceiling=float(np.log2(5)))
            add(f"| {row} | 0 <= I_s <= log2(5), per measured state within {N_SE:g} se (MAIN) | {tag} | 0 violations | "
                f"{b['n_violations']} violation(s) / {b['n_measured']} measured | {'OK' if b['ok'] else 'FAIL'} |")
            row += 1
        if kinds.get("uncontrollable_deterministic") == "uncontrollable_deterministic":
            conv = _zone_iz_stats_real(report, "uncontrollable_deterministic")
            ctrl = _zone_iz_stats_real(report, "controllable_deterministic")
            verdict = ordering_verdict(conv["mean"], conv["se"], ctrl["mean"], ctrl["se"])
            combined = float(np.sqrt(conv["se"] ** 2 + ctrl["se"] ** 2))
            add(f"| {row} | I_z(conveyor) < I_z(controllable-det) | {tag} | gap > {N_SE:g} x combined se | "
                f"{_fmt(conv['mean'])} vs {_fmt(ctrl['mean'])} (gap {_fmt(ctrl['mean'] - conv['mean'])}, combined se "
                f"{_fmt(combined)}) | {verdict} |")
        else:
            add(f"| {row} | I_z(conveyor) < I_z(controllable-det) | {tag} | - | no conveyor in this environment | non applicable |")
        row += 1

        b = bounds_verdict(report["I_z_real"], report["I_z_real_se"], ceiling=float(np.log2(5)))
        add(f"| {row} | 0 <= I_z <= log2(5), per measured state within {N_SE:g} se | {tag} | 0 violations | "
            f"{b['n_violations']} violation(s) / {b['n_measured']} measured | {'OK' if b['ok'] else 'FAIL'} |")
        row += 1

        add(f"| {row} | template accuracy ({ACC_LABELS['acc_replay_hard']}) high | {tag} | >= {MECHANICS_FLOOR} (mini; "
            f"calibration's C-a is 0.99) | {_fmt(acc_replay)} | {'OK' if ok_mech else 'FAIL'} |")
        row += 1

        accepted = [d for d in report["d1"] if d["accepted"]]
        n_anomalies = sum(d[kind]["n_anomalies"] for d in accepted for kind in ("local", "global"))
        add(f"| {row} | alpha_app defined only on accepted contrasts; anomalies listed | {tag} | contrast > max(0.25, "
            f"{N_SE:g} se) | {len(accepted)}/{len(report['d1'])} pairs accepted, {n_anomalies} anomalous state value(s) | "
            f"{'OK' if ok_mech else NOT_EVALUABLE} |")
        row += 1

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3 world-model run (see the module docstring).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (default: configs/e3/mini.yaml)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="overrides runtime.device")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="results root (default: results/)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="overrides the config's seeds")
    parser.add_argument("--conditions", nargs="+", choices=e3_rng.CONDITIONS, default=None, help="overrides the config's conditions")
    parser.add_argument("--max-steps", type=int, default=None, help="caps the number of gradient steps (smoke tests)")
    parser.add_argument("--resume", type=Path, default=None, help="continue the interrupted run in this directory")
    parser.add_argument("--combine", type=Path, nargs="+", default=None, help="write a combined report of finished runs")
    parser.add_argument("--wandb", action="store_true", help="force W&B tracking on (default: on if WANDB_API_KEY is set)")
    parser.add_argument("--wandb-artifact", action="store_true",
                        help="upload the run directory (weights included, resume.pt excluded) as a W&B artifact at the end")
    return parser.parse_args(argv)


def _resume(args: argparse.Namespace) -> int:
    run_dir = args.resume
    cfg = load_mini_config(run_dir / "config.yaml")
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = configure_runtime(cfg)
    print(f"[run_mini] resume {run_dir} | device {device}", flush=True)
    try:
        run_one_seed(cfg, int(meta["seed"]), run_dir, device, max_steps=cfg["runtime"].get("max_steps"),
                     condition=meta.get("condition", "experimental"), resume=True, wandb=args.wandb,
                     wandb_artifact=args.wandb_artifact)
    except (SeparationNotReachedError, CollapseAfterSwitchError) as err:
        print(f"[run_mini] run STOPPED by a training guard ({type(err).__name__}); see {run_dir / 'failure'}", flush=True)
        return 1
    except BaseException as err:
        write_run_meta(run_dir, int(meta["seed"]), status="failed", error=f"{type(err).__name__}: {err}")
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts (GPU determinism policy, B7)
    if args.combine:
        reports = [load_run_report(Path(d)) for d in args.combine]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        combined = make_run_dir(args.out_dir, EXPERIMENT_ID, f"report__combined__{stamp}")
        print(f"[run_mini] combined report: {write_report(reports, combined, cfg_name='combined')}", flush=True)
        return 0
    if args.resume:
        return _resume(args)
    cfg = load_mini_config(args.config)
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.conditions is not None:
        cfg["conditions"] = args.conditions
    cfg.setdefault("conditions", ["experimental"])
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.max_steps is not None:
        cfg["runtime"]["max_steps"] = args.max_steps
    device = configure_runtime(cfg)
    config_name = Path(args.config).stem
    print(f"[run_mini] config {args.config} | device {device} | seeds {cfg['seeds']} | conditions {cfg['conditions']} | "
          f"out-dir {args.out_dir}", flush=True)

    reports, failures = [], []
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            run_dir = make_seed_run_dir(args.out_dir, config_name, seed, condition)
            try:
                reports.append(run_one_seed(cfg, seed, run_dir, device, max_steps=cfg["runtime"].get("max_steps"), condition=condition,
                                            wandb=args.wandb, wandb_artifact=args.wandb_artifact))
            except (SeparationNotReachedError, CollapseAfterSwitchError) as err:
                failures.append((seed, condition, run_dir, err))
                print(f"[run_mini] seed {seed} {condition} STOPPED by a training guard ({type(err).__name__}); checkpoint and "
                      f"diagnostic in {run_dir / 'failure'}", flush=True)
            except BaseException as err:
                write_run_meta(run_dir, seed, status="failed", error=f"{type(err).__name__}: {err}")
                raise
    if len(reports) > 1:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = "-".join(f"{r['seed']:02d}{r['condition'][0]}" for r in reports)
        combined = make_run_dir(args.out_dir, EXPERIMENT_ID, f"report__{config_name}__{tag}__{stamp}")
        path = write_report(reports, combined, cfg_name=config_name)
        print(f"[run_mini] combined report: {path}", flush=True)
    for seed, condition, run_dir, err in failures:
        print(f"[run_mini] FAILED seed {seed} {condition}: {err}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
