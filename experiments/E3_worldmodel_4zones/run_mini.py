"""E3 world-model run on one configuration (configs/e3/mini.yaml by default;
configs/e3/mini_long.yaml for the long diagnostic run). For each seed:
collects fixed epsilon=1 data (plus N[s,a,s'] counts for the tabular-twin
comparison), trains the RSSM with the retained procedure (docs/e3/DECISIONS.md),
optionally logs a set of training diagnostics every
`measure.diagnostics_every` steps, then runs every Phase-3
measure/diagnostic (canonical probe, I_z, kernel-in-s-space, z-only probe,
context dispersion, D1/D2) and writes a report.

Outputs (docs/e3/DECISIONS.md, 2026-09-25) -- nothing is written to docs/:
one directory per seed, never overwritten (helplessness.runs.make_run_dir):
    <out-dir>/E3_worldmodel_4zones/grid12x12__4zones__pair-experimental__worldmodel-<config>__seedNN__<UTC>/
        config.yaml         the resolved config (after CLI overrides)
        meta.yaml           status, seed, commit + dirty flag, python/torch/CUDA/cuDNN/GPU, device,
                            TF32 and determinism flags, threads
        console.log         everything printed to the console
        diagnostics.jsonl   training diagnostics, one JSON line per point, appended as they are computed
        results.json        scalars and curves of the seed (written at the end of the seed)
        arrays.npz          N, T_hat_B, P, p_hat, I_z, posteriors/priors of the canonical probe, ...
        REPORT.md           the seed's report; figures in artifacts/
        failure/            checkpoint.pt + diagnostic.json, only if a training guard fired
With several seeds, a combined REPORT.md goes to
<out-dir>/E3_worldmodel_4zones/report__<config>__seeds...__<UTC>/.

Usage, from the repo root (Linux / macOS):
    python experiments/E3_worldmodel_4zones/run_mini.py --config configs/e3/mini.yaml --device auto
PowerShell:
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\run_mini.py --config configs\\e3\\mini.yaml --device auto
Options: --config, --device {auto,cpu,cuda}, --out-dir (default: results/), --seeds, --max-steps
(caps the number of gradient steps, for smoke tests). Exit code 1 if a seed stopped on a training guard.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
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
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, categorical_kl, resolve_device
from helplessness.envs.appearance.theme_sampling import build_themes
from helplessness.envs.observation.allocentric_pixel_backend import render, render_background
from helplessness.envs.zones import geographic_support_mask
from helplessness.measures.diagnostics import (
    alpha_app_with_se,
    d1_contrast,
    jensen_shannon_categorical,
    mean_random_pair_jensen_shannon,
    summarize_alpha_app,
)
from helplessness.measures.latent_controllability import latent_state_information
from helplessness.runs import make_run_dir, make_run_name, write_run_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e3" / "mini.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results"
EXPERIMENT_ID = "E3_worldmodel_4zones"
ZONE_PRESET = "4zones"
GRID_SHAPE = (12, 12)
N_STATES, N_ACTIONS = 144, 5

NOISY_ZONES = ("controllable_noisy", "uncontrollable_noisy")
DETERMINISTIC_ZONES = ("controllable_deterministic", "uncontrollable_deterministic")
LOSS_TERMS = ("total", "recon", "recon_patch", "kl_dyn", "kl_rep")
N_SE = 3.0  # every MC-based verdict in the report uses a 3-standard-error margin
MECHANICS_FLOOR = 0.5  # below this template-match accuracy, downstream indicators are not evaluable
NOT_EVALUABLE = "non évaluable (mécanique en échec)"
ACC_LABELS = {
    "acc_replay_hard": "replay windows (real contexts h_t), hard sample",
    "acc_canonical_hard": "canonical probe (fresh h0), hard sample, mean of 5 draws",
    "acc_canonical_mode": "canonical probe (fresh h0), posterior mode",
}


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


def shape_verdicts(records: list[dict], switch_step: int | None, n_se: float = N_SE) -> list[dict]:
    """The expected shapes of the long diagnostic run, fixed in advance
    (docs/e3/DECISIONS.md, 2026-09-25), judged between the FIRST training
    diagnostic at or after the effective switch and the LAST one:
      1. KL(posterior || prior) in each deterministic zone decreases and
         ends below 1 bit;
      2. L1(world model, P) decreases in each zone (its distance to
         L1(B, P), what the data allow, is reported);
      3. I_z of the conveyor (uncontrollable_deterministic) decreases;
      4. I_z of the controllable-deterministic zone increases.
    3 and 4 use the MC standard error of the zone means: a change within
    n_se combined se is "non concluant". Returns one row per check."""
    post = [r for r in records if switch_step is not None and r["step"] >= switch_step]
    if len(post) < 2:
        return [{"check": "all", "verdict": "non évaluable (moins de 2 points de diagnostic après la bascule)"}]
    first, last = post[0], post[-1]
    rows = []
    for zone in DETERMINISTIC_ZONES:
        a, b = first["kl_by_zone"][zone], last["kl_by_zone"][zone]
        ok = b < a and b < 1.0
        rows.append({"check": f"KL decreasing to < 1 bit ({zone})", "first": a, "last": b,
                     "verdict": "conforme" if ok else "non conforme"})
    for zone in e3_env.ZONE_CATEGORIES:
        a, b = first["l1_vs_p_by_zone"][zone], last["l1_vs_p_by_zone"][zone]
        ref = last["l1_b_vs_p_by_zone"][zone]
        rows.append({"check": f"L1(WM, P) decreasing toward L1(B, P) = {ref:.3f} ({zone})", "first": a, "last": b,
                     "verdict": "conforme" if b < a else "non conforme"})
    for zone, direction in (("uncontrollable_deterministic", -1), ("controllable_deterministic", +1)):
        a, b = first["I_z_by_zone"][zone], last["I_z_by_zone"][zone]
        se = float(np.sqrt(a["mc_se"] ** 2 + b["mc_se"] ** 2))
        change = b["mean"] - a["mean"]
        if abs(change) <= n_se * se:
            verdict = "non concluant"
        else:
            verdict = "conforme" if np.sign(change) == direction else "non conforme"
        rows.append({"check": f"I_z {'decreasing' if direction < 0 else 'increasing'} ({zone})", "first": a["mean"],
                     "last": b["mean"], "verdict": verdict})
    for row in rows:
        row["first_step"], row["last_step"] = first["step"], last["step"]
    return rows


# ---------------------------------------------------------------------
# config, runtime, metadata, logging
# ---------------------------------------------------------------------


def load_mini_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path is not None else DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_runtime(cfg: dict) -> torch.device:
    """Threads, TF32 and device, set once at startup and recorded in
    meta.yaml. TF32 is disabled explicitly for matmul AND cuDNN
    convolutions (A100 default: on for convolutions): the probabilities
    phase separates the states on tiny probability differences, which a
    reduced-precision convolution could change (docs/e3/DECISIONS.md,
    2026-09-25)."""
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return resolve_device(cfg["runtime"]["device"])


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD") or "unknown"
        changed = [line for line in run("status", "--porcelain").splitlines() if line.strip()]
    except OSError:
        return {"git_commit": "unknown", "git_dirty": None, "git_n_changed_paths": None}
    return {"git_commit": commit, "git_dirty": bool(changed), "git_n_changed_paths": len(changed)}


def environment_metadata(device: torch.device) -> dict:
    """Everything needed to tell two runs' environments apart. Bit-exact
    reproducibility holds only on the same machine, with the same number of
    threads and the same device (docs/e3/DECISIONS.md)."""
    cuda = device.type == "cuda"
    return {
        "device": str(device),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),  # TorchVersion is a str subclass yaml.safe_dump rejects
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "num_threads": torch.get_num_threads(),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        **git_state(),
    }


class RunLog:
    """print() that also appends to the run's console.log."""

    def __init__(self, path: Path | None, prefix: str):
        self.path, self.prefix = path, prefix

    def __call__(self, message: str) -> None:
        line = f"{self.prefix} {message}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def seed_streams(seed: int, device: torch.device) -> dict:
    """Separate random streams (docs/e3/DECISIONS.md, 2026-09-25): changing
    how often anything is MEASURED never changes the training trajectory.
    Training consumes only "collect" (environment actions), "replay"
    (minibatch windows), "switch_pairs" (the label-free switch statistic,
    evaluated on a fixed schedule) and "latent" (every latent draw of the
    training step); the network initialisation uses the global torch RNG,
    seeded just before the model is built. Every measurement draws from its
    own streams, created where it is used (measure_streams)."""

    def torch_gen(key: int) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(np.random.SeedSequence([seed, key]).generate_state(1)[0]))
        return gen

    return {
        "collect": np.random.default_rng(seed),  # same actions (hence same data) as the runs before 2026-09-25
        "replay": np.random.default_rng([seed, 1]),
        "switch_pairs": np.random.default_rng([seed, 4]),
        "latent": torch_gen(2),
        "torch_gen": torch_gen,
    }


def measure_streams(seed: int, key: int, streams: dict) -> tuple[np.random.Generator, torch.Generator]:
    """A (numpy, torch) pair of streams owned by one measurement."""
    return np.random.default_rng([seed, 100 + key]), streams["torch_gen"](100 + key)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------
# environment, data, twin
# ---------------------------------------------------------------------


def zone_of_state(layout, s: int) -> str:
    r, c = divmod(s, layout.grid_shape[1])
    return layout.zone_at((r, c)).appearance.category


def zone_states(layout, category: str) -> list[int]:
    return [s for s in range(N_STATES) if zone_of_state(layout, s) == category]


def labelled_between_state_js(post_probs: np.ndarray, labels: np.ndarray) -> float:
    """Console monitoring only -- never an input to the switch or to
    training. Mean JS (bits, summed over latent variables) between the
    posteriors of every pair of batch samples whose true states DIFFER;
    post_probs (n, n_vars, n_classes) and labels (n,) in the same order.
    NaN if every sample shows the same state."""
    pair_js = jensen_shannon_categorical(post_probs[:, None], post_probs[None, :]).sum(axis=-1)
    different = labels[:, None] != labels[None, :]
    return float(pair_js[different].mean()) if different.any() else float("nan")


def build_separation_switch(cfg: dict) -> tuple[SeparationSwitch, int]:
    """The retained estimator procedure from the config (docs/e3/DECISIONS.md,
    2026-09-24): posterior probabilities until the label-free batch-pair JS
    criterion fires, then straight-through; after the switch, the same
    statistic guards against a collapse (CollapseAfterSwitchError).
    switch_max_steps is an ABSOLUTE number of gradient steps (2026-09-25:
    never a fraction of the budget)."""
    est = cfg["train"]["latent_estimator"]
    if est["kind"] != "warmup_probs_until_separated":
        raise ValueError(f"run_mini.py implements the retained procedure only; got latent_estimator.kind={est['kind']!r}")
    if not isinstance(est["switch_max_steps"], int):
        raise ValueError(f"train.latent_estimator.switch_max_steps must be an absolute number of gradient steps, got "
                         f"{est['switch_max_steps']!r} (docs/e3/DECISIONS.md, 2026-09-25)")
    switch = SeparationSwitch(
        max_steps=est["switch_max_steps"], threshold_bits=est["switch_js_threshold"], consecutive=est["switch_consecutive"],
        eval_every=est["switch_eval_every"], min_steps=est["switch_min_steps"],
        collapse_threshold_bits=est["collapse_js_threshold"], collapse_consecutive=est["collapse_consecutive"],
    )
    return switch, est["switch_n_pairs"]


def build_run(cfg: dict, seed: int):
    """Layout, themes, environment, background and the 144 templates of a
    seed. Tensors are returned on the CPU; the caller moves them."""
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
    templates_np = np.stack([render(s, layout, themes, k=k) for s in range(N_STATES)])
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
    works in nats; converted here."""
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
    probability (the "frequent outcome"). Not the full T4 statistical
    test (that is Phase 4/E3-0 scope) -- a shape preview only."""
    out = {}
    for zone in NOISY_ZONES:
        max_probs = p_hat[zone_states(layout, zone)].max(axis=-1)  # (n_states_in_zone, n_actions)
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


def training_diagnostics(
    rssm: RSSM, templates_t: torch.Tensor, layout, P_true: np.ndarray, T_hat_B: np.ndarray, replay: ReplayBuffer, cfg: dict,
    rng: np.random.Generator, generator: torch.Generator,
) -> tuple[dict, dict]:
    """One point of the training diagnostics (docs/e3/DECISIONS.md,
    2026-09-25), also used for the final measurement. Returns a JSON-able
    record and the full arrays it was computed from (reused by the final
    report). All draws come from (rng, generator), owned by the caller."""
    m = cfg["measure"]
    was_training = rssm.training
    rssm.eval()
    t0 = time.perf_counter()
    iz = e3_probe.canonical_iz(rssm, templates_t, M=m["iz_mc_samples"], rng=rng, generator=generator)
    probe = iz["probe"]
    kernel = e3_probe.canonical_kernel_probe(rssm, templates_t, K_s=m["kernel_k_s"], margin_threshold=m["kernel_margin_threshold"],
                                             probe=probe, generator=generator)
    separation = e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=5, generator=generator)
    batch, labels = replay.sample(32, 16, rng)
    acc_replay = e3_probe.template_match_accuracy(rssm, batch, labels, templates_t, margin_threshold=m["kernel_margin_threshold"],
                                                  generator=generator)
    rssm.train(was_training)

    l1_vs_p = np.abs(kernel["p_hat"] - P_true).sum(axis=-1)  # (144, 5)
    l1_vs_b = np.abs(kernel["p_hat"] - T_hat_B).sum(axis=-1)
    l1_b_vs_p = np.abs(T_hat_B - P_true).sum(axis=-1)
    cats = list(e3_env.ZONE_CATEGORIES)

    def by_zone(arr: np.ndarray) -> dict[str, float]:
        return {cat: float(np.nanmean(arr[zone_states(layout, cat)])) for cat in cats}

    record = {
        "kl_by_zone": kl_post_prior_by_zone(probe, layout),
        "l1_vs_p_by_zone": by_zone(l1_vs_p),
        "l1_vs_b_by_zone": by_zone(l1_vs_b),
        "l1_b_vs_p_by_zone": by_zone(l1_b_vs_p),
        "I_z_by_zone": {cat: {"mean": float(iz["I_z"][zone_states(layout, cat)].mean()),
                              "mc_se": zone_mean_mc_se(iz["se"][zone_states(layout, cat)])} for cat in cats},
        "frequent_outcome": frequent_outcome_preview(kernel["p_hat"], layout),
        "posterior_entropy_canonical": float(probe["posterior_entropy"].mean()),
        "acc_replay_hard": acc_replay,
        "acc_canonical_hard": separation["acc_hard"],
        "acc_canonical_mode": separation["acc_mode"],
        "ghost_fraction_mean": float(np.nanmean(kernel["ghost_fraction"])),
        "seconds": time.perf_counter() - t0,
    }
    return record, {"iz": iz, "probe": probe, "kernel": kernel, "l1_vs_p": l1_vs_p, "l1_vs_b": l1_vs_b, "l1_b_vs_p": l1_b_vs_p}


def d1_analysis(rssm: RSSM, layout, themes, I_z: np.ndarray, I_z_se: np.ndarray, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """D1 (spec section 8, criteria of docs/e3/DECISIONS.md 2026-09-25): for
    every ordered pair of zones (own L, target L'), the contrast is accepted
    only if |I_L' - I_L| > max(d1_contrast_min_bits, 3 x combined MC se).
    For an accepted pair, alpha_app is computed for EVERY state of L, local
    (the agent's cell re-themed) and global (the two zones' themes swapped
    on the whole map), with its standard error; per-state values outside
    [0, 1] beyond 3 se are anomalies (listed, never averaged); the others
    are summarised by their mean over the zone's states +/- 3 sd/sqrt(n)."""
    m = cfg["measure"]
    cats = list(e3_env.ZONE_CATEGORIES)
    stats = {cat: (float(I_z[zone_states(layout, cat)].mean()), zone_mean_mc_se(I_z_se[zone_states(layout, cat)])) for cat in cats}
    I_by_category = {cat: v[0] for cat, v in stats.items()}
    results = []
    for own in cats:
        for target in cats:
            if own == target:
                continue
            (I_L, se_L), (I_Lp, se_Lp) = stats[own], stats[target]
            entry = {"own_category": own, "target_category": target, "I_L": I_L, "se_L": se_L, "I_Lprime": I_Lp,
                     "se_Lprime": se_Lp, **d1_contrast(I_L, se_L, I_Lp, se_Lp, min_bits=m["d1_contrast_min_bits"], n_se=N_SE)}
            if entry["accepted"]:
                states = zone_states(layout, own)
                for kind, global_swap in (("local", False), ("global", True)):
                    swaps = [e3_probe.d1_alpha_app_for_state(rssm, layout, themes, s, target, I_by_category, M=m["iz_mc_samples"],
                                                             rng=rng, k=cfg["render"]["k"], contrast_threshold=0.0,
                                                             global_swap=global_swap) for s in states]
                    alpha, se = alpha_app_with_se(np.array([d["I_swap"] for d in swaps]), np.array([d["I_swap_se"] for d in swaps]),
                                                  I_L, se_L, I_Lp, se_Lp)
                    summary = summarize_alpha_app(alpha, se, n_se=N_SE)
                    for anomaly in summary["anomalies"]:
                        anomaly["s"] = states[anomaly.pop("index")]
                    summary["per_state"] = [{"s": s, "alpha": float(a), "se": float(e)} for s, a, e in zip(states, alpha, se)]
                    entry[kind] = summary
            results.append(entry)
    return results


# ---------------------------------------------------------------------
# one seed
# ---------------------------------------------------------------------


def save_failure_diagnostic(fail_dir: Path, cfg: dict, seed: int, rssm: RSSM, optimizer: torch.optim.Optimizer,
                            templates_t: torch.Tensor, step: int, switch: SeparationSwitch,
                            error: SeparationNotReachedError | CollapseAfterSwitchError, meta: dict, log: RunLog) -> Path:
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
    diag = e3_probe.latent_usage_diagnostic(rssm, templates_t, M=cfg["measure"]["iz_mc_samples"],
                                            rng=np.random.default_rng([seed, 3000]))
    record = {"seed": seed, "steps_done": step, "kind": kind, "error": str(error), **meta, "config": cfg,
              "switch_history": [list(map(float, h)) for h in switch.history], "diagnostic": diag}
    (fail_dir / "diagnostic.json").write_text(json.dumps(_jsonable(record), indent=1), encoding="utf-8")
    log(f"FAILURE DIAGNOSTIC (saved to {fail_dir}):")
    for key, value in diag.items():
        log(f"    {key:22s} {value:.4f}")
    return fail_dir


def make_seed_run_dir(out_root: Path, config_name: str, seed: int) -> Path:
    name = make_run_name(GRID_SHAPE, ZONE_PRESET, "experimental", f"worldmodel-{config_name}", seed)
    return make_run_dir(out_root, EXPERIMENT_ID, name)


def run_one_seed(cfg: dict, seed: int, run_dir: Path, device: torch.device, max_steps: int | None = None) -> dict:
    """Trains and measures one seed, writing everything into run_dir.
    Raises SeparationNotReachedError / CollapseAfterSwitchError after
    saving the failure diagnostic (run_dir/failure/)."""
    log = RunLog(run_dir / "console.log", f"[seed={seed}]")
    meta = environment_metadata(device)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_run_meta(run_dir, seed, status="running", **meta)
    artifacts = run_dir / "artifacts"
    k = cfg["render"]["k"]
    streams = seed_streams(seed, device)
    layout, themes, env, background_t, templates_t = build_run(cfg, seed)
    background_t, templates_t = background_t.to(device), templates_t.to(device)
    zone_class = e3_env.block_class_for_seed(seed)
    log(f"class {zone_class}; device {device}; torch {torch.__version__}; threads {torch.get_num_threads()}; "
        f"TF32 matmul/cudnn {meta['allow_tf32_matmul']}/{meta['allow_tf32_cudnn']}; run dir {run_dir}")

    image_side = 12 * k
    budget, prefill = cfg["budget"]["total_steps"], cfg["budget"]["prefill_steps"]
    replay = ReplayBuffer.for_run(budget=budget, prefill_steps=prefill, image_shape=(image_side, image_side, 3))
    log(f"collecting {prefill + budget} steps under epsilon=1 ...")
    t0 = time.perf_counter()
    N = collect_and_count(env, replay, n_steps=prefill + budget, rng=streams["collect"], n_states=N_STATES, n_actions=N_ACTIONS)
    log(f"collection done in {time.perf_counter() - t0:.1f}s")
    T_hat_B = build_tabular_twin(N, N_STATES, N_ACTIONS)
    P_true = env.transition_tensor()

    rssm_cfg = RSSMConfig(
        image_size=(image_side, image_side), base_depth=cfg["model"]["base_depth"], embed_dim=cfg["model"]["embed_dim"],
        h_dim=cfg["model"]["h_dim"], n_vars=cfg["model"]["n_vars"], n_classes=cfg["model"]["n_classes"],
        n_actions=N_ACTIONS, unimix=cfg["model"]["unimix"],
    )
    torch.manual_seed(seed)  # network initialisation only: built on the CPU, then moved (same weights on every device)
    rssm = RSSM(rssm_cfg).to(device)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=cfg["train"]["lr"], eps=cfg["train"]["adam_eps"])
    loss_cfg = WorldModelLossConfig(
        k=k, w_patch=cfg["train"]["w_patch"], beta_dyn=cfg["train"]["beta_dyn"], beta_rep=cfg["train"]["beta_rep"],
        free_bits=cfg["train"]["free_bits"], free_bits_mode=cfg["train"]["free_bits_mode"],
    )

    planned_steps = (prefill + budget) * cfg["train"]["replay_ratio"]
    n_grad_steps = planned_steps if max_steps is None else min(planned_steps, max_steps)
    switch, n_pairs = build_separation_switch(cfg)
    rep_warmup_cfg = cfg["train"]["rep_warmup_steps"]
    rep_warmup_steps: int | None = None  # resolved at the switch (default: 20% of the planned budget left after it)
    monitor_every = cfg["measure"]["monitor_every"]
    separation_every = cfg["measure"]["separation_every"]
    diagnostics_every = cfg["measure"]["diagnostics_every"]
    lr, lr_after_switch = cfg["train"]["lr"], cfg["train"]["lr_after_switch"]
    _, sep_gen = measure_streams(seed, 1, streams)
    diag_rng, diag_gen = measure_streams(seed, 2, streams)
    log(f"lr {lr:g} until the switch, then {lr_after_switch}; clip_norm {cfg['train']['clip_norm']}; collapse guard: "
        f"batch-pair JS < {switch.collapse_threshold_bits} bits on {switch.collapse_consecutive} consecutive evaluations after "
        f"the switch")
    log(f"training for {n_grad_steps} gradient steps (planned {planned_steps}): posterior probabilities until the batch-pair JS "
        f"exceeds {switch.threshold_bits} bits on {switch.consecutive} consecutive evaluations (every {switch.eval_every} steps, "
        f"from step {switch.min_steps}, deadline {switch.max_steps} absolute), then straight-through; beta_dyn={loss_cfg.beta_dyn} "
        f"from step 0, beta_rep 0 until the switch then ramped to {loss_cfg.beta_rep}; training diagnostics every "
        f"{diagnostics_every} steps")

    history = {key: [] for key in ("step", *LOSS_TERMS, "grad_norm", "lr", "beta_dyn", "beta_rep")}
    separation_curve, diagnostics = [], []
    lr_switch_step = None
    last_stat = float("nan")

    def record_diagnostics(step: int) -> tuple[dict, dict]:
        record, extras = training_diagnostics(rssm, templates_t, layout, P_true, T_hat_B, replay, cfg, diag_rng, diag_gen)
        record = {"step": step, "switch_step": switch.switch_step, **record}
        diagnostics.append(record)
        with open(run_dir / "diagnostics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")
        save_training_diagnostics_figure(diagnostics, switch.switch_step, artifacts / f"training_diagnostics_seed{seed:02d}.png")
        log(f"diagnostics @ {step}: KL det {[round(record['kl_by_zone'][z], 2) for z in DETERMINISTIC_ZONES]} bits | "
            f"acc replay/canonical-hard/canonical-mode {record['acc_replay_hard']:.3f}/{record['acc_canonical_hard']:.3f}/"
            f"{record['acc_canonical_mode']:.3f} | {record['seconds']:.0f}s")
        return record, extras

    def fail(err: Exception, steps_done: int) -> None:
        save_failure_diagnostic(run_dir / "failure", cfg, seed, rssm, optimizer, templates_t, steps_done, switch, err, meta, log)
        write_run_meta(run_dir, seed, status="failed", error=str(err), steps_done=steps_done, switch_step=switch.switch_step,
                       switch_deadline=switch.max_steps)

    t0 = time.perf_counter()
    for step in range(n_grad_steps):
        if set_learning_rate(optimizer, phase_learning_rate(step, switch.switch_step, lr, lr_after_switch)):
            lr_switch_step = step
            log(f">>> LR changed to {optimizer.param_groups[0]['lr']:g} at step {step} (effective switch)")
        if step % separation_every == 0:
            separation_curve.append({"step": step, **e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=1, generator=sep_gen)})
        if diagnostics_every and step % diagnostics_every == 0:
            record_diagnostics(step)
        model_batch, label_batch = replay.sample(cfg["train"]["batch_size"], cfg["train"]["seq_len"], streams["replay"])
        beta_rep = rep_beta(step, switch.switch_step, rep_warmup_steps or 0, loss_cfg.beta_rep)
        step_cfg = dataclasses.replace(loss_cfg, beta_rep=beta_rep)
        phase = "straight-through" if switch.switched else "probabilities"
        eval_now = switch.is_eval_step(step)
        monitor_now = step % monitor_every == 0
        losses = train_world_model_step(
            rssm, optimizer, model_batch, background_t, step_cfg, clip_norm=cfg["train"]["clip_norm"],
            estimator=switch.estimator(), return_posteriors=eval_now or monitor_now, generator=streams["latent"],
        )
        post_probs = losses.pop("post_probs", None)
        if eval_now:
            last_stat = mean_random_pair_jensen_shannon(post_probs, streams["switch_pairs"], n_pairs)
            was_switched = switch.switched
            try:
                switch.update(step, last_stat)
            except CollapseAfterSwitchError as err:
                fail(err, step + 1)
                raise
            if switch.switched and not was_switched:
                rep_warmup_steps = rep_warmup_cfg if rep_warmup_cfg is not None else int(round(0.2 * (planned_steps - switch.switch_step)))
                log(f">>> SWITCH to straight-through effective at step {switch.switch_step} (batch-pair JS {last_stat:.3f} bits); "
                    f"deadline {switch.max_steps}, margin {switch.max_steps - switch.switch_step} steps; beta_rep ramps "
                    f"0 -> {loss_cfg.beta_rep} over {rep_warmup_steps} steps")
        try:
            switch.check_deadline(step + 1)
        except SeparationNotReachedError as err:
            fail(err, step + 1)
            raise
        history["step"].append(step)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["beta_dyn"].append(loss_cfg.beta_dyn)
        history["beta_rep"].append(beta_rep)
        for key in (*LOSS_TERMS, "grad_norm"):
            history[key].append(losses[key])
        if monitor_now:
            # MONITORING ONLY: the labelled JS uses this batch's state labels, which reach
            # neither the training step nor the switch rule (the switch is driven by the
            # label-free batch-pair JS above, exclusively).
            labelled_js = labelled_between_state_js(post_probs, label_batch.s.reshape(-1))
            log(f"step {step:6d}/{n_grad_steps} | batch-pair JS {last_stat:6.3f} bits | labelled JS {labelled_js:6.3f} bits | "
                f"phase {phase:16s} | beta_rep {beta_rep:.4f} | recon_patch {losses['recon_patch']:.4f} | "
                f"grad_norm {losses['grad_norm']:.3g} | lr {optimizer.param_groups[0]['lr']:g}")
    separation_curve.append({"step": n_grad_steps, **e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=1, generator=sep_gen)})
    log(f"training done in {time.perf_counter() - t0:.1f}s")

    rssm.eval()
    log("final measurements (canonical probe, I_z, s-space kernel, template accuracies) ...")
    final, extras = record_diagnostics(n_grad_steps)
    iz_result, probe, kernel_result = extras["iz"], extras["probe"], extras["kernel"]

    log("collapse diagnostics (I(z;s), posterior entropy, grid) ...")
    rng_izs, _ = measure_streams(seed, 3, streams)
    izs = latent_state_information(probe["posterior_probs"], M=cfg["measure"]["iz_mc_samples"], rng=rng_izs)
    zone_categories = list(e3_env.ZONE_CATEGORIES)
    grid_states = [s for cat in zone_categories for s in zone_states(layout, cat)[:2]]
    grid_name = f"collapse_grid_seed{seed:02d}.png"
    save_collapse_grid(rssm, templates_t, background_t, k, grid_states, artifacts / grid_name)

    log("z-only probe (every replay posterior) ...")
    rng_z, _ = measure_streams(seed, 4, streams)
    z_batch, z_labels = replay.chronological_windows(seq_len=64)
    z_probe_result = e3_probe.z_only_probe(rssm, z_batch, z_labels, n_states=N_STATES, test_frac=cfg["measure"]["z_only_test_frac"],
                                           rng=rng_z)

    log("context dispersion ...")
    rng_ctx, _ = measure_streams(seed, 5, streams)
    ctx_batch, ctx_labels = replay.sample(16, 32, rng_ctx)
    ctx_result = e3_probe.context_dispersion(rssm, ctx_batch, ctx_labels, iz_result["I_z"], M=cfg["measure"]["iz_mc_samples"],
                                             rng=rng_ctx, n_samples=cfg["measure"]["context_dispersion_n_samples"])

    log("D1 (every state of every zone with an accepted contrast) / D2 ...")
    rng_d1, _ = measure_streams(seed, 6, streams)
    d1_results = d1_analysis(rssm, layout, themes, iz_result["I_z"], iz_result["se"], cfg, rng_d1)
    d2_result = e3_probe.d2_neighbor_distances(probe["posterior_probs"], layout)

    hist = {key: np.asarray(history[key]) for key in LOSS_TERMS}
    loss_table = {key: {"start": float(v[:50].mean()), "end": float(v[-50:].mean())} for key, v in hist.items()}
    margin = None if switch.switch_step is None else switch.max_steps - switch.switch_step
    report = {
        "seed": seed,
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
            "lr": lr, "lr_after_switch": lr_after_switch, "lr_switch_step": lr_switch_step,
            "clip_norm": cfg["train"]["clip_norm"],
            "rep_warmup_steps": rep_warmup_steps,
            "batch_pair_js": switch.history,
            "n_grad_steps": n_grad_steps,
            "planned_grad_steps": planned_steps,
        },
        "separation_curve": separation_curve,
        "diagnostics": diagnostics,
        "reproducibility": meta,
        "loss_weights": {"recon": 1.0, "recon_patch": 0.0, "kl_dyn": loss_cfg.beta_dyn, "kl_rep": loss_cfg.beta_rep},
        "w_patch": loss_cfg.w_patch,
        "I_zs": izs["I_zs"],
        "I_zs_se": izs["se"],
        "canonical_posterior_entropy_mean": float(probe["posterior_entropy"].mean()),
        "posterior_entropy_ceiling": float(rssm_cfg.n_vars * np.log2(rssm_cfg.n_classes)),
        "collapse_grid": grid_name,
        "I_z": iz_result["I_z"],
        "I_z_se": iz_result["se"],
        "zone_of_state": [zone_of_state(layout, s) for s in range(N_STATES)],
        "I_by_category": {cat: float(iz_result["I_z"][zone_states(layout, cat)].mean()) for cat in zone_categories},
        "accuracies": {key: final[key] for key in ACC_LABELS},
        "ghost_fraction_mean": final["ghost_fraction_mean"],
        "z_only_probe": z_probe_result,
        "context_dispersion_mean_abs": float(ctx_result["abs_discrepancy"].mean()),
        "d1": d1_results,
        "d2_boundary_mean": float(d2_result["boundary"].mean()),
        "d2_within_zone_mean": float(d2_result["within_zone"].mean()),
        "kl_by_zone": final["kl_by_zone"],
        "l1_vs_p_by_zone": final["l1_vs_p_by_zone"],
        "l1_vs_b_by_zone": final["l1_vs_b_by_zone"],
        "l1_b_vs_p_by_zone": final["l1_b_vs_p_by_zone"],
        "frequent_outcome_preview": final["frequent_outcome"],
        "zone_categories": zone_categories,
        "truncated_by_max_steps": n_grad_steps < planned_steps,
    }

    np.savez_compressed(
        run_dir / "arrays.npz", N=N, T_hat_B=T_hat_B, P_true=P_true, p_hat=kernel_result["p_hat"],
        ghost_fraction=kernel_result["ghost_fraction"], I_z=iz_result["I_z"], I_z_se=iz_result["se"],
        posterior_probs=probe["posterior_probs"], prior_probs=probe["prior_probs"], l1_vs_p=extras["l1_vs_p"],
        l1_vs_b=extras["l1_vs_b"], l1_b_vs_p=extras["l1_b_vs_p"],
    )
    (run_dir / "results.json").write_text(json.dumps(_jsonable({k: v for k, v in report.items() if k not in ("I_z", "I_z_se")}),
                                                     indent=1), encoding="utf-8")
    write_report([report], run_dir, cfg_name=run_dir.name)
    write_run_meta(run_dir, seed, status="done", switch_step=switch.switch_step, switch_deadline=switch.max_steps,
                   switch_margin=margin, n_grad_steps=n_grad_steps, truncated_by_max_steps=n_grad_steps < planned_steps)
    log(f"seed done; results in {run_dir}")
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
    plt = _plt()
    cats = list(e3_env.ZONE_CATEGORIES)
    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    colors = dict(zip(cats, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    ax = axes[0, 0]
    for cat in cats:
        ax.plot(steps, [r["kl_by_zone"][cat] for r in records], marker="o", color=colors[cat], label=cat)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.set_title("KL(posterior || prior) by zone (bits; dotted: 1 bit)")
    ax = axes[0, 1]
    for cat in cats:
        ax.plot(steps, [r["l1_vs_p_by_zone"][cat] for r in records], marker="o", color=colors[cat], label=f"{cat}: WM vs P")
        ax.axhline(records[-1]["l1_b_vs_p_by_zone"][cat], color=colors[cat], linestyle="--", linewidth=1)
    ax.set_title("L1 of the s-space kernel vs P (dashed: L1(B, P), what the data allow)")
    ax = axes[0, 2]
    for cat in cats:
        ax.errorbar(steps, [r["I_z_by_zone"][cat]["mean"] for r in records],
                    yerr=[N_SE * r["I_z_by_zone"][cat]["mc_se"] for r in records], marker="o", color=colors[cat], label=cat, capsize=2)
    ax.set_title("canonical I_z by zone (bits, +/- 3 MC se)")
    ax = axes[1, 0]
    for zone in NOISY_ZONES:
        ax.plot(steps, [r["frequent_outcome"][zone] for r in records], marker="o", color=colors[zone], label=zone)
    ax.set_title("p_hat(frequent outcome), noisy zones")
    ax = axes[1, 1]
    ax.plot(steps, [r["posterior_entropy_canonical"] for r in records], marker="o", color="black")
    ax.set_title("canonical posterior entropy (bits)")
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


def _save_loss_curve_figure(seed_reports: list[dict], path: Path) -> None:
    plt = _plt()
    terms = [*LOSS_TERMS, "grad_norm"]
    fig, axes = plt.subplots(1, len(terms), figsize=(4.4 * len(terms), 4))
    for ax, term in zip(axes, terms):
        for report in seed_reports:
            ax.plot(report["history"]["step"], report["history"][term], label=f"seed={report['seed']} (class {report['zone_class']})")
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
    the switch. Per seed, the EFFECTIVE switch step (dashed), the end of
    the beta_rep ramp (dotted) and the switch deadline (dash-dot)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    panels = (("acc_hard", "template accuracy -- CANONICAL PROBE (fresh h0), HARD SAMPLE, 1 draw"),
              ("mean_js", "mean JS between states (bits)"), ("posterior_entropy", "mean posterior entropy (bits)"))
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
            ax.axvline(proc["switch_deadline"], color=color, linestyle="-.", linewidth=0.8)
        ax.set_title(title + "\n(dashed: switch; dotted: end of beta_rep ramp; dash-dot: switch deadline)", fontsize=9)
        ax.set_xlabel("gradient step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_iz_by_zone_figure(seed_reports: list[dict], path: Path) -> None:
    plt = _plt()
    categories = seed_reports[0]["zone_categories"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(seed_reports)
    x = np.arange(len(categories))
    for i, report in enumerate(seed_reports):
        values = [report["I_by_category"][c] for c in categories]
        ax.bar(x + i * width, values, width, label=f"seed={report['seed']} (class {report['zone_class']})")
    ax.set_xticks(x + width * (len(seed_reports) - 1) / 2)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("mean canonical I_z (bits)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _fmt(x) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}" if np.isfinite(x) else "NaN"


def _zone_iz_stats(report: dict, cat: str) -> tuple[float, float, float]:
    """(mean I_z over the zone's states, std over states, MC se of the mean)."""
    idx = [s for s in range(N_STATES) if report["zone_of_state"][s] == cat]
    I_z = np.asarray(report["I_z"])[idx]
    return float(I_z.mean()), float(I_z.std()), zone_mean_mc_se(np.asarray(report["I_z_se"])[idx])


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
    add(f"Generated by `experiments/E3_worldmodel_4zones/run_mini.py`, seeds {[r['seed'] for r in seed_reports]} "
        f"(classes {[r['zone_class'] for r in seed_reports]}). Run directories: "
        + ", ".join(f"`{Path(r['run_dir']).name}`" for r in seed_reports) + ".")
    add("")
    add(f"Verdict rules: MC-based checks use a {N_SE:g}-standard-error margin; an ordering whose gap is within "
        f"{N_SE:g} combined standard errors is \"non concluant\", never OK. If the template accuracy "
        f"({ACC_LABELS['acc_replay_hard']}) is below {MECHANICS_FLOOR}, every indicator built on decoded images or on the "
        f"posterior's discrimination (s-space kernel, L1, T4 preview, ghosts, D1, D2) is reported as \"{NOT_EVALUABLE}\".")
    add("")
    add("**Template accuracy -- what each number measures.** Nearest-template matching of a DECODED image against the "
        "144 renders. *Replay*: windows of the replay buffer unrolled with real contexts h_t, z_t a hard straight-through "
        "sample of the posterior (the calibration criterion C-a). *Canonical probe*: every state's render encoded from "
        "the learned initial state h0 (fresh-episode context), z a hard sample (mean of 5 draws) or the posterior mode. "
        "h0 is only ever seen at the start of a replay window, so the canonical numbers are expected to be lower.")
    add("")
    add("## Loss curves (per term)")
    add("")
    add("![loss curves](artifacts/loss_curves.png)")
    add("")
    add("Reconstruction is summed over channels and pixels of each image and averaged over batch x time; KL terms "
        "are exact, summed over latent variables, in nats. `recon_patch` is diagnostic only (unweighted error on the "
        "masked patch, same aggregation), not part of the loss. `grad_norm`: global gradient norm, before any clipping.")
    add("")
    add("## Separation between states along training")
    add("")
    add("![separation curves](artifacts/separation_curves.png)")
    add("")
    add("## Canonical I_z by zone type")
    add("")
    add("![I_z by zone](artifacts/iz_by_zone.png)")
    add("")

    for report in seed_reports:
        seed, cls = report["seed"], report["zone_class"]
        acc = report["accuracies"]
        ok_mech = mechanics_ok(acc["acc_replay_hard"])
        proc = report["training_procedure"]
        add(f"## Seed {seed} (class {cls})")
        add("")
        if report["truncated_by_max_steps"]:
            add(f"**Truncated by --max-steps: {proc['n_grad_steps']} of {proc['planned_grad_steps']} planned gradient steps.**")
            add("")
        add("### Training procedure")
        add("")
        add(f"Weights: recon x1 (patch weight w_patch={report['w_patch']:g} inside the mask); "
            f"kl_dyn x{report['loss_weights']['kl_dyn']:g} from step 0; kl_rep 0 until the switch, then ramped "
            f"linearly to x{report['loss_weights']['kl_rep']:g} over {proc['rep_warmup_steps']} steps; "
            f"recon_patch x0 (diagnostic). {proc['n_grad_steps']} gradient steps.")
        add("")
        add(f"Posterior estimator: probabilities until the label-free batch-pair JS exceeded "
            f"{proc['switch_rule']['threshold_bits']} bits on {proc['switch_rule']['consecutive']} consecutive "
            f"evaluations (every {proc['switch_rule']['eval_every']} steps, from step {proc['switch_rule']['min_steps']}), then "
            f"straight-through. **Effective switch step: {proc['switch_step']}; deadline {proc['switch_deadline']} (absolute); "
            f"margin (deadline - switch step): {proc['switch_margin']} steps.** Learning rate {proc['lr']:g} until the switch, "
            f"then {proc['lr_after_switch']} (changed at step {proc['lr_switch_step']}); global gradient-norm clipping: "
            f"{'none' if proc['clip_norm'] is None else format(proc['clip_norm'], 'g')}; collapse guard after the switch: "
            f"batch-pair JS < {proc['collapse_guard']['threshold_bits']} bits on {proc['collapse_guard']['consecutive']} "
            f"consecutive evaluations.")
        repro = report["reproducibility"]
        add("")
        add(f"Environment: device `{repro['device']}`, torch {repro['torch_version']}, CUDA {repro['cuda_version']}, "
            f"{repro['num_threads']} threads, TF32 matmul/cuDNN {repro['allow_tf32_matmul']}/{repro['allow_tf32_cudnn']}, commit "
            f"`{repro['git_commit']}`{' (working tree modified)' if repro['git_dirty'] else ''} -- bit-exact only on the same "
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
            add(f"![training diagnostics, seed {seed}](artifacts/training_diagnostics_seed{seed:02d}.png)")
            add("")
            add("Raw values: `diagnostics.jsonl` in the seed's run directory (one line per point).")
            add("")
            add("Expected shapes, fixed in advance (between the first diagnostic at or after the switch and the last one):")
            add("")
            add("| check | first | last | verdict |")
            add("|---|---|---|---|")
            for row in shape_verdicts(report["diagnostics"], proc["switch_step"]):
                add(f"| {row['check']} | {_fmt(row.get('first'))} | {_fmt(row.get('last'))} | {row['verdict']} |")
            add("")

        add("### Template accuracy (final)")
        add("")
        add("| what is decoded | accuracy |")
        add("|---|---|")
        for key, label in ACC_LABELS.items():
            add(f"| {label} | **{_fmt(acc[key])}** |")
        add("")
        if not ok_mech:
            add(f"Replay accuracy below {MECHANICS_FLOOR}: mechanics failing, downstream indicators not evaluable.")
            add("")
        add(f"z-only linear probe (every replay posterior): train accuracy **{_fmt(report['z_only_probe']['train_accuracy'])}**, "
            f"test accuracy **{_fmt(report['z_only_probe']['test_accuracy'])}** (n_train={report['z_only_probe']['n_train']}, "
            f"n_test={report['z_only_probe']['n_test']})")
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
        add(f"![collapse grid, seed {seed}](artifacts/{report['collapse_grid']})" if Path(report["run_dir"]) == out_dir else
            f"Collapse grid: `artifacts/{report['collapse_grid']}` in the seed's run directory.")
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

        add("### L1 of the s-space kernel vs P (true) and vs the tabular twin B, per zone; L1(B, P) = what the data allow")
        add("")
        if ok_mech:
            add("| zone | L1(WM, P) | L1(WM, B) | L1(B, P) |")
            add("|---|---|---|---|")
            for cat in categories:
                add(f"| {cat} | {_fmt(report['l1_vs_p_by_zone'][cat])} | {_fmt(report['l1_vs_b_by_zone'][cat])} | "
                    f"{_fmt(report['l1_b_vs_p_by_zone'][cat])} |")
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

        add(f"### Context dispersion (mean abs discrepancy vs canonical I_z): **{_fmt(report['context_dispersion_mean_abs'])}**")
        add("")

        add("### D1 (alpha_app, local and global)")
        add("")
        if ok_mech:
            add(f"Contrast accepted only if |I_L' - I_L| > max(0.25 bit, {N_SE:g} x combined MC se). alpha_app per state of the "
                f"own zone with its standard error (error propagation); a value outside [0, 1] by more than {N_SE:g} se is an "
                f"ANOMALY, listed and never averaged; the others: mean over states +/- {N_SE:g} sd/sqrt(n).")
            add("")
            add("| own zone L | target zone L' | I_L' - I_L | threshold | accepted | local: mean [interval] (n kept / n) | "
                "global: mean [interval] (n kept / n) | anomalies (local; global) |")
            add("|---|---|---|---|---|---|---|---|")
            for d in report["d1"]:
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
        acc_replay = report["accuracies"]["acc_replay_hard"]
        ok_mech = mechanics_ok(acc_replay)

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

        add(f"| {row} | template accuracy ({ACC_LABELS['acc_replay_hard']}) high | {tag} | >= {MECHANICS_FLOOR} (mini; "
            f"calibration's C-a is 0.99) | {_fmt(acc_replay)} | {'OK' if ok_mech else 'FAIL'} |")
        row += 1

        n_accepted = sum(1 for d in report["d1"] if d["accepted"])
        n_anomalies = sum(d[kind]["n_anomalies"] for d in report["d1"] if d["accepted"] for kind in ("local", "global"))
        add(f"| {row} | alpha_app defined only on accepted contrasts; anomalies listed | {tag} | contrast > max(0.25, "
            f"{N_SE:g} se) | {n_accepted}/{len(report['d1'])} pairs accepted, {n_anomalies} anomalous state value(s) | "
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
    parser.add_argument("--max-steps", type=int, default=None, help="caps the number of gradient steps (smoke tests)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_mini_config(args.config)
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.max_steps is not None:
        cfg["runtime"]["max_steps"] = args.max_steps
    device = configure_runtime(cfg)
    config_name = Path(args.config).stem
    print(f"[run_mini] config {args.config} | device {device} | seeds {cfg['seeds']} | out-dir {args.out_dir}", flush=True)

    reports, failures = [], []
    for seed in cfg["seeds"]:
        run_dir = make_seed_run_dir(args.out_dir, config_name, seed)
        try:
            reports.append(run_one_seed(cfg, seed, run_dir, device, max_steps=cfg["runtime"].get("max_steps")))
        except (SeparationNotReachedError, CollapseAfterSwitchError) as err:
            failures.append((seed, run_dir, err))
            print(f"[run_mini] seed {seed} STOPPED by a training guard ({type(err).__name__}); checkpoint and diagnostic in "
                  f"{run_dir / 'failure'}", flush=True)
        except BaseException as err:
            write_run_meta(run_dir, seed, status="failed", error=f"{type(err).__name__}: {err}")
            raise
    if len(reports) > 1:
        seeds_tag = "-".join(f"{r['seed']:02d}" for r in reports)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        combined = make_run_dir(args.out_dir, EXPERIMENT_ID, f"report__{config_name}__seeds{seeds_tag}__{stamp}")
        path = write_report(reports, combined, cfg_name=config_name)
        print(f"[run_mini] combined report: {path}", flush=True)
    for seed, run_dir, err in failures:
        print(f"[run_mini] FAILED seed {seed}: {err}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
