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
from helplessness.pipeline.world import (  # moved to the library on 2026-10-14, re-exported here
    EXPERIMENT_CONFIG_PATH,
    EXPERIMENT_ID,
    ZONE_PRESET,
    GRID_SHAPE,
    N_STATES,
    N_ACTIONS,
    PAIR_ROLE,
    NOISY_ZONES,
    zone_of_state,
    zone_states,
    _dynamics_kind,
    zone_kinds,
    load_episode_config,
    run_identity,
    themes_to_dict,
    themes_from_dict,
    World,
    build_world,
    build_run,
    collect_and_count,
    build_tabular_twin,
)
from helplessness.pipeline.measure import (  # moved to the library on 2026-10-14, re-exported here
    N_SE,
    ACC_LABELS,
    SLOT_SEPARATION,
    SLOT_DIAG,
    SLOT_IZS,
    SLOT_ZONLY,
    SLOT_D1_REAL,
    SLOT_FAILURE,
    SLOT_FINAL,
    SLOT_D1_CANONICAL,
    SLOT_HONLY,
    SLOT_PER_CONTEXT,
    zone_mean_mc_se,
    labelled_between_state_js,
    kl_post_prior_by_zone,
    frequent_outcome_preview,
    save_collapse_grid,
    _plt,
    real_context_settings,
    kernel_draws,
    controllability_of_kernel,
    per_context_is_check,
    is_by_zone,
    real_context_measures,
    training_diagnostics,
    _d1_pairs,
    d1_analysis,
    d1_analysis_real,
    measure_final,
)
from helplessness.pipeline.checkpoints import (  # moved to the library on 2026-10-14, re-exported here
    e2_grid_env_steps,
    checkpoint_steps,
    _atomic_torch_save,
    save_weights,
    _generator_state,
    _set_generator_state,
    save_failure_diagnostic,
    make_seed_run_dir,
    build_rssm,
    world_record,
    load_run_data,
)
from helplessness.pipeline.measure import (  # moved to the library on 2026-10-14, re-exported here
    ARRAY_FIELDS,
)

START_DISTRIBUTIONS = ("uniform_non_terminal",)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e3" / "mini.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results"
DETERMINISTIC_ZONES = ("controllable_deterministic", "uncontrollable_deterministic")
DETERMINISTIC_KINDS = DETERMINISTIC_ZONES
LOSS_TERMS = ("total", "recon", "recon_patch", "kl_dyn", "kl_rep")
MECHANICS_FLOOR = 0.5  # below this template-match accuracy, downstream indicators are not evaluable
NOT_EVALUABLE = "non évaluable (mécanique en échec)"
NOT_MEASURED = "non mesuré"


# ---------------------------------------------------------------------
# verdict rules (pure functions -- tested in tests/e3/test_e3_mini_report.py)
# ---------------------------------------------------------------------


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
# environment, data, twin; measurements; checkpoints, resume: moved to helplessness.pipeline
# (world.py, measure.py, checkpoints.py; runtime.py for the runtime part above) on 2026-10-14
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# one (seed, condition)
# ---------------------------------------------------------------------


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
