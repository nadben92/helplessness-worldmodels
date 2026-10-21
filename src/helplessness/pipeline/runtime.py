"""E3's run pipeline: runtime set-up and run bookkeeping -- threads, TF32 and determinism, the device, the code's
provenance and the environment's metadata, the console log, the training and measurement random streams, JSON
helpers, and the separation switch built from a config. Moved verbatim from
experiments/E3_worldmodel_4zones/run_mini.py on 2026-10-14 (run_mini.py re-exports every name); the module-level
names those functions read are imports here (REPO_ROOT from helplessness.config, E3's streams from
helplessness.streams.worldmodel).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.loop import SeparationSwitch
from helplessness.agents.worldmodel.rssm import resolve_device
from helplessness.config import REPO_ROOT
from helplessness.runs import code_provenance
from helplessness.streams import worldmodel as e3_rng


def configure_runtime(cfg: dict) -> torch.device:
    """Threads, TF32, determinism and device, set once at startup and
    recorded in meta.yaml. TF32 is disabled explicitly for matmul AND cuDNN
    convolutions (A100 default: on for convolutions): the probabilities
    phase separates the states on tiny probability differences, which a
    reduced-precision convolution could change (docs/e3/DECISIONS.md,
    2026-09-25). runtime.deterministic (GPU determinism policy, audit B7,
    docs/e3/DECISIONS.md 2026-09-28): deterministic algorithms requested
    (warn_only -- an op without a deterministic implementation warns in
    the console instead of stopping a multi-hour run), cuDNN deterministic,
    no cuDNN benchmark; CUBLAS_WORKSPACE_CONFIG is set by main() before
    CUDA starts."""
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deterministic = bool(cfg["runtime"].get("deterministic", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    return resolve_device(cfg["runtime"]["device"])


def git_state() -> dict:
    """The code's version (runs.code_provenance): the git commit when git
    works here, and always a content hash of the source files -- the first
    GPU batch ran where git was unavailable and recorded only "unknown"
    (docs/e3/DECISIONS.md, 2026-09-29)."""
    return code_provenance(REPO_ROOT)


def max_images_per_forward(cfg: dict) -> int | None:
    """measure.max_images_per_forward: at most this many images through
    the encoder at once in the real-context and z-only probes
    (loop.encode_windows); null (or absent, as in every config before
    2026-09-29) keeps the one-call encoding, bit for bit."""
    return cfg["measure"].get("max_images_per_forward")


def gpu_peak_start(device: torch.device) -> None:
    """Resets the peak-memory counters before a measurement (no-op off GPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_end(device: torch.device) -> dict | None:
    """Peak GPU memory (MiB) since gpu_peak_start: allocated by tensors and
    reserved by the caching allocator (what other processes cannot use).
    Then returns the cached blocks to the device (torch.cuda.empty_cache):
    runs sharing a GPU otherwise keep a measurement's peak reserved for the
    rest of their life (the first GPU batch: 29 GiB per spec-size process).
    None off GPU."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    peak = {"peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
    torch.cuda.empty_cache()
    return peak


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
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
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


def seed_streams(seed: int, device: torch.device, condition: str = "experimental") -> dict:
    """The training streams of a (seed, condition) (rng_streams.py: every
    stream derived by SeedSequence.spawn, as E2; docs/e3/DECISIONS.md,
    2026-09-25). Training consumes only "collect" (E2's explore stream: the
    collection actions), "start_state" (E2's), "replay" (minibatch
    windows), "switch_pairs" (the label-free switch statistic, on a fixed
    schedule) and "latent" (every latent draw of the training step); the
    network initialisation seeds the global torch RNG from network_init
    just before the model is built. The theme and dynamics streams are
    taken by build_world. Every measurement draws from its own child of
    the measure stream (measure_streams), so measuring more or less often
    never changes the training trajectory. A reference run's streams
    derive from their own root (rng_streams.py), except the themes."""
    streams = e3_rng.spawn_streams(seed, condition)
    seqs = e3_rng.spawn_sequences(seed, condition)
    return {
        "collect": streams["explore"],
        "start_state": streams["start_state"],
        "replay": streams["replay_sample"],
        "switch_pairs": streams["switch_pairs"],
        "latent": e3_rng.torch_generator(seqs["latent_sample"], device),
        "network_init_seed": e3_rng.torch_seed(seqs["network_init"]),
        "device": device,
        "seed": seed,
        "condition": condition,
    }


def measure_streams(seed: int, slot: int, streams: dict) -> tuple[np.random.Generator, torch.Generator]:
    """A (numpy, torch) pair owned by one measurement (rng_streams.measure_streams)."""
    return e3_rng.measure_streams(seed, slot, streams["device"], streams.get("condition", "experimental"))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _nan(value):
    """JSON null (a NaN written by _jsonable) back to NaN."""
    return float("nan") if value is None else value


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
    threshold, collapse = switch_thresholds_bits(est, cfg["model"]["n_vars"])
    switch = SeparationSwitch(
        max_steps=est["switch_max_steps"], threshold_bits=threshold, consecutive=est["switch_consecutive"],
        eval_every=est["switch_eval_every"], min_steps=est["switch_min_steps"],
        collapse_threshold_bits=collapse, collapse_consecutive=est["collapse_consecutive"],
        # E5's late threshold (docs/e5/DECISIONS.md (28)); absent in E3's and E4's configs: off
        late_threshold_bits=est.get("switch_late_threshold"), late_after_steps=est.get("switch_late_after_steps"),
    )
    return switch, est["switch_n_pairs"]


def switch_thresholds_bits(est: dict, n_vars: int) -> tuple[float, float]:
    """The switch and collapse thresholds in bits of the batch-pair JS
    (a SUM over the latent variables, each bounded by 1 bit). A config gives
    them either in bits (switch_js_threshold, collapse_js_threshold: every
    config written before 2026-09-30 (7)) or as a fraction of the ceiling,
    scale-free in the number of variables (switch_js_threshold_per_var,
    collapse_js_threshold_per_var: threshold = fraction x n_vars; 0.375 and
    0.125 reproduce 3 and 1 bits at n_vars = 8 exactly). Not both."""
    out = []
    for key in ("switch_js_threshold", "collapse_js_threshold"):
        per_var = est.get(f"{key}_per_var")
        if per_var is not None and key in est:
            raise ValueError(f"train.latent_estimator gives both {key} and {key}_per_var: choose one")
        out.append(float(per_var) * n_vars if per_var is not None else float(est[key]))
    return out[0], out[1]
