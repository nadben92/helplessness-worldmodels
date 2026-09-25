# E3 — World model, 4 zones

An RSSM world model (DreamerV3, simplified) trained on pixel observations of
the 4-zone torus, and the measures of how much controllability its learned
dynamics perceive. Experiment status: `docs/E3_worldmodel_4zones.md`.
Decisions, in order, with their evidence: `docs/e3/DECISIONS.md`.

**Status (2026-09-25): end of Phase 3.** The pipeline runs end to end on the
mini configuration. A long diagnostic run (`configs/e3/mini_long.yaml`) comes
next, then Phase 4a (calibration code). No agent policy yet (Phase 5).

## Install

Python ≥ 3.10. From the repository root:

**Linux / macOS, GPU machine** (PyTorch ≥ 2.2; pip's default `torch` wheel on
Linux is the CUDA build):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[e3,dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

If the machine's CUDA driver needs a specific build, install torch first
from the matching index, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu121`,
then `pip install -e ".[e3,dev]"`.

**Windows, CPU only** (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
.venv\Scripts\python.exe -m pip install -e ".[e3,dev]"
```

The editable install (`-e`) is required: the scripts locate the repository
from their own path.

## Run

One command per run. It runs every seed of the config, one after the other.

```bash
python experiments/E3_worldmodel_4zones/run_mini.py --config configs/e3/mini.yaml --device auto
```

```powershell
.venv\Scripts\python.exe experiments\E3_worldmodel_4zones\run_mini.py --config configs\e3\mini.yaml --device auto
```

| option | default | meaning |
|---|---|---|
| `--config` | `configs/e3/mini.yaml` | configuration file. `configs/e3/mini_long.yaml`: the long diagnostic run |
| `--device` | `runtime.device` of the config (`auto`) | `auto` (CUDA if available), `cpu`, `cuda` |
| `--out-dir` | `results/` | results root |
| `--seeds` | `seeds` of the config | e.g. `--seeds 0` to run one seed |
| `--max-steps` | none | caps the number of gradient steps (smoke tests only; the report says so) |

The device actually used is printed at startup and recorded in `meta.yaml`.
TF32 is disabled on purpose (see `docs/e3/DECISIONS.md`, 2026-09-25).

On Windows, keep the default `--out-dir` to stay under the 260-character path limit (run directory names are long).

To run seeds in parallel, start one process per seed (`--seeds N`), with
`runtime.num_threads` in the config adjusted to the number of cores.
Each seed writes its own directory, so parallel processes never collide.

Keep a copy of the console with `... 2>&1 | tee run.log` (bash) or
`... 2>&1 | Tee-Object run.log` (PowerShell). Each run directory also has
its own `console.log`.

| config | seeds | gradient steps | duration, this CPU (6 threads) |
|---|---|---|---|
| `mini.yaml` | 0, 5 | 10,512 each | ~1.1-1.3 h per seed |
| `mini_long.yaml` | 0 | 40,000 | ~5-5.5 h (see the config's header) |

## Where the results are

One directory per seed, never overwritten:

```
results/E3_worldmodel_4zones/grid12x12__4zones__pair-experimental__worldmodel-<config>__seed<NN>__<UTC timestamp>/
    REPORT.md            the report; figures in artifacts/
    config.yaml          the resolved configuration (after command-line overrides)
    meta.yaml            status (running / done / failed), seed, git commit and "working tree modified" flag,
                         python / torch / CUDA / cuDNN / GPU, device, TF32 and determinism flags, threads,
                         switch step, deadline and margin
    console.log          everything printed
    diagnostics.jsonl    training diagnostics, one JSON line per point, appended as soon as each one is computed
                         (a crash loses at most the last interval)
    results.json         scalars and curves of the seed, written when the seed ends
    arrays.npz           counts N, twin T_hat_B, true P, kernel p_hat, I_z, canonical posteriors / priors, L1 arrays
    failure/             only if a training guard stopped the seed: checkpoint.pt + diagnostic.json
```

With several seeds, a combined report is written to
`results/E3_worldmodel_4zones/report__<config>__seeds<..>__<UTC>/REPORT.md`.

**Training guards.** When they fire, the checkpoint and the diagnostic are
saved automatically, the seed is marked `failed` in `meta.yaml`, and the
remaining seeds still run. The exit code is 1 if a guard fired.
- `SeparationNotReachedError`: the posterior has not separated the states
  by the absolute deadline `switch_max_steps`.
- `CollapseAfterSwitchError`: the separation is lost after the switch.

## Tests

```bash
python -m pytest tests/agents/worldmodel tests/measures tests/e3 -q
```

`tests/e3/test_e3_run_outputs.py` runs the real entry point on a tiny
configuration (≈ 1 min). `tests/e3/test_e3_ae_isolation.py` is the slowest
test (≈ 2 min).

GPU smoke test (Colab T4): `docs/e3/GPU_SMOKE_TEST.md`.

## What this directory contains

- `run_mini.py`: the run entry point above.
- `probe.py`: the canonical probe and every measurement built on it. The
  measures themselves (entropies, KL, α_app, JS, ghost classification)
  live in `src/helplessness/measures/`.
- `env.py`: builds E3's pixel-observation `TorusZonedEnv` by importing
  `experiments/E2_online_4zones/env.py` unchanged. It adds only the
  observation backend (`allocentric_pixel_backend`) and the theme
  construction (`theme_sampling`).
- `ae_isolation.py`, `compare_estimators.py`, `compare_probs_phase_aids.py`:
  the autoencoder isolation test and the estimator / procedure comparisons
  behind `docs/e3/ESTIMATOR_COMPARISON.md`.
- `profile_throughput.py`: throughput of the environment and of a training
  step (`--device`).
- `config.yaml`: the environment-only configuration of the full experiment
  (1e6 steps, the 16-checkpoint grid).

## Reuse discipline

`experiments/E1_offline_4zones/` and `experiments/E2_online_4zones/` are
frozen for this task: imported, never modified. If E3 needs behaviour they
don't expose, the fix is a new function here, or in `src/helplessness/` if it
is genuinely reusable library code — never an edit to E1/E2's own files.

## Block-factor structure of E2's counterbalancing (T2)

See `tests/e3/test_e3_symmetry.py` and `docs/e3/DECISIONS.md` (the two
2026-09-23 entries). The symmetry group that transports the transition
tensor exactly, for every zone type (drift included), is the goal-centred
axial reflection group, not the rotation group. E2's 4 counterbalancing
rotations split into **class A = {rotation 0, rotation 2} = seeds {0-4,
10-14}** and **class B = {rotation 1, rotation 3} = seeds {5-9, 15-19}**.
This is a declared block factor (`env.py::block_class_for_seed`), recorded
in every run and reported as its own column — never pooled across classes
silently.
