# E3 — World model, 4 zones

An RSSM world model (DreamerV3, simplified) trained on pixel observations of
the 4-zone torus, and the measures of how much controllability its learned
dynamics perceive. Experiment status: `docs/E3_worldmodel_4zones.md`.
Decisions, in order, with their evidence: `docs/e3/DECISIONS.md`.

**Status (2026-09-28): first GPU run prepared (E3-0, `configs/e3/gpu_e30.yaml`).**
The main measure is taken in real contexts (`docs/e3/DECISIONS.md`,
2026-09-28); every run saves its weights on E2's checkpoint grid, its
compact replay and its world, and `recompute_measures.py` recomputes every
measure from a run directory. `mini.yaml` is retired; `mini_long.yaml` is
the CPU reference. No agent policy yet (Phase 5).

## Running on the GPU (A100)

**To launch the first GPU runs, follow [`RUN_GPU.md`](RUN_GPU.md)** (install, short check,
launch, results to send back). What they are: the 4 runs of `configs/e3/gpu_e30.yaml`
(seeds 0 and 5, experimental and reference environments, spec architecture) and the bridge
run `configs/e3/gpu_e30_bridge.yaml` (seed 0, mini_long's network); budget and reasons in
`docs/e3/DECISIONS.md` (2026-09-28). The rest of this README is for development.

## Install (development machine)

Python >= 3.10. From the repository root. **Linux / GPU machine**: see [`RUN_GPU.md`](RUN_GPU.md).
**Windows, CPU only** (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
.venv\Scripts\python.exe -m pip install -e ".[e3,dev]"
```

The editable install (`-e`) is required: the scripts locate the repository
from their own path.

## Run

One command per configuration; it runs every (seed, condition) of the
config, one after the other.

```bash
python experiments/E3_worldmodel_4zones/run_mini.py --config configs/e3/mini_long.yaml --device auto
```

```powershell
.venv\Scripts\python.exe experiments\E3_worldmodel_4zones\run_mini.py --config configs\e3\mini_long.yaml --device auto
```

| option | default | meaning |
|---|---|---|
| `--config` | `configs/e3/mini.yaml` (retired) | `mini_long.yaml`: CPU reference; `gpu_e30.yaml`: first GPU run |
| `--device` | `runtime.device` of the config | `auto` (CUDA if available), `cpu`, `cuda` |
| `--out-dir` | `results/` | results root |
| `--seeds` | `seeds` of the config | e.g. `--seeds 0` |
| `--conditions` | `conditions` of the config | `experimental`, `reference` |
| `--max-steps` | none | caps the number of gradient steps (smoke tests only; the report says so) |
| `--resume DIR` | - | continue the interrupted run in DIR |
| `--combine DIR...` | - | combined report of finished runs |
| `--wandb` | off unless `WANDB_API_KEY` is set | force the optional W&B tracking on (`wandb_logger.py`; install the `wandb` extra) |
| `--wandb-artifact` | off | with W&B on, upload the run directory as an artifact at the end (weights, not `resume.pt`) |

The device actually used is printed at startup and recorded in `meta.yaml`.
TF32 is disabled on purpose (see `docs/e3/DECISIONS.md`, 2026-09-25).
On Windows, keep the default `--out-dir` to stay under the 260-character path
limit (run directory names are long).

| config | runs | gradient steps | duration |
|---|---|---|---|
| `mini_long.yaml` | seed 0, experimental | 40,000 of 8 x 16 | ~7 h on this CPU (6 threads) |
| `gpu_e30.yaml` | seeds 0, 5 x experimental, reference | 60,000 of 16 x 64 | target <= ~6 h per run on an A100, 5 side by side |
| `gpu_e30_bridge.yaml` | seed 0, experimental | 60,000 of 8 x 16 | shorter (small network, 16-step windows) |

## Where the results are

One directory per (seed, condition), never overwritten:

```
results/E3_worldmodel_4zones/grid12x12__4zones__pair-<experimental|control>__worldmodel-<config>__seed<NN>__<UTC timestamp>/
    REPORT.md            the report; figures in artifacts/
    config.yaml          the resolved configuration (after command-line overrides)
    meta.yaml            status (running / done / failed), seed, condition, git commit and "working tree modified"
                         flag, python / torch / CUDA / cuDNN / GPU, device, TF32 and determinism flags, threads,
                         permutation, themes, switch step, deadline and margin
    world.json           seed, condition, permutation, themes, render, dynamics seed, hash of the 144 images
    replay.npz           compact replay (s, a, r, flags, s_next), N[s,a,s'], episode starts
    console.log          everything printed
    diagnostics.jsonl    training diagnostics (real contexts + canonical probe), one JSON line per point
    checkpoints/         weights_stepNNNNNNNN.pt (E2's grid), weights_final.pt, resume.pt, latest.json
    results.json         scalars and curves, written when the run ends
    arrays.npz           N, T_hat_B, P, real-context arrays (per state and per context), canonical arrays
    recomputed/          outputs of recompute_measures.py, if run
    failure/             only if a training guard stopped the run: checkpoint.pt + diagnostic.json
```

**Recomputing the measures** from a run directory, without training:

```bash
python experiments/E3_worldmodel_4zones/recompute_measures.py <run dir> --checkpoint final     # or: all, failure, <gradient step>
```

**Training guards.** When they fire, the checkpoint and the diagnostic are
saved automatically, the run is marked `failed` in `meta.yaml`, and the
remaining runs of the process still run. The exit code is 1 if a guard fired.
- `SeparationNotReachedError`: the posterior has not separated the states
  by the absolute deadline `switch_max_steps`.
- `CollapseAfterSwitchError`: the separation is lost after the switch.

## Tests

```bash
python -m pytest tests/agents/worldmodel tests/measures tests/e3 -q
```

`tests/e3/test_e3_run_outputs.py` and `tests/e3/test_e3_real_context.py` run
the real entry point on a tiny configuration (resume after an interruption,
recomputation from the directory). `tests/e3/test_e3_ae_isolation.py` is the
slowest test (about 2 min).

## What this directory contains

- `run_mini.py`: the run entry point above.
- `recompute_measures.py`: every measure recomputed from a run directory.
- `wandb_logger.py`: the optional W&B tracking (never stops a run).
- `preflight.py`, `launch_gpu_e30.sh`, `RUN_GPU.md`: the GPU check, the launcher and the instructions of the first GPU run.
- `probe.py`: the real-context measure, the canonical probe and every measurement built on them. The
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
