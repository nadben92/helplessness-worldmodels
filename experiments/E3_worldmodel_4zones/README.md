# E3 — World model, 4 zones (orchestration)

See `docs/E3_worldmodel_4zones.md` for the experiment's status, and
`docs/e3/` for the phase reports, architecture proposal, and running
decisions log (`docs/e3/DECISIONS.md`).

**Status: Phase 1 (environment, rendering, themes, permutation,
reference) in progress.** No agent, no world model, no training loop
exists yet — those are Phase 2 onward.

## What this directory contains so far

- `env.py`: builds E3's pixel-observation `TorusZonedEnv` (experimental
  and reference/paired-control) by importing
  `experiments/E2_online_4zones/env.py` unchanged (via
  `helplessness.experiment_loader.load_experiment_module`) for every
  piece of zone/dynamics/permutation logic — E3 adds only the
  observation backend (`helplessness.envs.observation.allocentric_pixel_backend.AllocentricPixelBackend`)
  and theme construction
  (`helplessness.envs.appearance.theme_sampling.build_themes`). See that
  module's own docstring for exactly what is reused vs. added.
- `config.yaml`: environment-only configuration (episode/budget/render
  parameters already decided). Model hyperparameters are not here — they
  are calibrated in Phase 4 and frozen into `configs/e3_frozen.yaml`.

## Reuse discipline

`experiments/E1_offline_4zones/` and `experiments/E2_online_4zones/` are
frozen for this task: imported, never modified. If E3 ever needs
behaviour those directories don't already expose, the fix is a new
function in this directory (or in `src/helplessness/`, if genuinely
reusable library code), never an edit to E1/E2's own files — see
`docs/e3/DECISIONS.md` for every place this reuse boundary was actually
exercised so far.

## Block-factor structure of E2's counterbalancing (T2)

See `tests/e3/test_e3_symmetry.py`'s module docstring and
`docs/e3/DECISIONS.md` (the two 2026-09-23 entries): the symmetry group
that actually transports the transition tensor exactly, for every zone
type (drift included), is the goal-centred AXIAL REFLECTION group
{identity, row-reflection, column-reflection, point-reflection} — not
the full rotation group. E2's own 4 counterbalancing rotations (the
cyclic group C4, reused as-is) split under this group into two classes:
**class A = {rotation 0, rotation 2} = seeds {0-4, 10-14}**, **class B =
{rotation 1, rotation 3} = seeds {5-9, 15-19}** (10/10 split, verified).
No reflection relates a class-A rotation to a class-B one. This is now
a declared block factor: `env.py::block_class_for_seed(seed_index)`,
recorded in every run's metadata and reported as its own column in
aggregated tables from Phase 4 onward — never pooled across classes
silently.
