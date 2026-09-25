# E3 — proposed module structure

Written after Phase 0 validation and your structural decisions (see
`docs/e3/DECISIONS.md`). Nothing under this tree exists yet except the
files explicitly marked `[exists]`. This is a proposal — waiting for your
go-ahead before Phase 1 writes any of it.

Guiding rule from your decision #4: reusable, torch-dependent library
code goes into `src/helplessness/` (the reserved
`agents/worldmodel/` slot, plus new files elsewhere in `src/helplessness/`
that do not touch anything E1/E2 depends on); run-specific orchestration,
configs and reports stay under `experiments/E3_worldmodel_4zones/`,
`configs/`, `docs/e3/`. E1/E2 files are read-only; new library files are
allowed.

```
src/helplessness/
  agents/
    worldmodel/                      [new -- the CLAUDE.md-reserved slot]
      __init__.py
      rssm.py         Encoder, Decoder, GRU dynamics core, prior/posterior MLPs,
                       straight-through categorical z, latent_hook(h, z, context)
                       (identity by default -- spec 3.1's E6 hook, wired but unused).
      losses.py        Masked reconstruction loss (background-diff dilation mask,
                       computed from images only, no state label), KL(dyn)/KL(rep)
                       with free_bits/free_bits_mode kept as inert config knobs.
      heads.py          Goal head g(h,z)->p_but (BCE, sg([h,z]) input), critic
                       V(h,z) + its slow EMA target (tau=0.02).
      agent.py           WorldModelAgent: one-step-anticipation epsilon-greedy
                       policy (K_act rollouts, Q formula from spec 4.3), relative
                       tie-break tolerance (your decision #8), per-step filtering
                       (h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}), z_t ~ posterior).
      imagination.py      Critic-in-imagination training: horizon 15, K_imag
                       rollouts, lambda-returns (spec 4.4), c_k-weighted loss.
      replay.py            Episode buffer: capacity >= budget, uniform 64-step
                       window sampling across episode boundaries, is_first-driven
                       reset; stores x, a, r, terminal, truncated, is_first, and
                       the true label s (log-only, NEVER read by agent/rssm code
                       -- see "agent/instrument boundary" below).
      loop.py               Training loop: env interaction, replay_ratio-scheduled
                       updates, prefill_steps (epsilon=0.1, same policy as
                       training -- your decision #2), checkpoint hooks. Mirrors
                       agents/online/loop.py's role for this agent family.
  envs/
    observation/
      allocentric_pixel_backend.py  [new -- pixel_backend.py untouched]
                       render(s, layout, themes, k=6, grid_lines=False) -> uint8
                       [H,W,3]; render_background(layout, themes, k) (no agent,
                       for the reconstruction-loss mask); render_counterfactual
                       (s, layout, themes, override, k) (diagnostics only, never
                       used for training). AllocentricPixelBackend(ObservationBackend)
                       -- thin adapter for TorusZonedEnv (see open point A below).
    appearance/
      cielab.py         [new -- themes.py untouched]
                       srgb_to_lab(rgb), lab_distance(lab1, lab2) (deltaE76
                       Euclidean by default) -- self-contained, numpy only,
                       tested against published sRGB/Lab reference pairs (T1).
      theme_sampling.py  [new]
                       Themes (category -> RGB), sample_themes(categories, rng,
                       min_dist, metric="cielab", pair_distance=None) enforcing
                       pairwise CIELAB distance via cielab.py; agent-marker-colour
                       selection (>= threshold CIELAB distance from every zone
                       theme, the corridor, and the goal colour).
  measures/
    latent_controllability.py  [new -- controllability.py untouched]
                       Exact categorical H(z'|s,a); Monte-Carlo H(z'|s) via
                       logsumexp with reported MC standard error; I_z(s) with the
                       [0, log2(5)] bounds assertion (spec 5.2). Pure numpy,
                       array-in/array-out -- no torch import (see open point B).
    kernel_probe.py             [new -- belief.py untouched]
                       Nearest-template assignment from decoded-image L2
                       distances -> p_hat(s'|s,a); ghost classification by
                       margin threshold; H_s(s'|s,a), I_s; L1 vs P_true and vs
                       T_hat_B. Pure array-in/array-out.
    diagnostics.py              [new]
                       alpha_app(s) (spec 8), D2 Jensen-Shannon distances between
                       canonical-probe posteriors at neighbouring cells. Pure
                       array-in/array-out.

configs/
  e3/                    [new]
    model_defaults.yaml  n_vars/n_classes/h_dim/w_patch/replay_ratio/
                       prefill_steps -- the values phase 4's sweep varies.
    render.yaml            k, grid_lines, cielab min_dist, pair_distance
                       (reserved, unused in E3), agent-marker CIELAB threshold.
  e3_frozen.yaml         [written by you at the end of phase 4 -- spec section 7]

experiments/
  E3_worldmodel_4zones/  [new -- orchestration only, nothing imported from here]
    README.md
    config.yaml
    env.py             Builds the E3 ZoneLayout + reference layout: imports
                       experiments/E2_online_4zones/env.py's counterbalancing
                       functions AS-IS (see "permutation reuse" below), wires
                       AllocentricPixelBackend + sample_themes + TorusZonedEnv
                       (reusing GoalReward/terminal_states exactly as E2's own
                       build_env does).
    seed_table.py       Documents/tests that E3's seed k reuses E2's own
                       rotation_for_seed_index/quadrant_assignment_for_rotation
                       output bit-for-bit (your decision #5) -- imported, not
                       reimplemented.
    rng.py                E3's stream layout: extends E2's STREAM_NAMES
                       (env_dynamics, explore, tiebreak, start_state,
                       matched_control, reference) with the E3-only streams
                       spec section 6 requires (theme, network_init,
                       replay_sample, latent_sample).
    probe.py               The canonical-probe forward pass through the frozen
                       model (torch-dependent) -- produces the plain numpy
                       arrays measures/latent_controllability.py,
                       measures/kernel_probe.py and measures/diagnostics.py
                       consume. This is the one file that bridges "the model"
                       and "the measures," analogous to
                       agents.base.materialize_transition_tensor's role for the
                       tabular agents.
    twin.py                 Condition B (and B_ref): reuses
                       experiments/E2_online_4zones/twin.py::twin_from_counts
                       AS-IS (imported via the same mechanism as env.py's reuse)
                       -- no reimplementation of T_hat/I_hat/b/Delta.
    calibration.py          Phase 4 capacity sweep driver.
    diagnostics_run.py      Phase 8 D1-local/D1-global/D2 counterfactual-
                       rendering diagnostics driver.
    run_e3_0.py / run_a.py / run_ref.py / run_b.py  (or one run.py with a
                       --condition flag -- decided in phase 5/6, not now)

docs/
  e3/
    PHASE0_REPORT.md     [exists]
    ARCHITECTURE.md       [this file]
    DECISIONS.md            [exists -- running log, updated every phase]
    CALIBRATION_REPORT.md  [phase 4 deliverable]
  E3_worldmodel_4zones.md  [exists -- status doc, updated per "Adding an
                       experiment" convention as phases complete]

tests/                    [mirrors src/, as always]
  agents/worldmodel/
    test_rssm.py, test_losses.py, test_heads.py, test_agent.py,
    test_imagination.py, test_replay.py, test_loop.py
  envs/observation/
    test_allocentric_pixel_backend.py
  envs/appearance/
    test_cielab.py, test_theme_sampling.py
  measures/
    test_latent_controllability.py, test_kernel_probe.py, test_diagnostics.py
  e3/                    [experiment-local orchestration tests, mirroring the
                       tests/agents/test_e2_*.py convention for
                       experiments/E2_online_4zones/]
    test_e3_env.py, test_e3_seed_table.py, test_e3_probe.py, test_e3_twin.py, ...
```

## Permutation reuse (your decision #5)

`experiments/E2_online_4zones/env.py` is importable cleanly today via the
existing `tests/_experiment_loader.py::load_experiment_module` helper
(already used by `tests/envs/test_e2_4zone_layout.py` for exactly this
purpose — loads an experiment-directory file by explicit path, under a
caller-chosen name, with sibling-import isolation, no `sys.modules`
collision risk). That loader is generic infrastructure (no E1/E2 domain
logic in it), so `experiments/E3_worldmodel_4zones/env.py` gets its own
copy of the ~60-line loader (or a shared one factored out of `tests/` —
open point C below) and calls
`load_experiment_module("E2_online_4zones", "env.py", "e3_e2_env")` to
reach `quadrant_assignment_for_rotation`, `rotation_for_seed_index`,
`QUADRANT_ORDER`, `identity_quadrant_assignment`, `build_layout`,
`build_control_layout` **directly** — zero duplication, zero move. The
"write a new function + equivalence test" fallback is not needed.

Verified (Phase 0 follow-up): E2's seed→permutation table is a properly
balanced cyclic Latin square — every (dynamics type, quadrant) pair
occurs in exactly 5 of the 20 seeds. E3 reusing it as-is means E3's seed
`k` gets the identical quadrant assignment as E2's seed `k`, satisfying
your comparability requirement directly.

## Open points inside this proposal (flagging, not deciding silently)

**A. `AllocentricPixelBackend`'s relationship to `ObservationBackend`.**
The ABC's `observe(position, zone_layout, theme_seed)` takes a single
int `theme_seed`; spec 2.4's `render(s, layout, themes)` takes a `Themes`
object (the fixed, pre-sampled per-run theme set from `sample_themes`,
not a rotation angle). My default: the backend is constructed with a
`Themes` object already resolved for the run, and its `observe()` ignores
the `theme_seed` argument it's handed each call (spec 2.5 forbids
per-episode re-theming anyway, so there is nothing for that argument to
do). This satisfies the existing ABC without changing it, but it is a
slightly unusual use of that parameter — flagging rather than assuming.

**B. Keeping torch fully out of `measures/`.** Not explicitly stated in
your decision #4 ("mesures ... rangées selon les conventions
existantes"), but I'm reading "existing conventions" as including
`CLAUDE.md`'s "measures/ ... never import an environment or agent class"
— by extension, never import torch either, since the whole point is
array-in/array-out portability to whatever backend (tabular today, RSSM
now, something else later). Concretely: `probe.py` (torch-dependent, in
`experiments/E3_worldmodel_4zones/`) does every forward pass through the
frozen model and hands plain numpy arrays to
`measures/latent_controllability.py` etc. — the same
"agent-side bridge, never a measures-side one" pattern
`agents.base.materialize_transition_tensor` already establishes for the
tabular case. Confirm this reading is what you intended.

**C. `tests/_experiment_loader.py`.** Currently test-only. E3's
*production* code (`experiments/E3_worldmodel_4zones/env.py`, `twin.py`)
needs the same loader at runtime, not just in tests. Proposing a small,
identical copy inside `experiments/E3_worldmodel_4zones/` (it's ~60 lines
of generic import machinery, not E1/E2 domain logic, so copying it isn't
"duplicating E1/E2's logic" in the sense your decision #5 was guarding
against) — alternative is promoting it out of `tests/` into a shared
utility location, which is a bigger, less obviously-scoped change.
