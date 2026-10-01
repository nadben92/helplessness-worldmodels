# E3 — status of the initial prompt, section by section

The initial E3 prompt ([`prompt_e3_worldmodel.md`](prompt_e3_worldmodel.md),
sections §0–§11) is the reference for everything `docs/e3/DECISIONS.md`
does not change. **When the two conflict, DECISIONS.md wins.** The prompt is
kept verbatim in `docs/e3/prompt_e3_worldmodel.md` (added 2026-09-25,
checked complete: §0–§11 and all their subsections).

Each section is **unchanged**, **amended** or **under review**. For an
amendment, the entry of DECISIONS.md that makes it is cited by its date and
title. "Not implemented yet" means unchanged but not built: the phase that
builds it is given.

Last updated: 2026-09-28.

| § | Section | Status | Where it changed (DECISIONS.md) |
|---|---|---|---|
| 0 | Context, role, working rules | amended | 2026-09-23 Phase 0 validation, points 4 and 7; 2026-09-25 after the mini run |
| 1 | Phase 0 — reading the repo | amended | 2026-09-23 Phase 0 validation, points 1 and 4 |
| 2 | Pixel environment | amended (§2.2) | 2026-09-23 Phase 0 validation, point 5; 2026-09-23 T2 entries |
| 3 | World model | amended (§3.1–§3.3) | 2026-09-24 and 2026-09-25 entries, see below |
| 4 | Agent | amended (§4.1, §4.3, §4.5, §4.6) | 2026-09-23 Phase 0 validation, points 1, 2, 8; 2026-09-25 episode starts; 2026-09-25 independent streams (reward convention) |
| 5 | Measures | amended (§5.1, §5.2 -- main measure I_s, §5.3 -- K_s, §5.4, §5.5, via §8) | 2026-09-30 I_s main measure; 2026-09-29 K_s; 2026-09-28 first GPU run (real contexts); 2026-09-25 independent streams (Monte-Carlo counts) |
| 6 | Conditions, runs, seeds | amended | 2026-09-25 after the mini run; 2026-09-25 episode starts; 2026-09-25 independent streams; 2026-09-28 first GPU run (reference streams) |
| 7 | Phase 4 — calibration | amended | 2026-09-23 Phase 0 validation, point 8; 2026-09-24 lr 1e-4 failure |
| 8 | Position / appearance diagnostic | amended | 2026-09-25 after the mini run (D1 rule) |
| 9 | Shape tests | amended (T2) | 2026-09-23 T2 reformulated |
| 10 | Logs and outputs | amended; parts not implemented yet | 2026-09-23 Phase 0 validation, point 1; 2026-09-25 after the mini run; 2026-09-28 first GPU run |
| 11 | Order of the phases | amended (Phase 4 split; first GPU run = E3-0) | 2026-09-24 and 2026-09-25 entries (Phase 4a / 4b); 2026-09-28 first GPU run |

## Detail

### §0 — Context, role, working rules — amended

- **Module location.** The prompt's `e3_worldmodel/` module becomes the
  repository's existing layout: `src/helplessness/agents/worldmodel/`,
  `src/helplessness/measures/`, `experiments/E3_worldmodel_4zones/`,
  `configs/e3/`, `docs/e3/`. *Phase 0 validation, point 4.*
- **Environment (§0, "PyTorch with CUDA, PowerShell commands").**
  - Local development runs on the CPU; the GPU runs are on a remote Linux
    machine.
  - The device is a config value (`runtime.device`: auto / cpu / cuda,
    and `--device`).
  - Entry points are cross-platform Python scripts. Commands are given in
    bash and PowerShell.
  - TF32 is disabled.
  - *Phase 0 validation, point 7; 2026-09-25 after the mini run (device, run outputs).*
- Reproducibility: config, commit (with a "working tree modified" flag)
  and seed are saved with every run. Bit-exact only on the same machine,
  device and number of threads. *2026-09-24 (i) validated (reproducibility);
  2026-09-25 after the mini run.*
- Everything else (phase gates, E1/E2 frozen, asking instead of
  improvising, the label s never entering the agent): unchanged.

### §1 — Phase 0 — amended

- The report lives at `docs/e3/PHASE0_REPORT.md`. *Phase 0 validation, point 4.*
- "Grid of 11 checkpoints" is corrected to E2's real grid of **16**
  log-spaced checkpoints (`sweep.py::CHECKPOINTS`). *Phase 0 validation, point 1.*

### §2 — Pixel environment — amended (§2.2)

- §2.1 dynamics: unchanged (E2's kernels imported).
- **§2.2 permutation and T2.**
  - E3 reuses **E2's seed → permutation table** by import, not an
    independent Latin square: E2's table is a balanced cyclic Latin
    square, and seed k stays comparable with E2's seed k. *Phase 0
    validation, point 5.*
  - The group that transports the kernel exactly is the group of
    goal-centred axial **reflections**, not the rotations. E2's rotations
    split into classes A = {0, 2} and B = {1, 3}, a declared **block
    factor**. *2026-09-23 T2 entries.*
- §2.3 reference environment, §2.4 rendering (k = 6, 72×72), §2.5 themes:
  unchanged. CIELAB is computed by our own code (*Phase 0 validation,
  point 6* — an implementation choice, not a change).

### §3 — World model — amended (§3.1–§3.3)

- **§3.1 architecture: amended.**
  - Straight-through alone collapses the categorical posterior. Training
    therefore uses a **probabilities phase**: the decoder receives the
    posterior probabilities until a **label-free JS criterion** (batch
    pairs) fires, then straight-through samples.
  - The switch deadline is an **absolute** number of steps; missing it
    raises `SeparationNotReachedError`.
  - The prior/posterior logit layers are **initialised small**.
  - A **guard after the switch** raises `CollapseAfterSwitchError`.
  - *2026-09-24 posterior-collapse diagnosis; categorical-bottleneck fix;
    option (c) validated; (i) validated (guard); 2026-09-25 after the
    mini run (absolute deadline).* The rest of §3.1 is unchanged.
- **§3.2 losses: amended.**
  - The reconstruction is **summed** over pixels and channels per image
    (averaged over batch × time).
  - β_dyn = 0.5 from step 0; β_rep = 0 until the switch, then a ramp to 0.1.
  - Free bits stay at 0.
  - *2026-09-24 posterior-collapse diagnosis; option (c) validated.*
- **§3.3 optimisation: amended.**
  - lr **1e-3 until the switch, then 1e-4**.
  - **No gradient clipping** (`clip_norm: null`).
  - TF32 disabled.
  - *2026-09-24 lr 1e-4 failure; (i) validated; 2026-09-25 global
    clipping disabled; 2026-09-25 after the mini run.*
  - The batch (16 × 64) is unchanged for the real runs (gpu_e30 uses it); the mini configs
    use 8 × 16.
- §3.4 replay: unchanged, and made stricter:
  - capacity is checked at startup;
  - a transition is never overwritten;
  - the model batch and the label batch are two separate objects.
  - *Phase 0 validation, adjustment D; 2026-09-24 Phase 3 points 1-2.*
- §3.5 replay ratio: unchanged (calibrated in Phase 4). Note: run_mini's `replay_ratio` counts gradient steps per environment step, the spec's replayed steps per environment step; configs state both. *2026-09-28 first GPU run.*

### §4 — Agent — amended (§4.1, §4.3, §4.5, §4.6); first version implemented on the CPU (Phase 5, 2026-09-30 (5))

- **§4.1 goal head: amended.** The reward read from the goal head is
  r = 50·p_goal − 1·(1 − p_goal), not r = 50·p_goal. This matches E2's
  convention: +50 for the step that reaches the goal (the goal reward
  *replaces* the step cost; `src/helplessness/envs/rewards/base.py:52-53`)
  and −1 for every other step. **Decided** (user, 2026-09-25: E2's
  convention confirmed, E3 aligns with it). *2026-09-25 independent
  streams, point 5; 2026-09-25 theme sensitivity.* Not implemented yet
  (Phase 5).
- §4.2, §4.4: unchanged; implemented (`agents/worldmodel/heads.py`, `imagination.py`). Untrained heads start with a
  zero output layer, and train from the estimator switch on. *2026-09-30 (5), F.*
- **Replay of the closed loop (§3.4, §4.1): amended.** A stored step is an observation with the reward received on
  arriving at it; the goal's observation is stored (terminal). E3-0's replay never held it. *2026-09-30 (5), F.*
- §5.7 Delta of the world-model agent: implemented in closed loop (`run_agent.py`, every checkpoint; real contexts main,
  canonical probe secondary); the E3 measures (I_s, I_s projected, off-support, I_z) at the end of a closed-loop run,
  paired by `aggregate_b.py` with the bias-control reading rule; resume after interruption. *2026-09-30 (7), (8).*
- Training-step accelerations (torch.compile, stacked seeds): implemented behind options, off by default, timed by the
  preflight's bench; no decision yet. *2026-09-30 (8).*
- **§4.3:** the tie-break tolerance is **relative**,
  `tol = max(1e-6, 0.01·std(Q))`. *Phase 0 validation, point 8.*
- **§4.5:** E2 had **no burn-in**; E3 has none either. A minimal prefill
  is played under the same policy as the run it precedes and counted in
  the budget. *Phase 0 validation, point 2.*
- **§4.6 frozen E2 parameters:**
  - The grid is E2's 16 checkpoints. *Phase 0 validation, point 1.*
  - **Uniform starts over the non-terminal states** were specified but
    not implemented until 2026-09-25: every earlier run started on the goal,
    state 0 = (0, 0). *2026-09-25 episode starts.*
  - H = 36, ε = 0.1 (closed loop) and the 10⁶-step budget are unchanged.
    `tests/e3/test_e3_protocol_conformance.py` checks them against E2's
    code.

### §5 — Measures — amended (§5.1, §5.2, §5.3, §5.4, §5.5)

- **§5.1 canonical probe: amended -- the main measure is taken in real contexts.**
  - The canonical probe (h0) is not valid as the main measure: template
    accuracy 0.40 from h0 against 0.98 in real contexts (mini_long), and
    the agent almost never acts from h0.
  - Main measure: for each state, the contexts (h_t, z_t) of replay windows
    unrolled with their true history where s_t = s, excluding the first 8
    steps of each window and the is_first steps; up to 50 per state, drawn
    with the measurement stream. I_z per context (same exact densities),
    averaged per state; contexts per state reported; a state without a
    context is "non mesuré". KL by zone (one-step KL of the transition taken,
    the training loss's pairing), the s-space kernel, L1 vs P and vs B,
    p_hat(frequent outcome), alpha_app and D2 are all computed in this
    regime.
  - The canonical probe stays reported as secondary. Its "KL by zone" pairs
    the posterior of s with the prior of s' and is not a fit measure.
  - *2026-09-28 first GPU run.*
- **§5.2: amended -- the main measure is I_s, I_z becomes secondary (an upper bound).**
  - **I_s** = I(a ; s' | s) of the world model's s-space kernel in real contexts: for each
    state, prior draws of z' from its contexts, decoded and matched to the templates (§5.3),
    pooled over the state's contexts (K = 1,358 draws per (s, a), spread over up to 50
    contexts), then `measures.controllability.controllability_per_state` -- the same function
    as for the true P (pooled: the analogue of E2's per-state kernel and of twin B; the per-context
    I_s is a secondary diagnostic with every final measurement, 2026-09-30 (2)). Standard error and plug-in bias by parametric bootstrap
    (`controllability_bootstrap`, 200 replicates); plug-in bias <= 0.004 bits for an exact
    noisy kernel at K = 1,358. b is computed on I_s (`aggregate_b.py`).
  - **Why:** on the bridge, I_z / I_s / truth = 2.07 / 1.71 / 1.65 bits in the controllable-noisy
    zone and 0.29 / 0.07 / 0 in the uncontrollable zones: I_z overestimates the
    controllability; the predicted code carries information on the action beyond the arrival
    state (action_in_latent.py); this bias would compress b. I_s recovers the truth where it is
    known. *2026-09-30 I_s main measure.*
  - I_z (z-space, exact densities): unchanged estimator, reported as secondary. M = 2000 for the
    diagnostics during training; M = 4096, the spec's value, for the final measures.
    *2026-09-25 independent streams, point 4.*
  - Diagnostics during training use a reduced K = 256 (plug-in bias <= 0.013 bits, L1 floor
    0.046), documented in each record; the final measures use K = 1,358. *2026-09-30.*
- **§5.3 s-space kernel: amended (K_s).** Computed from the real contexts (main) and from the
  canonical probe (secondary); the estimator (decode, nearest template, ghosts) is unchanged.
  K_s = 1,358 draws per (s, a), the smallest K whose expected L1 for an exact kernel is <= 0.02
  in a 0.9 / 4 x 0.025 zone; the floor is reported next to every L1. *2026-09-29 K_s.* It is also
  used as the bridge for `Agent.transition_distribution`. *2026-09-24 Phase 3 points 1-2.*
- **§5.4 z-only probe: amended (h-only probe added).** The same linear probe on the
  deterministic state h_t (h -> s), over the whole replay at the end, and h -> s / z -> s on the
  real contexts at every diagnostic point: how much of the state information travels through h.
  *2026-09-30.*
- §5.5 context dispersion: read from the real contexts (per zone, mean |I_real - I_canonical| and the sd over a state's contexts). *2026-09-28 first GPU run.*
- **§5.3 (added): off-support mass and I_s projected on the support.** I_s stays computed on the kernel as it is;
  the share of the kernel's mass outside the 5 reachable successors is reported by zone, and a secondary I_s on the
  kernel projected on that support (for the comparison with twin B). *2026-09-30 (5), C.*
- §5.6: unchanged. §5.7 (Delta under P): computed for the twin; not yet for the world-model agent. D1 (§8) is still
  computed on I_z (no s-space version yet).
- The measures are pure numpy functions in `src/helplessness/measures/`
  (the prompt's `measure/`). *Phase 0 validation, adjustment B.*

### §6 — Conditions, runs, seeds — amended; one deviation open

- **Every stream is derived by `SeedSequence(seed).spawn`, as in E2.** No
  component receives the integer seed (`experiments/E3_worldmodel_4zones/rng_streams.py`).
  - The first six children are E2's own streams (env_dynamics, explore,
    tiebreak, start_state, matched_control, reference).
  - E3 adds theme, network_init, replay_sample, latent_sample,
    switch_pairs and measure.
  - Measuring more or less often never changes the training trajectory
    (tested).
  - *2026-09-25 after the mini run (RNG streams); 2026-09-25 episode
    starts; 2026-09-25 independent streams, point 1.*
- The earlier deviation (actions, dynamics and themes seeded with the
  same integer) is corrected. Every run before the correction had
  correlated streams, and their noisy-zone statistics are not reliable.
- The reference environment's streams derive from their own root, SeedSequence([seed, 1]); only the permutation and the themes are shared with the experimental run. *2026-09-28 first GPU run.*
- The conditions (E3-0, A, Ref, B, E2 null) and the 20 seeds are unchanged.

### §7 — Phase 4 calibration — amended

- **C-b** is **relative to the tabular twin B**:
  L1_WM ≤ L1_B + 0.05, per zone, on pairs with N[s,a] ≥ 100. *Phase 0
  validation, point 8.*
- **The learning rate joins the calibrated hyperparameters.** *2026-09-24
  lr 1e-4 failure (selection of option (i)).*
- C-a, C-c, the selection rule and the stop on α_app ≈ 0: unchanged.

### §8 — Position / appearance diagnostic — amended

- **D1 acceptance:** |Î_L' − Î_L| > max(0.25 bit, 3 × combined MC standard
  error).
- α_app is computed for **every state** of the zone, with its **standard
  error** by propagation.
- Values outside [0, 1] beyond 3 standard errors are **anomalies**:
  listed, never averaged. The others get a mean over states with an
  interval.
- *2026-09-25 after the mini run (D1 rule).*
- The executability criterion (bootstrap CI over seeds of the local α_app
  excludes 0) and D2 are unchanged.

### §9 — Shape tests — amended (T2)

- **T2** is reformulated: the reflection group and a population test
  within each class A/B. *2026-09-23 T2 reformulated.*
- T0, T1, T3–T10: unchanged.

### §10 — Logs and outputs — amended; parts not implemented yet

- **Run directories:**
  `results/E3_worldmodel_4zones/<run name>/` through `runs.make_run_dir`
  (never overwritten). The prompt said `runs/e3/{condition}/{seed}/`.
  - Contents: config.yaml, meta.yaml, console.log, diagnostics.jsonl,
    results.json, arrays.npz, REPORT.md, artifacts/, and failure/ when a
    guard fires.
  - *2026-09-25 after the mini run (run outputs).*
- Checkpoints follow E2's 16-point grid. *Phase 0 validation, point 1.*
- Metadata: config.yaml, commit, seed and random-stream names, plus the
  permutation (rotation, quadrant assignment), the block class A/B, the
  theme colours and the episode parameters in meta.yaml. *2026-09-25
  independent streams, point 2.*
- **Implemented 2026-09-28 first GPU run:** model weights at every checkpoint of E2's grid
  (E3-0: grid point c -> gradient step round(c x replay_ratio)) and at the
  end, `--resume`, the compact replay (`replay.npz`, which carries every
  column of the spec's transitions table except the episode index), the
  world (`world.json`) and `recompute_measures.py`.
- **Not implemented yet:**
  - `transitions.parquet` as such (the content is in `replay.npz`);
  - aggregation scripts comparable to E2's.

### §11 — Order of the phases — amended

- P0–P3 are done. **P4 is split**:
  - 4a: calibration code, periodic checkpoints and the GPU-readiness
    items, run locally;
  - 4b: calibration runs on the remote GPU machine, after the GPU
    determinism policy is set.
  - *2026-09-24 and 2026-09-25 entries (notes for Phase 4a / 4b).*
- A long diagnostic run on the mini configuration (`mini_long.yaml`) came
  before 4a (done 2026-09-26; `mini.yaml` retired, `mini_long.yaml` is the CPU
  reference). The first GPU run is E3-0 (`gpu_e30.yaml`: seeds 0 and 5,
  environment and reference, 8x8 latent). *2026-09-25 after the mini run; 2026-09-28 first GPU run.*
- P5, P6 and the CLAUDE.md update at the end of P6: unchanged.
