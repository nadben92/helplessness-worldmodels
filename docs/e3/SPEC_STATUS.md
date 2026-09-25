# E3 — status of the initial prompt, section by section

The initial E3 prompt (`prompt_e3_worldmodel.md`, sections §0–§11) is the
reference for everything `docs/e3/DECISIONS.md` does not change. **When the
two conflict, DECISIONS.md wins.**

Each section is **unchanged**, **amended** or **under review**. For an
amendment, the entry of DECISIONS.md that makes it is cited by its date and
title. "Not implemented yet" means unchanged but not built: the phase that
builds it is given.

Last updated: 2026-09-25.

| § | Section | Status | Where it changed (DECISIONS.md) |
|---|---|---|---|
| 0 | Context, role, working rules | amended | 2026-09-23 Phase 0 validation, points 4 and 7; 2026-09-25 after the mini run |
| 1 | Phase 0 — reading the repo | amended | 2026-09-23 Phase 0 validation, points 1 and 4 |
| 2 | Pixel environment | amended (§2.2) | 2026-09-23 Phase 0 validation, point 5; 2026-09-23 T2 entries |
| 3 | World model | amended (§3.1–§3.3) | 2026-09-24 and 2026-09-25 entries, see below |
| 4 | Agent | amended (§4.3, §4.5, §4.6) | 2026-09-23 Phase 0 validation, points 1, 2, 8; 2026-09-25 episode starts |
| 5 | Measures | under review (§5.1) | 2026-09-25 after the mini run, note on h0 |
| 6 | Conditions, runs, seeds | amended; one deviation open | 2026-09-25 after the mini run; 2026-09-25 episode starts |
| 7 | Phase 4 — calibration | amended | 2026-09-23 Phase 0 validation, point 8; 2026-09-24 lr 1e-4 failure |
| 8 | Position / appearance diagnostic | amended | 2026-09-25 after the mini run (D1 rule) |
| 9 | Shape tests | amended (T2) | 2026-09-23 T2 reformulated |
| 10 | Logs and outputs | amended; parts not implemented yet | 2026-09-23 Phase 0 validation, point 1; 2026-09-25 after the mini run |
| 11 | Order of the phases | amended (Phase 4 split) | 2026-09-24 and 2026-09-25 entries (Phase 4a / 4b) |

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
  - The batch (16 × 64) is unchanged for the real runs; the mini configs
    use 8 × 16.
- §3.4 replay: unchanged, and made stricter:
  - capacity is checked at startup;
  - a transition is never overwritten;
  - the model batch and the label batch are two separate objects.
  - *Phase 0 validation, adjustment D; 2026-09-24 Phase 3 points 1-2.*
- §3.5 replay ratio: unchanged (calibrated in Phase 4).

### §4 — Agent — amended (§4.3, §4.5, §4.6); not implemented yet (Phase 5)

- §4.1, §4.2, §4.4: unchanged, not implemented yet.
- **§4.3:** the tie-break tolerance is **relative**,
  `tol = max(1e-6, 0.01·std(Q))`. *Phase 0 validation, point 8.*
- **§4.5:** E2 had **no burn-in**; E3 has none either. A minimal prefill
  is played under the same policy as the run it precedes and counted in
  the budget. *Phase 0 validation, point 2.*
- **§4.6 frozen E2 parameters:**
  - The grid is E2's 16 checkpoints. *Phase 0 validation, point 1.*
  - **Uniform starts over the non-terminal states** were specified but
    not implemented until 2026-09-25: every earlier run started at
    (0, 0). *2026-09-25 episode starts.*
  - H = 36, ε = 0.1 (closed loop) and the 10⁶-step budget are unchanged.
    `tests/e3/test_e3_protocol_conformance.py` checks them against E2's
    code.

### §5 — Measures — under review (§5.1)

- **§5.1 canonical probe: under review.**
  - The probe starts from h0, a context under-represented in training:
    ~8.9% of the training steps show h0 with an arbitrary state
    (seq_len 16, uniform starts); the spec's seq_len of 64 would give ~1/64.
  - As a result the canonical template accuracy is lower than in real
    contexts: 0.70–0.87 against 0.99.
  - The long run (`configs/e3/mini_long.yaml`) says whether the gap
    closes. That result is valid for the mini configuration only; it has
    to be measured again with the full configuration in Phase 4a.
  - If the gap does not close, the main measure is redefined in Phase 4a:
    a short standardised context, or replay contexts, based on the
    context dispersion.
  - *2026-09-25 after the mini run (note on h0); 2026-09-25 episode starts.*
- §5.2 I in z-space: unchanged. The default is M = 4096; the configs use
  2000, which DECISIONS.md does not cover (reported, to decide).
- §5.3 s-space kernel: unchanged. It is also used as the bridge for
  `Agent.transition_distribution`. *2026-09-24 Phase 3 points 1-2.*
- §5.4–§5.7: unchanged.
- The measures are pure numpy functions in `src/helplessness/measures/`
  (the prompt's `measure/`). *Phase 0 validation, adjustment B.*

### §6 — Conditions, runs, seeds — amended; one deviation open

- **Dedicated generators.**
  - Latent draws of the training step: a torch generator.
  - Each measurement: its own streams.
  - Episode starts: E2's `start_state` stream, by import.
  - Measuring more or less often never changes the training trajectory
    (tested).
  - *2026-09-25 after the mini run (RNG streams); 2026-09-25 episode starts.*
- **Open deviation, reported, not corrected:** the collection actions,
  the environment dynamics and the themes are seeded with the same
  integer. That gives identical bit streams, where the spec and E2 derive
  them by `SeedSequence(seed).spawn`. *2026-09-25 episode starts, other
  deviations, point 1.* Pinned by a strict xfail in the conformance test.
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
- **Not implemented yet:**
  - model weights per checkpoint and `--resume` (Phase 4a, B5);
  - `transitions.parquet`;
  - permutation, themes and block class A/B in meta.yaml (reported
    2026-09-25);
  - aggregation scripts comparable to E2's.

### §11 — Order of the phases — amended

- P0–P3 are done. **P4 is split**:
  - 4a: calibration code, periodic checkpoints and the GPU-readiness
    items, run locally;
  - 4b: calibration runs on the remote GPU machine, after the GPU
    determinism policy is set.
  - *2026-09-24 and 2026-09-25 entries (notes for Phase 4a / 4b).*
- A long diagnostic run on the mini configuration (`mini_long.yaml`)
  comes before 4a. *2026-09-25 after the mini run.*
- P5, P6 and the CLAUDE.md update at the end of P6: unchanged.
