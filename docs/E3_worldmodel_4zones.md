# E3 — World model validation, 4 zones

**Status (2026-10-14): campaign gpu_campaign analysed; E3's analysis closed.** Results section at the end of this file
(b compatible with 0 in the controllable zones with bounds of ~0.2-0.3 bit; the hybrid-code rule not satisfied in CN
and CD; budget unchanged at 300,000; seed 1 environment an exploratory observation). docs/e3/DECISIONS.md (31)-(36).

**Status (2026-10-13): gpu_campaign running; its analysis pipeline ready and committed before any campaign value** --
`experiments/E3_worldmodel_4zones/campaign_report.py --handoff handoff/gpu_campaign` (integrity, analyses, RSA,
figures, results section appended at the end of this file); definitions and rules in docs/e3/DECISIONS.md (31).

**Status (2026-10-07): the campaign's final batch gpu_campaign ready, not launched** -- N = 4 seeds: the pilot's runs (seeds 0, 5) measured again with the campaign's protocol, and 4 new closed-loop runs (seeds 1, 6; environment and reference); measurement protocol x4 and budget 300,000 retained from the pilot (gpu_pilot_cl, gpu_pilot_remeasure, gpu_pilot_noise); full measures at every checkpoint after the switch; analyses recorded before the launch (`campaign_analysis.py`). Details: `docs/e3/DECISIONS.md`, 2026-10-07 (28), (29), (30); instructions: `experiments/E3_worldmodel_4zones/RUN_GPU.md`.

**Status (2026-10-01, later): agent_mini (closed loop, CPU) recorded as FAILED (shapes 3 and 4 contradicted); read-only diagnostics: the goal head separates perfectly, the world model's prior never predicts the goal, so r_hat = -1 everywhere and the policy follows critic noise. Closed-loop GPU pilot suspended; options pending.** Details: `docs/e3/DECISIONS.md`, 2026-10-01 (11).

**Status (2026-10-01): measurement batch gpu_e30b_measures ready, not launched** -- the four win64 runs of gpu_e30b measured again on the A100 from their own weights (no training), for I_s projected and the reading rule's thresholds (`tau_gpu_e30b.py`); reading rule by zone type and one execution mode per campaign decided; agent_mini (closed loop, CPU) run, analysis deferred. Details: `docs/e3/DECISIONS.md`, 2026-09-30 (9) and 2026-10-01 (10); instructions: `experiments/E3_worldmodel_4zones/RUN_GPU.md`.

**Status (2026-09-30, decisions after gpu_e30b): window shape 16 x 64 retained; C-b kept, hybrid-codes test ready (`hybrid_codes_check.py`, waiting for three runs' weights); off-support mass and projected I_s added to the measures; Phase 5 started on the CPU (agent heads, policy, imagination, closed loop with twin B: `run_agent.py`, `configs/e3/agent_mini.yaml`), first version tested, no full run yet.** Details: `docs/e3/DECISIONS.md`, 2026-09-30 (5).

**Status (2026-09-30, after the second GPU batch): gpu_e30b analysed, 8 / 8 runs done, all separate; calibration C-a and C-c met, C-b met in the deterministic zones and not in the noisy ones (8 / 8 runs); I_s in the controllable-noisy zone above the truth by about 0.29 bits; window shape (16 x 64 vs 64 x 16) not decided.** Analysis: `reports/E3_GPU_gpu_e30b_analysis_2026-09-30.md`; decisions pending in `docs/e3/DECISIONS.md`, 2026-09-30 (4).

**Status (2026-09-30): main measure I_s (controllability of the decoded s-space kernel, real contexts), I_z secondary; second GPU batch gpu_e30b ready, not launched** (bridge network, 1,024 images per step as 16 x 64 and 64 x 16 windows, seeds 0 and 5, experimental and reference: 8 runs). Objectives fixed before the launch, b aggregation (`aggregate_b.py`), h -> s probe: `docs/e3/DECISIONS.md`, 2026-09-30.

**Status (2026-09-29): first GPU batch analysed, no run relaunched.** Outcomes recorded as they are (1 done, 2 separation failures, 2 out-of-memory). Bridge final measures recovered on CPU: C-a at the threshold, C-c met, C-b met in the deterministic zones and not in the noisy ones (also at the corrected K_s = 1,358, now in every config). Seed 0 reference: T4 0.92 / 0.90, I_z above the s-space value in the noisy zones (+0.49 / +0.31 bits). Engineering for the next batch: memory-bounded probes, preflight memory test, run subsets and --max-parallel, code content hash. Details: `docs/e3/DECISIONS.md`, 2026-09-29.

**Status (2026-09-28, 22:44 UTC): GPU batch finished; seed 0 reference completed and produced its final report. Two main runs stopped at the separation deadline; seed 5 experimental and the bridge exited with CUDA OOM (bridge training itself completed).** Main measure in real contexts. No agent policy yet (Phase 5). Collaborator handoff: `reports/E3_GPU_2026-09-28.md`. Details: `docs/e3/SPEC_STATUS.md`, `docs/e3/DECISIONS.md`, `experiments/E3_worldmodel_4zones/README.md`.

**GPU preflight (2026-09-28, 15:39-15:48 UTC): passed.**
UCL ARC Kubeflow TrainJob `bc7e34cec174`, NVIDIA A100-SXM4-80GB,
PyTorch 2.10.0+cu128. CUDA placement, five-step determinism, the 200-gradient-step
smoke run and exact recomputation of the saved final arrays passed.
The main and bridge configurations are unchanged; the job requests 3 CPUs,
64 GiB RAM and one GPU. With all five processes sharing those resources, projected
durations are 9.7 hours per main run and 3.1 hours for the bridge (estimates, not
completed-run timings). This exceeds the original six-hour target; the budget
was not reduced.
Persistent log:
`/scratch/helplessness-worldmodels/logs/kubeflow/preflight-20260928T153952251521Z/preflight.log`.
Submission metadata and child status are saved alongside the log.
An earlier setup-only attempt (`w559ed7b7be3`) failed because the image lacks
`ensurepip`; its log was retained and the job removed. The wrapper now uses
the image's pip to populate its isolated virtual environment without `ensurepip`.
W&B is disabled for preflight. See `experiments/E3_worldmodel_4zones/RUN_GPU.md`
for submission, supervision and resume instructions.

**Full GPU batch:** TrainJob `m3e9461fad23`, submitted at 15:48 UTC after
preflight passed; all five workers confirmed to have reached gradient training.
Four `gpu_e30.yaml` runs (seeds 0 and 5, experimental and
reference) plus `gpu_e30_bridge.yaml` (seed 0, experimental), 60,000 gradient
steps each, on one GPU. W&B is disabled at the user's request. Supervisor logs,
submission metadata and per-process status are under
`/scratch/helplessness-worldmodels/logs/kubeflow/train-20260928T154816605454Z`
(submission JSON is adjacent). Per-run outputs and checkpoints persist under
`/scratch/helplessness-worldmodels/results/E3_worldmodel_4zones/`.

**Full-batch update (16:51 UTC):** seed 5 experimental exited at 16:02 UTC
with CUDA out-of-memory during `real_context_probe` at the first diagnostic
checkpoint (600 gradient steps). This is an execution failure, not a scientific
training guard. `weights_step00000600.pt` exists, but there is no `resume.pt`
or `latest.json`; exact full-state resume is not available for this run.
The other three main runs remain active (latest logged progress around
3,786-4,000 / 60,000 steps), as does the bridge (14,000 / 60,000).
The bridge has switched to straight-through training; the main runs have not
yet reached the separation criterion. GPU memory use at inspection was
80,040 MiB and utilization 92%. The preflight's concurrent throughput benchmark
did not establish safety for simultaneous diagnostic memory peaks. No run was
restarted and no live configuration was changed.

**Full-batch update (21:37 UTC):** only seed 0 reference remains active,
latest logged step 47,000 / 60,000, in the straight-through phase. Seed 0
experimental and seed 5 reference stopped at the 40,000-step separation
deadline (`SeparationNotReachedError`) at 20:54 and 20:53 UTC respectively:
their batch-pair JS remained near zero. Both saved failure checkpoints and
diagnostics; these are scientific guard outcomes, not infrastructure failures.
The bridge completed all 60,000 training steps in 9,638.6 seconds, then
exited at 18:33 UTC with CUDA OOM in the final z-only probe. Its
`weights_final.pt`, `resume.pt`, replay and world remain on disk, but its final
measurements/report did not complete (`results.json` is absent). Seed 5
experimental remains stopped at its earlier OOM. No failed run was restarted.
With only one worker left, GPU memory use was 29,701 MiB and utilization 38%.

**Final batch outcome (22:44 UTC):** seed 0 reference finished all 60,000
gradient steps (training time 24,275.8 s), completed final measurements and
produced `REPORT.md`, `results.json` and `arrays.npz`. Its run status is `done`;
the other four remain `failed` as described above. Supervisor exit codes are
0 for seed 0 reference and 1 for each other worker. TrainJob `m3e9461fad23`
is marked Failed because its supervisor propagates worker failures, not because
all five runs failed. The local checkout now holds copies of all five run
directories and the batch logs; see `reports/E3_GPU_2026-09-28.md` for
paths, scientific findings and recovery options.

## What it tests

Whether a controllability axis emerges in a recurrent world model's latent
representation and is decodable, on the same 4-zone environment as E1/E2
(pixel version) -- the first step of the neural phase (see CLAUDE.md
roadmap step 2). Gated on E1 and E2 producing clean results first.

## Environment

Same 4 zones and dynamics as E1/E2, `PixelObservationBackend`
(`helplessness.envs.observation.pixel_backend.PixelObservationBackend`)
instead of the tabular index backend -- same underlying dynamics, only the
observation changes (see `helplessness.envs.torus_env.TorusZonedEnv`).

## Agent

Does not exist yet: `helplessness.agents.worldmodel/` (or equivalent) is
reserved but empty. Whatever gets built there must implement
`helplessness.agents.base.Agent` (specifically `transition_distribution`)
so `helplessness.measures.controllability.controllability_from_tensor` --
completely unchanged -- can be run on its belief exactly as it already
runs on the tabular agents' T_hat (see
`helplessness.agents.base.materialize_transition_tensor` and
`tests/agents/test_base.py`). How a world model actually produces a
comparable transition distribution over a large or continuous latent
space is an open design question for whoever builds this agent, not
something this restructuring resolves in advance.

## Configuration

Not yet defined.

## Caveat (from CLAUDE.md)

A successful linear decode from the latent does not prove the information
is used. Only intervention (E6) proves that.

<!-- e3-campaign-results:start -->

## Results of the campaign (gpu_campaign, N = 4 seeds)

Seeds 1 and 6 are new closed-loop runs; seeds 0 and 5 are the pilot's runs measured again with the campaign's protocol
(x4 contexts, every measure at every checkpoint after the switch). Each seed has an environment run and a paired
reference run (the uncontrollable zones made controllable). Budget 300,000 environment steps. Every rule, threshold and
expected shape was recorded in docs/e3/DECISIONS.md before the data were read ((16), (29)-(32)). Corrections made after a
result are recorded as such ((33), (35)), and so is the exploratory analysis of seed 1 ((34)-(36)). With N = 4 every
statement is descriptive. A null result is reported as a bound (the largest end of the 95 % Student-t interval over the
seeds), not as the absence of an effect. Pipeline: `experiments/E3_worldmodel_4zones/campaign_report.py`. Figures:
`docs/e3/campaign_figures/`.

### Integrity

All 8 jobs finished with exit code 0, status done, 300,000 environment steps and complete archives. The code hash was
d84dec84 for every job, and the training code of the pilot runs was 3a71c745. There were no interruptions. Switches
occurred between environment steps 8,175 and 17,008 (seed 6 environment, the latest). The pilot runs' final replay is the
one at 290,000 (flagged).

### Main result: b by zone at 300,000

| zone | b = I_s(env) - I_s(ref), mean [95 %] | bound of a null | alpha_env - alpha_ref [95 %] |
|---|---|---|---|
| UD (uncontrollable, deterministic) | -2.258 [-2.289, -2.228] | | not defined |
| UN (uncontrollable, noisy) | -1.779 [-1.951, -1.607] | | not defined |
| CN (controllable, noisy) | -0.011 [-0.206, +0.183] | 0.21 bit | -0.003 [-0.086, +0.080] |
| CD (controllable, deterministic) | -0.038 [-0.295, +0.218] | 0.29 bit | +0.011 [-0.077, +0.100] |
| corridor | -0.050 [-0.208, +0.108] | 0.21 bit | +0.022 [-0.042, +0.087] |

- In the controllable zones, b and alpha_env - alpha_ref are compatible with 0 at the budget, as expected ((29), (31)).
  No effect of the uncontrollable zones on the controllable ones larger than ~0.2-0.3 bit is detected.
- **The hybrid-code rule (29) is not satisfied**: in CN and CD the 95 % intervals of b(I_s) - b(I_s projected) and of
  the off-support difference exceed +- 0.01 bit. b is therefore **not interpretable** in these zones under the rule
  fixed before the campaign.
- E3-0's aberration control flags 12 values (none excluded):
  - in all 4 seeds, the environment's off-support mass is below E3-0's intervals in UD and UN, as already in the pilot
    ((22));
  - in 3 seeds, |off-support env - ref| in CD exceeds tau_off;
  - in seed 6, |b(I_s) - b(I_s projected)| in CD exceeds tau_b (0.029 against 0.0285).

### Learning and behaviour

At the budget every run is successful and its goal is learnt:
- success over the last interval 0.999-1.000;
- median prior mass on the goal 0.997-1.000;
- Q gaps next to the goal 6.6-15.7, far above the tie tolerance;
- C-a >= 0.996 everywhere;
- C-c >= 0.996 everywhere except seed 1 environment (0.976).

The tabular twin's L1 falls in every zone. C-b holds in UD in all runs and fails in the noisy zones in all runs, as in
calibration. In CD and the corridor it fails in some runs (seed 1 environment in both zones; seed 6 reference in the
corridor). The prior's mass on the true successor at 300,000 is 0.94-0.99 in the deterministic zones, lower in seed 1
environment's CD (0.91).

### Measurement error in the noisy zones

In CN, I_s exceeds the true kernel's controllability by +0.211 (sd 0.072) in the environment and +0.222 (0.073) in the
reference at 300,000. The excess was negative early (-0.83 / -0.95 at 25,119) and grows with training. The resolution a
four-seed extension to the noisy zones would have on b (95 % half-width) is **0.19 bit in CN and 0.17 bit in UN** at
300,000.

### Secondary analyses of propagation

- **The early CN transient is a visitation effect, not a propagation.** b in CN is positive at 39,811 (+0.136 [+0.091,
  +0.180]) and alpha_env - alpha_ref is negative at 25,119 and 39,811. This goes opposite to a propagation of
  uncontrollability, which would give b < 0. At equal visits b in CN is ~0 (b0 = -0.000 [-0.198, +0.197]). Early on, CN
  receives a larger share of the visits in the environment than in the reference in 3 of 4 seeds (seed 1: 0.27 and 0.29
  of all visits at 25,119 and 39,811, against 0.10 and 0.11). (Corrected reading, (34)-(35).)
- **b at equal visits** (training time as a covariate): b0 is compatible with 0 in CN, CD and the corridor (bounds 0.20,
  0.54, 0.38). The training-time term is positive in every zone (b1 = +0.75 to +1.03 per decade of the checkpoint).
- **b against the distance to the nearest uncontrollable zone** (main zone CD): the propagation shape is **absent at
  every checkpoint**. In the secondary zones the rule's "present" verdict occurs in CN at 100,000 and 300,000 and in the
  corridor at 100,000.
- **Useful generalisation** (world model against twin at n <= 10; "negative in all 4 seeds" is a one-sided sign test at
  p = 1/16):
  - the world model does not generalise better than the twin in UD, UN and the corridor of the environment up to 63,096;
  - every other cell is not testable (fewer than 5 such pairs in some run), including every reference cell.

### Convergence of the deterministic zones: budget of the next runs

**No candidate checkpoint satisfies the criterion of (31) E, so the budget of the next runs stays at 300,000.**
- Agent learning (a): fails at every candidate because of seed 1 environment.
- Stability of I_s in UD, CD and the corridor (Bonferroni-corrected thresholds, 0.007-0.011 bit): fails up to
  251,189 -> 300,000, mostly in seed 1 environment.

### RSA of the latent

- At the state level, the model distances are explained modestly by the theoretical matrices (R^2 at 300,000: h 0.32 env
  / 0.54 ref; z 0.12 / 0.17), mostly through the position on the torus (unique share h 0.24 / 0.42, z 0.08 / 0.13).
- The controllability matrix's unique share is 0.001-0.005.
- The predictors are weakly collinear (VIF <= 1.8) but **theme, position and zone are confounded in E3** (each theme
  occupies a fixed region). E3 cannot separate theme from position; this is the object of the next step.
- No regression weight is read in isolation.
- The confusion ratio between controllable and uncontrollable zones (by appearance, (33)): on h, env - ref is negative
  from 39,811 to 251,189 (interval excluding 0), -0.086 [-0.187, +0.016] at 300,000; on z it is compatible with 0 at
  every checkpoint. The CD vs UD and CN vs UN ratios are compatible with 0 everywhere.

### Seed 1 (environment): an exploratory observation

Seed 1's environment run learns CD much more slowly than the other runs: the prior's mass on the true successor goes
from 0.28 (15,849) to 0.91 (300,000), against >= 0.97 from 63,096 in seed 6. b(CD) = -0.27 at 300,000, against -0.04 to
+0.10 in the other seeds. This run alone makes the convergence criterion fail. The following diagnostics are exploratory
and post hoc (b(CD) had been seen), each with its rule recorded before computing it.
- (34) "Helplessness" versus "failed learning" signature: **undetermined**. alpha in CD exceeds the other runs and its
  reference (+0.091 > 0.05), but the errors are not concentrated near the boundary (rho(error, distance) = +0.22) and
  they are concentrated on little-visited states (rho(error, n) = -0.55).
- **Design flaw of (34)'s rule, recorded after the result ((35))**: a high alpha is the normal signature of early
  learning in every run (an action-independent prediction before the actions' effects are learnt). So alpha compared at
  the same checkpoint cannot distinguish a delay of learning from a belief of helplessness.
- (35) alpha against the visits n(s):
  - per state, seed 1's alpha stays above the other runs' range at equal visits in 3 of 4 qualifying bins: **"effect
    proper to this model"**;
  - on zone means, the rule's verdict is undetermined.
  - Over its first ~25,000 steps seed 1 environment gave the smallest share of its visits to CD (0.14, against 0.20-0.34)
    and the largest to CN (0.27, against 0.10-0.18).
- (36) Timing of the visits (alpha regressed on log10 n and on the mean step of the state's visits):
  - the timing coefficient is **negative** (-0.172 per 100,000 steps [-0.207, -0.134]): later visits go with a smaller
    alpha, opposite to the loss-of-plasticity hypothesis;
  - seed 1's mean residual stays outside the other runs' range with or without the term (+0.160 without, +0.180 with,
    against [-0.091, +0.013]);
  - **verdict: the gap persists; it is proper to this model, with no explanation identified.**

Candidate explanations, as they stand after these tests:
- a delay of learning due to few early visits to CD: consistent with the visit shares, but it does not account for the
  per-state gap at equal visits;
- a loss of plasticity or primacy bias: not supported, the timing term has the opposite sign;
- a helplessness-like belief propagated from the uncontrollable zones: not supported by the spatial pattern of the
  errors;
- a property of this run's training trajectory (initialisation, optimisation) not captured by visits or timing: not
  tested.

### Limits

- **N = 4**: every result is descriptive. The bounds on b in the controllable zones are wide, about **0.2 to 0.3 bit**,
  roughly ten times E2's effect in CN (0.02 bit). E3 can only exclude effects larger than these bounds.
- **The hybrid-code rule (29) is not satisfied** in CN and CD: b is not interpretable there under the rule fixed before
  the campaign.
- **The budget is unchanged (300,000)**: the deterministic zones did not converge by the criterion of (31) E.
- The pilot runs' last checkpoint rests on the replay at 290,000.
- The confusion ratio of the RSA was corrected after the first run of the pipeline ((33)).
- The seed 1 analyses are exploratory and post hoc.
- In E3, theme, position and zone are confounded.

### Detailed tables (generated by campaign_report.py, second run after the correction of (33))


Generated 2026-10-14 13:14 UTC. Rules and definitions: docs/e3/DECISIONS.md (16), (29), (30), (31), recorded before the data. With N = 4 everything is descriptive; a null result is a bound, not an absence.

#### Integrity of the handoff

`handoff\gpu_campaign`: **passed**.

| job | seed | condition | code (source_sha256) | switch (env step) | checkpoints measured | final replay at | interruptions |
|---|---|---|---|---|---|---|---|
| seed1_experimental | 1 | experimental | d84dec84a3bd | 10092 | 8 | 300000 | none |
| seed1_reference | 1 | reference | d84dec84a3bd | 8592 | 9 | 300000 | none |
| seed6_experimental | 6 | experimental | d84dec84a3bd | 17008 | 7 | 300000 | none |
| seed6_reference | 6 | reference | d84dec84a3bd | 8175 | 9 | 300000 | none |
| pilot_win64_seed0_experimental | 0 | experimental | d84dec84a3bd (training 3a71c7453c4a) | 11342 | 8 | 290000 (FLAGGED: last resume point) | none |
| pilot_win64_seed0_reference | 0 | reference | d84dec84a3bd (training 3a71c7453c4a) | 10758 | 8 | 290000 (FLAGGED: last resume point) | none |
| pilot_win64_seed5_experimental | 5 | experimental | d84dec84a3bd (training 3a71c7453c4a) | 8758 | 9 | 290000 (FLAGGED: last resume point) | none |
| pilot_win64_seed5_reference | 5 | reference | d84dec84a3bd (training 3a71c7453c4a) | 10342 | 8 | 290000 (FLAGGED: last resume point) | none |


#### Main: b by zone at the budget (300,000), hybrid-code rule (29), aberration control (16)

Hybrid-code rule: **applied** (bound +-0.0100 bit).

| zone | b(I_s) - b(I_s projected), mean [95 %] | off-support env - ref, mean [95 %] | verdict |
|---|---|---|---|
| CN | -0.0027 [-0.0209, +0.0156] | -0.0040 [-0.0249, +0.0170] | NOT interpretable |
| CD | -0.0082 [-0.0395, +0.0230] | +0.0026 [-0.0346, +0.0399] | NOT interpretable |

Aberration flags (to be explained, never exclusions): 12:

- seed 1, UD: off-support (env) 0.0224 outside [0.0297, 0.0770]
- seed 1, UN: off-support (env) 0.0589 outside [0.1207, 0.1787]
- seed 1, CD: |off-support env - ref| = 0.0362 > 0.0035
- seed 6, UD: off-support (env) 0.0067 outside [0.0297, 0.0770]
- seed 6, UN: off-support (env) 0.0382 outside [0.1207, 0.1787]
- seed 6, CD: |b(I_s) - b(I_s projected)| = 0.0293 > 0.0285
- seed 6, CD: |off-support env - ref| = 0.0179 > 0.0035
- seed 0, UD: off-support (env) 0.0116 outside [0.0297, 0.0770]
- seed 0, UN: off-support (env) 0.0516 outside [0.1207, 0.1787]
- seed 5, UD: off-support (env) 0.0069 outside [0.0297, 0.0770]
- seed 5, UN: off-support (env) 0.0501 outside [0.1207, 0.1787]
- seed 5, CD: |off-support env - ref| = 0.0060 > 0.0035

| zone | b(I_s) at 300,000, mean [95 %] over the seeds | bound of a null | alpha_env - alpha_ref [95 %] |
|---|---|---|---|
| UD | -2.2582 [-2.2889, -2.2275] | +2.2889 | not defined |
| CN | -0.0114 [-0.2055, +0.1827] | +0.2055 | -0.0029 [-0.0858, +0.0800] |
| UN | -1.7788 [-1.9508, -1.6068] | +1.9508 | not defined |
| CD | -0.0382 [-0.2949, +0.2184] | +0.2949 | +0.0114 [-0.0769, +0.0997] |
| corridor | -0.0501 [-0.2084, +0.1081] | +0.2084 | +0.0222 [-0.0421, +0.0866] |

**Expectation (29)/(31): in the controllable zones b ~ 0 and alpha_env - alpha_ref ~ 0 at the budget; a transient early effect possible.**

- CN: b compatible with 0 (bound +0.2055 bit); alpha difference compatible with 0 (bound +0.0858); earlier checkpoints whose interval excludes 0: [39811]
- CD: b compatible with 0 (bound +0.2949 bit); alpha difference compatible with 0 (bound +0.0997); earlier checkpoints whose interval excludes 0: none
- corridor: b compatible with 0 (bound +0.2084 bit); alpha difference compatible with 0 (bound +0.0866); earlier checkpoints whose interval excludes 0: none

#### b and alpha_env - alpha_ref by zone at every checkpoint measured in every pair (mean [95 %] over the seeds)

| checkpoint | b UD | b CN | b UN | b CD | b corridor | dalpha CN | dalpha CD | dalpha corridor |
|---|---|---|---|---|---|---|---|---|
| 25,119 | -0.7981 [-1.2389, -0.3573] | +0.1231 [-0.1642, +0.4103] | -0.5906 [-0.9541, -0.2270] | +0.0145 [-0.6894, +0.7184] | +0.1703 [-0.2276, +0.5683] | -0.1291 [-0.2294, -0.0289] | -0.0559 [-0.4579, +0.3461] | -0.0997 [-0.2636, +0.0643] |
| 39,811 | -1.5561 [-1.8291, -1.2830] | +0.1355 [+0.0911, +0.1798] | -1.0616 [-1.4342, -0.6891] | -0.1948 [-1.0971, +0.7076] | -0.1296 [-0.4371, +0.1780] | -0.1247 [-0.1571, -0.0922] | +0.0816 [-0.3614, +0.5247] | +0.0729 [-0.0455, +0.1913] |
| 63,096 | -1.9984 [-2.1681, -1.8287] | +0.0010 [-0.1127, +0.1147] | -1.4664 [-1.8506, -1.0823] | -0.2449 [-1.1749, +0.6851] | -0.2349 [-0.6499, +0.1800] | -0.0289 [-0.0639, +0.0060] | +0.1143 [-0.3234, +0.5520] | +0.1287 [-0.0592, +0.3165] |
| 100,000 | -2.1425 [-2.2565, -2.0286] | -0.0417 [-0.2033, +0.1199] | -1.6541 [-1.8774, -1.4309] | -0.2152 [-1.0600, +0.6295] | -0.1984 [-0.6113, +0.2144] | +0.0069 [-0.0742, +0.0880] | +0.0990 [-0.2762, +0.4743] | +0.1058 [-0.0993, +0.3109] |
| 158,489 | -2.2188 [-2.2577, -2.1799] | -0.0329 [-0.2058, +0.1400] | -1.7326 [-1.9284, -1.5368] | -0.1554 [-0.7998, +0.4889] | -0.1601 [-0.5364, +0.2161] | +0.0079 [-0.0667, +0.0824] | +0.0650 [-0.2008, +0.3307] | +0.0771 [-0.1023, +0.2566] |
| 251,189 | -2.2509 [-2.2727, -2.2292] | -0.0193 [-0.2053, +0.1666] | -1.7677 [-1.9445, -1.5909] | -0.0732 [-0.4535, +0.3070] | -0.0709 [-0.2804, +0.1385] | +0.0007 [-0.0811, +0.0825] | +0.0271 [-0.1116, +0.1659] | +0.0337 [-0.0564, +0.1237] |
| 300,000 | -2.2582 [-2.2889, -2.2275] | -0.0114 [-0.2055, +0.1827] | -1.7788 [-1.9508, -1.6068] | -0.0382 [-0.2949, +0.2184] | -0.0501 [-0.2084, +0.1081] | -0.0029 [-0.0858, +0.0800] | +0.0114 [-0.0769, +0.0997] | +0.0222 [-0.0421, +0.0866] |

#### Propagation (31) D

**b at equal visits** (per state, interpolated in log n; regression d = b0 + b1 dtau + b2 log n, dtau the difference of training time at equal visits; b0: b at equal visits and equal training time):

| zone | b0, mean [95 %] over seeds | b1 (dtau) | states used per seed |
|---|---|---|---|
| CN | -0.0004 [-0.1981, +0.1974] | +0.7502 [+0.4282, +1.0722] | [24, 18, 21, 25] |
| CD | -0.0393 [-0.5392, +0.4605] | +1.0263 [+0.6390, +1.4135] | [25, 18, 25, 25] |
| corridor | -0.0440 [-0.3755, +0.2875] | +0.9671 [+0.7736, +1.1606] | [36, 29, 30, 35] |

**b against the distance to the nearest uncontrollable zone** (Spearman rho over the zone's states; main zone CD). Expected shape of a propagation by position, recorded before: |b| larger near the boundary, decreasing with the distance, in the environment only, rather early after the switch. The thresholds (95 % t intervals over 4 seeds) are descriptive.

| checkpoint | zone | rho(abs b, distance) | rho(env error) | rho(ref error) | verdict |
|---|---|---|---|---|---|
| 25,119 | CN | -0.167 [-0.722, +0.389] | -0.145 [-0.435, +0.144] | -0.240 [-0.466, -0.014] | absent (rho(abs b, distance) compatible with 0) |
| 25,119 | CD (main) | -0.032 [-0.496, +0.433] | -0.215 [-0.462, +0.032] | -0.357 [-0.645, -0.068] | absent (rho(abs b, distance) compatible with 0) |
| 25,119 | corridor | +0.125 [-0.050, +0.301] | -0.064 [-0.176, +0.047] | -0.166 [-0.368, +0.037] | absent (rho(abs b, distance) compatible with 0) |
| 39,811 | CN | -0.183 [-0.546, +0.180] | -0.309 [-0.838, +0.220] | +0.021 [-0.376, +0.418] | absent (rho(abs b, distance) compatible with 0) |
| 39,811 | CD (main) | -0.100 [-0.704, +0.505] | -0.151 [-0.292, -0.010] | -0.189 [-0.559, +0.181] | absent (rho(abs b, distance) compatible with 0) |
| 39,811 | corridor | -0.033 [-0.179, +0.113] | -0.127 [-0.188, -0.066] | -0.123 [-0.368, +0.121] | absent (rho(abs b, distance) compatible with 0) |
| 63,096 | CN | -0.159 [-0.704, +0.387] | -0.314 [-0.619, -0.010] | -0.031 [-0.383, +0.321] | absent (rho(abs b, distance) compatible with 0) |
| 63,096 | CD (main) | -0.090 [-0.560, +0.381] | -0.185 [-0.348, -0.021] | -0.179 [-0.468, +0.109] | absent (rho(abs b, distance) compatible with 0) |
| 63,096 | corridor | -0.088 [-0.242, +0.065] | -0.169 [-0.296, -0.042] | -0.055 [-0.286, +0.176] | absent (rho(abs b, distance) compatible with 0) |
| 100,000 | CN | -0.269 [-0.497, -0.042] | -0.254 [-0.475, -0.033] | -0.029 [-0.583, +0.525] | propagation shape present |
| 100,000 | CD (main) | -0.126 [-0.418, +0.167] | -0.078 [-0.319, +0.164] | -0.172 [-0.235, -0.108] | absent (rho(abs b, distance) compatible with 0) |
| 100,000 | corridor | -0.092 [-0.174, -0.010] | -0.154 [-0.285, -0.024] | -0.020 [-0.210, +0.170] | propagation shape present |
| 158,489 | CN | -0.194 [-0.302, -0.086] | -0.264 [-0.681, +0.154] | -0.138 [-0.867, +0.591] | undetermined |
| 158,489 | CD (main) | -0.207 [-0.605, +0.190] | -0.153 [-0.439, +0.134] | -0.192 [-0.334, -0.050] | absent (rho(abs b, distance) compatible with 0) |
| 158,489 | corridor | -0.134 [-0.322, +0.054] | -0.160 [-0.339, +0.019] | -0.022 [-0.189, +0.145] | absent (rho(abs b, distance) compatible with 0) |
| 251,189 | CN | -0.176 [-0.296, -0.057] | -0.291 [-0.616, +0.033] | -0.203 [-0.633, +0.227] | undetermined |
| 251,189 | CD (main) | -0.068 [-0.544, +0.408] | -0.033 [-0.329, +0.264] | -0.177 [-0.356, +0.002] | absent (rho(abs b, distance) compatible with 0) |
| 251,189 | corridor | -0.035 [-0.213, +0.143] | -0.147 [-0.327, +0.033] | +0.029 [-0.162, +0.221] | absent (rho(abs b, distance) compatible with 0) |
| 300,000 | CN | -0.249 [-0.457, -0.041] | -0.379 [-0.596, -0.163] | -0.181 [-0.668, +0.306] | propagation shape present |
| 300,000 | CD (main) | -0.045 [-0.518, +0.429] | -0.132 [-0.529, +0.264] | -0.162 [-0.439, +0.115] | absent (rho(abs b, distance) compatible with 0) |
| 300,000 | corridor | -0.035 [-0.206, +0.136] | -0.112 [-0.197, -0.028] | +0.018 [-0.194, +0.231] | absent (rho(abs b, distance) compatible with 0) |

**Useful generalisation** (median L1_WM - L1_twin against P over the pairs with n <= 10; rule: negative in every seed). Descriptive: "negative in all 4 seeds" is a sign test at p = 1/16 (one-sided).

| checkpoint | condition | UD | CN | UN | CD | corridor |
|---|---|---|---|---|---|---|
| 25,119 | experimental | does not generalise | not testable | does not generalise | not testable | does not generalise |
| 25,119 | reference | not testable | not testable | not testable | not testable | not testable |
| 39,811 | experimental | does not generalise | not testable | does not generalise | not testable | does not generalise |
| 39,811 | reference | not testable | not testable | not testable | not testable | not testable |
| 63,096 | experimental | does not generalise | not testable | does not generalise | not testable | does not generalise |
| 63,096 | reference | not testable | not testable | not testable | not testable | not testable |
| 100,000 | experimental | mixed | not testable | not testable | not testable | not testable |
| 100,000 | reference | not testable | not testable | not testable | not testable | not testable |
| 158,489 | experimental | not testable | not testable | not testable | not testable | not testable |
| 158,489 | reference | not testable | not testable | not testable | not testable | not testable |
| 251,189 | experimental | not testable | not testable | not testable | not testable | not testable |
| 251,189 | reference | not testable | not testable | not testable | not testable | not testable |
| 300,000 | experimental | not testable | not testable | not testable | not testable | not testable |
| 300,000 | reference | not testable | not testable | not testable | not testable | not testable |

#### Measurement error in the noisy zones (31) C

| checkpoint | zone | excess env, mean (sd) | excess ref, mean (sd) | b, mean (sd over seeds) | resolution (95 % half-width, N = 4) |
|---|---|---|---|---|---|
| 25,119 | CN | -0.8270 (+0.1310) | -0.9501 (+0.2328) | +0.1231 (+0.1805) | +0.2872 |
| 25,119 | UN | +0.1765 (+0.0577) | -0.8859 (+0.2493) | -0.5906 (+0.2285) | +0.3636 |
| 39,811 | CN | -0.3782 (+0.2590) | -0.5137 (+0.2709) | +0.1355 (+0.0279) | +0.0444 |
| 39,811 | UN | +0.1461 (+0.0488) | -0.4452 (+0.2620) | -1.0616 (+0.2341) | +0.3726 |
| 63,096 | CN | -0.0902 (+0.2156) | -0.0912 (+0.2011) | +0.0010 (+0.0714) | +0.1137 |
| 63,096 | UN | +0.1019 (+0.0449) | -0.0846 (+0.2454) | -1.4664 (+0.2414) | +0.3841 |
| 100,000 | CN | +0.0455 (+0.1626) | +0.0872 (+0.1057) | -0.0417 (+0.1016) | +0.1616 |
| 100,000 | UN | +0.0708 (+0.0391) | +0.0720 (+0.1434) | -1.6541 (+0.1403) | +0.2233 |
| 158,489 | CN | +0.1320 (+0.1162) | +0.1649 (+0.0567) | -0.0329 (+0.1087) | +0.1729 |
| 158,489 | UN | +0.0461 (+0.0247) | +0.1258 (+0.1297) | -1.7326 (+0.1230) | +0.1958 |
| 251,189 | CN | +0.1979 (+0.0780) | +0.2172 (+0.0566) | -0.0193 (+0.1169) | +0.1860 |
| 251,189 | UN | +0.0316 (+0.0111) | +0.1463 (+0.1171) | -1.7677 (+0.1111) | +0.1768 |
| 300,000 | CN | +0.2108 (+0.0719) | +0.2222 (+0.0728) | -0.0114 (+0.1220) | +0.1941 |
| 300,000 | UN | +0.0295 (+0.0101) | +0.1554 (+0.1155) | -1.7788 (+0.1081) | +0.1720 |

Hybrid-code rule (29): applied; CN NOT interpretable; CD NOT interpretable

#### Convergence of the deterministic zones (31) E

**no candidate satisfies (a) and (b): the budget stays at 300,000**

sigma by zone (x4): UD 0.0020, CD 0.0021, corridor 0.0016; family-wise alpha 0.05.

| candidate | comparisons | k | thresholds (UD / CD / corridor) | (a) agent finished | (b) I_s stable | first failures |
|---|---|---|---|---|---|---|
| 25,119 | 168 | 3.62 | 0.0100 / 0.0108 / 0.0082 | no | no | seed1_experimental (a) at 25119: goal mass 0.1171398583995144, next interval overlaps False, max Delta 23.671; seed1_experimental (a) at 39811: goal mass 0.48302361447416964, next interval overlaps False, max Delta 17.203; seed1_experimental (a) at 63096: goal mass 0.5452172552166934, next interval overlaps True, max Delta 16.606 ... |
| 39,811 | 144 | 3.58 | 0.0099 / 0.0107 / 0.0082 | no | no | seed1_experimental (a) at 39811: goal mass 0.48302361447416964, next interval overlaps False, max Delta 17.203; seed1_experimental (a) at 63096: goal mass 0.5452172552166934, next interval overlaps True, max Delta 16.606; seed1_experimental (a) at 100000: goal mass 0.5844231328800815, next interval overlaps True, max Delta 14.971 ... |
| 63,096 | 120 | 3.53 | 0.0098 / 0.0106 / 0.0080 | no | no | seed1_experimental (a) at 63096: goal mass 0.5452172552166934, next interval overlaps True, max Delta 16.606; seed1_experimental (a) at 100000: goal mass 0.5844231328800815, next interval overlaps True, max Delta 14.971; seed1_experimental (a) at 158489: goal mass 0.6831374786881959, next interval overlaps True, max Delta 10.113 ... |
| 100,000 | 96 | 3.47 | 0.0096 / 0.0104 / 0.0079 | no | no | seed1_experimental (a) at 100000: goal mass 0.5844231328800815, next interval overlaps True, max Delta 14.971; seed1_experimental (a) at 158489: goal mass 0.6831374786881959, next interval overlaps True, max Delta 10.113; seed1_experimental (b) uncontrollable_deterministic 100000 -> 158489: 0.0862 >= 0.0096 ... |
| 158,489 | 72 | 3.39 | 0.0094 / 0.0101 / 0.0077 | no | no | seed1_experimental (a) at 158489: goal mass 0.6831374786881959, next interval overlaps True, max Delta 10.113; seed1_experimental (b) uncontrollable_deterministic 158489 -> 251189: 0.0409 >= 0.0094; seed1_experimental (b) controllable_deterministic 158489 -> 251189: 0.3688 >= 0.0101 ... |
| 251,189 | 24 | 3.08 | 0.0085 / 0.0092 / 0.0070 | no | no | seed1_experimental (b) controllable_deterministic 251189 -> 300000: 0.1591 >= 0.0092; seed1_experimental (b) corridor 251189 -> 300000: 0.0909 >= 0.0070; seed1_reference (b) corridor 251189 -> 300000: 0.0220 >= 0.0070 ... |

#### Learning and behaviour at the budget (31) B (all checkpoints: tables/learning.csv, figures/curves_learning.png)

| run | C-a | C-c | C-b ok by zone | twin L1 by zone | mass on true successor (UD / CD / corridor) | success last interval [95 %] | max Delta | median goal mass |
|---|---|---|---|---|---|---|---|---|
| seed 1 experimental | 0.998 | 0.976 | UD:ok CN:FAIL UN:FAIL CD:FAIL corridor:FAIL | UD:+0.021 CN:+0.071 UN:+0.068 CD:+0.016 corridor:+0.016 | +0.975 / +0.907 / +0.897 | 0.999 [0.998, 1.000] | +0.78 | +1.000 |
| seed 1 reference | 0.996 | 0.999 | UD:ok CN:FAIL UN:FAIL CD:ok corridor:ok | UD:+0.015 CN:+0.068 UN:+0.070 CD:+0.014 corridor:+0.016 | +0.991 / +0.988 / +0.969 | 1.000 [0.999, 1.000] | +0.29 | +1.000 |
| seed 6 experimental | 1.000 | 0.999 | UD:ok CN:FAIL UN:FAIL CD:ok corridor:ok | UD:+0.019 CN:+0.057 UN:+0.072 CD:+0.010 corridor:+0.014 | +0.992 / +0.996 / +0.954 | 1.000 [0.999, 1.000] | +1.29 | +1.000 |
| seed 6 reference | 0.996 | 0.996 | UD:ok CN:FAIL UN:FAIL CD:ok corridor:FAIL | UD:+0.018 CN:+0.078 UN:+0.078 CD:+0.017 corridor:+0.017 | +0.975 / +0.963 / +0.944 | 1.000 [0.999, 1.000] | +0.22 | +0.997 |
| seed 0 experimental | 1.000 | 0.999 | UD:ok CN:FAIL UN:FAIL CD:ok corridor:ok | UD:+0.019 CN:+0.076 UN:+0.066 CD:+0.014 corridor:+0.018 | +0.984 / +0.993 / +0.951 | 1.000 [0.999, 1.000] | +0.25 | +1.000 |
| seed 0 reference | 0.996 | 0.997 | UD:ok CN:FAIL UN:FAIL CD:ok corridor:ok | UD:+0.015 CN:+0.080 UN:+0.078 CD:+0.015 corridor:+0.018 | +0.987 / +0.991 / +0.959 | 1.000 [0.999, 1.000] | +0.32 | +1.000 |
| seed 5 experimental | 1.000 | 0.999 | UD:ok CN:FAIL UN:FAIL CD:ok corridor:ok | UD:+0.025 CN:+0.068 UN:+0.067 CD:+0.009 corridor:+0.017 | +0.993 / +0.996 / +0.962 | 1.000 [0.999, 1.000] | +0.75 | +1.000 |
| seed 5 reference | 0.998 | 0.999 | UD:ok CN:FAIL UN:FAIL CD:ok corridor:ok | UD:+0.015 CN:+0.062 UN:+0.069 CD:+0.015 corridor:+0.015 | +0.989 / +0.988 / +0.971 | 1.000 [0.999, 1.000] | +0.42 | +1.000 |

#### RSA of the latent (31) F

CPU time per checkpoint: median 24 s (66 checkpoints).

| checkpoint | latent | R^2 env / ref | controllability unique share env | confusion ratio (C vs U) env / ref, env - ref [95 %] | CD vs UD env - ref | CN vs UN env - ref |
|---|---|---|---|---|---|---|
| 25,119 | h | +0.245 / +0.452 | +0.002 | +1.341 / +1.442, -0.101 [-0.257, +0.055] | -0.192 [-0.529, +0.145] | -0.017 [-0.363, +0.329] |
| 25,119 | z | +0.129 / +0.141 | +0.005 | +1.194 / +1.118, +0.076 [-0.058, +0.209] | +0.016 [-0.132, +0.164] | +0.138 [-0.113, +0.389] |
| 39,811 | h | +0.268 / +0.474 | +0.003 | +1.365 / +1.458, -0.093 [-0.150, -0.036] | -0.149 [-0.387, +0.088] | -0.058 [-0.431, +0.316] |
| 39,811 | z | +0.128 / +0.164 | +0.003 | +1.183 / +1.130, +0.053 [-0.050, +0.156] | +0.015 [-0.084, +0.114] | +0.100 [-0.134, +0.334] |
| 63,096 | h | +0.291 / +0.518 | +0.001 | +1.359 / +1.473, -0.114 [-0.159, -0.070] | -0.135 [-0.331, +0.060] | -0.103 [-0.453, +0.247] |
| 63,096 | z | +0.131 / +0.171 | +0.004 | +1.184 / +1.137, +0.048 [-0.046, +0.142] | +0.006 [-0.085, +0.096] | +0.093 [-0.106, +0.293] |
| 100,000 | h | +0.303 / +0.539 | +0.001 | +1.358 / +1.482, -0.124 [-0.174, -0.073] | -0.127 [-0.322, +0.067] | -0.143 [-0.469, +0.184] |
| 100,000 | z | +0.132 / +0.172 | +0.004 | +1.180 / +1.138, +0.043 [-0.043, +0.129] | +0.003 [-0.084, +0.089] | +0.086 [-0.119, +0.291] |
| 158,489 | h | +0.311 / +0.546 | +0.001 | +1.354 / +1.474, -0.120 [-0.192, -0.048] | -0.112 [-0.308, +0.084] | -0.155 [-0.437, +0.126] |
| 158,489 | z | +0.128 / +0.171 | +0.003 | +1.179 / +1.138, +0.040 [-0.042, +0.122] | +0.000 [-0.082, +0.083] | +0.086 [-0.125, +0.298] |
| 251,189 | h | +0.317 / +0.541 | +0.001 | +1.346 / +1.443, -0.097 [-0.189, -0.005] | -0.087 [-0.245, +0.070] | -0.142 [-0.400, +0.116] |
| 251,189 | z | +0.124 / +0.170 | +0.003 | +1.176 / +1.138, +0.038 [-0.039, +0.116] | +0.001 [-0.080, +0.081] | +0.083 [-0.121, +0.286] |
| 300,000 | h | +0.317 / +0.539 | +0.001 | +1.342 / +1.428, -0.086 [-0.187, +0.016] | -0.084 [-0.221, +0.053] | -0.123 [-0.373, +0.126] |
| 300,000 | z | +0.123 / +0.169 | +0.003 | +1.175 / +1.138, +0.038 [-0.034, +0.110] | -0.001 [-0.081, +0.080] | +0.085 [-0.119, +0.289] |

**Partition of the explained variance and collinearity of the predictors** (state level; unique share of each theoretical matrix, shared share, largest variance inflation factor over the seeds, mean correlations between predictors). No regression weight is read in isolation. In E3 theme, position and zone are confounded (each theme occupies one fixed region of the torus): E3 cannot separate theme from position -- the object of the next step (two-zone models on permuted themes and positions).

| checkpoint | latent | run | unique shares | shared | VIF (max) | predictor correlations | removed (constant) |
|---|---|---|---|---|---|---|---|
| 25,119 | h | experimental | controllability +0.002, position +0.184, stochasticity +0.004, theme +0.008 | +0.046 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 25,119 | h | reference | position +0.350, stochasticity +0.006, theme +0.020 | +0.075 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 25,119 | z | experimental | controllability +0.005, position +0.070, stochasticity +0.008, theme +0.011 | +0.035 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 25,119 | z | reference | position +0.097, stochasticity +0.003, theme +0.008 | +0.032 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 39,811 | h | experimental | controllability +0.003, position +0.195, stochasticity +0.005, theme +0.012 | +0.054 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 39,811 | h | reference | position +0.369, stochasticity +0.008, theme +0.023 | +0.075 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 39,811 | z | experimental | controllability +0.003, position +0.076, stochasticity +0.005, theme +0.009 | +0.034 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 39,811 | z | reference | position +0.119, stochasticity +0.003, theme +0.008 | +0.034 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 63,096 | h | experimental | controllability +0.001, position +0.215, stochasticity +0.006, theme +0.011 | +0.059 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 63,096 | h | reference | position +0.397, stochasticity +0.007, theme +0.023 | +0.091 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 63,096 | z | experimental | controllability +0.004, position +0.078, stochasticity +0.006, theme +0.009 | +0.035 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 63,096 | z | reference | position +0.126, stochasticity +0.003, theme +0.007 | +0.035 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 100,000 | h | experimental | controllability +0.001, position +0.224, stochasticity +0.007, theme +0.010 | +0.060 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 100,000 | h | reference | position +0.410, stochasticity +0.006, theme +0.025 | +0.098 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 100,000 | z | experimental | controllability +0.004, position +0.081, stochasticity +0.005, theme +0.009 | +0.033 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 100,000 | z | reference | position +0.127, stochasticity +0.003, theme +0.008 | +0.033 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 158,489 | h | experimental | controllability +0.001, position +0.231, stochasticity +0.008, theme +0.011 | +0.060 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 158,489 | h | reference | position +0.414, stochasticity +0.007, theme +0.026 | +0.099 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 158,489 | z | experimental | controllability +0.003, position +0.079, stochasticity +0.005, theme +0.009 | +0.032 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 158,489 | z | reference | position +0.127, stochasticity +0.004, theme +0.009 | +0.032 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 251,189 | h | experimental | controllability +0.001, position +0.236, stochasticity +0.008, theme +0.012 | +0.060 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 251,189 | h | reference | position +0.415, stochasticity +0.008, theme +0.025 | +0.094 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 251,189 | z | experimental | controllability +0.003, position +0.076, stochasticity +0.005, theme +0.009 | +0.031 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 251,189 | z | reference | position +0.126, stochasticity +0.004, theme +0.009 | +0.030 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 300,000 | h | experimental | controllability +0.001, position +0.235, stochasticity +0.008, theme +0.012 | +0.061 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 300,000 | h | reference | position +0.415, stochasticity +0.008, theme +0.024 | +0.091 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |
| 300,000 | z | experimental | controllability +0.003, position +0.075, stochasticity +0.005, theme +0.009 | +0.031 | controllability 1.4, position 1.2, stochasticity 1.3, theme 1.8 | controllability~stochasticity +0.08, position~controllability -0.02, position~stochasticity +0.26, position~theme +0.31, theme~controllability +0.47, theme~stochasticity +0.47 | - |
| 300,000 | z | reference | position +0.126, stochasticity +0.005, theme +0.009 | +0.029 | position 1.1, stochasticity 1.3, theme 1.3 | position~stochasticity +0.26, position~theme +0.31, theme~stochasticity +0.47 | controllability |

In the reference every zone is controllable: its controllability predictor is constant and removed explicitly before the regression (DECISIONS (32)); its confusion ratios contrast appearances only (the control). The distance on z is the sum over the 8 categorical variables of the per-variable Jensen-Shannon divergences, not the JS divergence of the joint distribution.

#### Figures

- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): maps_025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): maps_100000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): maps_300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): curves_b_alpha.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): curves_learning.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed1_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed1_env300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed1_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed1_env300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed6_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed6_env300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed6_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed6_env300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed0_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed0_env300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed0_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed0_env300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed5_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_experimental_seed5_env300000.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed5_env025119.png`
- `results/E3_worldmodel_4zones/campaign/figures/ (not versioned; key figures copied to docs/e3/campaign_figures/): rsa_reference_seed5_env300000.png`

Tables: `results/E3_worldmodel_4zones/campaign/tables`; full output: `results/E3_worldmodel_4zones/campaign/analysis.json`; per-pair detail: `results/E3_worldmodel_4zones/campaign/CAMPAIGN_ANALYSIS.md`.

<!-- e3-campaign-results:end -->
