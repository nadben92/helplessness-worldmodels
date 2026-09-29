# E3 — World model validation, 4 zones

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
