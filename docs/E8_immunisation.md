# E8 — Immunisation

**Status (2026-10-22): not started.** Moved out of E5 (docs/e5/DECISIONS.md (29)), as an annex experiment: prevention,
beside the therapy.

## What it tests

Whether prior exposure to control protects the belief of controllability (Maier & Seligman): the reference's dynamics
(everything controllable, the training image) until a switch step, then the experimental dynamics; the belief is
measured with E4's probes against the baseline reference of the same seed.

## What exists

E5's implementation and tests (docs/e5/DECISIONS.md (7)-(11)): `experiments/E5_belief_training/run_e5.py --training
immunisation` (the switch at the first episode starting at switch_step, post_steps after it, the run on the reference's
streams, `--branch-from` a resumable state), `pipeline.episode_sources.DynamicsSwitch`, the switch step proposed by
`choose_switch_step.py` (100,000 for p1-b-d1, (27)). Seed B's baseline reference (E5's wave 2) keeps its full state at
100,000 steps for the branch. E5's batches list it as deferred (`batches.DEFERRED["e8_immunisation"]`).
