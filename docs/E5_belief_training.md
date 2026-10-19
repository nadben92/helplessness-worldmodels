# E5 — Manipulating the belief through training

**Status (2026-10-19): code ready and tested on CPU (mini configuration), not launched.** Branch `e5` (from `e4`; `e4` is
merged into it before the launch). The E4 configuration of E5 is set after E4's analysis. Code and usage:
`experiments/E5_belief_training/README.md`; launch: `experiments/E5_belief_training/RUN_GPU.md`; decisions:
`docs/e5/DECISIONS.md`; implementation report: `experiments/E5_belief_training/IMPLEMENTATION_REPORT.md`. Deterministic
dynamics only.

## What it tests

E4's first results (wave 1, extreme probes) suggest that the world model's belief of controllability is carried by the
state's **position**, hardly by its **theme** (b at the first step: training image at the UD slot about -2.31; position
only about -2.2 to -2.3; theme only about -0.01 to -0.29). E5 asks whether a training condition can move that support:

- **shuffle** -- if the position stops predicting controllability during training, does the belief move to the theme?
  At each episode, with probability f (1 or 0.5), the assignment of the zones to the quadrants is redrawn among the 12
  (UD slot, CD slot) pairs; the theme follows the type (T_inc with the conveyor, T_ctrl with CD).
- **immunisation** (Maier & Seligman) -- prior exposure to control: the reference's dynamics (everything controllable,
  the same image) until the switch, then the experimental dynamics.

## Design

One configuration of E4 for all of E5 (`--configuration`), the same rendering, colours and canonical orientation for
every condition. Two seeds: A = E4's seed of the configuration, B = A + 100. Every run has E4's parameters (collection,
epsilon, horizon, windows, latent, learning rates, budget of 300,000 steps); `configs/e5/e5.yaml` reads E4's config whole
(`base`) and adds the `e5` block only.

| Condition | Exposed | Reference | Runs |
|---|---|---|---|
| baseline (E4) | seed A: E4's run, reused; seed B trained | same | 2 (seed B) |
| shuffle f = 1 | assignment redrawn at every episode | same draws, UD -> CD | 4 (A, B) |
| shuffle f = 0.5 | redrawn with probability 0.5, else canonical | same draws, UD -> CD | 4 (A, B) |
| immunisation | reference dynamics until the first episode starting at switch_step, then exposed dynamics for post_steps (175,000) | the baseline reference of the same seed | 2 (A, B) |

Three waves of four (`batches.py`): wave 1 seed A shuffle; wave 2 seed B baseline and shuffle f = 1; wave 3 seed B
shuffle f = 0.5 and the two immunisations (A recomputed: E4 kept no full state; B branched from wave 2's baseline
reference at the switch step).

Shuffle: the assignment stream is the seed's own (`streams/assignment.py`), shared by EPISODE INDEX between a run and its
reference; the conveyor's drift is resolved for its slot; the corridor, the goal, the torus never change; the
neutral slots have the corridor's dynamics and colour. Pairing tested for each of the 12 assignments. f = 0 reproduces
E4 bit for bit (tested). The replay keeps every assignment's images and per-step backgrounds, so the loss's mask stays the
agent's patch.

Immunisation: the run equals the reference of its seed until the switch (tested); the dynamics change at the reset of
the first episode starting once switch_step transitions are done (logged: `dynamics_switch_step`); the replay is not
emptied and the optimiser not reset; its checkpoints are E4's grid in post-switch steps; beta_rep's ramp stays E4's
planned budget. switch_step is proposed by `choose_switch_step.py` (the first checkpoint where the future UD slot of E4's
reference meets E4's thresholds: mass >= 0.97 and |I at the first step - log2 5| <= 0.1 bit).

## Measures

E4's, unchanged (the library's `probing.theme_probe_measure`, `pipeline.two_zones_records`):

1. Probes at the first step and recovery (steps 2-25), the full grid of 93 images, the same walks, frozen weights,
   positions on the canonical layout for every condition; b against the reference of the same condition and seed (the
   immunisation: the baseline reference of its seed at the same number of post-switch steps).
2. The belief in the training world at every checkpoint: E4's record; under shuffle, on the canonical assignment and by
   assignment over the 12 (with their mean by zone type); visits by zone type.
3. RSA (`rsa_e5.py`, E3's `measures.rsa`): unique shares of position, theme and controllability on h and z; under
   shuffle the points are (assignment, state) pairs, which decorrelates the three.
4. Basic checks: the agent's success, the quadrants' mass, the visits by zone type.

## Config, code, results

`configs/e5/`; `experiments/E5_belief_training/`; results under `results/E5_belief_training/` on the cluster's volume,
brought back as one archive per batch (`export_batch.py`).
