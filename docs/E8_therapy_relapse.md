# E8 — Therapy and relapse

**Status (2026-11-13): finished, merged into `main`. Waves 1 (baselines) and 2 (shuffle A, E, F with reference B,
therapy on nw) run and analysed; waves 3-5 (therapy on ne, sw, se) configured, not launched (future work).
Wave 1 (report section 11): control forms held, P1 held, P2 not decidable on the baselines, P3 did not hold. Wave 2
(report section 13): F0 exact (A on its exposed side, DECISIONS (26)), M held; R failed on the untreated assignments
(partial forgetting of the untrained arrangements), H failed for F (assignment 0, 5,000-25,000 steps); P1 held, P2 failed
on (a) and (b), P3 held for F only, P4 "transfer" (3 of 3, read on the 6 arrangements where nw is neutral).** Branch `e8` (from `v0.2-e7`). The user's brief:
`docs/e8/prompt_e8_therapie.md`. Decisions: `docs/e8/DECISIONS.md`. Expected forms and predictions, committed before any
computation: `docs/e8/EXPECTED_FORMS.md`. Report: `experiments/E8_therapy_relapse/IMPLEMENTATION_REPORT.md`.

## Why

E7 showed that the transition belief causes the behaviour (the oracle restores it in all 8 agents), that the
experience leaves two traces (the transition belief and the value cached by the critic: light in the baselines and F,
as heavy as the belief in shuffle A and E), and that editing h corrects the belief only for the step it is applied at.
A lasting correction has to change the weights. E8 re-trains exposed agents in a world that has become controllable.

## Questions

1. Does an exposed agent, re-trained in a controllable world, correct its belief and its behaviour, and how fast?
2. Does behaviour lag behind the belief, and does the lag depend on the weight of the cached value?
3. In the shuffle agents, does a therapy limited to some contexts carry over to the untreated ones (relapse, or
   renewal, on the untreated contexts)?

## Agents (DECISIONS (2), (3))

| Role | Agent | Start | Paired reference (re-trained alike) |
|---|---|---|---|
| baseline | base_A (E4's `p1-b-d1`, seed 2) | 300,000 | base_A's reference (seed 2) |
| baseline | base_B (E5, seed 102) | 300,000 | base_B's reference (seed 102) |
| shuffle | A (seed 2), E (402), F (502) | 650,000 | shuffle reference B (seed 102), one run for the three |

Every start file is checked by its SHA-256 (`experiments/E5_belief_training/analysis_robustness/tables/inventory.csv`)
before a run starts. The shuffle agents are short of E4's convergence criterion at 650,000: their level is declared
(E5 (37)).

## Therapy

Every therapy environment has the **reference's dynamics** (everything controllable, UD -> CD; corridor, goal and torus
unchanged) and shows **only images seen in training** (E5: an unseen colour breaks the localisation).

- **Baselines, full exposure**: the paired reference environment itself (canonical layout, original themes: T_inc still
  in nw, now controllable). For the reference, its own training environment.
- **Shuffle, partial therapy on quadrant q** (`treated_quadrant`, nw in this campaign): each episode draws uniformly one
  of the 3 assignments with T_inc on q (the CD on one of the 3 other quadrants, the theme following the type as in E5).
  The 9 other assignments are never shown. `all` (the 12, the shuffle reference environment) is the full therapy,
  available, not run.
- **Treated zone** (visit counts): nw for the baselines; for the shuffle agents, the quadrant carrying T_inc in the
  episode (q).

Procedure: E5's training procedure through the library's closed loop (`pipeline.agent_run.run_one`), with new options
whose defaults reproduce E3-E7 bit for bit. At the start (DECISIONS (4)): the weights of the world model, goal head,
critic and slow critic loaded from the start file; the replay empty, refilled by E5's prefill (256 steps under the
agent's own policy, no gradient step); the three optimisers new; the schedule post-switch from step 0
(straight-through, beta_rep at its full value, the post-switch learning rate), the switch unable to fire again.
Policy, starts, epsilon, truncation, gamma, reward and the critic's training in imagination are E5's.
`start_in_zone_frac` (default 0) is a parameter, unchanged here. Budget `max_env_steps`: 150,000 in wave 1 (a pilot);
wave 2's set after wave 1. Full checkpoints (weights and resume state) on a log-spaced grid (0, 500, 1,000, 2,000,
3,500, 5,000, 7,500, 10,000, 15,000, 25,000, 40,000, 60,000, 100,000, 150,000); checkpoint 0 before any update.
Every step is logged (state, quadrant, assignment), so recovery is read against environment steps, gradient steps and
visits of the treated zone.

## Measures at every therapy checkpoint (frozen weights, `--phase probes`)

In the test environment (the reference's dynamics), on the exposed run and on its reference at the same checkpoint:
baselines on the original image; shuffle agents on the 12 assignment images, split into the 3 treated (T_inc on q) and
the 9 untreated (the relapse test). Zone of measurement: the quadrant carrying T_inc in the image.

1. b at step 1: E5's 12-assignment protocol (and E4's probe for the original image), E4's validity criterion applied
   at every checkpoint; invalid points reported, never dropped silently.
2. C1 (the controllability the critic reads) and C3 (the planner's greedy choice), as in E7's 650,000 campaign.
3. Behaviour: E6's protocol (9 interior starts of the zone, initial h, epsilon 0.1, H = 36, the training policy, k = 1):
   directed share (main), recovery latency and zone share (secondary), with these episodes' b at step 1.
4. Localisation, judged by the decoder (E4/E6's rule, with the quadrant the agent is drawn in when mislocalised) and
   from the true cell (the directed share computed from the true cell, as for F in E7).
5. At some checkpoints (0, 2,000, 10,000, 40,000 and the last): the oracle at k = 1 (true transitions in the planner,
   the agent's critic) on the training image, for the decomposition: belief effect = oracle k=1 - exposed, cache effect
   = reference - oracle k=1.

`measures/controllability.py` stays the only implementation of I(a; s'|s); every measure calls E4-E7's code, moved to
the library where it lived in a finished experiment's scripts (DECISIONS (11)).

## Analysis (`experiments/E8_therapy_relapse/analyze_e8.py`)

Recovery curves of b, C1, C3 and the directed share against environment steps, gradient steps and visits of the
treated zone (shuffle: treated and untreated assignments separately); t½ and threshold crossings fixed in
EXPECTED_FORMS, censored when not reached (never extrapolated); the lag of behaviour behind C3; the belief / cache
decomposition at the oracle's checkpoints; transfer (untreated recovery over treated); the (b, directed share) points
of every checkpoint over E6's 650,000 cloud. Secondary, baselines only: a tabular twin (start counts + therapy counts)
-- blocked: the start runs' count tables are not available locally (DECISIONS (14)).

## Batches

`e8_w1_base` (base_A, base_B, exposed and references: 4 runs; the pilot), `e8_w2_shuf_nw` (shuffle A, E, F and
reference B, therapy on nw: 4 runs; launched after wave 1 is validated), and, generated but not launched,
`e8_w3_shuf_ne`, `e8_w4_shuf_sw`, `e8_w5_shuf_se`. One 3-CPU pod (`nadim-launcher` stopped during the runs); always a
`--dry-run` first; results back through `export_batch.py`, analysed locally. Re-acquisition (back to uncontrollable
after the therapy) is not implemented, but a run can start from a therapy checkpoint and the therapy environment is a
parameter.
