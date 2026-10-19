# E5 — decisions

Numbered in order. "User" marks a decision taken by the user; "implementer" one taken while implementing, within the
user's design, recorded here before any launch.

## 2026-10-19 (1) -- one E4 configuration, E4's parameters unchanged (user; implementer for the mechanism)

One configuration of E4 for all of E5, a run parameter (`--configuration`), set after E4's analysis; every condition
derives from it (the same rendering, colours, canonical orientation). `configs/e5/e5.yaml` holds `base:
configs/e4/e4.yaml`, read whole, and the `e5` block; `e5_setup.load_config` refuses any other key, so no training
parameter of E5 can differ from E4's (tested). Seeds: A = E4's seed of the configuration (`e4.configurations.<id>.seed`),
B = A + 100 (`e5.seed_b_offset`).

## 2026-10-19 (2) -- the shuffle's draw and its stream (user; implementer for the stream)

At each episode's reset, with probability f, an assignment drawn uniformly among the 12 ordered (UD slot, CD slot)
pairs of the four quadrants (the two others neutral); otherwise the canonical one (the configuration's). Every episode
draws u ~ U[0, 1) then k ~ U{0..11}, always both, so the stream advances by one episode whatever is drawn; the canonical
assignment can itself be drawn (it is one of the 12).

The stream is dedicated: root `SeedSequence([seed, 2])` (`streams/assignment.py`), distinct from both conditions' roots
(`SeedSequence(seed)`, `SeedSequence([seed, 1])`); it is never touched by the training, and no other stream is touched
by it (f = 0 equals E4 bit for bit while drawing from it, tested).

**The assignment is shared between a run and its reference by EPISODE INDEX, not by environment step** (user): the k-th
episode of the experimental run and the k-th episode of its reference get the same assignment, whatever their lengths
(the two runs' episodes differ in length, so at a given environment step they are generally in different episodes).

## 2026-10-19 (3) -- what an assignment changes (user)

The slot that receives UD takes T_inc and the conveyor, whose drift (`away_from_goal`) is resolved for that slot's rows;
the CD slot takes T_ctrl; the two neutral slots the corridor's colour and dynamics. The corridor, the goal, the torus and
the region of every state never change (tested for the 12). Library: `two_zones.build_layout(..., ud_slot=None)`,
`training_colors(..., ud_slot=None, cd_slot=None)`, `pipeline.two_zones.build_world(..., ud_slot=None)` (None = E4's,
tested). The reference: the same draws on `derive_control` of each assignment's layout (UD -> CD, T_inc kept). Pairing
tested for each of the 12: H(s'|s,a) = 0 and a support of 1 for every pair in both, I(a;s'|s) differs exactly on the UD
slot (0 vs log2 5).

## 2026-10-19 (4) -- per-episode appearances in the replay and the loss's mask (implementer, approved by the user)

The replay's table mode stored one image per state; under shuffle a state's image depends on the episode's assignment.
The replay (option, off by default) holds the 12 assignments' images (row = 12-block index x 144 + state) and, per row,
the assignment's agent-free background; every window then carries its per-step backgrounds (`ModelBatch.background`),
which the loss uses instead of the run's single background, so the reconstruction mask stays the agent's patch (with the
canonical background, a moved quadrant would get w_patch = 10 over its whole surface). `ModelBatch` gains that one field,
an image, never a state label; the library test pinning its fields was updated accordingly
(`tests/agents/worldmodel/test_replay.py`). Tests: E4's run with these options off is unchanged (E4's bit-exact identity
test passes); f = 0 with these options ON equals E4 with them off, bit for bit (weights, losses, stored steps); under
shuffle the background of every sampled step is its own assignment's and the observation differs from it on the agent's
cell only.

## 2026-10-19 (5) -- episode boundaries (implementer)

The assignment changes only at an episode's reset (the collector's episode source), so the first step of a new
assignment is always an `is_first` step, where the world model resets its state (the windows of 64 steps may straddle two
episodes, as in E3 and E4, but never carry a transition across the boundary). Tested on the replay of a shuffled run:
every change of assignment falls on an `is_first` step, every episode's last step is followed by one, and the
episodes' assignments are the stream's draws in order.

## 2026-10-19 (6) -- choosing switch_step (user)

`choose_switch_step.py` reads E4's REFERENCE run of the configuration and proposes the FIRST checkpoint of the grid (not
two consecutive ones) at which the future UD slot meets E4's thresholds as a reference can: mass on the true successor
>= 0.97 and |I at the first step - log2 5| <= 0.1 bit (a reference has no b: its I is compared with the true value; 0.1
is E4's |b| threshold at CD and in the corridor). It draws both curves (`switch_step.png`) for a visual check.
switch_step remains a parameter. E4's wave-2 configurations have diagnostics at 10,000, 100,000 and 300,000 only: the
script recomputes the missing checkpoints on the cluster from their weights and `replay_final.npz` (the replay as of a
checkpoint is its first `stored_steps` steps), with E4's own functions, measurement slots and seeds (`--verify` compares a
recomputed record with the recorded one).

## 2026-10-19 (7) -- the reference of the immunisation's b (user)

b of the immunisation at k post-switch steps is taken against the baseline reference of the same seed at checkpoint k.
With this choice, b_immun(k) - b_E4(k) = I_immun(k) - I_E4exp(k): the contrast between the immunised and the naive model
does not depend on the reference, which serves the absolute b only.

## 2026-10-19 (8) -- when the dynamics switch (user)

At the reset of the first episode that starts once switch_step transitions are done, so that no episode mixes two
dynamics. The step it actually happens (`dynamics_switch_step` = the transitions done then, between switch_step and
switch_step + 35) is logged in every checkpoint record with the post-switch step; the run lasts dynamics_switch_step +
post_steps transitions; its checkpoints are dynamics_switch_step + each point of E4's grid below post_steps, and +
post_steps. No checkpoint before the switch (the run equals the reference there). post_steps: a parameter, default
175,000 (the grid point nearest to 175,000, user). Tested: every transition before the switching episode follows the
reference's tensor, every one after it the experimental tensor, and the run's length and checkpoints are as stated.

## 2026-10-19 (9) -- the immunisation's budget: beta_rep's ramp stays E4's (user)

beta_rep's ramp after the estimator switch is 20% of the PLANNED gradient steps left (E3's rule). With switch_step +
post_steps as the planned budget, phase 1 would differ from E4's reference. The planned budget stays E4's (299,744 + 256);
`run_one(total_env_steps=...)` (option, off by default) runs past it. The ramp ends long before any plausible switch
step. Note: "switch" in the code also names E3's estimator switch (probabilities -> straight-through); the dynamics switch
is always `dynamics_switch_step`.

## 2026-10-19 (10) -- the replay and the optimiser across the switch (user; implementer for the numbers)

The replay is not emptied, the optimiser not reset: the memory of control is part of the manipulation. The buffer's
capacity is at least every step the run stores (E3's rule, `ReplayBuffer.for_run`, strict mode: nothing is ever
overwritten), so phase-1 data stay in the buffer for the whole of phase 2 and are sampled uniformly with the rest: after
k post-switch steps, the share of phase-1 steps in the buffer is about S / (S + k) (S = switch_step; e.g. S = 100,000:
0.53 at k = 88,000, 0.36 at k = 175,000; S = 175,000: 0.50 at k = 175,000).

## 2026-10-19 (11) -- branching: E4 kept no full state (implementer, as the user's rule foresaw)

E4's checkpoint files (`agent_env<N>.pt`) hold the weights only; the full state (`resume.pt`: optimisers, replay,
environment, collector, every stream) is one file overwritten every 10,000 steps and at each checkpoint (the last one at
290,000), never exported. No exact branch from E4's reference at an arbitrary step is possible: immunisation A recomputes
its phase 1 (on the reference's streams of seed A, identical to E4's reference up to the switch on the same device).
Seed B: the baseline reference keeps full states (`run_one(save_resumable_at=...)`, `checkpoints/resumable_env<N>.pt`) at
`e5.resumable_candidates` (E4's grid from 10,000 to 250,000) and at switch_step when wave 2 is launched with it;
immunisation B branches from the one at switch_step (`run_one(branch_from=...)`: same planned budget required, the
replay's capacity grown, its own records from there). The immunisation runs on the reference's streams throughout.
Tested: the immunisation's full state at switch_step equals the reference's (weights, optimisers, replay, streams,
progress), and the branched run equals the recomputed one after the switch (weights, losses, replay).

## 2026-10-19 (12) -- the checkpoint measures by assignment (implementer)

Under shuffle, E4's checkpoint measures must be taken assignment by assignment: a real context's one-step prediction is
matched against the state images of ITS assignment and scored against its successors; E4's record mixes every
assignment and matches against the canonical images (`pipeline.assignment_records`). The real contexts are drawn per
(assignment, state) row (`AssignmentReplayView`), up to e4.checkpoint's 20 per row, the kernel per assignment; the walks
are E4's on each assignment's image in its probe environment. Slots 22 (checkpoints) and 23 (final) of the run's
measurement streams; walk seeds: E4's for the canonical assignment (image index 0), image index 1000 + k for the others.
The canonical assignment's record is also written in E4's format (`"e4"`). Every assignment is measured at the
checkpoints of `e5.shuffle_records.all_assignments_at` (null: every checkpoint) and at the last one; cost and choice:
IMPLEMENTATION_REPORT.md. E4's final protocol control (full settings) on the canonical assignment only.

## 2026-10-19 (13) -- probes, positions and the RSA's points (user; implementer for the RSA)

The probes' positions are those of the canonical layout for every condition (under f = 1, "former UD slot" is the
canonical UD slot). The RSA's points are replay rows, i.e. (assignment, state) pairs, the goal left out: one appearance
(baseline, immunisation) gives E3's 143 states (theme and position confounded); under shuffle the same position appears
with several themes and controllabilities. The output reports the theoretical matrices' correlations and VIFs
(`predictor_correlations`, `vif`) for the canonical points and for all of them, which shows the decorrelation.
Observed on the mini runs (CPU): canonical points alone, theme ~ controllability = 1.0 and position ~ theme = 0.44; all
points of a shuffled exposed run, position ~ theme = 0.05 and position ~ controllability = 0.00, but theme ~
controllability = 0.53. Under shuffle the position separates from theme and controllability; theme and controllability
stay confounded BY DESIGN (the theme follows the type: T_inc always carries the conveyor), so the RSA cannot separate
them in an exposed shuffled run. Only a theme condition (excluded from E5 for now) could.

## 2026-10-19 (14) -- E5 imports no experiment (implementer, CLAUDE.md)

E5 reuses E4's set-up, checkpoint records and probe measure: they moved verbatim to the library
(`pipeline/two_zones_setup.py`, `pipeline/two_zones_records.py`, `probing/theme_probe_measure.py`) in their own commit,
E4's scripts re-exporting them (E4's tests pass, its bit-exact identity test included).

## 2026-10-19 (15) -- launch (user)

From a separate clone, `/scratch/helplessness-worldmodels-e5` (branch `e5`; `submit_kubeflow.py` refuses another
branch), the e4 clone keeping E4's runs and probes; `e4` is merged into `e5` before the launch. The configuration,
switch_step and post_steps are batch parameters set at launch. Durations are estimated from E4's 40-45 h for 300,000
steps (four runs in parallel).

## 2026-10-19 (16) -- the estimator switch under shuffle: a risk, logged (user's question; implementer's analysis)

E3's estimator switch (posterior probabilities -> straight-through, with the learning-rate drop, the heads' training
and beta_rep's ramp) fires when the label-free statistic -- the mean Jensen-Shannon divergence between the posteriors
of random PAIRS of batch images -- exceeds 3.0 bits (summed over the 8 variables) on 5 consecutive evaluations, after
500 gradient steps. In E3/E4 the background is fixed, so two images differ only by the agent's cell: a high JS means the
latent separates the agent's states. Under shuffle two images of different episodes also differ by the colours of the
quadrants (11 of 12 cross-assignment pairs, large areas), which the encoder separates easily: the statistic can reach
3 bits on a latent that codes the ASSIGNMENT but not yet the agent's position, so the switch may fire earlier. The same
statistic guards against a collapse after the switch (1.0 bit): it may miss a collapse of the position while the
assignment stays coded. Not a risk for the baseline and the immunisation (one appearance).

Logged in every E5 run: the switch's environment step (`e5.estimator_switch.env_step`; the gradient step is the record's
`switch_step`), the statistic (`batch_pair_js`) at every checkpoint; the agent state's decodability (the share of real
contexts whose posterior reconstruction matches the image of their own state, E3's real_context_template_accuracy): by
assignment at every checkpoint in the shuffle runs (`decode_accuracy`, `e5.decode_accuracy_canonical`), and at the
checkpoints up to 100,000 steps in the baseline and immunisation runs (`e5.decodability`). Reading: a shuffle run whose
switch comes much earlier than its reference's or the baseline's, with a low decodability at the first checkpoints after
it, is the failure this point describes.

## 2026-10-19 (17) -- the parade, proposed, NOT active (implementer)

The switch statistic on pairs of items with the SAME background image (`train.latent_estimator.switch_pairs:
same_background`; set from `e5.switch_pairs`, default `any` = E3/E4's statistic): the groups are computed from the
batch's per-step background images (no state label), i draws among the items that have a partner, j among i's partners
(`measures.diagnostics.mean_within_group_pair_jensen_shannon`, `pipeline.agent_run.background_groups`). Off: the
default path is E3/E4's, unchanged (the identity tests). On, it would draw differently from the `switch_pairs` stream,
so a run with it is not comparable step for step with one without it; it applies to the shuffle runs only (the others
have one background). Tested: the groups are exactly the items' assignments on a shuffled replay; the statistic never
pairs across groups.

## 2026-10-19 (18) -- checkpoint measures by assignment: defaults (user)

The canonical assignment at every checkpoint; the 12 assignments at 8 checkpoints, [10000, 25119, 63096, 100000,
150000, 200000, 250000, 300000] (`e5.shuffle_records.all_assignments_at`, a parameter; null = every checkpoint), and at
the run's last checkpoint in any case. Measured cost: 305 s per assignment per checkpoint on CPU (IMPLEMENTATION_REPORT.md).

The collapse of the mini configuration (JS and mass at 0) is E4's: E4's mini at the commit before the move (3657b33)
gives the same logs and bit-identical weights at 200, 400 and 600 steps, both conditions. It comes from the mini config
itself (`switch_js_threshold: -1.0` forces the switch at gradient step 16, `collapse_js_threshold: -inf`), not from E5.
