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

## 2026-10-19 (17) -- the parade, proposed (implementer); activated by (19)

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

## 2026-10-19 (19) -- the same-background switch statistic, active on every E5 run (user, after the short runs)

**Methodological observation (for the paper).** E3/E4's label-free separation criterion -- the Jensen-Shannon divergence
between the posteriors of random pairs of batch images -- reads "the latent separates the agent's positions" only in a
world whose background is fixed: two images then differ by the agent's cell alone. In a world whose background varies
(E5's shuffled assignments), it is confounded by the background: the encoder separates the backgrounds (large, easy
areas) long before the agent's position, the statistic crosses its threshold, and the switch (straight-through, the
learning-rate drop, the heads' training, beta_rep's ramp) fires on a latent that does not code the agent's state; the
collapse guard, which reads the same statistic, would miss a collapse of the position while the background stays coded.
Restricted to pairs of images that share their background, the criterion keeps its meaning -- distinguishing the
agent's positions -- in a world with a variable background.

Evidence (short runs on CPU, the real config `e5.yaml`, p1-a-d1, seed A, experimental, 4,000 steps = 2,246 gradient
steps; decodability = share of real contexts whose posterior reconstruction matches their own state's image):

| Run | grad 446 | grad 797 | grad 1,353 | grad 2,234 | switch |
|---|---|---|---|---|---|
| baseline (E4's procedure) | JS 0.00, dec 0.7% | 0.00, 1.0% | 1.06, 12% | 3.84, 33% | not yet (JS above 3 at the end) |
| shuffle f = 1, E3/E4's statistic | 4.57, 3.5% | 4.22, 2.4% | 4.61, 4.9% | -- | gradient step 701 (env 1,425), decodability about 3% |
| shuffle f = 1, same-background statistic | 0.00, 3.5% | 0.00, 2.4% | 0.00, 8.1% | 0.00 | none up to 2,234 |

At the runs' end (gradient step 2,246): baseline JS 3.84, decodability 34.8%, no switch; shuffle with E3/E4's statistic, switched at 701, decodability 1.0%; shuffle with the same-background statistic, JS 0.00, decodability 4.2%, no switch.
(The two shuffle runs are identical until the first one switches: the pairs draw from their own stream. The canonical
decodability of a shuffle run rests on about 85 contexts, 1/12 of the data: imprecise.) Validation criteria (user):
the switch comes markedly later than 701 -- met (none up to 2,234, at least 3.2 times later); the decodability at the
switch at least comparable to the baseline's -- NOT observable in 4,000 steps (neither the baseline nor the parade
switched); it is read on the cluster from the first hours of wave 1 (`e5.estimator_switch`, `decode_accuracy`).

Scope: every E5 run (user's points 1-2). Under shuffle, f = 1 and f = 0.5, experimental and reference (their
backgrounds vary; a run and its reference follow one procedure). With one background (the baseline, the immunisation, a
batch of a single assignment) the statistic is E3/E4's, the same draws: f = 0 with the option equals E4 bit for bit, and
so does the baseline with it (tested: weights, losses, stored steps), so the option is set on every E5 run for one
uniform procedure (`e5.switch_pairs: same_background`, copied by run_e5.py into `train.latent_estimator.switch_pairs`).
The collapse guard reads the same statistic (SeparationSwitch.update takes one value for both). Pairs: logged at every
evaluation and summarised at every checkpoint (`switch_pairs`: their number, the statistic's mean, sd, min, max, the
items with a partner, the groups). Offline on the short run's replay (2,000 batches of 16 x 64 drawn as in training):
1,023 to 1,024 of the 1,024 items have a same-background partner, 10.5 groups per batch on average; the statistic was
<= 5e-5 bit on all 16 evaluations -- pairs are not scarce and the statistic not noisy: no running mean or minimum number
of pairs is needed.

## 2026-10-19 (20) -- the separation deadline, moved back for every E5 run (user, 2026-10-20)

E3/E4's procedure stops a run whose switch statistic has not crossed its threshold by `switch_max_steps` gradient steps
(E4: 25,000; `SeparationNotReachedError`). That deadline is a guard calibrated on a world with a FIXED background: E4's
wave 1 switched at gradient steps 2,551 / 3,901 (p1-a-d1 experimental / reference, environment steps 4,508 / 6,758) and
3,051 / 3,451 (p1-b-d1, 5,342 / 6,008), the latest at 3,901 (E4's wave 2 is not available here). Under shuffle the
agent's position is learnt more slowly (short runs, gradient step 2,234: same-background statistic about 4e-5 bit and
reconstruction error on the agent's patch 3.79, against 2.1-3.2 bits and 0.38-1.05 for E4's four runs), so a shuffle
run with the same-background statistic could meet E4's deadline.

The deadline is moved back UNIFORMLY, for every E5 run, to max(75,000, 10 x E4's latest switch = 39,010) = **75,000
gradient steps** (`e5.switch_max_steps`, copied by `run_e5.apply_e5_procedure` into `train.latent_estimator.switch_max_steps`;
about 125,000 environment steps). The deadline only stops a run (`SeparationSwitch.check_deadline`): a run that switches
before E4's 25,000 is unaffected, step for step (tested: f = 0 and the baseline with E5's procedure equal E4 bit for bit).

Logged: at every checkpoint, the switch's gradient and environment steps and the environment steps elapsed since
(`e5.estimator_switch`: grad_step, env_step, env_steps_since); `switch_monitor.py final` gives one line per run (switch
steps, environment steps trained after it). Check of wave 1's first hours (`switch_monitor.py early`, RUN_GPU.md): at
every checkpoint up to the switch, the switch statistic, the agent state's decodability and the reconstruction error on
the agent's patch, shuffle against E4's baseline. ALERT rule (user; a notification, never an automatic stop): a shuffle
run that has not switched at 50% of the deadline (37,500 gradient steps) with a decodability that has not risen over its
last 3 checkpoints (last minus third-to-last <= 0.01).

**The comparison between conditions is made after checking convergence, not at the budget alone**: a shuffle run that
switches late trains fewer steps after its switch than E4's runs (about 295,000); the final table of
`switch_monitor.py` gives it per run, and the probes are compared between converged models.


## 2026-10-21 (21) -- the configuration: p1-b-d1 (user)

E5 runs on **p1-b-d1** (pair 1, red/blue; T_inc = (22, 0, 244) blue, T_ctrl = (227, 17, 15) red): seed A = 2 (E4's), seed
B = 102. Reason, fixed before seeing any E5 result: in E4, p1-b is the configuration where the theme alone carries no
belief (theme only: b at step 1 = -0.021 +/- 0.007, against -0.289 in p1-a). Any belief the theme carries in E5 can then
be attributed to the training manipulation, not to an effect already present in the baseline. Pair 2 is set aside: its
near-black colour (17, 29, 14) is salient and close to the brown agent (RGB distance 86), and two of p2-a's extreme probes
are invalid (E4's final report). `batches.CONFIGURATION`; the submission refuses another configuration.

## 2026-10-21 (22) -- probe validity is part of the protocol (user)

E4's final criterion (results/E4_generalization_gradient/final/REPORT.md): a probe's b is interpretable only if the
REFERENCE of the same condition and seed (the baseline reference for the immunisation) predicts the transitions at the
first step in the zone -- mass on the true successor >= 0.90 and |I_ref - log2 5| <= 0.15 bit. In E4, 203 of 400 probes
failed it, and validity fell with the distance between the probed colour and the slot's training colour: every model's
reconstruction rendered the probed quadrant in its training colour. E5's `measure_probes.py` writes, for every probe,
mass / ghost / I at step 1 of both models and `valid` (read from the same measurement), `validity_summary.json` (the share
of valid probes, overall and by axis and position) and `reconstruction.csv` (the reconstruction diagnostic, both
models: reconstruction error, the agent's decoded position, the colour reconstructed in the zone of interest and its
distance to the probe's colour and to every colour the slot takes in training). The code is the library's
(`probing/probe_validity.py`, moved from E4's `final_analysis.py`). Invalid probes are excluded from the gradients and the
comparisons; the analysis reports the share of valid probes BY CONDITION.

**Prediction, recorded before any E5 result:** under shuffle the colour of a slot no longer follows from its position, so
the model must code the quadrants' colours in its latent; we expect markedly more valid probes than in E4 (where the
decoder rendered every quadrant in its training colour), most of all under f = 1, and f = 0 (= E4's training) about as
many as E4's p1-b (48 of 100).

## 2026-10-21 (23) -- probes under shuffle (user; implementer for the check)

Unchanged: the probes' positions are the canonical layout's in every condition; f = 0 replicates E4 and is the control.
Under f = 1 the canonical layout is one of the 12 training assignments, so "position only" (the corridor's colour at the
canonical UD slot) tests whether a position without its theme still carries a belief: b about 0 is expected if the belief
moved to the theme. Checked: nothing in the probe script assumes one colour per slot -- the probe images are built from
the configuration's colours and the canonical roles only, and the reconstruction diagnostic compares the reconstructed
colour with every colour a slot takes in training (canonical training colour, T_inc, T_ctrl, the corridor's), not with
one.

## 2026-10-21 (24) -- the conveyor in the 12 assignments (user's check)

Only the UD slot's dynamics differ from the corridor's (CD and the neutral slots are controllable-deterministic alike),
so every assignment equals the canonical one, for the dynamics, up to the goal-centred axial reflection that sends the
canonical UD slot to its own (`pipeline.episode_sources.assignment_symmetry`); tested for the 12, both conditions, with
E3's exact transport (every (s, a), actions permuted). The drift (`away_from_goal`, along the rows) is well defined in
every slot: (+1, 0) for UD at nw and ne, (-1, 0) at se and sw. E3's two rotation classes (identity = 180 degrees: "A",
UD at nw or se; 90 = 270 degrees: "B", UD at ne or sw) were inequivalent in E3, whose four zone types differed; here a
column reflection maps one onto the other. The images differ (where T_ctrl sits: diagonal / same rows / same columns). The
outputs carry the class for the analysis: `episode_rotation_class` per episode in `assignments.npz`, and at every
checkpoint `e5.assignment_symmetry` (reflection, rotation class, the CD slot's relation, drift) of each measured
assignment.

| k | UD | CD | reflection | class | CD vs UD | drift |
|---|---|---|---|---|---|---|
| 0 | nw | ne | identity | A | same rows | (+1, 0) |
| 1 | nw | se | identity (canonical) | A | diagonal | (+1, 0) |
| 2 | nw | sw | identity | A | same columns | (+1, 0) |
| 3 | ne | nw | col_reflect | B | same rows | (+1, 0) |
| 4 | ne | se | col_reflect | B | same columns | (+1, 0) |
| 5 | ne | sw | col_reflect | B | diagonal | (+1, 0) |
| 6 | se | nw | point_reflect | A | diagonal | (-1, 0) |
| 7 | se | ne | point_reflect | A | same columns | (-1, 0) |
| 8 | se | sw | point_reflect | A | same rows | (-1, 0) |
| 9 | sw | nw | row_reflect | B | same columns | (-1, 0) |
| 10 | sw | ne | row_reflect | B | diagonal | (-1, 0) |
| 11 | sw | se | row_reflect | B | same rows | (-1, 0) |

## 2026-10-21 (25) -- 150 walks for the probes; seed A's baseline re-probed (user)

The zero-data probes use the first 150 of E4's fixed walks (`e5.probes.n_walks`; E4: 300): over E4's 192 valid probes the
standard error of b at step 1 has a median of 0.012 and a maximum of 0.048 at 100 walks (0.010 and 0.035 at 150). For
one protocol across conditions, seed A's baseline -- E4's own pair of p1-b-d1 -- is re-probed with E5's script at 150
walks: batch `e5_probes_baseline_A` (probes only, `--e4-results`). The walks of the measures during training stay E4's
(100 at the checkpoints).

## 2026-10-21 (26) -- STALE runs (user)

E4's wave 2 was frozen for 24 hours by a failure of the cluster's storage with nothing visible in `status.json`. The
batch status (`submit_kubeflow.py --status <supervisor directory>`, E4 and E5, `helplessness.cluster.batch_status`) flags
STALE every running job whose log has not changed for more than an hour (exit code 2 when one is). A batch can be resumed
in one pod (`--phase resume` without `--resume`; resume points checked on the CPU first, `helplessness.cluster.resume`).

## 2026-10-21 (27) -- budget 300,000; switch step 100,000; separation deadline unchanged (user; implementer for the numbers)

The budget stays 300,000 steps. E4's analysis suggests 200,000-225,000 (its recomputed criterion gives 175,000), but the
shuffle runs will be harder to learn and switch later; 300,000 with diagnostics at several checkpoints lets the
analysis checkpoint be chosen afterwards at the convergence criterion. No code change.

switch_step: `choose_switch_step.py` on E4's reference of p1-b-d1 (diagnostics at every checkpoint) proposes **100,000**,
the first checkpoint where the future UD slot (nw) meets mass >= 0.97 and |I at step 1 - log2 5| <= 0.1: mass 0.978,
I 2.289 (63,096: 0.929, 2.200). It is one of `e5.resumable_candidates`.

Separation deadline: E4's estimator switches, both waves -- wave 1: 2,551 / 3,901 (p1-a exp / ref), 3,051 / 3,451 (p1-b);
wave 2: 4,801 / 6,001 (p2-a), 3,701 / 3,701 (p2-b). The latest, 6,001, is below 7,500: the deadline stays 75,000 gradient
steps (max(75,000, 10 x 6,001)).

## 2026-10-21 (28) -- wave 1: the same-background statistic plateaus under shuffle; a late threshold (user's question; implementer's diagnosis and proposal)

**Observed** (wave 1, p1-b-d1, seed A, `switch_monitor.py early`, two checkpoints): no run switched by gradient step
9,355. f = 1: the agent's patch is reconstructed almost perfectly (0.0055 reference, 0.027 exposed) but the
same-background statistic plateaus at 1.9-2.1 bits, below the 3-bit threshold; f = 0.5 reference: statistic about 1e-5
bit, reconstruction 1.28 and decreasing; f = 0.5 exposed: 0.50 then 0.16 bit. E4's eight runs switched at gradient steps
2,551-6,001, their reconstruction of the agent's patch still 0.12-1.0 and their statistic rising without a plateau.

**Semantics (the code).** At `switch_max_steps` the run STOPS (`SeparationSwitch.check_deadline` raises
`SeparationNotReachedError`; the run is failed): never a forced switch. The collapse guard acts only AFTER the switch
(before it, `update` only counts the streak above the threshold): the f = 0.5 runs are not exposed to it now; after a
switch it stops a run whose statistic stays below 1.0 bit on 5 consecutive evaluations. The statistic: the posterior
PROBABILITIES q(z_t | h_t, e(x_t)) of the training forward pass (the "probs" estimator before the switch), summed over
the 8 variables of each pair's per-variable Jensen-Shannon divergence; h enters only through the posterior's
conditioning. Ceiling: 8 x 0.979 = 7.83 bits (disjoint point masses under the 1% unimix); 6.85 bits in expectation over
random pairs for a perfect, uniform code -- E4's 3 bits are 44% of it. The monitor's decodability unrolls the windows
with hard straight-through SAMPLES and decodes (h, z): before the switch the model is trained on probabilities, so the
decodability reads a regime the model is not trained in (it can fall while the reconstruction improves).

**Diagnosis.** (i) The same-background groups are right under f = 0.5 (mini, both conditions: 0 wrong partition in 200
batches). (ii) The f = 0.5 reference's near-zero statistic is not an anomaly of E5: E4's p2-b-d1 exposed run had a
statistic of 0.000 for 2,846 gradient steps with a reconstruction about 3, then jumped (switch at 3,701). (iii) The f = 1
plateau is structural: the latent first codes the background. On the local short run under f = 1 (gradient step 2,234,
`latent_ceiling.py`), all 8 variables code the assignment (1.25-2.3 bits of mutual information each, posteriors sharp:
largest probability 0.90-0.98) and none the agent's state within an assignment (0.0 bit); between different backgrounds
the statistic is 5.35 bits, within the same background 0. A variable whose class follows the background is identical
within a same-background pair and adds nothing to the statistic, so the statistic a converged shuffled model can reach
is the share of the latent left to the position -- below E4's, whose background never varies. (To quantify on the
cluster's runs: `latent_ceiling.py` on their resume.pt.)

**Criterion.** The switch keeps E4's rule (3 bits on 5 consecutive evaluations) and ALSO fires on a lower threshold, 1.5
bits on 5 consecutive evaluations, which can act only from gradient step 10,000 on -- after every switch of E4's eight
runs (6,001 the latest). Library option `SeparationSwitch(late_threshold_bits, late_after_steps)`, off by default;
`e5.late_switch: {threshold_bits: 1.5, after_steps: 10000}`, copied by `run_e5.apply_e5_procedure` into every E5 run;
the rule that fired is logged (`switch_rule`, `e5.estimator_switch.rule`). Label-free (the same statistic). Identical to
E4 where E4 switched: tolerance zero by construction, the late rule never acts before 10,000 and every E4 run switched
by 6,001 by the main rule (tested: a run that switches by the main rule is unchanged; f = 0 and the baseline with E5's
procedure equal E4 bit for bit). 1.5 bits: half of E4's threshold and 0.5 bit above the collapse guard (1.0), which keeps
watching after the switch; the f = 1 runs (1.9-2.1 bits) would switch soon after 10,000, the f = 0.5 runs not yet.

For the paper: "A separation threshold calibrated on a fixed background is unreachable when part of the latent codes a
varying background, so after the latest switch of the fixed-background experiment the threshold is halved; it never
acts where that experiment switched."

**Risks that remain.** A shuffle run switched by the late rule could see its statistic fall below 1.0 bit after the
switch (the collapse guard would then stop it: a result). The f = 0.5 runs may still not reach 1.5 bits before the
deadline (75,000).

## 2026-10-22 (29) -- the first wave 1 failed; wave 1 = f = 1 on seeds A and B; late threshold from 6,100; f = 0.5 out; the immunisation to E8 (user; implementer for the analysis)

**What happened.** The first wave 1 (p1-b-d1, seed A, f = 1 and f = 0.5, both conditions) was not stopped: its four
runs failed at the separation deadline (75,000 gradient steps, `SeparationNotReachedError`), before the late threshold
of (28) was pushed. f = 1 reference: statistic 1.8-2.6 bits throughout, agent's patch reconstructed (0.00016 at the
end): the plateau of (28). f = 1 experimental: **collapsed while waiting for the switch**, between gradient steps 9,355
and 14,917: statistic 2.06 -> 3e-11, decodability 0.267 -> 0.005, reconstruction of the agent's patch 0.027 -> 9.06,
never recovered. f = 0.5, both conditions: statistic about 0 throughout, reconstruction 1.0 (step 9,355) -> 2.2-3.0
(end), decodability falling: they degrade instead of learning.

**Reading of the collapse (implementer, to be confirmed on the copied logs).** Before the switch the world model trains
at lr 1e-3 (`train.lr`; 1e-4 from the switch on, `lr_after_switch`), with no gradient clipping (`clip_norm: null`).
E3 diagnosed and reproduced a late single-step collapse at lr 1e-3 (docs/e3/DECISIONS.md, 2026-09-24): a rare batch
gives the encoder a gradient about 100 times its usual scale, Adam's second moment (starved by near-zero gradients) turns
it into a large coordinated step, the posterior falls into a saturated, state-independent code and never comes back --
the reason the procedure drops the lr to 1e-4 at the switch. E3/E4 never stayed long at lr 1e-3 (E4: at most 6,001
gradient steps); the shuffled runs stayed tens of thousands. The f = 1 experimental collapse has that signature (an
abrupt, permanent fall of the statistic and of the reconstruction). **Waiting long before the switch is not safe**: an
argument for an early late threshold.

**Decisions (user).**
- **Wave 1 = shuffle f = 1, seeds A (2) and B (102), experimental and reference** (4 runs).
- **Late threshold from gradient step 6,100** (`e5.late_switch.after_steps`; 1.5 bits on 5 consecutive evaluations
  unchanged): just above E4's latest switch (6,001), so every E4 switch is unchanged by construction (the late rule
  never acts before 6,100; E4's eight runs switched by the main rule by 6,001); tests: a run that switches by the main
  rule is unchanged, f = 0 and the baseline with E5's procedure equal E4 bit for bit.
- **f = 0.5 out of the campaign for now**, until its diagnosis (`batches.DEFERRED["f050"]`, never submitted).
- **The immunisation moves to E8** (an annex experiment: prevention, beside the therapy): out of E5's waves
  (`batches.DEFERRED["e8_immunisation"]`), its code kept; seed B's baseline reference keeps its full state at 100,000
  steps (`batches.BASELINE_REFERENCE_RESUMABLE_AT`) so that E8 can branch from it.
- **Wave 2 = seed B's baseline, experimental and reference** (2 runs). Two places are free on its GPU (see the
  implementation report for the proposals).
- The failed runs move to `results/E5_belief_training_aborted/wave1_20261021/` after their light files are copied
  (RUN_GPU.md, step 3b).

**The replay.** `replay_switch.py` replays a criterion on the switch statistic a run recorded at every evaluation
(`switch_pairs.evals` in checkpoints.jsonl): exact up to the switch, since the training does not depend on the rule
before it (tested: on a run of the test suite, the replay finds the recorded switch). On the failed runs' series it
gives, for each run, the switch under the corrected criterion and the rule, the collapse onset (the last stretch below
the collapse guard's 1.0 bit) and the switch's margin before it, and a grid of thresholds (1.0-2.0 bits) and
activations (6,100 / 8,000 / 10,000). Results: IMPLEMENTATION_REPORT.md, once the series are copied.

## 2026-10-22 (30) -- the replay on the failed wave 1's series: the late threshold from 6,100 at 1.5 bits is kept (implementer; user's request)

`replay_switch.py` on the four failed runs (seed 2; the same-background statistic at every evaluation, 1,497 evaluations
from 0 to 74,800, no gap; E4's main rule: 3 bits on 5 consecutive evaluations from 500; the deadline 75,000):

| Run | E4's rule alone | corrected criterion (1.5 bits from 6,100) | statistic (mean +/- sd) | collapse |
|---|---|---|---|---|
| f = 1 reference | no switch | **6,301**, late threshold | 1.84 +/- 0.14 bits (5,000-13,500), max 3.25 | none |
| f = 1 experimental | no switch | **7,151**, late threshold | 1.49 +/- 0.08 (6,100-7,500), 1.89 +/- 0.19 (7,500-13,000) | **13,250**: from 1.37 to 0.0 in one evaluation |
| f = 0.5 reference | no switch | no switch | about 0 throughout (max 0.058) | -- |
| f = 0.5 experimental | no switch | no switch | up to 1.28 (5,000-6,100), then down: 0.33 (6,100-7,500), 0.06 (7,500-13,000), 0.004 after | below 1 bit for good from 6,550 |

The f = 1 experimental collapse is abrupt: 1.44 / 1.37 bits at 13,150 / 13,200, 0.0 from 13,250 on; the reconstruction
error on the agent's patch goes from 0.007 (gradient step 13,046) to 8.97 (13,646) and stays -- E3's single-step
collapse at lr 1e-3. **The corrected criterion would have switched it at 7,151, 6,099 gradient steps before the
collapse**, after which the lr is 1e-4. For seed 2, the relaunched runs are identical to the failed ones up to their
switch (same code up to it, same seeds, same device type), so these are their switch steps.

Grid (late threshold 1.0-2.0 bits x activation 6,100 / 8,000 / 10,000): every setting switches the f = 1 experimental
run before its collapse; at 6,100 the thresholds 1.0 and 1.25 switch it at 6,301, 1.5 at 7,151, 1.75 at 9,101, 2.0 at
12,051 (900 steps before the collapse); the reference switches at 6,301 for every threshold up to 1.75 (12,451 at 2.0).
A later activation only delays the switches (8,201 / 10,201). No setting makes f = 0.5 switch. **Kept: 1.5 bits from
6,100** -- earliest activation compatible with E4's identity by construction; 1.5 bits keeps 0.5 bit above the collapse
guard and switches both seed-2 runs within 1,050 steps of the activation, 6,099 steps before the collapse. A lower
threshold (1.25) would switch the experimental run 850 steps earlier but sits only 0.25 bit above the guard; it stays an
option if seed B plateaus between 1.25 and 1.5 (read in wave 1's first hours). Non-regression test on these series:
`tests/e5/test_e5_batches.py::test_replay_of_the_corrected_criterion_on_the_failed_wave_1` (the two f = 1 series in
tests/e5/data, 50 KB).

f = 0.5: its statistic peaks around 1.3 bits near 5,000-6,000 gradient steps (experimental) and then decays, before any
long stay at lr 1e-3 -- not the same signature as the f = 1 collapse (gradual, not a single step); the reference never
separates. Its diagnosis waits for `latent_ceiling.py` on the failed runs (RUN_GPU.md, step 3b.4).

## 2026-10-22 (31) -- the latent of the failed runs; f = 0.5 abandoned; more seeds of f = 1; waves 2 and 3 (user; implementer for the reading and the code)

**`latent_ceiling.py` on the four failed runs** (their resume.pt at environment step 125,000, gradient step 74,846; 100
batches drawn as training draws them, the "probs" regime; sharpness = mean largest posterior probability, 0.991 = the
1% unimix's point mass):

| Run | statistic same background / sharpened / different backgrounds | per variable |
|---|---|---|
| f = 1 reference | 2.05 / **4.74** / 3.95 bits | vars 3, 5, 6: the agent's position (state within an assignment 2.5-2.9 bits; assignment <= 0.21); vars 0, 2, 4: both (state 1.7-2.1, assignment 0.7-1.0); vars 1, 7: mostly the assignment (1.84; state 0.74-0.88); 8 classes used; sharpness 0.47-0.78 |
| f = 1 experimental | 0 / 0 / 0 | ONE class per variable, sharpness 0.991: one saturated code for every input |
| f = 0.5 reference | 0 / 0 / 6.34 | all 8 variables: the assignment only (1.17-2.12 bits; state within an assignment 0.000), 3-7 classes, sharpness 0.991 |
| f = 0.5 experimental | 0 / 0 / 6.76 | the same (assignment 1.40-1.83 bits; state 0.000), sharpness 0.991 |

Reading (the user's, confirmed with two precisions). f = 1 reference: a healthy split of the latent between the position
and the assignment; the plateau near 2 bits comes mainly from SOFT posteriors (sharpness about 0.5, 0.78 on the two
assignment variables), not from missing information: with confident posteriors on their current classes the statistic
would be 4.74 bits, above E4's threshold. This amends (28): the latent first codes the background (at gradient step
2,234 every variable coded the assignment), then frees most of its variables for the position, softly; the switch is
what makes the posteriors hard (straight-through), so switching such a run on the late threshold is sound. f = 1
experimental: its final state is E3's collapse signature exactly -- one saturated code for every state -- the collapse
of gradient step 13,250 (its statistic is 0 from 13,250 to the end, (30)). f = 0.5: more precise than "the background is
coded": every variable carries a deterministic, SATURATED code of the assignment alone (0.991 = the unimix ceiling, 3-7
classes per variable) and no information on the agent within an assignment. A saturated posterior starves its own
gradient (E3, 2026-09-24: the softmax saturates, the gradient through the probabilities vanishes, Adam's state is starved),
so the latent cannot move any more: a lock-in by saturation. Its statistic had begun to rise (1.28 bits near gradient
steps 5,000-6,000 in the experimental run) and then decayed: the position was starting to be coded and was lost as the
background code saturated. The cause of the saturation under f = 0.5 and not under f = 1 is not established by these
data (the user's hypothesis -- the canonical layout in about 54% of the episodes drives an early, confident
specialisation on recognising the background -- is compatible with them but untested).

**f = 0.5 is abandoned for E5 (user).** Repairing it would change the training procedure: for f = 0.5 alone, the
comparison between conditions would be confounded; for every run, everything would be redone -- too costly for a
secondary question. It stays defined (`batches.DEFERRED["f050"]`), to be retried only if time remains at the end and a
fix applicable to every condition exists. For the paper (a methodological limit): "When the zone assignment was redrawn
in only half of the episodes, the world model never learned the agent's position: its latent locked onto the
background, every latent variable saturating on a code of the assignment alone, and the label-free separation criterion
was never met. Full variability (every episode redrawn) did not show this lock-in. Repairing it would have required
changing the training procedure for every condition, so we report the partial-variability condition as a limit of the
method rather than as a result."

**The immunisation is E8** ((29); `docs/E8_immunisation.md`); seed B's baseline reference keeps its full state at
100,000 steps.

**More seeds of f = 1; waves 2 and 3 (user).** Seeds by B's rule (A + 100 per label): **C = 202, D = 302**
(`e5_setup.SEED_LABELS` A-D; the export reads the same from the configs' text). No stream collides: every run's roots --
SeedSequence(seed), [seed, 1] (reference), [seed, 2] (assignments) -- are distinct across E4's eight seeds (1-8) and E5's
2, 102, 202, 302, seed 2 being E4's p1-b-d1 run on purpose (tested).
- `e5_wave2`: seed B's baseline (experimental, reference; the reference with `--save-resumable-at 100000`) and shuffle
  f = 1 seed C (experimental, reference).
- `e5_wave3`, **to be confirmed after wave 1's results**, submittable: seed C's baseline and shuffle f = 1 seed D.
- Together: the baseline on 3 seeds (A = E4's pair, B, C) and f = 1 on 4 (A, B, C, D), paired by seed. The switch
  criterion is the current one, unchanged ((28)-(30)).
- Probes: every wave's pairs (`batches.PROBE_PAIRS`) with the full grid at 150 walks, plus `e5_probes_baseline_A`;
  `analyse_probes.py` aggregates b at step 1 probe by probe over 2 to 4 seeds per condition (valid probes only; mean,
  sd, number of valid seeds; the extreme probes; tested).

**(31), completed the same day (user): no probe between the waves; one probe batch after wave 3.** Between two waves:
the export and the learning check only -- `check_wave.py --batch <wave>`, one command, one line per run (status; the
estimator switch and its rule; after it, the longest stretch of the switch statistic below the collapse guard's 1 bit,
its lowest and last value; the smallest quadrant mass at the last checkpoint, by zone type over the 12 assignments for
the shuffle runs; the success rate) and one per pair (b at step 1 in the training world by slot, and by zone type over
the assignments under shuffle; the checkpoints where E4's convergence criterion holds), from the light files only, with
the standard library (the notebook). The probes of every wave and of seed A's baseline go in ONE batch after wave 3,
`e5_probes_all` (baselines A = E4's pair, B, C; shuffle f = 1 A-D: 7 pairs), at ONE common checkpoint chosen from the
checks' convergence lists (a point of E4's grid). Checkpoints until then: nothing in the code deletes or overwrites a
weight file (`agent_env<N>.pt` is written once under its step's name, by an atomic write of that file only; only
`resume.pt` and `latest.json` are rewritten; a resumed run rewrites the checkpoints after its resume point with the same
training; `make_run_dir` refuses an existing directory; the probes write new directories); E4's runs of p1-b-d1 must stay
on the volume too (seed A's baseline). What the code cannot rule out -- a purge of the volume by the cluster, `git clean`
in a checkout (`results/` is untracked), a manual deletion -- is written in RUN_GPU.md (step 7).

## 2026-10-23 (32) -- the relaunched wave 1: one f = 1 run locks in; the lock-in is a seed-dependent failure mode of the shuffle (user; implementer for the reading and the code)

**Observed (user).** Seed A (2): experimental and reference switched at gradient steps 7,151 and 6,301, as the replay
predicted ((30)); seed B (102): the reference switched at 6,301; decodability about 0.85 after the switch. Seed B's
EXPERIMENTAL run does not switch: its statistic has stayed <= 0.003 bit from the start -- the lock-in seen under f = 0.5
-- and it will go to the deadline (75,000 gradient steps).

**The NaN of its console is not numerical.** The line prints `goal_loss {last.get('goal_loss', nan)}` and
`critic_loss`, and the agent's heads (goal head, critic) train only from the estimator switch on (`run_one`: `if
switch.switched: train_heads_step(...)`): a run that has not switched has no such loss, printed as nan; with untrained
heads every Q value is tied, hence `ties 1.000` (random actions). E4's runs printed the same before their switch. The
run's log: `results/E5_belief_training/grid12x12__2zones-p1-b-d1-sh100__pair-experimental__worldmodel-agent-e5__seed102__<UTC>/console.log`
in the e5 checkout, and the supervisor's copy `logs/kubeflow/e5-train-e5_wave1-20261022T074754459275Z/B_sh100_experimental.log`.
After its failure, `latent_ceiling.py` on it (RUN_GPU.md) is to confirm the lock-in (every variable a saturated code of
the assignment, no information on the agent).

**The lock-in is a seed-dependent failure mode of the shuffle, not a property of f = 0.5.** It hit both f = 0.5 runs
(2/2) and one of the four f = 1 runs of the relaunched wave 1 (1/4: seed B, experimental); the other three f = 1 runs
switched and learnt. (The first wave 1's f = 1 runs, before the late threshold, did not lock in: their reference
plateaued, their experimental run collapsed at 13,250.) Decision (31)'s formulation for the paper becomes (to be
confirmed by `latent_ceiling.py` on seed B's experimental run):

"Under shuffled zone assignments, the world model's latent could lock onto the background before learning the agent's
position: every latent variable saturated on a code of the assignment alone, and the label-free separation criterion
was never met. This failure depended on the seed: it hit one of four runs with full variability (every episode
redrawn) and both runs with partial variability (half of the episodes redrawn). Runs that locked in never reached the
learning criterion and are excluded from the comparison; partial variability, where it hit every run, is reported as a
limit of the method rather than as a result."

**Alerts in `check_wave.py`.** LOCK-IN: a run that has not switched and whose statistic stayed below 0.01 bit on every
evaluation from gradient step 6,100 on (at least 20 evaluations); NaN/inf: a training loss present in a checkpoint record
or the statistic not finite -- the heads' losses ABSENT before the switch are not flagged. Tested.

**Completing f = 1 (proposal, the user decides).** Seed E = 402 (B's rule) is defined (`e5_setup.SEED_LABELS`), in no
wave yet. Complete f = 1 pairs (experimental and reference both switched, no collapse) so far: A; B is incomplete. Waves
2 (B's baseline, f = 1 C) and 3 (C's baseline, f = 1 D) are full (four runs each, the GPU's four parallel runs), so there
is no free place: see IMPLEMENTATION_REPORT.md, section 10, for the options.

**(32), updated the same day (user): two waves only; seed B's experimental run replaced.**
- **No wave 3**: seed C's baseline and seed D are dropped (labels C = 202 and D = 302 stay defined, unused).
- **`e5_wave2`** = seed B's baseline, experimental and reference (the reference with `--save-resumable-at 100000`, for
  E8), and two shuffle f = 1 EXPERIMENTAL runs replacing seed B's (locked in): seeds **E = 402** and **F = 502** (B's rule,
  `e5_setup.SEED_LABELS`), both **measured against seed B's f = 1 reference** of wave 1 (`batches.REFERENCE_SEED_OF`;
  `measure_probes.py --reference-seed-label B`, refused unless asked; `check_wave.py` pairs them by itself).
- **Why another seed's reference is legitimate (user).** Already in E4 the experimental run and its reference had
  different streams (`streams.worldmodel`: the reference's root is [seed, 1]): the reference cancels the REPRESENTATION's
  bias (what a model trained on fully controllable dynamics believes of the same image), it does not pair the chance of
  training. Consequence, recorded: the replacements draw their zone assignments from their own stream ([402, 2],
  [502, 2]), so decision (2)'s shared draws by episode index do not hold for these two pairs -- by the same argument, it
  is not needed for b.
- **Probes**: ONE batch after wave 2, `e5_probes_all`, at **K = 300,000** (`batches.PROBES_CHECKPOINT`): every healthy
  pair of waves 1 and 2 plus seed A's baseline (E4's own pair). At submission (dry run included) a pair is left out,
  with its reason printed, when one of its runs is not done, never switched, locked in, or lacks its weights at K
  (`submit_kubeflow.probe_plan`): seed B's own f = 1 pair is left out this way; if both replacements succeed, both are
  probed (two f = 1 experimental runs against one reference).
- **`check_wave.py` excludes a locked-in run from every pair** and lists the excluded pairs apart (`EXCLUDED`), with the
  LOCK-IN and NaN/inf alerts of (32).
- Tested: the two-wave table, the replacements' commands and pairing, the probe plan (healthy pairs kept; the locked-in
  and the failed left out; seed A's baseline from E4's runs), the cross-seed guard of `measure_probes.py`.
- Calendar: IMPLEMENTATION_REPORT.md, section 11.

## 2026-10-26 (33) -- validity of waves 1 and 2 before the probes: the shuffle runs are still learning; continuation proposed (implementer; the user decides)

**`check_wave.py` fixed.** It crashed on wave 2 (`None >= float`): the records write a NaN as null, and a quadrant has no
measured state at the first checkpoints (no context of it in the replay yet): seed B's f = 1 reference at 1,000 and
1,585 (nw), F at 1,000-3,981. Wave 1 had not met it because seed B's pair was excluded; E and F are paired with that
reference. A value not measured is now flagged (`NOT MEASURED`, `criterion not computable` at those checkpoints), never
fatal (test). Two more corrections, seen on these exports: the LOCK-IN alert now needs the statistic below 0.01 bit on
at least 95% of the evaluations from 6,100 on (seed B's experimental run: 98.6%, 19 isolated values up to 0.042 bit,
which the "every evaluation" rule of (32) missed), and a run never switched or not done is excluded from every pair, as
in the probe plan.

**Validity at 300,000** (light files; `check_wave.py`, `learning_trajectory.py`):

| Run | switch (grad, rule) | after it | success | smallest quadrant mass | mass by type over the 12 assignments (UD / CD / neutral) |
|---|---|---|---|---|---|
| E4 baseline A exp / ref | (E4: 3,051 / 3,451) | -- | -- | 0.994 / 0.994 | -- |
| baseline B exp | 6,101, threshold | min 3.29 bits | 1.00 | 0.997 | -- |
| baseline B ref | 3,401, threshold | min 2.04 | 1.00 | 0.994 | -- |
| f = 1 A exp | 7,151, late | min 1.32 | 1.00 | 0.838 | 0.860 / 0.904 / 0.905 |
| f = 1 A ref | 6,301, late | min 1.67 | 1.00 | 0.785 | 0.811 / 0.738 / 0.831 |
| f = 1 B ref | 6,301, late | min 1.87 | 1.00 | 0.890 | 0.928 / 0.925 / 0.932 |
| f = 1 E exp (402) | 6,301, late | min 2.02 | 1.00 | 0.690 | 0.821 / 0.839 / 0.859 |
| f = 1 F exp (502) | 6,301, late | min 1.38 | 1.00 | 0.774 | 0.853 / 0.757 / 0.841 |

| Pair | b at step 1, UD (canonical) | by type over assignments (UD) | E4's criterion |
|---|---|---|---|
| E4 baseline A | -2.301 | -- | met |
| baseline B | -2.317 (CD +0.012, neutral -0.007, other +0.003) | -- | met from 100,000 on |
| f = 1 A | -1.734 (CD +0.15, neutral +0.24, other -0.15) | -1.62 | never |
| f = 1 E vs B ref | -1.935 (CD -0.51, neutral -0.25, other -0.20) | -1.69 | never |
| f = 1 F vs B ref | -1.789 (CD -0.15, neutral -0.18, other -0.20) | -1.86 | never |

Seed B's own f = 1 pair is excluded (its experimental run locked in). **Baseline B is healthy and equals E4's baseline A**
(mass 0.994-0.997, b(UD) -2.32 vs -2.30, converged from 100,000): E5's procedure is not at fault; the deficit is the
shuffle's. The shuffle runs are alike: every switch by the late threshold, no fall below 1 bit after it, success 1.0,
mass 0.74-0.93 by type, b(UD) -1.6 to -1.9 -- the same stage of learning, short of E4's criterion.

**Trajectory: still learning, not a plateau.** From 200,000 to 300,000 every shuffle run's mass rises (smallest
quadrant +0.08 to +0.17, UD-type mass +0.09 to +0.16 per 100,000 steps), decelerating (UD-type gains per 50,000 steps:
e.g. 0.106 then 0.038 for A's reference, 0.059 then 0.028 for B's), and b(UD) still falls (A: -1.30 at 200,000, -1.73
at 300,000). The baselines converged by 60,000-100,000 steps; the shuffle runs lag by a factor of about 3-5 in steps (12
images per state to learn). Linear extrapolation reaches mass 0.97 at about 340,000-455,000 steps; the deceleration makes
that a lower bound -- likely 450,000-650,000. The deficit is diffuse: every zone type, both rotation classes (A / B) and
every UD-CD relation alike, the canonical assignment included (0.77-0.92 at its UD slot); within a run the three weakest
assignments share one UD slot, a different one in each run (sw, ne, se, nw): a general lag, not a targeted failure.

**Recommendation (implementer): (c) continue the shuffle runs before probing.** At 300,000 the shuffle models are short
of the learning the baselines reached (b(UD) -1.6 to -1.9 against -2.3; their references' masses 0.79-0.93), so a
probe's b would mix where the belief sits with how far it has formed; and the probes' validity criterion (the reference
must predict at step 1, mass >= 0.90) would exclude more shuffle probes than baseline ones, an asymmetric selection.
Normalising (option (b): a probe's b over the same model's b(UD) in its training world) assumes the belief scales
linearly with learning and does not fix the validity asymmetry. Continuing is cheap: from each run's last resume point
(290,000), `run_e5.py --extend` (the run's own config and planned budget, checkpoints every 25,000 and every assignment
every 50,000 past 300,000; identical to the original up to 300,000 on the same device, tested), batch `e5_extend` (A exp,
A ref, B ref, E, F) to 500,000; the continuation of a continuation starts from its own resume point if 500,000 is not
enough. Baselines stay at 300,000 (converged). Cost: 210,000 steps per run, five runs in one pod: about 35 hours (one
GPU; observed 7,700-9,000 steps per hour per run with three or four runs in parallel). The probes then go at the first
common checkpoint where the shuffle pairs meet E4's criterion (`check_wave.py` on the continuations, readable while they
train), the baselines at 300,000 (`e5_probes_all`: baselines at `batches.PROBES_CHECKPOINT`, shuffle pairs at
`--checkpoint`).

**A resume bug, found by the continuation test and fixed (library).** The resume state of `run_one` did not keep the
current episode's row offset in the replay table (`ClosedLoopCollector._obs_offset`, the episode's assignment): a shuffle
run resumed in the middle of an episode of a non-canonical assignment looked its images up in the wrong block and
stopped (`ObservationTableMismatch`). No run of waves 1-2 was resumed, so none is affected; a resume after an incident
would have failed. The offset is now saved and restored (only with an episode source: E3's and E4's resume states are
unchanged); tested (a shuffle run interrupted and resumed equals the uninterrupted run; the continuation equals the
original at 600).
