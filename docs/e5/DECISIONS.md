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
