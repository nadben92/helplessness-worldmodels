# E8 — expected forms, thresholds and predictions

**Written and committed on 2026-11-05, before any E8 computation** (brief, §6). Every analysis added later gets its
forms here first, in a new commit before it is computed. Verdicts are reported as computed; thresholds are never changed
after a result is seen; what is exploratory is marked so. Definitions of the measures: `docs/e8/DECISIONS.md` (10),
(13).

## 0. Notation

- t: a therapy checkpoint (environment steps since the start of the therapy; checkpoint 0 = the start file). T: the last
  checkpoint of the run (its budget).
- For a model (exposed "exp", reference "ref") at t, on an image: b(t) = I_exp(t) - I_ref(t) at step 1 in the zone
  (12-assignment protocol for every image; also E4's probe and E6's episodes on the training image); C1(t), C3(t), D(t)
  (directed share, E6's protocol); localisation by the decoder L_dec(t) and the directed share from the true cell.
- Gaps: g_X(t) = X_ref(t) - X_exp(t) for X in {C1, C3, D}.
- Value of a curve at t: baselines, the training image (zone nw); shuffle, the mean over the **valid** treated
  assignments (UD slot = nw: 3 images), and separately over the valid untreated ones (9), with their counts. A point is
  valid when the reference at t passes E4's criterion on that image's zone (mass >= 0.90, |I_ref - log2 5| <= 0.15).
- Curves are read against three abscissae, all from the same checkpoints: environment steps (main), gradient steps,
  visits of the treated zone. Every verdict below is stated in environment steps; the other two are reported.

## 1. Forms (implementation and measurement checks)

A failure points to an implementation or measurement problem, is reported, and stops the analysis until understood.

**F0 -- identity at checkpoint 0.**
- (a) Same machine, CPU, 1 thread: on the training image, every quantity measured at checkpoint 0 equals the 650,000
  campaign's (the job outputs of `C:\nrw\wt-e6\scratch\all\`, which `results/E6_phenotypes_latent/all_agents_650k/`
  summarises): E6's episodes (states, actions, length: equal arrays), b at step 1, directed share, latency and zone
  share (`e6_<agent>_training`), C1 and C3 per step and their means (`c12_<agent>_training`), the oracle's directed
  share at k = 1 (`oracle_<agent>`). **Exact equality** (values equal as floats; arrays `np.array_equal`).
  For the shuffle agents the campaign's reference was reference B at 650,000 (A, E, F alike): the pair at checkpoint 0
  is the campaign's pair.
- (b) Where a campaign value does not exist (the 11 non-canonical assignment images, E5's 12-assignment walks on CPU):
  a direct call of the existing code (E5's `walks_by_assignment`, E6's `run_model`) on the start files, same seeds, gives
  the same values exactly.
- (c) Against E5's own training records at 650,000 (`checkpoints.jsonl`, `e5.by_assignment.<k>.I_step1_by_type.ud`,
  computed on the cluster's GPU): |difference| <= 0.05 bit for each model and assignment (other device: CLAUDE.md's
  tolerance rule). Baselines: E5's record of base_B's run at 300,000 (canonical walks), same tolerance; base_A (E4's
  run) has no local record and is checked by (a) and (b) only.
- (d) The weights of the therapy's checkpoint 0 are bit for bit the start file's (rssm, goal head, critic, slow critic).

**R -- the reference stays a controllable agent.** For every reference run, on every image whose point is valid at
checkpoint 0:
- |I_ref(t) - log2 5| <= 0.15 at step 1 in the zone, and C1_ref(t), C3_ref(t), D_ref(t) within 0.05 of their value at
  checkpoint 0, at **every checkpoint t >= 5,000**;
- before 5,000, a transient is allowed (new optimiser, empty replay; DECISIONS (4)); it is reported (depth and
  checkpoints), and the values must be back within the bounds above from 5,000 on.
- A reference point that becomes invalid at t >= 5,000 is a failure of R (reported with the image).

**M -- the exposed agent does not get worse in the treated zone.** On the exposed run, treated zone (baselines: the
training image; shuffle: the treated mean), for X in {b, C3, D}, over the checkpoints:
- Spearman's rho between t and X(t) >= 0 ("non-decreasing in trend"), and
- maximal drawdown (the largest drop of X(t) below the maximum of X over the earlier checkpoints) <= 0.5 bit for b and
  <= 0.15 for C3 and D.

**TW -- tabular twin.** Its T_hat equals the count mixture (N_start + N_therapy) / (n_start + n_therapy) exactly
(`np.array_equal` on the counts it is built from). N_start rebuilt from the start run's `replay_final.npz` (DECISIONS (20)), checked first by TW0-TW3
(written 2026-11-06, before any count is computed; baselines only):
- **TW0 -- accounting.** Transitions retained = the run's logged environment steps (`meta.yaml` `env_steps`, the last
  record of `checkpoints.jsonl`), line by line: stored steps = retained transitions with a stored successor + pending
  transitions (0 or 1, completed from the run's environment) + end observations; end observations = completed
  episodes = `checkpoints.jsonl`'s `episodes`; terminal end observations = its `successes`; every end observation is
  followed by an `is_first` (or ends the replay) and every `is_first` but the first follows an end. Any other gap fails.
- **TW1 -- the exposed conveyor.** On the cells of the exposed run's UD zone (nw for `p1-b-d1`), every counted
  transition goes to the conveyor's successor of its cell (the row P_exp[s, a] of the run's environment, the same for
  the 5 actions), whatever the action: zero counts elsewhere, and every zone cell with counts has them under at least
  2 distinct actions.
- **TW2 -- the reference's same zone.** On the same cells in the reference run, the successor depends on the action:
  for every cell, distinct counted actions have distinct counted successors (each action one successor), so every cell
  counted under >= 2 actions has >= 2 distinct successors.
- **TW3 -- agreement with the dynamics.** Every counted transition of either run, anywhere, has P[s, a, s'] = 1 in its
  run's environment (deterministic dynamics), and the rebuilding function equals `run_one`'s `counts.npz` at every
  checkpoint of a mini run (`np.array_equal`; tested).

## 2. Recovery times (definitions used by the predictions)

- **t½ for b** (exposed, treated zone; defined when b(0) <= -0.5 bit): the first checkpoint t > 0 with
  b(t) >= b(0) / 2.
- **t½ for X in {C1, C3, D}** (defined when g_X(0) >= 0.10): the first checkpoint t > 0 with g_X(t) <= g_X(0) / 2.
- **Sustained t½** (secondary, the same definitions): the first t after which every later checkpoint also satisfies the
  condition. Reported beside the main t½; no verdict reads it.
- **Crossings** (reported): b(t) > -0.5 bit (E6 (63)'s "in control" class); g_D(t) <= 0.05 (behaviour at the
  reference's level).
- **Censoring**: a t½ or crossing not reached by T is reported as "censored at T" (> T), never extrapolated. A quantity
  whose t½ is undefined at checkpoint 0 (b(0) > -0.5 or g(0) < 0.10) is reported as "nothing to recover".
- Resolution: t½ is a checkpoint of the grid, so two t½ are equal or ordered at the grid's resolution only.

## 3. Predictions (scientific hypotheses)

For each: the operational criterion, then the reading of both outcomes. Per agent; shuffle agents on the treated
assignments unless said otherwise.

**P1 -- the correction holds.** Holds for an agent when (i) the t½ of b is not censored, (ii) b(T) > -0.5 bit, and
(iii) after the first checkpoint with b > -0.5, b never falls back below -1.0 bit.
- Holds: re-training in the controllable world rewrites the transition belief of the treated zone, and the new belief
  is stable while the therapy goes on.
- Does not hold: either the belief does not move within the budget (censored: the old belief resists the new data, or
  the budget is too short -- read with the visits of the treated zone), or it rises then falls back (the correction is
  not stable). Each case is reported as such.

**P2 -- behaviour lags the belief, more where the cache is heavy.** Lag(agent) = t½(D) - t½(C3), in environment steps
(a censored t½ enters at T, and the lag is then a bound, reported as such).
- (a) Lag > 0 (behaviour after the planner's reading) for at least 4 of the 5 exposed agents.
- (b) min(lag of shuffle A, lag of shuffle E) > max(lag of base_A, base_B, shuffle F). If both t½ are censored for an
  agent its lag is unknown and (b) is not decidable for it (reported).
- Holds: the critic, trained only in imagination through the prior, follows the corrected belief with a delay, and the
  delay grows with the cached value (E7's decomposition at 650,000: cache effect 0.22-0.24 in A and E, 0.06-0.16 in the
  baselines and F).
- Does not hold: lag <= 0 (behaviour moves with or before the planner's reading: the cache is rewritten as fast as the
  belief), or a lag that does not order the agents by their cache (the cache's weight at 650,000 does not predict the
  delay).
- Belief / cache decomposition at the oracle checkpoints (descriptive, no verdict): belief effect(t) = D_oracle1(t) -
  D_exp(t), cache effect(t) = D_ref(t) - D_oracle1(t); expected shape: the cache effect of A and E falls more slowly than
  that of the baselines and F (its share of its checkpoint-0 value at the last oracle checkpoint).

**P3 -- recovery as a step.** For an agent with a gain of D over the therapy >= 0.20 and a gain of b >= 0.5 bit
(gain = value at T minus value at 0): let s_X = the largest increase of X between two consecutive checkpoints divided by
X's total gain.
- Holds: s_D >= 0.5 and s_b < 0.5 (the directed share moves in one interval of the grid, b over several), as E6's
  class relation (behaviour follows the class of b, not its value) predicts.
- Does not hold: s_D < 0.5 (behaviour recovers gradually) or s_b >= 0.5 (b too moves in one interval: no gradual belief
  to compare with). Agents with smaller gains are reported, without verdict.
- Limit (written before): the grid is log-spaced, so late intervals are long; a jump in a long interval may be a
  gradual change under-sampled. The (b, D) points of every checkpoint are also placed on E6's 650,000 cloud
  (descriptive): expected, the points stay near the exposed cluster while b <= -1.5 and move to the reference cluster
  once b > -0.5, with few points in between.

**P4 -- transfer in the shuffle agents (open).** For a shuffle agent at checkpoint t, on the untreated assignments whose
point is valid with b(0) <= -1.0 and on the treated ones: recovered fraction of b, f_b(t) = 1 - b(t) / b(0); likewise
for D with its gap, f_D(t) = 1 - g_D(t) / g_D(0) (when g_D(0) >= 0.10). Transfer index at T: TI_X = mean f_X over the
untreated / mean f_X over the treated (defined when the treated mean f_X >= 0.25).
- TI_D >= 0.5: **transfer** -- the cure carried by the colour (T_inc is the same colour in the untreated assignments):
  re-learning T_inc as controllable in one quadrant changes the belief tied to that colour elsewhere.
- TI_D <= 0.2: **no transfer** -- the cure attached to the context where it took place (quadrant and arrangement), the
  untreated contexts keep their belief: relapse (renewal) expected there.
- Between: partial, reported with the numbers. TI_b is reported beside TI_D (the verdict reads D, as E6/E7: behaviour
  over the decoder's reading). Per agent; then "transfer" if 2 of 3 shuffle agents agree, "no transfer" likewise,
  otherwise mixed.
- Untreated assignments are also split by where T_inc is (each of ne, se, sw) and by whether nw is neutral or CD in
  them (descriptive).
