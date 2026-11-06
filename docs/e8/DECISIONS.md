# E8 — decisions

E8 (therapy and relapse, `docs/E8_therapy_relapse.md`; the user's brief `docs/e8/prompt_e8_therapie.md`). Numbered from
(1); other logs are cited by experiment ("E5 (39)", "E6 (63)"). "User" marks a decision taken in the brief, "implementer"
one taken while implementing it. Decisions (1)-(16) were written on 2026-11-05, before any E8 code.

## 2026-11-05 (1) -- where the code lives (implementer, CLAUDE.md)

`experiments/E8_therapy_relapse/` (runners, analysis, batch table, README, `IMPLEMENTATION_REPORT.md`), `configs/e8/`
(YAML only), `docs/e8/`, `tests/e8/`, `results/E8_therapy_relapse/` (light outputs only: tables, figures, verdict JSON).
Weights, resume states, replays and episode arrays never enter the repository. No absolute path in code or config: the
runs root, the output root and the start runs' root are arguments.

Library changes: new modules, or new options whose default reproduces E3-E7 bit for bit, each with a test pinning the
default. E8 imports the library only, never another experiment's directory; E7 code that E8 needs (the per-step critic
readout of D7, the oracle's q_fn, D7's C1/C3 from the arrays) moves to the library first, in its own commit, E7's scripts
importing it back, verified bit for bit (11).

## 2026-11-05 (2) -- the start checkpoints (user's proposal; implementer for the check)

- **Baselines at 300,000** (the user's proposal, kept). Reasons: (a) it is the checkpoint of E6/E7's 650,000 campaign, so
  E8's checkpoint 0 is checked against `results/E6_phenotypes_latent/all_agents_650k/` and its runs; (b) the weights at
  100,000 are not local (`inventory.csv` lists them as MISSING), so a 100,000 start could not be verified against any
  existing measurement here; (c) 300,000 is the level at which every baseline result of E4-E7 was read.
  **Alternative rejected: 100,000**, whose learning level is closer to the shuffle agents' at 650,000 (E5 (37)'s matched
  comparison). It is not retained: no verifiable checkpoint 0, and E8 compares each agent with its own reference at the
  same therapy step, not baselines with shuffle agents at a matched level. The level difference is declared instead.
- **Shuffle at 650,000** (user). Short of E4's convergence criterion; level declared as in E5 (37).
- **The files** (relative to the runs root, the cluster's layout; the local root is `C:\nrw\e7_650k_root`, junctions to
  the extracted archives, as E6 (63)'s relaunch), each checked by its SHA-256 at the start of a run (a mismatch stops the
  run). Hashes from `experiments/E5_belief_training/analysis_robustness/tables/inventory.csv`; all 10 verified on
  2026-11-05 on both local copies (`C:\nrw\e6all` / `C:\nrw\e5_extends\extracted` and the root above):

  | Agent | Role | Seed | Run (under `helplessness-worldmodels-e4/results/E4_generalization_gradient/` or `-e5/results/E5_belief_training/`) | Step | SHA-256 |
  |---|---|---|---|---|---|
  | base_A | exposed | 2 | `grid12x12__2zones-p1-b-d1__pair-experimental__worldmodel-agent-e4__seed02__20261016T144543Z` | 300,000 | `fc813201...08dc71bd` |
  | base_A | reference | 2 | `grid12x12__2zones-p1-b-d1__pair-control__worldmodel-agent-e4__seed02__20261016T144543Z` | 300,000 | `b738b3fe...310d9d1b` |
  | base_B | exposed | 102 | `grid12x12__2zones-p1-b-d1-base__pair-experimental__worldmodel-agent-e5__seed102__20261024T110418Z` | 300,000 | `d56d22ee...fd2df628` |
  | base_B | reference | 102 | `grid12x12__2zones-p1-b-d1-base__pair-control__worldmodel-agent-e5__seed102__20261024T110418Z` | 300,000 | `d1df54e6...72a902a5` |
  | shuffle A | exposed | 2 | `grid12x12__2zones-p1-b-d1-sh100__pair-experimental__worldmodel-agent-e5__seed02__20261027T094524Z` | 650,000 | `86da724c...fecba154` |
  | shuffle E | exposed | 402 | `grid12x12__2zones-p1-b-d1-sh100__pair-experimental__worldmodel-agent-e5__seed402__20261027T154746Z` | 650,000 | `7653fcea...a948a975` |
  | shuffle F | exposed | 502 | `grid12x12__2zones-p1-b-d1-sh100__pair-experimental__worldmodel-agent-e5__seed502__20261027T154746Z` | 650,000 | `eed6acff...de84a5ca` |
  | shuffle ref. B | reference | 102 | `grid12x12__2zones-p1-b-d1-sh100__pair-control__worldmodel-agent-e5__seed102__20261027T154746Z` | 650,000 | `7cf05785...814429ca` |

  The full hashes go in `configs/e8/agents.yaml`. Shuffle A's own reference (seed 2, `66927663...111c01d0`) is verified but not
  used (3).

## 2026-11-05 (3) -- the pairs (user)

- base_A exposed with base_A's reference (seed 2: E4's `p1-b-d1` pair); base_B exposed with base_B's reference (seed 102).
- Shuffle A (2), E (402) and F (502) with shuffle reference B (102), **one** reference therapy run for the three. A is
  paired with B as in E5 (39): A's own reference is weak on the neutral quadrants (no valid neutral probe).
- Every reference undergoes the same therapy as its exposed agents (its own training environment for the baselines; the
  same partial therapy on q for reference B), so that "exposed minus reference at the same therapy step" separates the
  exposure's trace from what re-training does to any agent.

## 2026-11-05 (4) -- what is reset at the start of the therapy (user; implementer for the details)

The start files are `agent_env<N>.pt`: they hold the weights of the RSSM, goal head, critic and slow critic, the agent
config, the environment and gradient steps and the switch step; they hold no optimiser, replay or stream state (the
cluster's `resume.pt` of these runs are not local). So the therapy cannot continue the training state exactly, and
nothing below is a loss relative to what exists.

- **Weights kept**: RSSM, goal head, critic and **slow critic** (the critic's EMA target) loaded from the start file,
  bit for bit (tested). The critic is kept (user): resetting it would erase the cached value that P2 and the belief /
  cache decomposition follow. The slow critic is kept with it: it is the critic's training target, and replacing it with
  a copy of the critic would change the first targets of the therapy. No technical reason to reset either was found.
- **Replay: empty** (user, 23 October), then E5's prefill: `budget.prefill_steps` = 256 environment steps in the therapy
  environment under the agent's own policy, no gradient step during them (`run_one`'s rule `env_step > prefill`),
  exactly as at the start of an E5 run. The replay's capacity covers the whole therapy (nothing is overwritten).
- **Optimisers: new**, the same for every run: the three Adams of E5 (world model `train.lr` then the schedule below,
  goal head `agent.goal_lr`, critic `agent.critic_lr`, `adam_eps`), built as at the start of E5's runs. The first
  steps of a new Adam are larger relative updates than a warmed-up one's: a transient at the start is expected, in the
  references too (EXPECTED_FORMS, form R).
- **Schedule: post-switch from step 0, counters restarted.** The therapy counts its own environment and gradient steps
  from 0 (checkpoint 0 = the start file). The switch state is set as switched at gradient step 0 (`switch_step = 0`,
  rule kept from the start file and recorded), so `phase_learning_rate` gives `lr_after_switch` (1e-4) at every step,
  the estimator is straight-through from step 0 and the heads train from the first gradient step; beta_rep's warm-up is
  set to 0, so `rep_beta` gives `beta_rep` (0.1) from step 0. This is numerically the schedule the start runs were on:
  every start file's ramp was over (switch at gradient step 3,051-7,151, warm-up 34,539-35,359 steps, files at gradient
  step 179,846 or 389,846: read from the files on 2026-11-05). The origin's switch step, rule and gradient step are
  recorded in the run's meta.
- **The switch cannot fire again**: `SeparationSwitch.update` returns before its switch rule once `switch_step` is set
  (only the collapse guard runs), and `check_deadline` acts only before a switch (tested). The **collapse guard stays
  active**, with E5's same-background statistic (E5 (19)): if it fires, the run stops and the user is asked (brief, §11);
  no E5 hyperparameter is changed to make a therapy run.

## 2026-11-05 (5) -- the therapy's random streams (implementer)

- Training streams (explore, tiebreak, start_state, replay_sample, switch_pairs, latent, act, imagination, the global
  torch RNG): children of a **therapy root** `SeedSequence([seed, 3])`, in E3's `STREAM_NAMES` order. 3 is a new
  entropy word: distinct from the experimental root `SeedSequence(seed)`, the reference root `[seed, 1]` and E5's
  assignment root `[seed, 2]`, so no therapy stream repeats or correlates with a training stream.
- The root depends on the seed only, not on the condition: an exposed baseline and its reference (same seed) draw the
  same starts and the same exploration uniforms step for step (common random numbers, as E5's assignment stream and E6's
  episodes); they diverge as their actions differ.
- The shuffle therapy's assignment stream: `SeedSequence([e8.seed, 4])`, the SAME for every shuffle therapy run, drawn
  once per episode, so the k-th episode of every shuffle run (A, E, F and reference B) gets the same treated assignment.
- The world's dynamics seed comes from the world builder as in E5 (deterministic dynamics: it never acts).

## 2026-11-05 (6) -- the therapy environments (user; implementer for the construction)

- **Baselines**: `two_zones_setup.world(cfg, conf, "reference")` of the agent's configuration and seed -- the very
  environment of the paired E5 reference run (dynamics and images equal, tested), for the exposed agent and its reference.
- **Shuffle on q**: a new episode source (`pipeline.episode_sources`, new class, nothing else changed) over the 12
  assignment worlds of E5 in the **reference** condition (UD -> CD, T_inc kept on the UD slot), drawing per episode
  `u ~ U[0, 1)` then an index uniform among the assignments whose UD slot is q (3 of them; `all`: the 12). With `all`
  the draws equal `ZoneAssignmentShuffle(fraction = 1)`'s on the same generator (tested). The replay table keeps E5's 12
  blocks (row = assignment x 144 + state), so a therapy replay reads like an E5 shuffle replay; the 9 untreated blocks are
  never visited.
- `start_in_zone_frac` (default 0): with probability f, an episode starts uniformly in the treated zone instead of
  uniformly on the non-terminal states. 0 draws nothing more (the start stream is E5's draw, tested). Not used in this
  campaign.

## 2026-11-05 (7) -- the checkpoints (user's grid; implementer for the content)

- Grid (config): 0, 500, 1,000, 2,000, 3,500, 5,000, 7,500, 10,000, 15,000, 25,000, 40,000, 60,000, 100,000, 150,000,
  cut at `max_env_steps` (and the end). 500 falls 244 steps after the prefill (about 146 gradient steps).
- At each: the weights (`agent_env<NNNNNN>.pt`, the format E6/E7's loader reads) and the full resume state
  (`resumable_env<NNNNNN>.pt`: weights, optimisers, replay, streams, progress), kept; plus the rolling `resume.pt`
  (every checkpoint and every `checkpoints.resume_every` steps) used by `--phase resume`. A run can start from any
  therapy checkpoint's resume state (re-acquisition later), the therapy environment being a parameter.
- **Checkpoint 0** is written before the first environment step: its weights are the start file's, bit for bit.

## 2026-11-05 (8) -- the per-step log and the treated zone (user; implementer)

- Every step of the therapy is in the replay (state row, action, reward, flags); at each checkpoint `steps.npz` (state,
  assignment, quadrant, treated-zone flag, episode index; not versioned) is written from it, and the record gets the
  cumulative visits of the treated zone and of every quadrant, the episodes by assignment, the success rate and length.
- **Treated zone**: nw for the baselines (the canonical UD slot of `p1-b-d1`); for the shuffle runs the UD slot of the
  episode's assignment (q during the therapy). Recovery is read against environment steps, gradient steps and these
  visits.

## 2026-11-05 (9) -- the health record at the therapy checkpoints (implementer)

E4's `real_context_record` (the prior's mass on the true successor and I_s by region, E4's checkpoint settings) on the
therapy world, from the run's own replay (shuffle: by treated assignment, E5's `real_context_by_assignment`); own
measurement slots of the therapy root (26 and 27; E3 1-16, E4 20-21, E5 22-25 unused here); nothing drawn from a
training stream (the measures never change the trajectory, tested by the resume test). No probe walks during training:
they belong to `--phase probes`.

## 2026-11-05 (10) -- the measures of `--phase probes` (user's list; implementer for the definitions)

Frozen weights, CPU or GPU, the test environment (reference dynamics). For a run and checkpoint t, the exposed model at
t is measured with its reference **at the same therapy checkpoint t** (b(t) = I_exp(t) - I_ref(t)); I_exp and I_ref are
also reported apart. Images: baselines, the training image (zone nw); shuffle, the 12 assignment images (E5's
`assignment_colors`), zone = the assignment's UD slot (where T_inc is), split treated (UD slot = q) / untreated.
The measurement seeds are the same at every checkpoint (only the weights change along a curve).

1. **b at step 1**, E5's 12-assignment protocol: `pipeline.assignment_records.walks_by_assignment`'s walks (E4's fixed
   walks, `e4.checkpoint.n_walks` = 100, purpose `checkpoint`, image index 0 for the canonical assignment and 1000 + k
   otherwise), on the measured zone only (each slot has its own seed, so its value is the one a full call gives). Validity
   (E4's criterion, `probing.probe_validity`): the reference's step-1 mass on the true successor >= 0.90 and
   |I_ref - log2 5| <= 0.15, from the same walks (an option of `walks_by_assignment` keeping the step-1 summary, off by
   default). For the training image also E4's probe (`frozen_setup.probe_validity_pair`, 150 walks), as in E6.
2. **C1 / C3**, D7's definitions (E6 (62)): per zone step of E6's episodes, C1 = the strict argmax of V(h'_a, z_bar)
   is directed, C3 = the greedy directed share (the training's tie set) of Q with a common h; per (model, image) the
   mean over the zone steps measured and well localised (`loc_template` = 1), as E6 (63); also over all zone steps.
3. **Behaviour**: E6's episodes (`frozen_setup.run_model`, E6's seeds, `e6` block of `configs/e6/e6.yaml` unchanged: 20
   starts on each of the 9 interior states, H = 36, epsilon of the checkpoint (0.1), k = 1), run once with D7's
   counterfactual q_fn (which returns the policy's own Q, checked bit for bit), so one run gives the behaviour, b at step 1
   of the episodes (`belief_row`), C1/C3 and the localisation. Directed share (main), recovery latency and zone share
   (`zone_phenotypes`).
4. **Localisation**: by the decoder (`loc_template`, and the quadrant of `decoded_template` on mislocalised zone steps);
   from the true cell: the share of directed actions judged from the true cell and from the decoded cell, on the well and
   the badly localised zone steps (as `free_offsets_650k.py`, moved to the library with it).
5. **Oracle at k = 1** at the oracle grid (config; default 0, 2,000, 10,000, 40,000 and the last checkpoint): the
   exposed model's E6 episodes on the training image with the policy on the oracle Q (`lookahead.k_step_q`, E7's
   lookahead phase), for belief effect = oracle(k=1) - exposed and cache effect = reference - oracle(k=1) (the 650,000
   campaign's definitions, on the training image).

## 2026-11-05 (11) -- E7 code moved to the library (implementer, CLAUDE.md)

`run_e7_d7.natural`'s q_fn (and its `clone` / `assemble`), `run_e7_diagnostics`' oracle q_fn and `episode_row`,
`run_e7_dissociation.seed_of`, `analyse_e7_d7.c_measures` and the true-cell computation of `free_offsets_650k.py` move to
a new library module (`probing/`, torch side; the array-only part to `measures/`), unchanged; E7's and E6's scripts
import them back under their old names. Verified bit for bit on a reduced D7 natural run and a reduced lookahead run
(outputs before and after the move, same machine), in a commit of its own before any E8 use.

## 2026-11-05 (12) -- cost estimate (implementer; refined after the first timed job)

- **Training** (cluster pod, 3 CPUs): E5 ran about 6,200 environment steps per hour per run with 5 runs on the pod
  (E5's report, 650,000 continuation). Wave 1 (4 runs x 150,000): about 20-25 h of wall time, about 60-75 CPU-hours.
- **Probes** (local CPU, 1 thread per process, up to 12 processes): per (model, image, checkpoint) about 380 s for a
  shuffle image (walks on one zone about 200 s, episodes with the readout about 150 s, the reference's fresh I about
  30 s) and about 660 s for a baseline image (plus E4's 150-walk probe, about 250 s). Wave 1: 4 models x 1 image x 14
  checkpoints, about 10 CPU-hours (under 1 h on 12 processes). Wave 2: (3 exposed + 1 reference) x 12 images x 14
  checkpoints = 672 units, about 71 CPU-hours (about 6 h on 12 processes). Oracle: 60 s per set, negligible.
- So the probes cost about as much as the training (wave 2: about 71 against about 75 CPU-hours), not markedly more:
  no reduction is proposed now. Reductions available if the timed job says otherwise (not applied without the user):
  the 12 assignments at a sub-grid of checkpoints and the 3 treated ones at every checkpoint, as E5 (18); 10 starts per
  interior state instead of 20.

## 2026-11-05 (13) -- the analysis's definitions (implementer; thresholds in EXPECTED_FORMS)

The value of a curve at checkpoint t: baselines, the training image; shuffle, the mean over the valid treated (resp.
untreated) assignments at t, with the count of valid ones. Gaps for C1, C3 and directed share: g(t) = reference(t) -
exposed(t); for b, b(t) itself. t½, crossings, censoring, trend, step and transfer as written in EXPECTED_FORMS before any
computation; on three abscissae (environment steps, gradient steps, treated-zone visits) read off the same checkpoints.

## 2026-11-05 (14) -- the tabular twin: blocked (brief, §4.5 and §11)

The start runs' count tables (`counts.npz`, written by `run_one` at the end of a run) are **not available locally**: the
batch archives (`e5_wave1`, `e5_wave2`, `e5_extend_a`, `e5_extend_b`) and the local copies hold `replay_final.npz`,
`checkpoints.jsonl` and weights, never `counts.npz` (E5's `export_batch.py` leaves it out); E4's base_A run has weights
only here. As the brief says, they are **not reconstructed** (not even from `replay_final.npz`). The twin waits for the
user: do the `counts.npz` of the four baseline runs exist on the cluster's `/scratch`?

## 2026-11-05 (15) -- batches (user)

`e8_w1_base` (base_A and base_B, exposed and references, full exposure: 4 runs, budget 150,000; the pilot),
`e8_w2_shuf_nw` (shuffle A, E, F and reference B, `treated_quadrant: nw`: 4 runs, budget set after wave 1), and
`e8_w3_shuf_ne`, `e8_w4_shuf_sw`, `e8_w5_shuf_se` generated from the same table by the parameter, never launched without
the user. One 3-CPU pod; `--dry-run` first; `nadim-launcher` stopped during the runs; results back by
`export_batch.py create --batch ...`, analysed locally.

## 2026-11-05 (16) -- local long runs (user's model)

`--phase probes` on the local CPU is chained like `experiments/E6_phenotypes_latent/chain_all_cpu.py` (scheduled task,
heartbeat in `STATUS.txt`, `KeyboardInterrupt` and every exception written as FAILED with its cause, phases already done
skipped on a restart); a first short job must finish before the full chain is launched.

## 2026-11-05 (17) -- amendments to (9) and (11), written while coding, before any E8 run (implementer)

- **(9), measurement streams of the health record.** E4's `real_context_record` and E5's `real_context_by_assignment`
  draw from their own measurement slots of the RUN SEED's measure stream (E4's slot 20, E5's slot 22, keyed by the
  checkpoint), not from new slots of the therapy root as (9) wrote. They are reused unchanged rather than given a stream
  option: a measurement stream never touches a training stream, so the therapy's trajectory does not depend on it either
  way (tested by the resume test). The health record is skipped at checkpoint 0 (empty replay: no real context).
- **(11), one more function moved**: `apply_e5_procedure` (E5's changes to E4's procedure: the same-background switch
  statistic, which also drives the collapse guard, the later separation deadline, the late threshold) moves from
  `experiments/E5_belief_training/run_e5.py` to `pipeline/two_zones_setup.py`, body unchanged; `run_e5.py` imports it
  back. The other moves landed as listed: `probing/critic_readout.py` (readout and oracle q_fns, `assemble`, the generator
  clone, the E7 seed derivations), `measures.decision_rule.critic_readouts` (D7's C1/C2/C3) and
  `measures.zone_phenotypes.localisation_split` (the decoder / true-cell split of `free_offsets_650k.py`).
  Verified on 2026-11-05, same machine, CPU: `run_e7_d7.py --phase natural --no-overlay --image training` and
  `run_e7_diagnostics.py --phase lookahead`, base_A, `--n-reps 1`, before (main) and after the move: the 29 and 52
  arrays of `episodes.npz` equal (`np.array_equal`, NaN equal) and the summary rows equal (durations aside); the array
  functions against verbatim copies of their old bodies (`tests/e8/test_e8_library_moves.py`).
- **Reference measured with its partner's seeds.** E4's walk seeds depend on the pair's seed (`measurement_seed` reads
  `conf["seed"]`), so reference B is measured by the walks once per exposed partner, with that partner's seed (common
  random numbers within the pair, as E5 (39)); E6's episode seeds do not depend on the agent, so its episodes are run
  once per checkpoint and image. Cost: wave 2's walks count 3 + 3 models x 12 images per checkpoint (12): about +20 %.

## 2026-11-05 (18) -- the probe jobs and the images' seeds (implementer, before any probe)

- **One E6 run per (model, image, checkpoint)** with E6's `measure: all` and D7's readout q_fn
  (`probing.critic_readout.readout_q_fn`, which returns the policy's own Q, checked): its episodes are E6's, its I and
  localisation E6's (the readout draws from clones of the act generator and from its own z_bar generator, never from the
  measurement generator), so at checkpoint 0 it reproduces both the campaign's `e6_<agent>_training` rows and its
  `c12_<agent>_training` C1/C3. The oracle reuses that run's reference episodes and fresh I (as E7's lookahead phase:
  reference with `measure: all`, the oracle's episodes with `measure: first`, its generator `diagnostics_seed(e7
  diagnostics seed, "lookahead", k)`). E6's place / history split (`fresh_split`) is not computed (not an E8 measure).
- **Image indices** (E6's episode seeds read the image index and the zone's slot): the canonical assignment is the
  training image, with E4's index (so checkpoint 0 is the campaign's); the 11 others use 3000 + k, E7's convention for the
  12 assignment images (`run_e7_context.py`'s collect). The walks' seeds are E5's (image index 0 for the canonical
  assignment, 1000 + k otherwise; `assignment_records`).
- **Job unit**: (the pairs sharing a reference, one checkpoint): the reference's episodes and fresh I are computed once
  per image and checkpoint and serve its three shuffle partners; its walks once per partner seed ((17)); E4's 150-walk
  probe of the training image once per pair (its seeds are the pair's). Outputs per job:
  `probes__<reference model>__k<NNNNNN>__<UTC>/` (rows.csv / rows.json: one row per pair, image and role; episodes.npz;
  meta.yaml with the weights' SHA-256 and their hash before / after; status.json).
- C1 / C3 use the model's own tie tolerance (`agent_config.tie_tol_abs`, 0.0751 for every E4/E5 run: E7's
  `analyse_e7_diagnostics.TIE_TOL`).

## 2026-11-06 (19) -- E7's lookahead keys fixed; E8's oracle checked (user, after the report of 2026-11-05)

- **The bug (E7, finished).** `run_e7_diagnostics.phase_episodes` saved the lookahead's k-step episodes of the exposed
  model and of its reference under one key (`none__0__k<k>`): in `episodes.npz` the reference's overwrote the exposed
  model's. The rows of `summary.json` were computed before the save: no published value changes.
- **The fix, minimal (user).** `episode_tag` gives the lookahead's arrays a model suffix (`none__0__k<k>__exposed`,
  `__reference`); the induction's and softmax's keys are unchanged; no analysis, slide or paper script read these arrays
  (checked). Tested (`tests/e6/test_e7_lookahead_keys.py`). Reduced E7 rerun (base_A, `--n-reps 1`, CPU) before / after:
  the 6 summary rows bit for bit identical; the old key's arrays equal the new reference key's; the new exposed arrays
  differ from them and give exactly the exposed row's directed share, at k = 1, 5 and 10. Nothing of the campaign is
  rerun. Erratum: `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`, section 16.
- **E8 does not inherit it.** The library's `critic_readout.oracle_q_fn` computes Q only and stores nothing; `probes_e8.py`
  runs the oracle on the exposed model only and saves it under `<pair>__<image>__oracle__*` (one key per pair and image);
  the checkpoint-0 check of base_A found E8's oracle arrays at the exposed row's directed share (0.8752, not the
  reference's 0.9387). A guard now stops a probe job on any second write to a key of its `episodes.npz`; the shared
  reference's arrays are saved once per image (its rows stay one per pair).
