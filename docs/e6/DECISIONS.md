# E6 and E7 — decisions

E6 (the behavioural signature of the controllability belief) and E7 (the causal test: a correction of h) share their
protocol and their code, so they share this file. Numbered after `docs/e5/DECISIONS.md`'s (32). "User" marks a decision
taken by the user (the design of 2026-10-23); "implementer" one taken while implementing, within that design, recorded
here before any agent of E5 is measured.

## 2026-10-23 (33) -- where the code lives (implementer, CLAUDE.md)

The design asked for `e6/` and `e7/` folders, `DECISIONS.md` and `e6/IMPLEMENTATION_REPORT.md`. The repository's
convention (CLAUDE.md) is kept: orchestration in `experiments/E6_phenotypes_latent/` and `experiments/E7_latent_manipulation/`,
configs in `configs/e6/` and `configs/e7/`, this file in `docs/e6/`, the report in
`experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`. Everything both experiments need is in the library (new
modules only): `measures/zone_phenotypes.py`, `measures/monotone_fit.py`, `interventions/latent_direction.py` (numpy);
`probing/frozen_episodes.py`, `probing/belief_steering.py`, `probing/context_walks.py` (torch / walks);
`pipeline/frozen_setup.py` (agents file, test images, seeds, validity, rows). E7 never imports E6: both import the
library. One submission script (`experiments/E6_phenotypes_latent/submit_kubeflow.py`, phases `e6`, `e7-directions`,
`e7-apply`) runs E7's scripts by path, as a command, never as an import. One change to an existing module:
`probing.probe_validity.decoded_agent_cell` split out of `reconstruction_diagnostic` (the same operations; E4's and E5's
probe tests pass, E5's probe identity test included).

## 2026-10-23 (34) -- the test images and the zone of interest (user's list; implementer for the zone)

The images are E4's probe images (`probing.theme_probes`, the canonical layout), given by name in `e6.images` (a
parameter, set after E5's probes): axis `training`, `position_only` or one of E4's axes with its point k, and a position
(ud, cd, neutral). **The zone of interest is the slot the probe changes, as in E4** (for the training image, the slot of
the given position). It is the former UD slot ("zone AI") for the training image and position only. E4's "theme only"
probe colours the NEUTRAL slot T_inc and keeps the training image elsewhere (the UD slot stays T_inc), so its zone is the
neutral slot: starting the agent at the UD slot of that image would measure the training image again. The development
list's "neutral image" is the training image with the neutral slot as zone (a position never uncontrollable, the
corridor's colour). Starts: the nine states of the zone's central 3 x 3 square, `e6.n_reps` (20) episodes each, episode
i at start i mod 9.

## 2026-10-23 (35) -- the speed of behavioural recovery, fixed before any agent was measured (implementer)

Curve: the directed share at the j-th step of the FIRST stay in the zone (j = 1 .. T_exit), pooled over the episodes, with
the number of episodes still in the zone at each j. Scalar summary, `recovery_latency` = **j_theta, the first index at
which the pooled directed share reaches theta**, linearly interpolated between consecutive indices (1 if the first step
already reaches it), read only on indices with at least n_min = 20 episodes, in order up to the first index below n_min;
**theta = halfway between chance and the epsilon-greedy ceiling**: chance = 2/5 (inside a quadrant two of the five actions
bring the agent closer to the goal, in every quadrant), ceiling = 1 - epsilon + epsilon x 2/5; theta = 0.67 at
epsilon = 0.1. Censored (never reached on the readable indices): NaN with a flag; in the curves a censored latency enters at
its first unreadable index (max_index_read + 1), flagged.

Why: the phenotype must measure how fast behaviour becomes directed, in steps, on a scale shared by every agent and
image (absolute, not relative to the agent's reference: the references' own points are on the curves). A slope over the
first k steps is not monotone in recovery (a reference already at the ceiling has slope 0, like an agent that never
recovers) and is biased by survivorship (the episodes still in the zone at large j are the undirected ones); a threshold
crossing reads only the indices up to the crossing, where most episodes are still in the zone, and its two anchors
(chance, ceiling) are interpretable and fixed by epsilon. Recorded in `measures/zone_phenotypes.py` (module docstring)
before the first run on E4's weights.

## 2026-10-23 (36) -- the episodes' random streams (implementer)

Per (image, zone), seeds from `SeedSequence([e6.seed, image index, slot index, purpose])`, the SAME for every agent and
for both models of a pair (common random numbers). Per episode, one numpy generator (spawned children, so episode i's
stream does not depend on the batch) draws at EVERY step u ~ U[0, 1) (explore if u < epsilon), a uniform action and one
uniform score per action (the tie-break: the highest score in the tie set): the exposed model and its reference see the
same starts and the same epsilon draws step by step, whatever their actions. The training's `act()` draws its tie-break
only on greedy steps: same law, not the same draws. The torch draws use two generators: act (the posterior draws and the
lookahead's prior draws) and measure (I_s's draws), so the measures never change the behaviour (measure: all, zone or
first give the same episodes; tested). E7's episodes call the same function with the same seeds, so alpha = 0 reproduces
E6 exactly (tested; checked on the real runs by `analyse_e7.py`).

## 2026-10-23 (37) -- the belief at step 1, b_local and the reference's points (implementer)

b at step 1 = I_exposed(t = 0) - I_reference(t = 0), each by state then zone over the nine starts, on the episodes of
the pair (same starts). The reference's own rows have b = 0 by construction (I_ref - I_ref) and are in the curves
(`e6.analysis.include_references: true`): they are (agent, image) points of a fully controllable training history. b_local at
every step = the model's I_s at that step minus the reference's I_s at the FIRST step of an episode started at the same
state (`reference_fresh`: every non-goal state, 8 posterior draws): a later step of the reference is not available at the
same state and context (the two models' trajectories differ). Under an E7 intervention, b = I_model(intervened, t = 0) -
I_reference(not intervened, t = 0).

## 2026-10-23 (38) -- validity and the coherence check (implementer, E4's criterion)

A pair (agent, image) is valid if the agent's REFERENCE passes E4's criterion on E4's probe of that image and zone:
mass on the true successor >= 0.90 and |I_ref - log2 5| <= 0.15 at the first step in the zone. Computed by running E4's
probe (`probing.theme_probe_measure.measure`: the first `e6.validity.n_walks` = 150 of E4's fixed walks -- E5's -- E4's
n_in rule, E4's measurement seeds with the agent's seed): the same values as E5's probes of the same pair on the same
device. The same measure gives E4's b at step 1 (corridor walk, then the zone's border), against which E6's b at step 1
(fresh start in the zone's centre) is compared (`coherence_gap`); a large gap is reported, never corrected. Invalid pairs
are listed (`invalid_pairs.csv`) and left out of every curve.

## 2026-10-23 (39) -- the policy during the episodes (implementer; phase 0)

No evaluation mode exists: the training's policy is used with frozen weights -- the one-step lookahead's Q
(`policy.q_values`, k_act = 16 prior draws per action), epsilon = the checkpoint's `agent_config.epsilon` (0.1 for every
E4/E5 run; `e6.epsilon` overrides it), the training's absolute tie tolerance (`policy.tie_set`, 0.0751), the posterior drawn
straight-through (the training's filter after its estimator switch). The test environment's successors come from its
`transition_tensor()`, which the library keeps consistent with `step()`.

## 2026-10-23 (40) -- what phase 0 measured on the RSSM, and what E7 does with it (implementer)

On E4's eight models (`IMPLEMENTATION_REPORT.md`, phase 0, point 5): the posterior's probabilities depend on h far less
than on the image (JS to the matched posterior 0.1-0.6 bit with an h from another state's context, 0.20-0.68 bit with the
first step's h, against 3.4-5.1 bits between two neighbouring states' images; 92-98% of the variables keep their mode; a
perturbation of h of 50% of its norm moves it by at most 0.05 bit), and its mode identifies the 144 states. But the
DECODER reads h: with the exposed model's own h from a walk in the UD zone the decoded position is right in only 52-93% of
the cases (the "55-91%" of E4), 100% with the first step's h. Consequences for E7: the target posterior is computed with
the unmodified h (the bias is small); the localisation under intervention is measured on the INTERVENED (h, z) -- a
correction of h can move the decoded position while z stays the image's.

## 2026-10-23 (41) -- where and how h is corrected (implementer)

The correction is added to h_t after the posterior draw: z_t is drawn from the unmodified h_t (z is recomputed from the
image, never modified); the modified h_t is the one the measures, the policy (lookahead) and the recurrence
(h_{t+1} = GRU(h_t, z_t, a_t)) use, at every step in the zone (`e7.apply_in: zone`; `all`: every step). This is the
design's h_{t+1}^(a) = f(h_t + dh, z_t, a). The loss is the design's (sum over actions of KL(target || prior), summed over
the variables, + lam |dh|^2, no pixel loss); lam = 0.01, Adam lr 0.05, at most 1,000 steps, plateau = every situation's
loss changed by less than 1e-4 (relative) over 50 steps. The common direction v: the mean of the ESTIMATION half's
corrections on `e7.direction_images` (default: the training image), per agent and task, applied at the corrections' mean
norm (v_hat); the random control: a random direction of the same norm, fixed per agent and task. Tasks: forward (the
exposed model, target = the test environment's successors) and reverse (the reference model, target = the exposed
TRAINING dynamics, on the images whose zone is the former UD slot only: elsewhere the training dynamics are
controllable). Stepwise: the correction optimised at every step on the current h (at most 300 Adam steps,
`e7.stepwise_max_steps`, for its cost). alpha = 0 is measured once per image (mode `none`; `reverse_none` for the
reference).

## 2026-10-23 (42) -- the situations of E7 (implementer)

Every state of the zone of interest, 10 context walks each (`probing.context_walks`): a corridor start, 8-32 corridor
steps (random, or heading for the goal), an entry across each side of the zone in turn (north, east, south, west; the
side's border cell nearest to the target), the shortest path inside the zone to the target. The model filters the walk
from its initial state; the situation is the last step. Contexts of even index form the estimation half, odd the test
half. A situation is optimised only if the model localises the agent there (nearest template of the decoded (h, z)).
These contexts start in the corridor and reach the state with a history; E6's episodes start in the zone without one.

## 2026-10-23 (43) -- the analysis's settings (implementer)

Monotone curves: isotonic regression (direction: the smaller SSE), its smooth version (PCHIP through the blocks), the
band of a bootstrap over the AGENTS (1,000 resamples). Strict monotonicity: at each grid point b (0.01 bit), the change
over [b, b + 0.25 bit] is in the main direction and larger than 2% of the phenotype's observed range in at least 95% of
the resamples defined there; the invertible range is the longest run of such points. Held out: by condition (the design's
test) and by agent (finer), the curve fitted without the group, its points' residuals and the blind test (b predicted
from the phenotype by inverting that curve). Localisation: E4's nearest-template rule at step 1, threshold 0.9 for
"poorly localised" (`e6.analysis.localisation_min`). The development agents' conditions (dev_pair1, dev_pair2) are
artificial labels on E4's four baseline configurations, only to exercise the per-condition code.

## 2026-10-23 (44) -- one main phenotype (user, after the first report)

The first exit time does not separate a directed agent from a wandering one (from the zone's centre, the shortest path
to the goal follows the inside of the quadrant to its corner nearest the goal): it stays in the rows (`exit_median_km`,
`exit_censored_share`; points.csv), out of the curves and the figures. **Main phenotype: the directed share in the zone.**
No time to the goal (it carries the same information). Secondary, unchanged: recovery latency j_theta (with its
censoring), share of steps in the zone, stay share, revisit share. The roles are in the config
(`e6.analysis.phenotypes`: primary, secondary, logged), not in the library. E6's protocol is unchanged (starts, epsilon,
H, images): its rerun gives the first report's values again.

## 2026-10-23 (45) -- localisation against behaviour; the nature of the errors (user's questions; implementer for the measures)

Every step now records the decoded positions (`decoded_template`: E4's nearest-template rule, the localisation of the
checks; `decoded_cell`: the cell nearest to the agent's colour). Reported (`analyse_e6.py`, `localisation_*.csv`): the
localisation in the zone (mean over the steps taken in it) of every (agent, image) next to b and the directed share; the
correlation of the directed share with it across the exposed models (valid pairs, all, and b < -1); within each agent,
the directed share of the steps well and badly localised (`measures.localisation_errors.steps_by_localisation`). The errors
(`error_nature`, former UD slot only, where the conveyor is defined): the share on the cell where the exposed TRAINING
dynamics would have taken the agent from its PREVIOUS state, conveyor[s_{t-1}] (the conveyor ignores the action, checked);
steps where the agent moved exactly as the conveyor would have are left out (no error can land on its cell); against
chance, the share on that cell among the errors that fall on one of the four neighbours of s_t when the conveyor's cell
is one of them (chance 1/4, Wilson interval). **Added after seeing the data (post hoc, flagged)**: the share of the errors
by region of the decoded cell and on the homologous cell of the other quadrants -- in the b-orientation configurations
most errors fall in the CD quadrant, not near the agent.

## 2026-10-23 (46) -- separating b's gap with E4's probe (user; implementer for the measure)

`frozen_setup.fresh_split`: b at the first step of an episode started from the initial state (no context), at the
centre (the 9 starts) and on the border (E4's 16 entry states), each model's I with the same function, seed and draws
(`fresh_start_controllability`). place_effect = b_fresh_border - b_fresh_center (both fresh); history_effect =
b_step1_probe - b_fresh_border (both on the border: E4's corridor walk against no context). Reported, nothing corrected.

## 2026-10-23 (47) -- the constrained correction (manipulation A) (user's loss; implementer for D_pos and the sweep)

L(d) = sum_a KL(target_a || prior_a(d)) + lam |d|^2 + mu_pos D_pos(d) + mu_goal (p_goal(h + d, z) - p_goal(h, z))^2
(`probing.belief_steering.constraint_terms`, optimise_delta's `extra`; mu_pos = mu_goal = 0 gives the first report's
correction step for step -- tested, and checked on the real runs: max |difference| 0.0 on the four agents).

**D_pos: the soft position read in the decoded image**, KL(pos(h) || pos(h + d)), pos = softmax over the cells of -(the
smallest squared distance of a cell's pixels to the agent's colour) / tau, tau = 0.02. Why: the localisation that must not
move is read from the DECODER (phase 0: the decoder reads h; the posterior's mode identifies the state but the decoded
position depends on h), so the constraint reads the same decoder output; it needs no training (a linear probe trained on
the references' (h, z) would read the references' code, not what the exposed model's decoder draws), and its argmax is
exactly E4's agent-cell rule (`decoded_agent_cell`, tested). Checked against the localisation of the checks (the
nearest-template rule) on E6's episodes: on the 14 valid pairs the two rules' accuracies differ by at most 0.05 (most by
< 0.01); they disagree on p2-a's two invalid images, where the template rule fails (E4's finding). p_goal: the goal head
through a forward pass that keeps the gradient (its own forward detaches the input).

Sweep (`run_e7_constrained.py`, e7.constrained): lam in {0.01, 0.1, 1, 10} x mu_pos in {0, 1, 10, 100}; mu_goal: the
smallest of {0, 10, 1e2, 1e3, 1e4, 1e5} keeping max |dp_goal| <= 0.05 at the most permissive setting (lam = 0.01,
mu_pos = 0), per agent, reported. Situations: 4 contexts per state of the training image's zone (the position term decodes
at every Adam step: 10 contexts would take about 4 h per agent on a CPU), the same walk generator, 1,000 Adam steps and
the same plateau rule. **Every context is corrected, localised or not** (unlike (42), which restricts the direction
estimation to localised situations): the localisation without intervention is then the exposed model's own (below 1),
the reference line the design asks for. Per setting: gain of b = the mean shift of I_s (the reference's I is unchanged),
the localisation under intervention, p_goal's shift, and the random control: each situation's own random direction at
the norm of its correction. Specific: the paired gain minus the control's has a 95% bootstrap interval above 0 and is at
least half the gain (`specific_share`). Kept: specific and localisation under intervention >= the exposed model's without
- 0.02 (`loc_tol`).

## 2026-10-23 (48) -- applying A (implementer)

For an agent's best kept setting (the largest specific gain; `apply_settings: all` for every kept one -- the constrained
stepwise mode decodes at every Adam step of every episode step), E6's episodes with, at every step in the zone: mode
constrained_direction (h + v_hat, v_hat = the setting's estimation-half mean correction at their mean norm) and mode
constrained_stepwise (the constrained correction optimised on the current h, at most 300 Adam steps), alpha in {0, 1}.

## 2026-10-23 (49) -- the induction in the references (manipulation B) (user; implementer for the grid)

The reference's reverse direction of `run_e7_directions.py` (target: the exposed training dynamics, the conveyor;
unconstrained, as in the first report) and its random control at the same norm, alpha in {0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.75, 1} (`e7.induction.alphas`), on the images whose zone is the former UD slot. Checks: b, the directed share, the
localisation and p_goal under intervention against the reference without (`reverse_none`). "b decreases monotonically
with alpha": no rise larger than 0.02 bit between consecutive alphas.

## 2026-10-23 (50) -- the agents file (user)

`configs/e6/agents_e5.yaml` follows the two-wave plan: baseline A (E4's pair, trained: active), baseline B and f = 1 seed
A (commented out with a TODO until trained), and the relaunch of the failed f = 1 run. (32) relaunched it as TWO runs,
seeds E = 402 and F = 502, both against seed B's f = 1 reference: both are listed (commented, TODO), the user keeps the one(s)
that learnt. `batches.agent_ids` ignores comments.

## 2026-10-26 (51) -- phase 0 of the diagnostics: how the critic is trained (implementer, read from the code)

Read before any diagnostic was computed (`agents/worldmodel/imagination.py`, `policy.py`, `heads.py`,
`pipeline/agent_run.py:565`, `configs/e4/e4.yaml`).
1. **The critic is trained only in imagination, through the prior.** At every gradient step after the estimator switch,
   `train_heads_step` starts from every posterior state (h, z) of the world model's replay batch (16 windows x 64 steps,
   detached) and `imagine` rolls out `horizon` = 15 steps with `rssm.imagination_step` (GRU, then one prior draw). No real
   return enters the critic's target: R_k = r_{k+1} + gamma c_{k+1} ((1 - lam) V_slow(s_{k+1}) + lam R_{k+1}),
   R_H = V_slow(s_H), lam = 0.95, gamma = 0.9; the loss is weighted by the cumulative continuation (w_k = prod c), the
   target is the slow critic (EMA, tau 0.02). H1 keeps its original form: a value learnt through the exposed prior.
2. **The imagined trajectories follow the agent's own policy**: epsilon-greedy (epsilon = 0.1, `epsilon_greedy`) on the
   one-step Q computed with K_imag = 4 prior draws per action (`q_values`, with the critic being trained), tie set of the
   training tolerance (0.0751). The starts are the replay's real posterior states (in the exposed model, the states of the
   UD slot with their real conveyor history); the rollouts are imagined.
3. **Reward and continuation from the goal head**: p = sigmoid(g(h, z)), r = 50 p - 1 (1 - p), c = 1 - p (`heads.py`);
   the goal head is trained by BCE on the replay's posterior states against 1[r_t > 0] (the arrival reward).
4. **The policy's Q** (`q_values`): for every action a, h'_a = GRU(h, z, a), K_act = 16 draws z'_k ~ p(. | h'_a),
   Q(a) = mean_k [r(h'_a, z'_k) + gamma c(h'_a, z'_k) V(h'_a, z'_k)]; epsilon-greedy with the random tie-break among the
   actions within 0.0751 of the max.

## 2026-10-26 (52) -- the diagnostics' definitions (implementer)

New modules: `measures/decision_rule.py` (numpy: directed actions, Q gaps, the greedy tie set's directed share, softmax,
bins) and `probing/lookahead.py` (torch: the cell values, the oracle one-step Q, the k-step Q in three variants). Runner
`experiments/E7_latent_manipulation/run_e7_diagnostics.py` (phases values, induction, softmax, lookahead), analysis
`analyse_e7_diagnostics.py`. Every weight is frozen; the runner hashes every model's state dict before and after and
fails if it moved.
- **Context of a cell** (diagnostics 1 and 4): E6's step 1, an episode started on the cell (h_1 = GRU(h0, 0, 0), z_1 a
  straight-through posterior draw of the cell's image, `fresh_start_contexts`), training image, test environment (all
  controllable). V of a cell: the critic's mean over 8 such draws (E6's `fresh_samples`). The map covers the 143
  non-goal cells; the zone AI is the former UD slot (25 cells), its ring the 24 corridor cells around it.
- **Directed action**: an action whose successor in the test environment is strictly closer to the goal (toric
  distance; E6's directed step). **Greedy directed share** of a Q row: the share of directed actions in the training's
  tie set (within 0.0751 of the max) -- the probability that the greedy branch of the training policy picks a directed
  action. The strict argmax is reported next to it.
- **Oracle one-step Q**: Q_or(a) = mean over K posterior draws of r + gamma c V(h'_a, z'), h'_a = GRU(h, z, a),
  z' ~ q(z | h'_a, e(x(s'_a))), s'_a the TRUE successor; r and c TRUE (50 and c = 0 on the goal, -1 and c = 1
  elsewhere): the transition belief AND the goal head are replaced by the truth, only V is the model's.
- **k-step Q**: Q_k(a) = sum_{i<k} gamma^i (prod_{j<=i} c_j) r_{i+1} + gamma^k (prod_{j<=k} c_j) V(s_k), the mean over
  K_act = 16 particles per (context, action) (the policy's 16 draws); after the first action each particle follows the
  GREEDY one-step action (epsilon = 0, random tie-break in the training's tie set) of its own variant, computed with
  K_imag = 4 draws (the number of the critic's training imagination). Variants: **oracle** (true successors, posterior of
  the true image, true reward; greedy on Q_or), **model** (the prior, r and c from the goal head; greedy on `q_values`),
  **model_corrected** (model from h_1 + d, d the constrained correction of (47) at the agent's setting of (48), optimised
  on h_1 only, at most 300 Adam steps as the stepwise mode; exposed models only). One rollout of k_max = 10 steps gives
  every k. Model, k = 1: `q_values` exactly (same draws; tested). b along the imagined steps (model and
  model_corrected): I_s at the particles' (h_i, z_i) minus the reference's fresh-start I at the particle's decoded cell
  (nearest template), on 2 particles per (context, action); the conveyor control: among the imagined transitions from a
  cell decoded in the zone, the share whose decoded successor is the conveyor's cell (exposed training dynamics) and the
  share that is the played action's cell (test environment).
- **Diagnostic 2's steps**: E6's training-image episodes (exposed and reference, rerun with the same seeds: identical to
  E6's) and plan B's induction (reverse and reverse_random, every alpha, rerun with every per-step array saved: plan B's
  runs kept no Q). A step counts if it is taken in the zone, measured, and well localised (nearest template = true state).
  b = b_local (I at the step minus the reference's fresh-start I at the state). Bins of 0.25 bit.
- **Diagnostic 3**: run_episodes' `tau` option: the action drawn from softmax(Q / tau) by the inverse CDF of the
  episode's explore uniform u (no epsilon branch; every step still draws E6's u, uniform action and scores, so the
  streams are consumed as before). tau = None (default) is E6's epsilon-greedy, unchanged (tested); tau -> 0 is the
  argmax (tested). The grid is fixed from diagnostic 2's gaps before diagnostic 3 runs ((53)).
- **Diagnostic 4's episodes**: run_episodes' `q_fn` option (default None: `q_values` with the act generator, unchanged;
  tested): Q_k oracle at every real step, k in {1, 5, 10}, epsilon-greedy as in training (epsilon 0.1, the same
  streams), exposed and reference, training image, E6's starts; the oracle's particles draw from their own generator.

## 2026-10-26 (53) -- the expected forms, written before any diagnostic was computed (implementer)

- **D1 (H1, first link)**. Reference: V rises towards the goal (Spearman of V with minus the distance >= 0.8 over the
  143 cells) and the oracle Q's greedy choice is directed on >= 0.9 of the zone's contexts. Exposed, if H1 holds: V is
  flat or does not rise towards the goal inside the zone AI, and the oracle greedy directed share is well below the
  reference's (<= 0.6; chance is 0.4, 2 directed actions of 5). If the exposed oracle share is close to the
  reference's (within 0.1), the cached value at one step is not the deficit's cause: H1's first link is refuted.
- **D2 (H2)**. If H2 holds: on well-localised steps, the greedy directed share stays near the references' level from
  b = 0 down to about -1.5 bit and drops in the last bins (b near -2 to -2.3), while the median best-minus-second Q gap
  shrinks gradually with b. A share that declines steadily with b from 0 refutes the step form.
- **D3 (H2)**. tau's grid: 5 values, log-spaced from 0.1 x to 10 x the median best-minus-second gap of the references'
  well-localised zone steps of D2 (quasi-greedy to nearly uniform at the scale of the gaps that drive the choice). If H2
  holds: at the smallest tau the curve is the epsilon-greedy one (flat on B, low at the exposed point); at an
  intermediate tau (around the median gap) the directed share decreases monotonically with b over B's alphas that keep
  the localisation; at the largest tau every point is near chance. If B's Q gaps do not shrink with b, the intermediate
  curve stays flat: the references' Q does not carry the partial belief, and H2 does not explain B's null effect.
- **D4 (H1)**. Oracle: the reference's directed share stays high (>= 0.9) for every k; if H1 holds, the exposed share
  rises with k towards the reference's (most of the gap closed at k = 10, where the path leaves the zone). Model: the
  exposed share stays low for every k and the imagined transitions follow the conveyor (the control). Model_corrected:
  b at the root is A's (about -0.3 to -0.7 bit) but falls back towards -2 from the first imagined step, and the share
  for k >= 2 returns to the model variant's. Episodes with Q_k oracle: if H1 holds, the exposed directed share rises
  with k towards the reference's 0.94.
- **Methodological status.** E6's phenotype stays the one measured with the training policy (k = 1, epsilon-greedy).
  Long lookahead and softmax are manipulations of the decision rule at test time, used to separate the transition
  belief from the cached value, not new phenotypes. The critic learnt under epsilon-greedy: the softmax is a reading
  rule, not the training policy.

## 2026-10-26 (54) -- what was settled while running the diagnostics (implementer)

- **tau's grid**, by (53)'s rule, before diagnostic 3 ran: the median best-minus-second gap of the references'
  well-localised zone steps of D2 (E6's reference and plan B, both modes, the 4 agents) is 0.4975, so tau in {0.0498,
  0.157, 0.498, 1.57, 4.98}.
- **n_trace = 1** particle per (context, action) for b along the rollout (not 2, as written in (52)): cost. b along the
  rollout is also reported on the particles decoded in the zone only (`b_mean_in_zone`): read at the decoded cell, b rises
  as soon as the imagined conveyor carries a particle out of the zone, which is location, not belief.
- D2 pools plan B's two modes (reverse and reverse_random) to cover b's range; D3 runs plan B's reverse mode only. D3's
  checks (localisation and p_goal) compare each point to the run without intervention at the same tau.
- D4's episodes ran on the 4 agents (under 1 h each), with the policy on Q_k oracle at every step of the episode.
- Results and verdicts: `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`, section 16.
