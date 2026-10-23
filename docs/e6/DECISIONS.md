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
