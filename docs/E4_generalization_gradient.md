# E4 — Generalisation gradient

**Status (2026-10-16): code ready, wave 1 (the pilot) prepared, not launched.** Branch `e4`. Code and usage:
`experiments/E4_generalization_gradient/README.md`; launch: `experiments/E4_generalization_gradient/RUN_GPU.md`.
Deterministic dynamics only.

## What it tests

When a world model has learnt that a zone is uncontrollable, is that belief carried by the zone's position, by its
appearance (theme), by both, or by neither? A model trained in an environment with one uncontrollable zone and its
paired reference are probed with zero data on test environments where theme and position are dissociated.

## Training environments

E3's environment: 12x12 torus, four 5x5 zone slots, cross corridor, goal at the corridor intersection (0, 0), `stay`
action (|A| = 5), gamma 0.9, R = 50, step cost 1, horizon 36, conveyor drift `away_from_goal`, E3's data collection
and training procedure (`configs/e3/gpu_campaign.yaml`'s blocks, unchanged).

- **Experimental (exposed):** on one diagonal, a controllable-deterministic zone (CD) of colour T_ctrl and an
  uncontrollable-deterministic conveyor (UD) of colour T_inc; the two slots of the other diagonal have the corridor's
  dynamics and colour.
- **Reference:** the same, UD made CD by `derive_control`: same colours, geometry and goal.
- Pairing (tested): for every (s, a), H(s'|s,a) = 0 and a support of 1 in both; I(a;s'|s) differs only in the UD zone
  (0 vs log2 5).
- Conventions (`envs/layouts/two_zones.py`): diagonal 1 = (nw, se), diagonal 2 = (ne, sw), UD at the first slot, CD at
  the second (the 180-degree turn about the goal maps one placement onto the other exactly, E3's symmetry tools). The
  neutral slot (probed) is the slot of the other diagonal in UD's row band (nw <-> ne, se <-> sw). E3's class-A
  rotations put the conveyor on diagonal 1, class-B ones on diagonal 2; all four slots are 2 steps from the goal.
- Finding: with two zones only, the column reflection about the goal maps diagonal 1 onto diagonal 2 exactly -- the
  diagonal changes the image, not the dynamics. It was therefore removed as a factor of the plan (2026-10-16).

## Plan

A configuration = colour pair x orientation (which colour is T_inc), on diagonal 1: 4 configurations, each with its own
seed shared by its experimental run and its reference (8 runs), in two waves of four (`batches.py`), each one pair with
both orientations:

| Wave | Configurations | Budget | Checkpoint diagnostics |
|---|---|---|---|
| 1 (pilot) | `p1-a-d1`, `p1-b-d1` | 300,000 steps | every checkpoint (the choice of B) |
| 2 | `p2-a-d1`, `p2-b-d1` | B (after the pilot) | B and the control checkpoints 10,000 and 100,000 |

The diagonal was a factor of the first plan (8 configurations, 4 waves). It is no longer one: the column reflection
about the goal exchanges the diagonals exactly, so a diagonal-2 run is the same dynamics with a mirrored image. The
diagonal-2 configurations stay defined as optional replicas (`batches.REPLICAS`: same configuration, reflected image,
new seed), outside the default waves.

Checkpoints: E2's grid to 100,000 steps, then every 25,000 steps to 300,000 (and E2's 158,489, 251,189); weights saved at
every one. The only budget dependence is beta_rep's ramp (20% of the PLANNED gradient steps left after the switch):
later waves keep the planned 300,000 steps and stop at B, which reproduces checkpoint B exactly (tested).

## Colours

`configs/e4/themes.yaml`, by `choose_themes.py` (deterministic). Distance between themes and D: Euclidean in RGB.
Distinctness from the agent, the goal and the corridor: E3's minimum, 30 in CIELAB deltaE76 (E3's metric), for every
point of every axis, both orientations.

| Pair | a | b | D |
|---|---|---|---|
| 1 | (227, 17, 15) red | (22, 0, 244) blue | 307.8 |
| 2 | (17, 29, 14) near black | (208, 0, 252) magenta | 306.5 |

Agent marker (fixed): (100, 45, 0). Corridor (128, 128, 128) and goal (255, 215, 0): E3's. Smallest deltaE76 of any axis
point to agent / goal / corridor: pair 1 45.5 / 48.2 / 35.4; pair 2 45.3 / 33.6 / 36.9. Pair 2's colours are at least
40 deltaE76 from pair 1's and its direction 80 degrees from pair 1's. The best feasible pair among E3's campaign colours
had D = 255.5.

Axes, 11 points each: the segment [T_inc, T_ctrl]; the arc centred on T_ctrl from T_inc and the arc centred on T_inc from
T_ctrl (radius D, 0 to 60 degrees, equal steps of distance to the start). Plane of an arc: among the planes containing
the arc's radius (360 angles), those keeping the 11 rounded points in the RGB cube; the one maximising the smallest
deltaE76 to agent, goal and corridor; the arc centred on b is chosen first, the arc centred on a must stay D/10 away from
it (otherwise both end at the same colour). Compromise with the cube: D could reach 441 (the cube's diagonal) for a
segment alone; with the arcs of radius D kept in the cube and every point away from agent, goal and corridor, the
search (400,000 random pairs, then a hill climb) reached D = 307.8 and 306.5 -- the best found, not a proven optimum.

## Probes

A probe environment has the reference's dynamics and the training image, except one slot's colour: the 11 points of
each axis at three positions (former UD slot, former CD slot, neutral slot), plus "position only" (the corridor's colour
at the UD slot): 100 probes, 93 distinct images (tested), each keeping the list of its probes. Zone of interest: the
changed slot; the training image has the four slots in turn (96 image x zone pairs).

## Measure

Fixed walks (`probing/walks.py`, saved by seed, the same for every image and both models): a start drawn in the corridor,
n_pre = 32 steps of uniform actions within the corridor, a shortest corridor path to a border state of the zone of
interest drawn uniformly, then the zone: the entry state is the 1st step in the zone and the walk takes n_in - 1 = 24
more uniform actions within it (n_in = 25 states in the zone, steps since entry 1..25); never the goal nor another
slot; n_walks = 300 (wave 2's from the pilot's sufficiency analysis). The extreme probes (training image, theme only,
position only) use the 25 steps; the rest of the grid the same walks cut after 10 steps (n_in_grid), enough for the main
measure (m = 5). I_s is computed at the steps in the zone only; the corridor steps are filtered, never decoded. Each model filters the walk with frozen weights from its learned initial state; at each step in the
zone, I_s from the prior (16 draws per action, decoded and matched to the probe environment's state images, E3's rule;
identical draws decoded once). Main measure: b at entry = mean of I over steps 1..m (m = 5), by state then by zone,
experimental minus reference; secondary: b by step since entry; states covered reported.

## Pilot (wave 1)

`pilot_analysis.py`: convergence and choice of B (mass on the true successor >= 0.97 in every region, |b| <= 0.1 bit at
CD and in the corridor, |b_UD + log2 5| <= 0.15 bit, on two consecutive checkpoints, for every run of the wave; else
B = 300,000 and the run is flagged), protocol control (b from the walks vs E3's end-of-training b), forgetting of the
initial state, evolution in the zone, extreme probes, full grid.

## Config, code, results

`configs/e4/`; `experiments/E4_generalization_gradient/`; results: on the cluster's volume, brought back as one archive
per batch (`export_batch.py`: light files, final weights, probes, logs, manifest); the push to a branch `results/<batch>`
is optional and off by default.
