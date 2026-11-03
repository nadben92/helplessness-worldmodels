# Robustness diagnostic: why F stays localised and A, E do not

Frozen weights, local CPU, no training. Predictions, thresholds and decision rule: PREDICTIONS.md (commit 5648864,
before any mechanistic measurement). Anything not planned there is marked **EXPLORATORY**. Tables in `tables/`, figures
in `figures/`, raw per-walk outputs outside the repo (`mechanism.py --out-dir`, `C:\nrw\e5_robust\raw`).

## Summary

- **Test 0: the loss is in the posterior, and already in the encoder.** At E's lost arc_ctrl points (t >= 0.5),
  the encoder alone (a2, clean h) is lost at every point (0.67 -> 0.17-0.19), in the recoloured slot and in the slots
  whose colour did not change (0.69 -> 0.16). The recurrence amplifies the loss (a1 0.27 -> 0.01, a2 always above a1)
  but does not carry it alone: **H6 not supported**. A shows the same pattern from t = 0.7. The stop rule did not fire.
- **P1 holds, P2 holds, P3 partial, P4 holds (floor effect), P5 consistent (descriptive).** The decision rule gives
  **"not decided"**: H1 is neither supported (P3 not fully met) nor refuted (F localises far better than E on new
  backgrounds: +0.44 +/- 0.01 on 4a + 4b, carried by the five novel-background draws of 4b; on the three uniform
  colours of 4a the order depends on the colour).
- **Two results qualify H1 strongly (EXPLORATORY).** (i) Before the loss (t <= 0.3), the locality index of E (0.40) and
  A (0.30) is the same as F's (0.31): P2's contrast is carried by the collapse itself, not by a "global" code that E
  would use in distribution. (ii) The most local model, ref_B (L = 0.19), fails the novel backgrounds of 4b (0.48),
  while F, less local, is >= 0.89 on 4 of 5 draws (2 localised, 2 indeterminate).
- **H2 and H3 go the wrong way**: F has the least separated latent (Sep 0.56 vs E 0.82) and the least sharp posteriors
  of the shuffle models. **H4**: E and F have almost the same training history (same switch step, same min-mass
  trajectory); A differs. **H5**: every exposed model is more fragile than its reference on arc_ctrl; F nearly matches
  its reference (0.980 vs 0.995).

## 1. Models and robustness scores (step 1)

Localisation = the swap probes' own measure (agent_cell_accuracy at the first step in the zone, cluster values,
`tables/localisation.csv`). Restricted rule (PREDICTIONS.md section 0): only the swaps localised at t = 0 (mean over
zones >= 0.90). Score = first t below 0.90 / area under the curve. `tables/scores_restricted.csv`; figure R1.

| model | retained swaps | arc_ctrl | segment | arc_inc |
|---|---|---|---|---|
| baseA_exp_k100k | sw | 0.5 / 0.583 | never / 0.957 | 0.4 / 0.690 |
| baseA_ref_k100k | ne+sw | never / 0.981 | 0.7 / 0.833 | 0.0 / 0.748 |
| baseA_exp_k300k | ne+se+sw | 0.7 / 0.699 | never / 0.991 | 0.7 / 0.893 |
| baseA_ref_k300k | ne+sw | never / 0.997 | 0.7 / 0.882 | 0.0 / 0.867 |
| baseB_exp_k100k | ne+sw | 0.6 / 0.860 | 0.7 / 0.822 | 0.0 / 0.426 |
| baseB_ref_k100k | ne+sw | 1.0 / 0.989 | 0.5 / 0.731 | 0.0 / 0.765 |
| baseB_exp_k300k | ne+sw | 0.9 / 0.949 | 0.9 / 0.971 | 0.0 / 0.960 |
| baseB_ref_k300k | ne+sw | never / 1.000 | 0.6 / 0.813 | 0.0 / 0.842 |
| A | ne+se+sw | 0.7 / 0.658 | never / 0.986 | never / 0.981 |
| ref_A | ne+se+sw | never / 0.992 | never / 0.981 | 0.8 / 0.824 |
| E | ne+se+sw | **0.5 / 0.479** | never / 0.982 | never / 0.982 |
| F | ne+se+sw | **never / 0.980** | never / 0.982 | never / 0.972 |
| ref_B | ne+se+sw | never / 0.995 | never / 0.990 | never / 0.991 |

- The four 100,000 models have no local weights: cluster scores only, no mechanistic measurement.
- All local weights match the SHA-256 recorded by the cluster's probes (`tables/inventory.csv`).
- Consistency check (E, F, 150 walks): on the curve points, max |local - cluster| 0.006 (E), 0.013 (F). 8 of 744
  single cells differ by > 0.05 (max 0.087).
- Two of F's cells change side of 0.90 between cluster and CPU: image 2010 at ne (0.887 -> 0.940) and image 2072 at sw
  (0.913 -> 0.860). **Neither enters (38)'s counted points**: F's points are segment t = 0.0-0.7 and arc_inc t = 0.2,
  while both images are segment t = 1.0 = arc_inc t = 0 (and 2010 at ne is not valid). Nothing to add to
  analysis_swap/REPORT.md.

**The baselines' se swap at t = 0** (T_inc and T_ctrl exchanged; `tables/se_swap_t0_cluster.csv`,
`tables/se_swap_t0_test0.csv`): every baseline except baseA_exp_k300k loses the agent in all four zones (cluster mean
0.003-0.28). On that image, with the canonical training image as origin:

| model | a1 (nw/ne/se/sw) | a2, clean h (nw/ne/se/sw) | origin |
|---|---|---|---|
| baseA_exp_k300k | 0.94 / 1.00 / 1.00 / 0.99 | 0.97 / 1.00 / 1.00 / 1.00 | 1.00 |
| baseA_ref_k300k | 0.00 / 0.00 / 0.05 / 0.00 | 0.01 / 0.10 / 0.37 / 0.19 | 1.00 |
| baseB_exp_k300k | 0.00 / 0.02 / 0.14 / 0.10 | 0.13 / 0.48 / 0.31 / 0.46 | 1.00 |
| baseB_ref_k300k | 0.02 / 0.02 / 0.04 / 0.03 | 0.02 / 0.21 / 0.43 / 0.22 | 1.00 |

The baselines' global loss on the exchanged image is also in the encoder (a2 lost everywhere), amplified by h, as E's.

## 2. Test 0: encoder or dynamics? (arc_ctrl, 100 walks)

a1 = the job's localisation (own h); a2 = encoder alone (h of the swap's base image); b = the prior's prediction of
the first step in the zone, where a1 was correct one step before. Mean over the retained swaps and the four zones
(`tables/test0.csv`, also by moved / other slots).

| t | E a1 | E a2 | E b | A a1 | A a2 | A b | F a1 | F a2 | F b |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.94 |
| 0.3 | 1.00 | 1.00 | 0.97 | 0.98 | 0.99 | 0.96 | 0.98 | 0.99 | 0.93 |
| 0.4 | 0.97 | 0.99 | 0.86 | 0.99 | 0.98 | 0.96 | 0.98 | 0.98 | 0.90 |
| 0.5 | **0.27** | **0.67** | 0.57 | 0.99 | 0.99 | 0.95 | 0.98 | 0.99 | 0.91 |
| 0.6 | 0.04 | 0.39 | 0.36 | 0.98 | 0.98 | 0.95 | 0.98 | 0.98 | 0.90 |
| 0.7 | 0.02 | 0.28 | 0.00 | **0.08** | **0.53** | 0.23 | 0.97 | 0.98 | 0.91 |
| 0.8 | 0.01 | 0.19 | n/a | 0.02 | 0.39 | 0.20 | 0.98 | 0.98 | 0.89 |
| 1.0 | 0.01 | 0.18 | n/a | 0.01 | 0.31 | 0.62 | 0.94 | 0.94 | 0.84 |

SE of a2 <= 0.015 everywhere. b is n/a where a1 is never correct.

- **The shape is the same for E and A.** a1 drops abruptly, by 0.7-0.9 within one step of t. a2 drops at the same t but
  less far, and decreases monotonically after. The prior fails with the posterior: b falls where a1 falls.
- **The loss is not only in the prior**, so the stop rule does not fire. The encoder with a clean history already loses
  the agent, so H1's premise holds: the encoder's posterior is affected.
- **The encoder loss is global.** For E at t = 0.5, a2 is 0.62 when the agent is in the recoloured slot and 0.69 when
  it is in an unchanged slot. At t >= 0.6 it is 0.16-0.40 in both.
- **h amplifies the loss.** a1 - a2 is -0.41 at E's t = 0.5 and -0.45 at A's t = 0.7, far beyond 2 SE.
- **Note on b (added after the predictions, EXPLORATORY).** The prior of an exposed model does not predict the
  controllable successor in its UD slot: that is its learnt belief (F's b over the whole zone m is 0.28 at t = 0).
  b must therefore be read against its value at t = 0, not against 0.85. At step 1 this changes nothing here: b
  stays >= 0.84 wherever a1 is localised.

## 3. Predictions, one by one (`tables/reading.json`)

**P1 (step 2): holds.** At E's 6 lost points (t = 0.5-1.0, a1 < 0.85), a2 < 0.85 at all 6 (0.67, 0.39, 0.28, 0.19,
0.17, 0.18); none is in the indeterminate band. Same for A: 4 of 4 points (0.53, 0.39, 0.32, 0.31).

**P2 (step 3): holds.** Position variables are classified on the 12 training assignments (`tables/latent.csv`).
Locality index L (TV of the Pos variables, image effect only, t = 0.1-1.0; figure R2):

| model | L | SE |
|---|---|---|
| F | 0.46 | 0.008 |
| E | 1.33 | 0.012 |
| A | 1.10 | 0.013 |

- L_E - L_F = 0.87 +/- 0.016. A goes in E's direction.
- F's Pos variables move in the recoloured slot (TV 0.25-0.60) and much less elsewhere: 0.05-0.18, except one variable
  at 0.43.
- E's and A's move in every place once the loss starts.
- With the posterior of the model's own filtering (tv_a1), the ordering is the same: F 0.36, E 1.34, A 1.08.

**P3 (4a, 4b): partial.** a2, mean over images and zones (`tables/step4.csv`, figure R3):

| family | F | E | A |
|---|---|---|---|
| 4a (uniform) | 0.73 | 0.61 | 0.55 |
| 4b (novel) | 0.83 | 0.19 | 0.26 |

- F is not localised (< 0.85 on both families), so the strict form fails.
- But F - E = +0.44 +/- 0.01 on 4a + 4b, so H1 is not refuted. The same holds for a1: +0.40 +/- 0.01.
- Image by image on 4b, F is above E on all 5 draws (0.95, 0.92, 0.97, 0.43, 0.89 against 0.06-0.28). In the band
  rule, 2 of F's draws are localised and 2 indeterminate.
- Image by image on 4a, F is above E on 1 colour of 3 (green: 0.98 vs 0.26), level on magenta (0.945 vs 0.948) and
  **below** E on the dark colour (0.25 vs 0.61). The 4a half of P3 is therefore not a clear F > E (see section 5).
- The pooled difference is carried by 4b.

**P4 (4c): holds by the rule, but as a floor effect.**
- Drop from the canonical image to 4c: F 0.57, E 0.52; the difference is 0.054 +/- 0.019 (2.9 SE).
- The order reverses: F is below E on 4c (0.43 vs 0.48), above E on 4a + 4b. The gap shrinks by 0.50 +/- 0.02.
- **But every model collapses on 4c** (a2 0.43-0.57, all nine), on the same two agent colours (green and magenta:
  0.15-0.56 for all nine). The dark third colour (20, 0, 85) is localised by most (0.73-1.00, see section 5).
- Decoding at the new agent colour instead of the brown (a1x, a2x) does not rescue anyone (0.27-0.42).
- So every model relies on the agent's colour. F's dependence is barely specific: F is below E on 2 of 3 agent colours,
  by 0.09 and 0.12, and above on the third.

**P5 (9 models, descriptive): consistent.**
- Spearman rho(area arc_ctrl, -L) = 0.73; rho(area, min training mass) = 0.13.
- Five models are tied near the ceiling (area >= 0.97), so this is descriptive only.
- The scatter (figure R4) shows two groups rather than a gradient: the delocalised exposed models (E, A,
  baseA_exp_k300k) at L >= 1.0 and low area, and the others. One exception: ref_A, L = 1.17 with area 0.99 (see
  section 5).

**Decision rule: not decided.**

| condition | status |
|---|---|
| P1 | holds |
| P2 | holds |
| A's direction | holds: a2 lost at 4/4 points, L_A - L_F = 0.64 +/- 0.015, F - A on 4a + 4b = 0.42 +/- 0.013 |
| P3 | partial, so "H1 supported" is not reached |
| P3 not failed | so H1 is not refuted |
| H6 | not supported |

## 4. Competing hypotheses

**H2 (organisation of the latent): not consistent.**
- Separation Sep = 1 - cos(I_pos, I_asg): F 0.56, E 0.82, A 0.82, ref_B 0.68, ref_A 0.69.
- Sep_F - Sep_E = -0.26 +/- 0.003: F is less separated, the opposite of the prediction.
- rho(area, Sep) = -0.53.
- F has no "background" variable (5 position, 3 mixed); E and A have 3 each (`tables/latent_summary.csv`).
- The baselines' classification is read on 11 images they never saw: there, Sep is not a property of their training
  code.

**H3 (margin): not consistent.**
- Margin of Pos variables: F 0.91, E 0.94, A 0.97. Margin of all variables: F 0.90, E 0.95.
- F is the least sharp shuffle model (F - E = -0.034 +/- 0.001 and -0.050 +/- 0.001).
- rho(area, margin_pos) = -0.43.

**H4 (training history, records only): no difference between E and F** (`tables/history_summary.csv`,
`tables/history.csv`).

| | switch (grad step) | min switch statistic after the switch | position decoding >= 0.9 from | min mass at 650k |
|---|---|---|---|---|
| E | 6301 | 2.02 bits | 15,849 | 0.935 |
| F | 6301 | 1.38 bits | 25,119 | 0.917 |
| A | 7151 | 1.32 bits | 25,119 | 0.948 |

- No run approached the lock-in rule (0.01 bit).
- The min-mass trajectories of E and F are nearly identical (0.18-0.19 at 300-400k, then 0.87-0.89 at 500k). A
  differs (0.91 at 400k).
- E reaches 0.9 position decoding one record earlier (15,849 vs 25,119 steps). The records are too coarse for more.

**H5 (exposure, descriptive): consistent.** Area on arc_ctrl, exposed vs reference:

| pair | exposed | reference |
|---|---|---|
| base A 300k | 0.70 | 1.00 |
| base B 300k | 0.95 | 1.00 |
| A / ref_A | 0.66 | 0.99 |
| E / ref_B | 0.48 | 0.99 |
| F / ref_B | 0.98 | 0.99 |

F is the exception, as stated. The references are not robust everywhere: ref_A fails arc_inc from t = 0.8, and the
baseline references fail segment and arc_inc.

**H6 (contamination through h): not supported** (section 2). It is a real but secondary contribution: h deepens a loss
that the encoder already shows.

## 5. What remains ambiguous

1. **Locality before the collapse (EXPLORATORY, not planned).** L restricted to the points before E's loss:

   | | L, t <= 0.3 | L, t <= 0.4 |
   |---|---|---|
   | E | 0.40 | 0.79 |
   | A | 0.30 | 0.31 |
   | F | 0.31 | 0.40 |

   - While E and A are still localised, their position variables respond to the recolouring as locally as F's.
   - Before the loss, F's Pos variables move **more** than E's in the recoloured slot (0.36 vs 0.06, summed over
     t <= 0.3).
   - So the committed index separates the models mostly through the collapse, which is partly circular with the
     robustness score. The difference shows at t = 0.4 for E, the step before its loss.
   - Reading: E and A do not run a global code in distribution. They switch abruptly into a global failure beyond a
     colour distance; F does not.
2. **Locality is not sufficient for new backgrounds.**
   - ref_B is the most local model (L = 0.19) and robust on all swap axes, yet it fails 4b (0.48; 4 of 5 draws
     0.28-0.49).
   - F (L = 0.46) is >= 0.89 on 4 of 5 draws of 4b.
   - Whatever makes F robust to fully new backgrounds is not captured by L on arc_ctrl.
3. **ref_A: global index, robust curve.** L = 1.17, but its Pos variables barely move anywhere (summed TV 0.10 in the
   slot): L is a ratio of two small numbers there, above the 0.02 floor set in PREDICTIONS.md. ref_A localises on
   arc_ctrl (0.99) and on 4a (0.97), and fails arc_inc.
4. **4a and 4c depend on the colour more than on the model.** As a uniform background:

   | colour | a2 range over the 9 models | notable |
   |---|---|---|
   | magenta | 0.94-1.00 | localised by every model |
   | green | 0.18-0.99 | separates the models: F 0.98, ref_A 0.99, baseB_exp 0.97; E 0.26, A 0.23, baseA_exp 0.18 |
   | dark (20, 0, 85) | 0.16-0.97 | F 0.25, ref_B 0.24 and baseB_exp 0.16 among the worst; E 0.61; ref_A 0.97 |

   - As an agent colour, the dark one is the only one localised (0.73-1.00); green and magenta break every model.
   - The dark colour is dark like the brown (100, 45, 0) the decoder renders, which its deltaE to the brown (74.5)
     does not capture.
   - So 4a gives no single order: F leads on green, trails on the dark colour. Three colours are too few to tell a
     model property from a colour property.
   - The choice of the far colours (farthest-point in CIELAB) did not anticipate this.
5. **Bootstrap scope.** The SE covers the walks only, not the choice of images (3 colours in 4a and 4c, 5 draws in 4b).
   For P3 and P4 the image-by-image orders (section 3) are the robust part: F above E on 5/5 draws of 4b, F below E on
   2/3 agent colours of 4c.
6. **The baselines' latent statistics** (classification, Sep, margins) are measured on 11 assignments they never saw.
   They enter P5, H2 and H3 descriptively only.
7. **Models at 100,000** have no local weights: they appear in section 1 only.

## 6. Files

- Scripts: `inventory_scores.py`, `local_localisation.py` (step 1); `stimuli.py`, `mechanism.py`, `analyse.py` (steps
  2-4); `training_history.py` (step 5); `figures.py`.
- Raw outputs: `mechanism.py` wrote to `C:\nrw\e5_robust\raw\<model>__<set>__w100__<UTC>\`, with `walk_sums.npz`,
  `images.json` and `meta.json` (weights SHA-256, seeds, CPU time), plus `frames.npz` for the training set.
- CPU time:
  - arc_ctrl: about 1,215-1,420 s per model;
  - training set: about 300 s per model;
  - step 4: about 545 s per model;
  - total with 3 processes x 4 threads: about 2 h 05 of wall time.
- Figures:

  | figure | content |
  |---|---|
  | R1_localisation | localisation against t, the three axes |
  | R2_sensitivity | TV per variable against t, three panels, E / F / A |
  | R3_step4 | 4a / 4b / 4c by model |
  | R4_scatter | area against L |
