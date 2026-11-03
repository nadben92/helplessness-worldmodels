# Robustness diagnostic: predictions (written before any mechanistic measurement)

Question: why does the exposed shuffle model F (seed 502) stay localised on the far arc_ctrl colours of the swap probes
while A (seed 2) and E (seed 402) lose the agent (E in the whole image)? Same data, same hyperparameters, frozen
weights at 650,000; the training performance does not separate them (min quadrant mass 0.917 F, 0.935 E, 0.948 A).

Committed after step 1 (inventory, consistency check, cluster scores: `tables/inventory.csv`, `tables/scores.csv`,
`tables/consistency_*.csv`) and **before** any measurement of steps 2-4. Every analysis added after this commit is
marked **exploratory** in REPORT.md.

## 0. Rules fixed after the inventory (before any mechanistic measurement)

- **Restricted score.** The inventory showed that the baselines (trained on the canonical image only) lose the agent in
  the whole image at t = 0 of the swap to se (T_inc and T_ctrl exchanged: localisation 0.00-0.15 in all four zones),
  so their score on the full set measures the arrangement, not the colour gradient. Rule, for **every** model: the
  robustness score is computed on the swaps whose localisation at t = 0 (mean over the four zones, cluster value) is
  >= 0.90. The baselines' se swap is reported apart (section 2, and test 0 at t = 0 below). This rule was fixed after
  the inventory and before any mechanistic measurement.
- **Consistency check.** It bears on curve points (mean over swaps and zones): it passes (max |local - cluster| 0.006
  for E, 0.013 for F). 8 of 744 single (image, zone) cells differ by > 0.05 (max 0.087); hence the band rule below.
- **Band rule.** No conclusion rests on a value within 0.05 of the 0.90 threshold: a localisation >= 0.95 is
  "localised", < 0.85 is "lost", [0.85, 0.95) is "indeterminate" and does not count for or against a prediction.
- **Scope (cost).** Steps 2 and 3 on arc_ctrl only (33 images: 3 swaps x 11 t), 9 models with local weights, 100 walks
  per (image, zone); step 4 complete. Every measurement on the local CPU. The four models at 100,000 have no local
  weights: cluster scores only, no mechanistic measurement.
- **Statistics.** Uncertainty = bootstrap over the walks (200 replicates, the same resampled walk indices for every
  image and model, so comparisons are paired). A difference below 2 SE is not interpreted. The **shape** of a result
  (sign, monotonicity in t, order of the models) is judged before its values.

## 1. Models (models.yaml)

| id | role | seed | checkpoint | mechanistic steps |
|---|---|---|---|---|
| baseA_exp_k300k / baseA_ref_k300k | baseline exp / ref | 2 | 300,000 | yes |
| baseB_exp_k300k / baseB_ref_k300k | baseline exp / ref | 102 | 300,000 | yes |
| A / ref_A | shuffle f = 1 exp / own ref | 2 | 650,000 | yes |
| E, F / ref_B | shuffle f = 1 exp, exp / ref | 402, 502 / 102 | 650,000 | yes |
| baseA_*_k100k, baseB_*_k100k | baseline | 2, 102 | 100,000 | cluster scores only |

## 2. Robustness score (step 1, cluster localisation)

Localisation of a (model, image, zone) = the swap probes' own measure: agent_cell_accuracy of part zone_step1 of
`probe_validity.reconstruction_diagnostic` (the agent's cell decoded from the posterior, h filtered on the probe image).
Curve: for each axis and t, the mean over the retained swaps (section 0) of the mean over the four zones. Scores:
**first t below 0.90** and the **area under the curve** over t in [0, 1] (trapezoids). P5 uses the area on arc_ctrl.
The baselines' se swap at t = 0 (mean over zones) is reported apart.

## 3. Measures of steps 2-4 (mechanism.py)

Same walks (`two_zones_setup.walks_for_slot`, first 100, n_in = n_in_grid), same probe worlds and image renderer as the
swap probes; the a1 posterior draws are seeded as the probes' diagnostic (`measurement_seed(..., "probe")`); every other
draw from its own SeedSequence slot.

- **(a1) job's localisation**: the model filters the walk on the probe image (h contaminated by the history), its
  posterior sample decoded, the agent's cell (`probe_validity.decoded_agent_cell`, nearest pixel to the agent's brown)
  against the true state.
- **(a2) encoder alone**: h_t taken from the filtering of the **origin** image along the same walk (same states and
  actions), z_t ~ posterior(h_t^origin, e(x_t^probe)), decoded with decode(h_t^origin, z_t). Origin = the swap's base
  image (t = 0, a training assignment) for the swap images; the canonical training image for step 4 and for the
  baselines' se swap at t = 0.
- **(b) prior where the posterior is correct**: at a frame j, from the a1 context (h_{j-1}, z_{j-1}) and the walk's
  action, one prior sample decoded; correct if the agent's cell is the true state at j; counted only where a1 is
  correct at j - 1. Primary: j = the first step in the zone (zone_step1, the job's part); also all zone steps.
- **Training images (step 3.1)**: the 12 assignments (`two_zones.training_colors` with every (UD, CD) pair), walks into
  the four zones, posterior along the model's own filtering. Per variable v of z (8 variables x 8 classes, rssm.py):
  I_pos(v) = MI between its most probable class and the agent's state within an assignment, averaged over the
  assignments; I_asg(v) = MI between its most probable class and the assignment (the definitions of `latent_ceiling.py`,
  `latent_ceiling.mutual_information`).
  - classes: **active** if max(I_pos, I_asg) >= 0.3 bit; among active, **position** if I_asg <= 0.25 I_pos, **background**
    if I_pos <= 0.25 I_asg, **mixed** otherwise; **inactive** if max < 0.3 bit. Position set Pos = the position
    variables; if none, position + mixed; if still none, the two variables with the largest I_pos.
  - **H2 separation** Sep = 1 - cos(I_pos, I_asg) over the 8 variables (1 = disjoint profiles).
  - **H3 sharpness** = mean largest posterior probability; **margin** = mean (top1 - top2); over all variables and over Pos.
- **Sensitivity (step 3.2)**: per frame and variable, the total-variation distance between the posterior on the probe
  frame and on the origin frame, **with the same h_t^origin** (the image's effect alone; the a1 version, own h, is
  reported beside), by where the agent is: **moved** (walks into m, zone steps), **other** (walks into the other three
  slots, zone steps), **corridor** (corridor steps of every walk); by t on arc_ctrl. "Moves little" = mean TV < 0.10.
- **Locality index (step 3.3)**: L = sum over v in Pos of mean_t S_v(other, t) / sum over v in Pos of mean_t S_v(moved, t),
  t = 0.1 ... 1.0 on arc_ctrl (also L_corridor with the corridor). Undefined if the denominator < 0.02.
- **Step 4** (stimuli.py; geometry, goal and agent states unchanged; colours in CIELAB, >= 30 deltaE from T_inc, T_ctrl,
  the corridor grey, the goal gold and the agent brown):
  - 4a: quadrants and corridor one colour: (0, 255, 150), (255, 35, 185), (20, 0, 85), farthest-point sampling (the first
    is the colour farthest from every training colour and the agent: min deltaE 87.4);
  - 4b: each quadrant and the corridor its own new colour, 5 draws (SeedSequence([4004, draw]));
  - 4c: canonical training image, agent drawn in the three 4a colours (localisation decoded at the brown the decoder
    renders; nearest to the new colour reported beside).
  - a family's localisation = mean over its images and the four zones at zone_step1; a1 and a2 both reported, **a2 is
    the primary measure** for P3 and P4.

## 4. Predictions

### Under H1 (local vs global shortcut)

- **P1 (step 2)**: E's loss is already in the encoder: at E's lost arc_ctrl points (a1 < 0.85), a2 < 0.85 too.
- **P2 (step 3)**: L_F < 0.5 and L_E >= 0.75, with L_E - L_F > 2 SE.
- **P3 (4a, 4b)**: F localised (a2 >= 0.95) and E lost (a2 < 0.85), on 4a and on 4b.
- **P4 (4c)**: F more affected than E by the recoloured agent: drop_F > drop_E + 2 SE, drop = a2 on the canonical
  training image - a2 on 4c. Minimal form: (a2_F - a2_E) on 4c smaller than on 4a + 4b by > 2 SE.
- **P5 (9 models, descriptive)**: Spearman rho(area arc_ctrl, -L) >= 0.6 and |rho(area, min training mass)| < 0.3.
  n = 9 with ties at the ceiling (F, ref_A, ref_B and the restricted baselines near 1): descriptive only.
  Min training mass = min over the four quadrants of `e4.real_context.mass_true_by_region` in the run's record at the
  probed checkpoint.

### Competing hypotheses

- **H2 (latent organisation)**: Sep_F > Sep_E by > 2 SE, and rho(area, Sep) >= 0.6 (descriptive).
- **H3 (margin)**: margin and sharpness (all variables and Pos) larger for F than for E by > 2 SE, and rho(area, margin)
  >= 0.6 (descriptive).
- **H4 (training history, logs only, step 5)**: descriptive: switch step, position decodability
  (`e5.decode_accuracy_canonical`, shuffle runs only), the minimal JS after the switch (LOCK-IN rule), the evolution of
  the min mass. Expected under H4: F differs from E and A on at least one. Already seen in the inventory: the switch
  step is 6301 for E, F and both references, 7151 for A.
- **H5 (exposure, descriptive)**: the exposed models are more fragile than their references (area: base A 300k, base B
  300k, A vs ref_A, E and F vs ref_B), F an exception.
- **H6 (contamination through h)**: the global loss goes through the recurrence, not the encoder: at E's lost arc_ctrl
  points, a2 >= 0.95 while a1 < 0.85. Under H1, a2 fails too (P1).

### Test 0 stop rule

If, at E's arc_ctrl points with t >= 0.5, a1 >= 0.95 while b < 0.85 (the loss in the prior, not in the posterior),
stop after test 0 and report. (H6 is not a stop: it is reported and the rest runs.)

## 5. Decision rule

- **H1 supported** if P1, P2 and P3 hold for the pair E/F and A goes in E's direction relative to F (a2 < 0.85 at A's
  lost points, L_A > L_F, A's 4a + 4b a2 below F's by > 2 SE).
- **H1 refuted** if P3 fails: F does not localise better than E on new backgrounds (a2_F - a2_E <= 2 SE on 4a + 4b).
- **H6 supported** if a2 >= 0.95 at the majority of E's lost points (then P1 fails).
- Otherwise not decided. P1-P5 are reported one by one, with H2-H6, whatever the verdict.
