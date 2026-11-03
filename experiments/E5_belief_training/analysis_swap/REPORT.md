# E5 -- theme-swap probes: results (docs/e5/DECISIONS.md (38))

Batch `e5_probes_swap` (archive `e5_probes_swap__20261103T085130Z`, 7 jobs, all `done`, manifest SHA-256 checked on
the 66 files). 93 images x 4 zones = 372 measurements per agent. `swap_probes.csv` lists 396 rows per agent: the
shared images appear on two axes (segment t = 0 = arc_ctrl t = 0 = the base image; segment t = 1 = arc_inc t = 0 = T_ctrl).
Counts below are per row (/396), as the tables compute them, unless "unique" is written.

Tables: `tables/` (`analyse_swap.py ... --compare-probes`, `point0_records.py`, `posthoc_swap.py`). Figure:
`E5f_swap_gradient.png` (branch e6, `slides/out_e5/png_white`).

## 1. Verdict by (38)'s rule: not readable on any axis

| axis | role | A | E | F | conclusion |
|---|---|---|---|---|---|
| arc_ctrl | primary | not counted (0 counted points) | not counted (0 counted points at t >= 0.5) | **+0.13** (t = 0.6, 0.7, 0.8; Delta(0) = -2.04) -> colour | **not readable** (1 agent) |
| arc_inc | primary | not counted (0 valid zones) | not counted (0 counted points) | not counted (1 counted point, t = 0.2) | **not readable** |
| segment, all t | secondary | not counted | not counted (2 far points: -0.45, +0.06) | **-0.73** (t = 0.5, 0.6, 0.7) -> both | not readable |
| segment, without t >= 0.8 | secondary | not counted | not counted | -0.73 (same points) -> both | not readable |

Plateau = mean of the 3 furthest counted Delta points at t >= 0.5, Delta = b(m) - b(neutral), a point counting with
Delta in >= 2 of the 3 swaps. No axis has 2 counted shuffle agents, so (38) gives **no conclusion** on colour vs
arrangement.

Why the agents drop out:
- **A**: the pair's reference is invalid at `ne` in the swaps to ne and to sw, on every image (t = 0 included). There
  `ne` is the moved slot (swap to ne) or the only neutral slot (swap to sw). Delta then exists only in the swap to se:
  never 2 of 3 swaps, not even at t = 0.
- **E**: on arc_ctrl the exposed model is delocalised at t >= 0.5 in the whole image (section 5).
- **arc_inc (all three)**: the reference is invalid at almost every point (A 0/132 valid, E 26/132, F 26/132), so Delta
  rarely exists in 2 swaps.
- **F's segment**: from t = 0.8 on, the points come from 1 swap only.

## 2. Shape check (before any reading)

**Point 0 agrees with the training records** (`point0_records.csv`, GPU probe against `e5.by_assignment` at 650,000, 36
(swap, slot)):
- |difference| has a mean of 0.018 and a maximum of 0.063 (F, swap to sw, slot sw).
- Slot m sits at -1.86 to -2.24. Every other slot sits at -0.16 to +0.10.
- All 36 are kept except A's `ne` in the swap to ne (moved slot) and in the swap to sw (neutral slot): invalid, as
  predicted. A's `ne` is kept in the swap to se.

**Baselines: the expected shape, exactly** (kept zones only):
- old UD slot nw: -2.13 to -2.31 at 300,000 (B's arc_inc t = 0.9 / 1.0: -2.07 / -2.00) and -1.96 to -2.19 at 100,000,
  whatever the colour (grey or T_ctrl) and the axis;
- slot m, the neutral slots and CD: within ±0.15 of 0 at 300,000, and within ±0.27 at 100,000 (single-swap points
  there).

**Shuffle: the witnesses depart, which contradicts the expected shape of (38).** (38) expected nw and the neutral slots
at about 0 along the whole gradient. They are not. Kept zones:

| | nw (old UD) | neutral slots | slot m |
|---|---|---|---|
| F, arc_ctrl t >= 0.3 (3 swaps) | **-0.77 to -0.95** (departs at t = 0.3 ... 0.9) | -0.28 to -0.52 | -0.71 -> -0.29 |
| F, arc_ctrl, nw in the swap to se (it holds T_ctrl there) | -1.06 / -1.04 (kept at t = 0.6 / 0.9) | | |
| A, arc_ctrl t >= 0.3 | -0.23 to -0.46 (2-3 swaps) | -0.22 to -0.26 | -0.82 to -0.62 (swap to se about -1.1, swap to sw about -0.4) |
| A, segment t = 0.6 / 0.7 | -0.31 / -0.66 (departs at 0.7) | | |
| E, segment t >= 0.5 | -0.19 / -0.52 (1-2 swaps) | -0.36 to -0.63 (departs at 1.0) | -0.80 -> -0.39 |
| F, segment t >= 0.7 | | -0.58 to -0.89 (departs at 0.7 ... 1.0) | |
| E / F, arc_inc t <= 0.2 (two T_ctrl quadrants) | | -0.60 to -0.64 / -0.88 to -0.90 | |

- When the colour of m leaves T_inc, b at m rises, but **the other quadrants of the same image fall**. The fall is
  strongest at nw for F, even where nw holds T_ctrl.
- Delta therefore reaches 0 partly because the neutral slots fall. F's arc_ctrl plateau of Delta = +0.13 corresponds to
  b(m) = -0.34 and b(neutral) = -0.48.
- (38) states that a witness that departs "qualifies the plateau there: it marks an effect of the image, not of the slot
  m". F's only counted primary verdict ("colour") is qualified in this way.

Other checks:
- Signs are consistent.
- Variance is non-zero.
- The seeds agree on the rise of b(m) when the colour of m leaves T_inc, but not on where it rises:
  - arc_ctrl: F at t = 0.2-0.3, A at t = 0.3, E at t = 0.4 then lost;
  - segment: E at t = 0.4-0.5, F at t = 0.5-0.6.
- The level reached at m differs between seeds: about -0.35 (F, arc_ctrl), -0.7 (A, arc_ctrl: -1.1 in the swap to se,
  -0.4 in the swap to sw) and -0.4 (E, segment).

## 3. Witnesses (reading.json)

| axis | A nw / neutral | E nw / neutral | F nw / neutral |
|---|---|---|---|
| arc_ctrl | about 0 / about 0 | about 0 / about 0 | **departs (0.3-0.9)** / departs (0.9) |
| arc_inc | no counted point / no counted point | no counted point / departs (0-0.2) | no counted point / departs (0-0.3) |
| segment | departs (0.7) / about 0 | about 0 / departs (1.0) | about 0 / departs (0.7-1.0) |

The threshold of -0.5 hides part of the departure:
- E's "about 0" on arc_ctrl is vacuous: its points at t >= 0.5 are not kept.
- A's arc_ctrl witnesses ("about 0") are at -0.22 to -0.46, so below 0 but above the threshold.
- F's neutral slots on arc_ctrl sit at -0.47 to -0.52, right at the threshold.

## 4. Comparison with "position only" (-0.46 / -0.48 / -0.89 for A / E / F)

**By the rule: nothing to compare.** No arc plateau exists except F's (Delta +0.13, i.e. b(m) = -0.34), which is far
above F's position-only value of -0.89.

**Post hoc observation** (`posthoc.json`, far_levels). On images whose slot m has left T_inc, nw takes about the
position-only value of each agent:

| agent | nw on these images | position only |
|---|---|---|
| F | -0.90 (arc_ctrl, t >= 0.5, 11 cells) | -0.89 |
| A | -0.36 (arc_ctrl, t >= 0.5, 5 cells) | -0.46 |
| E | -0.30 (segment, t >= 0.5, 3 cells) | -0.48 |

- On these images, the arrangement designates m, not nw. The low b at nw is therefore not arrangement.
- It looks like what (38) withdrew: a residual tie to nw. A possible mechanism, not tested: the canonical assignment
  before the switch at 100,000.
- This is post hoc and is not a conclusion.

## 5. Where the localisation removes probes

| | valid / 396 | kept | removed by localisation | where |
|---|---|---|---|---|
| A | 142 | 127 | 15 | all on arc_ctrl at t = 0.7-0.9, in every slot |
| **E** | 246 | 180 | **66** (65 unique) | **60 on arc_ctrl at t >= 0.4** (12 at each t from 0.5 to 0.7, 11 at 0.8, 10 at 0.9: the four slots of the three swaps); 3 on the segment (0.6, 0.7, 1.0); 3 on arc_inc (0-0.2) |
| F | 247 | 244 | 3 | arc_ctrl, swap to se, slot nw (t = 0.5, 0.7, 0.8; localisation 0.85-0.87) |

- **They are not the near-duplicate points.** Only 4 of E's 66 are flagged near-duplicates. arc_ctrl has no
  near-duplicates.
- E's mean localisation on arc_ctrl drops from 0.97 (t = 0.4) to 0.25 (t = 0.5), then to 0.00-0.03. The drop happens at
  every slot at once, so the exposed model loses the agent in the whole image once the colour of m is far along arc_ctrl.
- A shows the same drop at t >= 0.7 (0.99 -> 0.09), and so does baseline A at 300,000 (0.92 -> 0.43 -> 0.08 at
  t = 0.6-0.8).
- F never drops: 0.93-0.98 up to t = 1.0. Its 3 removals are borderline values at one slot.
- So the far arc_ctrl colours delocalise some models (E, A, baseline A) and not others (F), whatever their condition.
  F is the only shuffle agent whose arc_ctrl plateau survives.

## 6. Pair A, against (38)'s prediction

(38) predicted few valid zones at ne and se for pair A. The data are worse:
- **ne: 0 valid zones in the swaps to ne and to sw** (t = 0 included). In the swap to se it is valid at low t.
- se: valid up to about t = 0.4-0.6 on the segment and arc_ctrl.
- arc_inc: 0 valid zones anywhere.

A's Delta therefore exists only in the swap to se (moved se against neutral sw):
- segment, t = 0-0.4: -2.02 to -1.96;
- arc_ctrl, t = 0-0.6: -2.02, -1.98, -1.87, -0.89, -1.00, -0.91, -0.81.

These are single-swap points. Under (38) A counts at no point, t = 0 included, so it is out of every conclusion. Post hoc
with 1 swap, its arc_ctrl Delta far points are -0.91 / -0.81 (2 points, t = 0.5-0.6), still not 3.

## 7. Baselines (qualitative, kept zones only)

| | valid / 396 | kept |
|---|---|---|
| A 100,000 | 111 (28 %) | 60 |
| A 300,000 | 218 (55 %) | 165 |
| B 100,000 | 145 (37 %) | 91 |
| B 300,000 | 208 (53 %) | 181 |

- The swap to se is entirely invalid for all four: T_inc and T_ctrl exchange, two quadrants the baseline's reference
  never saw.
- On the kept zones the expected shape holds (section 2): nw at about -2.3 whatever its colour, every other slot at
  about 0. This replicates E4: the belief is tied to the position.

## 8. Reproducibility control: the image shared with E4 (`shared_image.csv`)

T_ctrl at nw and se; b at nw. Swap set (GPU) against E4 set (GPU), same model and checkpoint:

| | swap | E4 set | difference |
|---|---|---|---|
| baseline A 100,000 | -0.632 | -0.598 | -0.035 |
| baseline B 100,000 | -0.432 | -0.435 | +0.003 |
| baseline A 300,000 | -0.975 | -0.981 | +0.006 |
| baseline B 300,000 | -0.431 | -0.483 | +0.051 |
| shuffle A / E / F 650,000 | -0.152 / -0.403 / -0.976 | -0.171 / -0.441 / -0.961 | +0.019 / +0.038 / -0.015 |

- |difference| <= 0.051: the draws' noise, as expected.
- The probe is **invalid in every measurement** (both sets). The number reproduces but cannot be interpreted, as found
  in the CPU preview of (38).
- Baseline A at 300,000 has three E4-set directories (28 Oct, 1 Nov twice); their curves are identical, so the same
  difference appears three times in the CSV.

## 9. Where the rule fits the data poorly (flagged; the rule is applied as written above)

1. **The minimum of 2 of 3 swaps removes A entirely**, including its anchor at t = 0. Its reference loses `ne` in the
   swaps to ne and sw, where `ne` is the moved slot of one swap and the only neutral slot of the other. (38) predicted pair A's weakness, not that it would erase
   the pair.
2. **arc_inc was meant to give the plateau directly, and is almost entirely invalid** (reference criterion) for the three
   shuffle pairs. (38) did not foresee this. The primary reading therefore rests on arc_ctrl alone, and only F survives
   there.
3. **"The 3 furthest counted points" is not a plateau when the far points are lost.** On F's segment it takes
   t = 0.5-0.7 (-1.41, -0.52, -0.24), which is the rise itself.
4. **Delta assumes the neutral slots stay at about 0.** They fall to about -0.5 (F), so Delta = 0 does not mean b(m) = 0.

**Post hoc variants** (`posthoc_swap.py`, `posthoc.json`; NOT part of (38)):

| variant | segment | arc_ctrl | arc_inc |
|---|---|---|---|
| Delta, 1 swap enough | E +0.03, F +0.00 -> "colour" | F +0.13 only -> not readable | not readable |
| b(m), 2 swaps | F -1.10 ("both"; near-duplicates included) | F -0.34 ("colour") only | not readable |
| b(m), 1 swap | E -0.41 colour, F -0.82 both -> no majority | F -0.34 only | **E -0.45, F -0.32 -> "colour"** |

**In no variant does any agent keep b(m) or Delta at about -2 once the colour of m has left T_inc.** The hypothesis "the
arrangement alone" (plateau <= -1.5) is not supported anywhere the data exist. The single exception is F's segment
close to T_ctrl (b(m) about -1.0 at t >= 0.7, swap to se, two T_ctrl slots), which is a near-duplicate.

What the data show instead, post hoc and on few points:
- b(m) settles at about -0.3 to -0.7, between the two verdicts;
- the rest of the image falls by 0.2-0.9 at the same time;
- F's nw falls to its position-only level.

This is neither "colour only" nor "arrangement". The belief becomes diffuse over the image once T_inc is gone, with a
residue at nw for F. The reserve on colours never seen in training applies here in full: a diffuse image-wide effect is
what an out-of-distribution colour could produce.

## 10. Exploratory analysis, not planned by (38): the start of the gradient

**Status: exploratory** (user's request, 3 November, after (38)'s results). The reading below was written before the
bootstrap and the differences were computed. The pooled curves of sections 1-3 had already been seen; what is new here
is the test and its rule.

**Question.** On the part of the gradient where the agents stay well localised, does b(m) change when only the colour of m
changes, the arrangement staying fixed?

**Portion read (fixed before the numbers).** For each agent and axis, the kept points only (valid AND localised >= 0.90;
no criterion relaxed):
- t_last is the largest t such that, at every t' <= t, slot m is kept in the same set of swaps;
- that set holds at least 2 swaps where the agent has 2 at t = 0, else 1 (reported);
- the difference is taken between t = 0 and t_last on those same swaps (paired).
- For b(m) - neutral, a swap also needs at least one neutral slot kept at both ends.

**Interval.** Bootstrap over the walks (2,000 resamples). The walks of a slot are the same on every image, so the
resample is shared across t, which makes it paired. Each resample recomputes the measure's own estimator: the mean
over states of each state's mean at step 1 (`fixed_trajectory.summarize_zone`), exposed and reference separately.
Interval: 95 % percentile of the difference. Standard errors per point: the bootstrap SD.

**Reading (written before the numbers).**
- **b(m) rises significantly** (the 95 % interval of Delta b(m) above 0) on a portion where slot m stays localised
  >= 0.90 and the arrangement is fixed:
  - the agent reads the colour of m, at least in part;
  - this holds as a slot-specific reading only if b(m) - neutral also rises (interval above 0). If b(m) rises and
    b(m) - neutral does not, the change is an image-wide shift, not a reading of slot m.
- **b(m) stays at about -2** (interval containing 0, or a rise < 0.2 bit): **not conclusive**. The agent may generalise
  the colour T_inc to the nearby colours, or read the arrangement; the start of the gradient does not separate them.
- In both cases the witnesses (nw, neutral slots) and the localisation at each t are reported beside.

### Results (`gradient_start.py`, `tables/gradient_start.json`, `tables/gradient_points.csv`; figure `E5g_color_gradient.png`)

Interval: 95 % bootstrap over the walks (paired). For each agent and axis with a portion beyond t = 0:

| agent | axis | portion (swaps) | loc of m < 0.90 from t (portion's swaps / all three) | Delta b(m) | Delta (b(m) - neutral) | Delta neutral / Delta nw | reading (rule above) |
|---|---|---|---|---|---|---|---|
| A (own ref) | arc_ctrl | 0 -> 0.6 (sw, se; Delta: se only) | 0.7 / 0.7 | **+1.35** [+1.29, +1.41] (-1.97 -> -0.62) | **+1.16** [+1.05, +1.28] | -0.21 / -0.54 | rises, slot-specific |
| A (own ref) | segment | 0 -> 0.4 (sw, se; Delta: se) | 0.6 / 0.6 | +0.06 [+0.01, +0.12] | +0.04 [-0.02, +0.11] | +0.00 / -0.00 | stays: not conclusive |
| E | arc_ctrl | 0 -> 0.4 (ne, sw) | 0.5 / 0.4 | **+0.49** [+0.43, +0.55] (-2.04 -> -1.55) | **+0.66** [+0.59, +0.73] | -0.17 / -0.16 | rises, slot-specific |
| E | segment | 0 -> 0.6 (ne, se) | none / 0.6 | **+1.55** [+1.49, +1.60] (-1.97 -> -0.43) | **+1.97** [+1.90, +2.03] | -0.42 / -- | rises, slot-specific |
| F | arc_ctrl | 0 -> 0.8 (ne, se) | 1.0 / 1.0 | **+1.89** [+1.84, +1.93] (-2.18 -> -0.29) | **+2.24** [+2.19, +2.31] | -0.36 / -0.67 | rises, slot-specific |
| F | segment | 0 -> 0.7 (ne, se) | 0.8 / 0.7 | **+1.24** [+1.19, +1.30] (-2.18 -> -0.94) | **+1.84** [+1.77, +1.91] | -0.60 / -- | rises, slot-specific |
| F | arc_inc | 0 -> 0.7 (sw only) | none / 0.0 | +0.24 [+0.15, +0.33] (-0.50 -> -0.26) | -0.02 [-0.15, +0.11] | -- | rises, not slot-specific |

Other notes on the table:
- The baselines (qualitative, kept zones only, over their valid portions to t = 0.3-0.6):
  - Delta b(m) lies between -0.16 and +0.03, with b(m) at about 0 throughout;
  - b(nw) stays at -2.0 to -2.3 (Delta -0.03 to +0.17).
- **The start is flat**: b(m) stays within 0.1 of its t = 0 value up to t = 0.1 on arc_ctrl (A, F), 0.3 (E), and on the
  segment up to t = 0.2 (E), 0.3 (F) and 0.4 (A, the end of its portion). The rise starts at:
  - arc_ctrl: t = 0.2 (A -1.72, F -1.77), t = 0.4 (E -1.55);
  - segment: t = 0.4 (E -1.54), t = 0.5 (F -1.56).
- A's segment portion stops at t = 0.4, before that onset: its "not conclusive" is consistent with E and F at the same t.
- "--" in the witness column: nw is not kept at both ends of the portion (or the portion uses only the swap to se, where
  nw is the CD slot).
- arc_inc has a portion for F only (1 swap). No agent has one for A against its own reference or for E.

**Reading, by the rule written above:**
- On arc_ctrl (A, E, F) and on the segment (E, F), b(m) rises from about -2 by 0.5 to 1.9 bits. The rise happens where
  slot m is still kept, so it is valid and localised >= 0.90 at every t of the portion, while the three other quadrants,
  hence the arrangement, are fixed.
- b(m) - neutral rises too, by more: the neutral slots fall a little while b(m) rises. The change is slot-specific.
- Under the rule, **each of the three shuffle agents reads the colour of m, at least in part**, on the start of the
  gradient. That is: with the arrangement fixed, the belief at m depends on the colour of m.

**Limits (to state with the result):**
- **Exploratory.** (38)'s plateau rule, the planned reading, stays "not readable".
- **The interval is the walks' sampling only.** It excludes the measurement draws (about 0.05 bit, `shared_image.csv`),
  the choice of t_last, and the seed. The effects (0.5-1.9 bit) are far above the first two. Across seeds, the three
  agents agree in sign on arc_ctrl, E and F agree on the segment, and A's segment portion is too short.
- **Colours never seen in training** ((38)'s reserve). From t > 0 slot m carries a new colour.
  - The data show that the belief at m does not survive a change of the colour of m alone. They do not show which
    feature of the colour is read, nor how wide the generalisation around T_inc is (flat up to 0.1-0.3 D, depending on the axis and the agent).
  - An out-of-distribution effect specific to slot m is not excluded by the witnesses, which only filter image-wide
    effects.
- **The portion is selected by the data** (kept points). Where the localisation fails (E on arc_ctrl from 0.4-0.5, A from
  0.7), nothing is read, and the hollow points of E5g are not interpreted.
- **A against its own reference**: its Delta rests on the swap to se alone (`ne` invalid). A against B's reference ((39))
  will be added here when its data arrive.
