# E5 -- probes of the shuffle pairs at 475,000 against the baselines (PRELIMINARY, docs/e5/DECISIONS.md (36))

Expected shapes committed before any b was computed: `EXPECTED_SHAPES.md` (45a0fd8). Tables: `tables/` (written by
`analyse_450k.py`); figures: `slides/out_e5/` (`slides/figures_e5/make_all_e5.py`, `slides/MANIFEST_e5.md`). b is
measured on the world model's prior; per E6/E7's lesson, b separates classes of belief (about -2.1 vs about 0) and
differences of a few tenths of a bit within a class are not read as what the agent uses.

## 1. Data inventory

Three archives in `C:\nrw\e5_probes_450k` (read-only, extracted elsewhere): `e5_extend_b` (training records, configs,
logs and the 475,000 weights of seed B's f = 1 reference, E and F), `e5_probes_b` (probes of E and F),
`e5_probes_baselines` (probes of baseline A and B). **The checkpoint is 475,000, not 450,000** (`e5_extend_b` stopped
there, (36)). Every measurement: configuration p1-b-d1, 93 images, 96 (image, zone) pairs, 100 probes (93 distinct
(image, slot) once the shared images are counted once), 150 walks, measure_probes.py status "done".

| Measurement | Exposed | Reference | k | Valid probes |
|---|---|---|---|---|
| baseline A | seed 2 (E4's run) | seed 2 (E4's) | 300,000 | 49/100 |
| baseline B | 102 | 102 | 300,000 | 44/100 |
| baseline B, matched level | 102 | 102 | 63,096 | 20/100 |
| shuffle f = 1, E | 402 | 102 (f = 1) | 475,000 | 55/100 |
| shuffle f = 1, F | 502 | 102 (f = 1) | 475,000 | 53/100 |

Missing: shuffle seed A (pair stopped at 350,000, never probed); shuffle seed B's own experimental run (locked in,
replaced by E and F, (32)). The shuffle condition is therefore **two exposed seeds sharing one reference model**.

## 2. Controls

**Validity, by position** (valid / probes; `validity_by_position.csv`):

| | old UD | old CD | neutral |
|---|---|---|---|
| baseline A 300k | 17/34 | 7/33 | 25/33 |
| baseline B 300k | 17/34 | 3/33 | 24/33 |
| baseline B 63k | 11/34 | 0/33 | 9/33 |
| shuffle E | 15/34 | 13/33 | 27/33 |
| shuffle F | 14/34 | 13/33 | 26/33 |

Every extreme probe and the four zones of the training image are valid in every measurement, except baseline B at
63,096 (theme only and the training image's CD zone invalid). The shuffle "theme only" probe passes the mass threshold
narrowly (reference mass 0.907 for E, 0.903 for F).

**Training image, b at step 1** (`extremes.csv`): UD -2.312 (A), -2.306 (B), -2.180 (B at 63,096), **-2.077 (E),
-1.941 (F)**; CD, neutral and other zones within 0.01 of 0 for the 300,000 baselines, -0.04 to -0.20 for E and F.
The pairs' level from the training records (canonical quadrant mass 0.868 E, 0.887 F; reference 0.939; E4's criterion
not met) is in (36); not recomputed here.

**Replication of E4** (`e4_replication.csv`): baseline A against E4's `probes_b_step1.csv` (p1-b-d1, 300 walks), on the
48 probes valid in both: mean |difference| 0.007 bit, max 0.048, none above 0.05; validity agrees on 99/100 probes.
**Replicated.**

## 3. Position vs theme (central question)

b at step 1 (`extremes.csv`; all valid):

| | position only | theme only |
|---|---|---|
| baseline A | -2.301 | -0.020 |
| baseline B | -2.274 | -0.004 |
| mean | -2.287 | -0.012 |
| shuffle E | -0.376 | -0.907 |
| shuffle F | -0.922 | -0.919 |
| mean | -0.649 | -0.913 |

**What the two extreme probes contain** (from `images.csv`): a probe changes ONE quadrant and leaves the training
image's other quadrants. "Theme only" (T_inc on the neutral slot ne) therefore shows **two** T_inc quadrants (nw and
ne); "position only" (the corridor's colour at nw) shows **none**. Under shuffle, every training image has exactly one
T_inc quadrant: both extreme probes are outside the shuffle training distribution in that respect, which they are not
for the baseline's reading. A b of about -0.9 on one of two T_inc quadrants is consistent with a belief split between
them; this design does not separate that from a partial reliance on the theme.

Matched comparison ((36), E and F against B at 63,096, probes valid in both; `matched_summary.csv`): only 17 of 100
probes per comparison are valid in both (none at the old CD position; theme only invalid in the baseline). Shuffle minus
matched baseline: old UD +0.38 (E, n = 11) and +0.61 (F, n = 11); neutral -0.43 (E, n = 6) and -0.18 (F, n = 6);
position only +1.31 (mean of 2). Same direction as the comparison at 300,000.

## 4. Theme gradient (`gradient_binned.csv`, valid probes, bins of 0.1 D centred on multiples of 0.1 D, seeds pooled)

Abscissa: the distance of the probed colour to T_inc, over D. Values: bin centre: mean b (n).

- Baseline, old UD: -2.31 from 0 to 0.4 D, -2.20 at 0.6, -1.80 to -2.05 from 0.7 to 0.9 (E4's result: the position
  carries the belief). Neutral: 0.00 up to 0.5 D, then -0.30 to -0.45 between 0.7 and 0.9 D, -0.21 at 1.0. Old CD: valid
  only from 0.7 D, about 0.
- Shuffle, old UD: -2.01 (0), -1.97 (0.1), -1.82 (0.2), -1.71 (0.3), -1.44 (0.4), -0.85 (0.5), -0.76 (0.6), -0.72 (0.7),
  -0.23 (1.0, n = 1). Neutral: -0.91 (0), -0.76 (0.1), -0.55 (0.2, 0.3), -0.49 (0.4), then -0.17 to -0.28 from 0.5 to
  1.0. Old CD: valid only from 0.4 D: -0.46 (0.4), then -0.13 to -0.20.
- Both shuffle seeds have the same profile (per-seed bins in the same table).

Under shuffle, b falls as the probed colour approaches T_inc at both positions measurable near T_inc, but the curves do
not overlap: at the same distance the old UD slot is about 1 to 1.3 bits below the neutral slot up to 0.4 D. The same
caveat as above applies: near T_inc, a UD-position probe shows one T_inc quadrant, a neutral-position probe two.

## 5. Distribution of beliefs (`distribution.csv`, valid probes, (image, slot) counted once)

Classes of b at step 1: helpless <= -1.5, intermediate in (-1.5, -0.5], in control > -0.5.

| | n | helpless | intermediate | in control |
|---|---|---|---|---|
| baseline A | 45 | 15 | 6 | 24 |
| baseline B | 40 | 15 | 0 | 25 |
| shuffle E | 50 | 8 | 11 | 31 |
| shuffle F | 48 | 6 | 11 | 31 |

The baselines are bimodal (near -2.3 and near 0); the shuffle pairs have about a fifth of their valid probes in the
intermediate class, their helpless probes stop near -2.1 rather than -2.3, and their "in control" probes sit at -0.1 to
-0.3 rather than 0.

## 6. Recovery in the episode (`recovery.csv`, training image, old UD, secondary)

Gain from step 1 to the mean of steps 6-10: baseline A +0.43, B +0.37 (E4: +0.35 to +0.42); shuffle E +0.50, F +0.36.
Plateau about -1.9 (baselines) and about -1.55 (shuffle).

## 7. Verdict per expected shape

1. **Baseline replicates E4: confirmed.** Position only -2.30 / -2.27, theme only -0.02 / -0.00; replication of E4
   within 0.05 bit.
2. **Shuffle, belief moved to the theme: not met as stated.** Theme only -0.91 / -0.92 (clearly negative, against
   about 0 for the baseline, but not <= -1); position only -0.38 (E, meets >= -0.5) and -0.92 (F, does not). And both
   extreme probes depart from the shuffle training images' single T_inc quadrant (above).
3. **Shuffle gradient, curves overlapping: partly.** b decreases toward T_inc at the old UD and neutral positions (the
   old CD position has no valid probe below 0.4 D); the curves do not overlap (old UD about 1 to 1.3 bits lower).
4. **Shuffle less extreme on the training image, intermediate b present: confirmed.** UD -2.08 / -1.94 against -2.31 /
   -2.31; 11 + 11 intermediate probes against 6 + 0.
5. **Fewer valid probes under shuffle: refuted.** 55 and 53 against 49 and 44 at 300,000 (20 for B at 63,096); the
   difference is at the old CD position (13 and 13 against 7 and 3). The shuffle reference was trained on the 12
   assignments, so it predicts the colours at more slots.

**Outcome.** Neither of the two pure outcomes: under shuffle the belief depends on BOTH the theme and the position. The
theme now matters at every position (at the old UD slot b rises from about -2.0 to about -0.7 as the colour leaves
T_inc, where the baseline stays below -1.8), but T_inc at the old UD slot still gives a stronger belief than T_inc on a
neutral slot, and position only stays negative for F.

## 8. Points of attention

- PRELIMINARY: E4's criterion is not met (every sub-criterion missed for E, all but |b_CD| for F, (36)); the final
  probes (to 650,000) are deferred.
- Two shuffle seeds sharing one reference; shuffle seed A not probed. No inference across seeds beyond "both seeds".
- About half of the probes are invalid in every measurement; the matched comparison keeps only 17 probes per pair and
  none at the old CD position.
- The extreme probes and the near-T_inc gradient probes have one or two T_inc quadrants depending on the slot probed;
  under shuffle this is a confound for "position vs theme".
- Differences of a few tenths of a bit within a class (e.g. -0.15 against 0 in the "in control" class) are not read.

## 9. Localisation filter for E5e (LOCALISATION_CRITERION.md, committed before computing it)

Kept = valid and the exposed model's localisation at step 1 (E4's direct rule, the probe's 150 walks) >= 0.90, both
conditions. Valid -> kept: baseline A 45 -> 33, B 40 -> 32, shuffle E 50 -> 43, F 48 -> 47 (`loc_removed.csv`). The
baseline's dip at the neutral slot beyond 0.5 D **disappears**: every baseline neutral probe between 0.7 and 0.9 D is
poorly localised (0.00-0.89) and removed; the kept ones are -0.04 to +0.02 from 0 to 0.6 D and -0.00 at 1.0 D (9 probes).
The shuffle's neutral gradient keeps its shape (-0.91 at 0, -0.35 at 0.4 D, -0.16 to -0.23 beyond 0.6 D). The four
extreme probes are all kept (0.96-1.00): section 3 is unchanged.
