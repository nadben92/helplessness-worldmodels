# E5 -- final probes of the shuffle pairs at 650,000 against the baselines (docs/e5/DECISIONS.md (37))

Tables: `tables/` (written by `analyse_650k.py`, which reuses `analysis_450k/analyse_450k.py`'s tables and
`analyse_probes.matched`; `tables/analyse_probes/` is `analyse_probes.py --pair-k ... --matched ...` itself). Figures:
`slides/out_e5/` of the `e6` worktree (`slides/figures_e5/make_e5_v2.py --set 650k | 650k_matched`). As before, b separates
classes of belief (about -2.1 vs about 0); differences of a few tenths of a bit within a class are not read as what the
agent uses.

**E4's criterion is not met by any pair at 650,000** (the ceiling); the level is declared per pair in (37). These are the
final probes of E5, not converged models.

## 1. Data

Archives in `C:\nrw\e5_extends` (read-only, extracted outside the repo in `extracted\`). Every measurement: p1-b-d1, 93
images, 100 probes (93 distinct (image, slot)), 150 walks, status "done".

| Condition | Exposed / reference | k | Valid probes | Probe directory |
|---|---|---|---|---|
| baseline A | 2 / 2 (E4's run) | 300,000 | 49/100 | `probes__p1-b-d1__base__seed02__k300000__20261028T125628Z` |
| baseline B | 102 / 102 | 300,000 | 44/100 | `probes__p1-b-d1__base__seed102__k300000__20261028T125628Z` |
| baseline A, matched | 2 / 2 | 100,000 | 40/100 | `probes__p1-b-d1__base__seed02__k100000__20261101T185641Z` |
| baseline B, matched | 102 / 102 | 100,000 | 39/100 | `probes__p1-b-d1__base__seed102__k100000__20261101T185013Z` |
| shuffle f = 1, A | 2 / 2 (f = 1) | 650,000 | **23/100** | `probes__p1-b-d1__sh100__seed02__k650000__20261101T183552Z` |
| shuffle f = 1, E | 402 / 102 (f = 1) | 650,000 | 54/100 | `probes__p1-b-d1__sh100__seed402__k650000__20261101T175710Z` |
| shuffle f = 1, F | 502 / 102 (f = 1) | 650,000 | 53/100 | `probes__p1-b-d1__sh100__seed502__k650000__20261101T175710Z` |
| (shuffle E, F at 475,000) | | 475,000 | 55, 53 | the preliminary probes, for the comparison |

- The baselines at 300,000 exist in three copies (28 October; 1 November 18:29 and 18:43): byte-identical `curves.npz`,
  b identical, same validity. The 28 October copies are used (those of the 475,000 analysis; every table row shared with
  `analysis_450k/tables/extremes.csv` is identical, 24/24).
- The matched checkpoint (`matched_checkpoint.py --at 650000`, recomputed locally from the training records): 100,000 for
  the three pairs; L = 0.9226 (A), 0.9350 (E), 0.9168 (F); baselines at 100,000: A 0.9630, B 0.9746 (63,096: 0.8890,
  0.9010). Baseline A's "before the switch" flag is a format artefact (E4's switch at environment steps 5,342 / 6,008).
- Replication of E4 (baseline A at 300,000 against E4's `probes_b_step1.csv`): 48 probes valid in both, mean |difference|
  0.007 bit, max 0.048, validity agreeing on 99/100 -- as at 475,000.

## 2. Validity

| | old UD | old CD | neutral | all |
|---|---|---|---|---|
| baseline A 300k | 17/34 | 7/33 | 25/33 | 49 |
| baseline B 300k | 17/34 | 3/33 | 24/33 | 44 |
| baseline A 100k | 14/34 | 6/33 | 20/33 | 40 |
| baseline B 100k | 15/34 | 3/33 | 21/33 | 39 |
| shuffle A 650k | 17/34 | 6/33 | **0/33** | **23** |
| shuffle E 650k | 15/34 | 15/33 | 24/33 | 54 |
| shuffle F 650k | 15/34 | 15/33 | 23/33 | 53 |
| shuffle E 475k | 15/34 | 13/33 | 27/33 | 55 |
| shuffle F 475k | 14/34 | 13/33 | 26/33 | 53 |

**Pair A's 23/100 is its reference's**, as expected, and more precisely its reference's weakest quadrant: 74 of its 77
invalid probes fail the mass threshold (60 both mass and I, 14 mass only), 3 fail I only. Every neutral-slot probe is
invalid: the reference's median mass on the true successor there is 0.838 (E and F's reference: 0.947, 0.936), and the
neutral slot `ne` is precisely the quadrant where the reference A's canonical mass is lowest (0.923, the pair's L).
30 probes of A have a reference mass between 0.85 and 0.90 (7 for E, 11 for F). Consequence: **pair A contributes nothing
at the neutral slot** (no theme-only, no neutral gradient, no UD-vs-neutral pair).

**Theme only is invalid in all three shuffle pairs at 650,000** (reference mass 0.838 A, 0.842 E, 0.854 F; it passed
narrowly at 475,000: 0.907 E, 0.903 F). More generally, for E and F every neutral-slot probe from 0 to 0.2 D (the colours
nearest T_inc, i.e. images with two T_inc quadrants) became invalid between 475,000 and 650,000, while the shuffle
reference's canonical mass rose (0.939 -> 0.962). The reference predicts those two-T_inc images worse after more
training on the one-T_inc images of the shuffle; no explanation is drawn here. The invalid values are reported below for
information only, never used.

## 3. Position vs theme

b at step 1 (`extremes.csv`):

| | position only | theme only |
|---|---|---|
| baseline A 300k / B 300k | -2.301 / -2.274 | -0.020 / -0.004 |
| baseline A 100k / B 100k (matched) | -2.123 / -2.241 | -0.033 / +0.015 |
| shuffle A 650k | -0.457 | invalid (-0.486) |
| shuffle E 650k | -0.484 | invalid (-0.984) |
| shuffle F 650k | -0.892 | invalid (-0.710) |
| mean of the shuffle pairs | -0.611 (SD 0.24, n = 3) | -- |
| shuffle E / F 475k | -0.376 / -0.922 | -0.907 / -0.919 |

- **Position only**: the shuffle stays far above the baselines. Shuffle minus baseline: +1.67 (A), +1.76 (E), +1.35 (F)
  against the matched baselines; +1.84, +1.79, +1.38 against 300,000. E moved by -0.11 from 475,000, F by +0.03. Under
  shuffle the position alone still lowers b by about half a bit (A, E) to 0.9 (F), never close to the baselines' -2.3.
- **Theme only**: no valid measurement under shuffle at 650,000. The 475,000 result (-0.91 / -0.92, valid) is not
  confirmed nor refuted by the final probes; the invalid values have the same sign and order (-0.5 to -1.0).

## 4. Old UD against neutral, same colour (`ud_vs_neutral_summary.csv`)

b(old UD) - b(neutral) for the same probed colour (same axis and k), probes valid at both slots:

| | pairs valid at both | mean | SD |
|---|---|---|---|
| baseline A / B 300k | 16 / 16 | -2.11 / -2.17 | 0.39 / 0.24 |
| baseline A / B 100k | 13 / 14 | -1.97 / -2.09 | 0.46 / 0.33 |
| shuffle A 650k | 0 | -- | -- |
| shuffle E / F 650k | 7 / 6 | **-0.64 / -1.11** | 0.82 / 0.36 |
| shuffle E / F 475k | 11 / 9 | -0.88 / -1.07 | 0.63 / 0.16 |

The old UD slot stays below the neutral slot at the same colour under shuffle (about 0.6 to 1.1 bit), far less than in
the baselines (about 2 bits). The valid pairs are at 0.3 D and beyond for E and F at 650,000 (the near-T_inc neutral probes
are invalid); the means at 475,000 included nearer colours.

Pooled bins (valid probes, `gradient_binned.csv`, unit = group; bin centre: mean b (n)):
- shuffle, old UD: -2.10 (0), -2.09 (0.1), -2.01 (0.2), -1.70 (0.3), -1.47 (0.4), -1.03 (0.5), -0.67 (0.6), -0.70 (0.7);
- shuffle, neutral: first valid bin 0.3 D: -0.56 (0.3), -0.33 (0.4), -0.28 to -0.32 (0.5-0.8), -0.12 (1.0, n = 22);
- baseline 300k, old UD -2.31 from 0 to 0.4 D; neutral -0.01 to 0.00 up to 0.5 D (then the arcs' dip, see section 6);
- baseline 100k, old UD -2.27 (0) to -2.15 (0.4); neutral -0.01 to 0.00 up to 0.3 D, -0.04 (0.4), -0.16 (0.5).

The shuffle's old-UD gradient (b rising from -2.1 to -0.7 as the colour leaves T_inc) is the same as at 475,000; the
baselines' old UD stays below -1.8 everywhere. The neutral side of the shuffle can no longer be read below 0.3 D.

## 5. Distribution (`distribution.csv`, valid probes, (image, slot) once)

| | n | helpless (<= -1.5) | intermediate (-1.5, -0.5] | in control (> -0.5) |
|---|---|---|---|---|
| baseline A / B 300k | 45 / 40 | 15 / 15 | **6 / 0** | 24 / 25 |
| baseline A / B 100k | 36 / 35 | 11 / 13 | **5 / 0** | 20 / 22 |
| shuffle A 650k | 20 | 8 | **6** | 6 |
| shuffle E 650k | 50 | 9 | **9** | 32 |
| shuffle F 650k | 49 | 9 | **8** | 32 |
| shuffle E / F 475k | 50 / 48 | 8 / 6 | 11 / 11 | 31 / 31 |

Intermediate beliefs remain under shuffle (6, 9, 8), slightly fewer for E and F than at 475,000 (9 and 8 against 11 and
11), against 0 to 6 in the baselines. Pair A's distribution
has no neutral probe, so its share of "in control" probes is not comparable.

## 6. Localisation filter (LOCALISATION_CRITERION.md, unchanged)

Kept = valid and the exposed model's localisation at step 1 (E4's direct rule, the probe's 150 walks) >= 0.90.
Localisation computed (`analysis_450k/localisation_probes.py`, frozen weights, SHA-256 checked before and after) for
shuffle A and E at 650,000 (`tables/localisation_shuffle_{A,E}_k650000.csv`) and for the baselines at 300,000 (recomputed:
byte-identical to `analysis_450k/tables/localisation_baseline_{A,B}_k300000.csv`, copied here). **Not computed: shuffle F
(weights missing locally) and the matched baselines at 100,000 (weights not local)**; E5e_loc_650k therefore shows the
shuffle side with E alone at the neutral slot (A has no valid neutral probe).

- Valid -> kept: baseline A 45 -> 33, B 40 -> 32 (as at 475,000), shuffle A 20 -> 18, shuffle E 50 -> 42.
- Extreme probes: position only kept everywhere (localisation 0.98-1.00); theme only is invalid under shuffle anyway.
  No E5a_v3.
- Shuffle E, neutral slot, removed: one probe each at 0.5, 0.6, 0.7, 0.8 D (b -0.64, -0.73, -0.83, -0.65; localisation
  0.00-0.45). **The shuffle's neutral plateau of about -0.3 between 0.5 and 0.8 D (section 4) came from poorly localised
  probes**: with the filter, E's neutral slot is -0.55 (0.3 D, n = 1), -0.26 (0.4 D, n = 2), then -0.08 to -0.10 from
  0.5 to 1.0 D (n = 1 per bin, 11 at 1.0 D). At 475,000 the filtered shuffle curve was -0.16 to -0.23 beyond 0.6 D.
- Baselines: unchanged from 475,000 (flat at -0.02 to 0.00 wherever a probe is kept).
- Reading: under shuffle the neutral slot reacts to the colour only near T_inc (0.3-0.4 D, the nearest valid colours),
  and is within 0.1 bit of 0 from 0.5 D on; the near-T_inc part (0-0.2 D) cannot be measured at 650,000.

## 7. Comparison with the preliminary probes (475,000), E and F

- Probe by probe (valid at both checkpoints, `matched_summary.csv`, kind `vs_k475000`): E +0.03 (n = 49, SD 0.11), F -0.04
  (n = 48, SD 0.18): the measured beliefs barely moved over 175,000 steps.
- Training image, old UD: -2.077 -> -2.132 (E), -1.941 -> -2.027 (F); the other zones moved towards 0 (E: CD -0.20 ->
  -0.09, neutral -0.16 -> -0.11).
- Position only: -0.376 -> -0.484 (E), -0.922 -> -0.892 (F).
- Theme only: valid -0.907 / -0.919 -> **invalid** (-0.984 / -0.710).
- Intermediate class: 11 / 11 -> 9 / 8.

## 8. What changes, and the two comparisons

- **What changes from the preliminary results**: (1) theme only, the central probe of "the belief moves to the theme", is
  invalid in every shuffle pair at 650,000 -- the final probes give no valid reading of it, and the neutral gradient
  under shuffle starts at 0.3 D; (2) pair A is added but is uninformative at the neutral slot (0/33 valid); its position
  only (-0.46) agrees with E (-0.48); (3) everything else is stable: position only far above the baselines, the old UD
  slot about 0.6-1.1 bit below the neutral slot at equal colour (about 2 bits in the baselines), intermediate beliefs
  present; (4) under the localisation filter, the shuffle's neutral slot is within 0.1 bit of 0 from 0.5 D on (E only;
  F not computed), where the unfiltered curve and the filtered one at 475,000 stayed at -0.16 to -0.3.
- **Matched (100,000) vs 300,000 baselines**: the same conclusions. The baselines at 100,000 are slightly less extreme
  (position only -2.12 / -2.24 against -2.30 / -2.27; old UD minus neutral -1.97 / -2.09 against -2.11 / -2.17), so every
  shuffle-minus-baseline gap is 0.03 to 0.18 bit smaller with the matched baseline (old UD probes: +0.59 to +0.67 against
  +0.73 to +0.78), well below the size of the effects (1.3 to 1.8 bits on position only). As noted in (37), a baseline
  above L enlarges the gap at the old UD position; at the observed sizes this bias cannot produce the effect.
- **Shape checks**: signs as expected (b <= 0 in the classes read, theme only about 0 in every baseline), values within
  [-log2 5, about +0.1], non-zero variance within each measurement (e.g. ud_vs_neutral SD 0.24-0.82), seeds of a condition
  in agreement in sign and order.

## 9. Points of attention

- E4's criterion not met by any pair (level per pair in (37)); three shuffle pairs, E and F sharing a reference.
- Theme only invalid in all shuffle pairs at 650,000; pair A invalid at every neutral probe.
- The weights of F at 650,000 are missing from the local copy of `e5_extend_b` (the download stopped): F's localisation is
  not computed. The baselines' weights at 100,000 are not local: the matched comparison is not localisation-filtered.
