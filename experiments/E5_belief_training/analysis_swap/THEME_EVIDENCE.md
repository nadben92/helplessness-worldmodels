# E5 -- the evidence on the theme (shuffle f = 1, 650,000)

What supports, and what limits, the claim that under shuffle the belief b at step 1 is carried by the **colour** (the
theme) of the UD zone rather than by its **position** or by the **arrangement** of the other three quadrants. Each item
gives:
- its status: planned (written before its data) or exploratory;
- its numbers;
- what it separates and what it does not;
- its limit.

Nothing here relaxes a criterion: valid = reference mass >= 0.90 and |I_ref - log2 5| <= 0.15; kept = valid and
localisation at step 1 >= 0.90.

Agents:
- shuffle A (seed 2, own reference);
- E (402) and F (502), against reference B (seed 102);
- A against reference B ((39), pending).

Baselines A and B are qualitative only.

## 1. The 12 training assignments: the belief follows the T_inc quadrant wherever it is

- **Status**: planned measurement, descriptive reading.
  - The per-assignment records are part of every checkpoint (DECISIONS (12), (18)). The swap probes' point 0 checks them
    on three of the assignments ((38)).
  - Using them as evidence on the theme is a reading made here.
- **Numbers.** Records at 650,000 (`assignments_records.py`, `tables/assignments_records.csv`; b = I_curve[0] exposed
  minus reference, in each assignment's probe world, every action controllable):

  | agent | b at the T_inc (UD) slot, 12 assignments: mean [min, max] | CD slot: mean | neutral slots: mean | T_inc slot the lowest of the four | min gap to the next slot |
  |---|---|---|---|---|---|
  | A (own ref) | -2.07 [-2.20, -1.90] | +0.07 | +0.03 | 12/12 | 2.01 |
  | E | -2.03 [-2.14, -1.89] | -0.07 | -0.05 | 12/12 | 1.84 |
  | F | -2.10 [-2.24, -1.96] | -0.11 | -0.09 | 12/12 | 1.92 |
  | A against B's reference (records, preview of (39)) | -2.11 [-2.20, -1.95] | -0.02 | -0.02 | 12/12 | 1.95 |

  - Against its own reference, A's reference mass is >= 0.90 at 10/12 UD slots, 6/12 CD slots and 23/24 neutral slots.
    E, F and A against B: 48/48.
  - Point 0 of the swap probes (assignments 4, 11, 6; GPU, the probes' own walks and draws) agrees with these records:
    |difference| <= 0.063, mean 0.018 (`tables/point0_records.csv`).
- **What it separates**: colour against **fixed position**.
  - Each slot is UD in 3 of the 12 assignments, and b falls by about 2 bits at whichever slot holds T_inc.
  - A belief tied to one position would show at that position in all 12 assignments. Yet b at nw is about 0 in the 9
    assignments where nw is not UD: -0.19 to +0.10, with F the lowest at -0.19 to -0.06.
- **What it does not separate**: colour against **arrangement**. In every training assignment, the colour T_inc and the
  elimination by the other three quadrants (one T_ctrl, two neutral) designate the same slot.
- **Limit**:
  - The images are in the training distribution.
  - The records' walks and draws are not the probes'. The records of A against B pair two runs measured with their own
    seeds.

## 2. F along the gradient: (38)'s verdict "colour" on arc_ctrl, one agent only

- **Status**: planned ((38), reading rule written before the data).
- **Numbers** (REPORT.md sections 1-3):
  - F's Delta plateau on arc_ctrl is **+0.13** (t = 0.6, 0.7, 0.8; Delta(0) = -2.04): verdict "colour";
  - b(m) = -0.34 and b(neutral) = -0.48 over those points;
  - F's localisation stays at 0.93-0.98 along the axis;
  - A and E are not counted, so **(38)'s conclusion is "not readable" on both primary axes**.
- **What it separates**: colour against arrangement, at the far points of arc_ctrl. The colour of m is at D from T_inc
  and T_ctrl, and the arrangement still designates m.
- **What it does not separate**: an out-of-distribution effect of the new colour from a reading of it. The colours at
  t > 0 were never seen.
- **Limits**:
  - One agent, so not a conclusion under (38).
  - The witnesses depart. F's nw falls to -0.77 ... -0.95 from t = 0.3 in all three swaps (-1.06 / -1.04 where nw holds
    T_ctrl), and the neutral slots fall to about -0.5. So Delta = +0.13 reflects b(m) at -0.34 with the rest of the
    image lowered. It does not reflect a return of the image to 0. (38) itself says that a witness that departs
    qualifies the plateau.

## 3. A against reference B ((39)): pending

- **Status**: planned ((39), committed before launch). Batch `e5_probes_swap_A_refB`.
- **Numbers**: to be filled when the data arrive.
- **Expected count (projection in (39), not a result)**:
  - arc_ctrl: at most 2 counted points at t >= 0.5 (t = 0.5, 0.6), since A is delocalised from t = 0.7;
  - segment: 1-2 points, given B's validity;
  - arc_inc: 0 points.
  - So A is expected to stay not counted under (38). Its verdict is reported whatever it is.
- **What it can add**: A's start of the gradient against a reference valid at `ne` (item 4), on 2-3 swaps instead of 1.

## 4. The start of the gradient: b(m) rises with the colour alone, arrangement fixed

- **Status: exploratory** (REPORT.md section 10). Requested after (38)'s results. Its reading rule was written before
  its numbers, but the pooled curves had been seen.
- **Numbers.** Kept points only, paired 95 % bootstrap over the walks, from t = 0 to the last t where slot m is kept in
  the same swaps:

  | agent | axis | t_last | Delta b(m) | Delta (b(m) - neutral) | witnesses on the portion (Delta neutral / Delta nw) |
  |---|---|---|---|---|---|
  | A (own ref) | arc_ctrl | 0.6 | +1.35 [+1.29, +1.41] | +1.16 [+1.05, +1.28] (swap to se only) | -0.21 / -0.54 |
  | E | arc_ctrl | 0.4 | +0.49 [+0.43, +0.55] | +0.66 [+0.59, +0.73] | -0.17 / -0.16 |
  | E | segment | 0.6 | +1.55 [+1.49, +1.60] | +1.97 [+1.90, +2.03] | -0.42 / -- |
  | F | arc_ctrl | 0.8 | +1.89 [+1.84, +1.93] | +2.24 [+2.19, +2.31] | -0.36 / -0.67 |
  | F | segment | 0.7 | +1.24 [+1.19, +1.30] | +1.84 [+1.77, +1.91] | -0.60 / -- |
  | A (own ref) | segment | 0.4 | +0.06 [+0.01, +0.12] | +0.04 [-0.02, +0.11] | 0.00 / 0.00 (portion ends before the rise) |

  - The rise starts at t = 0.2 (A, F) to 0.4 (E) on arc_ctrl, and at 0.4-0.5 on the segment.
  - On the same portions the baselines' b(m) stays at about 0 (Delta -0.16 to +0.03) and their b(nw) at -2.0 to -2.3.
- **What it separates**: colour against arrangement.
  - With the three other quadrants fixed and slot m kept (valid and localised >= 0.90), changing the colour of m alone
    raises b(m) by 0.5 to 1.9 bits in all three agents (arc_ctrl), and in E and F on the segment.
  - b(m) - neutral rises as much or more, so the change is specific to slot m.
  - The arrangement alone does not hold the belief at m.
- **What it does not separate**:
  - which feature of the colour is read;
  - the width of the generalisation around T_inc (flat up to 0.1-0.3 D);
  - an out-of-distribution effect of the unseen colours that would be specific to slot m. The witnesses only filter
    image-wide effects.
- **Limits**:
  - Exploratory.
  - The interval covers the walks' sampling only: neither the measurement draws (about 0.05 bit) nor the seed are in
    it.
  - The portion is selected by the kept points.
  - A's Delta rests on one swap.

## 5. "Theme only" at 475,000, valid, and invalid at 650,000

- **Status**: planned (E4's probe set, (36), (37)).
- **Numbers** (`analysis_450k/tables/extremes.csv`, `analysis_650k/tables/extremes.csv`; T_inc painted on the neutral
  slot ne of the canonical image, b at ne):

  | | b | reference mass | valid |
  |---|---|---|---|
  | E, F at 475,000 | **-0.91 / -0.92** | 0.907 / 0.903 | yes (narrowly) |
  | baseline A / B at 300,000 | -0.02 / -0.00 | 0.995 / 0.995 | yes |
  | baseline A / B at 100,000 (matched) | -0.03 / +0.01 | 0.948 / 0.975 | yes |
  | A, E, F at 650,000 | -0.49 / -0.98 / -0.71 | 0.838 / 0.842 / 0.854 | **no** (not interpreted) |

- **What it separates**: colour against position, at 475,000. A position the agent does not associate with UD becomes
  low when it carries T_inc. The baselines do not show this.
- **What it does not separate**:
  - The image holds **two** T_inc quadrants (nw and ne), which never occurs in shuffle training. A b of -0.9 at one of
    them fits a belief split between two T_inc quadrants as well as a partial reliance on the theme.
  - The arrangement is ambiguous there (two candidate UD slots).
- **Limits**:
  - Valid by 0.003-0.007 above the mass threshold.
  - At 475,000, short of E4's criterion.
  - Not confirmed at the final checkpoint, where the reference predicts the two-T_inc image worse (0.84-0.85).

## Against a colour-only reading (reported in REPORT.md; not an item of the list above, but needed for a fair summary)

- **"Position only"** (nw painted neutral, no T_inc anywhere) gives -0.46 / -0.48 / -0.89 (A, E, F at 650,000), against
  -2.3 in the baselines. The arrangement designates nw there, so this residue cannot be attributed to position alone.
- **On the swap images whose slot m has left T_inc, nw falls to about that same value**: F -0.90 (arc_ctrl, t >= 0.5), A
  -0.36, E -0.30, although the arrangement designates m. This is post hoc and rests on few points for A and E.
  - It suggests a residual tie to nw, the canonical UD slot before the switch at 100,000. No test of that mechanism has
    been made.

## What the evidence supports, and what it does not

- **Supported:**
  - Under shuffle the belief is not tied to a fixed position (item 1, in distribution, 3 agents, 12/12).
  - With the arrangement fixed, it depends on the colour of the T_inc quadrant on the start of the gradient (item 4:
    exploratory, 3 agents, effects far above the walks' and draws' noise).
- **Not established:**
  - **Colour against arrangement under (38)'s planned rule** (not readable; one counted agent, F, reads "colour").
  - The far-gradient level of the belief, where the localisation, the validity and the witnesses all limit the reading.
  - That the colour is the ONLY cue: a residue at nw remains when T_inc is absent ("position only", and nw on the far
    swap images).
- **Pending:** A against reference B ((39)), expected not counted under (38) and usable for item 4.
