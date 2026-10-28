# E5, probes of the shuffle pairs at 475,000 -- expected shapes (written before any b is computed)

Committed before `analyse_450k.py` is run. The shapes are the user's (analysis request of 2026-10-28), transcribed
without change; each gets a verdict in `REPORT.md`.

**Seen before this commit** (inventory only): the batch manifests, the probe directories' `meta.yaml` (pairing,
checkpoints, number of walks, and the validity counts written there by `measure_probes.py`: E 55/100, F 53/100,
baseline A at 300,000 49/100, baseline B at 300,000 44/100, baseline B at 63,096 20/100), the column headers of
`probes.csv`, `images.csv`, `curves.npz` and `steps.npz`, and, while checking the array layout, the first row of E's
`curves.npz` (training image, UD slot: I_exp 0.204, I_ref 2.281 at step 1). No other b was read.

**The data are at 475,000, not 450,000.** `e5_extend_b` was stopped at 475,000 (docs/e5/DECISIONS.md (36)); the probe
batch `e5_probes_b` measured E and F at `k475000`. The "450k" in this directory's name follows the analysis request.

## Conditions and runs (from the probe directories' metadata)

| Condition | Exposed | Reference | Checkpoint | Batch |
|---|---|---|---|---|
| baseline, seed A | seed 2 (E4's p1-b-d1 run) | seed 2 (E4's) | 300,000 | e5_probes_baselines |
| baseline, seed B | seed 102 | seed 102 | 300,000 | e5_probes_baselines |
| baseline, seed B, matched level | seed 102 | seed 102 | 63,096 | e5_probes_baselines |
| shuffle f = 1, E | seed 402 | seed 102 (f = 1) | 475,000 | e5_probes_b |
| shuffle f = 1, F | seed 502 | seed 102 (f = 1) | 475,000 | e5_probes_b |

Shuffle seed A (pair stopped at 350,000) was not probed; shuffle seed B's experimental run (locked in) was replaced by
E and F (docs/e5/DECISIONS.md (32)). E and F share one reference model.

## Expected shapes

1. Baseline: replicates E4 -- "position only" about -2.2 to -2.3, "theme only" about 0.
2. Shuffle, if the belief moves to the theme: "theme only" clearly negative (<= -1), "position only" near 0 (>= -0.5).
3. Shuffle, gradient: b decreases as the probed theme approaches T_inc, whatever the position (the three position
   curves overlap).
4. Shuffle: b(UD) less extreme than the baseline on the training image (unfinished learning), hence intermediate b
   present in the distribution.
5. A smaller share of valid probes in the shuffle pairs than in the baselines.

If another outcome appears (the belief on neither the position nor the theme, or on both), it is described as it is,
with the numbers.

## Measures (fixed now)

- b at step 1 = `curve_exp[:, 0] - curve_ref[:, 0]` of `curves.npz` (as `analyse_probes.load_probe_dir`), valid probes
  only (`probes.csv` column `valid`: the reference's mass on the true successor >= 0.90 and |I_ref - log2 5| <= 0.15 at
  step 1, docs/e5/DECISIONS.md (22)); invalid probes are counted, never imputed.
- "position only": axis `position_only` (the corridor's colour at the canonical UD slot); "theme only": axis
  `segment`, k = 0, position `neutral` (T_inc on a neutral slot) -- `analyse_probes.EXTREMES`.
- Theme distance: the RGB distance of the probed colour to T_inc, over D = |T_inc - T_ctrl| (the training theme of the
  UD role in both conditions; under shuffle a slot has no single training colour). Probe images shared by several axes
  are counted once per (image, slot).
- Baseline condition in the figures: A and B at 300,000 (each condition at the end of its training, the secondary
  comparison of (36)); the matched comparison of (36) (E, F against B at 63,096, probes valid in both) is reported in
  the tables and the report.
- Replication of E4: baseline A against E4's `probes_b_step1.csv` (p1-b-d1, 300 walks) on the same probes, expected
  difference < 0.05 bit.
