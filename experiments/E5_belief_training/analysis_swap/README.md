# E5 -- the theme-swap probes (docs/e5/DECISIONS.md (38))

Written before any probe of this set was launched. The data come from `e5_probes_swap` (RUN_GPU.md, step 7 ter).

- `check_point0.py`: the test before launch. At t = 0, b per quadrant on a swap image is compared with the training
  record of the same assignment (local weights of A and E at 650,000).
- `analyse_swap.py <probes-swap dirs> --out tables`: the tables and `reading.json`. This is (38)'s reading rule:
  - a point counts when Delta is computed in at least 2 of the 3 swaps;
  - an agent's plateau on an axis is the mean of its 3 furthest counted Delta points, all at t >= 0.5 (at least 3 such
    points; on the segment and arc_ctrl, it must also count at t = 0, with Delta(0) about -2); the plateau without the
    near-duplicate points is reported beside it;
  - the verdict is "colour" (plateau > -0.5), "arrangement" (plateau <= -1.5) or "both" (in between);
  - nw and the neutral slots are witnesses: "about 0" or "departs", with the t at which they depart;
  - the main conclusion is read on arc_ctrl and arc_inc; the segment is secondary, with and without t >= 0.8;
  - `--compare-probes <probes__ dirs>`: `shared_image.csv`, the reproducibility control on the one image shared with
    E4's set (T_ctrl at nw and se), against the E4-set measurement of the same model and checkpoint.
- `fig_swap_gradient.py --tables tables --out-dir <e6 worktree>/slides/out_e5/png_white`: `E5f_swap_gradient.png`, on a
  white background.

## Reading E5f_swap_gradient (the figure carries no text but axis names and ticks)

- **Columns**: the primary axes arc_ctrl and arc_inc, then the secondary segment (its t >= 0.8 make two CD-coloured
  slots: no assignment matches), with x = t = k / 10.
- **Rows**: the shuffle agents A, E and F at 650,000 on top. The baselines A and B at 300,000 below; they are
  qualitative only (`--baseline-k 100000` for the matched checkpoint).
- **Colours**:
  - the swapped slot m: violet (shuffle) or vermilion (baseline), solid line;
  - the neutral slots: grey;
  - the old UD slot nw: the UD zone colour, dash-dot line, over the three swaps (a witness).
- **Markers**: one per agent (A circle, E square, F triangle; baseline B square). A filled marker counts for the reading
  rule, a hollow one does not.
- **Bottom crosses**: no kept moved slot at that t (invalid or badly localised).
- **Shaded areas**: the near-duplicate points (segment t >= 0.8, arc_inc t <= 0.2).
- **Dashed lines**: 0 (grey) and -log2 5 (black).

Values are means over the swaps of the kept zones, where kept means valid (E4's criterion on the reference) and the
exposed model localised at step 1 (>= 0.90).
