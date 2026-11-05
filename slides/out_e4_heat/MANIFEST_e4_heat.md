# MANIFEST e4_heat

Quantity: I(a;s'|s) in bits (prior, 5 actions, max log2(5) = 2.322), per state of the TRAINING environment (training image), final checkpoint
(env step 300000), frozen weights, CPU. b = I_exposed - I_reference. Goal state masked (white). Contours: thin grey, the 4 quadrants.
Geometry per configuration (not realigned between configurations; array row = image row, top to bottom):
- p1-a-d1 (seed 01, diagonal 1): UD = slot nw (rows 1-5, cols 1-5); CD = slot se (rows 7-11, cols 7-11); neutral quadrants = ne (rows 1-5, cols 7-11) and sw (rows 7-11, cols 1-5); corridor = the cross, rows/cols rows 0-11, cols 0-11 (bounding box; goal excluded); goal = state 0 (row, col) = (0, 0) (masked)
- p1-b-d1 (seed 02, diagonal 1): UD = slot nw (rows 1-5, cols 1-5); CD = slot se (rows 7-11, cols 7-11); neutral quadrants = ne (rows 1-5, cols 7-11) and sw (rows 7-11, cols 1-5); corridor = the cross, rows/cols rows 0-11, cols 0-11 (bounding box; goal excluded); goal = state 0 (row, col) = (0, 0) (masked)
- p2-a-d1 (seed 05, diagonal 1): UD = slot nw (rows 1-5, cols 1-5); CD = slot se (rows 7-11, cols 7-11); neutral quadrants = ne (rows 1-5, cols 7-11) and sw (rows 7-11, cols 1-5); corridor = the cross, rows/cols rows 0-11, cols 0-11 (bounding box; goal excluded); goal = state 0 (row, col) = (0, 0) (masked)
- p2-b-d1 (seed 06, diagonal 1): UD = slot nw (rows 1-5, cols 1-5); CD = slot se (rows 7-11, cols 7-11); neutral quadrants = ne (rows 1-5, cols 7-11) and sw (rows 7-11, cols 1-5); corridor = the cross, rows/cols rows 0-11, cols 0-11 (bounding box; goal excluded); goal = state 0 (row, col) = (0, 0) (masked)

## Figures
- `E4_heat_p1b.png` (p1-b-d1, seed 02, n = 1 seed): panel 1 I exposed, panel 2 I reference (viridis, fixed [0, 2.322], shared colorbar), panel 3 b (coolwarm, fixed [-2.322, +2.322], own colorbar).
- `E4_heat_all.png`: 4 rows (p1-a-d1 seed 01, p1-b-d1 seed 02, p2-a-d1 seed 05, p2-b-d1 seed 06; n = 1 seed each) x the same 3 panels and scales.

## Source and estimator
- Weights: `agent_env300000.pt` of the E4 wave-1 runs `grid12x12__2zones-<config>__pair-{experimental,control}__worldmodel-agent-e4__seed<NN>__*` (extract at /c/nrw/e6all/E4_generalization_gradient).
- No per-state map existed: E4's records hold region means only, and the replay buffers (real-context probe) are not in the exports. Recomputed with `slides/out_e4_heat/measure_e4_heat.py`
  (data: `data_<config>.npz`), using the existing E4 walk estimator `probing.fixed_trajectory` (`fixed_trajectory_context` + `step_controllability`, I via `controllability_of_kernel`), with E4's settings
  (`configs/e4/e4.yaml`): fixed walks of `walks_for_slot` for each of the 4 slots (n_pre 32, n_in 25, seed 2026), probe environment = reference dynamics on the training image, measurement seed `measurement_seed(..., 'probe')`.
  Deviation from E4's reported quantity: instead of the zone steps only (entry / curve), I_s is taken at EVERY walk step (corridor, path, zone) after a burn-in of 8 steps, and averaged per state.
  Walks per slot: 100 (E4's full set is 300); draws per action: 16 (E4's probe setting); contexts per state: min 0 (exposed).
- Context = filtered walk history (not replay contexts); the exposed model's I is read at the reference's (all controllable) true successors for the mass only.

## b by region (mean ± sd over states; goal excluded)
| config | UD | CD | neutral | corridor | max abs b outside UD |
|---|---|---|---|---|---|
| p1-a-d1 | -1.92 ± 0.18 | +0.00 ± 0.01 | -0.01 ± 0.02 | -0.02 ± 0.04 | 0.16 |
| p1-b-d1 | -1.92 ± 0.29 | -0.00 ± 0.01 | +0.00 ± 0.02 | -0.00 ± 0.03 | 0.11 |
| p2-a-d1 | -1.95 ± 0.22 | +0.00 ± 0.01 | +0.00 ± 0.01 | -0.00 ± 0.02 | 0.07 |
| p2-b-d1 | -1.83 ± 0.28 | -0.00 ± 0.03 | -0.00 ± 0.02 | -0.01 ± 0.03 | 0.15 |

## Validity: reference mass on the true successor (mean over states, criterion >= 0.90)
| config | UD | CD | neutral | corridor |
|---|---|---|---|---|
| p1-a-d1 | 0.997 | 0.997 | 0.997 | 0.996 |
| p1-b-d1 | 0.995 | 0.996 | 0.992 | 0.991 |
| p2-a-d1 | 0.997 | 0.997 | 0.996 | 0.995 |
| p2-b-d1 | 0.995 | 0.989 | 0.993 | 0.991 |
