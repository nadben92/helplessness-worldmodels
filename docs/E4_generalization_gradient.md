# E4 — Generalisation gradient

**Status: not started.** Depends on E3 (finished). Deterministic dynamics only.

## What it tests

Whether the world model's belief of controllability is carried by the position of a zone, by its appearance,
by both, or by neither.

## Training environments

- **Experimental:** two zones on one diagonal of the 12x12 torus -- one controllable-deterministic zone of
  theme T_ctrl, one uncontrollable-deterministic zone of theme T_inc; the other diagonal takes the corridor's
  theme.
- **Reference (paired):** the same layout with both zones controllable, derived by
  `envs.dynamics.pairing.derive_control` (never hand-written).

## Probes (zero data)

After training, both models are probed on fully controllable test environments, without any further
training:

- the theme of a single quadrant is changed, along three axes in RGB: the segment [T_inc, T_ctrl] and two
  circular arcs, 11 points per axis;
- at three positions: the former uncontrollable position, the former controllable position, a neutral
  position.

## Measure

b by zone (I_s of the experimental model minus that of the reference model), with a context h obtained by
letting both models filter one same fixed trajectory in the test environment.

## Config, code, results

- Configs: `configs/e4/`.
- Code: `experiments/E4_generalization_gradient/` (orchestration only); new library modules or options whose
  default reproduces the old behaviour exactly (`CLAUDE.md`, "Extending the library").
- Results: one orphan branch `results/<batch>` per cluster batch (`docs/RESULTS_CONVENTION.md`).
