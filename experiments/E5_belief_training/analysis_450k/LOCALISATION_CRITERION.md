# E5e without the localisation artefact -- the criterion (written and committed before it is computed)

Request of 2026-10-28 (user): the baseline's dip in E5e beyond 0.5 D (neutral slot, colours far from every training
colour) is probably an artefact of localisation -- far from the training colours, the model may not locate the agent.
An objective criterion, the same for both conditions, decides which probes are kept; the axis is not cut by hand.

**Nothing of the localisation has been computed or read when this is committed.**

## The measure

For every probe (image, slot) of the four measurements shown in the figures (baseline A and B at 300,000, shuffle E and
F at 475,000), the EXPOSED model's localisation at step 1 in the zone of interest:

- the agent's decoded cell, E4's direct rule (`probe_validity.reconstruction_diagnostic`: the cell holding the pixel of
  the posterior's reconstruction nearest to the agent's colour) against the true cell; `agent_cell_accuracy` of the part
  `zone_step1` (the frames at the first step in the zone);
- on the probe's own walks: the same probe world, the first 150 of E4's fixed walks (the probe's n_walks), the same
  measurement generator as measure_probes.py's diagnostic (`two_zones_setup.measurement_seed(..., "probe")`);
- frozen weights, read-only; every weight file's SHA-256 checked against the probe's meta.yaml.

The kernel rule (nearest template) is computed alongside and reported, not used by the criterion.

## The criterion

A probe is **kept** if it is valid (E4's criterion on the reference, `probes.csv` column `valid`) **and** the exposed
model's localisation at step 1 (direct rule) is **>= 0.90**. Same threshold for baseline and shuffle.

## What is done with it

- E5e is redrawn with the kept probes only (`E5e_loc`); E5a_v2 is redrawn (`E5a_v3`) if a "position only" or "theme
  only" probe is removed. Existing figures are not overwritten.
- Reported: the number of probes removed per condition and per distance bin, and whether the baseline's dip beyond 0.5 D
  disappears. If it does not, it is said with the numbers and the figure stays as the criterion gives it.
