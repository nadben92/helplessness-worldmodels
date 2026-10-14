"""E3's world-model run pipeline, moved verbatim out of experiments/E3_worldmodel_4zones/run_mini.py on
2026-10-14 (run_mini.py re-exports every moved name, so every existing caller keeps working):

runtime.py      threads/TF32/determinism and the device, provenance and environment metadata, the console log,
                the training and measurement random streams, JSON helpers, the separation switch
world.py        the world of a (seed, condition): layout, themes, pixel environment, templates, episode
                parameters, zone bookkeeping, data collection with counts, the tabular twin's T_hat
measure.py      real-context and canonical measures, training diagnostics, D1/D2, the final measurement
checkpoints.py  the checkpoint grid, weight saves, generator states, failure diagnostics, run directories,
                building the RSSM, reloading a finished run's data

What stays in run_mini.py: reading its default config (DEFAULT_CONFIG), the training loop of one seed
(run_one_seed), the report and its figures, and the command line. E4-E7 extend this package with new modules or
options whose default reproduces E3's behaviour exactly.
"""
