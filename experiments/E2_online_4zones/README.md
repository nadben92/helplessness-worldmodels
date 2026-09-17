# E2 — online validation, 4 zones

See `docs/E2_online_4zones.md` for what this tests and the project's
working conventions. This file documents how to run the pilot/sweep and
how to regenerate data that isn't stored by default.

## Running

```powershell
cd experiments\E2_online_4zones
python pilot.py                 # Temps 2: 3 seeds, full trajectory logging, budget=1e6
python sweep.py                 # Temps 3: 4 conditions x 20 seeds, resumable, parallelised
python episode_analysis.py      # after the sweep: episode length/truncation by starting zone
```

## What the sweep stores, per checkpoint

Each run (`results/E2_online_4zones_sweep/<condition_key>_seed<NN>/arrays.npz`)
saves, at every checkpoint `t`:

- `N_checkpoint_<t>`: the full `(S, A, S')` counts table, `int32`.
- `N_start_checkpoint_<t>`: `(S, A)`, the subset of those counts coming
  only from a transition that was an episode's *departure* step (the
  first action taken right after a reset) -- lets you check whether a
  pair's visitation floor is explained by departures alone.
- `departure_counts`, `departure_pair_fraction`, `episode_lengths`,
  `episode_start_state`, `episode_truncated`: run-level (not
  per-checkpoint) episode bookkeeping -- see `episode_analysis.py`.

`per_state.csv`/`per_pair.csv` hold the derived measurement chain
(I_hat, b, Delta, L1, is_optimal_under_P, ...) for both the online and
matched conditions at every checkpoint -- see `measurement.py`/`artifact.py`.

## Regenerating a full trajectory

The sweep does **not** log the full step-level trajectory
`(episode_id, t, s, a, r, s', terminated, truncated)` for every one of
its 80 runs -- at budget=1e6 that would be ~20MB/run compressed, ~1.6GB
total, for data the sweep's own report doesn't need (episode length and
truncation rate only need the per-episode summary already in
`arrays.npz`, not the per-step record -- see `episode_analysis.py`'s
docstring). This is safe **because every run is reproducible
bit-for-bit**: every RNG stream `sweep.py` uses is a pure, deterministic
function of `(seed_index, alpha, epsilon[, checkpoint_step])`
(`rng.py::spawn_streams`/`spawn_checkpoint_streams`), so a run computed
again with the exact same `(condition, seed_index)` retraces the exact
same sequence of states, actions, and dynamics outcomes.

To regenerate the full trajectory of a specific run (say
`a1_e0.1_seed03`, i.e. `condition={"alpha": 1.0, "epsilon": 0.1, "key": "a1_e0.1"}`,
`seed_index=3`):

```python
# from experiments/E2_online_4zones/, with the repo's Python on sys.path
import sweep

condition = {"alpha": 1.0, "epsilon": 0.1, "key": "a1_e0.1"}
result = sweep.run_condition_seed(condition, seed_index=3, log_trajectory=True)

traj = result["trajectory"]  # dict of numpy arrays: t, episode_id, s, a, r, s_next, terminated, truncated
```

`log_trajectory=True` builds the exact same env/agent/RNG-stream setup
`run_condition_seed` always uses (never a second, separately-maintained
reconstruction script that could silently drift from the real sweep
code) and additionally records every transition into pre-allocated
compact arrays, exactly like `pilot.py`'s own trajectory logging. This
is verified directly, not just argued: `result["arrays"]["N_checkpoint_<t>"]`
from a plain call (`log_trajectory=False`, what the real sweep always
uses) and from a `log_trajectory=True` call of the *same*
`(condition, seed_index)` are bit-identical (see
`tests/agents/test_e2_sweep.py::test_log_trajectory_does_not_perturb_the_run_and_reconstructs_n_checkpoints`),
and `trajectory.py::reconstruct_N_from_trajectory` applied to the logged
trajectory reproduces that same `N_checkpoint_<t>` exactly.

Because `log_trajectory=True` pre-allocates one array per field sized to
`budget`, it costs some extra memory/time for a large budget -- fine for
regenerating one run on demand, not meant to be turned on for the whole
sweep.
