# Results of the GPU cluster

## What is actually done (E4 on)

A GPU batch (UCL ARC Kubeflow) comes back as **one archive**, made on the cluster by the experiment's
`export_batch.py create` (`experiments/E4_generalization_gradient/`, `experiments/E5_belief_training/`): each run's
light files, the weights of the checkpoints the analyses need, the probe measurements, the supervisor's logs, and a
`MANIFEST.json` with every file's size and SHA-256. The archive is transferred to the local machine, checked by
`export_batch.py verify`, and extracted **outside the repository**. The analyses (and E6/E7, which run on the local
CPU with frozen weights, `--runs-root <root>`) read the extracted directory. Their **light outputs** (tables, JSON,
reports, small figures) are committed on the experiment's work branch, under `results/<experiment_id>/` or the
experiment's own analysis directory, then merged into `main` like the rest of the work. Archives, weights and raw
arrays are never committed (`.gitignore`).

**The orphan-branch convention described below was not used from E4 on**: no `results/<batch>` branch exists in the
repository. Its tooling (`helplessness.cluster.results_push`, `handoff.write_results_layout`, `submit_kubeflow.py
--push-secret`) is still there and off by default. The rest of this file documents it as it was designed.

## The orphan-branch convention (designed 2026-10-14, not used)

What a GPU batch returns to git would go to **its own orphan branch** `results/<batch>`, never to `main` or to a work
branch. Heavy artefacts (weights, buffers, bulk arrays) stay on the cluster's storage; the branch records where they
are and their SHA-256.

The batches run before this convention (E3's `handoff/<batch>/`, `logs/kubeflow/*.tgz`) remain in the history
of the work branch up to tag `e3-final`; they are no longer versioned and were not moved to branches.

## The branch

- Name: `results/<batch>` (the batch's key in the experiment's batch table, e.g. `results/gpu_e4_pilot`).
- Orphan: it shares no history with the code; it is **never merged** into `main` or into any work branch.
- Read it through a worktree, never by checking it out in the main working tree:

  ```
  git fetch origin results/<batch>
  git worktree add ../results-<batch> results/<batch>
  ```

- Its single tree:

  ```
  results/<experiment_id>/<batch>/
      handoff.json        every job's row (exit code, status, steps, switch step...), the archives, and
                          "heavy_files": [{"path": <absolute path on the cluster>, "sha256", "bytes"}]
      handoff.md          the same, readable
      status.json         the supervisor's own status.json (commands, exit codes, start/end times)
      logs.tgz            the supervisor's and preflight's logs, light (no .pt, no checkpoints/, no wandb/)
      <run name>/         one directory per run (helplessness.runs.make_run_name)
          config.yaml  meta.yaml  results.json  summary.json  metrics.csv  diagnostics.jsonl  REPORT.md
          console.log.gz
  ```

  Only the files of that list that the run wrote are copied; nothing else of the run directory is.

## Never on the branch

Weights (`*.pt`), replay buffers (`replay*.npz`), `arrays.npz`, `checkpoints/`, `measures/` bulk arrays,
`wandb/`. They stay under the cluster's results root (`/scratch/...`); `handoff.json` lists each with its
absolute path on the cluster and its SHA-256, so a later measurement can check it reads the same bytes.

## E4: an archive per batch; the push is optional

E4 brings a batch back as one archive, made in the notebook by `experiments/E4_generalization_gradient/export_batch.py
create` (each run's light files, the weights of its final checkpoint, the probe measurements, the supervisor's logs,
`MANIFEST.json` with every file's size and SHA-256) and checked locally by `export_batch.py verify`; the analyses read the
extracted directory. The push below is off by default (`submit_kubeflow.py --push-secret` turns it on).

## Producing it, automatically (optional)

Each run pushes its own light results when it ends, success or failure (`run_e4.py --push-batch <batch>`, through
`helplessness.cluster.results_push`): under `results/<experiment_id>/<batch>/<run name>/`, its light files
(`config.yaml`, `meta.yaml`, `results.json`, `metrics.csv`, `checkpoints.jsonl`, ...), `console.log.gz`,
`run_status.json` (status, error, traceback) and `heavy_files.json` (the run's weights, buffers and arrays left on the
volume, with absolute path, SHA-256 and size). At the end of the batch, the supervisor pushes `status.json` and a
`handoff.json` gathering every run's status and heavy files. Pushes retry on network errors, and when another run
of the batch pushed first they commit again on top of the branch's new head. The token comes from a Kubernetes secret mounted as a file (`RESULTS_PUSH_TOKEN`), handed to
git through GIT_ASKPASS; it never appears in the code, the job spec, a command line or a log.

## Producing it by hand (E3's handoffs)

`helplessness.cluster.handoff.write_results_layout` builds the tree above from a supervisor directory and the
run directories. E3's `make_handoff.py` exposes it as `--results-layout <dir>` (off by default: without it,
the handoff is exactly E3's). The person running the batch then commits that tree on the branch:

```
git worktree add --orphan -b results/<batch> ../results-<batch>     # git >= 2.42
cp -r <dir>/results ../results-<batch>/
git -C ../results-<batch> add results
git -C ../results-<batch> commit -m "<batch>: résultats"
```

Pushes are made by the user, never by an assistant.
