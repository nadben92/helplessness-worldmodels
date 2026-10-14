# Results of the GPU cluster: one orphan branch per batch

From E4 on, what a GPU batch (UCL ARC Kubeflow) returns to git goes to **its own orphan branch**
`results/<batch>`, never to `main` or to a work branch. Heavy artefacts (weights, buffers, bulk arrays) stay on
the cluster's storage; the branch records where they are and their SHA-256.

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

## Producing it

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
