# Running the current E3 GPU batch (UCL ARC Kubeflow, A100)

**Current batch: gpu_e30b_measures (4 measurement jobs, no training). Follow this file from top to
bottom, skipping nothing.**

What it is: the four win64 runs of the finished batch gpu_e30b are **measured again** with the current
measurement code (`recompute_measures.py`), from their own final weights, on the A100, all four in the
same job. **Nothing is trained.** The measures gain a quantity that did not exist when gpu_e30b ran (I_s
projected on the support), from which the thresholds of the reading rule are computed. Its purpose is
written in `docs/e3/DECISIONS.md` (2026-10-01 (10)). Earlier batches are in `RUN_GPU_HISTORY.md`:
**do not use it for this batch.**

The four runs are read **in place, where gpu_e30b wrote them** -- the weights do not move:

```
/scratch/helplessness-worldmodels-gpu_e30b/results/E3_worldmodel_4zones/
    grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260929T183427Z
    grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed00__20260929T183428Z
    grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z
    grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z
```

Each job reads `checkpoints/weights_final.pt`, `replay.npz`, `world.json` and `config.yaml` of its run,
checks the weights' SHA-256 before measuring, and writes **only** into the job's own directory
(`logs/kubeflow/measure-<date>/runs/`). Nothing is written into the directories above.

## Rules for the person or the assistant running this file

1. Do not modify any file tracked by git (code, configs, docs). The only files you create are the
   ones this file tells you to create, in `handoff/gpu_e30b_measures/`.
2. Do not modify, move, copy or delete anything under `/scratch/helplessness-worldmodels-gpu_e30b/`:
   it is only read. Never relaunch a failed job and never change a run, a seed or an option. A job that
   fails is a result: it is reported, not repaired.
3. Never commit weights (`.pt` files) or complete result directories. The only commit is the one of
   step 5, of the directory `handoff/gpu_e30b_measures/` alone.
4. If anything happens that this file does not foresee (an error, a message not described here, a
   command that fails, a pod that does not start), stop: do not try to fix it. Go to step 5 with what
   you have, and report the complete output of the command that failed.

Every command below runs in a terminal of the Kubeflow notebook (`language-mind-v1-0`, namespace
`kubeflow-v-weilnhammer`), from the checkout `/scratch/helplessness-worldmodels` (not the gpu_e30b
one). Steps 2 to 5 use the shell variables `PRE` and `MEAS` set on the way. If the terminal was closed
in between, go back to the checkout and set them again from the newest directories:

```bash
cd /scratch/helplessness-worldmodels
ls -dt logs/kubeflow/preflight-*/ logs/kubeflow/measure-*/
```

## Step 1 -- update the checkout and check the code version

```bash
cd /scratch/helplessness-worldmodels
git status --short
```

This must print nothing. If it prints anything, stop (rule 4).

```bash
git checkout main
git pull origin main
python -c "import sys; sys.path.insert(0, 'src'); from helplessness.runs import source_tree_hash; print(source_tree_hash('.'))"
```

The last command must print exactly:

```
{'source_sha256': '99efffaf255bdf99944721415278ae53654ed4fc162b326af07dbe58c81c26e9', 'source_n_files': 261}
```

If `git pull` fails or the hash differs, stop: do not launch anything, go to step 5 (without measurement).
The preflight checks the same hash again on the GPU node and fails if it differs.

Use the commands of steps 2 and 3 exactly as written, `--cpu 5` included: the batch is prescribed **5
CPUs in total**. `submit_kubeflow.py` refuses another `--cpu`, and the preflight fails unless 5 CPUs are
both requested and usable in the container. If the submission is refused, or if the namespace cannot
grant 5 CPUs (pod `Pending`), stop (rule 4): do not lower the number yourself.

## Step 2 -- preflight (about 30 minutes)

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_e30b_measures --max-parallel 4 --cpu 5
```

It prints a JSON record and "Submitted, not yet confirmed running". Copy the value of `"output"` from
that record (a path ending in `logs/kubeflow/preflight-<date>`) into the variable `PRE`:

```bash
PRE=<the "output" path printed above>
```

The preflight checks: CUDA, TF32 off and deterministic algorithms on as in the measured runs; **5 CPUs
requested and 5 usable**; the code hash above; for each of the four runs, that its directory exists with
its files, that its name, `meta.yaml` and window are the job's, and that `weights_final.pt` has the
expected SHA-256; then it measures the first run once in full on the GPU (into `$PRE/runs/smoke/`) and
projects the batch's duration from it. Follow the pod until it has finished:

```bash
kubectl get pods
cat "$PRE/status.json"
```

The preflight has finished when `status.json` shows an `"exit_code"` that is a number (0 or 1), not
`null`. If the pod stays `Pending` for more than 15 minutes, stop (rule 4). When it has finished:

```bash
tail -n 30 "$PRE/preflight.log"
cat "$PRE/runs/preflight_verdict.json"
```

## Step 3 -- decision rule (apply it exactly)

Read `$PRE/runs/preflight_verdict.json`:

- **Launch** if `passed` is `true`, `problems` is empty, `projected_batch_hours` is a number <= 4, and
  `cpus.requested` and `cpus.usable` are both 5 (the last lines of `preflight.log` say `CPUs: 5
  requested, 5 usable` and `PREFLIGHT DECISION: launch`). Launch the measurement with exactly these
  options:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase measure --batch gpu_e30b_measures --max-parallel 4 --cpu 5
  ```

- **Stop** in every other case (any problem, a projection above 4 hours or missing, no verdict file, a
  number of CPUs other than 5). Do not launch; go to step 5 (without measurement) with the preflight's
  output. There is no retry branch for this batch.

The last line of `preflight.log` is `PREFLIGHT DECISION: launch` or `PREFLIGHT DECISION: stop`. If that
line and your reading of the rule disagree, stop (rule 4).

## Step 4 -- follow the measurement (under 1 hour)

After the submission, copy its `"output"` path (ending in `logs/kubeflow/measure-<date>`):

```bash
MEAS=<the "output" path printed by the measurement submission>
```

```bash
kubectl get pods
python -c "import json, sys; s = json.load(open(sys.argv[1])); print({k: v.get('exit_code') for k, v in s.items()})" "$MEAS/status.json"
tail -n 2 "$MEAS"/*.log
```

The four jobs run at once. Each log ends with `written to .../recomputed/final__<date>` when its run is
measured. `exit_code` is `null` while a job runs, `0` when it finished, `1` when it failed (a failure is
a result: rule 2). The batch is finished when every `exit_code` is a number and the pod is `Completed` or
`Error`. If the pod disappears while an `exit_code` is still `null`, stop (rule 4): go to step 5.

## Step 5 -- send back the results

With the measurement:

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --measure-dir "$MEAS" --preflight-dir "$PRE" --out handoff/gpu_e30b_measures
```

Without measurement (stop at step 1 or 3), with the preflight directory if it exists:

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --preflight-dir "$PRE" --out handoff/gpu_e30b_measures
```

The script writes `handoff/gpu_e30b_measures/handoff.md` (each job: exit code, output, code hash, weights
checksum, whether it reproduced the run's original arrays), one light archive per run
(`<run>.recomputed.tgz`: the measures, no weights) and `logs.tgz`. If it ends with `STOP: archive(s)
above 90 MB`, do not commit: stop (rule 4). After a stop under rule 4, add at the end of
`handoff/gpu_e30b_measures/handoff.md` a section `## Stop` with the command that failed and its complete
output. Then commit this directory alone and push:

```bash
git status --short
git add handoff/gpu_e30b_measures
git commit -m "E3 gpu_e30b_measures: results handoff" -- handoff/gpu_e30b_measures
git push origin main
```

`git status --short` must show only `handoff/` as new. If `git commit` or `git push` fails, stop and
report its output.
