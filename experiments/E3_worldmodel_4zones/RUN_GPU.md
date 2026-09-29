# Running the current E3 GPU batch (UCL ARC Kubeflow, A100)

**Current batch: gpu_e30b (8 runs). Follow this file from top to bottom, skipping nothing.**

The 8 runs: configs `configs/e3/gpu_e30b_win64.yaml` (16 windows x 64 steps) and
`configs/e3/gpu_e30b_win16.yaml` (64 windows x 16 steps), seeds 0 and 5, experimental environment
and reference. Their purpose is written in `docs/e3/DECISIONS.md` (2026-09-30). Earlier batches and
other ways of launching are in `RUN_GPU_HISTORY.md`: **do not use it for this batch.**

## Rules for the person or the assistant running this file

1. Do not modify any file tracked by git (code, configs, docs). The only files you create are the
   ones this file tells you to create, in `handoff/gpu_e30b/`.
2. Never relaunch a run marked failed, and never change a seed or a config. A run that fails is a
   result: it is reported, not repaired.
3. Never commit weights (`.pt` files) or complete result directories. The only commit is the one of
   step 5, of the directory `handoff/gpu_e30b/` alone.
4. If anything happens that this file does not foresee (an error, a message not described here, a
   command that fails, a pod that does not start), stop: do not try to fix it. Go to step 5 with what
   you have, and report the complete output of the command that failed.

Every command below runs in a terminal of the Kubeflow notebook (`language-mind-v1-0`, namespace
`kubeflow-v-weilnhammer`), from the checkout on the persistent volume. Steps 2 to 5 use the shell
variables `PRE`, `PRE4` and `TRAIN` set on the way. If the terminal was closed in between, go back to
the checkout and set them again from the newest directories (`PRE` = the newest preflight directory
of the first preflight, `PRE4` of the second one if any):

```bash
cd /scratch/helplessness-worldmodels
ls -dt logs/kubeflow/preflight-*/ logs/kubeflow/train-*/
TRAIN=$(ls -dt logs/kubeflow/train-*/ | head -n 1)
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
{'source_sha256': '2c65bb74d1df6ea150cb186f61ff8977a92d4fa081c396641fa7b32d75d40728', 'source_n_files': 251}
```

If `git pull` fails or the hash differs, stop: do not launch anything, go to step 5 (without training).

## Step 2 -- preflight

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_e30b --max-parallel 8 --cpu 5
```

It prints a JSON record and "Submitted, not yet confirmed running". Copy the value of `"output"` from
that record (a path ending in `logs/kubeflow/preflight-<date>`) into the variable `PRE`:

```bash
PRE=<the "output" path printed above>
```

Follow the pod until it has finished (about 1 hour):

```bash
kubectl get pods
cat "$PRE/status.json"
```

The preflight has finished when `status.json` shows an `"exit_code"` that is a number (0 or 1), not
`null`. If the pod stays `Pending` for more than 15 minutes (for example, a quota message in
`kubectl describe pod <pod name>`), stop (rule 4). When it has finished:

```bash
tail -n 40 "$PRE/preflight.log"
cat "$PRE/runs/preflight_verdict.json"
```

## Step 3 -- decision rule (apply it exactly)

Read `$PRE/runs/preflight_verdict.json` (fields `passed`, `problems`, `memory_problems`, `warnings`,
`projected_batch_hours`, `parallel`):

- **Launch** if `passed` is `true`, `warnings` is empty and `projected_batch_hours` is a number
  <= 20. Launch the training with exactly the options of the preflight that passed:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_e30b --max-parallel 8 --cpu 5
  ```

- **Retry with 4 processes** if the preflight had `"parallel": 8` and its only problems are the
  memory test's (`memory_problems` is not empty and holds all of `problems`). Run the preflight again
  with `--max-parallel 4`, follow it as in step 2 with the variable `PRE4`, and apply this rule again
  to `$PRE4/runs/preflight_verdict.json`:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_e30b --max-parallel 4 --cpu 5
  PRE4=<the "output" path printed above>
  ```

  If this second preflight gives **Launch**, launch with `--max-parallel 4`:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_e30b --max-parallel 4 --cpu 5
  ```

- **Stop** in every other case: any other failure, any warning, `projected_batch_hours` above 20 or
  missing, a memory failure at 4 processes, no verdict file. Do not launch; go to step 5 (without
  training) with the preflight's output.

The preflight applies the same rule itself: the last line of `preflight.log` is
`PREFLIGHT DECISION: launch`, `PREFLIGHT DECISION: retry-max-parallel-4` or `PREFLIGHT DECISION: stop`.
If that line and your reading of the rule disagree, stop (rule 4).

## Step 4 -- follow the training

After the training submission, copy its `"output"` path (ending in `logs/kubeflow/train-<date>`):

```bash
TRAIN=<the "output" path printed by the training submission>
```

To see where the runs are (the batch takes about `projected_batch_hours` of the preflight):

```bash
kubectl get pods
python -c "import json, sys; s = json.load(open(sys.argv[1])); print({k: v.get('exit_code') for k, v in s.items()})" "$TRAIN/status.json"
tail -n 2 "$TRAIN"/*.log
grep -h "SWITCH\|STOPPED\|Error" "$TRAIN"/*.log
```

`exit_code` is `null` while a run is running or queued (at most `--max-parallel` run at once; the
others start as runs finish), `0` when it finished, `1` when it failed (a failure is a result: rule 2).
Each run's log prints a line every 1,000 gradient steps (`step  12000/60000 | ...`). The batch is
finished when every `exit_code` is a number and the pod is `Completed` or `Error`. If the pod
disappears or the job ends while some `exit_code` is still `null`, stop (rule 4): go to step 5.

## Step 5 -- send back the results

With the training (use every preflight directory you created: `"$PRE"`, and `"$PRE4"` if there was one):

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --train-dir "$TRAIN" --preflight-dir "$PRE" --out handoff/gpu_e30b
```

Without training (stop at step 1 or 3): the same without `--train-dir`, with the preflight
directories that exist (none after a stop at step 1):

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --preflight-dir "$PRE" --out handoff/gpu_e30b
```

The script writes `handoff/gpu_e30b/handoff.md` (status of every run: exit code, status, gradient
steps done, switch step, error), one light archive per run (no weights) and `logs.tgz`. If it ends
with `STOP: archive(s) above 90 MB`, do not commit: stop (rule 4). After a stop under rule 4, add at
the end of `handoff/gpu_e30b/handoff.md` a section `## Stop` with the command that failed and its
complete output. Then commit this directory alone and push:

```bash
git status --short
git add handoff/gpu_e30b
git commit -m "E3 gpu_e30b: results handoff" -- handoff/gpu_e30b
git push origin main
```

`git status --short` must show only `handoff/` as new. If `git commit` or `git push` fails, stop and
report its output.
