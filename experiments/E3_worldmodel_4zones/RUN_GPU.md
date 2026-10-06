# Running the current E3 GPU batch (UCL ARC Kubeflow, A100)

**Current batch: gpu_pilot_noise (5 measurement jobs, no training). Follow this file from top to bottom, skipping
nothing.**

What it is: the measurement noise of the main measure. On the final weights of the four runs of the closed-loop pilot
(gpu_pilot_cl), I_s is measured 8 times with independent draws of real contexts, with the present number of contexts
per state and with 2 and 4 times more (`measure_noise.py --noise`); a fifth job measures, on REF seed 0 at its
checkpoint 15,849, the prior's mass on the goal with 10,000 draws and the agent's Q gaps next to the goal
(`measure_noise.py --goal-control`). **Nothing is trained.** Its purpose is written in `docs/e3/DECISIONS.md`
(2026-10-06 (26)). Earlier batches are in `RUN_GPU_HISTORY.md`: **do not use it for this batch.**

The four runs are read **in place, where the pilot wrote them** -- the weights do not move:

```
/scratch/helplessness-worldmodels/results/E3_worldmodel_4zones/
    grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z
    grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z
    grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z
    grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z
```

Each job reads its run's `config.yaml`, `meta.yaml`, `checkpoints.jsonl`, `checkpoints/resume.pt` (the replay) and
`checkpoints/agent_final.pt` (the goal control: `checkpoints/agent_env015849.pt` of REF seed 0), and writes **only** into
the job's own directory
(`logs/kubeflow/measure-<date>/runs/`). Nothing is written into the directories above.

## Rules for the person or the assistant running this file

1. Do not modify any file tracked by git (code, configs, docs). The only files you create are the ones this file
   tells you to create, in `handoff/gpu_pilot_noise/`.
2. Do not modify, move, copy or delete anything in the four run directories above: they are only read. Never
   relaunch a failed job and never change a run, a checkpoint or an option. A job that fails is a result: it is
   reported, not repaired.
3. Never commit weights (`.pt` files) or complete result directories. The only commit is the one of step 5, of the
   directory `handoff/gpu_pilot_noise/` alone.
4. If anything happens that this file does not foresee (an error, a message not described here, a command that
   fails, a pod that does not start), stop: do not try to fix it. Go to step 5 with what you have, and report the
   complete output of the command that failed.

Every command below runs in a terminal of the Kubeflow notebook (`language-mind-v1-0`, namespace
`kubeflow-v-weilnhammer`), from the checkout `/scratch/helplessness-worldmodels`. Steps 2 to 5 use the shell variables
`PRE` and `MEAS` set on the way. If the terminal was closed in between, go back to the checkout and set them again from
the newest directories:

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
{'source_sha256': '6b1eee46a9a01e35f631c61a080f1a27c9ad260b8552965f1b1a8875321413fa', 'source_n_files': 271}
```

If `git pull` fails or the hash differs, stop: do not launch anything, go to step 5 (without measurement). The
preflight checks the same hash again on the GPU node.

Use the commands of steps 2 and 3 exactly as written, `--cpu 3` included: the batch is prescribed **3 CPUs in total**.
`submit_kubeflow.py` refuses another `--cpu`, and the preflight fails unless 3 CPUs are both requested and usable. If
the submission is refused, or if the pod stays `Pending`, stop (rule 4).

## Step 2 -- preflight (about 30 minutes)

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_pilot_noise --max-parallel 4 --cpu 3
```

It prints a JSON record and "Submitted, not yet confirmed running". Copy the value of `"output"` from that record (a
path ending in `logs/kubeflow/preflight-<date>`) into the variable `PRE`:

```bash
PRE=<the "output" path printed above>
```

The preflight checks CUDA, TF32 off and determinism on, **3 CPUs requested and 3 usable**, the code hash above, and for
each of the four runs that its directory exists with the files listed above, that its name and `meta.yaml` are the
job's, and that the goal control's weights exist; then it makes one measurement of the first run with 4 times the
contexts (the most expensive one) and projects the batch's duration from it. Follow the pod until it has finished:

```bash
kubectl get pods
cat "$PRE/status.json"
```

The preflight has finished when `status.json` shows an `"exit_code"` that is a number (0 or 1), not `null`. If the pod
stays `Pending` for more than 15 minutes, stop (rule 4). When it has finished:

```bash
tail -n 30 "$PRE/preflight.log"
cat "$PRE/runs/preflight_verdict.json"
```

## Step 3 -- decision rule (apply it exactly)

- **Launch** if the last line of `preflight.log` is `PREFLIGHT DECISION: launch`, and
  `$PRE/runs/preflight_verdict.json` says `passed` `true`, `problems` empty, `projected_batch_hours` a number <= 8,
  `cpus.requested` and `cpus.usable` both 3:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase measure --batch gpu_pilot_noise --max-parallel 4 --cpu 3
  ```

- **Stop** in every other case (`PREFLIGHT DECISION: stop`, any problem, no verdict file, another number of CPUs): do
  not launch; go to step 5 (without measurement) with the preflight's output. There is no retry branch.

If the decision line and the verdict file disagree, stop (rule 4).

## Step 4 -- follow the measurement (a few hours)

After the submission, copy its `"output"` path (ending in `logs/kubeflow/measure-<date>`):

```bash
MEAS=<the "output" path printed by the measurement submission>
```

```bash
kubectl get pods
python -c "import json, sys; s = json.load(open(sys.argv[1])); print({k: v.get('exit_code') for k, v in s.items()})" "$MEAS/status.json"
tail -n 2 "$MEAS"/*.log
```

The four noise jobs run at once, the fifth (`goal_control`) starts when one of them finishes; each noise job prints a
line `[noise] ... x<m> repeat <r>: ...` after each of its 24 measurements. `exit_code` is `null` while a job runs, `0` when it finished, `1` when it failed (a failure is a
result: rule 2). The batch is finished when every `exit_code` is a number and the pod is `Completed` or `Error`. If the
pod disappears while an `exit_code` is still `null`, stop (rule 4): go to step 5.

## Step 5 -- send back the results

With the measurement:

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --measure-dir "$MEAS" --preflight-dir "$PRE" --out handoff/gpu_pilot_noise
```

Without measurement (stop at step 1 or 3), with the preflight directory if it exists:

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --preflight-dir "$PRE" --out handoff/gpu_pilot_noise
```

The script writes `handoff/gpu_pilot_noise/handoff.md` (each job: exit code, output), one
light archive per job (`<run>.recomputed.tgz` and `goal_control.recomputed.tgz`: JSON results, no weights) and
`logs.tgz`. If it ends
with `STOP: archive(s) above 90 MB`, do not commit: stop (rule 4). After a stop under rule 4, add at the end of
`handoff/gpu_pilot_noise/handoff.md` a section `## Stop` with the command that failed and its complete output.
Then commit this directory alone and push:

```bash
git status --short
git add handoff/gpu_pilot_noise
git commit -m "E3 gpu_pilot_noise: results handoff" -- handoff/gpu_pilot_noise
git push origin main
```

`git status --short` must show only `handoff/` as new. If `git commit` or `git push` fails, stop and report its output.
