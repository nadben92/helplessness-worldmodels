# Running the current E3 GPU batch (UCL ARC Kubeflow, A100)

**Current batch: gpu_campaign -- the E3 campaign (8 closed-loop runs). Follow this file from top to bottom, skipping
nothing.**

What it is: the agent in closed loop (`run_agent.py`), configuration `configs/e3/gpu_campaign.yaml` -- the pilot's
configuration (gpu_pilot_cl) unchanged, with 4 times more real contexts per state in the measures and the measures
taken at every checkpoint after the switch --, seeds 1, 2, 6 and 7, each in the experimental environment and in its
reference, 300,000 environment steps each. **8 runs, queued, at most 4 at a time.** Its purpose and the analyses
decided before the launch are written in `docs/e3/DECISIONS.md` (2026-10-07 (28), (29)). The pilot's runs are not
used, except the final weights of one of them, read only by the preflight to time the measures. Earlier batches are
in `RUN_GPU_HISTORY.md`: **do not use it for this batch.**

The eight runs, in launch order (two pairs per round of four):

```
seed1_experimental   seed1_reference   seed6_experimental   seed6_reference
seed2_experimental   seed2_reference   seed7_experimental   seed7_reference
```

**Expected duration: about 3.5 days in total** -- the preflight about 1 hour, then two rounds of about 42 hours (each
run: about 37 hours of training at ~2.2 environment steps per second with 4 runs on the GPU, plus about 5 hours of
measures). The preflight measures both and prints its own projection.

## Rules for the person or the assistant running this file

1. Do not modify any file tracked by git (code, configs, docs). The only files you create are the ones this file
   tells you to create, in `handoff/gpu_campaign/`.
2. **No relaunch and no code correction without telling Nadim first** and receiving his
   agreement. Never relaunch a run marked failed and never change a seed, a config or an option. A run that fails is a
   result: it is reported, not repaired.
3. Never commit result directories, nor weights outside the `*.weights.tgz` archives that step 5 writes. The only
   commit is the one of step 5, of the directory `handoff/gpu_campaign/`.
4. If anything happens that this file does not foresee (an error, a message not described here, a command that
   fails, a pod that does not start), stop: do not try to fix it. Go to step 5 with what you have, and report the
   complete output of the command that failed.
5. The pilot's run directory read by the preflight
   (`/scratch/helplessness-worldmodels/results/E3_worldmodel_4zones/grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z`)
   is only read: do not modify, move or delete anything in it.

Every command below runs in a terminal of the Kubeflow notebook (`language-mind-v1-0`, namespace
`kubeflow-v-weilnhammer`), from the checkout `/scratch/helplessness-worldmodels`. Steps 2 to 5 use the shell variables
`PRE` and `TRAIN` set on the way. If the terminal was closed in between, go back to the checkout and set them again
from the newest directories:

```bash
cd /scratch/helplessness-worldmodels
ls -dt logs/kubeflow/preflight-*/ logs/kubeflow/train-*/
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
{'source_sha256': '77015377ea0e67ab9d81e2be067274bb1a8b4c615100a962c749d36950464e1c', 'source_n_files': 273}
```

If `git pull` fails or the hash differs, stop: do not launch anything, go to step 5 (without training). The
preflight checks the same hash again on the GPU node.

Use the commands of steps 2 and 3 exactly as written, `--cpu 3` included: the batch is prescribed **3 CPUs in
total** (the notebook keeps 2 of the namespace's 5). `submit_kubeflow.py` refuses another `--cpu`, and the preflight
fails unless 3 CPUs are both requested and usable. If the submission is refused, or if the pod stays `Pending`, stop
(rule 4): do not change the number yourself.

## Step 2 -- preflight (about 1 hour)

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_campaign --max-parallel 4 --cpu 3
```

It prints a JSON record and "Submitted, not yet confirmed running". Copy the value of `"output"` from that record (a
path ending in `logs/kubeflow/preflight-<date>`) into the variable `PRE`:

```bash
PRE=<the "output" path printed above>
```

The preflight checks CUDA, TF32 off and determinism on, **3 CPUs requested and 3 usable**, the code hash above, that
no acceleration is configured; then the campaign's measures on the pilot's final weights (rule 5: read only; about 30
to 45 minutes, written to `$PRE/runs/measure_smoke/`), a truncated real run (1,100 environment steps, about 15
minutes, its outputs checked), and the closed loop in 2 and then 4 processes side by side to choose how many runs
share the GPU; it projects the batch's duration. There is no speed bench this time. Follow the pod until it has
finished:

```bash
kubectl get pods
cat "$PRE/status.json"
```

The preflight has finished when `status.json` shows an `"exit_code"` that is a number (0 or 1), not `null`. If the
pod stays `Pending` for more than 15 minutes, stop (rule 4). When it has finished:

```bash
tail -n 30 "$PRE/preflight.log"
cat "$PRE/runs/preflight_verdict.json"
```

## Step 3 -- decision rule (apply it exactly)

The last line of `preflight.log` is one of:

- `PREFLIGHT DECISION: launch-max-parallel-4` -- launch the eight runs, four at a time (the next one starts when one
  finishes):

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_campaign --max-parallel 4 --cpu 3
  ```

- `PREFLIGHT DECISION: launch-max-parallel-2` -- launch the eight runs, two at a time:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_campaign --max-parallel 2 --cpu 3
  ```

- `PREFLIGHT DECISION: stop` -- do not launch; go to step 5 (without training) with the preflight's output.

Check it against `$PRE/runs/preflight_verdict.json`: launch only if `passed` is `true`, `problems` is empty,
`cpus.requested` and `cpus.usable` are both 3, and `decision` is the line above. If they disagree, stop (rule 4).
`projected_batch_hours` in the verdict is the projected duration of the whole batch (under 336 hours, else the
decision is stop).

## Step 4 -- follow the training (about 3.5 days)

After the training submission, copy its `"output"` path (ending in `logs/kubeflow/train-<date>`):

```bash
TRAIN=<the "output" path printed by the training submission>
```

```bash
kubectl get pods
python -c "import json, sys; s = json.load(open(sys.argv[1])); print({k: v.get('exit_code') for k, v in s.items()})" "$TRAIN/status.json"
tail -n 2 "$TRAIN"/*.log
grep -h "SWITCH\|full measures at\|STOPPED\|Error" "$TRAIN"/*.log
```

Each run logs a line every 1,000 environment steps (`env step  12000/300000 | ...`), then, at every checkpoint after
the switch, `full measures at environment step ...` and `full measures at ... done in ... s` (about 20 to 40 minutes
each, during which the run's step counter does not move: this is expected). Its directory is
`results/E3_worldmodel_4zones/grid12x12__4zones__pair-<experimental|control>__worldmodel-agent-gpu_campaign__seed<NN>__<date>/`;
`checkpoints.jsonl` and `measures/` there grow at each checkpoint and can be read before the end. `exit_code` is
`null` while a run is running or queued, `0` when it finished, `1` when it failed (a failure is a result: rule 2). The
batch is finished when every `exit_code` is a number and the pod is `Completed` or `Error`.

### Step 4b -- if the pod disappears before the end

If the pod disappeared (evicted, node restarted, job killed) while some `exit_code` is still `null`: **do not resume
anything yourself (rule 2).** Tell Nadim, with the output of the commands of step 4 and, for each unfinished run, its
directory and its `checkpoints/latest.json`. Only if he agrees, each unfinished run is continued from its last resume
point (every 10,000 environment steps and every checkpoint), in its own directory, one submission per run:

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase resume --batch gpu_campaign --cpu 3 --resume <the run's directory>
```

and a run never started (`not_started` in `status.json`) is launched with
`--phase train --batch gpu_campaign --runs <its name> --max-parallel 4 --cpu 3`, also only with his agreement. Write
down in `handoff/gpu_campaign/handoff.md` (step 5, section `## Interruptions`) what was done, when, from which
environment step, and his agreement. A run whose log ends with an error of its own (`SeparationNotReachedError`,
`CollapseAfterSwitchError`, a Python traceback) is never resumed.

## Step 5 -- send back the results

With the training (and every preflight directory you created):

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --train-dir "$TRAIN" --preflight-dir "$PRE" --out handoff/gpu_campaign --with-weights
```

Without training (stop at step 1 or 3):

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --preflight-dir "$PRE" --out handoff/gpu_campaign
```

The script writes `handoff/gpu_campaign/handoff.md` (each run's status and the number of checkpoints measured), and
for each run three archives: `<run>.tgz` (its records, final measures, counts, twin kernels -- about 3 MB),
`<run>.measures.tgz` (the measures at its checkpoints -- about 10 MB) and `<run>.weights.tgz` (the weights at every
checkpoint and the final replay -- about 40 MB; never `resume.pt`), and `logs.tgz`. In total about 420 MB. If it ends
with `STOP: archive(s) above 90 MB`, do not commit: stop (rule 4). After a stop under rule 4, add at the end of
`handoff/gpu_campaign/handoff.md` a section `## Stop` with the command that failed and its complete output. Then
commit this directory alone and push:

```bash
git status --short
git add handoff/gpu_campaign
git commit -m "E3 gpu_campaign: results handoff" -- handoff/gpu_campaign
git push origin main
```

`git status --short` must show only `handoff/` as new. If `git commit` or `git push` fails, stop and report its
output. The run directories stay on the volume, untouched, after the handoff.
