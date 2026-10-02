# Running the current E3 GPU batch (UCL ARC Kubeflow, A100)

**Current batch: gpu_pilot_cl -- the closed-loop pilot (4 runs). Follow this file from top to bottom, skipping
nothing.**

What it is: the agent in closed loop (`run_agent.py`), configuration `configs/e3/gpu_pilot_closed_loop.yaml`, seeds 0
and 5, experimental environment and reference, 300,000 environment steps each, the world model trained on 16 windows
of 64 steps with 0.6 updates per environment step (the configuration retained for E3, gpu_e30b_win64), no acceleration. Its purpose and the shapes expected from it are written in `docs/e3/DECISIONS.md` (2026-10-02 (19)).
The preflight starts with a speed bench (about 30 minutes) whose numbers are informative only. Earlier batches are in
`RUN_GPU_HISTORY.md`: **do not use it for this batch.**

The four runs, in launch order (the experimental runs first):

```
seed0_experimental   seed5_experimental   seed0_reference   seed5_reference
```

## Rules for the person or the assistant running this file

1. Do not modify any file tracked by git (code, configs, docs). The only files you create are the ones this file
   tells you to create, in `handoff/gpu_pilot_cl/`.
2. Never relaunch a run marked failed and never change a seed, a config or an option. A run that fails is a result:
   it is reported, not repaired. A run is continued only by the resume of step 4b, and only after an interruption
   that was not a failure of the run itself (pod evicted, node restarted, job killed).
3. Never commit weights (`.pt` files) or complete result directories. The weights at the checkpoints stay on the
   volume, in the run directories. The only commit is the one of step 5, of the directory `handoff/gpu_pilot_cl/`.
4. If anything happens that this file does not foresee (an error, a message not described here, a command that
   fails, a pod that does not start), stop: do not try to fix it. Go to step 5 with what you have, and report the
   complete output of the command that failed.

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
{'source_sha256': '3a71c7453c4aaa51175fe714ad7fc2e3ea3a825e51659fc4219cabd550bc7e15', 'source_n_files': 268}
```

If `git pull` fails or the hash differs, stop: do not launch anything, go to step 5 (without training). The
preflight checks the same hash again on the GPU node.

Use the commands of steps 2 and 3 exactly as written, `--cpu 3` included: the batch is prescribed **3 CPUs in
total** (the notebook keeps 2 of the namespace's 5). `submit_kubeflow.py` refuses another `--cpu`, and the preflight
fails unless 3 CPUs are both requested and usable. If the submission is refused, or if the pod stays `Pending`, stop
(rule 4): do not change the number yourself.

## Step 2 -- preflight, with the speed bench (about 1 hour)

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_pilot_cl --cpu 3
```

It prints a JSON record and "Submitted, not yet confirmed running". Copy the value of `"output"` from that record (a
path ending in `logs/kubeflow/preflight-<date>`) into the variable `PRE`:

```bash
PRE=<the "output" path printed above>
```

The preflight checks CUDA, TF32 off and determinism on, **3 CPUs requested and 3 usable**, the code hash above, that
no acceleration is configured, then runs the speed bench (informative, about 30 minutes, written to
`$PRE/runs/bench_agent.json`), a truncated real run (1,100 environment steps, about 15 minutes, its outputs checked),
and the closed loop in 2 and then 4 processes side by side to choose how many runs share the GPU. Follow the pod until
it has finished:

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

- `PREFLIGHT DECISION: launch-max-parallel-4` -- launch the four runs at once:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_pilot_cl --max-parallel 4 --cpu 3
  ```

- `PREFLIGHT DECISION: launch-max-parallel-2` -- launch two at a time (the experimental runs first, the references
  when they finish):

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_pilot_cl --max-parallel 2 --cpu 3
  ```

- `PREFLIGHT DECISION: stop` -- do not launch; go to step 5 (without training) with the preflight's output.

Check it against `$PRE/runs/preflight_verdict.json`: launch only if `passed` is `true`, `problems` is empty,
`cpus.requested` and `cpus.usable` are both 3, and `decision` is the line above. If they disagree, stop (rule 4). The
speed bench never changes the decision.

## Step 4 -- follow the training (one to several days)

After the training submission, copy its `"output"` path (ending in `logs/kubeflow/train-<date>`):

```bash
TRAIN=<the "output" path printed by the training submission>
```

```bash
kubectl get pods
python -c "import json, sys; s = json.load(open(sys.argv[1])); print({k: v.get('exit_code') for k, v in s.items()})" "$TRAIN/status.json"
tail -n 2 "$TRAIN"/*.log
grep -h "SWITCH\|STOPPED\|Error" "$TRAIN"/*.log
```

Each run logs a line every 1,000 environment steps (`env step  12000/300000 | ...`). Its directory is
`results/E3_worldmodel_4zones/grid12x12__4zones__pair-<experimental|control>__worldmodel-agent-gpu_pilot_closed_loop__seed<NN>__<date>/`;
`checkpoints.jsonl` there grows at each checkpoint and can be read before the end. `exit_code` is `null` while a run
is running or queued, `0` when it finished, `1` when it failed (a failure is a result: rule 2). The batch is finished
when every `exit_code` is a number and the pod is `Completed` or `Error`.

### Step 4b -- only after an interruption (pod evicted, node restarted, job killed)

If the pod disappeared while some `exit_code` is still `null` (and the run's log does not end with an error of the
run itself), each unfinished run is continued from its last resume point (every 10,000 environment steps and every
checkpoint), in its own directory, one submission per run:

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase resume --batch gpu_pilot_cl --cpu 3 --resume <the run's directory>
```

Write down in `handoff/gpu_pilot_cl/handoff.md` (step 5, section `## Interruptions`) which runs were resumed, from
which environment step (`checkpoints/latest.json` of the run) and why. A run whose log ends with an error of its own
(`SeparationNotReachedError`, `CollapseAfterSwitchError`, a Python traceback) is NOT resumed (rule 2).

## Step 5 -- send back the results

With the training (and every preflight directory you created):

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --train-dir "$TRAIN" --preflight-dir "$PRE" --out handoff/gpu_pilot_cl
```

Without training (stop at step 1 or 3):

```bash
python experiments/E3_worldmodel_4zones/make_handoff.py --preflight-dir "$PRE" --out handoff/gpu_pilot_cl
```

The script writes `handoff/gpu_pilot_cl/handoff.md` (each run's status), one light archive per run (its measures,
checkpoint records, metrics and counts -- **without** the weights, which stay on the volume), `logs.tgz` (logs,
preflight, speed bench) and `bench_agent.json` (the speed bench). If it ends with `STOP: archive(s) above 90 MB`, do
not commit: stop (rule 4). After a stop under rule 4, add at the end of `handoff/gpu_pilot_cl/handoff.md` a section
`## Stop` with the command that failed and its complete output. Then commit this directory alone and push:

```bash
git status --short
git add handoff/gpu_pilot_cl
git commit -m "E3 gpu_pilot_cl: results handoff" -- handoff/gpu_pilot_cl
git push origin main
```

`git status --short` must show only `handoff/` as new. If `git commit` or `git push` fails, stop and report its
output.
