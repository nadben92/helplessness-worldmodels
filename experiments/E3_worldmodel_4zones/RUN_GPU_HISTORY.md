# HISTORIQUE -- NE PAS UTILISER POUR LE BATCH EN COURS

> Ce fichier conserve les instructions des lancements précédents (premier batch gpu_e30 et son pont, puis le batch gpu_e30b,
> lanceur serveur `launch_gpu_e30.sh`, installation manuelle, W&B par `export`), telles qu'elles
> étaient au 2026-09-30. **Il ne s'applique pas au batch en cours.** Le batch en cours se lance
> uniquement en suivant `experiments/E3_worldmodel_4zones/RUN_GPU.md`.

---

# Running the first E3 GPU runs (A100)

Hands-on time: about 20 minutes. The runs then take a few hours on their own.

## UCL ARC Kubeflow

Use `submit_kubeflow.py` instead of the server launcher below. Run the submission
commands from a Kubeflow notebook with the Trainer SDK installed. Keep this checkout
on `scratch-volume`, mounted at `/scratch`, and do not edit it while a job is running.
The wrapper uses the cluster's PyTorch image, an isolated `.venv-cluster` retaining
the image's CUDA torch, and a writable `.cluster-home` in the checkout. It requests
one GPU, 3 CPUs and 64 GiB RAM; the notebook must fit in the remaining namespace
quota (currently 5 CPUs, 120 GiB and one A100 in total).

```bash
cd /scratch/helplessness-worldmodels
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight
```

Submission returns a job ID, not proof of execution. Check its pod with
`kubectl get pods`, then `kubectl logs -f <pod-name>`. Setup/GPU messages appear
in pod logs; the preflight's full output is in the printed persistent output
directory's `preflight.log`. `status.json` records child exit codes; the adjacent
`*.submission.json` records the job ID. Preflight outputs are under that directory's
`runs/`. W&B is disabled for preflight. Send the preflight output and duration
projection back before proceeding.

Only after preflight review:

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train
```

This supervises all five processes on the same GPU, waits for every exit and returns
a failure if any run fails. It never silently retries a scientific failure.

**Subset and simultaneous processes (2026-09-29).** `--runs` selects runs among
`seed0_experimental seed0_reference seed5_experimental seed5_reference
bridge_seed0_experimental` (default: all), `--max-parallel N` caps how many share
the GPU at once (default: all; the others wait in a queue). Pass the **same**
`--runs` and `--max-parallel` to the preflight and to the training: the preflight
then tests that number of processes, including a memory test in which every
process takes a diagnostic point and a final measurement at the same time (it
fails if one runs out of memory or if their peaks add up to more than 90% of the
GPU). Example:

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --runs seed5_experimental bridge_seed0_experimental --max-parallel 2
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --runs seed5_experimental bridge_seed0_experimental --max-parallel 2
```

**Second batch, gpu_e30b (2026-09-30).** Eight runs (`win64_` and `win16_` x `seed0` / `seed5` x
`experimental` / `reference`), the bridge's network at 1,024 images per gradient step:

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_e30b --max-parallel 8
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_e30b --max-parallel 8
```

If the preflight's memory test fails at 8, run both commands with `--max-parallel 4`.

**CPUs.** `--cpu N` sets the job's CPU request (default 3, as the first batch). Pass the
same N to the preflight and to the training: the preflight runs under that allocation,
checks it against the container's limit (a warning if they differ) and prints the
projected wall time of the whole batch for it (`projected_batch_hours`, counting the
queue when `--max-parallel` is below the number of runs). The request must fit in the
namespace quota (5 CPUs in all, the notebook included). Example with 5 CPUs:

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_e30b --max-parallel 8 --cpu 5
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase train --batch gpu_e30b --max-parallel 8 --cpu 5
```

**Code version.** A run records its git commit, and always a sha256 of the
contents of `src/`, `experiments/`, `configs/` and `pyproject.toml`
(`meta.yaml`: `source_sha256`); it refuses to start with neither. The container
has no git: compare `source_sha256` with
`python -c "from helplessness.runs import source_tree_hash; print(source_tree_hash('.'))"`
on a checkout of the intended commit.
Do not use `launch_gpu_e30.sh` as the container entry point: it backgrounds its
children and exits. Kubernetes needs a foreground supervisor. The five processes
each retain their configured four CPU threads; the preflight measures contention
under the job's actual CPU allocation rather than changing the experiment.

Training results remain under `results/E3_worldmodel_4zones/` on the volume.
Resume one interrupted run with:

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase resume \
  --resume /scratch/helplessness-worldmodels/results/E3_worldmodel_4zones/<run-directory>
```

Optional W&B: provision a namespace Secret containing the key `WANDB_API_KEY`,
then add `--wandb-secret <secret-name>` to train/resume submissions. Never include
the credential in committed files or command arguments. The Secret is mounted
read-only and read inside the worker because this cluster rejects container `env`
patches. Confirm `logging.wandb.entity` in both configs for the intended W&B team.
Without a Secret, tracking is disabled. Retain the persistent volume even with
artifact uploads: W&B artifacts exclude `resume.pt`.

## 1. Install (once)

    git clone https://github.com/nadben92/helplessness-worldmodels.git
    cd helplessness-worldmodels
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[e3,dev,wandb]"
    python -c "import torch; ok = torch.cuda.is_available(); print(ok, torch.cuda.get_device_name(0) if ok else '')"

The last line must print `True` and the GPU's name. If it prints `False`, install the torch build matching the CUDA driver shown by `nvidia-smi` (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`), then run `pip install -e ".[e3,dev,wandb]"` again.

## 2. Short check (about 15 minutes)

    python experiments/E3_worldmodel_4zones/preflight.py --config configs/e3/gpu_e30.yaml --extra-config configs/e3/gpu_e30_bridge.yaml --device cuda

It ends with `PREFLIGHT PASSED` or `PREFLIGHT FAILED`. Please send me its output before going to step 3.
For a subset of runs or fewer simultaneous processes, add `--parallel <N>` (the number that will share the GPU) and drop `--extra-config` if the bridge is not launched.

## 3. Launch the runs

Before launching, in the same terminal:

    export WANDB_API_KEY=<key sent to you privately>

If the machine has no internet access, also run `export WANDB_MODE=offline` (see step 4). Then:

    bash experiments/E3_worldmodel_4zones/launch_gpu_e30.sh

This starts a supervisor in the background that runs the 5 runs; it keeps running if you close the terminal. To follow them: `tail -f logs/gpu_e30/train-*/*.log`. For a subset or fewer simultaneous runs (tested by the preflight with the same number): `RUNS="seed5_experimental bridge_seed0_experimental" MAX_PARALLEL=2 bash experiments/E3_worldmodel_4zones/launch_gpu_e30.sh`.

- If the machine restarts, resume each unfinished run with (after the same `export` line(s)):
  `python experiments/E3_worldmodel_4zones/run_mini.py --resume results/E3_worldmodel_4zones/<run folder>`
- A run may stop by itself and be marked `failed`. That is a possible result, not a bug: nothing to fix, just include it in what you send back.
- If W&B cannot connect, the runs print a warning and continue normally.

## 4. Send back the results

Right away, the light archive (everything except the model weights, about 75 MB):

    tar czf gpu_e30_results_light.tgz --exclude='*/checkpoints/*' --exclude='*/failure/*.pt' results/E3_worldmodel_4zones/*gpu_e30* logs/gpu_e30

If you used `WANDB_MODE=offline`, from a machine with internet access (or the same one later), upload the tracking with:

    wandb sync results/E3_worldmodel_4zones/*gpu_e30*/wandb/offline-run-*

Only if I ask for it, the full archive with the weights (about 2 GB), shared through a OneDrive or Google Drive link:

    tar czf gpu_e30_results.tgz results/E3_worldmodel_4zones/*gpu_e30* logs/gpu_e30

---

<!-- RUN_GPU.md of the gpu_e30b batch, as it was when that batch finished (moved here 2026-10-01). -->

# (HISTORY) Running the gpu_e30b batch (UCL ARC Kubeflow, A100)

**Current batch: gpu_e30b (8 runs). Follow this file from top to bottom, skipping nothing.**

> **Status, 2026-09-30 (5): this batch is FINISHED** (8 / 8 runs done on 2026-09-30, results in
> `handoff/gpu_e30b/`, analysis in `reports/E3_GPU_gpu_e30b_analysis_2026-09-30.md`). **Do not launch it
> again.** No next batch is defined yet; this file will be rewritten for it. The steps below are kept as the
> template, with the CPU check added after this batch (it was prescribed 5 CPUs and ran on 3).

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
{'source_sha256': '01f66ef67b756221a3dbb702ce23e27ae1f733c36d2d2760196c7bb6a9a0e9c9', 'source_n_files': 259}
```

If `git pull` fails or the hash differs, stop: do not launch anything, go to step 5 (without training).

Use the commands of steps 2 and 3 exactly as written, `--cpu 5` included. `submit_kubeflow.py` refuses a
`--cpu` other than the batch's prescribed number (5 for gpu_e30b), and requests that number when `--cpu` is
left out. If the submission is refused, or if the namespace cannot grant 5 CPUs (pod `Pending`), stop
(rule 4): do not lower the number yourself.

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

- **Launch** if `passed` is `true`, `warnings` is empty, `projected_batch_hours` is a number
  <= 20, **and `cpus.requested` and `cpus.usable` are both 5** (the number of CPUs this batch is
  prescribed; the last lines of `preflight.log` also say `CPUs: 5 requested, 5 usable`). Launch the
  training with exactly the options of the preflight that passed:

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
  missing, a memory failure at 4 processes, no verdict file, **a number of CPUs other than 5** (requested
  or usable). Do not launch; go to step 5 (without
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

---

<!-- RUN_GPU.md of the gpu_e30b_measures batch (3 CPUs, as run on 2026-10-01; moved here 2026-10-02). -->

# (HISTORY) Running the gpu_e30b_measures batch (UCL ARC Kubeflow, A100)

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
{'source_sha256': 'dae7ff2e302b16dc6744833e04f040098f5076082ef3414eeec8a3b291fddb2d', 'source_n_files': 261}
```

If `git pull` fails or the hash differs, stop: do not launch anything, go to step 5 (without measurement).
The preflight checks the same hash again on the GPU node and fails if it differs.

Use the commands of steps 2 and 3 exactly as written, `--cpu 3` included: the batch is prescribed **3
CPUs in total** (user-approved capacity adjustment; see `docs/e3/DECISIONS.md`). `submit_kubeflow.py`
refuses another `--cpu`, and the preflight fails unless 3 CPUs are both requested and usable in the
container. If the submission is refused, or if the namespace cannot grant 3 CPUs (pod `Pending`), stop
(rule 4): do not lower the number yourself.

## Step 2 -- preflight (about 30 minutes)

```bash
python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase preflight --batch gpu_e30b_measures --max-parallel 4 --cpu 3
```

It prints a JSON record and "Submitted, not yet confirmed running". Copy the value of `"output"` from
that record (a path ending in `logs/kubeflow/preflight-<date>`) into the variable `PRE`:

```bash
PRE=<the "output" path printed above>
```

The preflight checks: CUDA, TF32 off and deterministic algorithms on as in the measured runs; **3 CPUs
requested and 3 usable**; the code hash above; for each of the four runs, that its directory exists with
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
  `cpus.requested` and `cpus.usable` are both 3 (the last lines of `preflight.log` say `CPUs: 3
  requested, 3 usable` and `PREFLIGHT DECISION: launch`). Launch the measurement with exactly these
  options:

  ```bash
  python experiments/E3_worldmodel_4zones/submit_kubeflow.py --phase measure --batch gpu_e30b_measures --max-parallel 4 --cpu 3
  ```

- **Stop** in every other case (any problem, a projection above 4 hours or missing, no verdict file, a
  number of CPUs other than 3). Do not launch; go to step 5 (without measurement) with the preflight's
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

---

<!-- RUN_GPU.md of the gpu_pilot_cl batch (the closed-loop pilot, run 2026-10-03 to 05; moved here 2026-10-05). -->

# (HISTORY) Running the gpu_pilot_cl batch (UCL ARC Kubeflow, A100)

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
