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
