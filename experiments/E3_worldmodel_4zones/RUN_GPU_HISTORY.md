# HISTORIQUE -- NE PAS UTILISER POUR LE BATCH EN COURS

> Ce fichier conserve les instructions des lancements précédents (premier batch gpu_e30 et son pont,
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
