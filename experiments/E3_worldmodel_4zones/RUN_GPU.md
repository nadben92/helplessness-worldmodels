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

## 3. Launch the runs

Before launching, in the same terminal:

    export WANDB_API_KEY=<key sent to you privately>

If the machine has no internet access, also run `export WANDB_MODE=offline` (see step 4). Then:

    bash experiments/E3_worldmodel_4zones/launch_gpu_e30.sh

This starts 5 runs in the background; they keep running if you close the terminal. To follow them: `tail -f logs/gpu_e30/*.log`.

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
