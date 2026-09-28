# Running the first E3 GPU runs (A100)

Hands-on time: about 20 minutes. The runs then take a few hours on their own.

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
