#!/usr/bin/env bash
# The first GPU run: the 4 runs of configs/e3/gpu_e30.yaml (seeds 0 and 5 x experimental /
# reference) and the bridge run of configs/e3/gpu_e30_bridge.yaml (seed 0, experimental,
# mini_long's network) -- 5 processes side by side on one GPU. Run from the repository root, after preflight.py has passed:
#
#     bash experiments/E3_worldmodel_4zones/launch_gpu_e30.sh
#
# Each process writes its own run directory under results/E3_worldmodel_4zones/ and its console
# to logs/gpu_e30/seed<N>_<condition>.log (bridge: bridge_seed0_experimental.log); nohup keeps the runs alive if the terminal closes.
# The script prints the PIDs and returns at once. Wait for the runs with `wait` or
#     tail -f logs/gpu_e30/*.log
# An interrupted run is continued, in its own directory, with
#     python experiments/E3_worldmodel_4zones/run_mini.py --resume results/E3_worldmodel_4zones/<run directory>
# A run that stops on a training guard (no separation by the deadline, collapse) is a result:
# it is reported as a failure, never re-launched with another seed without it being recorded.
#
# Optional W&B tracking (wandb_logger.py): set WANDB_API_KEY before running this script (and
# WANDB_MODE=offline if the machine has no internet; then `wandb sync` the runs' wandb/
# directories afterwards). Both variables are passed to the 5 processes. Without a key, W&B is
# off and the runs are unchanged. With W&B on, each run directory is also uploaded as a W&B
# artifact at the end (weights included, resume.pt excluded).
set -euo pipefail

CONFIG=configs/e3/gpu_e30.yaml
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
mkdir -p logs/gpu_e30
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
  export WANDB_MODE=${WANDB_MODE:-online}
  echo "W&B tracking on (mode $WANDB_MODE)"
else
  echo "W&B tracking off (WANDB_API_KEY not set)"
fi

for seed in 0 5; do
  for condition in experimental reference; do
    log="logs/gpu_e30/seed${seed}_${condition}.log"
    nohup "$PYTHON" experiments/E3_worldmodel_4zones/run_mini.py --config "$CONFIG" --device "$DEVICE" \
      --seeds "$seed" --conditions "$condition" --wandb-artifact > "$log" 2>&1 &
    echo "seed $seed $condition: pid $! -> $log"
    sleep 2   # distinct UTC timestamps in the run names
  done
done
log="logs/gpu_e30/bridge_seed0_experimental.log"
nohup "$PYTHON" experiments/E3_worldmodel_4zones/run_mini.py --config configs/e3/gpu_e30_bridge.yaml --device "$DEVICE" \
  --seeds 0 --conditions experimental --wandb-artifact > "$log" 2>&1 &
echo "bridge seed 0 experimental: pid $! -> $log"
echo "5 runs launched. Results: results/E3_worldmodel_4zones/grid12x12__4zones__pair-*__worldmodel-gpu_e30*__seed0{0,5}__*/"
