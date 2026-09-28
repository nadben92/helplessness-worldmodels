#!/usr/bin/env bash
# The 4 runs of configs/e3/gpu_e30.yaml (seeds 0 and 5 x experimental / reference), one process
# each, side by side on one GPU. Run from the repository root, after preflight.py has passed:
#
#     bash experiments/E3_worldmodel_4zones/launch_gpu_e30.sh
#
# Each process writes its own run directory under results/E3_worldmodel_4zones/ and its console
# to logs/gpu_e30/<seed>_<condition>.log; nohup keeps the runs alive if the terminal closes.
# The script prints the PIDs and returns at once. Wait for the runs with `wait` or
#     tail -f logs/gpu_e30/*.log
# An interrupted run is continued, in its own directory, with
#     python experiments/E3_worldmodel_4zones/run_mini.py --resume results/E3_worldmodel_4zones/<run directory>
# A run that stops on a training guard (no separation by the deadline, collapse) is a result:
# it is reported as a failure, never re-launched with another seed without it being recorded.
set -euo pipefail

CONFIG=configs/e3/gpu_e30.yaml
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
mkdir -p logs/gpu_e30
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0 5; do
  for condition in experimental reference; do
    log="logs/gpu_e30/seed${seed}_${condition}.log"
    nohup "$PYTHON" experiments/E3_worldmodel_4zones/run_mini.py --config "$CONFIG" --device "$DEVICE" \
      --seeds "$seed" --conditions "$condition" > "$log" 2>&1 &
    echo "seed $seed $condition: pid $! -> $log"
    sleep 2   # distinct UTC timestamps in the run names
  done
done
echo "4 runs launched. Results: results/E3_worldmodel_4zones/grid12x12__4zones__pair-*__worldmodel-gpu_e30__seed0{0,5}__*/"
