"""Tooling for runs on the GPU cluster (UCL ARC Kubeflow), shared by every
experiment from E3 on: the job supervisor and runtime patch (kubeflow.py),
the results handoff and the results-branch layout (handoff.py), optional
Weights & Biases tracking (wandb_logger.py). Experiment-specific batch tables
stay in each experiment's own submit script."""
