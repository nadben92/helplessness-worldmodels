"""Re-export shim: this module moved to helplessness.cluster.wandb_logger on 2026-10-14
(reorganisation, docs/EXPERIMENTS.md). Every name of that module is re-exported
here, so the existing imports (bare sibling imports, load_experiment_module, the
tests, the paper's scripts) keep working unchanged. New code imports
helplessness.cluster.wandb_logger directly."""

from helplessness.cluster import wandb_logger as _moved

globals().update({name: value for name, value in vars(_moved).items() if not name.startswith("__")})
