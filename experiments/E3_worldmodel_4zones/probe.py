"""Re-export shim: this module moved to helplessness.probing.probe on 2026-10-14
(reorganisation, docs/EXPERIMENTS.md). Every name of that module is re-exported
here, so the existing imports (bare sibling imports, load_experiment_module, the
tests, the paper's scripts) keep working unchanged. New code imports
helplessness.probing.probe directly."""

from helplessness.probing import probe as _moved

globals().update({name: value for name, value in vars(_moved).items() if not name.startswith("__")})
