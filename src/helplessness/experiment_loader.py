"""Production copy of tests/_experiment_loader.py::load_experiment_module
(see docs/e3/DECISIONS.md, point C): E3's own orchestration code
(experiments/E3_worldmodel_4zones/) needs to import
experiments/E2_online_4zones/env.py (and twin.py) AS-IS, at runtime, not
just from tests -- and production code cannot import from tests/. This
is a single, exact duplicate of the test helper's logic (generic import
machinery, no E1/E2 domain knowledge in it), kept here so
tests/_experiment_loader.py stays untouched and test-only, per your
instruction. If the two ever need to diverge, that is itself worth
flagging -- they are expected to stay identical.

Why this exists at all: several experiment directories reuse the same
filename (both E1 and E2 have an env.py -- by design, see CLAUDE.md's
"Adding an experiment" convention). sys.modules caches an import by its
bare name regardless of sys.path, so a plain
`sys.path.insert(...); import env` pattern is only safe by whichever
caller happens to import "env" first in a given process. This module
makes that impossible by construction: every load gets its own name in
sys.modules, keyed by a caller-chosen unique_name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def load_experiment_module(experiment_dir_name: str, module_filename: str, unique_name: str) -> ModuleType:
    """experiment_dir_name: e.g. "E2_online_4zones". module_filename:
    e.g. "env.py". unique_name: any string not already used for another
    loaded module in this process.

    Also isolates the loaded module's OWN internal bare sibling imports
    (e.g. experiments/E2_online_4zones/twin.py does
    `from matched_control import collect_matched`) so they resolve via
    that experiment's own directory, not whatever some unrelated caller
    already cached under that bare name.
    """
    exp_dir = EXPERIMENTS_DIR / experiment_dir_name
    path = exp_dir / module_filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module

    exp_dir_str = str(exp_dir)
    inserted = exp_dir_str not in sys.path
    if inserted:
        sys.path.insert(0, exp_dir_str)

    sibling_stems = {p.stem for p in exp_dir.glob("*.py")}
    evicted = {stem: sys.modules.pop(stem) for stem in sibling_stems if stem in sys.modules}

    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(exp_dir_str)
        for stem in sibling_stems:
            sys.modules.pop(stem, None)  # drop whatever THIS call cached under that bare name
        sys.modules.update(evicted)  # restore exactly what was there before this call
    return module
