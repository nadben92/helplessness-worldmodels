"""Shared test helper: loads an experiment-local module (experiments/<dir>/<file>.py)
by explicit file path under a caller-chosen unique name, bypassing
sys.modules entirely.

Why this exists: several experiment directories reuse the same filename
(both E1 and E2 have an env.py, a measurement.py, an artifact.py -- by
design, see CLAUDE.md's "Adding an experiment" convention, each
experiment's own orchestration files are named for what they do, not
namespaced per experiment). sys.modules caches an import by its bare
name regardless of sys.path, so a plain `sys.path.insert(...); import env`
pattern is only safe by test-collection-order luck: whichever test file
happens to import "env" first in a given pytest session silently wins for
every later "import env" anywhere else in that session, even after the
inserting test file removes its own sys.path entry. This module makes
that impossible by construction -- every load gets its own name in
sys.modules, keyed by caller-chosen unique_name, so two experiments'
same-named files can be loaded side by side in one process without
either shadowing the other.

A loaded module's OWN sibling imports (e.g. E2's twin.py doing
`from matched_control import collect_matched`) still resolve via normal
sys.path lookup -- load_experiment_module leaves the experiment's own
directory on sys.path for the duration of exec_module() so those resolve,
then restores sys.path exactly as it found it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def load_experiment_module(experiment_dir_name: str, module_filename: str, unique_name: str) -> ModuleType:
    """experiment_dir_name: e.g. "E1_offline_4zones" or "E2_online_4zones".
    module_filename: e.g. "env.py". unique_name: any string not already
    used for another loaded module in this process (a plain descriptive
    string is enough -- collisions raise cleanly by overwriting
    sys.modules[unique_name], never by silently returning the wrong
    module, as long as callers pick distinguishable names).

    Also isolates the loaded module's OWN internal bare sibling imports
    (e.g. experiments/E2_online_4zones/pilot.py does `import env as
    e2_env` at its top level, matching the same bare-import convention
    every experiment's own orchestration files already use, e.g. E1's
    run_etape_b.py). Without this, such an import inside the module being
    exec'd here would hit whatever some UNRELATED test file already
    cached under that bare name (e.g. tests/envs/test_e1_4zone_layout.py
    permanently caches sys.modules["env"] = E1's env.py at its own
    top-level sys.path.insert) instead of resolving fresh via sys.path --
    exactly the collision this whole module exists to prevent, but
    inside the loaded file's own body rather than the caller's. Every
    filename actually present in exp_dir is evicted from sys.modules for
    the duration of exec_module() and restored exactly afterward, so this
    call has zero side effects on the rest of the pytest session.
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
