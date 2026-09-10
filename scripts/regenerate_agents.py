"""Regenerates results/agents/ (and the offline sweep CSVs) from scratch.

results/agents/**/*.npz is gitignored -- see CONTRIBUTING.md, "what gets
versioned". Running this script costs nothing but a few minutes of CPU and
reproduces the saved agents bit-for-bit (fixed seeds throughout), so not
versioning them costs nothing either.

Every downstream experiment (02_swaps, 03_intervention, 04_rescue_injection)
depends on the agents this script (re)creates. Run it once after a fresh
checkout, before running any other experiment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_SWEEP_DIR = REPO_ROOT / "experiments" / "E0_offline_5x5" / "01_offline_sweep"

CONFIGS = ["sweep_5000.yaml", "sweep_50000.yaml"]


def main() -> None:
    for config_name in CONFIGS:
        config_path = OFFLINE_SWEEP_DIR / "configs" / config_name
        print(f"=== regenerating agents from {config_path} ===")
        subprocess.run(
            [sys.executable, "run.py", "--config", str(config_path)],
            cwd=OFFLINE_SWEEP_DIR,
            check=True,
        )


if __name__ == "__main__":
    main()
