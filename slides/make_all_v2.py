"""Regenerate every v2 figure (out_v2/png transparent, out_v2/svg transparent, out_v2/png_white) and run the v2
palette check. The E7d induction runs are extracted first if their CSV is missing (read-only, needs scratch/).

    .venv/Scripts/python.exe slides/make_all_v2.py            # all figures
    .venv/Scripts/python.exe slides/make_all_v2.py E4 T1      # ids starting with these prefixes
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SLIDES = Path(__file__).resolve().parent


def run(p: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(p)], cwd=SLIDES.parent, capture_output=True, text=True)


def main(argv: list[str]) -> int:
    if not (SLIDES / "data_v2" / "e7d_induction_runs.csv").is_file():
        r = run(SLIDES / "figures_v2" / "_extract_e7d_induction.py")
        print(r.stdout.strip() or r.stderr)
    scripts = sorted(p for p in (SLIDES / "figures_v2").glob("*.py") if not p.name.startswith("_"))
    if argv:
        scripts = [p for p in scripts if any(p.stem.startswith(a) for a in argv)]
    failed = []
    for p in scripts:
        t = time.perf_counter()
        r = run(p)
        print(f"{'ok' if r.returncode == 0 else 'FAILED':6} {p.stem:36} {time.perf_counter() - t:5.1f} s {r.stdout.strip()}")
        if r.returncode:
            failed.append(p.stem)
            print(r.stderr)
    r = run(SLIDES / "style_v2" / "check_palette.py")
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr)
    print(f"{len(scripts) - len(failed)}/{len(scripts)} figures written to {SLIDES / 'out_v2'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
