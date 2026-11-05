"""Regenerate every figure of the progress presentation (out/png, out/svg, out/png_transparent) and run the palette
check. Each figure script is run in its own process, so a failure is reported without stopping the others.

    .venv/Scripts/python.exe slides/make_all.py            # all figures
    .venv/Scripts/python.exe slides/make_all.py E4 T1      # ids starting with these prefixes
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SLIDES = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    scripts = sorted(p for p in (SLIDES / "figures").glob("*.py") if not p.name.startswith("_"))
    if argv:
        scripts = [p for p in scripts if any(p.stem.startswith(a) for a in argv)]
    failed = []
    for p in scripts:
        t = time.perf_counter()
        r = subprocess.run([sys.executable, str(p)], cwd=SLIDES.parent, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"{status:6} {p.stem:36} {time.perf_counter() - t:5.1f} s {r.stdout.strip()}")
        if r.returncode:
            failed.append(p.stem)
            print(r.stderr)
    subprocess.run([sys.executable, str(SLIDES / "style" / "check_palette.py")], cwd=SLIDES.parent,
                   stdout=subprocess.DEVNULL)
    print(f"{len(scripts) - len(failed)}/{len(scripts)} figures written to {SLIDES / 'out'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
