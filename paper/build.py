"""Builds the paper, or one of the reports: figures, then numbers, then the PDF.

Usage (from any directory):
  python paper/build.py                  the paper: all three steps, in order
  python paper/build.py --figs           only the paper's figures
  python paper/build.py --numbers        only paper/numbers.tex
  python paper/build.py --pdf            only paper/main.pdf (latexmk)
  python paper/build.py --report e1|e3   a report (paper/reports/e1_report.pdf, e3_report.pdf): its figures, the numbers, its PDF;
                                         --figs / --numbers / --pdf select steps of the report the same way

Flags can be combined; the steps always run in the order above. The build stops at the first failing step and prints
its output.

Figures: a report owns the scripts matching its pattern (REPORTS); the paper builds every other fig_*.py. A figure
script that exits with PENDING_EXIT (paper/figures/e3_sources.py: its sources belong to a batch not run yet, the E3
final campaign) is reported as pending, not as a failure; the report shows \\pending{} in its place.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"
NUMBERS_SCRIPT = PAPER_DIR / "numbers" / "gen_numbers.py"
PENDING_EXIT = 3  # e3_sources.PENDING_EXIT
REPORTS = {
    "e1": {"tex": PAPER_DIR / "reports" / "e1_report.tex", "figures": "fig_e1_*.py"},
    "e3": {"tex": PAPER_DIR / "reports" / "e3_report.tex", "figures": "fig_e3_*.py"},
}


def run(label: str, cmd: list[str], cwd: Path, quiet: bool = False, allow_pending: bool = False) -> bool:
    """Runs one step; with quiet=True its output is shown only if it fails. Returns False if the step reported
    itself pending (allowed only with allow_pending), True if it succeeded; stops the build on any failure."""
    print(f"[build] {label}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.stdout.strip() and (not quiet or result.returncode != 0):
        print(result.stdout.rstrip())
    if allow_pending and result.returncode == PENDING_EXIT:
        return False
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        print(f"\n[build] FAILED: {label} (exit code {result.returncode})\n  command: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(1)
    return True


def figure_scripts(report: str | None) -> list[Path]:
    if report is not None:
        scripts = sorted(FIGURES_DIR.glob(REPORTS[report]["figures"]))
    else:
        owned = {p for r in REPORTS.values() for p in FIGURES_DIR.glob(r["figures"])}
        scripts = sorted(p for p in FIGURES_DIR.glob("fig_*.py") if p not in owned)
    if not scripts:
        print(f"[build] FAILED: no figure script for {'report ' + report if report else 'the paper'} in {FIGURES_DIR}",
              file=sys.stderr)
        sys.exit(1)
    return scripts


def build_figures(report: str | None) -> None:
    pending = [s.name for s in figure_scripts(report)
               if not run(f"figure {s.name}", [sys.executable, str(s)], cwd=FIGURES_DIR, allow_pending=True)]
    if pending:
        print(f"[build] pending figures (the report shows [pending]): {', '.join(pending)}")


def build_numbers() -> None:
    run("numbers", [sys.executable, str(NUMBERS_SCRIPT)], cwd=PAPER_DIR)


def build_pdf(tex: Path) -> None:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        print("[build] FAILED: latexmk not found on PATH (MiKTeX)", file=sys.stderr)
        sys.exit(1)
    run(f"pdf (latexmk {tex.name})",
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", tex.name],
        cwd=tex.parent, quiet=True)
    print(f"[build] wrote {tex.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper, or a report.")
    parser.add_argument("--figs", action="store_true", help="build the figures only")
    parser.add_argument("--numbers", action="store_true", help="build numbers.tex only")
    parser.add_argument("--pdf", action="store_true", help="build the PDF only")
    parser.add_argument("--report", choices=sorted(REPORTS), default=None, help="build this report instead of the paper")
    args = parser.parse_args()
    run_all = not (args.figs or args.numbers or args.pdf)
    tex = REPORTS[args.report]["tex"] if args.report else PAPER_DIR / "main.tex"

    if run_all or args.figs:
        build_figures(args.report)
    if run_all or args.numbers:
        build_numbers()
    if run_all or args.pdf:
        build_pdf(tex)


if __name__ == "__main__":
    main()
