"""Shared matplotlib style for every paper figure.

One place for the physical size, font sizes, palette and PDF export, so
every fig_*.py script produces figures that match the body text of
paper/main.tex. If the document class or the target venue changes, only
the constants below change.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fontTools warns about the bundled cmr10 font's zero timestamps when
# subsetting it into the PDF; harmless, and it buries real errors.
logging.getLogger("fontTools").setLevel(logging.ERROR)

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parent
OUT_DIR = PAPER_DIR / "figures" / "out"

# \textwidth of the 10pt article class used by main.tex, in TeX points.
TEXT_WIDTH_PT = 345.0
PT_PER_INCH = 72.27
TEXT_WIDTH_IN = TEXT_WIDTH_PT / PT_PER_INCH

# Body text is 10pt; figure text is set slightly smaller, as for \small.
FONT_SIZE = 9
SMALL_FONT_SIZE = 8

# Paul Tol's "bright" qualitative palette (colour-blind safe) -- the same
# one experiments/E2_online_4zones/figures.py already uses for its zones.
TOL_BRIGHT = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

# Colour of each E2 zone type, matching experiments/E2_online_4zones/figures.py's ZONE_STYLE.
ZONE_COLOR = {
    "corridor": "#000000",
    "uncontrollable_deterministic": TOL_BRIGHT["blue"],
    "controllable_noisy": TOL_BRIGHT["red"],
    "controllable_deterministic": TOL_BRIGHT["green"],
    "uncontrollable_noisy": TOL_BRIGHT["yellow"],
}


def apply_style() -> None:
    """Sets the shared rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "legend.fontsize": SMALL_FONT_SIZE,
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        # Embed TrueType fonts (no Type 3) so the PDF stays vector and searchable.
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def figure(width_fraction: float = 1.0, aspect: float = 0.55, nrows: int = 1, ncols: int = 1, **subplots_kw):
    """(fig, ax or axes) sized as a fraction of \\textwidth; height = width * aspect. Extra keyword arguments go to
    plt.subplots (sharey, gridspec_kw, ...)."""
    width = TEXT_WIDTH_IN * width_fraction
    return plt.subplots(nrows, ncols, figsize=(width, width * aspect), **subplots_kw)


def save_pdf(fig, name: str, out_dir: Path = OUT_DIR) -> Path:
    """Writes <out_dir>/<name>.pdf (vector; OUT_DIR, the paper's, by default) and closes the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    # Fixed metadata so a rerun on unchanged data gives a byte-stable PDF.
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "Producer": None})
    plt.close(fig)
    return path


def long_path(path: Path) -> str:
    """The path as a string the OS can open even beyond Windows' 260-character limit (some E3 result files are
    deeper than that); unchanged elsewhere."""
    p = str(Path(path).resolve())
    return "\\\\?\\" + p if os.name == "nt" and not p.startswith("\\\\?\\") else p


def require_file(path: Path) -> Path:
    """Fails with an explicit message if a source file is missing."""
    if not os.path.isfile(long_path(path)):
        raise SystemExit(f"Missing source file: {path}")
    return path
