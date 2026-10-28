"""Style of the E5 slide figures: the v2 chart (branch `slides`, slides/style_v2/palette.py and common.py) reproduced,
plus the shuffle colour. matplotlib's default style (boxed axes, no grid), axis names 18 pt, ticks 15 pt, thick lines,
no title, legend or annotation; reference lines dashed, without text.

save(fig, id) writes slides/out_e5/png/<id>.png (300 dpi, transparent), svg/<id>.svg (transparent) and
png_white/<id>.png (300 dpi, white). Data are read read-only from the E5 analysis tables.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLIDES = Path(__file__).resolve().parents[1]
REPO = SLIDES.parent
OUT = SLIDES / "out_e5"
TABLES = REPO / "experiments" / "E5_belief_training" / "analysis_450k" / "tables"
FULL = (13.33, 7.5)

# palette v2 (slides/style_v2/palette.py on branch `slides`), copied
EXPOSED = "#E8412C"      # vermilion: baseline exposed
REFERENCE = "#1F6FEB"    # bright blue
THEORY = "#000000"
CHANCE = "#888888"
ZONE_LINE = {"UD": "#FF5A36", "CD": "#4DA3FF", "corridor": "#6E6E6E"}
LW = 2.8
MS = 9
SIZE_LABEL = 18
SIZE_TICK = 15
# E5 only: the shuffle exposed models. Not in the v2 palette: violet-magenta, distinct from CORRECTION (#6718AB, E7);
# deuteranopia deltaE76 (Machado 2009, style/common.py of `slides`) >= 24.6 against every colour it shares a figure with.
SHUFFLE = "#A21CAF"

CONDITIONS = [("baseline", EXPOSED), ("shuffle", SHUFFLE)]
POSITIONS = [("ud", ZONE_LINE["UD"]), ("cd", ZONE_LINE["CD"]), ("neutral", ZONE_LINE["corridor"])]


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({"axes.labelsize": SIZE_LABEL, "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_TICK,
                         "lines.linewidth": LW, "lines.markersize": MS, "svg.fonttype": "none"})


apply_style()


def full(nrows: int = 1, ncols: int = 1, **kw):
    return plt.subplots(nrows, ncols, figsize=FULL, **kw)


def ref_lines(ax, theory: bool = True) -> None:
    """Zero (grey) and -log2 5 (black), dashed, no text."""
    ax.axhline(0.0, color=CHANCE, ls="--", lw=2.2, zorder=1)
    if theory:
        ax.axhline(-2.321928094887362, color=THEORY, ls="--", lw=2.2, zorder=1)


def table(name: str) -> list[dict]:
    p = TABLES / name
    if not p.is_file():
        raise FileNotFoundError(f"missing data source: {p}")
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def save(fig, fig_id: str) -> list[Path]:
    for sub in ("png", "svg", "png_white"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [OUT / "png" / f"{fig_id}.png", OUT / "svg" / f"{fig_id}.svg", OUT / "png_white" / f"{fig_id}.png"]
    fig.savefig(paths[0], dpi=300, transparent=True)
    fig.savefig(paths[1], transparent=True)
    fig.savefig(paths[2], dpi=300, facecolor="white")
    plt.close(fig)
    return paths
