"""Helpers of the v2 slide figures: matplotlib's default style (boxed axes, default ticks, no grid), larger fonts, no
titles or annotations, transparent export. Data are read read-only from the repository.

save(fig, id) writes out_v2/png/<id>.png (300 dpi, transparent), out_v2/svg/<id>.svg (transparent) and
out_v2/png_white/<id>.png (300 dpi, white).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLIDES = Path(__file__).resolve().parents[1]
REPO = SLIDES.parent
if str(SLIDES) not in sys.path:
    sys.path.insert(0, str(SLIDES))

from style_v2 import palette as P  # noqa: E402

OUT = SLIDES / "out_v2"
FULL = (13.33, 7.5)


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({"axes.labelsize": P.SIZE_LABEL, "xtick.labelsize": P.SIZE_TICK,
                         "ytick.labelsize": P.SIZE_TICK, "lines.linewidth": P.LW, "lines.markersize": P.MS,
                         "svg.fonttype": "none"})


apply_style()


def full(nrows: int = 1, ncols: int = 1, **kw):
    return plt.subplots(nrows, ncols, figsize=FULL, **kw)


def theory(ax, y: float, **kw) -> None:
    ax.axhline(y, color=P.THEORY, ls=P.THEORY_LS, lw=2.2, zorder=1, **kw)


def chance(ax, y: float, **kw) -> None:
    ax.axhline(y, color=P.CHANCE, ls=P.CHANCE_LS, lw=2.2, zorder=1, **kw)


# ---------------------------------------------------------------------------------------------------------- data
def path(*parts: str) -> Path:
    """A repository path, checked to exist (a missing source is an error, never a fallback)."""
    p = REPO.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"missing data source: {p}")
    return p


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------------------------------------- export
def save(fig, fig_id: str, tight: bool = True) -> list[Path]:
    for sub in ("png", "svg", "png_white"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    paths = [OUT / "png" / f"{fig_id}.png", OUT / "svg" / f"{fig_id}.svg", OUT / "png_white" / f"{fig_id}.png"]
    fig.savefig(paths[0], dpi=300, transparent=True)
    fig.savefig(paths[1], transparent=True)
    fig.savefig(paths[2], dpi=300, facecolor="white")
    plt.close(fig)
    return paths
