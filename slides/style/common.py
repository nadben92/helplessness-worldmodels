"""Helpers shared by every figure script: style, figure sizes, annotations, data reading, export.

A figure script builds its figure with `full()` or `half()`, draws with the palette tokens, and calls `save(fig, id)`,
which writes the three exports (beige PNG, SVG, transparent PNG). Data are read read-only from the repository.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SLIDES = Path(__file__).resolve().parents[1]
REPO = SLIDES.parent
if str(SLIDES) not in sys.path:
    sys.path.insert(0, str(SLIDES))

from style import palette as P  # noqa: E402

OUT = SLIDES / "out"
FULL = (13.33, 7.5)
HALF = (6.5, 6.0)


def font_in_use() -> str:
    """The first family of palette.FONT_FAMILY that matplotlib can find."""
    names = {f.name for f in fm.fontManager.ttflist}
    for fam in P.FONT_FAMILY:
        if fam in names:
            return fam
    return "DejaVu Sans"


def apply_style() -> None:
    plt.style.use(str(SLIDES / "style" / "slides.mplstyle"))
    plt.rcParams.update({
        "font.sans-serif": [font_in_use()],
        "figure.facecolor": P.BG, "axes.facecolor": P.BG, "savefig.facecolor": P.BG,
        "text.color": P.TEXT, "axes.labelcolor": P.TEXT, "axes.edgecolor": P.AXIS,
        "xtick.color": P.AXIS, "ytick.color": P.AXIS, "xtick.labelcolor": P.TEXT, "ytick.labelcolor": P.TEXT,
        "grid.color": P.GRID, "axes.titlesize": P.SIZE_TITLE, "axes.labelsize": P.SIZE_AXIS,
        "xtick.labelsize": P.SIZE_TICK, "ytick.labelsize": P.SIZE_TICK,
    })


apply_style()


# ------------------------------------------------------------------------------------------------------------ figures
def full(nrows: int = 1, ncols: int = 1, **kw):
    """A full-slide 16:9 figure (13.33 x 7.5 in)."""
    return plt.subplots(nrows, ncols, figsize=FULL, **kw)


def half(nrows: int = 1, ncols: int = 1, **kw):
    """A half-slide figure (6.5 x 6 in)."""
    return plt.subplots(nrows, ncols, figsize=HALF, **kw)


def title(fig, text: str, sub: str | None = None, y: float = 0.965) -> None:
    """The figure title (the message), left-aligned, and an optional secondary line under it."""
    fig.text(0.045, y, text, fontsize=P.SIZE_TITLE, fontweight="bold", color=P.TEXT, ha="left", va="top")
    if sub:
        fig.text(0.045, y - 0.062, sub, fontsize=P.SIZE_ANNOT, color=P.TEXT_2, ha="left", va="top")


def footnote(fig, text: str, y: float = 0.025, width: int = 150) -> None:
    """Secondary note at the bottom, wrapped to the slide width."""
    import textwrap
    text = "\n".join(textwrap.fill(part, width) for part in text.split("\n"))
    fig.text(0.045, y, text, fontsize=P.SIZE_SMALL, color=P.TEXT_2, ha="left", va="bottom")


def hgrid(ax) -> None:
    ax.yaxis.grid(True, color=P.GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def theory_line(ax, y: float, x0=None, x1=None, label: str | None = None, side: str = "right", **kw) -> None:
    """A theoretical value: dark grey dotted line, optionally labelled at one end."""
    if x0 is None:
        ax.axhline(y, color=P.THEORY, ls=P.THEORY_LS, lw=1.6, zorder=1, **kw)
    else:
        ax.hlines(y, x0, x1, color=P.THEORY, ls=P.THEORY_LS, lw=1.6, zorder=1, **kw)
    if label:
        xa = ax.get_xlim()[1] if side == "right" else ax.get_xlim()[0]
        ax.annotate(label, (xa, y), xytext=(4 if side == "right" else -4, 0), textcoords="offset points",
                    ha="left" if side == "right" else "right", va="center", fontsize=P.SIZE_SMALL, color=P.THEORY,
                    annotation_clip=False)


def chance_line(ax, y: float, label: str | None = "chance", where: float | None = None, **kw) -> None:
    """Chance / floor: light grey dashed line, labelled just above its left or given x."""
    ax.axhline(y, color=P.CHANCE, ls=P.CHANCE_LS, lw=1.6, zorder=1, **kw)
    if label:
        x = ax.get_xlim()[0] if where is None else where
        ax.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", ha="left", va="bottom",
                    fontsize=P.SIZE_SMALL, color=P.TEXT_2)


def end_label(ax, x, y, text: str, color: str, dx: float = 6, dy: float = 0, **kw) -> None:
    """A direct label at the end of a curve."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha="left", va="center",
                fontsize=P.SIZE_ANNOT, color=color, fontweight="bold", annotation_clip=False, **kw)


def fmt(x: float, nd: int = 2, sign: bool = False) -> str:
    """A number with a true minus sign (U+2212), optionally with an explicit plus."""
    if round(x, nd) == 0:
        x = 0.0
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


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
def save(fig, fig_id: str) -> list[Path]:
    """Write out/png/<id>.png (300 dpi, beige), out/svg/<id>.svg (beige), out/png_transparent/<id>.png."""
    paths = []
    for sub in ("png", "svg", "png_transparent"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    p = OUT / "png" / f"{fig_id}.png"
    fig.savefig(p, dpi=300, facecolor=P.BG)
    paths.append(p)
    p = OUT / "svg" / f"{fig_id}.svg"
    fig.savefig(p, facecolor=P.BG)
    paths.append(p)
    p = OUT / "png_transparent" / f"{fig_id}.png"
    fig.savefig(p, dpi=300, transparent=True)
    paths.append(p)
    plt.close(fig)
    return paths


# ------------------------------------------------------------------------------------------- colour-vision check
# Machado, Oliveira & Fernandes (2009), deuteranopia at severity 1.0, applied in linear RGB.
_DEUTAN = np.array([[0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _delin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def deuteranopia(h: str) -> np.ndarray:
    return _delin(_DEUTAN @ _lin(hex_rgb(h)))


def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-1) -> CIELAB (D65)."""
    x = _lin(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M @ x / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(lab(a) - lab(b)))
