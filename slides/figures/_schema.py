"""Boxes and connectors for the schematic figures (T2, T3, E3a, E4a, E7e), drawn on a full-figure axes in [0, 1]."""
from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import palette as P


def canvas(fig, rect=(0, 0, 1, 1)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, text="", fc=P.BG, ec=P.AXIS, tc=P.TEXT, size=P.SIZE_ANNOT, weight="normal", lw=1.2, ls="-",
        r=0.012, sub=None, subsize=P.SIZE_SMALL, zorder=3):
    """A rounded box centred on (x, y); optional second line in secondary text."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=zorder))
    if sub:
        ax.text(x, y + h * 0.16, text, ha="center", va="center", fontsize=size, color=tc, fontweight=weight,
                zorder=zorder + 1)
        ax.text(x, y - h * 0.22, sub, ha="center", va="center", fontsize=subsize, color=P.TEXT_2, zorder=zorder + 1)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=tc, fontweight=weight, zorder=zorder + 1)


def conn(ax, p0, p1, color=P.TEXT_2, lw=1.6, rad=0.0, label=None, ls="-", head=10, zorder=2, lsize=P.SIZE_SMALL,
         loff=(0, 8)):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={head * 0.04:.2f},head_width={head * 0.025:.2f}",
                                 mutation_scale=20, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=zorder))
    if label:
        ax.annotate(label, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2), xytext=loff, textcoords="offset points",
                    ha="center", va="bottom", fontsize=lsize, color=P.TEXT_2)
