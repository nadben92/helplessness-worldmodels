"""E3a (v2) -- The agent: RSSM (image, encoder, GRU, posterior, prior, decoder) and the decision side (imagination,
goal head, critic, policy). Block names only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "E3a_architecture"


def main():
    fig = C.plt.figure(figsize=C.FULL)
    ax = D.canvas(fig)
    bw, bh = 0.14, 0.12
    D.box(ax, 0.09, 0.15, bw, bh, "image", fc=P.ZONE_CORRIDOR)
    D.box(ax, 0.09, 0.47, bw, bh, "encoder")
    D.box(ax, 0.29, 0.8, bw, bh, "GRU")
    D.box(ax, 0.29, 0.47, bw, bh, "posterior")
    D.box(ax, 0.49, 0.8, bw, bh, "prior")
    D.box(ax, 0.49, 0.15, bw, bh, "decoder")
    D.conn(ax, (0.09, 0.21), (0.09, 0.41))
    D.conn(ax, (0.16, 0.47), (0.22, 0.47))
    D.conn(ax, (0.29, 0.74), (0.29, 0.53))
    D.conn(ax, (0.36, 0.8), (0.42, 0.8))
    D.conn(ax, (0.36, 0.44), (0.47, 0.21), rad=0.15)
    D.conn(ax, (0.47, 0.74), (0.34, 0.535), ls="--")
    D.box(ax, 0.75, 0.8, 0.17, bh, "imagination", fc="#EDEDED")
    D.box(ax, 0.66, 0.47, bw, bh, "goal head")
    D.box(ax, 0.86, 0.47, bw, bh, "critic")
    D.box(ax, 0.76, 0.15, 0.18, bh, "policy", lw=3.4)
    D.conn(ax, (0.56, 0.8), (0.665, 0.8))
    D.conn(ax, (0.72, 0.74), (0.67, 0.53))
    D.conn(ax, (0.79, 0.74), (0.85, 0.53))
    D.conn(ax, (0.66, 0.41), (0.72, 0.21))
    D.conn(ax, (0.86, 0.41), (0.8, 0.21))
    ax.plot([0.585, 0.585], [0.05, 0.95], color=P.CHANCE, lw=1.6, ls="--")
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
