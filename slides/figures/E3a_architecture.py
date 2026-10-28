"""E3a -- The agent of E3-E7: an RSSM world model (encoder, GRU, prior, posterior, decoder), a goal head, a critic
trained in imagination, and a one-step look-ahead policy (agents/worldmodel/, IMPLEMENTATION_REPORT E6 section 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _schema as S  # noqa: E402

FIG_ID = "E3a_architecture"
MODEL = "#EFE8DA"


def main():
    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "A world model that imagines the next step for each action")
    ax = S.canvas(fig, (0.02, 0.08, 0.96, 0.78))
    # World model (left block)
    ax.add_patch(C.plt.Rectangle((0.01, 0.06), 0.56, 0.88, fc="none", ec=P.GRID, lw=1.5, ls=(0, (4, 3))))
    ax.text(0.025, 0.91, "world model (RSSM)", fontsize=P.SIZE_ANNOT, color=P.TEXT_2, fontweight="bold")
    S.box(ax, 0.08, 0.2, 0.11, 0.12, "image xₜ", fc=P.ZONE_CORRIDOR, weight="bold")
    S.box(ax, 0.08, 0.5, 0.11, 0.12, "encoder", fc=MODEL)
    S.box(ax, 0.27, 0.78, 0.15, 0.13, "GRU", sub="hₜ = f(hₜ₋₁, zₜ₋₁, aₜ₋₁)", fc=MODEL, weight="bold")
    S.box(ax, 0.27, 0.5, 0.15, 0.13, "posterior", sub="q(zₜ | hₜ, xₜ)", fc=MODEL)
    S.box(ax, 0.46, 0.78, 0.15, 0.13, "prior", sub="p(zₜ | hₜ)", fc=MODEL)
    S.box(ax, 0.46, 0.2, 0.15, 0.13, "decoder", sub="(h, z) → image", fc=MODEL)
    S.conn(ax, (0.08, 0.26), (0.08, 0.44))
    S.conn(ax, (0.135, 0.5), (0.195, 0.5))
    S.conn(ax, (0.27, 0.715), (0.27, 0.565))
    S.conn(ax, (0.345, 0.78), (0.385, 0.78))
    S.conn(ax, (0.345, 0.5), (0.46, 0.265), rad=0.15, label="z", loff=(14, -4))
    S.conn(ax, (0.46, 0.715), (0.33, 0.57), ls=(0, (3, 2)), label="KL", loff=(-14, -6))
    # Heads (right block)
    ax.add_patch(C.plt.Rectangle((0.6, 0.06), 0.39, 0.88, fc="none", ec=P.GRID, lw=1.5, ls=(0, (4, 3))))
    ax.text(0.615, 0.91, "decision", fontsize=P.SIZE_ANNOT, color=P.TEXT_2, fontweight="bold")
    S.box(ax, 0.72, 0.74, 0.18, 0.13, "imagine each action", sub="h′ₐ = GRU(h, z, a), z′ ~ prior", fc=MODEL,
          weight="bold")
    S.box(ax, 0.68, 0.43, 0.12, 0.13, "goal head", sub="p(goal | h′, z′)", fc=MODEL)
    S.box(ax, 0.88, 0.43, 0.12, 0.13, "critic", sub="V(h′, z′)", fc=MODEL)
    S.box(ax, 0.78, 0.15, 0.32, 0.15, "policy: Q(a) = mean of r + γ·c·V", fc=P.BG, ec=P.TEXT,
          sub="over 16 prior draws; ε-greedy, ε = 0.1", lw=2.0, weight="bold")
    S.conn(ax, (0.535, 0.80), (0.63, 0.77), label="h, z", loff=(0, 4))
    S.conn(ax, (0.70, 0.675), (0.68, 0.495))
    S.conn(ax, (0.76, 0.675), (0.87, 0.495))
    S.conn(ax, (0.68, 0.365), (0.73, 0.22))
    S.conn(ax, (0.88, 0.365), (0.83, 0.22))
    C.footnote(fig, "The critic is trained only in imagination, through the prior (15-step λ-returns). r and c come "
                    "from the goal head (r = 50·p − (1 − p), c = 1 − p), γ = 0.9. h: 64 units; z: 8 categorical "
                    "variables × 8 classes. b reads the same prior through the decoder (T2).", y=0.03)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
