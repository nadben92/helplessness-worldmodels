"""E7e -- Three readers of the same predicted z': the decoder and templates (b), the critic (the action), the KL (the
optimisation of h by the latent correction). Schematic (IMPLEMENTATION_REPORT sections 17-20)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _schema as S  # noqa: E402

FIG_ID = "E7e_three_readers"
MODEL = "#EFE8DA"


def main():
    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "Three readers of the same prediction can disagree",
            "What b measures is not necessarily what the policy uses")
    ax = S.canvas(fig, (0.02, 0.08, 0.96, 0.78))
    S.box(ax, 0.1, 0.5, 0.14, 0.16, "h′ₐ = GRU(h, z, a)", sub="one per action", fc=MODEL, weight="bold")
    S.box(ax, 0.32, 0.5, 0.14, 0.16, "predicted z′", sub="16 draws of the prior", fc=P.BG, ec=P.TEXT, lw=2.2,
          weight="bold")
    S.conn(ax, (0.17, 0.5), (0.25, 0.5))
    rows = [(0.84, "1", "decoder + nearest template", "→ predicted cell → I(a; s′|s) → b", "the measure",
             P.REFERENCE),
            (0.5, "2", "critic V(h′ₐ, z′) + goal head", "→ Q(a) → action", "what drives behaviour", P.EXPOSED),
            (0.16, "3", "KL(target ‖ p(z′ | h′ₐ))", "→ gradient on h (latent correction A)", "what the correction "
             "optimises", P.CORRECTION)]
    for y, n, name, out, role, col in rows:
        S.box(ax, 0.6, y, 0.26, 0.17, name, sub=out, fc=MODEL, ec=col, lw=2.2, weight="bold")
        S.conn(ax, (0.39, 0.5), (0.47, y), color=col, lw=2.2)
        ax.text(0.47, y + 0.105, n, fontsize=P.SIZE_AXIS, fontweight="bold", color=col, ha="center")
        ax.text(0.75, y, role, fontsize=P.SIZE_ANNOT + 1, color=col, va="center", fontweight="bold")
    ax.text(0.75, 0.33, "Correction A raises 3 and 1\n(b +1.2 to +1.9 bit), but 2 stays\nat chance for 3 of 4 "
            "agents (D7)", fontsize=P.SIZE_SMALL + 0.5, color=P.TEXT, va="center")
    ax.text(0.75, 0.67, "Induced references: 1 says the\naction fails in about half the\nsteps; 2 still steers "
            "(D5, D6)", fontsize=P.SIZE_SMALL + 0.5, color=P.TEXT, va="center")
    C.footnote(fig, "All three read the same prior p(z′ | h′ₐ); they differ in what they extract from it. Findings "
                    "on E4's four agents (CPU, frozen weights): experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md "
                    "§17–20.", y=0.03)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
