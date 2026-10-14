"""E3: switch step of the probabilities phase across colour draws and across replicates at fixed colours
(autoencoder isolation test, lr 1e-3, k = 6, deadline 8,000 steps). Left: the 10 colour draws. Right: draws 0 and 1
with their 3 replicates (same colours, other latent draws and minibatches); a run that did not separate before the
deadline is drawn at the deadline with an open marker. Writes paper/figures/out/fig_e3_switch_themes.pdf.

Source: results/E3_worldmodel_4zones/theme_sensitivity/{draw_NN,replicate_dNN_rN}.json
(experiments/E3_worldmodel_4zones/theme_sensitivity.py; docs/e3/THEME_SENSITIVITY.md). Values plotted as recorded.
"""

from __future__ import annotations

import common
import e3_sources as src

NAME = "fig_e3_switch_themes"
ROOT = src.E3_RESULTS / "theme_sensitivity"
N_DRAWS = 10
REPLICATED_DRAWS = (0, 1)
N_REPLICATES = 3
DEADLINE = 8000  # the characterisation's deadline (docs/e3/THEME_SENSITIVITY.md, design)


def switch_of(run: dict) -> tuple[float, bool]:
    """(switch step, censored); a run that did not separate is censored at the deadline."""
    if run["status"] == "ok":
        return float(run["switch_step"]), False
    if run["status"] == "separation_not_reached":
        return float(DEADLINE), True
    raise SystemExit(f"Unexpected status {run['status']!r} in a theme-sensitivity record")


def main() -> None:
    draws = [src.read_json(ROOT / f"draw_{d:02d}.json") for d in range(N_DRAWS)]
    common.apply_style()
    fig, (ax_d, ax_r) = common.figure(aspect=0.42, ncols=2, sharey=True, gridspec_kw={"width_ratios": [2, 1]})
    color = common.TOL_BRIGHT["blue"]
    for d, run in enumerate(draws):
        y, censored = switch_of(run)
        ax_d.plot(d, y, "o", ms=4, color=color, mfc="none" if censored else color)
    ax_d.set_xticks(range(N_DRAWS))
    ax_d.set_xlabel("colour draw")
    ax_d.set_ylabel("switch step")
    for i, d in enumerate(REPLICATED_DRAWS):
        y0, c0 = switch_of(draws[d])
        ax_r.plot(i, y0, "o", ms=4, color=color, mfc="none" if c0 else color)
        for r in range(1, N_REPLICATES + 1):
            y, censored = switch_of(src.read_json(ROOT / f"replicate_d{d:02d}_r{r}.json"))
            ax_r.plot(i + 0.12 * r, y, "s", ms=3.5, color=common.TOL_BRIGHT["red"], mfc="none" if censored else common.TOL_BRIGHT["red"])
    ax_r.set_xticks(range(len(REPLICATED_DRAWS)), [f"draw {d}" for d in REPLICATED_DRAWS])
    ax_r.set_xlim(-0.4, len(REPLICATED_DRAWS) - 0.2)
    ax_r.set_xlabel("replicates at fixed colours")
    for ax in (ax_d, ax_r):
        ax.axhline(DEADLINE, color="gray", lw=0.6, ls="--")
    fig.tight_layout(pad=0.2)
    path = common.save_pdf(fig, NAME)
    print(f"wrote {path.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
