"""E7f (v2) -- The optimisation of the latent correction delta h (probing/belief_steering.py): h <- h + delta h,
min over delta h of sum_a KL(target_a || p(z | GRU(h + delta h, z, a))) + lam |delta h|^2 + mu_pos D_pos, weights frozen.
The mu_goal term of the code is left out on purpose (MANIFEST_v2.md). Mathtext only (no LaTeX)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E7f_correction_equations"
SIZE = 28
GREY = "#777777"
TERMS = [
    (r"$\min_{\delta h}\ $", None, None),
    (r"$\sum_{a}\ \mathrm{KL}(\,q_{\mathrm{target}}(z\,|\,a)\ \|\ p(z\,|\,\mathrm{GRU}(h+\delta h,\,z,\,a))\,)$",
     "controllability", P.CORRECTION),
    (r"$\ +\ \lambda\,\|\delta h\|^{2}$", "small change", GREY),
    (r"$\ +\ \mu_{\mathrm{pos}}\,D_{\mathrm{pos}}$", "keep position", GREY),
]


def main():
    fig = C.plt.figure(figsize=C.FULL)
    fig.text(0.5, 0.78, r"$h\ \leftarrow\ h+\delta h$", ha="center", va="center", fontsize=SIZE + 4)
    renderer = fig.canvas.get_renderer()
    size = SIZE
    while True:  # largest size <= SIZE whose equation fits in 92 % of the slide width
        texts = [fig.text(0, 0.5, t, ha="left", va="center", fontsize=size) for t, _, _ in TERMS]
        widths = [t.get_window_extent(renderer).width / fig.bbox.width for t in texts]
        if sum(widths) <= 0.92:
            break
        for t in texts:
            t.remove()
        size -= 1
    x = 0.5 - sum(widths) / 2
    for t, w, (_, label, col) in zip(texts, widths, TERMS):
        t.set_x(x)
        if label:
            fig.text(x + w / 2, 0.36, label, ha="center", va="center", fontsize=P.SIZE_LABEL, color=col)
        x += w
    fig.text(0.5, 0.16, "weights frozen", ha="center", va="center", fontsize=P.SIZE_TICK, color=GREY)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
