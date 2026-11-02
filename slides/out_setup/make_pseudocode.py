"""Renders the training pseudo-code (values of configs/e4/e4.yaml) -> training_pseudocode.png/.svg.
Usage: python slides/out_setup/make_pseudocode.py"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LINES = [
    "# 1. collecte (1 pas d'environnement)",
    "o, h, z  <-  posterior filtré :  h = GRU(h, z, a),  z ~ q(z | h, o)",
    "pour chaque action a :   h'_a = GRU(h, z, a),   z'_k ~ p(z | h'_a),   k = 1..16",
    "    Q(a) = moyenne_k [ r̂(h'_a, z'_k) + γ · c · V(h'_a, z'_k) ]      γ = 0.9",
    "a  <-  uniforme avec proba ε = 0.1,  sinon argmax Q (égalités tirées au hasard)",
    "o', r  <-  env.step(a) ;   replay.ajouter(o, a, r, o')",
    "",
    "# 2. world model (RSSM) : 0.6 mise à jour par pas, lots de 16 séquences × 64",
    "pour chaque pas t de la séquence :",
    "    h_t = GRU(h_t-1, z_t-1, a_t-1)",
    "    q(z_t | h_t, encodeur(o_t)) ;   p(z_t | h_t) ;   z_t ~ q (8 × 8 catégories)",
    "    ô_t = décodeur(h_t, z_t)",
    "L = Σ_pixels w · (o − ô)²          w = 1 + 9·masque(patch de l'agent)",
    "  + 0.5 · KL( sg[q] ‖ p )  +  0.1 · KL( q ‖ sg[p] )",
    "Adam(L),  lr 1e-3   [ z <- probabilités tant que les latents ne sont pas séparés,",
    "                       puis straight-through, lr 1e-4 ]",
    "",
    "# 3. tête « but » et critic, une fois les latents séparés",
    "tête_but( sg[h], sg[z] ) → p(but)  ;  r̂ = 50·p + (−1)·(1 − p) ;  c = 1 − p",
    "départs :  tous les (h, z) postérieurs du lot, sg",
    "pour k = 0..14 :  a_k  <-  ε-greedy sur Q (comme en 1, K = 4) ;  (h, z)_k+1 via le prior",
    "R_15 = V_lent(h_15, z_15)",
    "R_k  = r̂_k+1 + γ · c_k+1 · [ (1 − λ) · V_lent_k+1 + λ · R_k+1 ]      γ = 0.9,  λ = 0.95",
    "L_V = moyenne  w_k · ( V(h_k, z_k) − sg[R_k] )²        w_k = Π_j≤k c_j",
    "Adam(L_V),  lr 3e-5 ;   V_lent  <-  EMA(V, 0.02)",
]
KW = r"\b(pour|chaque|sinon|avec|proba|tant|que|puis|départs|tous)\b"
KW2 = r"(GRU|argmax|uniforme|env\.step|replay\.ajouter|Adam|décodeur|encodeur|tête_but|EMA|KL|sg|moyenne)"
FS, W, LH = 15, 0.602, 0.5  # font size (pt), advance in em, line height (em-ish)

def colour(tok_line):
    out = [(c, "#222222") for c in tok_line]
    def paint(rx, col):
        for m in re.finditer(rx, tok_line):
            for i in range(m.start(), m.end()):
                out[i] = (tok_line[i], col)
    paint(KW2, "#1f5fa8"); paint(KW, "#a8321f")
    paint(r"\b(Q|V|V_lent|r̂|p|q|h|z|a)\b", "#222222")
    paint(r"[−+·Σ←<\-]{0}", "#222222")
    paint(r"\b\d+(\.\d+)?(e-?\d+)?\b", "#2a7a3b")
    return out

ncol = max(len(l) for l in LINES)
fig_w = ncol * W * FS / 72 + 0.4
fig_h = len(LINES) * FS * 1.45 / 72 + 0.3
fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
cw = W * FS / 72
for i, line in enumerate(LINES):
    y = fig_h - 0.15 - (i + 0.8) * FS * 1.45 / 72
    if line.startswith("#"):
        ax.text(0.2, y, line, family="DejaVu Sans Mono", size=FS, color="#777777", weight="bold", va="baseline"); continue
    cells = []  # (column, text, colour); a combining mark stays with its base character
    for ch, col in colour(line):
        if ch == "̂" and cells:
            cells[-1] = (cells[-1][0], cells[-1][1] + ch, cells[-1][2]); continue
        cells.append((len(cells), ch, col))
    for j, ch, col in cells:
        if ch != " ":
            ax.text(0.2 + j * cw, y, ch, family="DejaVu Sans Mono", size=FS, color=col, va="baseline")
fig.savefig(HERE / "training_pseudocode.png", dpi=200, facecolor="white")
fig.savefig(HERE / "training_pseudocode.svg", facecolor="white")
