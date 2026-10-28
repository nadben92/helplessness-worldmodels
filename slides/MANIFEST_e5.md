# Figures E5 des slides — manifeste (seule source pour présenter les figures)

Les figures ne portent aucun texte autre que les noms d'axes, les graduations et les noms de catégories. Tout le reste
(message, lecture des couleurs, valeurs, sources) est ici. Résultats **PRÉLIMINAIRES** (docs/e5/DECISIONS.md (36)).

- **Régénérer** : `.venv/Scripts/python.exe experiments/E5_belief_training/analysis_450k/analyse_450k.py <5 dossiers de
  sondes> --out experiments/E5_belief_training/analysis_450k/tables`, puis
  `.venv/Scripts/python.exe slides/figures_e5/make_all_e5.py`. Sorties : `out_e5/png/` (300 dpi, transparent),
  `out_e5/svg/` (transparent), `out_e5/png_white/` (300 dpi, fond blanc).
- **Style** : charte v2 de la branche `slides` (`style_v2/palette.py`, `common.py`) reproduite dans
  `figures_e5/style_e5.py` : `plt.style.use('default')`, noms d'axes 18 pt, graduations 15 pt, traits 2,8–3,4 pt,
  16:9 (13,33 × 7,5 in).
- **Couleurs** : baseline exposé vermillon `#E8412C` ; **shuffle exposé violet-magenta `#A21CAF`** (absent de la
  palette v2 ; distinct du violet de la correction d'E7 `#6718AB` ; deutéranopie, ΔE76 ≥ 24,6 contre toutes les
  couleurs qu'il côtoie, contrôle de `style/common.py`) ; positions : ancienne UD `#FF5A36`, ancienne CD `#4DA3FF`,
  neutre `#6E6E6E` (couleurs de lignes des zones). Lignes de référence en tirets sans texte : noir = −log2 5 ≈ −2,32
  (théorie, tapis roulant), gris = 0.
- **Données** : sondes `e5_probes_b` (shuffle f = 1, graines E = 402 et F = 502, **475 000 pas**, une seule référence
  partagée, la f = 1 de la graine B) et `e5_probes_baselines` (baselines A = 2 et B = 102 à 300 000). La « baseline »
  des figures = A et B à 300 000 ; la baseline B au niveau apparié (63 096) n'est que dans les tableaux. b au pas 1 =
  exposé − référence, sondes valides seulement (critère d'E4). Rapport : `experiments/E5_belief_training/analysis_450k/REPORT.md`.

## E5a — position contre thème (figure principale)
- **Tracé** : deux catégories, `position only` (couleur du couloir à l'ancienne UD) et `theme only` (T_inc sur le
  quadrant neutre nord-est) ; une barre par condition (vermillon baseline, violet shuffle) = moyenne des deux graines,
  points = graines.
- **Valeurs** : position seule : baseline −2,30 (A), −2,27 (B) ; shuffle −0,38 (E), −0,92 (F), moyenne −0,65. Thème
  seul : baseline −0,02, −0,00 ; shuffle −0,91, −0,92.
- **Message** : quand la position cesse d'être prédictive, la croyance ne reste plus sur la position seule et devient
  en partie sensible au thème — mais sans se reporter entièrement sur lui.
- **À dire** : une sonde ne change qu'un quadrant. « Thème seul » montre donc deux quadrants T_inc, « position seule »
  aucun ; le shuffle n'a jamais vu que des images à un seul T_inc. −0,9 sur l'un des deux T_inc est compatible avec une
  croyance partagée entre eux.
- **Source** : `tables/extremes.csv`. **Oral** : « En shuffle, la position seule ne suffit plus et la couleur seule
  compte un peu : la croyance n'est plus attachée à un seul indice. »

## E5b — gradient de thème
- **Tracé** : panneau gauche = baseline (cadre vermillon), panneau droit = shuffle (cadre violet), même ordonnée.
  Abscisse = distance RVB de la couleur sondée à T_inc, divisée par D = |T_inc − T_ctrl|. Une couleur par position
  sondée (ancienne UD rouge-orangé, ancienne CD bleu clair, neutre gris). Points pâles = sondes valides (graines
  réunies, image partagée comptée une fois) ; traits = moyennes par classe de 0,1 D centrée sur les multiples de
  0,1 D (≥ 2 sondes).
- **Valeurs** : baseline, ancienne UD −2,31 de 0 à 0,4 D, −1,80 à −2,05 de 0,7 à 0,9 D ; neutre ≈ 0 jusqu'à 0,5 D.
  Shuffle, ancienne UD −2,01 (0) → −1,44 (0,4) → −0,72 (0,7) ; neutre −0,91 (0) → −0,49 (0,4) → −0,17 (1,0) ; ancienne
  CD valide seulement à partir de 0,4 D (−0,46 puis ≈ −0,15).
- **Message** : en baseline, seule la position compte ; en shuffle, b dépend de la couleur aux deux positions mesurables
  près de T_inc, mais l'ancienne UD reste environ 1 bit plus basse que le neutre.
- **Source** : `tables/gradient_probes.csv`, `gradient_binned.csv`. **Oral** : « À gauche la croyance suit la position,
  à droite elle suit la couleur — et encore un peu la position. »

## E5c — distribution des croyances
- **Tracé** : b au pas 1 de toutes les sondes valides (image partagée comptée une fois), une colonne par condition,
  dispersion horizontale aléatoire ; ronds = graine A (baseline) / E (shuffle), carrés = graine B / F.
- **Valeurs** : sondes valides 45 + 40 (baseline), 50 + 48 (shuffle). Classes (≤ −1,5 / ]−1,5 ; −0,5] / > −0,5) :
  baseline 15/6/24 (A), 15/0/25 (B) ; shuffle 8/11/31 (E), 6/11/31 (F).
- **Message** : la baseline est bimodale (≈ −2,3 ou ≈ 0) ; le shuffle produit des croyances intermédiaires (utile pour
  E6), et ses deux pôles sont moins extrêmes (≈ −2,1 et −0,1 à −0,3).
- **Source** : `tables/gradient_probes.csv`, `distribution.csv`. **Oral** : « Le shuffle remplit l'entre-deux. »
  Rappel : b sépare bien des classes, ses petites variations à l'intérieur d'une classe ne se lisent pas.

## E5d — récupération dans l'épisode (secondaire)
- **Tracé** : b par pas dans l'ancienne UD, image d'entraînement, poids gelés ; traits fins = graines, trait épais =
  moyenne ; vermillon baseline, violet shuffle.
- **Valeurs** : pas 1 −2,31 (baseline), −2,01 (shuffle) ; gain pas 6–10 moins pas 1 : +0,43, +0,37 (A, B), +0,50,
  +0,36 (E, F) ; plateau ≈ −1,9 et ≈ −1,55.
- **Message** : même récupération rapide puis plateau, à un niveau moins extrême en shuffle.
- **Source** : `tables/recovery.csv`. **Oral** : « Dans l'épisode, les deux conditions récupèrent autant, puis
  plafonnent. »
