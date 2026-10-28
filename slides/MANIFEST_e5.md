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

## Figures pour la présentation (`figures_e5/make_e5_v2.py`, n'écrit pas E5a–E5d)

Message des deux figures : **avant (baseline), la croyance tenait à la position seule ; en shuffle, le thème joue.**
Résultat intermédiaire ; les limites ci-dessous ne sont pas sur les figures.

### E5a_v2 — position contre thème, deux panneaux
- **Tracé** : panneau gauche baseline (vermillon), droit shuffle (violet), nom de la condition sous chaque panneau ;
  barres `position only` et `theme only` = moyenne des deux graines, points = graines ; ordonnée −2,4 à 0,2 ; tirets
  gris = 0. Mêmes valeurs qu'E5a (`tables/extremes.csv`) : baseline −2,30 / −2,27 et −0,02 / −0,00 ; shuffle −0,38 /
  −0,92 (moyenne −0,65) et −0,91 / −0,92 (moyenne −0,91).
- **Oral** : « Avant, seule la position portait la croyance ; en shuffle, la couleur seule suffit à la faire baisser. »

### E5e — le thème seul (quadrant neutre)
- **Tracé** : sondes valides au quadrant neutre (nord-est) seulement ; abscisse `theme distance` = distance RVB de la
  couleur sondée à T_inc, divisée par D ; par graine, moyenne de ses sondes par classe de 0,1 D (trait fin) ;
  moyenne des graines (trait épais) sur les classes où les deux graines ont au moins une sonde valide (une graine peut
  n'avoir qu'une sonde dans une classe). Tirets gris = 0. Source : `tables/gradient_binned.csv` (unit = label).
- **Valeurs (moyenne des graines)** : shuffle −0,91 (0) → −0,58 (0,2) → −0,48 (0,4) → −0,26 à −0,17 (0,5 à 1,0) ;
  baseline entre −0,02 et 0,00 de 0 à 0,5 D.
- **Écart à la forme demandée (« baseline plate à 0 »)** : la baseline n'est plate que jusqu'à 0,5 D ; elle descend à
  −0,09 (0,6), −0,30 (0,7), −0,43 (0,8), −0,45 (0,9), −0,21 (1,0). Ces points viennent des sondes des arcs (couleurs
  hors du segment T_inc–T_ctrl, loin des deux thèmes d'entraînement), surtout de la graine A (−0,41 à −0,68 sur l'arc
  centré en T_ctrl, jusqu'à −0,55 sur l'arc centré en T_inc ; graine B −0,18 à −0,26 et ≈ 0). Déjà vu dans E4 (thème
  seul −0,29 pour une configuration). Lecture : une couleur jamais vue sur un quadrant neutre fait un peu baisser b chez
  la baseline ; près de T_inc, rien. L'axe n'a pas été tronqué.
- **Oral** : « Le même quadrant neutre, peint de plus en plus près de la couleur du tapis : rien chez la baseline, une
  croyance qui baisse chez le shuffle. »

### Limites (à dire, pas sur la figure)
- « Theme only » montre deux quadrants T_inc (la sonde ne change qu'un quadrant, l'ancienne UD garde T_inc) et
  « position only » aucun ; le shuffle n'a vu que des images à un seul T_inc. La même chose vaut pour E5e près de
  T_inc (deux quadrants T_inc).
- Convergence inachevée : 475 000 pas, critère d'E4 non atteint (DECISIONS (36)) ; résultats PRÉLIMINAIRES.
- Deux graines shuffle (E, F) partageant une référence ; shuffle A non sondé.

### E5e_loc — le thème seul, sondes bien localisées (critère objectif, même pour les deux conditions)
- **Critère** (`experiments/E5_belief_training/analysis_450k/LOCALISATION_CRITERION.md`, commité avant le calcul) : une
  sonde est retenue si elle est valide (critère d'E4) **et** si la localisation au pas 1 de l'exposé est ≥ 0,90 (règle
  directe d'E4 : case où le décodeur dessine l'agent contre vraie case ; mêmes marches que la sonde, 150). Localisation
  recalculée en lecture seule (`localisation_probes.py`, poids gelés, hash inchangés), tables
  `tables/localisation_*.csv`, `gradient_binned_loc.csv`, `loc_removed.csv`.
- **Sondes retirées par la localisation** (toutes positions ; valides → retenues) : baseline A 45 → 33, baseline B 40 → 32,
  shuffle E 50 → 43, shuffle F 48 → 47. Au quadrant neutre : baseline A, 0,7 / 0,8 / 0,9 D (1 chacune, loc. 0,36 / 0,00 /
  0,00) et 6 sur 8 à 1,0 D (loc. 0,00–0,82) ; baseline B, 0,6 à 0,9 D (1 chacune, loc. 0,69–0,89) et 1 sur 8 à 1,0 D ;
  shuffle E, 1 à 0,4, 0,6 et 0,7 D (loc. 0,00–0,23) ; shuffle F, aucune.
- **Résultat : la descente de la baseline disparaît.** Baseline au quadrant neutre (moyenne des graines) : −0,01 à 0,00
  de 0 à 0,5 D, −0,02 à 0,6 D (graine A seule), −0,00 à 1,0 D (9 sondes) ; plus aucune sonde retenue entre 0,7 et 0,9 D
  pour les deux graines, donc le trait épais relie 0,5 et 1,0 D (les traits fins sont les graines). Les sondes qui
  descendaient (−0,41 à −0,68) étaient celles où l'exposé ne localisait pas l'agent. Shuffle inchangé dans sa forme :
  −0,91 (0) → −0,58 (0,2) → −0,35 (0,4) → −0,16 à −0,23 (0,6 à 1,0 D).
- **E5a** : les quatre sondes extrêmes (position seule, thème seul) sont toutes retenues (localisation 0,96–1,00) : pas
  d'E5a_v3, E5a_v2 reste valable.
- **Oral** : « On ne garde que les sondes où le modèle sait encore où est l'agent : la baseline est plate partout, seul
  le shuffle réagit à la couleur du tapis. »
