# E2 — Résultats du sweep

Produit par `analysis.py`, relançable depuis les artefacts du sweep
(`results/E2_online_4zones_sweep/`), sans jamais relancer l'entraînement
(à l'exception de la courbe régénérée F6c, qui tire des comptages
génératifs frais — jamais de re-training). Faits uniquement — aucune
interprétation scientifique.

Les figures (titres, étiquettes, légendes) et les en-têtes de colonnes
des CSV sont en anglais. Ce rapport reste en français ; glossaire :
couloir=corridor, tapis roulant=conveyor, contrôlable-bruitée=
controllable-noisy, contrôlable-déterministe=controllable-deterministic,
incontrôlable-bruitée=uncontrollable-noisy.

## Paramètres figés

| paramètre | valeur |
|---|---|
| Départs | uniformes sur les 143 états non terminaux (état but exclu) |
| H (troncature) | 36 pas |
| ε | 0,1 (référence), 1,0 (pont E1) |
| α retenu comme référence | 1,0 (balayé sur {0,5 ; 1 ; 5} à ε=0,1) |
| Budget de référence | 10⁶ pas par graine |
| Checkpoints | 16, log-espacés, 1000 à 10⁶ |
| Graines | 20 par condition, carré latin 4 rotations × 5 graines |
| γ | 0,9 |
| Conditions | α∈{0,5 ; 1 ; 5} à ε=0,1 (online + apparié) ; α=1, ε=1 (online + apparié, pont E1) |

## Phase A — statuts

Voir `artifacts/phase_a_status.json` pour le détail complet. Résumé :

| vérification | statut |
|---|---|
| b̂ = 0 exact, couloir + contrôlable-déterministe, online et apparié | observé |
| T̂ online = T̂ apparié, tapis roulant + contrôlable-déterministe (via proxy L1) | observé |
| Δ ≥ 0 partout, après clamp | observé |
| variance non nulle de b̂ entre graines, zones bruitées | observé |
| N_start ≤ N, tout couple, tout checkpoint | observé |
| ε=1 : aucune différence appariée online-apparié (b̂, L1, Δ) | observé |
| tests du jumeau A-D, échantillon (2 graines × 4 conditions, checkpoint final) | observé (8/8) |

Contre-balancement : dispersion inter-rotation ≤ dispersion inter-graine dans toutes les zones à composante stochastique (voir `artifacts/phase_a_status.json`).

**Tout passe : Phase B produite sans interruption.**

## 1 — Cartes F4 : rotation et repère canonique

Vérification du `summary.json` de chaque run de la condition de référence
(a1_e0.1) : les 20 graines couvrent bien les 4 rotations du contre-
balancement (5 graines par rotation, `seed_index // 5 == assignment`,
`env.py::rotation_for_seed_index`) — ce n'est ni une vue à rotation
unique, ni un écart au protocole.

Les cartes F4 étaient déjà construites dans le repère canonique par type
de zone (`analysis_data.py::canonical_rc_for_state`, `fig_f4.py::
_canonical_grid_average`) : la valeur de chaque graine est replacée au
quadrant canonique de son propre type de zone avant la moyenne sur les
20 graines — jamais moyennée par quadrant physique brut. Les quatre
quadrants distincts visibles sur les cartes sont donc l'apparence
ATTENDUE de cette transformation : quatre types de zone différents,
chacun à sa position canonique fixe, et non quatre quadrants physiques
bruts que le contre-balancement aurait dû rendre identiques. Titres et
fichiers mis à jour (`(canonical zone-type frame, N=20 seeds, 4
rotations)`) pour lever toute ambiguïté.

## 2 — F5c reconstruite

Ancienne version : un seul point par (zone, α) au checkpoint final,
scindé en deux classes (`N<10` / `N≥10`) — vide pour `N<10` dans
presque toutes les zones au checkpoint final (voir vérification
« couples encore mal échantillonnés » plus bas).

Nouvelle version (`F5c_b_vs_alpha_by_count_bin`) : b̂ en fonction du
comptage minimal du couple (min sur les actions), binné en
`1-3 / 3-10 / 10-30 / 30-100 / 100-300 / >300`, en agrégeant TOUS les
checkpoints et TOUTES les graines (une courbe par α, un panneau par type
de zone, IC 95 %, effectif annoté par bin). Le couloir et les deux zones
contrôlables restent à b̂≈0 sur toute la plage de comptage ; le tapis
roulant et l'incontrôlable-bruitée convergent vers leur valeur
analytique (−log₂5 et −1,6529) à mesure que le comptage minimal
augmente.

## 3 — F6a par zone

Refait sans moyenne sur les zones : 3 figures (`F6a_epsilon_bridge_
paired_diff_b/_delta/_l1`), un panneau par type de zone chacune. L'écart
b̂ online−apparié à ε=0,1 (négatif, ~−0,01 à −0,04) n'est visible que
dans le panneau contrôlable-bruitée ; il est nul dans le couloir, le
tapis roulant et la contrôlable-déterministe, et plus faible/plus
bruité dans l'incontrôlable-bruitée. À ε=1, l'écart reste centré sur 0
dans toutes les zones, comme prédit. La moyenne inter-zones de l'ancienne
version diluait cet effet localisé.

## 4 — F6c avec incertitude

La courbe régénérée (générative, α=1) utilise maintenant 20 tirages
indépendants par valeur de n (flux RNG indépendants via
`SeedSequence(12345).spawn(20)`), avec bande d'IC 95 % sur la figure. Par
type de zone, écart apparié « point du pont ε=1 − courbe (moyenne des 20
tirages, interpolée en log n) », par graine, avec statistique appariée
(`stats_utils.paired_comparison`, même primitive que partout ailleurs) :

| zone | n (graines) | écart moyen | IC 95 % | p Wilcoxon | d de Cohen |
|---|---|---|---|---|---|
| couloir | 20 | 0,0000 | [0,0000 ; 0,0000] | — (tout à 0) | 0,00 |
| tapis roulant | 20 | +0,0158 | [0,0156 ; 0,0159] | 1,9e-06 | 41,79 |
| contrôlable-bruitée | 20 | +0,0017 | [−0,0010 ; 0,0043] | 0,12 | 0,29 |
| contrôlable-déterministe | 20 | 0,0000 | [0,0000 ; 0,0000] | — (tout à 0) | 0,00 |
| incontrôlable-bruitée | 20 | +0,0045 | [0,0008 ; 0,0082] | 0,024 | 0,56 |

Le pont ε=1 se situe systématiquement légèrement AU-DESSUS de la courbe
régénérée dans le tapis roulant (écart quasi constant entre graines, d
très élevé — la dispersion inter-graine du pont y est très faible) et
l'incontrôlable-bruitée ; l'écart n'est pas significatif dans la
contrôlable-bruitée.

(valeurs exactes : `F6c_e1_bridge_comparison_gap_summary.csv` ; par
graine : `F6c_e1_bridge_comparison_gap.csv` ; courbe : `F6c_e1_bridge_
comparison_curve.csv`.)

## 5 — Tableaux chiffrés (α=1, ε=0,1 sauf mention contraire)

### 5.1 — Écart apparié online−apparié par zone (`table1_paired_diff_by_zone.csv`)

b̂, L1, Δ, aux checkpoints 10⁴/10⁵/10⁶, n=20 comparaisons partout. Extrait
(zones bruitées, b̂) :

| checkpoint | zone | écart moyen | IC 95 % | p Wilcoxon | d |
|---|---|---|---|---|---|
| 10⁴ | contrôlable-bruitée | −0,0146 | [−0,0286 ; −0,0005] | 0,040 | −0,49 |
| 10⁵ | contrôlable-bruitée | −0,0221 | [−0,0370 ; −0,0072] | 0,008 | −0,69 |
| 10⁶ | contrôlable-bruitée | −0,0110 | [−0,0209 ; −0,0010] | 0,064 | −0,52 |
| 10⁴ | incontrôlable-bruitée | −0,0147 | [−0,0240 ; −0,0055] | 0,006 | −0,74 |
| 10⁵ | incontrôlable-bruitée | −0,0008 | [−0,0080 ; 0,0064] | 0,73 | −0,05 |
| 10⁶ | incontrôlable-bruitée | −0,0011 | [−0,0019 ; −0,0003] | 0,009 | −0,66 |

Couloir, tapis roulant et contrôlable-déterministe : écart exactement 0
aux 3 checkpoints (cohérent avec la Phase A). Table complète : b̂, L1 et
Δ, 5 zones × 3 checkpoints = 45 lignes.

### 5.2 — Ventilation contrôlable-bruitée (`table2_controllable_noisy_ventilated.csv`)

Écart L1 online−apparié, restreint à contrôlable-bruitée :

- **Par statut de l'action sous P** : action optimale — écart positif
  significatif aux 3 checkpoints (10⁴ : +0,038, p<0,001 ; 10⁵ : +0,034,
  p<0,001 ; 10⁶ : +0,0046, p=0,002). Action non optimale — écart plus
  faible, significatif seulement à 10⁴ (+0,016, p=0,02), non significatif
  ensuite.
- **Par tranche de comptage** : `N<10` — écart positif à 10⁴ (+0,032,
  p<0,001), non significatif à 10⁵, **0 comparaison à 10⁶** (plus aucun
  couple contrôlable-bruitée sous N=10 à ce stade). `10≤N<100` — écart
  significatif mais de signe instable (−0,011, p=0,048 à 10⁴ ; +0,013,
  p=0,040 à 10⁵ ; non significatif à 10⁶). `N≥100` — écarts petits, non
  significatifs aux 3 checkpoints.

### 5.3 — Tendance budget, loi de puissance, contrôlable-bruitée (`table3_budget_trend_power_law.csv`)

Ajustement OLS log-log de \|écart moyen\| vs budget, checkpoints ≥ 10⁴
(11 points, df=9, t_crit=2,262) :

| quantité | exposant | IC 95 % | r | erreur type |
|---|---|---|---|---|
| b̂ | +0,005 | [−0,148 ; 0,159] | 0,03 | 0,068 |
| L1 | −0,415 | [−0,616 ; −0,215] | −0,84 | 0,091 |

L'écart sur L1 décroît avec le budget (exposant négatif, IC excluant 0,
corrélation forte).

Pour b̂, « plat » et « non détectable » ne sont pas équivalents — les deux
sont vérifiés séparément :

- L'IC de l'exposant b̂ ([−0,148 ; 0,159], demi-largeur 0,153) **exclut**
  la valeur de l'exposant L1 (−0,415) et **exclut** −0,5 (décroissance
  attendue d'une erreur d'échantillonnage en 1/√N). L'ajustement n'est
  donc pas simplement sous-puissant : il rejette activement une
  décroissance de l'ampleur de celle observée sur L1 ou de celle attendue
  par du bruit d'échantillonnage pur.
- Seuil de détection à cet effectif (11 points, test bilatéral α=0,05,
  puissance 80 %) : |exposant| ≳ 0,21. Un exposant plus petit en valeur
  absolue (décroissance plus lente que ≈0,21) ne serait pas fiablement
  distingué de 0 à ce nombre de graines/checkpoints.
- Conclusion correcte : l'écart sur b̂ n'a **pas** de décroissance de
  l'ampleur de celle sur L1, ni de l'ampleur attendue par du seul bruit
  d'échantillonnage — mais une décroissance plus faible (|exposant|
  < ~0,21) ne peut ni être confirmée ni exclue avec ces données.

Écart brut sur b̂ (contrôlable-bruitée, avec IC 95 %, n=20 comparaisons) :

| checkpoint | écart moyen | IC 95 % |
|---|---|---|
| 10⁵ | −0,0221 | [−0,0370 ; −0,0072] |
| 10⁶ | −0,0110 | [−0,0209 ; −0,0010] |

L'écart reste non nul (IC excluant 0) aux deux checkpoints ; le point
central diminue de moitié entre 10⁵ et 10⁶, mais les IC se chevauchent
largement (pas de diminution significative point à point entre ces deux
checkpoints seuls).

### 5.4 — Δ extérieur, majorant (`table4_delta_exterior_max.csv`)

Maximum, sur les états extérieurs (couloir), de la moyenne de graine de
Δ, à α=1 ε=0,1, checkpoint final :

| condition | état (argmax) | Δ moyen max | IC 95 % |
|---|---|---|---|
| online | 1 | 0,0000 | [0,0000 ; 0,0000] |
| apparié | 1 | 0,0000 | [0,0000 ; 0,0000] |

Exactement 0 aux deux conditions à α=1 (cohérent avec la Phase A et
`b_exact_zero`/`delta_nonneg`). À α=5, l'écart online−apparié moyen sur
le couloir entier est non nul (`synthesis_table_delta_exterior.csv` :
+0,059 online, +0,058 apparié, IC excluant 0) — déjà noté au Temps
précédent, confirmé inchangé ici.

### 5.5 — b̂ au checkpoint final vs valeur analytique (`table5_b_final_checkpoint_vs_analytic.csv`)

Étend `synthesis_table_b.csv` avec l'écart à la valeur analytique, par
zone × α, online et apparié :

| zone | α | b̂ online | b̂ apparié | analytique | écart online | écart apparié |
|---|---|---|---|---|---|---|
| tapis roulant | 0,5 | −2,276 | −2,276 | −2,322 | 0,046 | 0,046 |
| tapis roulant | 1,0 | −2,237 | −2,237 | −2,322 | 0,084 | 0,084 |
| tapis roulant | 5,0 | −2,024 | −2,024 | −2,322 | 0,298 | 0,298 |
| incontrôlable-bruitée | 0,5 | −1,640 | −1,637 | −1,653 | 0,013 | 0,016 |
| incontrôlable-bruitée | 1,0 | −1,620 | −1,619 | −1,653 | 0,033 | 0,034 |
| incontrôlable-bruitée | 5,0 | −1,490 | −1,488 | −1,653 | 0,163 | 0,165 |

Couloir et zones contrôlables : b̂=0 exact, analytique=0 (couloir : pas
de valeur analytique définie, colonnes vides). L'écart à l'analytique
croît avec α pour les deux zones incontrôlables (prior plus fort → sous-
estimation plus lente de I(a;s'|s) au même budget).

### 5.6 — Contraste apparié et budget à 90 % de l'analytique (`table6_contrast_budget_to_90pct.csv`)

| contraste | α | contraste analytique | contraste checkpoint final | budget à 90 % |
|---|---|---|---|---|
| contrôlable-dét. − tapis roulant | 0,5 | 2,322 | 2,276 | 158 489 |
| contrôlable-dét. − tapis roulant | 1,0 | 2,322 | 2,237 | 398 107 |
| contrôlable-dét. − tapis roulant | 5,0 | 2,322 | 2,024 | non atteint |
| contrôlable-bruitée − incontrôlable-bruitée | 0,5 | 1,653 | 1,636 | 158 489 |
| contrôlable-bruitée − incontrôlable-bruitée | 1,0 | 1,653 | 1,601 | 251 189 |
| contrôlable-bruitée − incontrôlable-bruitée | 5,0 | 1,653 | 1,487 | non atteint |

À α=5, le seuil de 90 % n'est atteint pour aucun des deux contrastes
dans la plage de budget balayée (≤10⁶).

## Vérifications

### Anisotropie de la visite : géométrie vs boucle

Trois sources, comparées en PARTS relatives (chaque branche divisée par
le total des 4 branches — les échelles absolues diffèrent, la part est
comparable) :

- **Référence ε=0,1** (`verif_corridor_branch_visitation.csv`) : N(s)
  sommé par branche, moyenne sur les 20 graines, checkpoint final —
  observé (politique + exploration).
- **Pont ε=1** (`verif_corridor_branch_visitation_epsilon1_bridge.csv`) :
  même calcul, condition ε=1 (politique purement aléatoire, aucun rôle du
  but) — la dynamique seule, sans boucle de décision.
- **DP optimal sous P** (`verif_corridor_branch_dp_optimal.csv`) : pour
  chaque rotation, valeur/politique optimale exacte par itération de la
  valeur sur le noyau vrai P, puis nombre de visites attendu par état
  via la matrice fondamentale de la chaîne absorbante induite
  ((I−Q)⁻¹, Q restreinte aux 143 états non-but), moyenné uniformément
  sur les 143 états de départ non-terminaux, puis moyenné sur les 4
  rotations. Aucune collecte, aucun RNG — calcul exact.

| branche | part DP optimal | part pont ε=1 | part référence ε=0,1 |
|---|---|---|---|
| ligne 0 | 0,224 | 0,056 | 0,200 |
| ligne 6 | 0,226 | 0,635 | 0,256 |
| colonne 0 | 0,505 | 0,147 | 0,504 |
| colonne 6 | 0,045 | 0,163 | 0,040 |

(table complète : `verif_corridor_branch_comparison.csv`)

**Colonne 0 / colonne 6 : géométrique, via la politique optimale.** La
part DP (0,505 / 0,045) reproduit quasiment exactement la part observée
à ε=0,1 (0,504 / 0,040 — écart ≤0,005). Le pont ε=1 (dynamique seule,
sans but) ne reproduit PAS ce motif (0,147 / 0,163 — colonne 0 n'y est
pas dominante). L'asymétrie colonne 0 ≫ colonne 6 n'est donc pas un
effet de la dynamique brute : c'est la signature de la politique
optimale sous P (la colonne 0, qui contient le but (0,0), est
l'artère de corridor la plus empruntée par tout chemin optimal,
quelle que soit la rotation) — géométrique au sens où elle est
entièrement déterminée par (P, but), sans dépendre de la politique
apprise réelle au-delà de son optimalité.

**Ligne 0 / ligne 6 : dynamique, via la dérive.** Sous le pont ε=1
(aucune politique), l'asymétrie ligne 6 ≫ ligne 0 est encore plus
marquée (0,635 vs 0,056) que sous DP (0,226 vs 0,224, quasi symétrique)
ou sous la référence observée (0,256 vs 0,200) — cohérent avec la
dérive résolue (`helplessness.envs.zones.resolve_drift`, direction
`away_from_goal`, but en (0,0)), rotation-invariante puisque dérivée de
la plage de lignes du quadrant géométrique, jamais du type qui l'occupe :

| bande | plage de lignes | dérive (ligne, colonne) |
|---|---|---|
| nw/ne | 1–5 | (+1, 0) |
| sw/se | 7–11 | (−1, 0) |

Les deux bandes dérivent vers la ligne 6 : sous une politique purement
aléatoire, cette dérive domine et pousse la quasi-totalité du trafic
extérieur vers la ligne 6. Sous la politique optimale (DP), le but
étant en ligne 0, la dérive est presque entièrement compensée (parts
quasi égales, 0,226 vs 0,224). La référence observée (0,256 vs 0,200)
se situe entre les deux extrêmes DP et pont, cohérent avec ε=0,1 :
90 % des pas suivent la politique apprise (proche de DP), 10 %
explorent au hasard (tirant vers le motif du pont).

**Ce qui reste inexpliqué.** Sous l'hypothèse d'un mélange {DP, pont}
(la référence ε=0,1 devrait s'interpoler entre les deux extrêmes), 3
des 4 branches vérifient bien part(DP) ≤ part(référence) ≤ part(pont)
ou l'inverse selon le sens de l'écart — sauf colonne 6 : part observée
(0,040) légèrement SOUS la part DP (0,045), donc hors de l'intervalle
[DP, pont]=[0,045 ; 0,163]. L'écart est faible (0,005) et colonne 6 est
la branche avec le N total le plus petit (24 340, la plus sensible au
bruit d'échantillonnage) — probablement du bruit, mais signalé tel
quel : ce point précis n'est pas strictement couvert par le modèle à
deux mécanismes (DP + dynamique pure).

### Couples encore mal échantillonnés (`verif_undersampled_pairs_final_checkpoint.csv`)

Checkpoint final, α=1, ε=0,1, online (identique par construction pour
apparié, `matched_control.py::collect_matched` redessine exactement le
même N par couple) — nombre de couples (état, action) avec N<10 et
N<100, moyenné sur les 20 graines :

| zone | couples totaux | N<10 (moy.) | N<100 (moy.) |
|---|---|---|---|
| couloir | 215 | 0,35 | 62,1 |
| tapis roulant | 125 | 0,00 | 57,35 |
| contrôlable-bruitée | 125 | 0,00 | 79,1 |
| contrôlable-déterministe | 125 | 0,00 | 49,75 |
| incontrôlable-bruitée | 125 | 0,00 | 49,05 |

Au checkpoint final, quasiment plus aucun couple sous N=10 (0 à 0,35 en
moyenne selon la zone) ; entre 29 % (couloir) et 63 % (contrôlable-
bruitée) des couples restent sous N=100 selon la zone. C'est ce
déséquilibre (comptages faibles concentrés aux checkpoints précoces) qui
motivait la reconstruction de F5c en agrégeant tous les checkpoints.

## Figures et tableaux

Tous les PNG/PDF/CSV sont sous `artifacts/figures/`, répartis dans les
sous-dossiers `png/`, `pdf/`, `csv/` (même nom de fichier dans chacun).

- **F1** (`F1_b_vs_budget_alpha{0.5,1.0,5.0}`) — b̂ par type de zone vs budget, online/apparié, une figure par α.
- **F2** (`F2_null_model_paired_diff`) — écart apparié online−apparié sur b̂/Δ/L1 par zone, α=1 ε=0,1. Figure centrale.
- **F3** (`F3a_l1_diff_by_zone_and_optimal_status`, `F3b_l1_diff_by_count_bin`) — localisation de l'écart L1 par statut d'action et par tranche de comptage.
- **F4** (`F4_N_total`, `F4_N_start_share`, `F4_N_min_per_state`, `F4_b_online`, `F4_delta_online`, `F4_b_diff`, `F4_delta_diff`, `F4_delta_diff_exterior`) — cartes spatiales, checkpoint final, repère canonique par type de zone (N=20 graines, 4 rotations — voir section 1).
- **F5** (`F5a_alpha_contrast_vs_budget`, `F5b_alpha_contrast_final_checkpoint_table`, `F5c_b_vs_alpha_by_count_bin`) — effet du prior α ; F5c reconstruite (section 2).
- **F6** (`F6a_epsilon_bridge_paired_diff_{b,delta,l1}`, `F6b_visitation_comparison`, `F6c_e1_bridge_comparison`) — pont ε=1 par type de zone (section 3), comparaison à une courbe générative régénérée à α=1 avec IC sur 20 tirages (section 4 ; jamais aux artefacts α=0 d'E1).
- **F7** (`F7a_counts_profile`, `F7b_departure_share`, `F7c_episode_profile`) — profil de visite de référence (le dernier réutilise `episode_analysis.py` sans recalcul).
- **Tableau de synthèse** (`synthesis_table_b.csv`, `synthesis_table_delta_exterior.csv`) — checkpoint final, par zone × α : b̂ online, b̂ apparié, écart apparié + IC 95 % + taille d'effet + p Wilcoxon ; Δ moyen extérieur (couloir) par α et condition.
- **Tableaux chiffrés** (`table1` à `table6`) — section 5 ci-dessus.
- **Vérifications** (`verif_corridor_branch_visitation`, `verif_corridor_branch_visitation_epsilon1_bridge`, `verif_corridor_branch_dp_optimal`, `verif_corridor_branch_comparison`, `verif_resolved_drift`, `verif_undersampled_pairs_final_checkpoint`) — section « Vérifications » ci-dessus.

## Formes prédites — observé / non observé

| prédiction | statut |
|---|---|
| b̂ = 0 exact, couloir et contrôlable-déterministe | observé |
| online et apparié identiques en distribution, deux zones déterministes | observé |
| écart adaptatif concentré sur l'action optimale sous P, zone contrôlable-bruitée | observé (table2, section 5.2) |
| Δ extérieur ≈ 0, pas de différence online/apparié | observé à α=0,5/1 (table4) ; non nul à α=5 (synthesis_table_delta_exterior) |
| aucun écart online−apparié à ε=1 | observé |
| écart online−apparié à ε=0,1 localisé à la zone contrôlable-bruitée | observé (F6a par zone, section 3) |
