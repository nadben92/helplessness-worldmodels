# Figures de la présentation d'avancement — manifeste

Régénérer : `.venv/Scripts/python.exe slides/make_all.py` (22 figures, ~1 min). Sorties : `out/png/` (300 dpi, fond
`#F7F3EA`), `out/svg/`, `out/png_transparent/`. Style : `style/palette.py` (seule source des couleurs),
`style/slides.mplstyle`, `style/common.py`. Contrôle daltonisme : `style/check_palette.py`.

- **Police** : DejaVu Sans (ni Inter ni Source Sans 3 ne sont installées pour matplotlib ; graisse « bold », DejaVu
  n'a pas de semi-gras).
- **Ajustements de palette** (luminosité seule, teinte gardée ; ΔE76 sous simulation deutéranopie Machado 2009, seuil 12) :
  correction A `#7D4E7A` → `#6A4268` (9,3 contre la référence → 12,8) ; UD `#E3A792` → `#DB8D72` (9,6 contre UN → 16,0) ;
  UN `#E8C9B0` → `#E5C1A4` (10,9 contre le couloir → 14,8) ; CN `#B7D1C6` → `#A4C5B7` (11,3 contre le couloir → 15,3) ;
  trait « couloir » des courbes `#766F63`. Après ajustement : 0 paire sous le seuil.
- **E6 et E7** : toutes les données viennent des 4 agents d'E4 (runs de développement CPU, conditions artificielles), pas
  encore des agents d'E5.

| id | titre-message | format | données (chemins) | valeurs clés affichées → concordance | statut | légende orale |
|---|---|---|---|---|---|---|
| T1 | Four zones, four kinds of control | 16:9 | `configs/zones/e1_4zones.yaml` ; tenseur de transition de `envs.layouts.four_zones` | flèches lues du tenseur (CN : 0,9 / 0,025 ; UD : 5 actions → même case) — conforme au doc E1 | complète | « Un tore 12 × 12, quatre zones qui ne diffèrent que par ce que font les actions. » |
| T1b | E4: one conveyor, one controllable zone | 16:9 | `envs.layouts.two_zones` (diagonale 1) | UD en nw, CD en se, référence = UD rendu contrôlable | complète | « En E4, un seul tapis roulant ; la référence a la même image mais tout est contrôlable. » |
| T2 | b asks the model: do my actions lead to different places? | 16:9 | aucune (schéma) ; I = log₂5 = 2,32 analytique | b ≈ 0 / b ≈ −2,3 = bornes analytiques, pas des données | complète | « b compare la diversité des cases prédites pour les 5 actions à celle de la référence. » |
| T3 | Two months: from measuring the belief to testing what it does | 16:9 | statuts : `docs/EXPERIMENTS.md`, `docs/E*.md` | E1–E4 fait, E5 en cours, E6–E7 fait* (agents d'E4), E8 à venir | complète (voir écart E8 ci-dessous) | « Quatre parties : mesurer, localiser, relier au comportement, corriger. » |
| E1a | The tabular measure recovers the true controllability | 16:9 | `experiments/E1_offline_4zones/artifacts/measurements.csv` (n = 1000, région all) | UD −2,32 (théorie −2,32) ; UN −1,65 (théorie −1,65) ; CD, CN 0,00 → conforme (doc E1 : −2,32193 ; −1,6518) | complète | « Sur données tabulaires, b retrouve la valeur théorique dans les quatre zones. » |
| E1b | 100 samples per state-action resolve b to a few hundredths of a bit | 16:9 | `experiments/E1_offline_4zones/artifacts/permutation_null.csv` (CN, b, bootstrap) | seuil 0,045 bit à n = 100 ; pente −0,42 → conforme (rapport : 0,04460 ; −0,415) | complète | « Le seuil de détection décroît avec n ; on a retenu n = 100, marge à 300. » |
| E2a | More data closes the gap in model error, not the bias in b | 16:9 | `experiments/E2_online_4zones/artifacts/figures/csv/F2_null_model_paired_diff.csv`, `table3_budget_trend_power_law.csv` | exposant b +0,01 [−0,21 ; +0,27] ; L1 −0,42 [−0,78 ; −0,16] → conforme à table3 | complète (voir note L1) | « L'écart online − apparié sur b ne décroît pas avec les données, alors que l'écart sur l'erreur du modèle décroît. » |
| E2b | Outside the zones, online and matched agents act identically | 16:9 | `…/csv/F4_delta_diff.csv`, `F4_delta_diff_exterior.csv` | 43 cases de couloir à Δ = 0 exactement (vérifié par assertion) ; zones −0,016 à +0,068 → conforme (table4 : max 0) | complète | « Hors des zones, la politique de l'agent online ne perd rien par rapport au contrôle apparié. » |
| E3a | A world model that imagines the next step for each action | 16:9 | schéma ; `IMPLEMENTATION_REPORT.md` E6 § 1 et § 16 | K = 16 tirages, ε = 0,1, γ = 0,9, h 64, z 8×8, λ-returns 15 pas | complète | « Un RSSM, un critic entraîné en imagination, et une politique qui imagine un pas par action. » |
| E3b | The world model learns the conveyor; controllable zones stay at 0 | 16:9 | `results/E3_worldmodel_4zones/campaign/tables/b_alpha_by_checkpoint.csv` | UD −2,26 (théorie −2,32) ; UN −1,78 (théorie −1,65) ; CN −0,01, CD −0,04, couloir −0,05 → conforme (contrôle : −2,26 ; −1,78) | complète | « Le world model apprend le tapis roulant ; les zones contrôlables restent à 0. » |
| E3c | The world model's state is organised by position, not by controllability | 16:9 | `results/E3_worldmodel_4zones/campaign/analysis.json` (`rsa_over_seeds/300000/h/partition`) | position 0,235 / 0,415 ; thème 0,012 / 0,024 ; contrôlabilité 0,001 / n.a. ; R² 0,32 / 0,54 → conforme au RESULTS_SECTION | complète | « La géométrie de h suit la position ; la contrôlabilité n'explique presque rien en propre. » |
| E4a | Train, freeze, then move colours and positions apart | 16:9 | `results/E4_generalization_gradient/final/tables/probes_b_step1.csv` (couleurs de p1-a) | 11 couleurs du segment, 3 positions + position seule + thème seul | complète | « On gèle les poids et on dissocie couleur et position dans un monde tout contrôlable. » |
| E4b | The belief is tied to position, not colour | 16:9 | `…/final/tables/probes_b_step1.csv` (sondes valides, images dupliquées comptées une fois) | ancienne UD −2,31 à distance 0, ≈ 0 en CD et neutre ; sondes valides UD 44/124, CD 47/124, neutre 90/124 | complète (voir écart « plateau ») | « Quelle que soit la couleur, b reste très négatif à l'ancienne place du tapis et ≈ 0 ailleurs. » |
| E4c | Remove the colour, keep the place: the belief stays | 16:9 | `…/final/tables/validity_image_zone.csv`, `theme_only_position_only.csv` | entraînement UD −2,31 ×4 ; position seule −2,15 / −2,30 / invalide / −2,27 ; thème seul p1-a −0,29 → conforme (rapport E4 : −2,150, −2,296, −2,274 ; −0,289) | complète | « Sans la couleur, la croyance reste ; avec la couleur ailleurs, elle ne suit pas. » |
| E4d | After one step that works, b recovers a little, then stalls | 16:9 | `…/final/tables/recovery_curves_extremes.csv`, `recovery_summary_by_position.csv` | +0,35 à +0,42 bit (pas 1 → plateau 6–10) → conforme | complète | « Une transition réussie fait remonter b de 0,4 bit, puis il plafonne ; le critic ne lit pas cette variation (D7). » |
| E6a | Same start, same controllable world: one wanders, one goes home | 16:9 | `scratch/e6dev/results/E6_phenotypes_latent/e6__p2-b-d1_e4__20261023T134444Z/episodes.npz` (non versionné) ; `results/E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv` | agent p2-b (0,40, le plus proche de la moyenne 0,38 des 4 exposés) ; départ case 39 (centre) ; épisode exposé 148 (part 0,36), référence 4 (part 1,00, la plus proche de 0,94 — les épisodes courts ont des parts discrètes) | complète | « Même départ : l'exposé erre 20 pas, la référence rejoint le but en 6. » |
| E6b | Where b is low, behaviour drops to chance | 16:9 | `results/E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv` (valides) | amas b ≈ −2,08, part 0,32–0,49 (n = 7) ; b ≈ 0, part 0,93–0,95 (n = 21) → conforme (contrôle : 0,32–0,49 ; 0,94) | complète | « Deux amas : b ≈ −2,1 au hasard, b ≈ 0 dirigé à 0,94 ; rien entre les deux. » |
| E7a | Behaviour follows the transition belief, not the critic | 16:9 | `results/E7_dissociation/dev_e4_cpu/analysis/table_2x2.csv` | k = 1 : 0,94 / 0,85 / 0,39 / 0,38 ; k = 5 : 0,95 / 0,95 / 0,39 / 0,39 → conforme | complète | « La croyance de transition fait l'essentiel ; le critic ne pèse qu'à k = 1, sous la vraie croyance. » |
| E7b | Forcing the conveyor belief into a reference: behaviour falls with it | 16:9 | `…/E7_dissociation/dev_e4_cpu/analysis/t1_runs.csv` (références, h moyen, k = 5) ; `points.csv` d'E6 | p = 0 : 0,94 ; p = 0,6 : 0,85 à b −1,45 ; p = 1 : 0,39 → conforme | complète | « Chez une référence, imposer le tapis dans l'anticipation fait baisser le comportement graduellement. » |
| E7c | True transitions fix behaviour; the latent correction plateaus | 16:9 | `results/E7_latent_manipulation/dev_e4_cpu_diag/analysis/d4_static.csv`, `d4_episodes.csv` | oracle en épisodes 0,75–0,93 (k = 1), 0,94–0,95 (k = 5) ; correction 0,67 à k = 10 (bande 0,65–0,71) → conforme | complète (voir note statique/épisodes) | « Avec les vraies transitions, l'exposé rejoint la référence dès k = 5 ; la correction latente plafonne vers 0,7. » |
| E7d | The probe is fooled: b moves, the critic's controllability does not | 16:9 | `results/E7_dissociation/d7/analysis/d7_correction.csv` (z̄ tiré), `d7_bins.csv` | A : Δb +1,2 à +1,8, ΔC1 ≈ 0 ou négatif sauf p1-b (+0,19 / +0,18) ; pente naturelle 0,28 C1/bit | **incomplète** : B par run absent | « La correction fait monter b sans changer ce que lit le critic, sauf chez p1-b. » |
| E7e | Three readers of the same prediction can disagree | 16:9 | schéma ; `IMPLEMENTATION_REPORT.md` E6 § 17–20 | Δb +1,2 à +1,9 (D7), « environ la moitié des pas » (D5 : 40–56 %) | complète | « Le même z′ est lu par la mesure, par le critic et par la correction ; ils peuvent diverger. » |

Format : toutes en 16:9 (13,33 × 7,5 in) ; aucune demi-slide n'a été nécessaire.

## Données manquantes, ambiguës, écarts

- **E7d, induction B** : les sorties légères ne gardent pas Δb et ΔC1 par run d'induction. Les points B sont les classes
  de b regroupées (`d7_bins.csv`, références induites) : Δb = centre de classe − 0, ΔC1 = C1 de la classe − C1 des
  références sans intervention à b ≈ 0. Pour un point par run, il faut relancer l'analyse D7 en exportant C1 par run.
  Ordonnée choisie : C1 (pas C2).
- **E4b, « plateau ≈ −2,3 quelle que soit la couleur »** : vrai près de la couleur d'entraînement ; la moyenne des sondes
  UD valides remonte vers −1,8 au-delà de 0,5 D (arc centré en T_ctrl, p2-b à −0,98 ; rapport E4 § 3 : −2,31 → −1,74).
  Le contraste avec CD/neutre (≈ 0) tient ; le plateau strict, non. À dire à l'oral.
- **E7c** : les courbes sont la part statique des choix greedy (100 contextes, `d4_static.csv`), la seule disponible pour
  k = 1, 2, 3, 5, 10 et pour les quatre variantes ; l'oracle statique est à 0,91 à k = 5 et 0,99 à k = 10. Le « rejoint
  la référence dès k = 5 » (0,94–0,95) vient des épisodes, montrés en losanges.
- **E2a** : « erreur L1 » = écart online − apparié de l'erreur L1 (comme l'exposant de la table 3), pas l'erreur L1
  absolue du modèle.
- **E1a** : budget n = 1000 (le résultat de référence du doc E1) ; à n = 100 les valeurs sont dans la tolérance de
  0,03 bit (rapport, étape B).
- **E3b** : théorie UN = −1,65 (b analytique de l'environnement, mesuré exactement en E1) ; le prompt citait −1,78, qui
  est la valeur observée en E3.
- **T3, E8** : le prompt place E8 en « IV corriger », mais `docs/E8_immunisation.md` décrit l'immunisation (prévention).
  Libellé de partie gardé, question tirée du doc.
- **E6a** : épisodes lus dans `scratch/` (non versionné) ; la figure ne se régénère que sur cette machine.
- Aucune figure E5 (en cours) : la frise T3 réserve sa place.
