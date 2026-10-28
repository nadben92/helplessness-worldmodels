# Figures v2 des slides — manifeste (seule source pour présenter les figures)

Les figures v2 ne portent aucun texte autre que les noms d'axes, les graduations et les noms de catégories ou de blocs.
Tout le reste (message, lecture des couleurs, valeurs, sources) est ici.

- **Régénérer** : `.venv/Scripts/python.exe slides/make_all_v2.py` (22 figures). Sorties : `out_v2/png/` (300 dpi,
  fond transparent), `out_v2/svg/` (transparent), `out_v2/png_white/` (300 dpi, fond blanc). La v1 (`out/`,
  `figures/`, `style/`, `MANIFEST.md`) n'a pas été modifiée.
- **Style** : `plt.style.use('default')` (axes en boîte, graduations par défaut, sans grille), DejaVu Sans, noms d'axes
  18 pt, graduations 15 pt, traits 2,8 pt. Palette : `style_v2/palette.py`.
- **Ajustement de palette** (contrôle deutéranopie `style_v2/check_palette.py`, Machado 2009, ΔE76, seuil 12) : la
  correction A `#8E2DE2` se confondait avec la référence `#1F6FEB` (ΔE 3,1) ; assombrie en `#6718AB` (ΔE 19,7), teinte
  gardée. Après ajustement : 0 paire sous le seuil.
- **Code couleur commun** : exposé vermillon, référence bleu vif, oracle vert, anti-oracle orange, correction A violet,
  induction B rose ; théorie en pointillé noir ; hasard en tirets gris. Zones : CD bleu clair, UD rouge-orangé, CN
  turquoise, UN orange, couloir gris clair, but = étoile jaune.
- **E6 et E7** : données des 4 agents d'E4 (runs de développement CPU), pas encore des agents d'E5.

## Par figure

Pour chaque figure : ce qui est tracé (lecture des couleurs), le message, les valeurs clés, les sources, la légende
orale.

### T1 — environnement d'E1–E3
- **Tracé** : tore 12 × 12, quatre zones 5 × 5 (UD, CN, CD, UN), couloir gris, but (étoile) en (0, 0). Seules les
  flèches du tapis roulant UD sont dessinées (une par case, lues dans le tenseur de transition de
  `envs.layouts.four_zones`). Le tapis pousse d'un rang vers le bas, à l'opposé du but.
- **Message** : quatre zones, quatre degrés de contrôle ; dans UD, toutes les actions mènent à la même case.
- **À dire** : CN = l'action marche 90 % du temps ; UN = tapis 90 % du temps ; récompense +50 au but, −1 par pas.
- **Source** : `configs/zones/e1_4zones.yaml`. **Oral** : « Un tore, quatre zones qui ne diffèrent que par ce que font
  les actions. »

### T1b — environnement d'E4
- **Tracé** : à gauche l'exposé, à droite la référence, avec les vrais thèmes d'entraînement de la configuration
  p1-a-d1 (la même que E4a). Les deux panneaux ont exactement les mêmes couleurs, quadrant par quadrant ; seule la
  dynamique diffère : flèches du tapis roulant dans le quadrant nord-ouest de l'exposé, aucune flèche chez la référence
  (où ce quadrant est rendu contrôlable par `derive_control`).
- **Couleurs RGB vérifiées** (`two_zones.training_colors` avec T_inc et T_ctrl lus dans `probes_b_step1.csv`, identiques
  au tableau des configurations de `results/E4_generalization_gradient/final/REPORT.md`) : nord-ouest (ancienne UD)
  T_inc = (227, 17, 15) ; sud-est (CD) T_ctrl = (22, 0, 244) ; nord-est et sud-ouest = couleur du couloir
  (128, 128, 128), comme le couloir lui-même.
- **Oral** : « Même image, même but ; seul le tapis distingue l'exposé de sa référence. »

### T2 — mesure de b
- **Tracé** : (h, z) → GRU (une fois par action) → prior (16 tirages) → décodeur → case prédite (gabarit le plus
  proche) → I(a ; s′|s) → b = I − I_ref. En bas : cas contrôlable (5 actions, 5 cases : I = log₂5 = 2,32, b ≈ 0) et cas
  tapis (5 actions, une case : I = 0, b ≈ −2,32). Valeurs analytiques, pas des données.
- **Oral** : « b demande au modèle si ses actions mènent à des endroits différents, comparé à la référence. »

### T3 — frise
- **Tracé** : E1–E8 en quatre parties (Mesurer, Localiser, Croyance → comportement, Corriger). Remplissage bleu = fait ;
  orange = en cours (E5) ; tirets = à venir (E8). E6 et E7 sont faits sur les agents d'E4, à refaire sur ceux d'E5.
- **E8** : thérapie (réapprentissage en environnement contrôlable) et rechute. La formulation « immunisation » de
  `docs/E8_immunisation.md` n'est pas reprise (dépassée, selon ta consigne).
- **Questions** : E1 la mesure retrouve-t-elle la contrôlabilité vraie ? E2 l'exploration de l'agent biaise-t-elle b ?
  E3 un world model apprend-il la zone incontrôlable ? E4 la croyance tient-elle à la position ou à la couleur ? E5 un
  entraînement peut-il déplacer la croyance sur la couleur ? E6 le comportement suit-il la croyance ? E7 croyance ou
  critic : qui pilote le comportement ? E8 peut-on corriger la croyance, et rechute-t-elle ?

### E1a — b tabulaire contre théorie
- **Tracé** : barres = moyenne sur 20 graines (n = 1000), barres d'erreur = IC t à 95 % (plus étroits que le trait),
  points = graines, pointillé = b analytique (I*(exp) − I*(ctrl), mêmes lignes).
- **Valeurs** : UD −2,32 (théorie −2,32) ; UN −1,65 (théorie −1,65) ; CD 0,00 ; CN 0,00 — conforme au doc E1
  (−2,32193 ; −1,6518).
- **Source** : `experiments/E1_offline_4zones/artifacts/measurements.csv`. **Oral** : « En tabulaire, b retrouve la
  valeur théorique dans les quatre zones. »

### E1b — seuil de détection contre budget
- **Tracé** : p95 du nul bootstrap (zone CN, où b = 0 est vrai), moyenne sur 20 graines, bande IC t 95 %, points =
  graines, log-log. Trait noir vertical : n = 100 retenu ; tirets gris : n = 300 (marge 3σ).
- **Valeurs** : 0,045 bit à n = 100 ; 0,015 à n = 1000 ; pente log-log −0,42 (rapport : 0,04460 ; −0,415).
- **Source** : `experiments/E1_offline_4zones/artifacts/permutation_null.csv`. **Oral** : « Avec 100 échantillons par
  couple, b est résolu à quelques centièmes de bit. »

### E2a — l'écart online ne décroît pas
- **Tracé** : gauche, |écart moyen apparié online − apparié| sur b̂ (vermillon) ; droite, idem sur l'erreur L1 du
  modèle (bleu) ; 16 checkpoints, 20 graines, bande IC 95 % (coupée au plancher de l'axe là où elle croise 0) ; trait
  noir = ajustement en loi de puissance sur les checkpoints ≥ 10⁴ (table 3 d'E2).
- **Valeurs** : exposant b +0,01 [−0,21 ; +0,27] ; exposant L1 −0,42 [−0,78 ; −0,16] (IC bootstrap).
- **Note** : « L1 » = écart online − apparié de l'erreur L1, pas l'erreur absolue.
- **Source** : `experiments/E2_online_4zones/artifacts/figures/csv/F2_null_model_paired_diff.csv`,
  `table3_budget_trend_power_law.csv`. **Oral** : « Plus de données réduit l'écart d'erreur du modèle, pas le biais
  sur b. »

### E2b — Δ hors des zones
- **Tracé** : carte de Δ (valeur perdue par la politique greedy), online − apparié, checkpoint final, moyenne de 20
  graines, repère canonique ; échelle divergente (bleu négatif, rouge positif, blanc = 0). Codes de zone en gris.
- **Valeurs** : les 43 cases de couloir (hors but) sont à Δ = 0 exactement (vérifié par assertion) ; dans les zones de
  −0,016 à +0,068 (seule CN s'écarte de 0).
- **Source** : `…/csv/F4_delta_diff.csv`, `F4_delta_diff_exterior.csv`. **Oral** : « Hors des zones, l'agent online
  ne perd rien par rapport au contrôle apparié. »

### E3a — architecture
- **Tracé** : à gauche le RSSM (image → encodeur → posterior ; GRU → prior ; KL en tirets entre prior et posterior ;
  posterior → décodeur) ; à droite la décision (imagination à partir du prior → tête but et critic → politique).
- **À dire** : politique = anticipation d'un pas, Q(a) = moyenne sur 16 tirages du prior de r + γ·c·V, ε-greedy
  ε = 0,1, γ = 0,9 ; critic entraîné uniquement en imagination (λ-returns sur 15 pas) ; h de dimension 64, z = 8 × 8
  catégories. Source : `IMPLEMENTATION_REPORT.md` d'E6, § 1 et § 16.

### E3b — b du world model pendant l'entraînement
- **Tracé** : b par zone (couleurs des zones de T1 ; couloir en gris foncé), 4 graines en traits fins, moyenne en trait
  épais sur les checkpoints communs ; pointillés : −log₂5, −1,65, 0.
- **Valeurs à 300 000 pas** : UD −2,26 (théorie −2,32) ; UN −1,78 (théorie −1,65) ; CN −0,01 ; CD −0,04 ; couloir
  −0,05 (conforme : −2,2582 ; −1,7788).
- **Source** : `results/E3_worldmodel_4zones/campaign/tables/b_alpha_by_checkpoint.csv`. Graines pilotes 0 et 5 :
  dernier replay à 290 000. **Oral** : « Le world model apprend le tapis ; les zones contrôlables restent à 0. »

### E3c — partition de variance (RSA de h)
- **Tracé** : part unique de variance expliquée par facteur, exposé (vermillon) contre référence (bleu), 300 000 pas,
  4 graines.
- **Valeurs** : position 0,235 / 0,415 ; thème 0,012 / 0,024 ; contrôlabilité 0,001 / absente (constante dans la
  référence) ; stochasticité 0,008 / 0,008 ; partagée 0,061 / 0,091 ; R² total 0,32 / 0,54.
- **Note** : dans E3, thème, position et zone sont confondus (E4 les sépare).
- **Source** : `results/E3_worldmodel_4zones/campaign/analysis.json` (`rsa_over_seeds/300000/h/partition`).
  **Oral** : « L'état du modèle est organisé par la position, pas par la contrôlabilité. »

### E4a — design d'E4
- **Tracé** : entraînement de l'exposé et de la référence (couleurs réelles de p1-a : T_inc rouge, T_ctrl bleu), gel,
  puis sondes : un quadrant recoloré (encadré) à l'ancienne UD, l'ancienne CD, la case neutre, et « position seule »
  (ancienne UD à la couleur du couloir). Bande : les 11 couleurs du segment T_inc → T_ctrl (deux arcs aussi).
  Pas de flèches de dynamique.
- **À dire** : monde de test entièrement contrôlable, poids gelés, zéro donnée ; « thème seul » = case neutre en
  T_inc ; b lu au premier pas dans le quadrant ; on ne garde que les sondes où la référence prédit encore bien
  (masse ≥ 0,90 et |I_ref − log₂5| ≤ 0,15). 4 configurations, 100 sondes chacune.
- **Source** : `results/E4_generalization_gradient/final/tables/probes_b_step1.csv`.

### E4b — la croyance tient à la position
- **Tracé** : b au pas 1 contre distance de la couleur sondée à la couleur d'entraînement du quadrant (/ D) ; sondes
  valides seulement (images en double comptées une fois) ; vermillon = ancienne UD, bleu = ancienne CD, gris = neutre ;
  traits = moyennes par tranches de 0,1 D (≥ 2 sondes), points = sondes ; pointillé = −log₂5.
- **Valeurs** : ancienne UD à −2,31 à distance 0 ; CD et neutre ≈ 0. Sondes valides : UD 44/124, CD 47/124,
  neutre 90/124. La neutre commence à 0,38 D (sa couleur d'entraînement est celle du couloir).
- **Écart à dire** : **la moyenne des sondes UD valides remonte vers −1,8 au-delà de 0,5 D** (arc centré en T_ctrl ;
  p2-b jusqu'à −0,98 ; rapport E4 § 3 : −2,31 → −1,74). Le contraste avec CD et neutre tient ; un plateau strict à
  −2,3, non.
- **Oral** : « Quelle que soit la couleur, b reste très négatif à l'ancienne place du tapis, et ≈ 0 ailleurs. »

### E4c — configurations × conditions
- **Tracé** : b au pas 1 (rouge = −2,32, blanc = 0) ; lignes = 4 configurations ; colonnes = image d'entraînement lue à
  l'ancienne UD, position seule, thème seul, image d'entraînement à la case neutre, à l'ancienne CD ; hachures = sonde
  invalide.
- **Valeurs** : ancienne UD −2,31 (×4) ; position seule −2,15 / −2,30 / invalide (p2-a, −0,66) / −2,27 ; thème seul
  −0,29 (p1-a) / −0,02 / invalide (p2-a, −0,18) / −0,03 ; neutre −0,02 / −0,00 / −0,00 / −0,02 ; CD ≈ 0.
- **Source** : `…/final/tables/validity_image_zone.csv`, `theme_only_position_only.csv`. **Oral** : « Sans la
  couleur, la croyance reste ; la couleur seule ailleurs ne l'emporte pas. »

### E4d — récupération dans l'épisode
- **Tracé** : b par pas dans l'ancien quadrant UD (image d'entraînement), 4 configurations (fins) et moyenne (épais) ;
  pointillé = −log₂5.
- **Valeurs** : de −2,31 au pas 1 à un plateau vers −1,85 ; écart pas 1 → plateau (pas 6–10) de +0,35 à +0,42 bit selon
  la configuration (toutes sondes UD valides).
- **À dire** : variation de b à l'intérieur d'une classe ; le critic ne la lit pas (D7).
- **Source** : `…/final/tables/recovery_curves_extremes.csv`, `recovery_summary_by_position.csv`.

### E6a — trajectoires
- **Tracé** : même départ (centre de l'ancien quadrant UD, cercle), monde de test contrôlable, image d'entraînement ;
  gauche = exposé (vermillon), droite = référence (bleu) ; clair → foncé = temps ; étoile = but. Au-dessus de chaque
  panneau, le nombre de pas de l'épisode tracé (seul texte de la figure) : « 20 steps » et « 6 steps », vérifiés contre
  `training__<rôle>__length` dans `episodes.npz` (épisodes 148 et 4 ; tous deux terminés au but, état 0).
- **Valeurs** : agent p2-b (part dirigée 0,40, la plus proche de la moyenne 0,38 des 4 exposés) ; épisode exposé 148 :
  20 pas, part dirigée 0,36 ; référence épisode 4 : 6 pas, part 1,00 (pool de la référence 0,94 ; épisodes courts à
  parts discrètes). Règle de choix fixée avant de voir les trajectoires.
- **Source** : `scratch/e6dev/results/E6_phenotypes_latent/e6__p2-b-d1_e4__20261023T134444Z/episodes.npz` (non
  versionné), `results/E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv`. **Oral** : « Même départ : l'exposé
  erre 20 pas, la référence rejoint le but en 6. »

### E6b — part dirigée contre b
- **Tracé** : un point par couple (agent, image, modèle) valide ; plein vermillon = exposé, creux bleu = référence
  (b = 0 par construction) ; forme = image (rond : entraînement, carré : position seule, triangle : thème seul,
  losange : case neutre) ; barres = IC bootstrap 95 % ; tirets = hasard 2/5.
- **Valeurs** : amas b ≈ −2,08, part 0,32–0,49 (7 couples) ; amas b ≈ 0, part 0,93–0,95 (21 couples) ; rien entre les
  deux. Invalides exclus : p2-a position seule et thème seul.
- **Source** : `results/E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv`. **Oral** : « Deux amas : là où b est
  bas, le comportement tombe au hasard. »

### E7a — le 2 × 2 croyance × critic
- **Tracé** : part dirigée (blanc = 0, bleu = 1), lignes = croyance (vraie, tapis), colonnes = critic (référence,
  exposé) ; gauche k = 1, droite k = 5 ; moyenne des 4 agents.
- **Valeurs** : k = 1 : 0,94 / 0,85 [0,75–0,93] / 0,39 / 0,38 [0,32–0,49] ; k = 5 : 0,95 / 0,95 / 0,39 / 0,39.
- **À dire** : la ligne « tapis » est le plancher du hasard par construction (toutes les actions prédisent la même case,
  les Q sont à égalité, 2 actions sur 5 sont dirigées) ; le critic ne pèse que sous la vraie croyance à k = 1 (0,09).
- **Source** : `results/E7_dissociation/dev_e4_cpu/analysis/table_2x2.csv`. **Oral** : « C'est la croyance de
  transition qui fait le comportement, pas le critic. »

### E7b — dose-réponse de l'anti-oracle (k = 5)
- **Tracé** : orange = références dont les particules d'anticipation suivent le tapis avec probabilité p (0 à 1),
  fins par agent, épais = moyenne ; abscisse = b effectif ; points = agents naturels d'E6 (plein vermillon = exposés,
  creux bleu = références) ; tirets = hasard.
- **Valeurs** : p = 0 : 0,94 à b 0,00 ; p = 0,6 : 0,85 à b −1,45 ; p = 1 : 0,39 à b −2,20 (conforme).
- **Note** : le b effectif est calculé sur le noyau mélangé, pas la même mesure que le b naturel.
- **Source** : `results/E7_dissociation/dev_e4_cpu/analysis/t1_runs.csv`, `points.csv` d'E6. **Oral** : « Imposer le
  tapis dans l'anticipation d'une référence fait baisser son comportement graduellement. »

### E7c — part dirigée contre anticipation k
- **Tracé** : une seule mesure pour les quatre variantes : part statique des choix greedy dirigés sur 100 contextes de
  la zone (`d4_static.csv`) ; bleu = référence, vert = exposé + oracle (vraies transitions), violet = exposé +
  correction latente à la racine, vermillon = exposé (son modèle) ; moyenne des 4 agents, bande = étendue ; tirets =
  hasard. Les épisodes ont été retirés de la figure.
- **Valeurs (statique)** : référence 0,98 ; oracle 0,78 (k = 1) → 0,91 (k = 5) → 0,99 (k = 10) ; correction 0,70 →
  0,67 (étendue 0,65–0,71 à k = 10) ; exposé 0,47 → 0,41.
- **En épisodes** (non tracé) : l'oracle est à 0,75–0,93 à k = 1 et **0,94–0,95 à k = 5** (comme la référence).
- **Source** : `results/E7_latent_manipulation/dev_e4_cpu_diag/analysis/d4_static.csv` (`d4_episodes.csv` pour les
  épisodes). **Oral** : « Avec les vraies transitions, l'exposé rejoint la référence en anticipant ; la correction
  latente plafonne vers 0,7. »

### E7d — « la sonde est trompée »
- **Tracé** : abscisse Δb, ordonnée ΔC1 (C1 = part d'argmax dirigé de V(h′ₐ, z̄), z̄ tiré du mélange des 5 priors).
  Violet = correction A, un point par (agent, réglage), 10 points, IC 95 % ; cerclés de noir = p1-b. Rose = induction B,
  **un point par run** (31 runs : losanges pleins = direction inverse, 8 runs ; creux = direction aléatoire de même
  norme, 23 runs). Pointillé = pente entre agents naturels (exposés b ≈ −2,02, C1 0,39 ; références b ≈ 0,14, C1 1,00 :
  0,28 C1 par bit).
- **Valeurs** : A : Δb +1,25 à +1,86 ; ΔC1 ≈ 0 ou négatif pour p1-a, p2-a, p2-b ; p1-b +0,20 et +0,18 (seul changement
  réel). B : Δb −0,80 à −0,01 (inverse), ΔC1 −0,02 à +0,00 : la contrôlabilité vue par le critic ne bouge pas non plus.
- **État** : B par run recalculé en lecture seule depuis les épisodes D7 existants (`scratch/e6dev/results/
  E7_dissociation/e7d7__*__natural__*`), avec la collecte de D7 elle-même (`analyse_e7_d7.collect_natural`), par
  `figures_v2/_extract_e7d_induction.py` → `slides/data_v2/e7d_induction_runs.csv`. Rien n'a été relancé. Les 31 runs
  sont ceux de D6/D7 (localisation tenue). Définition : Δ = moyenne du run − moyenne de la référence sans intervention
  du même agent, sur les pas de zone bien localisés (A, lui, compare pas à pas avant/après correction).
- **Oral** : « La correction fait monter b sans changer ce que lit le critic, sauf chez p1-b ; l'induction fait
  baisser b sans le changer non plus. »

### E7e — trois lecteurs de z′
- **Tracé** : h′ₐ → z′ prédit → (bleu) décodeur → b ; (vermillon) critic → action ; (violet) KL → correction de h.
- **À dire** : correction A : monte le lecteur 3 et le lecteur 1 (b +1,2 à +1,9 bit), mais 2 reste au hasard pour 3
  agents sur 4 (D7) ; références induites : 1 dit que l'action échoue dans environ la moitié des pas (40–56 %, D5),
  2 dirige quand même (D6). Source : `IMPLEMENTATION_REPORT.md` d'E6, § 17–20.

### E7f — équations de la correction latente δh
- **Tracé** : `h ← h + δh` ; `min_δh Σ_a KL(q_target(z|a) ‖ p(z | GRU(h+δh, z, a))) + λ‖δh‖² + μ_pos·D_pos` ; sous les
  termes : « controllability » (violet de la correction A), « small change », « keep position » ; « weights frozen ».
  Taille de l'équation ajustée automatiquement pour tenir dans la slide (28 pt au plus).
- **Conforme au code** (`src/helplessness/probing/belief_steering.py`) : même ordre des arguments de la KL (cible
  d'abord), somme sur les actions et sur les variables catégorielles, pas de perte en pixels. Précisions à dire :
  q_target(z|a) = posterior de l'image de la **vraie** case suivante pour l'action a, calculé avec le h **non
  modifié** (q(z | h′ₐ, e(x(s′ₐ)))) ; D_pos = KL(pos(h) ‖ pos(h+δh)), pos = position douce de l'agent lue dans l'image
  décodée (softmax, τ = 0,02) ; optimisation par Adam depuis δh = 0, une situation à la fois, poids gelés.
- **Omis volontairement** : le terme μ_goal·(p_goal(h+δh, z) − p_goal(h, z))² du code (μ_goal calibré par agent : 10
  pour p1-a et p1-b, 0 pour p2-a et p2-b ; rapport § 11).
- **Oral** : « On cherche le plus petit changement de h qui fait prédire au modèle que ses actions marchent, sans
  déplacer l'agent dans l'image. »

### E7g — pourquoi la correction échoue
- **Tracé** : deux panneaux, abscisse commune Δb (bits). Gauche : localisation dans la zone (part des pas où le gabarit
  le plus proche de l'image décodée est la bonne case) ; droite : part dirigée (épisodes).
- **Un point** = un (agent, réglage), image d'entraînement, 4 agents d'E4 :
  - **cercles violet clair** : correction **sans contrainte** appliquée en épisodes (premier rapport, modes
    « direction » et « pas à pas », α ∈ {0,25 ; 0,5 ; 0,75 ; 1 ; 1,5}), moyenne sur les 180 épisodes ;
  - **cercles violet foncé** : correction **contrainte** appliquée en épisodes (réglage retenu par agent, α = 1, modes
    « direction contrainte » et « pas à pas contraint ») ;
  - **carrés** (gauche seulement) : balayage statique (100 situations par agent, pas d'épisodes), λ = 0,01 :
    μ_pos = 0 en clair, μ_pos ∈ {1 ; 10 ; 100} en foncé ; abscisse = gain de b, ordonnée = localisation après correction.
  - Δb en épisodes = b au pas 1 sous intervention − b au pas 1 sans intervention, même agent.
- **Repères** : point vermillon = exposés sans intervention (moyenne des 4 : localisation 0,79, part dirigée 0,38) ;
  correction complète = −b sans intervention, moyenne des 4 = **2,13 bit** ; étoile gauche = (2,13 ; localisation 1) ;
  étoile droite = (2,13 ; part dirigée de l'oracle à k = 1 en épisodes, moyenne des 4 = **0,85**, étendue 0,75–0,93) ;
  tirets gris à gauche = localisation sans intervention (0,79) ; tirets bleus à droite = références (0,94) ; tirets gris
  à droite = hasard (0,4).
- **Valeurs de contrôle retrouvées** : sans contrainte, pas à pas, α = 1 : b −2,13 → −0,31, part dirigée 0,38 → 0,58,
  localisation 0,33 au pas 1 (0,34 dans la zone) ; balayage λ = 0,01, p1-a : localisation 0,27 (μ_pos = 0) → 0,74
  (μ_pos = 1), gain +1,52 à +1,69 bit selon l'agent ; pas à pas contraint : part dirigée 0,49 → 0,71 (p1-b), 0,40 → 0,48
  (p2-b), 0,32 → 0,44 (p1-a), 0,33 → 0,61 (p2-a).
- **Ce que montrent les données** (ne correspond qu'en partie au message attendu) :
  - sans contrainte : la localisation s'effondre quand Δb monte (beaucoup de points vers 0) ; en direction, Δb reste
    souvent sous 1 bit ;
  - direction contrainte : la localisation s'effondre quand même (0,00–0,05 dans la zone) ; Δb 0,5–1,2 ;
  - **pas à pas contraint : la localisation est conservée** (0,63–0,88, autour du niveau sans intervention) **et Δb
    atteint 1,4–1,9 bit**, près de la correction complète. Ces points ne tombent pas « en bas à gauche » : l'échec est
    sur le **comportement**, à 0,44–0,71, sous l'oracle (0,85) et la référence (0,94). D7 explique pourquoi (E7d,
    E7g_critic) : le critic ne lit pas ce gain de b, sauf chez p1-b.
- **Sources** : `results/E7_latent_manipulation/dev_e4_cpu/analysis/checks.csv` (sans contrainte),
  `dev_e4_cpu_v2/analysis/checks.csv` (contrainte appliquée), `dev_e4_cpu_v2/analysis/constrained_settings.csv`
  (balayage, données de `constrained_tradeoff.png`), `dev_e4_cpu_diag/analysis/d4_episodes.csv` (oracle),
  `results/E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv` (référence).
- **Oral** : « Sans contrainte, corriger b fait perdre l'agent dans l'image ; avec contrainte, b est corrigé et
  l'agent reste localisé, mais le comportement ne suit qu'à moitié. »

### E7g_critic — même figure en contrôlabilité vue par le critic
- **Tracé** : abscisse ΔC1 (D7, z̄ tiré ; C1 = part d'argmax dirigé de V(h′ₐ, z̄)) ; gauche localisation, droite part
  dirigée ; **un point = un run corrigé de D7** (pas à pas contraint, α = 1, 10 couples agent × (λ, μ_pos)), ΔC1 de
  `d7_correction.csv` (pas corrigés − non corrigés, mêmes pas), localisation et part dirigée du `summary.json` du run.
  Les runs sans contrainte n'ont pas de ΔC1 (D7 ne les a pas rejoués) : seuls les runs contraints sont tracés.
- **Repères** : vermillon = exposés sans intervention (runs naturels de D7, moyenne des 4 : localisation 0,79, part
  0,38) ; correction complète en C1 = C1 des références naturelles − C1 des exposés naturels = **0,61** (`d7_bins.csv`,
  pondéré par les pas) ; étoiles à (0,61 ; 1) et (0,61 ; 0,85) ; mêmes lignes qu'E7g.
- **Valeurs** : ΔC1 de −0,13 à +0,02 pour 8 runs ; +0,18 et +0,20 pour p1-b (les deux points à part dirigée 0,71–0,75).
  Le gain de Δb (+1,2 à +1,9 bit) devient presque nul en ΔC1 : il est en grande partie illusoire pour le critic.
- **Sources** : `results/E7_dissociation/d7/analysis/d7_correction.csv`, `d7_bins.csv`, `results/E7_dissociation/d7/runs/*/summary.json`.
- **Oral** : « Vu par le critic, la correction ne déplace presque rien, sauf chez p1-b ; c'est là que le comportement
  remonte le plus. »

### E7h — la correction latente en une figure (mêmes 4 agents partout)
- **Tracé** : 4 panneaux × 4 conditions (exposé vermillon, sans contrainte violet clair, contrainte violet foncé,
  référence bleu) ; un marqueur par agent, même forme dans tous les panneaux (rond p1-a, carré p1-b, triangle p2-a,
  losange p2-b) ; trait noir court = moyenne des agents disponibles ; tirets gris = hasard 2/5 (C1, part dirigée).
  Image d'entraînement, 4 agents d'E4. E7g et E7g_critic restent inchangées.
- **Conditions** :
  - exposé et référence sans intervention : épisodes d'E6 (`points.csv`) ;
  - sans contrainte : correction pas à pas, α = 1, λ = 0,01, sans μ (premier rapport, `dev_e4_cpu/analysis/checks.csv`) ;
  - contrainte : correction pas à pas contrainte, α = 1, **réglage enregistré** de chaque agent, rejoué par D7
    (`d7_identity.csv`, `recorded = True`) : p1-a λ = 0,1 μ_pos = 1 ; p1-b λ = 0,01 μ_pos = 1 ; p2-a λ = 0,01
    μ_pos = 10 ; p2-b λ = 0,1 μ_pos = 1. Les runs D7 reproduisent exactement les runs enregistrés (§ 20).
- **Mesures** :
  - **b** : b au pas 1 (exposé − référence, 9 départs au centre de la zone, 20 épisodes chacun) ; la référence vaut 0
    par construction. C'est la seule définition disponible dans les quatre conditions ;
  - **critic C1** : part des pas de zone bien localisés où l'argmax de V(h′ₐ, z̄) est une action dirigée, z̄ tiré du
    mélange des 5 priors (z̄ principal de D7). Exposé et référence : moyenne par agent sur leurs pas bien localisés,
    extraite en lecture seule des épisodes D7 (`figures_v2/_extract_e7h_c1.py` → `data_v2/e7h_c1_natural.csv`, avec
    `analyse_e7_d7.collect_natural`). Contrainte : `C1_corr` de `d7_correction.csv` (pas corrigés bien localisés) ;
  - **localisation** : part des pas dans la zone où le gabarit le plus proche de l'image décodée est la bonne case
    (`loc_template_zone`) ;
  - **part dirigée** : part des pas dans la zone qui rapprochent du but (ε-greedy, épisodes).
- **Moyennes par condition (b ; C1 ; localisation ; part dirigée)** :
  - exposé : −2,13 ; 0,40 ; 0,79 ; 0,38 ;
  - sans contrainte : −0,31 ; **manquant** ; 0,34 ; 0,58 ;
  - contrainte : −0,38 ; 0,49 ; 0,79 ; 0,56 ;
  - référence : 0,00 ; 1,00 ; 1,00 ; 0,94.
- **Par agent (p1-a, p1-b, p2-a, p2-b)** : exposé C1 0,39 / 0,54 / 0,35 / 0,30 ; contrainte C1 0,36 / 0,74 / 0,45 /
  0,41 ; contrainte part dirigée 0,44 / 0,71 / 0,61 / 0,48 ; sans contrainte localisation 0,16 / 0,53 / 0,26 / 0,41.
- **Case vide** : C1 sans contrainte. D7 ne l'a pas calculé ; le rejeu avec la collecte de D7 coûte 1 520–2 180 s par
  run (rapport § 20), soit environ 36 min de temps réel pour les 4 agents en parallèle, au-delà de la limite de 30 min.
  Non relancé.
- **Ce que montrent les données** :
  - contrainte : b remonte presque au niveau de la référence (−0,38) et la localisation est conservée (0,79), mais C1
    reste près du hasard pour 3 agents sur 4 (0,36–0,45 ; p1-b 0,74) ;
  - la part dirigée remonte **à mi-chemin** (0,56) entre l'exposé (0,38) et la référence (0,94), pas « proche de
    l'exposé » ; c'est p1-b, le seul dont C1 monte, qui remonte le plus (0,71) ;
  - sans contrainte : b remonte autant (−0,31) mais la localisation s'effondre (0,34) ; sa part dirigée (0,58) n'est
    donc pas interprétable comme un effet de la croyance.
- **Oral** : « Corriger h ramène b presque au niveau de la référence. Sans contrainte, l'agent n'est plus localisé ;
  avec contrainte, il l'est, mais le critic ne voit pas le changement, et le comportement ne remonte qu'à moitié. »

### E7g_light — E7g allégée
- **Tracé** : mêmes panneaux, axes, couleurs, étoiles et lignes qu'E7g (localisation à gauche, part dirigée à droite,
  en fonction de Δb). E7g, E7g_critic et E7h ne sont pas modifiées.
- **Un point** = un (agent, réglage), image d'entraînement, moyenne sur les épisodes du run :
  - **violet clair, 20 points** : correction sans contrainte, mode pas à pas seulement, 4 agents × α ∈ {0,25 ; 0,5 ;
    0,75 ; 1 ; 1,5} (`dev_e4_cpu/analysis/checks.csv`) ;
  - **violet foncé, 10 points** : correction contrainte pas à pas, α = 1, exactement les 10 runs corrigés de D7
    d'E7g_critic (p1-a : λ 0,1 μ_pos 1, λ 0,01 μ_pos 1, λ 0,01 μ_pos 10 ; p1-b : λ 0,01 μ_pos 1 et 10 ; p2-a : λ 0,01
    μ_pos 1 et 10 ; p2-b : λ 0,1 μ_pos 1, λ 0,01 μ_pos 1 et 10 ; `results/E7_dissociation/d7/runs/*corrected*/summary.json`) ;
  - **vermillon, 1 point** : exposés sans intervention (moyenne des 4) ; **étoile** comme dans E7g.
  - Retirés par rapport à E7g : le mode « direction » (contraint ou non) et les carrés du balayage statique.
- **Δb** = b au pas 1 du run − b au pas 1 du même agent sans intervention (mêmes références qu'E7g).
- **Valeurs** : violet foncé : Δb 1,40–1,98, localisation 0,63–0,88 (niveau sans intervention : 0,79), part dirigée
  0,39–0,75 (oracle 0,85, référence 0,94) ; violet clair : localisation de 0,97 à 0,07, d'autant plus basse que Δb est
  grand.
- **Oral** : « Sans contrainte, plus on corrige b, plus l'agent se perd dans l'image ; avec contrainte, b est presque
  corrigé et l'agent reste localisé, mais le comportement reste loin de l'étoile. »

### E7_bars — la correction latente en barres
- **Tracé** : 3 panneaux (b au pas 1, localisation dans la zone, part dirigée en épisodes à k = 1) × 5 catégories
  (exposed vermillon, latent fix violet clair, constrained fix violet foncé, oracle vert, reference bleu). Barre =
  moyenne des 4 agents d'E4 (image d'entraînement) ; points noirs = les 4 agents ; tirets gris = hasard 2/5. Les autres
  figures ne sont pas modifiées.
- **Sources et définitions** : celles d'E7h pour exposed, latent fix (pas à pas sans contrainte, α = 1), constrained fix
  (réglage enregistré de D7 par agent) et reference. Oracle : vraies transitions dans le planificateur, k = 1 (D4) :
  - **b = 0** : la croyance utilisée est la vraie ; c'est la valeur de la référence, pas une mesure ;
  - **localisation = celle de l'exposé dans E6** (même modèle, h non modifié). Les runs D4 de l'oracle n'ont mesuré
    que le pas 1 (`measure = "first"` dans `run_e7_diagnostics.py`) : leur « localisation dans la zone » est en fait celle
    du pas 1 (0,97 / 0,98 / 0,94 / 1,00) et n'a pas été utilisée ;
  - **part dirigée** : `dev_e4_cpu_diag/analysis/d4_episodes.csv`, exposé, k = 1.
- **Valeurs (moyenne ; p1-a / p1-b / p2-a / p2-b)** :

  | Catégorie | b (bits) | localisation | part dirigée |
  |---|---|---|---|
  | exposed | −2,13 (−2,12 / −2,17 / −2,06 / −2,17) | 0,79 (0,87 / 0,61 / 0,96 / 0,70) | 0,38 (0,32 / 0,49 / 0,33 / 0,40) |
  | latent fix | −0,31 (−0,48 / −0,37 / −0,12 / −0,27) | 0,34 (0,16 / 0,53 / 0,26 / 0,41) | 0,58 (0,42 / 0,69 / 0,61 / 0,59) |
  | constrained fix | −0,38 (−0,72 / −0,37 / −0,15 / −0,28) | 0,79 (0,63 / 0,79 / 0,87 / 0,88) | 0,56 (0,44 / 0,71 / 0,61 / 0,48) |
  | oracle | 0 (par construction) | 0,79 (= exposé, E6) | 0,85 (0,75 / 0,88 / 0,93 / 0,86) |
  | reference | 0,00 | 1,00 | 0,94 |

- **Lecture** : constrained fix ramène b presque au niveau de l'oracle et de la référence en gardant la localisation de
  l'exposé, mais la part dirigée reste à mi-hauteur (0,56), bien sous l'oracle (0,85). Latent fix remonte b autant mais
  la localisation s'effondre (0,34).
- **Oral** : « Corriger la croyance dans h donne presque le bon b ; donner la vraie croyance au planificateur donne
  presque le bon comportement. La correction n'y arrive qu'à moitié. »

### E6a_trajectories_loop — trajectoires, l'exposé qui tourne dans la zone (E6a_trajectories inchangée)
- **Tracé** : comme E6a (agent p2-b, départ au centre de l'ancien quadrant UD, image d'entraînement, monde de test
  contrôlable ; gauche exposé vermillon, droite référence bleue, clair → foncé = temps, étoile = but ; seul texte : le
  nombre de pas). Référence inchangée : épisode 4, 6 pas.
- **Épisode exposé : 121, 19 pas** (15 dans la zone, 7 cases distinctes, 2 boucles, puis 4 pas droit au but par le
  couloir de gauche) ; trajet 39 → 51 → 51 → 39 → 40 → 52 → 52 → 51 → 50 → 49 → 50 → 38 → 50 → 50 → 49 → 48 → 36 → 24 →
  12 → 0. Les « stay » (51, 52, 50 répétés) ne se voient pas sur le tracé.
- **Règle de choix (demande du 28 octobre : montrer la boucle dans la zone, pas l'hésitation dans le couloir de
  l'épisode 148)** : parmi les épisodes exposés partant de ce départ et finissant au but, ceux dont le trajet après la
  première sortie de la zone va droit au but en au plus 4 pas ; parmi eux le plus de boucles (retour sur une case
  quittée ≥ 3 déplacements plus tôt, stays retirés), puis le plus de pas dans la zone, puis la longueur la plus proche de
  20. **Choisie après avoir regardé les épisodes : c'est une illustration, pas un épisode typique.** Sur les 20 épisodes
  exposés depuis ce départ : 20 au but, 14 avec une sortie directe, 6 avec une sortie directe en ≤ 4 pas ; médiane de
  7,5 pas dans la zone (l'épisode tracé : 15). L'épisode typique reste dans E6a_trajectories (épisode 148, 20 pas).
- **Source** : `scratch/e6dev/results/E6_phenotypes_latent/e6__p2-b-d1_e4__20261023T134444Z/episodes.npz` (non
  versionné). **Oral** : « Même départ : l'exposé tourne en rond dans la zone où il a appris que ses actions ne
  servaient à rien, puis sort ; la référence rejoint le but en 6 pas. »
