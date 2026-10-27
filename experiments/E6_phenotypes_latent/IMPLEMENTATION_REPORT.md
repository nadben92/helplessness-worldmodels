# E6 et E7 — rapport d'implémentation (2026-10-23)

Branche `e6`, créée depuis `e5` (1d765d7). Rien n'est entraîné ; tout a tourné sur CPU, sur les poids d'E4 (les 4
configurations à 300 000 pas, exposés et références, extraits des archives `e4_wave{1,2}__20261021T071924Z.tar.gz`).
Décisions : `docs/e6/DECISIONS.md` (33)–(43), communes à E6 et E7. Design et statut : `docs/E6_phenotypes_latent.md`,
`docs/E7_latent_manipulation.md`. Sorties légères des runs de développement : `results/E6_phenotypes_latent/dev_e4_cpu/`
et `results/E7_latent_manipulation/dev_e4_cpu/` (analyses, figures, lignes par run ; les épisodes complets restent en local).

**Les conditions des agents de développement sont artificielles** (`dev_pair1` = p1-a, p1-b ; `dev_pair2` = p2-a, p2-b) :
les quatre sont des baselines d'E4. Les nuages « par condition » et les tests « sans une condition » ne servent ici qu'à
exercer le code.

## 1. Phase 0 — état des lieux

**1. Choix d'action** (`agents/worldmodel/policy.py`). Q(h, z, a) = moyenne sur K_act = 16 tirages du prior de
r + γ·c·V, avec h'_a = GRU(h, z, a) (`rssm.imagination_step`), z'_k ~ p(· | h'_a), r et c lus de la tête but
(r = 50·p − 1·(1 − p), c = 1 − p), V le critique, γ = 0,9. ε-greedy : avec probabilité ε une action uniforme ; sinon l'argmax
avec départage aléatoire parmi les actions à moins de la tolérance ABSOLUE du max (0,0751 = ½ × 6 × 0,9³⁵,
`heads.horizon_information_gap`). Flux : `explore` et `tiebreak` (numpy), `act_latent` (torch). **Il n'existe pas de mode
évaluation** : ε = 0,1 partout (config `agent.epsilon`, et `agent_config` de chaque checkpoint) ; les taux de succès
des checkpoints viennent des épisodes d'entraînement à ε = 0,1. E6 utilise ce ε (paramétrable).

**2. Début d'épisode** (`rssm.observation_core`, `closed_loop.ClosedLoopCollector._reset`). `is_first` remplace
(h, z, a) précédents par (h0 appris, 0, 0) avant le pas du GRU : h_1 = GRU(h0, 0, 0) est le même pour tout état, z_1 est
tiré du posterior q(z | h_1, e(x)). Départs d'entraînement : uniformes sur les 143 états non terminaux
(`collection.start_distribution: uniform_non_terminal`, `loop.draw_start_state`) — **les 9 états centraux de chaque
quadrant sont dans la distribution d'entraînement**.

**3. Localisation d'E4 (« 55–91 % contre 100 % »)** (`probing/probe_validity.reconstruction_diagnostic`). Le posterior
(h, z) du filtre est décodé ; deux règles : le gabarit le plus proche (la règle du noyau) et la cellule du pixel le plus
proche de la couleur de l'agent. Les chiffres sont la partie « zone » (25 pas dans la zone) de
`results/E4_generalization_gradient/final/tables/diagnose_*.csv`, image d'entraînement, zone UD : exposé 0,57 (p1-b),
0,76 (p1-a), 0,89 (p2-a), 0,61 (p2-b) ; référence 1,00 ; **au pas 1 dans la zone, les deux modèles sont à 1,00**. C'est un
décodage par contexte : appelable à chaque pas (un décodage par contexte). E6 l'appelle à chaque pas
(`probing/frozen_episodes.localise`, qui utilise `probe_validity.decoded_agent_cell`, extrait sans changement de calcul).

**4. b au pas 1 et validité** (`probing/fixed_trajectory.py`, `probe_validity.py`). Marches fixes : départ dans le couloir,
32 pas aléatoires dans le couloir, plus court chemin vers une case de bordure de la zone, entrée (pas 1). Au contexte
(h, z) du pas 1 : pour chaque action, h' = GRU(h, z, a), 16 tirages du prior, décodés et appariés aux gabarits (fantôme si
la marge < 1e-3), noyau p̂(s' | contexte, a), I_s = `controllability_of_kernel`. Moyenne par état puis par zone sur les
16 états de bordure ; b = I_exposé − I_référence. Validité (`is_valid`) : pour la RÉFÉRENCE, masse sur le vrai successeur
≥ 0,90 et |I_ref − log2 5| ≤ 0,15 au pas 1.

**5. RSSM** (`agents/worldmodel/rssm.py`). h_t = GRU(h_{t−1}, MLP([z_{t−1}, onehot(a_{t−1})])) ; prior p(z | h_t) =
softmax(MLP(h_t)) avec 1 % d'unimix ; posterior q(z | h_t, e(x_t)) = softmax(MLP([h_t, e_t])), même unimix ; 8 variables à
8 classes, h de dimension 64 (E4/E5), décodeur sur (h, z). Un `latent_hook` existe (identité) mais agit sur le h APRÈS le
GRU ; E7 corrige h_t avant le pas suivant (ci-dessous). **Mesure** (`scratch/e6p0/posterior_h.py`, les 8 modèles d'E4,
contextes = marches d'E4, 25 états de la zone UD) :

| | JS au posterior apparié, h d'un autre état | JS, h du premier pas | JS entre deux images voisines (même h) | variables gardant leur mode | bruit de 50 % de ‖h‖ |
|---|---|---|---|---|---|
| 8 modèles | 0,11–0,59 bit | 0,20–0,68 bit | 3,4–5,1 bits | 92–98 % | ≤ 0,05 bit |

Le posterior dépend surtout de l'image, et son mode identifie les 144 états (144 codes distincts au premier pas, les 8
modèles). **Mais le décodeur lit h** : le mode du posterior décodé avec le h propre de l'exposé dans la zone UD donne la
bonne position dans 52 % (p1-b), 61 % (p2-b), 78 % (p1-a), 93 % (p2-a) des cas, 100 % avec le h du premier pas ;
pour les références 100 % (50 % et 26 % avec le h d'un autre état pour p1-b et p2-b). Conséquences pour E7 ((40)) : la
cible calculée avec le h non modifié est peu biaisée ; la localisation sous intervention doit être mesurée sur le h
modifié — et c'est elle qui casse (section 6).

**Divergences entre le prompt et le code** (le code fait foi ; aucune ne bloque le design) :
- emplacement : `experiments/E6_phenotypes_latent/`, `experiments/E7_latent_manipulation/`, `docs/e6/DECISIONS.md` au lieu
  de `e6/`, `e7/`, `DECISIONS.md` ((33)) ; ce rapport est donc ici ;
- « z code la position de l'agent » : vrai pour le posterior (son mode identifie l'état), mais la position DÉCODÉE dépend
  aussi de h (point 5) ;
- « l'image thème seul » d'E4 colore la case NEUTRE en T_inc et garde T_inc en UD : sa zone d'intérêt est la case neutre,
  pas l'ancienne UD ((34)) ; « image neutre » = l'image d'entraînement avec la case neutre comme zone ;
- pas d'ε d'évaluation distinct (ε = 0,1) ;
- `submit_kubeflow.py` d'E5 refuse toute autre branche que `e5` : E6 a sa propre soumission (phases `e6`, `e7-directions`,
  `e7-apply`), dans `experiments/E6_phenotypes_latent/`.

## 2. Ce qui est implémenté

Bibliothèque (nouveaux modules ; un seul module existant touché, `probe_validity`, par extraction à l'identique) :

| Module | Rôle |
|---|---|
| `measures/zone_phenotypes.py` | les 6 phénotypes, Kaplan-Meier, courbe par indice, j_θ, bootstrap stratifié par départ (numpy) |
| `measures/monotone_fit.py` | isotone (PAVA), version lissée (PCHIP), inversion, bootstrap par agent, plage strictement monotone, tests sans un groupe |
| `interventions/latent_direction.py` | direction moyenne, cosinus « sans soi », nul isotrope, stabilité, efficacité (numpy) |
| `probing/frozen_episodes.py` | épisodes à poids gelés en lots, mesures à chaque pas, crochet d'intervention sur h |
| `probing/belief_steering.py` | cibles (posterior de la vraie image suivante), perte Σ_a KL(cible ‖ prior) + λ‖δh‖², Adam |
| `probing/context_walks.py` | marches de contexte vers un état de la zone (couloir aléatoire ou vers le but, entrée par chaque côté) |
| `pipeline/frozen_setup.py` | config en couches, fichier d'agents, images de test, zone et départs, graines, validité, lignes |

Orchestration : `run_e6.py`, `analyse_e6.py`, `batches.py`, `submit_kubeflow.py`, `export_batch.py` (E6) ;
`run_e7_directions.py`, `run_e7_apply.py`, `analyse_e7.py` (E7). Configs : `configs/e6/e6.yaml` (base E4),
`configs/e7/e7.yaml` (base E6), `configs/e6/agents_dev_e4.yaml`, `configs/e6/agents_e5.yaml` (**à confirmer** : 5 agents
sains à ce jour — baseline A, B ; f = 1 A, E, F — la graine B f = 1, verrouillée, est absente ; ajouter un agent = une
entrée).

Tests : `tests/measures/test_zone_phenotypes.py`, `test_monotone_fit.py`, `tests/interventions/test_latent_direction.py`,
`tests/e6/test_frozen_pipeline.py`, `tests/e6/test_e6_batches.py` (31 tests). **Suite complète : 1 103 passés, 5 sautés,
3 xfail** (82 min, CPU) ; les tests de sondes d'E4 et l'identité des sondes d'E5 passent après l'extraction.

## 3. Vitesse de récupération comportementale (fixée avant toute mesure, (35))

Courbe : la part de pas dirigés au j-ième pas du PREMIER séjour dans la zone, sur les épisodes encore dans la zone (leur
nombre est rapporté pour chaque j). Résumé : **j_θ, le premier indice où cette part atteint θ**, interpolé linéairement,
lu sur les indices d'au moins 20 épisodes ; **θ = mi-chemin entre le hasard (2/5 : dans un quadrant, deux actions sur cinq
rapprochent du but) et le plafond ε-greedy (1 − ε + ε·2/5)**, soit 0,67 à ε = 0,1. Censuré s'il n'est jamais atteint
(dans les courbes : au premier indice non lisible, marqué). Pourquoi : une pente sur les k premiers pas n'est pas monotone
dans la récupération (une référence au plafond a une pente nulle, comme un agent qui ne récupère jamais) et elle est
biaisée par la survie (les épisodes encore dans la zone à grand j sont les non dirigés) ; un seuil ne lit que les indices
jusqu'au franchissement, où la plupart des épisodes sont encore là, et ses deux ancres sont interprétables et fixées par ε.

## 4. Tests de forme

| Test | Attendu | Observé (poids d'E4) |
|---|---|---|
| E6, image d'entraînement, exposé baseline : b au pas 1 | ≈ −2,3 | −2,06 à −2,17 (départ au centre) ; −2,30 à −2,32 par la sonde d'E4 (écart en section 5) |
| idem : temps de sortie plus long que la référence | oui | **à peine** : médiane KM 5–6 contre 5 (voir ci-dessous) |
| idem : part de pas dirigés plus faible | oui | **oui** : 0,32–0,49 contre 0,94 ; j_θ censuré pour les 4 exposés, 1 pour les références |
| E6, référence : b ≈ 0 et dirigée sur toutes les images valides | oui | oui : b = 0 par construction ; dirigé 0,93–0,95 sur les 14 couples valides |
| E7, α = 0 = E6 exactement | identité | **oui** : 24/24 couples (agent, image), b et tous les phénotypes identiques (`identity_alpha0.csv`) |
| E7, contrôle aléatoire compatible avec α = 0 | oui | **non** pour l'exposé : une direction aléatoire de même norme déplace b autant que v (section 6) ; oui pour la référence à α ≤ 0,5 |
| Reproductibilité ; mêmes départs et tirages ε pour l'exposé et la référence | oui | oui (tests unitaires ; flux par épisode, (36)) |

Le temps de première sortie discrimine mal **par géométrie** : le plus court chemin depuis le centre de la zone vers le but
longe l'intérieur du quadrant jusqu'au coin le plus proche du but, donc un agent dirigé ne sort qu'à d − 1 pas
(médiane 5 pour les 9 départs), à peu près comme un marcheur aléatoire qui sort par n'importe quel bord. La part dirigée et
j_θ, eux, séparent nettement.

## 5. Résultats préliminaires d'E6 (poids d'E4, 4 agents × 4 images × 2 rôles)

**Validité.** 14 couples (agent, image) valides sur 16 ; invalides : p2-a position seule (masse de la référence 0,01,
I_ref 1,60) et p2-a thème seul (0,22, 1,94) — la couleur quasi noire de p2-a, déjà signalée dans E4.

**Cohérence de b au pas 1 avec la sonde d'E4** (`coherence.csv`). Image d'entraînement : écart (E6 − sonde) +0,14 à +0,25
bit ; position seule : +0,14 à +0,28 ; thème seul et neutre : −0,03 à +0,16. **Écart systématique** : partir sans contexte
au centre de la zone donne un b moins négatif que d'arriver par la bordure après 32 pas de couloir. Signalé, non corrigé.

**Nuages** (`results/E6_phenotypes_latent/dev_e4_cpu/analysis/`, figures `cloud_*.png`). Les points valides sont **en deux
amas, b ≈ −2,1 (exposés : entraînement et position seule) et b ≈ 0 (exposés : thème seul et neutre ; toutes les
références)** : aucun point entre −1,9 et −0,13 (plus grand écart sans point : 1,76 bit). Les courbes y sont donc de
l'interpolation (en tirets sur les figures), et les plages « inversibles » ci-dessous ne valent que là où il y a des points.

| Phénotype | sens | ρ de Spearman [IC 95 % agents] | plage strictement monotone | résidus × localisation (pas 1) |
|---|---|---|---|---|
| temps de sortie (médiane KM) | décroissant | −0,50 [−0,80 ; −0,19] | −2,17 à 0,06 | −0,33 |
| part dirigée | croissant | 0,75 [0,69 ; 0,87] | −2,14 à 0,06 | 0,33 |
| j_θ (7 points censurés) | décroissant | −0,80 [−0,81 ; −0,78] | −2,17 à −0,24 | 0,08 |
| part des pas dans la zone | croissant | 0,63 [0,54 ; 0,79] | −2,00 à 0,06 | 0,35 |
| part de « stay » | décroissant | −0,78 [−0,90 ; −0,67] | −1,96 à 0,06 | −0,04 |
| part de retours (k = 4) | décroissant | −0,74 [−0,78 ; −0,66] | −1,93 à 0,06 | −0,07 |

À b ≈ −2,1, la part dirigée varie de 0,32 (p1-a) à 0,49 (p1-b) entre agents pour des b à 0,1 bit près : dans l'amas
helpless, le comportement n'est pas une fonction de b seul sur ces quatre agents. Test à l'aveugle sans un agent (part
dirigée) : erreur absolue moyenne sur b de 0,05 à 0,21 bit ; sans une condition : 0,07 et 0,11. Exclure les 2 points valides mal
localisés (< 0,9 au pas 1 : p1-a exposé, position seule et thème seul, 0,72) change la courbe de la part dirigée d'au plus 0,09.
Courbe d'accompagnement : b_local de l'exposé p1-b à l'entraînement passe de −2,19 (pas 1) à −1,77 (pas 3) puis reste vers
−1,75 ; la part dirigée par indice reste à 0,46–0,55 : la croyance remonte un peu, le comportement non.

## 6. Résultats préliminaires d'E7

**Corrections et direction commune** (`run_e7_directions`, 10 contextes par état, λ = 0,01 ; image de direction :
entraînement). Situations localisées par l'exposé dans la zone UD : 154–232 sur 250 (le « 55–91 % » d'E4).

| Agent | tâche | ‖δh‖ moyen | cos « sans soi » (nul : moyenne, p95) | efficacité (moitié test) | ΔI direction / ΔI aléatoire même norme | stabilité (cos entre redémarrages) |
|---|---|---|---|---|---|---|
| p1-a | avant (exposé) | 11,0 | 0,17 (0,00 ; 0,02) | 0,36 | +0,55 / +0,47 | 0,22 |
| p1-b | avant | 11,4 | 0,21 (0,00 ; 0,02) | 0,28 | +0,53 / +0,31 | 0,27 |
| p2-a | avant | 10,2 | 0,28 (0,00 ; 0,02) | 0,35 | +0,55 / +0,46 | 0,15 |
| p2-b | avant | 13,0 | 0,18 (0,00 ; 0,03) | 0,43 | +0,71 / +0,69 | 0,19 |
| p1-a | inverse (référence) | 5,6 | 0,44 (0,00 ; 0,03) | 0,58 | −1,24 / −0,30 | 0,62 |
| p1-b | inverse | 5,4 | 0,51 (0,00 ; 0,03) | 0,72 | −1,58 / −0,12 | 0,67 |
| p2-a | inverse | 4,6 | 0,57 (0,00 ; 0,03) | 0,74 | −1,46 / −0,11 | 0,61 |
| p2-b | inverse | 5,7 | 0,54 (0,00 ; 0,03) | 0,70 | −1,50 / −0,34 | 0,69 |

‖h‖ ≈ 5–6,5. Les corrections avant sont deux fois plus grandes que h ; elles amènent I de 0,05–0,24 à 1,85–2,00 bit,
mais l'optimum n'est pas unique (cos 0,15–0,27 entre redémarrages) et **une direction aléatoire de même norme déplace I
presque autant que v**. Les corrections inverses sont plus petites, plus stables, et nettement plus efficaces qu'une
direction aléatoire. Je ne tranche pas l'existence de la direction.

**Application** (image d'entraînement, moyenne sur les 4 agents ; tableau complet `checks.csv`) :

| Mode, α | b au pas 1 | part dirigée | médiane de sortie | localisation pas 1 / zone | masse sur le successeur joué | p_goal |
|---|---|---|---|---|---|---|
| sans intervention | −2,13 | 0,38 | 5,5 | 0,97 / 0,79 | 0,23 | 0,00 |
| direction 0,25 / 0,5 / 1 | −1,82 / −1,33 / −1,65 | 0,46 / 0,43 / 0,38 | 6,0 / 7,3 / 7,5 | 0,79 / 0,16 / 0,02 | 0,09 / 0,03 / 0,02 | 0,09 / 0,54 / 0,73 |
| aléatoire 0,25 / 0,5 / 1 | −1,94 / −1,49 / −1,22 | 0,40 / 0,43 / 0,45 | 5,8 / 4,8 / 5,5 | 0,93 / 0,61 / 0,09 | 0,15 / 0,09 / 0,02 | 0,00 / 0,04 / 0,29 |
| pas à pas 0,25 / 0,5 / 1 | −2,10 / −1,77 / −0,31 | 0,47 / 0,54 / 0,58 | 6,5 / 4,5 / 3,5 | 0,97 / 0,69 / 0,33 | 0,18 / 0,20 / 0,68 | 0,00 / 0,00 / 0,02 |
| référence, inverse 0,25 / 0,5 / 1 | −0,43 / −0,93 / −1,54 | 0,88 / 0,45 / 0,35 | 4,0 / 6,8 / 7,8 | 0,85 / 0,43 / 0,08 | 0,64 / 0,05 / 0,01 | 0,05 / 0,58 / 0,72 |
| référence, aléatoire 0,25 / 0,5 / 1 | −0,07 / −0,25 / −0,55 | 0,94 / 0,93 / 0,84 | 4,8 / 4,5 / 5,0 | 0,99 / 0,94 / 0,58 | 0,99 / 0,89 / 0,47 | 0,00 / 0,00 / 0,06 |

**Vérifications.** (1) La croyance bouge : oui, dans tous les modes. (2) **Rien d'autre ne doit bouger : échoue** pour la
direction et l'aléatoire dès α = 0,5 — la localisation s'effondre et p_goal monte jusqu'à 0,73 (le modèle « se croit » au
but) ; le pas à pas à α = 1 garde p_goal ≈ 0 et fait passer la masse sur le successeur joué de 0,23 à 0,68 (la prédiction
devient juste), mais la localisation descend à 0,33. (3) Contrôle aléatoire : pour l'exposé, il bouge b comme la direction
(non spécifique) ; pour la référence, il bouge b quatre à six fois moins que la direction inverse. (4) Sens inverse :
à α = 0,25 seulement, b passe à −0,43 avec une localisation à 0,85 et le comportement reste dirigé (0,88) ; au-delà, même
effondrement. Sur l'image neutre (b ≈ 0), le pas à pas ne fait presque rien (‖δ‖ ≤ 2) et ne change pas le comportement.

**Les points d'E7 ne suivent pas la courbe d'E6** (`residuals.csv`) : sur la part dirigée, RMSE des résidus 0,17 (direction),
0,19 (aléatoire), 0,14 (pas à pas), 0,18 (inverse), contre 0,04 pour les points d'E6 ; les points d'E7 sont sous la courbe
(résidu moyen −0,10 à −0,12 en direction, aléatoire et inverse ; −0,03 en pas à pas). Ces points violent la vérification (2),
donc ils ne mesurent pas encore l'effet de la croyance seule.

**Ce que cela suggère (non appliqué : c'est un choix de design).** λ = 0,01 laisse des corrections de norme ≈ 2‖h‖. Balayage
sur p1-b, image d'entraînement (`scratch/e6p0/lambda_sweep.py`, 154 situations) :

| λ | ‖δh‖ | I avant → après | localisation après |
|---|---|---|---|
| 0,01 | 11,2 | 0,05 → 1,99 | 0,60 |
| 0,1 | 7,3 | 0,05 → 1,89 | 0,71 |
| 1 | 4,2 | 0,05 → 1,22 | 0,93 |
| 10 | 1,0 | 0,05 → 0,34 | 0,99 |

Deux options, à décider : un λ plus grand (1 garde la localisation à 0,93 en déplaçant I de 1,2 bit), ou un terme de perte
qui conserve explicitement la position décodée et p_goal (une perte sur le posterior/décodage de l'état courant, pas sur
les pixels).

## 7. Coût mesuré et ce qui tient sur CPU

Mesures sur CPU (Windows, 1 thread par processus, **8 processus en parallèle** sur 12 cœurs pour E6 et les directions,
4 pour l'application ; le premier essai seul : 67 s d'épisodes, 30 s de I frais et 130 s de sonde de validité pour une image) :

| Phase | par agent (4 images) | dont |
|---|---|---|
| e6 | 1 630–1 910 s | sonde de validité ≈ 50 % (150–365 s par image), épisodes 35–200 s, I frais 30–90 s |
| e7-directions | 580–650 s | les redémarrages de stabilité ≈ 40 % |
| e7-apply | 8 700–9 400 s | 86 jeux d'épisodes ; le pas à pas ≈ 2 × un mode fixe |

Pour 5 à 7 agents : e6 ≈ 3 h et e7-directions ≈ 1 h 15 en série sur un CPU, moins de 40 min et 15 min avec 8 processus ;
e7-apply ≈ 15–18 h en série, ≈ 4–5 h avec 4 processus. **Tout tient sur CPU en local** (aucune phase ne demande plus de
quelques centaines de Mo de mémoire). Le GPU n'a pas été mesuré ; les décodages par lots (la sonde de validité, I_s)
devraient en profiter, pas les boucles par épisode. Leviers sans effet sur les résultats d'E6 : `e6.measure: zone` dans
E7 (les comportements et b au pas 1 sont inchangés : flux torch séparés, (36)).

**Commandes exactes de dry-run** (depuis la racine du checkout `e6` ; sous Git Bash local préfixer `MSYS_NO_PATHCONV=1`,
sinon `/scratch` est réécrit) :

```bash
python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e6 --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e7-directions --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e7-apply --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
python experiments/E6_phenotypes_latent/export_batch.py create --batch e6_e5
```

Un job par agent (`--agents-per-job` pour grouper), 4 en parallèle sur le GPU ; `e7-apply` lit les sorties de
`e7-directions` dans `results/E7_latent_manipulation/` du checkout, donc se soumet après. Le clone `/scratch/helplessness-worldmodels-e6`
n'existe pas encore : il se crée comme celui d'E5 (`RUN_GPU.md` d'E5, étape 1, branche `e6`). Reproduire l'analyse locale :

```bash
python experiments/E6_phenotypes_latent/analyse_e6.py --runs "results/E6_phenotypes_latent/e6__*" --out <dir>
python experiments/E7_latent_manipulation/analyse_e7.py --e6-runs "results/E6_phenotypes_latent/e6__*" \
    --apply-runs "results/E7_latent_manipulation/e7apply__*" --directions-runs "results/E7_latent_manipulation/e7dir__*" --out <dir>
```

## 8. Points tranchés (détails dans `docs/e6/DECISIONS.md`)

1. Emplacement du code et des décisions selon CLAUDE.md, numérotation à la suite d'E5 ((33)).
2. Zone d'intérêt = la case que change la sonde ; « thème seul » → la case neutre ; « image neutre » = image
   d'entraînement, case neutre ((34)).
3. j_θ comme vitesse de récupération ((35)).
4. Flux par épisode à consommation fixe (mêmes tirages ε pour l'exposé et sa référence) ; mêmes graines pour tous les
   agents ; deux générateurs torch (action, mesure) ((36)).
5. b au pas 1 sur les mêmes départs ; les références dans les courbes avec b = 0 ; b_local contre le I « départ frais » de la
   référence au même état ((37)).
6. Validité et b de comparaison par la sonde d'E4 elle-même (150 marches, graines d'E4) ((38)).
7. Politique d'entraînement telle quelle, ε = 0,1, posterior échantillonné ((39)).
8. Intervention : z tiré du h non modifié, h modifié utilisé par les mesures, la politique et la récurrence ; direction
   estimée sur l'image d'entraînement, appliquée à la norme moyenne des corrections ; λ = 0,01 ; pas à pas limité à 300
   pas d'Adam ((41)) — **λ est à revoir (section 6)**.
9. Contextes d'E7 : 10 par état, entrée par chacun des quatre côtés, moitiés paire/impaire ((42)).
10. Analyse : seuil de stricte monotonie (fenêtre 0,25 bit, 2 % de l'étendue, 95 %), localisation à 0,9 ((43)).

**À décider par toi** : la liste des images de test (pour remplir l'écart entre −1,9 et −0,1 bit, il faut des images de
croyance intermédiaire : les sondes d'E5 diront lesquelles) ; la composition de `configs/e6/agents_e5.yaml` ; λ ou un terme de
conservation pour E7 ; l'existence de la direction commune.

---

# Corrections après le premier rapport (2026-10-23, suite)

Même branche `e6`. Décisions (44)–(50) dans `docs/e6/DECISIONS.md`. Sorties légères : `results/E6_phenotypes_latent/dev_e4_cpu_v2/`
et `results/E7_latent_manipulation/dev_e4_cpu_v2/` (analyses, figures, lignes et réglages par run). Toujours les 4 agents
d'E4 sur CPU, conditions artificielles. Le protocole d'E6 n'a pas changé : la relance donne les mêmes b et phénotypes.

## 9. E6 : phénotypes et localisation

**Phénotypes ((44)).** Principal : la part de pas dirigés. Secondaires : j_θ, part des pas dans la zone, « stay », retours.
Le temps de première sortie reste dans les lignes (`points.csv`), hors des courbes et des figures. Les rôles sont dans
`e6.analysis.phenotypes`.

**Localisation par couple** (`localisation_by_pair.csv` ; règle du gabarit le plus proche, moyenne sur les pas dans la zone) :

| Agent | entraînement : b / dirigé / loc. | position seule : b / dirigé / loc. | thème seul ; neutre : loc. |
|---|---|---|---|
| p1-a | −2,12 / 0,32 / 0,87 | −1,89 / 0,32 / 0,60 | 0,89 ; 1,00 |
| p1-b | −2,17 / 0,49 / 0,61 | −2,08 / 0,49 / 0,68 | 1,00 ; 1,00 |
| p2-a | −2,06 / 0,33 / 0,96 | invalide (0,10) | invalide (0,08) ; 1,00 |
| p2-b | −2,17 / 0,40 / 0,70 | −2,11 / 0,44 / 0,68 | 1,00 ; 1,00 |

**Comportement à croyance égale.** Sur les 14 couples exposés valides, la part dirigée et la localisation sont corrélées
(Pearson 0,74, Spearman 0,65), mais c'est l'effet des deux amas. **Parmi les 7 exposés à b < −1, la corrélation est
négative** (Pearson −0,53, Spearman −0,36) : p2-a, le mieux localisé (0,96), est le moins dirigé (0,33) ; p1-b, le plus mal
localisé (0,61), le plus dirigé (0,49).

**Pas à pas, dans chaque agent, les pas bien et mal localisés sont aussi dirigés les uns que les autres**
(`localisation_steps.csv`, image d'entraînement, bien / mal localisés) : p1-a 0,32 / 0,31 ; p1-b 0,50 / 0,47 ; p2-a 0,33 /
0,43 (47 pas mal localisés seulement) ; p2-b 0,40 / 0,39. Même chose en position seule. La localisation n'explique pas
l'écart de 0,32 à 0,49 à b ≈ −2,1.

**Nature des erreurs** (`localisation_errors.csv`, zone UD, exposés). Deux régimes, selon l'orientation de la configuration :

| Agent, image | erreurs | sur la case du tapis (depuis s_{t−1}) [IC 95 %] | parmi les voisines : sur la case du tapis (hasard 0,25) | région de la case décodée |
|---|---|---|---|---|
| p1-a, entraînement | 174 | 0,65 [0,58 ; 0,72] | 0,97 (n = 35) | UD 0,90, couloir 0,10 |
| p1-a, position seule | 410 | 0,24 [0,20 ; 0,29] | 0,43 (n = 30) | UD 0,98 |
| p2-a, entraînement | 36 | 0,46 [0,30 ; 0,62] | 1,00 (n = 5) | UD 0,97 |
| p1-b, entraînement | 427 | 0,02 [0,01 ; 0,04] | 1,00 (n = 2) | **CD 0,98** |
| p1-b, position seule | 365 | 0,13 [0,10 ; 0,17] | 1,00 (n = 24) | CD 0,81, UD 0,17 |
| p2-b, entraînement | 372 | 0,03 [0,02 ; 0,06] | 1,00 (n = 8) | **CD 0,96** |
| p2-b, position seule | 344 | 0,03 [0,02 ; 0,06] | 1,00 (n = 6) | CD 0,96 |

- Orientation **a** (T_inc = couleur a) : les erreurs restent dans la zone UD, et la plupart tombent là où le tapis
  roulant aurait poussé l'agent (un rang plus bas). Parmi les erreurs sur une voisine, c'est presque toujours la case du
  tapis, loin au-dessus du hasard (0,97 contre 0,25 pour p1-a).
- Orientation **b** : l'agent est dessiné dans le quadrant CD, à des cases variées (ce n'est pas la case homologue :
  2 %). C'est un saut de quadrant, pas une dérive.
- La mesure par région et la case homologue ont été ajoutées après avoir vu les erreurs lointaines de p1-b : c'est
  post hoc, et signalé comme tel ((45)).

## 10. E6 : séparation de l'écart de b ((46))

b au premier pas depuis l'état initial, au centre (9 états) et sur la bordure (16 états d'entrée d'E4), avec la même
fonction pour les deux modèles. Effet de l'endroit = bordure − centre ; effet de l'historique = sonde d'E4 − bordure.

| Agent, image | b centre (sans contexte) | b bordure (sans contexte) | b sonde d'E4 (bordure, après le couloir) | endroit | historique |
|---|---|---|---|---|---|
| p1-a, entraînement | −2,11 | −2,07 | −2,30 | +0,03 | −0,23 |
| p1-a, position seule | −1,84 | −1,68 | −2,17 | +0,16 | −0,49 |
| p1-b, entraînement | −2,22 | −2,21 | −2,31 | +0,01 | −0,10 |
| p1-b, position seule | −2,00 | −2,10 | −2,30 | −0,10 | −0,21 |
| p2-a, entraînement | −2,04 | −2,05 | −2,31 | −0,01 | −0,26 |
| p2-b, entraînement | −2,15 | −2,14 | −2,32 | +0,01 | −0,18 |
| p2-b, position seule | −2,21 | −2,11 | −2,25 | +0,10 | −0,14 |

À la zone UD, **l'écart vient surtout de l'historique** : 32 pas de couloir rendent b plus négatif de 0,10 à 0,49 bit.
L'endroit pèse de −0,10 à +0,16, sans signe constant. Sur thème seul et neutre (b ≈ 0), les deux effets sont de l'ordre de
±0,2 bit, sans signe constant. Rien n'est corrigé.

## 11. E7, A : la correction contrainte

**Lecture différentiable de la position ((47)).** La position douce est lue dans l'image décodée : un softmax sur les
cellules de −(distance minimale de leurs pixels à la couleur de l'agent) / τ, avec τ = 0,02. Elle lit le décodeur, comme
la mesure de localisation, et ne demande aucun entraînement. Son argmax est exactement la règle « cellule de l'agent »
d'E4 (testé).

Sur les épisodes d'E6, cette règle et celle du gabarit le plus proche (celle des vérifications) ont des précisions qui
diffèrent d'au plus 0,05 sur les 14 couples valides. Elles divergent sur les deux images invalides de p2-a, où la règle
du gabarit échoue (comme dans E4). Pourquoi pas une sonde linéaire entraînée sur les références : elle lirait le code des
références, pas ce que dessine le décodeur de l'exposé.

**Tests de forme.**
- **μ_pos = μ_goal = 0 et λ = 0,01 reproduisent le premier rapport** : écart max 0,0 sur les corrections des 4 agents
  (mêmes contextes, `reproduce_first_report`).
- **α = 0 reproduit E6** en A comme en B : 40 couples sur 40 identiques (`identity_alpha0.csv`).
- **Contrôle aléatoire compatible avec 0 pour les réglages retenus : non** pour λ ≤ 0,1 (gain aléatoire +0,15 à
  +0,46 bit, IC au-dessus de 0). À λ ≥ 1, il ne l'est qu'en partie : IC contenant 0 pour p1-a et p2-b, mais +0,10
  [+0,04 ; +0,17] pour p1-b et p2-a.
- **Induction monotone en α : partiellement** (section 12).

**Balayage** (`run_e7_constrained.py` ; 100 situations par agent, image d'entraînement, toutes corrigées, qu'elles soient
localisées ou non ; figure `constrained_tradeoff.png`, tableau `constrained_settings.csv`). μ_goal est calibré par agent :
10 pour p1-a et p1-b, 0 pour p2-a et p2-b, où p_goal ne bouge pas. Avec ce réglage, max |Δp_goal| ≤ 0,05 dans tous les
réglages.

| Agent (loc. sans intervention ; référence 1,00) | λ = 0,01, μ_pos = 0 : gain / loc. | λ = 0,01, μ_pos = 1–10 | λ = 0,1, μ_pos = 1 | λ = 1, μ_pos = 1 | λ = 10 | gain aléatoire à λ = 0,01 / 1 |
|---|---|---|---|---|---|---|
| p1-a (0,76) | +1,55 / 0,27 | +1,52 / 0,74 | +1,25 / 0,79 | +0,51 / 0,82 | +0,05 / 0,80 | +0,46 / +0,03 |
| p1-b (0,60) | +1,65 / 0,46 | +1,65 / 0,59 | +1,50 / 0,64 | +0,91 / 0,73 | +0,21 / 0,63 | +0,36 / +0,10 |
| p2-a (0,90) | +1,67 / 0,31 | +1,66 / 0,89 | +1,49 / 0,86 | +0,76 / 0,92 | +0,17 / 0,92 | +0,44 / +0,10 |
| p2-b (0,66) | +1,48 / 0,41 | +1,50 / 0,67 | +1,39 / 0,73 | +0,88 / 0,70 | +0,17 / 0,67 | +0,43 / +0,07 |

- **Le terme de position fait ce qu'on lui demande.** Sans lui (μ_pos = 0), la localisation tombe (0,27 à 0,46 pour
  λ = 0,01). Avec μ_pos = 1 ou 10, elle revient au niveau sans intervention pour le même gain de b (+1,5 à +1,7 bit).
- **Réglages retenus : 9 à 13 sur 16 selon l'agent.** Un réglage est retenu si sa localisation n'est pas inférieure à
  celle sans intervention moins 0,02, et si son gain est spécifique (IC de l'écart au contrôle au-dessus de 0, et écart
  ≥ ½ gain).
- **La spécificité dépend de λ.** Le gain spécifique (gain moins contrôle) va de +1,0 à +1,3 bit à λ ≤ 0,1. Mais le
  contrôle aléatoire de même norme y gagne encore 0,15 à 0,46 bit : une grande perturbation de h fait déjà monter I.

**Application** ((48) : le réglage retenu au gain spécifique le plus grand — p1-a λ 0,1 μ 1 ; p1-b λ 0,01 μ 1 ; p2-a
λ 0,01 μ 10 ; p2-b λ 0,1 μ 1). Image d'entraînement :

| Agent | sans : b / dirigé / loc. pas 1, zone / p_goal | direction contrainte | pas à pas contraint |
|---|---|---|---|
| p1-a | −2,12 / 0,32 / 0,97, 0,87 / 0,00 | −0,90 / 0,48 / 0,00, 0,00 / 0,06 | −0,72 / 0,44 / 0,85, 0,63 / 0,00 |
| p1-b | −2,17 / 0,49 / 0,98, 0,61 / 0,00 | −1,29 / 0,41 / 0,11, 0,05 / 0,93 | −0,37 / 0,71 / 0,88, 0,79 / 0,00 |
| p2-a | −2,06 / 0,33 / 0,94, 0,96 / 0,00 | −1,54 / 0,39 / 0,09, 0,01 / 0,36 | −0,15 / 0,61 / 0,94, 0,87 / 0,00 |
| p2-b | −2,17 / 0,40 / 1,00, 0,70 / 0,00 | −1,16 / 0,41 / 0,00, 0,04 / 0,97 | −0,28 / 0,48 / 0,91, 0,88 / 0,00 |

- **La direction contrainte (v moyen) casse encore tout pendant les épisodes** : la localisation tombe à 0,00–0,11,
  p_goal monte jusqu'à 0,97, et cela même sur les images à b ≈ 0. La contrainte vaut situation par situation, pas pour
  leur moyenne.
- **Le pas à pas contraint tient presque toutes les vérifications sur l'image d'entraînement** : p_goal = 0,
  localisation au pas 1 de 0,85 à 0,94, et la masse sur le successeur joué passe de 0,2 à 0,4–0,8. Seule exception, la
  localisation dans la zone chez p1-a : 0,63 contre 0,87 sans intervention (les autres agents restent à 0,79–0,88).
  b remonte de 1,4 à 1,9 bit.
- **Mais la part dirigée ne remonte qu'en partie** : 0,32 → 0,44, 0,49 → 0,71, 0,33 → 0,61, 0,40 → 0,48 (les références
  sont à 0,94).

## 12. E7, B : l'induction chez les références ((49))

Direction inverse (vers le tapis roulant), α ∈ {0 ; 0,1 ; 0,2 ; 0,3 ; 0,4 ; 0,5 ; 0,75 ; 1}, images à zone UD, contrôle
aléatoire de même norme (`summary.json`, clé `induction`). Image d'entraînement, α de 0,1 à 1 (en gras, le premier α où la
localisation casse) :

| Agent | b | part dirigée | loc. dans la zone | p_goal |
|---|---|---|---|---|
| p1-a | −0,05, −0,22, **−0,41**, −0,60, −0,77, −0,89, −1,90 | 0,94, 0,93, **0,54**, 0,42, 0,41, 0,40, 0,41 | 0,99, 0,97, **0,28**, 0,04, 0,01, 0,00, 0,00 | 0, 0, **0,48**, 0,79, 0,87, 0,95, 0,97 |
| p1-b | −0,03, −0,28, **−0,68**, −1,02, −1,26, −1,40, −1,82 | 0,94, 0,93, **0,67**, 0,46, 0,39, 0,33, 0,36 | 1,00, 0,99, **0,51**, 0,15, 0,12, 0,08, 0,05 | 0, 0, **0,28**, 0,69, 0,78, 0,88, 0,94 |
| p2-a | −0,06, −0,24, **−0,49**, −0,53, −0,62, −1,01, −1,12 | 0,94, 0,95, **0,93**, 0,73, 0,52, 0,37, 0,28 | 0,99, 0,85, **0,37**, 0,10, 0,06, 0,06, 0,00 | 0, 0, 0, 0, 0, 0,01, 0,08 |
| p2-b | −0,06, −0,39, −0,67, **−0,88**, −1,05, −1,76, −1,33 | 0,94, 0,94, 0,93, **0,58**, 0,48, 0,37, 0,36 | 1,00, 1,00, 0,99, **0,34**, 0,15, 0,04, 0,02 | 0, 0, 0, **0,39**, 0,66, 0,84, 0,90 |

- **b décroît avec α** : monotone sur 4 des 8 couples (agent, image). Sur les autres, il remonte aux grands α, quand la
  localisation s'est déjà effondrée (p2-a position seule : la référence y est invalide).
- **Tant que la localisation tient, b descend jusqu'à −0,67 bit (p2-b, α = 0,3) sans aucun changement de comportement**
  (part dirigée 0,93).
- **Le comportement change avec la localisation.** Chez p1-a et p1-b, la part dirigée chute au même α que la localisation,
  et p_goal monte en même temps. Chez p2-a, la localisation casse dès α = 0,3 (0,37), mais le comportement ne chute qu'à
  α = 0,4, sans que p_goal bouge. Chez p2-b, il chute à α = 0,4 avec la localisation et p_goal.
- Le contrôle aléatoire de même norme fait baisser b deux à cinq fois moins et laisse le comportement intact jusqu'à
  α ≈ 0,5.

## 13. Superposition aux courbes d'E6

Figure `e7_on_e6_directed_share.png` : carrés pour A, triangles pour B. Résidus de la part dirigée par rapport à la courbe
principale d'E6 (RMSE des points d'E6 : 0,04). Les points sont séparés selon qu'ils passent les vérifications ou non
(localisation dans la zone ≥ celle sans intervention − 0,02, et |Δp_goal| ≤ 0,05) :

| Groupe | passe | n | b | part dirigée | RMSE résidu | résidu moyen |
|---|---|---|---|---|---|---|
| A, direction contrainte | non | 14 | −2,21 à −0,57 | 0,38–0,57 | 0,18 | −0,15 |
| A, pas à pas contraint | oui | 9 | −0,37 à +0,19 | 0,47–0,95 | 0,21 | −0,11 |
| A, pas à pas contraint | non | 5 | −0,90 à +0,05 | 0,39–0,94 | 0,25 | −0,22 |
| B | oui | 12 | −0,67 à −0,03 | 0,93–0,94 | 0,05 | +0,03 |
| B | non | 37 | −2,22 à −0,22 | 0,28–0,95 | 0,22 | −0,17 |
| B aléatoire | oui / non | 33 / 16 | −1,00 à +0,01 | 0,64–0,95 | 0,02 / 0,06 | +0,01 / +0,02 |

**Lecture.**
- Les points A qui passent les vérifications sont sous la courbe : le comportement remonte moins que b.
- Les points B qui passent sont sur la courbe, mais entre −0,67 et −0,13 bit la courbe d'E6 n'est que de
  l'interpolation (aucun point d'E6, section 5). Ces points disent seulement que b peut baisser de 0,7 bit sans changer
  le comportement.
- Hors vérifications, A et B sont sous la courbe, et le comportement y chute en même temps que la localisation (et,
  sauf chez p2-a, que p_goal).
- Sur ces 4 agents, aucun point ne montre le comportement suivre b quand rien d'autre ne bouge.

## 14. Coût des nouvelles phases

Mesures sur CPU, 1 thread par processus, 8 processus en parallèle sur 12 cœurs :

| Phase | par agent |
|---|---|
| e6 (avec le b frais en bordure et au centre, et les positions décodées) | 1 480–1 980 s (≈ celui du premier rapport) |
| e7-constrained (16 réglages, calibration de μ_goal, reproduction ; 100 situations) | 3 530–4 400 s : ≈ 280–400 s par réglage avec μ_pos > 0, ≈ 30 s sans |
| e7-apply, plan A (meilleur réglage, 4 images) | ≈ 3 900–4 500 s, dont 600–960 s par image pour le pas à pas contraint |
| e7-apply, plan B (2 images × 7 α × 2 modes) | ≈ 3 400–4 300 s |

Le terme de position décode l'image à chaque pas d'Adam : c'est lui qui coûte. Tout tient encore sur CPU : pour 5 à 7
agents, compter environ 4 à 5 h avec 4 à 8 processus. Le GPU devrait aider surtout ici (décodages en lots).

Dry-runs (préfixer `MSYS_NO_PATHCONV=1` sous Git Bash) :

```bash
python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e6 --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e7-directions --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e7-constrained --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e7-apply --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
```

Ordre : e6 et e7-directions, puis e7-constrained, puis e7-apply (plans A et B par défaut, `e7.apply_plans`).
`configs/e6/agents_e5.yaml` ne contient pour l'instant que `base_A` (les autres en commentaire, TODO).

## 15. Points tranchés

1. **Lecture de la position** : position douce lue dans l'image décodée, τ = 0,02, plutôt qu'une sonde linéaire ((47)).
2. **Le balayage corrige toutes les situations**, localisées ou non. La référence « sans intervention » est ainsi la
   localisation propre de l'exposé (< 1). La restriction aux situations localisées ((42)) reste pour l'estimation de la
   direction du premier rapport.
3. **4 contextes par état pour le balayage**, à cause du coût. La reproduction du premier rapport se fait sur ses 10
   contextes.
4. **« Nettement supérieur » au contrôle aléatoire** : IC 95 % de l'écart au-dessus de 0, et écart ≥ ½ gain.
5. **μ_goal** : le plus petit de la grille qui garde max |Δp_goal| ≤ 0,05 au réglage le plus permissif, par agent.
6. **Plan A** : seul le meilleur réglage retenu est appliqué, pour le coût (`apply_settings: all` sinon), avec α ∈ {0 ; 1}.
7. **Plan B** : la direction inverse non contrainte du premier rapport, sur une grille d'α plus fine.
8. **Nature des erreurs** : la case du tapis est celle où l'état PRÉCÉDENT aurait mené sans contrôle, et les pas où
   l'agent a bougé comme le tapis sont exclus. Région et case homologue : post hoc.
9. **Agents** : la relance du run f = 1 échoué correspond à deux runs dans (32), E et F. Les deux sont listés en
   commentaire ; à toi de garder celui ou ceux qui ont appris.

**À décider par toi**
- **Le terme de position ne suffit pas pendant les épisodes.** En mode direction, la contrainte par situation ne passe pas
  à leur moyenne. En pas à pas, elle tient presque partout, mais le comportement ne suit b qu'en partie. Faut-il
  contraindre aussi le pas suivant (h_{t+1}) ?
- **Le réglage de référence pour la suite** : un λ ≥ 1 rapproche le contrôle aléatoire de 0, mais le gain est plus petit
  (+0,5 à +0,9 bit).
- **Les images de test intermédiaires**, toujours nécessaires pour lire la courbe d'E6 entre −1,9 et −0,1 bit.

---

# Diagnostics : valeur en cache, règle de décision, anticipation longue (2026-10-26)

## 16. Diagnostics

Même branche `e6`, mêmes 4 agents d'E4 sur CPU, poids gelés (hash des state dicts avant et après chaque run :
identiques, 16 runs sur 16). Image d'entraînement, environnement de test tout contrôlable ; zone AI = l'ancienne case UD.
Décisions (51)–(54) ; code : `probing/lookahead.py`, `measures/decision_rule.py`, options `tau` et `q_fn` de
`run_episodes` (désactivées par défaut, testées), `run_e7_diagnostics.py`, `analyse_e7_diagnostics.py`. Sorties
légères : `results/E7_latent_manipulation/dev_e4_cpu_diag/` (tableaux `d1_*` à `d4_*`, figures, `summary.json`).

**Phase 0 ((51)).** Le critic est entraîné **uniquement en imagination, à travers le prior** : à chaque pas de gradient
après la bascule, depuis chaque état posterior du batch du replay, 15 pas imaginés (`imagine` → `rssm.imagination_step`),
λ-returns avec λ = 0,95, γ = 0,9, cible = critic lent (EMA 0,02), poids = produit des continuations
(`agents/worldmodel/imagination.py`). Aucun retour réel n'entre dans la cible. Les trajectoires imaginées suivent la
politique de l'agent : ε-greedy (ε = 0,1) sur le Q à un pas calculé avec K_imag = 4 tirages du prior et le critic en
cours d'entraînement. Récompense et continuation viennent de la tête but : r = 50 p − (1 − p), c = 1 − p. Q en épisode
(`policy.q_values`) : moyenne sur 16 tirages du prior de r + γ c V au latent prédit, puis ε-greedy avec tie-break dans
une tolérance absolue de 0,0751. H1 garde donc sa forme d'origine.

**Formes attendues** : écrites et commitées avant tout calcul ((53), commit 5e1c483).

**Statut méthodologique.** Le phénotype d'E6 reste celui mesuré avec la politique d'entraînement (k = 1, ε-greedy).
L'anticipation longue et le softmax sont des manipulations de la règle de décision au test, utilisées pour séparer la
croyance de transition de la valeur en cache. Ce ne sont pas de nouveaux phénotypes. Le critic a appris sous ε-greedy :
le softmax est une règle de lecture, pas la politique d'entraînement.

### D1 — la valeur de l'exposé pointe-t-elle vers le but ? (H1)

Figure `d1_value_maps.png` ; `d1_values.csv`. V au pas 1 d'un épisode démarré sur la case (8 tirages) ; Q à
4 contextes par case de la zone (100 contextes).

| Agent | ρ(V, −distance), 143 cases : exp. / réf. | idem dans la zone AI | V moyen dans la zone | Q oracle : choix greedy dirigé, exp. / réf. | Q du modèle (= politique) : exp. / réf. |
|---|---|---|---|---|---|
| p1-a | 0,75 / 0,98 | 0,17 / 0,98 | 9,1 / 24,1 | 0,75 / 1,00 | 0,49 / 0,96 |
| p1-b | 0,75 / 0,98 | 0,15 / 0,97 | 8,7 / 24,4 | 0,73 / 1,00 | 0,44 / 0,98 |
| p2-a | 0,74 / 0,98 | 0,22 / 0,97 | 8,7 / 24,4 | 0,77 / 1,00 | 0,57 / 1,00 |
| p2-b | 0,74 / 0,98 | 0,20 / 0,99 | 8,6 / 24,2 | 0,85 / 1,00 | 0,40 / 0,99 |

- **Forme de la référence : tenue.** V monte vers le but (ρ = 0,98) et le Q oracle choisit une action dirigée sur toutes
  les cases.
- **Chez l'exposé, V est plate et basse dans la zone AI** (ρ = 0,15–0,22 ; V ≈ 9 contre 24) : c'est une trace de la
  croyance passée. L'argmax strict donne les mêmes parts à 0,01 près.
- **Verdict : ambigu selon le critère écrit avant.** L'exposé oracle est entre les deux seuils fixés (0,73–0,85 :
  au-dessus de 0,6, mais plus de 0,1 sous la référence). La valeur en cache coûte 0,15 à 0,27 de choix dirigés à un pas.
  Remplacer la seule croyance de transition fait passer l'exposé de 0,40–0,57 à 0,73–0,85 : la croyance de
  transition pèse plus que la valeur en cache.

### D2 — l'argmax ignore-t-il une croyance partielle ? (H2)

Figure `d2_gap_directed_vs_b.png` ; `d2_bins.csv`. Pas bien localisés dans la zone : 42 562, issus d'E6 (exposé et
référence, rejoués à l'identique : 8 sur 8 identiques à E6) et de B (reverse et reverse_random, 7 α). b = b_local,
tranches de 0,25 bit, n ≥ 20.

| b (tranche) | ≥ 0 | −0,25 | −0,5 | −0,75 | −1,0 | −1,25 | −1,5 | −1,75 | −2,0 | −2,25 |
|---|---|---|---|---|---|---|---|---|---|---|
| Références : part dirigée du choix greedy | 1,00 | 0,99 | 0,97 | 0,92 | 0,91 | 0,83 | 0,73 | 0,75 | 0,59 | 0,41 |
| Références : écart de Q médian (meilleure − deuxième) | 0,35 | 0,73 | 1,16 | 1,38 | 1,48 | 1,48 | 1,23 | 1,35 | 1,20 | 0,00 |
| Exposés (E6) : part dirigée | — | — | — | 0,76 | 0,53 | 0,46 | 0,34 | 0,34 | 0,33 | 0,36 |

- **Verdict : H2 infirmée dans sa forme écrite.** La part dirigée reste ≥ 0,92 jusqu'à −0,75 bit, puis **décroît
  progressivement** sur plus d'un bit (0,91 → 0,41). Ce n'est pas un plateau suivi d'une marche près de −2,3.
- **Le mécanisme supposé n'est pas là.** L'écart de Q ne rétrécit pas quand b baisse : il **grandit** (0,35 → 1,5) et ne
  s'effondre (égalité, écart 0) qu'en dessous de −2 bit. Une partie des pas bascule nettement vers une action non dirigée.
  Ce n'est pas un argmax qui garde son choix sur des écarts qui fondent.
- **L'absence d'effet de B s'explique par la plage de b.** B ne descend pas sous −0,7 bit tant que la localisation tient,
  et à ces b la part dirigée reste ≥ 0,92.
- **À b égal, les exposés sont bien moins dirigés que les références** (0,34 contre 0,73 à −1,5 ; 0,53 contre 0,91 à
  −1,0). Le comportement n'est pas une fonction de b seul, ce qui est compatible avec D1 (V plate dans la zone).

### D3 — lecture par une décision softmax (H2)

Figure `d3_softmax.png` ; `d3_points.csv`. Grille fixée par la règle écrite avant ((53)) : écart médian des pas de
référence bien localisés de D2 = 0,4975 ; τ ∈ {0,0498 ; 0,157 ; 0,498 ; 1,57 ; 4,98} (0,1× à 10×). Mêmes épisodes que
E6 et B (reverse ; mêmes graines), action ~ softmax(Q/τ). b au pas 1 est le même qu'en ε-greedy (mesuré avant l'action).

| τ | référence sans intervention | exposé | ρ(b, part dirigée), points B qui gardent localisation et p_goal |
|---|---|---|---|
| ε-greedy | 0,94 | 0,32–0,49 | −0,05 (n = 11) |
| 0,0498 / 0,157 / 0,498 | 0,99–1,00 | 0,30–0,50 | −0,25 / −0,34 / −0,43 (n = 10–11) |
| 1,57 | 0,94 | 0,35–0,42 | +0,32 (n = 10) |
| 4,98 | 0,69 | 0,38–0,39 | +0,24 (n = 8) |

- **Verdict : H2 infirmée pour B.** À tout τ, la part dirigée reste plate sur les points B qui gardent la localisation
  (b ≥ −0,4, et jusqu'à −0,67 pour p2-b). Au τ intermédiaire (1,57), elle ne perd que 0,01–0,04 (p1-b 0,95 → 0,91 à
  −0,28 ; p2-b 0,94 → 0,93 à −0,39). La chute reste brutale et tombe au même α que la perte de localisation, à toutes
  les températures.
- **C'était le cas prévu « écarts de Q qui ne rétrécissent pas » ((53))** : dans la plage de B, les Q des références ne
  portent presque aucune trace de la croyance partielle.
- Aux grands τ, tous les points tendent vers le hasard (0,4).

### D4 — une anticipation plus longue réduit-elle le poids de la valeur en cache ? (H1)

Figures `d4_lookahead.png`, `d4_b_trace.png` ; `d4_static.csv`, `d4_b_trace.csv`, `d4_conveyor.csv`, `d4_episodes.csv`.
Part de choix greedy dirigés sur les 100 contextes de la zone, k = 1 / 2 / 3 / 5 / 10 :

| Agent | oracle, exposé | modèle, exposé | modèle corrigé à la racine, exposé | oracle / modèle, référence |
|---|---|---|---|---|
| p1-a | 0,75 / 0,78 / 0,81 / 0,86 / 0,99 | 0,48 / 0,49 / 0,44 / 0,41 / 0,44 | 0,67 / 0,67 / 0,69 / 0,65 / 0,66 | 1,00 / 0,96–0,97 |
| p1-b | 0,73 / 0,84 / 0,89 / 0,91 / 0,98 | 0,44 / 0,36 / 0,35 / 0,42 / 0,39 | 0,64 / 0,75 / 0,74 / 0,70 / 0,71 | 1,00 / 0,97–1,00 |
| p2-a | 0,77 / 0,80 / 0,89 / 0,91 / 0,98 | 0,57 / 0,40 / 0,42 / 0,42 / 0,43 | 0,73 / 0,68 / 0,66 / 0,67 / 0,66 | 1,00 / 0,98–1,00 |
| p2-b | 0,85 / 0,93 / 0,90 / 0,95 / 1,00 | 0,40 / 0,41 / 0,42 / 0,38 / 0,38 | 0,76 / 0,75 / 0,76 / 0,66 / 0,65 | 1,00 / 0,98–0,99 |

**Épisodes avec la politique sur Q_k oracle** (E6 : mêmes départs, ε = 0,1, mêmes flux ; 4 agents, coût < 1 h par agent) :

| Agent | E6 (politique d'entraînement), exp. / réf. | k = 1 | k = 5 | k = 10 (exp. / réf.) |
|---|---|---|---|---|
| p1-a | 0,32 / 0,94 | 0,75 / 0,94 | 0,94 / 0,95 | 0,94 / 0,95 |
| p1-b | 0,49 / 0,94 | 0,88 / 0,94 | 0,94 / 0,95 | 0,95 / 0,95 |
| p2-a | 0,33 / 0,94 | 0,93 / 0,94 | 0,95 / 0,95 | 0,94 / 0,95 |
| p2-b | 0,40 / 0,94 | 0,86 / 0,94 | 0,94 / 0,95 | 0,94 / 0,95 |

- **Oracle : H1 confirmée.** La référence reste à 1,00 pour tout k. L'exposé rejoint la référence quand k augmente :
  0,98–1,00 à k = 10 en statique. En épisodes, il est à 0,94–0,95 dès k = 5, la part de la référence.
  - Le déficit d'E6 (0,45–0,62) se décompose ainsi : la croyance de transition en explique la plus grande part (k = 1 :
    0,75–0,93) ; la valeur en cache explique le reste (0,01–0,19 en épisodes), qu'une anticipation de 5 pas efface.
- **Modèle sans intervention : contrôle tenu.** La part de l'exposé reste à 0,35–0,57 pour tout k. Dans l'imagination,
  les transitions décodées depuis la zone suivent le tapis roulant : 0,94–0,98 des cas, et 0,93–0,98 là où le tapis
  diffère de l'action. Chez la référence, c'est 0,03, et l'action jouée dans 0,89–0,96 des cas.
- **Correction à la racine : la croyance ne tient pas dans la simulation.**
  - À la racine, b passe à −0,11 / −0,23 / −0,26 / −0,51 (localisation 0,93–0,97).
  - Dès le premier pas imaginé, b mesuré sur les particules décodées dans la zone retombe à −1,8 / −2,1, puis y reste.
    La moyenne sur toutes les particules remonte ensuite parce que le tapis imaginé les sort de la zone : c'est un effet
    de position, pas un retour de la croyance (`b_mean_in_zone` contre `b_mean`).
  - Les transitions imaginées suivent encore le tapis dans 0,48–0,61 des cas.
  - La part dirigée passe de 0,40–0,57 à 0,64–0,76 à k = 1, puis ne monte plus avec k (0,65–0,71 à k = 10).
  - Une correction appliquée seulement au pas réel ne peut donc pas agir à travers une anticipation longue.

**Tests de forme.**

| Test | Observé |
|---|---|
| V de la référence croît vers le but | oui (ρ = 0,98) |
| Q oracle de la référence dirigé sur la grande majorité des cases | oui (1,00) |
| k = 1 reproduit exactement le Q de la politique | oui (`torch.equal`, test) ; τ → 0 donne l'argmax (test) ; les options par défaut redonnent E6 (test) |
| Référence haute pour tout k | oui (oracle 1,00 ; modèle 0,96–1,00) |
| Aucun poids modifié | oui (hash identique avant/après, 16 runs) |
| E6 rejoué à l'identique | oui (8 sur 8 : états, actions, I, Q) |

### Lecture d'ensemble

- **Déficit résiduel de A.** Le pas à pas contraint donne 0,44–0,71. Le plafond d'une correction parfaite de la seule
  croyance de transition, avec le critic conservé (Q oracle, k = 1, en épisodes), est 0,75–0,93.
  - L'écart de A à ce plafond vient d'une correction incomplète : b n'est corrigé qu'au pas réel, et l'imagination à un
    pas repart du h corrigé.
  - L'écart du plafond à la référence (0,01–0,19) vient de la valeur en cache (V plate dans la zone AI), qu'une
    anticipation oracle de 5 pas efface.
- **Absence d'effet de B.** Elle ne vient pas d'un argmax insensible. Tant que la localisation tient, B reste au-dessus
  de −0,7 bit, et à ces b les Q des références restent nettement ordonnés (D2, D3).

### Coût (CPU, 1 thread par processus, jusqu'à 12 processus sur 12 cœurs)

| Phase | par agent |
|---|---|
| values (D1 + D4 statique, 3 variantes, traces) | 1 290–1 390 s |
| induction (D2 : E6 + B, 16 jeux d'épisodes, tout mesuré) | 1 750–2 140 s |
| softmax (D3 : 5 τ × 9 jeux, mesure dans la zone) | 2 320–2 850 s |
| lookahead (D4 épisodes : k = 1, 5, 10, deux rôles) | 1 360–1 380 s |

Environ 2 h de mur pour les 4 agents.

### Ce que j'ai tranché ((52), (54))

1. **Oracle** : la vraie récompense ET la vraie continuation, et non celles de la tête but ; seule V vient du modèle.
2. **Choix greedy à l'intérieur de l'anticipation** : sur le Q de la même variante (oracle : Q oracle, croyance parfaite
   tout du long), avec ε = 0, le tie-break de l'entraînement et 4 tirages (le K_imag du critic).
3. **« Argmax dirigé »** : la part d'actions dirigées dans le tie set de l'entraînement, c'est-à-dire ce que fait la
   politique. L'argmax strict est rapporté à côté. Il diffère d'au plus 0,04 en D1 et pour l'oracle. Pour la variante
   modèle de l'exposé à k ≥ 2, il diffère jusqu'à 0,11, dans les deux sens et autour de 0,4 : les égalités y sont
   nombreuses. Cela ne change aucun verdict.
4. **Contextes** : 4 départs frais par case de la zone (100) pour D1 et D4. V sur 8 départs par case pour les 143 cases.
5. **b le long de l'imagination** : 1 particule tracée par (contexte, action) au lieu des 2 écrites en (52), pour le coût.
   b est lu à la case décodée, et rapporté aussi sur les seules particules décodées dans la zone.
6. **D2** : uniquement l'image d'entraînement ; pas bien localisés uniquement ; B dans ses deux modes (reverse et
   reverse_random) pour couvrir la plage de b. D3 : B en mode reverse seulement, comme demandé.
7. **D3** : le softmax remplace tout l'ε-greedy (pas de branche ε). Il tire l'action par la CDF inverse de l'uniforme
   d'exploration déjà tiré à chaque pas : les flux sont consommés comme en E6. Les vérifications (localisation, p_goal)
   sont faites contre le run sans intervention au même τ.
8. **D4, épisodes** : les 4 agents (moins d'1 h chacun), politique ε-greedy sur Q_k oracle à chaque pas de l'épisode
   (pas seulement dans la zone). Les particules de l'oracle ont leur propre générateur.

---

# E7 comme dissociation croyance × critic : anti-oracle, E6 pas à pas, contrôle de D2 (2026-10-26)

## 17. Dissociation

Définitions : décisions (55) et (56). Même branche, mêmes 4 agents d'E4, CPU, poids gelés.

### Formes attendues (écrites et commitées avant tout calcul)

Critères chiffrés fixés ici ; « part dirigée » sans précision = la part ε-greedy d'E6 (transitions dans la zone qui
rapprochent du but).

1. **Anti-oracle, monotonie en p.** La part dirigée des références décroît avec p : entre deux p consécutifs, aucune
   hausse au-delà de l'IC bootstrap à 95 % (épisodes rééchantillonnés). La localisation dans la zone reste à 0,05 près de
   celle de p = 0 pour tout p.
2. **p = 1, k = 1.** La part dirigée des références est au niveau d'un marcheur aléatoire (2 actions dirigées sur 5 :
   0,4), dans [0,30 ; 0,50], le même ordre que les exposés d'E6 (0,32–0,49).
3. **p = 1, k = 5.** Pas d'amélioration nette par rapport à k = 1 : la part à k = 5 ne dépasse pas celle à k = 1 de plus
   de 0,05 (la trajectoire imaginée reste sous le tapis dans la zone).
4. **Contrôle croisé.** L'exposé avec p = 1 (k = 1) est à ± 0,05 de l'exposé d'E6.
5. **À b effectif égal.** À b ≈ −1,5 (interpolé par agent sur la courbe anti-oracle k = 1), la référence + anti-oracle
   est plus dirigée que l'exposé pas à pas (classe −1,5 de la tâche 2). L'écart est plus faible près de b = 0 et près de
   b = −2,1.
6. **Tâche 3.** D2 ne gardait déjà que les pas bien localisés ((56)). Forme attendue : la courbe des références sans
   filtre est sous celle de D2 ou égale, et le filtre au niveau du run la fait monter ou la laisse stable (à −1,5 en
   particulier, 0,73 dans D2). Elle ne descend pas.

### Ce qui a été fait

- **Tâche 1 — anti-oracle.** 4 références × p ∈ {0 ; 0,1 ; … ; 1} × k ∈ {1 ; 5}, plus le contrôle croisé sur les
  exposés (p ∈ {0 ; 1}, k ∈ {1 ; 5}) et la variante secondaire h'_a (références, p = 1). Épisodes d'E6 : image
  d'entraînement, mêmes départs, ε = 0,1, mêmes flux, mesurés dans la zone.
- **Tâche 2 — E6 pas à pas.** Calculée sur les épisodes d'E6 existants, où b_local est stocké à chaque pas ; rien n'a été
  relancé.
- **Tâche 3 — contrôle de D2.** Calculée sur les épisodes de la phase induction des diagnostics.
- Code : `run_e7_dissociation.py`, `analyse_e7_dissociation.py`, `probing/lookahead.py` (`mixed_policy_q`, variante
  `mixed`, `h_mode`), option `return_kernel` de `step_controllability`.
- Sorties légères : `results/E7_dissociation/dev_e4_cpu/` (`t1_runs.csv`, `t2_bins.csv`, `t3_bins.csv`,
  `table_2x2.csv`, `forms.json`, figures, méta des 16 runs). Épisodes dans `scratch/e6dev/results/E7_dissociation/`.

**Contrôles.**
- Poids identiques avant et après les 16 runs (hash).
- p = 0, k = 1 rejoue E6 à l'identique (8 sur 8) : états, actions, Q, et I au pas 1.
  - I aux autres pas diffère, comme attendu : ces runs mesurent dans la zone seulement, E6 partout, donc le générateur de
    mesure est consommé autrement. Le comportement n'en dépend pas ((36)).
- b effectif à p = 0 contre le b = 0 d'E6 chez les références : −0,010 à +0,004 bit (tirages de mesure seulement).
  Chez les exposés, b effectif à p = 0 : −2,06 / −2,11 / −2,17 / −2,18, contre −2,06 / −2,12 / −2,17 / −2,17 dans E6.
- Localisation à chaque p : 1,00 chez les références (écart maximal à p = 0 : 0,003).

### Écarts au prompt

1. **h de la branche tapis roulant ((57)).** Avec V(h'_a, z_tapis) comme l'oracle de D4, le critic de la référence lit
   encore l'action dans h'_a. À p = 1, la référence reste alors dirigée (variante h'_a : 0,92–0,94 à k = 1). La variante
   principale utilise un seul h pour toutes les actions (la moyenne des h'_a) : à p = 1, les Q sont exactement égaux,
   comme le prompt le suppose. La variante h'_a est reportée en secondaire.
2. **À k = 1, le mélange en espérance ne déplace pas l'argmax.** (1 − p) Q_modèle + p · constante garde l'argmax jusqu'à
   ce que les écarts passent sous la tolérance. La courbe k = 1 est donc une marche à p = 1, et ce n'est pas un résultat
   empirique. C'est k = 5, avec des particules tirées, qui donne une courbe graduée.
3. **Tâche 3.** D2 ne gardait déjà que les pas bien localisés. Le contrôle compare donc tous les pas, D2, et un filtre au
   niveau du run ((56)).
4. Image d'entraînement seulement ; b effectif calculé dans l'espace des cases (le noyau décodé, comme I_s), sans
   approximation supplémentaire.

### Tableau 2 × 2 (part dirigée ε-greedy dans la zone, moyenne des 4 agents ; extrêmes entre agents)

| k = 1 | Critic référence | Critic exposé |
|---|---|---|
| Vraie croyance | 0,94 (E6) | 0,85 [0,75–0,93] (oracle, D4) |
| Croyance tapis roulant | 0,39 [0,39–0,40] (anti-oracle p = 1) | 0,38 [0,32–0,49] (E6) |

| k = 5 | Critic référence | Critic exposé |
|---|---|---|
| Vraie croyance | 0,95 (oracle D4) ; 0,94 (son propre modèle) | 0,95 [0,94–0,95] (oracle, D4) |
| Croyance tapis roulant | 0,39 [0,38–0,41] (anti-oracle p = 1) | 0,39 [0,39–0,40] (anti-oracle p = 1) ; 0,37 (son propre modèle) |

- **La croyance de transition fait presque tout l'effet** : 0,55 points d'une ligne à l'autre.
- **L'effet du critic n'apparaît que sous la vraie croyance et à k = 1** : 0,09 en moyenne, de 0,01 à 0,19 selon
  l'agent. Il disparaît à k = 5.
- **Sous la croyance tapis roulant, les deux critics donnent le hasard.** À p = 1, toutes les actions sont à égalité.
  L'exposé d'E6, avec son propre modèle, s'écarte un peu du hasard (0,32–0,49) : son prior n'est pas exactement le tapis
  et son critic n'est pas neutre.

Figures : `t1_directed_vs_beff.png` (part dirigée en fonction de b effectif, k = 1 et k = 5), `t2_stepwise.png`,
`t3_localisation.png`.

### Verdicts des formes attendues

1. **Monotonie en p : confirmée.** Aucune hausse au-delà de l'IC entre deux p consécutifs, pour les 4 agents et les deux
   k ; localisation constante (écart maximal 0,003).
   - k = 1 : plate à 0,93–0,95 jusqu'à p = 0,9 (b effectif ≈ −2,0), puis 0,39–0,40 à p = 1. C'est l'effet mécanique de
     l'écart 2.
   - k = 5 : décroissance graduée, 0,94 → 0,91–0,92 (p = 0,4, b ≈ −1,05) → 0,85–0,86 (p = 0,6, b ≈ −1,45) → 0,68–0,72
     (p = 0,8) → 0,38–0,41 (p = 1).
2. **p = 1, k = 1 au niveau du hasard : confirmée.** 0,39–0,40 pour les 4 références, dans [0,30 ; 0,50]. Toutes les
   actions sont à égalité (100 % des pas de zone) et le choix est uniforme. L'argmax strict, lui, vaut 1,00 : il prend
   la première action d'indice, qui se trouve dirigée. Ce chiffre n'a pas de sens ici.
3. **p = 1, k = 5 sans amélioration : confirmée.** Écart k = 5 − k = 1 de −0,013 à +0,012.
4. **Exposé à p = 1 à ± 0,05 de l'exposé d'E6 : infirmée pour 3 agents sur 4.** Écarts : +0,08 (p1-a), −0,09 (p1-b),
   +0,06 (p2-a), +0,00 (p2-b).
   - L'exposé à p = 1 est à 0,40 partout, car ses Q sont à égalité. L'exposé d'E6 s'écarte du hasard dans les deux sens
     (0,32 à 0,49).
   - Même ordre de grandeur en moyenne (0,40 contre 0,38), mais pas à ± 0,05 agent par agent.
5. **À b effectif égal : confirmée à −1,5 et −2,1, non testable près de 0.**
   - À −1,5 : la référence + anti-oracle est à 0,94 (k = 1) et 0,85 (k = 5), l'exposé pas à pas à 0,41 [0,36 ; 0,46].
   - À −2,1 : écart plus faible, 0,69 (k = 1, interpolé sur la marche) et ≈ 0,50 (k = 5) contre 0,36.
   - Près de b = 0, l'exposé n'a aucun pas : toutes ses classes au-dessus de −0,75 ont moins de 30 pas.
6. **Tâche 3 : confirmée, et elle change la lecture de D2.**
   - La part de pas bien localisés tombe sous le niveau d'E6 référence − 0,05 dans toutes les classes sous 0 bit. Elle
     vaut 0,91 à −0,25, 0,25 à −1,5 et 0,03 à −2,25.
   - Sur tous les pas, la courbe est sous celle de D2 : 0,58 contre 0,73 à −1,5.
   - Avec le filtre au niveau du run (runs B dont la localisation dans la zone reste à 0,05 de celle d'E6), elle monte :
     0,98 à −0,9, 0,96 à −1,1 et **0,90 à −1,5** (n = 51 ; aucun pas plus bas).
   - **La décroissance progressive de D2 venait donc de pas bien localisés pris dans des runs où la localisation
     cassait.** Tant que la localisation tient au niveau du run, les références restent ≥ 0,90 jusqu'à −1,5 bit.

### Tâche 2 — E6 pas à pas (`t2_bins.csv`)

| Classe de b | −2,5 | −2,25 | −2,0 | −1,75 | −1,5 | −1,25 | −1,0 | −0,75 |
|---|---|---|---|---|---|---|---|---|
| Exposés : part dirigée [IC] | 0,39 [0,36 ; 0,42] | 0,36 [0,34 ; 0,38] | 0,36 [0,33 ; 0,40] | 0,36 [0,32 ; 0,40] | 0,41 [0,36 ; 0,46] | 0,38 [0,32 ; 0,44] | 0,45 [0,37 ; 0,53] | 0,54 [0,44 ; 0,64] |
| pas / épisodes | 761 / 367 | 2 143 / 655 | 663 / 385 | 519 / 330 | 397 / 276 | 241 / 170 | 146 / 118 | 85 / 77 |

- Les références n'ont des pas que près de 0 : 0,89 [0,81 ; 0,95] à −0,5 (63 pas), 0,92 à −0,25 et 0,94 à 0 et +0,25.
- Les deux courbes ne se recouvrent pas. La courbe anti-oracle est en b effectif, pas en b naturel : ce n'est pas la même
  mesure (indiqué sur la figure).
- Chez les exposés, la part dirigée ne remonte qu'au-dessus de −1 bit, sur peu de pas.

### Points d'attention

- **L'argmax est presque insensible à une croyance partielle quand elle entre par l'espérance (k = 1).** C'est la
  forme algébrique de H2, qui ne dépend pas des données. Une anticipation à particules (k = 5) rend le comportement
  gradué en b effectif.
- **La variante h'_a montre que h porte une information d'action que le critic lit.** Même quand la case prédite est la
  même pour toutes les actions, la référence reste dirigée à k = 1 (0,92–0,94) ; à k = 5, elle tombe à 0,41–0,54.
  - Même lecture possible pour l'oracle de D4 : la part « critic exposé » de 0,85 à k = 1 inclut cette voie.
- Le contrôle croisé (forme 4) dit que la ligne « croyance tapis roulant » de l'anti-oracle est un hasard exact. Ce n'est
  pas une copie fidèle de l'exposé, dont le prior n'est pas exactement le tapis.

### Coût (CPU, 1 thread par processus, 12 à 16 processus sur 12 cœurs)

| Runs | par agent |
|---|---|
| références, p = 0 à 0,5, k = 1 et 5 | 3 450–3 580 s |
| références, p = 0,6 à 1, k = 1 et 5 | 3 950–4 000 s |
| exposés, p = 0 et 1, k = 1 et 5 | 2 950–3 240 s |
| variante h'_a, p = 1 | 1 500–1 680 s |
| tâches 2 et 3, analyse | < 2 min |

Un épisode à k = 5 coûte environ 500 s par jeu de 180 épisodes (k = 1 : environ 60 s). Environ 1 h 10 de temps réel pour
l'ensemble.

---

# D5 : quelle action est crue inefficace, et lecture du prior par la politique (2026-10-26)

## 18. D5

Décisions (58)–(60). Mêmes 4 agents d'E4, CPU, poids gelés.

### Tâche 0 — lecture du code (avant tout run)

1. **z après l'action : 16 échantillons du prior par action.** `policy.q_values` (`agents/worldmodel/policy.py:59-69`)
   appelle `lookahead` (`:39-55`) : `rssm.imagination_step` donne h'_a = GRU(h, z, a) et le prior (`:50`), puis 16
   tirages (`sample_categorical_index`, `:52` ; `torch.multinomial`, `rssm.py:161-171`). Aucune moyenne ni aucun mode
   du prior n'est formé dans l'espace latent.
2. **V est évaluée sur (h'_a, z'_k), avec un h'_a propre à l'action**, pour chacun des 16 tirages (`:64-68`). r et c
   viennent de la tête but sur le même couple. **Q(a) est la moyenne des 16 valeurs** (`:69`).
3. **La politique n'a pas de « case prédite » : elle ne décode jamais.**
   - L'oracle de D4 utilise la VRAIE case (posterior de son image, `lookahead.oracle_q`).
   - La branche modèle du § 17 est le `q_values` de la politique elle-même (`lookahead.mixed_policy_q`, `:108`).
   - La prédiction en cases derrière b (I_s) est un autre objet : 16 AUTRES tirages du même prior (générateur de mesure),
     décodés et appariés au gabarit le plus proche (`fixed_trajectory.step_controllability`, `:112-139`).
4. **Égalités** (`frozen_episodes.run_episodes`, `:200-208`) : tie set de l'entraînement, Q ≥ max Q − 0,0751
   (`policy.tie_set`, `:80-82`). L'action greedy est le membre du tie set au plus haut score uniforme du pas, tiré dans le
   flux de l'épisode ; la branche ε utilise l'uniforme u du même pas.

**Conclusion.** La politique lit des échantillons, donc une croyance partielle entre bien dans Q tirage par tirage, mais
moyennée sur 16 tirages. H4 n'est plausible que comme le bruit de cette moyenne autour de l'espérance du § 17 (un nombre
Binomial(16, p) de tirages « tapis »). La tâche 2 est faite, avec les 16 tirages par action de la politique.

### Formes attendues (écrites et commitées avant tout calcul)

Critères chiffrés fixés ici ; « part dirigée » = part ε-greedy d'E6, sauf mention contraire.

1. **H3.** Sachant C_dir = 1, exposés et références induites ont des parts dirigées proches (écart ≤ 0,10, sur les
   classes de b couvertes par les deux groupes). Sachant C_dir = 0, les deux sont proches du hasard (0,39 ± 0,10).
2. **H3.** À b ≈ −1,5 (classes −1,75 et −1,5), la part de pas avec C_dir = 1 est nettement plus élevée chez les
   références induites que chez les exposés (écart ≥ 0,20).
3. **H3.** Dans la régression, l'ajout de C_dir réduit l'effet du groupe d'au moins la moitié (log-odds, M0 → M1).
4. **H4.** L'anti-oracle par tirage donne une courbe graduée dès k = 1 (pas de marche : la part à p = 0,8 est sous
   celle de p = 0 de plus de 0,05), plus basse que la version en espérance aux p intermédiaires (0,3–0,9).
5. **H4.** À b effectif ≈ −1,5, l'anti-oracle par tirage (k = 1) se rapproche de l'exposé naturel : l'écart à l'exposé
   (0,41) est réduit d'au moins la moitié par rapport au § 17 (0,94 − 0,41 = 0,53, donc ≤ 0,27).

Si H3 et H4 sont toutes deux confirmées, leurs parts dans l'écart de 0,5 sont estimées : pour H3, par l'écart absorbé
par C_dir dans la régression ; pour H4, par l'écart de l'anti-oracle par tirage à b ≈ −1,5.

### Ce qui a été fait

- **Tâche 1.** E6 (les deux modèles) et l'induction B (reverse et reverse_random, 7 α) ont été rejoués avec l'option
  `action_kernel` de `run_episodes`. À chaque pas et pour chaque action, elle enregistre la masse du noyau d'I_s sur la
  vraie case et dit si l'argmax du noyau est la vraie case. Les rejeux sont identiques à E6 (8 sur 8 : états, actions, Q,
  I, localisation, masse jouée).
  - Groupes : 3 968 pas bien localisés pour les exposés ; 30 996 pour les références induites (runs B dont la
    localisation reste à 0,05 près d'E6, plus E6 référence).
- **Tâche 2.** Anti-oracle par tirages à k = 1 sur les 4 références, p de 0 à 1 par pas de 0,1 : variante principale
  (un tirage par action et par draw) et secondaire (un tirage par pas). À k = 5, ce sont les particules du § 17
  (Bernoulli par particule et par pas imaginé : c'est déjà la version par tirages).
- Poids identiques avant et après les 12 runs (hash). p = 0 donne `q_values` lui-même (testé).
- Code : `run_e7_d5.py`, `analyse_e7_d5.py`, `lookahead.sampled_mixed_policy_q`, option `action_kernel`.
- Sorties légères : `results/E7_dissociation/d5/` (`d5_cdir_bins.csv`, `d5_cdir_share.csv`, `d5_regression.csv`,
  `d5_njuste.csv`, `d5_sampled.csv`, `d5_forms.json`, figures `d5_cdir.png` et `d5_sampled.png`, méta des 12 runs).

### Écarts au prompt

1. **« Lue avec le même objet que la politique » n'est pas possible au sens strict** : la politique ne décode jamais
   ((58).3). La prédiction par action est lue sur le noyau d'I_s : même prior, 16 autres tirages décodés.
2. **La décomposition de b par action n'est pas calculée.** I = (1/A) Σ_a KL(p_a ‖ p̄) demanderait une seconde
   implémentation de la mesure de contrôlabilité (règle de CLAUDE.md). La masse sur la vraie case par action est reportée
   à la place.
3. Régression à effets fixes par agent (IRLS en numpy, sans nouvelle dépendance), et non à effets aléatoires.
4. La variante secondaire par tirages n'a pas été faite à k = 5 (coût).

### Tâche 1 — résultats

Part de pas avec C_dir = 1 (argmax ; le seuil de masse ≥ 0,5 donne les mêmes chiffres à 0,02 près) :

| Classe de b | −2,5 à −1,0 | −1,5 | −1,25 | −1,0 | −0,75 | −0,5 | −0,25 | ≥ 0 |
|---|---|---|---|---|---|---|---|---|
| Exposés | 0,000–0,006 | 0,00 | 0,00 | 0,00 | — | — | — | — |
| Références induites | — | 0,44 (48 pas) | 0,62 | 0,59 | 0,81 | 0,94 | 0,99 | 1,00 |

Part dirigée sachant C_dir (pas bien localisés, IC à 95 % par épisodes) :

| Classe de b | Exposés, C_dir = 0 | Références, C_dir = 1 | Références, C_dir = 0 |
|---|---|---|---|
| −2,25 | 0,36 [0,34 ; 0,39] | — | — |
| −1,5 | 0,36 [0,30 ; 0,42] | — (< 30 pas) | — (< 30 pas) |
| −1,25 | 0,49 [0,37 ; 0,59] | 0,87 [0,82 ; 0,93] | 0,91 [0,84 ; 0,96] |
| −1,0 | 0,47 [0,35 ; 0,60] | 0,93 [0,90 ; 0,96] | 0,92 [0,88 ; 0,96] |
| −0,75 | — | 0,91 [0,89 ; 0,93] | 0,91 [0,88 ; 0,95] |

- **Les exposés ne prédisent jamais juste une action dirigée.** La masse maximale du noyau sur la vraie case d'une
  action dirigée vaut 0,003 en moyenne à b ≈ −1,5. Le seul n_juste = 1 fréquent (35 %) est probablement l'action dont
  la vraie case coïncide avec celle du tapis (non vérifié).
- **Les références induites prédisent faux l'action dirigée dans 40 à 56 % des pas sous −1 bit, et restent pourtant
  dirigées à 0,91–0,92 quand C_dir = 0.**
- **Régression** (`d5_regression.csv`, log-odds) :

| Plage | β groupe exposé, M0 | β groupe exposé, M1 (+ C_dir) | β C_dir, M1 | part absorbée (log-odds / probabilité) |
|---|---|---|---|---|
| tous les pas (n = 34 964) | −2,18 (0,13) | −2,11 (0,16) | +0,11 (0,14) | 3 % / 5 % |
| plage commune [−1,5 ; −0,75) (n = 1 040) | −1,91 (0,24) | −1,89 (0,30) | +0,03 (0,28) | 1 % / 1 % |

  En M2, C_dir × groupe n'est pas estimable dans la plage commune : C_dir = 1 n'existe pas chez les exposés.
- **n_juste à b ≈ −1,5** (classes −1,75 et −1,5) :
  - exposés : 0 action juste dans 64 % des pas, 1 dans 35 %, 2 dans 0,5 % ;
  - références induites : 0 dans 17 %, 1 dans 33 %, 2 dans 48 % (69 pas).

### Tâche 2 — résultats (moyenne des 4 références)

| p | 0 | 0,3 | 0,5 | 0,7 | 0,8 | 0,9 | 1 |
|---|---|---|---|---|---|---|---|
| En espérance (§ 17) | 0,94 | 0,94 | 0,94 | 0,94 | 0,94 | 0,94 | 0,39 |
| Par tirages, principal | 0,94 | 0,94 | 0,94 | 0,94 | 0,92 | 0,89 | 0,39 |
| Par tirages, un tirage par pas | 0,94 | 0,79 | 0,66 | 0,55 | 0,50 | 0,45 | 0,39 |
| Particules, k = 5 (§ 17) | 0,94 | 0,93 | 0,87 | 0,78 | 0,70 | 0,60 | 0,39 |

- La moyenne sur 16 tirages lisse presque tout le bruit d'échantillonnage. La version par tirages ne décolle de
  l'espérance qu'à p ≥ 0,8.
- La variante « un tirage par pas » est graduée, mais de façon triviale : chaque pas est entièrement sous le tapis ou
  pas du tout, et la part est le mélange linéaire (1 − p) · 0,94 + p · 0,39.
- Localisation : 1,00 à tout p.

### Verdicts des formes attendues

1. **H3, parts dirigées sachant C_dir : infirmée.**
   - Sachant C_dir = 1, la comparaison est impossible : les exposés n'ont aucun pas avec C_dir = 1.
   - Sachant C_dir = 0, les exposés sont au hasard (0,36–0,49), mais les références induites sont à 0,91–0,92, loin du
     hasard.
2. **H3, plus de C_dir = 1 chez les références à b ≈ −1,5 : confirmée.** 0,44 (48 pas) à −1,5 et 0,62 à −1,25, contre
   0,00 chez les exposés.
3. **H3, régression : infirmée.** L'ajout de C_dir absorbe 3 % de l'effet du groupe (1 % sur la plage commune) ; son
   coefficient est nul (+0,11 ± 0,14).
4. **H4, courbe graduée dès k = 1 : infirmée** pour la variante principale. À p = 0,8, la part ne baisse que de 0,02
   par rapport à p = 0 (critère : plus de 0,05). La courbe ne descend sous l'espérance qu'à p ≥ 0,8.
5. **H4, rapprochement de l'exposé à b effectif ≈ −1,5 : infirmée** pour la variante principale. La part y est à 0,94
   contre 0,36 pour l'exposé, soit un écart de 0,58, non réduit. La variante « un tirage par pas » est à 0,58 (écart
   0,22) mais pour une raison triviale (voir ci-dessus).

Ni H3 ni H4 n'explique l'écart d'environ 0,5 ; il n'y a donc pas de parts respectives à estimer.

### Points d'attention

- **La prédiction décodée n'est pas ce que lit la politique.** Chez les références induites, le noyau d'I_s se trompe
  sur l'action dirigée dans près de la moitié des pas sous −1 bit, sans effet sur le choix.
  - La politique ne lit pas de case : elle évalue V sur (h'_a, z'). Le § 17 l'a déjà montré : avec un h'_a propre à
    l'action, la référence reste dirigée à 0,92–0,94 même quand la case prédite est celle du tapis (variante h'_a).
  - **Hypothèse suivante, non testée ici : l'écart restant à b égal est porté par ce que h'_a transmet de l'action au
    critic (et à la tête but).** Ni b, ni la case décodée, ni le bruit des tirages ne le capturent. Un test direct serait
    un échange de h'_a entre exposé et référence, ou une mesure de l'information sur l'action contenue dans V(h'_a, ·) à
    z fixé.
- Les deux groupes se recouvrent mal en b. Les références induites bien localisées descendent rarement sous −1,25
  (48 pas à −1,5), les exposés remontent rarement au-dessus de −1. La plage commune ne compte que 1 040 pas.
- L'exposé à −1,5 est à 0,36 ici (pas bien localisés uniquement), contre 0,41 au § 17 (tous les pas).

### Coût (CPU, 1 thread par processus, 12 processus)

| Runs | par agent |
|---|---|
| kernels (E6 + induction B avec prédictions par action) | 1 530–1 900 s |
| par tirages, principal (11 p) | 750–840 s |
| par tirages, un tirage par pas (11 p) | 1 030–1 130 s |

Environ 32 min de temps réel ; l'analyse prend moins de 3 min.

---

# D6 : le canal h'_a (2026-10-26)

## 19. D6

Décision (61). Mêmes 4 agents d'E4, CPU, poids gelés, k = 1.

### Formes attendues (écrites et commitées avant tout calcul)

« Part dirigée » = part greedy dirigée (tie set de l'entraînement) sur les pas de zone bien localisés, sauf mention
contraire ; b ≈ −1,5 = classes −1,75 et −1,5 regroupées, ou la plus proche ayant au moins 30 pas, signalée.

1. **Variante (a)** : Q identique bit à bit à celui de la politique, pour les trois groupes ; les épisodes rejoués sont
   ceux d'E6 et de B (états et actions identiques).
2. **Variante (b), références induites à b ≈ −1,5** : la part dirigée chute vers le niveau de l'exposé naturel (≤ 0,50).
3. **Variante (b), références sans intervention (b ≈ 0)** : la part reste élevée (≥ 0,85) : le canal z suffit quand
   la prédiction est juste.
4. **Variante (b), exposés** : pas de changement (± 0,05) : le canal h'_a est déjà éteint.
5. **Test 2** : argmax de V(h'_a, z̄) dirigé à ≥ 0,85 chez les références induites, même sous −1 bit ; proche du hasard
   (0,4 ± 0,10) chez les exposés.
6. **Test 2** : chez les références induites, la part dirigée du test 2 ne dépend pas de b (écart ≤ 0,10 entre la
   classe la plus haute et la plus basse ayant ≥ 30 pas) ; chez les exposés, elle reste au hasard sur toute la plage.
7. **(c)** : chez les références induites, (c) reste dirigée (≥ 0,85) ; chez les exposés, (c) est au hasard
   (0,4 ± 0,10).

**Part de l'écart passant par h'_a** : (part (a) − part (b)) des références induites à b ≈ −1,5, rapportée à
l'écart (a) des références induites − (a) des exposés à la même classe.

### Ce qui a été fait

- **Contrefactuel pas à pas.** E6 (les deux modèles) et l'induction B (reverse et reverse_random, 7 α) ont été rejoués
  avec un `q_fn`. Il copie l'état du générateur de la politique dans un clone et calcule les variantes sur exactement ses
  16 tirages par action (`lookahead.channel_q`), puis rend le Q de la politique elle-même.
  - (a) normale ; (b) h commun ; (c) z commun, tiré du mélange des 5 priors et partagé par les actions ; test 2,
    V(h'_a, z̄).
- **Rejeu en épisodes de (b).** Les références (E6 et B) ont été rejouées avec la politique sur Q_b, en ε-greedy.
- Poids identiques avant et après les 8 runs (hash).
- Code : `run_e7_d6.py`, `analyse_e7_d6.py`. Sorties légères : `results/E7_dissociation/d6/` (`d6_bins.csv`,
  `d6_test2_bins.csv`, `d6_replay.csv`, `d6_identity.csv`, `d6_forms.json`, figures `d6_variants.png` et
  `d6_test2.png`, méta des 8 runs).

### Écarts au prompt

1. **Le contrefactuel n'est pas fait sur les épisodes enregistrés**, qui ne gardent pas (h, z) : il est fait sur des
   rejeux identiques (forme 1).
2. **z̄ est tiré du mélange des 5 priors** (une action uniforme, puis son prior), 16 tirages partagés par les actions.
   C'est symétrique en l'action, et chaque z̄ reste dans la distribution d'un prior.
3. **« Sous ε-greedy » est l'espérance** (1 − ε) × part greedy + ε × (actions dirigées)/5, car le choix contrefactuel
   n'est pas tiré. La part réelle ε-greedy de (b) vient du rejeu en épisodes.
4. Groupes et filtre de D5 : pas de zone mesurés et bien localisés ; références induites = runs B à localisation tenue.

### Résultats (part greedy dirigée ; entre crochets, IC à 95 % par épisodes)

| Groupe, plage de b | (a) normale | (b) h commun | (c) z commun | test 2 : argmax de V(h'_a, z̄) dirigé | test 2 : ρ(V, −distance) |
|---|---|---|---|---|---|
| Références induites, b ≈ −1,5 (69 pas) | 0,89 | 0,89 | 0,93 | 0,93 | 0,77 |
| Références induites, b < −1 (283 pas) | 0,95 | 0,95 | 0,95 | 0,94 | 0,82 |
| Références sans intervention (3 852 pas) | 1,00 | 1,00 | 1,00 | 1,00 | 0,93 |
| Exposés, b ≈ −1,5 (548 pas) | 0,34 | 0,38 | 0,32 | 0,36 | −0,14 |
| Exposés, tous les pas (3 969) | 0,37 | 0,38 | 0,38 | 0,39 | −0,07 |

- Par classe de b (`d6_bins.csv`, figure `d6_variants.png`), les références induites restent à 0,91–1,00 pour les trois
  variantes : à −1,5, 0,91 [0,81 ; 1,00] en (a), 0,93 en (b), 0,95 en (c). Les exposés restent à 0,27–0,54 dans les trois.
- Test 2 (figure `d6_test2.png`) : chez les références induites, l'argmax de V(h'_a, z̄) est dirigé à 0,93–1,00 dans
  toutes les classes. L'écart entre la meilleure et la deuxième valeur vaut 0,53–1,06, et ρ 0,79–0,94. Chez les exposés :
  0,32–0,50 ; écart 0,17–0,32 ; ρ −0,16 à −0,01.
- **Rejeu en épisodes de (b)** (`d6_replay.csv`), sur 31 runs de références induites : part ε-greedy dirigée de −0,006 en
  moyenne par rapport à l'original (de −0,087 à +0,007). Les seuls reculs nets concernent p2-b reverse_random α = 0,5 et
  0,75 (−0,09 et −0,06).

### Verdicts des formes attendues

1. **(a) reproduit les choix : confirmée.** Q identique bit à bit au Q de la politique à chaque pas (vérifié dans le run).
   États et actions identiques aux runs B de D5 ; états, actions et Q identiques à E6 (4 agents sur 4).
2. **(b) chez les références induites à b ≈ −1,5, part ≤ 0,50 : infirmée.** Elle vaut 0,89, comme (a).
3. **(b) chez les références sans intervention ≥ 0,85 : confirmée** (1,00).
4. **(b) chez les exposés à ± 0,05 de (a) : confirmée.** +0,01 sur tous les pas, +0,04 à b ≈ −1,5.
5. **Test 2, ≥ 0,85 chez les références induites sous −1 bit et au hasard chez les exposés : confirmée.** 0,94 contre
   0,39.
6. **Test 2, références induites indépendantes de b et exposés au hasard partout : confirmée.** Étendue de 0,93 à 1,00
   pour les références induites (≤ 0,10) ; de 0,32 à 0,50 pour les exposés.
7. **(c) dirigée chez les références induites et au hasard chez les exposés : confirmée.** 0,93 (b ≈ −1,5) et 0,95
   (b < −1), contre 0,38.

**Part de l'écart à b égal passant par h'_a : environ 0** (−0,7 %). (a) − (b) = −0,004 chez les références induites à
b ≈ −1,5, pour un écart (a) références induites − (a) exposés de 0,55.

### Lecture

- **H5 est infirmée telle que formulée** : l'écart ne passe pas spécifiquement par h'_a.
- **Chez les références induites, l'action atteint le critic par les deux canaux, de façon redondante.** Couper l'un ou
  l'autre ne change rien : (b) et (c) restent à 0,89–0,95.
- **Chez les exposés, les deux canaux sont éteints** : (a), (b) et (c) sont tous au hasard, et V(h'_a, z̄) ne classe
  pas les actions selon la distance (ρ ≈ −0,1).
- **Ce que b manque.** Sous −1 bit, la prédiction décodée des références induites se trompe sur l'action dirigée dans la
  moitié des pas (§ 18), mais les z' tirés par action portent encore, pour le critic, l'information sur la case d'arrivée
  ((b) : 0,95 à b < −1).
  - b est lu par le décodeur et l'appariement aux gabarits. Il mesure une contrôlabilité que l'induction dégrade dans
    l'image décodée, pas dans ce que le critic lit du latent.
  - L'écart d'environ 0,5 à b égal sépare donc la contrôlabilité décodée (b) de la contrôlabilité lisible par le critic,
    que le test 2 estime directement. À b égal, l'exposé l'a perdue et la référence induite l'a gardée.

### Points d'attention

- **Recouvrement en b faible.** Les références induites bien localisées n'ont que 40 pas dans la classe −1,5 et 69 à
  b ≈ −1,5 (IC [0,81 ; 1,00]), 283 sous −1. Les exposés n'ont que 40 pas à −1,0 et aucun au-dessus. Les conclusions à
  b ≈ −1,5 reposent sur ces 69 pas, mais le même résultat tient sur les 283 pas sous −1.
- Le test 2 dépend du choix de z̄ (mélange des priors). Un z̄ tiré d'un seul prior ou du h moyen n'a pas été essayé.
- Le constat de redondance est contrefactuel, pas par pas. Le rejeu en épisodes de (b) le confirme sur les runs à
  localisation tenue, mais ne teste pas (c).

### Coût (CPU, 1 thread par processus, 8 processus)

| Runs | par agent |
|---|---|
| contrefactuel (E6 + induction B, 16 jeux) | 1 240–1 410 s |
| rejeu de (b) (références, 15 jeux) | 1 080–1 330 s |

Environ 24 min de temps réel ; l'analyse prend moins de 3 min.

---

# D7 : concordance de b et de la contrôlabilité vue par le critic (2026-10-27)

## 20. D7

Décision (62). Mêmes 4 agents d'E4, CPU, poids gelés, k = 1. C1, C2, C3 comme en D6 (test 2 et variante (b)).

### Formes attendues (écrites et commitées avant tout calcul)

Pas de zone mesurés et bien localisés ; « agents naturels » = exposés et références d'E6, sans intervention.

1. **Tâche 1.**
   - C2 croît avec b : ρ de Spearman pas à pas entre b et C2 ≥ 0,5, tous agents naturels confondus.
   - Exposés à b ≈ −2,1 (classes −2,25 et −2,0) : C1 au hasard (0,4 ± 0,10).
   - Références à b ≈ 0 (classes −0,25 à +0,25) : C1 ≥ 0,85.
2. **Tâche 1, récupération des exposés.** C1 et C2 croissent avec b sans saut brutal : ρ(b, C2) > 0 chez les exposés,
   et aucun saut de C1 de plus de 0,25 entre deux classes voisines non masquées.
3. **Tâche 1.** À b égal, les références induites de D6 sont au-dessus de la courbe des agents naturels : C1 plus haut
   de plus de 0,10 dans les classes où les deux ont au moins 30 pas.
4. **Tâche 2**, deux issues fixées à l'avance :
   - **réelle** : ΔC1 > 0 et ΔC2 > 0 (IC au-dessus de 0), et les pas corrigés tombent sur la courbe des agents naturels
     (écart pondéré de C1 ≤ 0,10 en valeur absolue) ;
   - **mesure trompée** : Δb ≥ 1 bit mais ΔC1 ≤ 0,10, et les pas corrigés tombent sous la courbe (écart pondéré de C1
     < −0,10).
   Toute autre issue est rapportée comme intermédiaire, avec les chiffres. Le verdict est donné par agent et par réglage.
5. **Sensibilité.** Avec le second z̄ (moyenne des 5 priors), les verdicts des formes 1 à 4 sont inchangés.

### Ce qui a été fait

- **Tâche 1.** E6 (les deux modèles) et l'induction B ont été rejoués avec le `q_fn` contrefactuel de D6 et le second z̄
  (`channel_q`, option `mean_zbar`). Rejeux identiques à D6 (états, actions, Q, V(h'_a, z̄), Q_b) et à E6 (4 agents sur
  4).
- **Tâche 2.** La correction contrainte pas à pas (α = 1, image d'entraînement) a été rejouée pour 10 couples (agent,
  réglage) : le réglage enregistré de chaque agent, plus λ 0,01 avec μ_pos 1 et 10.
  - Le hook d'origine est enveloppé pour garder le h non corrigé et mesurer son I.
  - C1–C3 sont calculés avant et après correction sur les mêmes tirages (clones des générateurs).
  - Les 4 runs enregistrés sont reproduits exactement : part dirigée, b au pas 1, localisation au pas 1 et dans la zone.
- Poids identiques avant et après les 14 runs (hash).
- Code : `run_e7_d7.py`, `analyse_e7_d7.py`. Sorties légères : `results/E7_dissociation/d7/` (`d7_bins.csv`,
  `d7_corr.csv`, `d7_recovery.csv`, `d7_correction.csv`, `d7_identity.csv`, `d7_forms.json`, figure
  `d7_concordance.png`, méta des 14 runs).

### Écarts au prompt

1. **Les runs de correction enregistrés sont ceux du § 11** (application de A), pas du § 15. λ 0,01 μ_pos 1 n'est le
   réglage enregistré que de p1-b, et λ 0,01 μ_pos 10 que de p2-a. Ailleurs, ces réglages ont été rejoués sans run
   enregistré à comparer.
2. **Second z̄ : la moyenne des probabilités des 5 priors.** C'est un z « doux », déterministe, hors de la distribution
   des z one-hot que le critic a vus. C'est une sensibilité, pas une alternative équivalente.
3. **« Agents naturels » = exposés et références d'E6.** Leur courbe commune est faite des exposés sous −0,75 bit et des
   références au-dessus de −0,5 bit, presque sans recouvrement.

### Tâche 1 — agents naturels (z̄ principal ; le second donne les mêmes chiffres à 0,05 près)

| Classe de b | −2,5 | −2,25 | −2,0 | −1,75 | −1,5 | −1,25 | −1,0 | −0,5 | −0,25 | 0 | +0,25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exposés, C1 | 0,45 | 0,37 | 0,36 | 0,32 | 0,43 | 0,45 | 0,50 | — | — | — | — |
| Exposés, C2 | −0,11 | −0,05 | −0,09 | −0,16 | −0,10 | −0,03 | −0,01 | — | — | — | — |
| Références, C1 | — | — | — | — | — | — | — | 0,95 | 0,98 | 1,00 | 1,00 |
| Références, C2 | — | — | — | — | — | — | — | 0,84 | 0,86 | 0,93 | 0,94 |
| Références induites (D6), C1 | — | — | — | — | 0,93 | 0,95 | 0,99 | 0,98 | 0,98 | 0,99 | 1,00 |

- **ρ(b, C2) pas à pas** :
  - 0,77 sur tous les agents naturels confondus ;
  - 0,00 chez les exposés ;
  - 0,18 chez les références ;
  - 0,16 chez les références induites (second z̄ : 0,77, 0,01, 0,16, 0,19).
- **Récupération dans l'épisode** (`d7_recovery.csv`, indice j du premier séjour) : de j = 1 à j = 12, b passe de −2,13
  à −1,91–1,98. C1 passe de 0,52 au premier pas à 0,30–0,41 ensuite ; C2 reste entre −0,16 et +0,08.

### Tâche 2 — correction A (moyennes sur les pas corrigés bien localisés ; IC par épisodes dans `d7_correction.csv`)

| Agent, réglage | b avant → après | C1 avant → après [IC de ΔC1] | C2 avant → après | C3 avant → après | écart de C1 à la courbe naturelle | issue |
|---|---|---|---|---|---|---|
| p1-a λ 0,1 μ 1 (enregistré) | −1,96 → −0,71 | 0,39 → 0,36 [−0,09 ; 0,03] | −0,12 → −0,10 | 0,39 → 0,68 | −0,28 | mesure trompée |
| p1-a λ 0,01 μ 1 | −1,92 → −0,43 | 0,43 → 0,38 | −0,05 → −0,13 | 0,36 → 0,70 | −0,44 | mesure trompée |
| p1-a λ 0,01 μ 10 | −1,94 → −0,49 | 0,44 → 0,35 | −0,05 → −0,19 | 0,39 → 0,67 | −0,42 | mesure trompée |
| p1-b λ 0,01 μ 1 (enregistré) | −2,07 → −0,43 | 0,54 → 0,74 [0,15 ; 0,27] | 0,05 → 0,34 | 0,45 → 0,73 | −0,10 | **réelle** |
| p1-b λ 0,01 μ 10 | −2,09 → −0,38 | 0,55 → 0,73 [0,12 ; 0,24] | 0,05 → 0,39 | 0,50 → 0,74 | −0,12 | intermédiaire |
| p2-a λ 0,01 μ 10 (enregistré) | −1,99 → −0,23 | 0,43 → 0,45 [−0,03 ; 0,07] | 0,08 → 0,04 | 0,38 → 0,69 | −0,47 | mesure trompée |
| p2-a λ 0,01 μ 1 | −1,97 → −0,22 | 0,45 → 0,43 | 0,12 → 0,03 | 0,37 → 0,70 | −0,50 | mesure trompée |
| p2-b λ 0,1 μ 1 (enregistré) | −1,94 → −0,12 | 0,53 → 0,41 [−0,17 ; −0,05] | 0,13 → 0,04 | 0,48 → 0,58 | −0,53 | mesure trompée |
| p2-b λ 0,01 μ 1 | −1,92 → −0,08 | 0,56 → 0,48 | 0,14 → 0,06 | 0,39 → 0,63 | −0,47 | mesure trompée |
| p2-b λ 0,01 μ 10 | −1,91 → −0,05 | 0,57 → 0,44 | 0,16 → 0,05 | 0,40 → 0,65 | −0,53 | mesure trompée |

- La correction fait gagner 1,2 à 1,9 bit de b dans tous les cas.
- Pour 3 agents sur 4, la contrôlabilité vue par le critic à z commun (C1, C2) ne bouge pas ou baisse ; les pas corrigés
  tombent bien sous la courbe des agents naturels (de −0,28 à −0,53).
- Chez p1-b seulement, C1 et C2 montent (C1 +0,19, C2 +0,30), et les pas corrigés frôlent la courbe (−0,10 et −0,12).
- **C3 monte partout** (+0,10 à +0,34) : la correction rend les z' tirés par action lisibles par le critic.

### Verdicts des formes attendues

1. **Confirmée telle qu'écrite, mais portée par les deux amas.** ρ(b, C2) = 0,77 tous agents naturels confondus. Les
   exposés à b ≈ −2,1 ont C1 = 0,37 (hasard), les références à b ≈ 0 ont C1 = 1,00. Dans chaque groupe, la corrélation
   est nulle ou faible (exposés 0,00, références 0,18).
2. **Infirmée.** Il n'y a aucun saut brutal (le plus grand écart de C1 entre classes voisines est de 0,10), mais rien ne
   monte non plus : chez les exposés, ρ(b, C2) = 0,00, et C1 et C2 restent au hasard. La récupération de b dans l'épisode
   (+0,2 bit) ne s'accompagne d'aucune récupération de C1 ou C2.
3. **Confirmée sous −0,75 bit, sans objet au-dessus.**
   - Sous −0,75 bit, les références induites dépassent la courbe naturelle (où elle n'est faite que des exposés) de
     +0,40 à +0,50.
   - De −0,5 à +0,25, l'écart est de +0,06 à −0,01 : les deux courbes sont au plafond.
4. **Mesure trompée pour 7 réglages sur 10, dont 3 des 4 réglages enregistrés (p1-a, p2-a, p2-b) ; réelle pour p1-b au
   réglage enregistré ; intermédiaire pour p1-b λ 0,01 μ 10** (ΔC1 et ΔC2 positifs, mais écart de −0,12).
5. **Confirmée, à une exception limite près.** Avec le second z̄, les formes 1 à 3 et 7 issues sur 10 de la forme 4 sont
   inchangées. p1-b λ 0,01 μ 1 passe de « réelle » à « intermédiaire » : son écart de −0,098 devient −0,117, juste
   au-delà du seuil de 0,10.

### Points d'attention

- **Recouvrement en b.** Les agents naturels n'ont aucun pas entre −0,75 et −0,5 bit, et presque aucun recouvrement entre
   exposés (sous −0,75) et références (au-dessus de −0,5). Les classes masquées sont nombreuses : exposés au-dessus de
   −1,0 (40 pas), références sous −0,5 (62 pas).
- **La « courbe des agents naturels » est une marche entre deux amas.** Les pas corrigés (b ≈ −0,7 à −0,05) sont comparés
  surtout aux références près de 0. Un écart de C1 à la courbe dit donc « pas au niveau des références », pas « sous une
  courbe graduée ».
- **C1 et C2 lisent le canal h'_a à z commun.** D6 a montré que chez les références les deux canaux sont redondants.
  La correction A remonte le canal z (C3 de +0,10 à +0,34, comme b) sans remonter le canal h'_a, sauf chez p1-b.
- **Sensibilité au z̄.** Le second z̄ est hors distribution. Les deux définitions concordent pourtant à 0,05 près sur
   les agents naturels, et sur toutes les issues sauf une, limite.

### Coût (CPU, 1 thread par processus, 14 processus sur 12 cœurs)

| Runs | par run |
|---|---|
| naturels et induits (16 jeux par agent) | 1 830–2 100 s |
| correction A rejouée (un réglage) | 1 520–2 180 s |

Environ 37 min de temps réel ; l'analyse prend moins de 3 min.

### Conclusions

1. **b est-il une mesure valide de la croyance utilisée chez les agents naturels ?** Seulement entre groupes : là où b
   sépare les exposés (≈ −2,1) des références (≈ 0), la contrôlabilité lue par le critic les sépare aussi (hasard contre
   1,00). Mais b ne suit pas cette contrôlabilité dans un groupe : sa récupération de 0,2 bit dans l'épisode n'en
   apporte aucune.
2. **La correction (A) a-t-elle changé la croyance utilisée, ou seulement la mesure ?** Pour 3 agents sur 4, elle a
   changé la mesure (b) et le canal z que lit aussi le critic (C3), mais pas la contrôlabilité que le critic lit à z
   commun (C1, C2), qui reste au hasard. Chez p1-b seulement, elle a aussi changé cette contrôlabilité. C'est l'agent où
   A avait le plus remonté le comportement (0,49 → 0,71).
