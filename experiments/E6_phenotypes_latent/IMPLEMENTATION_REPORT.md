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
