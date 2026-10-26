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
