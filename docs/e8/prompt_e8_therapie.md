# E8 — Thérapie et rechute : implémentation complète

## 0. Contexte (à lire avant tout)

Lis d'abord :
- le rapport `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`, en particulier § 16–22 (diagnostics, anti-oracle, D5–D7, campagne à 650 000, contexte et persistance), et `docs/e6/DECISIONS.md` ;
- le code et les configs d'E5 (`run_e5.py`, dont `--extend` et les états de reprise, `check_wave.py`, mesures sur les 12 affectations, sondes), et `experiments/E5_belief_training/analysis_swap/REPORT.md` ;
- `results/E6_phenotypes_latent/all_agents_650k/` (valeurs de référence d'E6/E7 pour les 8 agents) ;
- `chain_all_cpu.py` (modèle pour les runs locaux longs).

Résultats qui fixent le design d'E8 :
- **E4** : chez les baselines, la croyance d'incontrôlabilité est attachée à la position de l'ancienne zone UD (P_UD, quadrant nw en disposition canonique ; son thème est T_inc).
- **E5** : chez les agents shuffle f = 1, elle est attachée à la couleur et à l'arrangement, qui sont confondus à l'entraînement. Toute couleur hors distribution casse la localisation et rend les sondes invalides. **E8 n'utilise donc que des images de la distribution d'entraînement.**
- **E6** : chez les agents naturels, la relation croyance → comportement est une marche entre classes de b. Les b intermédiaires sont lus par la sonde, pas utilisés par l'agent.
- **E7** : la croyance de transition cause le comportement (l'oracle le rétablit chez les 8 agents). Il existe deux traces de l'expérience : la croyance de transition, et la valeur en cache du critic. Le cache est faible chez les baselines et F, aussi lourd que la croyance chez shuffle A et E. Éditer h corrige la croyance sur le moment, mais l'effet a disparu dès le pas suivant et le cache n'est pas touché. Une correction durable demande de changer les poids : c'est E8.
- **Leçon de méthode** : b et la localisation, lus par le décodeur, peuvent être trompés. Juger par ce que l'agent lit lui-même (C3 : argmax du planificateur ; C1 : contrôlabilité vue par le critic) et par le comportement. Juger aussi la localisation depuis la vraie case (en E7, F agissait depuis sa vraie case alors que le décodeur le dessinait ailleurs).

Questions d'E8 :
1. Un agent exposé, réentraîné dans un monde redevenu contrôlable, corrige-t-il sa croyance et son comportement, et à quelle vitesse ?
2. Le comportement est-il en retard sur la croyance, et ce retard dépend-il du poids du cache ?
3. Chez les shuffle, une thérapie limitée à une partie des contextes se transmet-elle aux contextes non soignés (rechute / renewal sur les contextes non soignés) ?

## 1. Première étape : documentation et décisions, avant tout code

Dans cet ordre, en un ou plusieurs commits, avant d'écrire du code :
1. Copie ce prompt dans `docs/e8/` (`docs/e8/prompt_e8_therapie.md`).
2. Crée `docs/E8_therapy_relapse.md` à partir de ce prompt : le design en une à deux pages. Mets à jour la ligne E8 de `docs/EXPERIMENTS.md`. `docs/E8_immunisation.md` est dépassé (l'immunisation est abandonnée) : ajoute en tête une ligne qui renvoie vers `docs/E8_therapy_relapse.md`, sans le supprimer.
3. Crée `docs/e8/DECISIONS.md` (décisions numérotées) avec au minimum les décisions du § 3 : checkpoints de départ, appariements, ce qui est remis à zéro au début de la thérapie.
4. Écris et commite `docs/e8/EXPECTED_FORMS.md` (§ 6) avant le premier calcul.

## 2. Branche, environnement, emplacement des fichiers

- Crée la branche `e8` à partir de `main` à jour (tag `v0.2-e7`), dans un worktree dédié, par exemple `C:\nrw\wt-e8`.
- L'installation éditable pointe vers `v1_ucl\src`. **Dans le worktree, mets `wt-e8\src` en premier dans `PYTHONPATH`**, sinon les tests tournent sur l'ancien code. Vérifie-le au début de chaque session (par exemple en affichant le chemin du module importé).
- Emplacements (même convention qu'E4–E7) :
  - `experiments/E8_therapy_relapse/` : runners, analyses, README, rapport `IMPLEMENTATION_REPORT.md` ;
  - `configs/e8/` : YAML uniquement ;
  - `docs/e8/` : le prompt, DECISIONS, EXPECTED_FORMS ;
  - `tests/e8/` ;
  - `results/E8_therapy_relapse/` : sorties légères seulement (tables, figures, JSON de verdicts) ; les épisodes complets et les poids restent hors du repo.
- **Pas de chemin absolu dans le code** : tous les chemins (poids, sorties, racines de données) passent en arguments ou dans les YAML.

## 3. Points de départ (à trancher et écrire dans DECISIONS avant de coder)

### 3.1 Agents, checkpoints et appariement

| Rôle | Agent | Checkpoint de départ | Référence |
|---|---|---|---|
| baseline | base_A (run E4 `p1-b-d1`, graine 2) | à décider (voir ci-dessous) | sa référence appariée |
| baseline | base_B (graine 102) | idem | sa référence appariée |
| shuffle | shuffle A (graine 2) | 650 000 | réf. shuffle B (graine 102), comme en décision (39) |
| shuffle | shuffle E (graine 402) | 650 000 | réf. shuffle B |
| shuffle | shuffle F (graine 502) | 650 000 | réf. shuffle B |

- **Baselines, 100 000 ou 300 000** : à décider et justifier dans DECISIONS. Proposition : **300 000**.
  - C'est le checkpoint de la campagne E6/E7 à 650 000, donc le checkpoint 0 d'E8 peut être vérifié contre `all_agents_650k/`.
  - Les poids à 100 000 ne sont pas en local.
  - Argument pour 100 000 : niveau d'apprentissage plus proche de celui des shuffle (décision (37) d'E5). À mentionner comme alternative écartée, ou à retenir si tu trouves un argument plus fort ; dans ce cas, explique-le avant de coder.
- Shuffle A est apparié à la référence B (sa référence propre est faible sur les quadrants neutres, aucune sonde neutre valide).
- Les shuffle n'atteignent pas le critère de convergence d'E4 à 650 000 : leur niveau est déclaré, comme en décision (37).
- **Vérifie chaque poids de départ par son SHA-256**, contre `experiments/E5_belief_training/analysis_robustness/tables/inventory.csv` ou les manifestes d'export, en local et sur le cluster.
- Si un chemin, un checkpoint, une empreinte ou un appariement est introuvable ou ambigu, ne devine pas : arrête-toi et liste les cas.

### 3.2 Ce qui est remis à zéro au début de la thérapie

À écrire dans DECISIONS, avec la justification de chaque point :
- **Buffer** : vide (décidé le 23 octobre). Préremplissage minimal joué dans l'environnement de thérapie avec la politique d'entraînement, même procédure qu'au démarrage des runs E5.
- **Optimiseur** : réinitialisé, identique pour tous les runs.
- **Critic** : **conservé**, poids chargés depuis le checkpoint de départ. Le remettre à zéro effacerait la valeur en cache, que P2 et la décomposition croyance / cache cherchent justement à suivre pendant la thérapie. Si tu vois une raison technique de faire autrement, arrête-toi et demande.
- **Scheduler / taux d'apprentissage** : mode post-bascule dès le pas 0 (straight-through, β_rep actif, lr d'après la bascule). **Le mécanisme de bascule ne doit pas se redéclencher : vérifie-le.** Indique l'état exact du scheduler repris ou réinitialisé.

## 4. Design

### 4.1 Environnements de thérapie

Tous ont la **dynamique de la référence** : tout contrôlable, UD → CD, géométrie, but et dynamique du couloir identiques. Ils n'utilisent que des images vues à l'entraînement.

- **Baselines, exposition complète** : l'environnement de référence apparié lui-même, disposition canonique, thèmes d'origine (T_inc reste en nw, mais la zone est contrôlable). Pour la référence baseline, c'est son propre environnement d'entraînement.
- **Shuffle, thérapie partielle sur le quadrant q** : seules les 3 affectations où T_inc est sur le quadrant q sont tirées, uniformément à chaque épisode. Le CD varie parmi les 3 autres quadrants, et le thème suit le type comme en E5. Les 9 autres affectations ne sont jamais montrées pendant la thérapie.
  - Paramètre `treated_quadrant` ∈ {nw, ne, sw, se, all}. Valeur `nw` pour cette campagne : c'est le quadrant de la disposition canonique, où sont définies toutes les sondes.
  - `all` (les 12 affectations, soit l'environnement de référence shuffle) correspond à la thérapie complète, possible plus tard, sans la lancer.

**Zone traitée** (pour compter les visites) : chez la baseline, nw ; chez shuffle, le quadrant q qui porte T_inc dans l'épisode.

### 4.2 Protocole d'entraînement de thérapie

- **Réutilise la reprise d'entraînement d'E5** (`run_e5.py --extend`, états de reprise) et le pipeline existant, en ajoutant les options nécessaires : environnement de thérapie, remises à zéro du § 3.2. Ne recopie pas le code.
- Politique, départs (uniformes sur les états non terminaux), ε, troncature, γ, récompense, entraînement du critic en imagination : identiques à E5.
- Paramètre `start_in_zone_frac` (fraction des épisodes démarrant dans la zone traitée), défaut 0, non modifié dans cette campagne.
- Budget : paramètre `max_env_steps`. Vague 1 : 150 000 (plafond généreux, la vague 1 sert de pilote). Vague 2 : fixé après la vague 1.
- Checkpoints **complets** (poids et état de reprise) sur une grille log-espacée dense au début, par exemple 0, 500, 1 000, 2 000, 3 500, 5 000, 7 500, 10 000, 15 000, 25 000, 40 000, 60 000, 100 000, 150 000. Le checkpoint 0 est pris avant toute mise à jour.
- **Reprise par `--phase resume` prévue dès le début.**
- Logs à chaque pas : la case, le quadrant et, chez shuffle, l'affectation de l'épisode, pour exprimer la récupération en **visites de la zone traitée** en plus des pas d'environnement et de gradient. Plus les métriques de santé habituelles (masse par quadrant, succès).

### 4.3 Mesures à chaque checkpoint de thérapie (poids gelés)

Toutes les mesures réutilisent les sondes et le code d'E4/E5/E6/E7, sans les réécrire. **`measures/controllability.py` reste la seule implémentation de I(a; s'|s).** Les mesures se font dans l'environnement de test à dynamique de la référence.

**Images mesurées :**
- **baselines** : l'image d'origine (disposition canonique) ;
- **shuffle** : les 12 affectations, réparties entre les **3 soignées** (T_inc sur q) et les **9 non soignées**. Les 9 non soignées constituent le test de rechute (renewal) : elles n'ont jamais été montrées pendant la thérapie. Zone de mesure = le quadrant qui porte T_inc dans l'affectation.

**Mesures**, pour l'exposé et sa référence, à chaque checkpoint :
1. **b au pas 1**, protocole des mesures sur les 12 affectations d'E5 (et sonde d'E4 pour l'image d'origine), avec le critère de validité des sondes appliqué à chaque checkpoint. Les points invalides sont rapportés comme tels, jamais retirés en silence.
2. **C1** (contrôlabilité vue par le critic) et **C3** (argmax du planificateur), comme dans la campagne E7 à 650 000.
3. **Comportement** : part de pas dirigés, protocole d'E6 (départ sur les 9 états intérieurs de la zone de mesure, h initial, ε = 0,1, H = 36, politique d'entraînement k = 1), avec le b au pas 1 de ces épisodes. Secondaires d'E6 : latence j_θ, part des pas en zone.
4. **Localisation**, sur les épisodes de phénotype, jugée de deux façons :
   - par le décodeur (règle d'E4/E6), avec le quadrant où l'agent est dessiné quand il est mal localisé ;
   - depuis la vraie case : part dirigée calculée depuis la vraie case, comme en E7 pour F.

En plus, à plusieurs checkpoints de thérapie (grille en config ; par défaut 0, 2 000, 10 000, 40 000 et le dernier) :

5. **Oracle à k = 1** (vraies transitions dans le planificateur, critic de l'agent), pour la décomposition croyance / cache : effet croyance = oracle k=1 − exposé, effet cache = référence − oracle k=1 (définitions de la campagne à 650 000).

Les mesures sont lancées par une phase séparée (`--phase probes`) sur les checkpoints d'un batch, et doivent tourner sur CPU en local pour les tests. Nombres de marches et d'épisodes en config. **Estime le coût total des mesures** (checkpoints × images × agents × mesures) avant le lancement. S'il dépasse nettement celui de l'entraînement, propose une réduction sans l'appliquer avant validation.

### 4.4 Analyse (`experiments/E8_therapy_relapse/analyze_e8.py`)

- **Courbes de récupération** de b, C1, C3 et part dirigée, exposé et référence, en fonction des pas d'environnement, des pas de gradient et des visites de la zone traitée. Chez shuffle, deux courbes séparées, affectations soignées et non soignées.
- **Temps de récupération** : t½ et franchissement de seuils fixés à l'avance dans EXPECTED_FORMS. Pour b (déjà une différence exposé − référence), t½ = premier checkpoint où b atteint la moitié de sa valeur au checkpoint 0. Pour C1, C3 et part dirigée, t½ = premier checkpoint où l'écart exposé − référence atteint la moitié de sa valeur au checkpoint 0. **Censure** si ce n'est pas atteint au budget : rapporter la censure, ne jamais extrapoler.
- **Retard du comportement sur la croyance** : différence entre le t½ de la part dirigée et celui de C3, par agent.
- **Décomposition croyance / cache** aux checkpoints de l'oracle, par agent : le cache se résorbe-t-il plus lentement chez A et E ?
- **Transfert / rechute (shuffle)** : récupération sur les affectations non soignées rapportée à celle des affectations soignées, aux mêmes checkpoints.
- **Points (b, part dirigée)** de tous les checkpoints, superposés au nuage d'E6 à 650 000, exposés et références séparés.

### 4.5 Jumeau tabulaire (secondaire, baselines seulement)

Jumeau construit sur les cases, comptages de départ conservés plus comptages de la thérapie (dilution), donnant b_tab en nw en fonction des visites. C'est une courbe de référence interprétative, pas un contrôle apparié, puisque le world model repart d'un buffer vide. **Vérifie d'abord que les tables de comptage des runs de départ existent ; si non, signale-le et ne les reconstruis pas.**

## 5. Réutiliser sans casser

- Réutilise l'existant plutôt que de le recopier : reprise d'entraînement, pipeline, sondes, `submit_kubeflow.py`, `export_batch.py`, `measures/controllability.py`.
- **Aucune modification de `src/` ne doit changer le comportement d'E0–E7.** Toute nouvelle option a une valeur par défaut qui reproduit l'existant à l'identique. Vérifie-le par des tests sur des runs réduits : rejeu bit pour bit d'E4, E5 et E6. E0–E3 restent figés.
- Réacquisition (remise en incontrôlable après la thérapie) : ne l'implémente pas, mais garde le code compatible (l'environnement de thérapie est un paramètre, et un run peut repartir d'un checkpoint de thérapie).

## 6. Rigueur : formes attendues, seuils et prédictions

Commite `docs/e8/EXPECTED_FORMS.md` avant le premier calcul, puis complète-le (nouveau commit, avant calcul) pour chaque nouvelle analyse. Les verdicts sont rapportés tels que calculés, et tout ce qui est exploratoire est marqué comme tel.

**Formes** (vérifications ; un échec signale un problème d'implémentation ou de mesure) :
- au checkpoint 0, toutes les mesures reproduisent **à l'identique** celles de la campagne à 650 000 (`all_agents_650k/`) et les mesures d'E5 du même agent, aux mêmes graines de mesure, quand elles existent ; sinon, un appel direct du code existant sur le checkpoint de départ ;
- la référence garde Î_réf ≈ log2 5, ainsi que C1, C3 et sa part dirigée au niveau de son checkpoint 0, à tous les checkpoints (un transitoire est possible au tout début à cause du buffer vide, avec retour) ;
- l'exposé, dans la zone traitée : b, C3 et part dirigée non décroissants en tendance ;
- le jumeau tabulaire suit exactement le mélange des comptages, T̂ = (N_départ + N_thérapie) / (n_départ + n_thérapie).

**Prédictions** (hypothèses scientifiques ; écrire pour chacune la lecture des deux issues) :
- P1, la correction tient : la croyance remonte dans la zone traitée et y reste ;
- P2, retard du comportement : le comportement se corrige après la croyance, parce que le critic n'apprend qu'en imagination à travers le prior. Le retard est plus grand chez shuffle A et E, où le cache est lourd, que chez les baselines et F ;
- P3, récupération en marche : b remonte progressivement, la part dirigée change d'un coup (relation par classes d'E6) ;
- P4, transfert chez shuffle : issue ouverte. Une récupération sur les affectations non soignées signifie une guérison transmise par la couleur ; son absence, une guérison attachée au contexte où elle a eu lieu.

Les seuils (t½, censure, « non décroissant en tendance », marche) sont chiffrés dans EXPECTED_FORMS avant le calcul.

## 7. Tests

- **Tests ciblés `tests/e8/`** pendant le développement :
  - identité : un run de thérapie à 0 pas de mise à jour (et un à lr = 0 sur quelques pas, préremplissage compris) redonne exactement les mesures de référence du § 6 ; les poids sont identiques bit à bit ;
  - l'environnement de thérapie baseline est identique, en dynamique et en image, à l'environnement de référence d'E5 ;
  - `treated_quadrant: nw` ne tire que les 3 affectations où T_inc est en nw, uniformément ; la dynamique est identique à la référence shuffle ; `all` redonne les 12 ;
  - remises à zéro du § 3.2 effectives (buffer vide, optimiseur neuf, critic chargé à l'identique, pas de rebascule) ;
  - reprise : interruption puis `--phase resume` redonne les mêmes checkpoints ;
  - mini-run CPU de bout en bout (quelques centaines de pas, checkpoints, mesures, analyse), pour une baseline et pour un shuffle ;
  - rejeu bit pour bit d'E4, E5 et E6 sur runs réduits avec les options par défaut.
- **Suite complète** (environ 75 minutes) avant la fusion.

## 8. Batches et cluster

- **`e8_w1_base`** : base_A et base_B, exposés et références, exposition complète, soit 4 runs. C'est aussi le pilote : il sert à fixer le budget et la grille de checkpoints de la vague 2.
- **`e8_w2_shuf_nw`** : shuffle A, E, F et la référence shuffle B, thérapie partielle en nw, soit 4 runs. Préparé, lancé seulement après validation de la vague 1. Les shuffle ont appris 3 à 5 fois plus lentement : propose un budget avec marge après la vague 1.
- **Plus tard**, `e8_w3_shuf_ne`, `e8_w4_shuf_sw`, `e8_w5_shuf_se` : mêmes 4 runs avec un autre `treated_quadrant`. Génère-les par paramètre, sans les lancer.
- **Toujours un `--dry-run` d'abord.** Un seul pod de 3 CPU (quota) ; rappelle dans les commandes que `nadim-launcher` doit être arrêté pendant les runs.
- Lancement via `submit_kubeflow.py --batch ...`, mesures via `--phase probes`, reprise via `--phase resume`.
- Récupération des résultats selon la convention à jour : `export_batch.py create --batch ...`, transfert, analyse en local. Pas de push depuis le cluster, pas de branche orpheline.

## 9. Runs locaux longs (mesures, analyses sur CPU)

- Reprends le modèle de `chain_all_cpu.py` : tâche planifiée, heartbeat dans `STATUS.txt`, interception de `KeyboardInterrupt`, reprise qui saute les phases terminées.
- **Vérifie que des jobs se terminent, pas seulement qu'ils démarrent** : un premier job court doit aller au bout avant de lancer la chaîne complète.

## 10. Git

- Commits en français, par étapes logiques, **sans ligne `Co-Authored-By`**. Rien n'est poussé, aucune réécriture d'historique.
- Fusion finale dans `main` en `--no-ff`, après la suite complète de tests, puis mise à jour du README (section E8 et tableau de statut).

## 11. Quand t'arrêter et demander

- Chemin, checkpoint, SHA-256 ou appariement introuvable ou ambigu.
- `PYTHONPATH` qui ne pointe pas vers `wt-e8\src`.
- Une mesure du test d'identité, ou un rejeu d'E4/E5/E6, qui n'est pas reproduit à l'identique.
- Un hyperparamètre d'E5 qu'il faudrait changer pour que la thérapie tourne (effondrement, bascule qui se redéclenche).
- Une raison de remettre le critic à zéro.
- Un coût de mesure qui dépasse nettement celui de l'entraînement.
- Des tables de comptage absentes (jumeau tabulaire).

## 12. Rapport

`experiments/E8_therapy_relapse/IMPLEMENTATION_REPORT.md`, tenu à jour par sections numérotées comme pour E6/E7 :
- ce qui a été implémenté et les résultats des tests ;
- les décisions prises (renvoi à DECISIONS) et les points ouverts ;
- les niveaux déclarés des agents de départ et leurs empreintes ;
- le coût estimé par run, par batch et pour les mesures, à partir du coût par pas d'E5 ;
- les commandes exactes pour la vague 1 (dry-run, lancement, mesures, export, analyse).
