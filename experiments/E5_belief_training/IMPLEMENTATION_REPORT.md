# E5 — rapport d'implémentation (2026-10-19)

Branche `e5`, créée depuis `e4` (3657b33). Aucun entraînement lancé ; tout a été testé sur CPU, en config mini. Les
décisions sont numérotées dans `docs/e5/DECISIONS.md` ; le design et le statut sont dans `docs/E5_belief_training.md` ;
le lancement est décrit dans `RUN_GPU.md`.

## 1. État des lieux de la phase 0 (code d'E4)

- **Configurations.** Le mapping configuration → graine est dans `configs/e4/e4.yaml`, bloc `e4.configurations` (par
  exemple `p1-a-d1: {pair: pair1, inc: a, diagonal: 1, seed: 1}`). Une configuration comprend une paire de couleurs,
  une orientation et une diagonale (toujours 1 dans les vagues). Les flux d'entraînement dérivent de
  `(graine, condition)` : l'exposé et la référence ont donc des flux différents.
- **Checkpoints.** `agent_env<N>.pt` ne contient que des poids. L'état complet (`resume.pt` : optimiseurs, replay, env,
  collecteur, flux) est un fichier unique, écrasé tous les 10 000 pas ; le dernier date de 290 000 et il n'est pas
  exporté. Conséquence : aucun branchement exact n'est possible depuis la référence d'E4.
- **Logs.** À chaque checkpoint, `checkpoints.jsonl` contient sous `"e4"` la masse et I_s par région (contextes réels)
  et, par slot, `I_curve` (le pas 1 en tête). b est calculé après coup. La vague 2 d'E4 n'a ces diagnostics qu'à
  10 000, 100 000 et 300 000.
- **Divergences signalées avant le code.** Il n'existait pas de `DECISIONS.md` pour E4, d'où un nouveau
  `docs/e5/DECISIONS.md`. Le chemin `e5/` du prompt devient `experiments/E5_belief_training/`. Le masque de
  reconstruction est calculé contre un fond unique et le replay utilise une table de 144 images (→ décision (4)).
  « switch » désigne déjà le switch d'estimateur dans le code, d'où le nom `dynamics_switch_step`. La rampe de
  `beta_rep` dépend du budget prévu (→ (9)).

## 2. Ce qui a été implémenté

**Bibliothèque** (3 commits ; chaque option est inactive par défaut) :

- **Déplacement verbatim de code d'E4** vers `pipeline/two_zones_setup.py`, `pipeline/two_zones_records.py` et
  `probing/theme_probe_measure.py`, avec des shims dans E4. Ainsi E5 n'importe aucune expérience.
- **Options de la bibliothèque :**
  - `two_zones.build_layout` / `training_colors` / `build_world` reçoivent un `ud_slot` et un `cd_slot` ;
  - `ReplayBuffer` : `obs_index`, `background_table`, `background_of_row`, `grow()`, `obs_rows_and_flags()`, et le
    champ image `ModelBatch.background` ;
  - la perte lit le fond par pas quand il est présent ;
  - `ClosedLoopCollector` accepte `episode_source` ;
  - `run_one` accepte `hooks.episode_source`, `total_env_steps`, `save_resumable_at` et `branch_from`, ainsi qu'un
    calendrier dynamique ;
  - la source d'épisodes est passée aux hooks.
- **Nouveaux modules :**
  - `streams/assignment.py` : le flux d'affectation, de racine `[graine, 2]` ;
  - `pipeline/episode_sources.py` : `ZoneAssignmentShuffle` et `DynamicsSwitch` ;
  - `pipeline/assignment_records.py` : les mesures de checkpoint par affectation, les visites par type de zone et la
    vue du replay par affectation.

**Expérience** (`experiments/E5_belief_training/`) :

- `run_e5.py` : baseline, shuffle (`--fraction`), immunisation (`--switch-step`, `--post-steps`,
  `--branch-from <fichier|auto>`) ; `--save-resumable-at candidates` ; reprise avec `--resume`.
- `choose_switch_step.py` : propose le premier checkpoint qui satisfait le critère, trace les deux courbes, et recalcule
  les diagnostics manquants (`--recompute`, `--verify`).
- `measure_probes.py` : les sondes d'E4 à k, en pas post-bascule pour l'immunisation, contre la référence baseline à k ;
  `--find-runs`.
- `rsa_e5.py` : la RSA d'E3 ; sous shuffle, les points sont des couples (affectation, état).
- `batches.py`, `submit_kubeflow.py`, `export_batch.py` : les 3 vagues ; le clone e5 et la branche `e5` sont exigés.
- `time_assignment_records.py` : le chronométrage des mesures par affectation.
- Configs : `configs/e5/e5.yaml` et `e5_mini.yaml`, qui lisent `base` (E4) en entier ; toute autre clé est refusée.

## 3. Résultats des tests de forme

Tous les tests ont été lancés en local, sur CPU, avec Windows, Python 3.14 et torch de la venv.

| Test | Résultat |
|---|---|
| **f = 0 ≡ E4**, bit à bit, sur les 300 premiers pas, les 2 conditions. Côté E4, options désactivées ; côté E5, flux d'affectation, table de 12 affectations et fonds par pas activés. Comparés : poids à 100/200/300, pertes enregistrées, et chaque pas stocké (ligne d'observation, s, a, r, drapeaux) | **passe** |
| E4 inchangé par les options : test d'identité bit à bit d'E4 (poids SHA-256) et `tests/e4` | **45/45 passent** |
| **Fréquence des tirages** sur 10⁴ épisodes, f ∈ {0 ; 0,25 ; 0,5 ; 1} : écart à f inférieur à 4 écarts-types binomiaux ; uniformité des 12 affectations tirées (χ², p > 10⁻³) ; épisodes non tirés canoniques | **passe** |
| **Appariement** exposé/référence pour chacune des 12 affectations : H(s'\|s,a) = 0, support de taille 1 partout, I différente exactement sur le slot UD (0 contre log2 5) | **12/12 passent** |
| **Invariants** pour les 12 affectations : couloir, but et régions inchangés ; dérive recalculée pour le slot ; le thème suit le type ; accord modèle/simulation du tapis déplacé | **passent** |
| **Frontières d'épisode** (replay d'un run f = 1, exposé et référence) : chaque changement d'affectation tombe sur un pas `is_first`, chaque fin d'épisode est suivie d'un reset, et les affectations sont les tirages du flux, dans l'ordre | **passe** |
| Même affectation par indice d'épisode, exposé contre référence | **passe** |
| **Fond par pas** : chaque pas échantillonné porte le fond de son affectation, et l'observation n'en diffère que sur la case de l'agent. Le fond canonique masquerait davantage | **passe** |
| **Immunisation identique à la référence jusqu'à `switch_step`** : poids, 3 optimiseurs, replay, flux et progression à S | **passe** |
| **Bascule exacte** : toutes les transitions avant l'épisode de bascule suivent le tenseur de la référence, toutes celles d'après le tenseur exposé (UD rencontré) ; durée = bascule effective + post_steps ; checkpoints = bascule + grille | **passe** |
| **Branchement ≡ recalcul** après la bascule : poids, pertes, replay | **passe** |
| Vagues et commandes (12 runs, paramètres, `--branch-from auto` pour B seulement, états reprenables de la référence B, paires de sondes) ; config = E4 inchangée + e5 | **5/5 passent** |
| `choose_switch_step --recompute --verify` : enregistrement recalculé comparé à l'enregistrement d'origine | **identique bit à bit** (I au pas 1 = 0.12621262513823164 à 600, run mini de graine 1) |
| Export : `create` sur les runs mini de la vague 1, puis `verify` | **48 fichiers, 0 problème** |
| **Run mini CPU complet** par condition : shuffle f = 1 et f = 0,5 (exp/ref), baseline B (exp/ref, avec états reprenables), immunisation A (recalculée, bascule à 216, fin à 616) et B (branchée, `auto`), sondes (grille complète pour f = 1, extrêmes ailleurs, immunisation B à k = 400), RSA, export | **tout s'exécute** |
| Suite complète `tests/` | voir ci-dessous |

**Suite complète, relancée sur une arborescence stable** (commit f9f40c6, sortie complète dans
`logs/tests_e5_full.txt`) : **1 025 passés, 1 échec**, 5 ignorés, 3 xfail (52 min). L'unique échec,
`tests/e5/test_e5_training.py::test_background_groups_are_the_assignments`, était un défaut du test : la partition
des groupes est juste, mais `torch.unique` numérote les groupes dans l'ordre des octets des fonds, et le test comparait
les numéros. Corrigé (comparaison de partitions, commit e6f9314), puis `tests/e5/test_e5_training.py` relancé : 10/10
(`logs/tests_e5_training_rerun.txt`). Les 2 échecs de la première exécution, non identifiés, ne se sont pas reproduits.

**Ajouté depuis** : un test d'identité des sondes, `tests/e5/test_e5_probe_identity.py`. Sur les mêmes poids (un
checkpoint d'E4 mini entraîné et un modèle non entraîné, aux I non nuls), les sondes de la bibliothèque donnent
exactement les mêmes I, courbes, pas mesurés et lignes que le code E4 d'origine (`measure_probes.py` et `e4_setup.py`
lus par `git show 3657b33:…`), sur 6 images de la grille. **Passe.**

**Effondrement du latent en mini : ce n'est pas une régression.** Le mini d'E4 a été relancé au commit 3657b33
(avant le déplacement, dans un worktree) et sur `e5`, dans les deux conditions. Les logs sont identiques ligne à ligne
(JS 0,000, mêmes masses, mêmes I) et les **poids sont identiques bit à bit à 200, 400 et 600 pas**. L'effondrement
vient de la config mini elle-même : `switch_js_threshold: -1.0` force la bascule de l'estimateur au pas de gradient 16,
et `collapse_js_threshold: -inf` désactive le garde-fou. La config réelle a un seuil de 3,0 bits et un garde-fou à
1,0.

## 4. Coût mesuré des mesures par affectation

Conditions : `time_assignment_records.py`, modèle de taille réelle (8×8, h 64) non entraîné, réglages de checkpoint
d'E4 (20 contextes par ligne, 64 tirages, 100 marches, n_in = 25), replay de 20 579 pas collecté sous f = 1, CPU à 8
threads.

| Affectations mesurées | Contextes réels | Marches | Total |
|---|---|---|---|
| 1 (= un checkpoint d'E4) | 19 s | 286 s | 305 s |
| 2 | 33 s | 558 s | 592 s |
| 12 | 201 s | 3 533 s | 3 735 s (×12,2) |

Le coût est linéaire en nombre d'affectations, et les marches dominent (~290 s par affectation). Les contextes réels
augmentent avec la taille du replay, mais restent bornés par son nombre de fenêtres (~4 700 à 300 000 pas).

**Extrapolation à un run shuffle.** Avec X le coût d'un checkpoint d'E4 sur GPU (champ `"seconds"` des enregistrements
de la vague 1 d'E4, que je n'ai pas en local), mesurer les 12 affectations aux 21 checkpoints ajoute 231·X par run. Par
exemple, X = 30 s donne +1,9 h et X = 120 s donne +7,7 h, sur les 40 à 45 h. Le prior non entraîné majore le coût de
décodage, comme dans le chronométrage d'E4.

**Défaut retenu (décision (18))** : l'affectation canonique à chaque checkpoint, et les 12 affectations à 8
checkpoints, `[10000, 25119, 63096, 100000, 150000, 200000, 250000, 300000]`, plus le dernier dans tous les cas. Le
surcoût vaut alors 8·11·X = 88·X par run shuffle au lieu de 231·X (300 000, le dernier checkpoint, est l'un des 8). C'est
paramétrable par `e5.shuffle_records.all_assignments_at` (null = tous les checkpoints).

## 5. Commandes exactes de dry-run des trois vagues

Depuis `/scratch/helplessness-worldmodels-e5`, avec `CONF` et `S` fixés au lancement :

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 --configuration $CONF \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 --configuration $CONF --switch-step $S \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave3 --configuration $CONF --switch-step $S --post-steps 175000 \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
```

Sortie vérifiée avec `CONF=p1-a-d1` et `S=125000`. Chaque job est lancé par `run_e5.py`, avec `--device cuda
--out-dir results` ; 3 CPU, 1 GPU, le volume `scratch`.

- **wave1** :
  - `A_sh100_experimental` / `A_sh100_reference` : `--seed-label A --training shuffle --condition … --fraction 1.0`
  - `A_sh050_experimental` / `A_sh050_reference` : idem avec `--fraction 0.5`
- **wave2** :
  - `B_base_experimental` : `--seed-label B --training baseline --condition experimental`
  - `B_base_reference` : idem en `reference`, avec `--save-resumable-at candidates 125000`
  - `B_sh100_experimental` / `B_sh100_reference` : shuffle, f = 1
- **wave3** :
  - `B_sh050_experimental` / `B_sh050_reference` : shuffle, f = 0,5
  - `A_imm` : `--seed-label A --training immunisation --switch-step 125000 --post-steps 175000`
  - `B_imm` : idem, avec `--branch-from auto`

Pour les sondes : `--phase probes --checkpoint <k>`. La vague 3 exige en plus
`--e4-results /scratch/helplessness-worldmodels-e4/results`, où se trouve la référence d'E4 qui sert de référence à
l'immunisation A.

**Durées estimées** (base E4 : 40 à 45 h pour 300 000 pas, 4 runs en parallèle) :

- vagues 1 et 2 : environ 40 à 45 h, plus le surcoût des mesures par affectation (section 4) pour les runs shuffle ;
- vague 3 : la plus longue de ses trois durées : celle des runs shuffle (même chose), `S + 175 000` pas pour A_imm
  (recalculée), 175 000 pas pour B_imm (branchée). Avec S ≤ 125 000, on reste à 40-45 h (plus le surcoût) ; avec
  S = 175 000, A_imm fait 350 000 pas, soit environ 47 à 53 h.

## 6. Points tranchés en cours d'implémentation

1. **Le code d'E4 passe dans la bibliothèque** (shims dans E4, tests E4 verts, identité bit à bit) : la règle de
   CLAUDE.md interdit à une expérience d'en importer une autre.
2. **`ModelBatch` gagne un champ `background`.** C'est une image, pas un label. Le test de bibliothèque qui fige la
   liste des champs a été mis à jour (décision (4)).
3. **Tirage.** Chaque épisode consomme toujours deux tirages (u, puis k), quel que soit le résultat. L'affectation
   canonique peut elle-même sortir du tirage. La racine du flux est `[graine, 2]` (décision (2)).
4. **Les mesures sous shuffle se font affectation par affectation.** L'enregistrement d'E4 mélangerait les
   affectations et apparierait les prédictions aux images canoniques. Slots 22 et 23 ; graines de marche d'E4 pour
   l'affectation canonique, index d'image 1000 + k pour les autres ; contrôle de protocole final sur l'affectation
   canonique seulement (décision (12)).
5. **Pas de checkpoint avant la bascule.** Le run d'immunisation y est identique à la référence. La grille post-bascule
   est ancrée sur la bascule **effective**, et la durée vaut bascule effective + post_steps. La borne passée à
   `total_env_steps` (S + P + H) n'est qu'une capacité.
6. **Candidats reprenables pour la référence B** : la grille d'E4 de 10 000 à 250 000, plus S s'il est donné en
   vague 2. Environ 25 à 40 Mo par état ; S hors de cette liste impose de passer `--switch-step` dès la vague 2.
7. **`choose_switch_step` recalcule les diagnostics manquants** avec les fonctions, slots et graines d'E4, à partir du
   replay tronqué à `stored_steps`. Vérifié bit à bit sur un run mini.
8. **RSA.** Sous shuffle, la position se décorrèle du thème et de la contrôlabilité (corrélations 0,05 et 0,00 en
   mini). En revanche, **thème et contrôlabilité restent confondus par construction** (0,53 ; le thème suit le type) :
   la RSA d'E5 ne peut pas les séparer dans un run exposé. Seule une condition sur les thèmes, exclue d'E5 pour
   l'instant, le pourrait (décision (13)).
9. **Outils sans dépendance.** `batches.py`, `e5_names.py` et `export_batch.py` n'utilisent que la bibliothèque
   standard, comme les outils d'E4 lancés depuis le notebook. Les seeds sont lues dans le texte des configs.
10. **Ajouter une condition** (thèmes, après E4) se fait par une nouvelle entrée de `WAVES` / `PROBE_PAIRS` et, si
    besoin, une nouvelle source d'épisodes dans `pipeline/episode_sources.py`.

## 7. Diagnostics avant lancement (2026-10-19, soir)

**Risque de bascule précoce de l'estimateur sous shuffle (décision (16)) : confirmé sur la config réelle.** Deux runs
courts sur CPU (config `e5.yaml`, `p1-a-d1`, graine A, exposé, 4 000 pas, sortie dans `scratch/`) :

| Run | Pas de gradient 446 (pas d'env 1 000) | Bascule de l'estimateur | Décodabilité de l'état au pas 1 000 |
|---|---|---|---|
| baseline (procédure d'E4) | JS = 0,000 bit (toujours 0,000 au pas de gradient 797) | pas encore au pas d'env 1 585 | 0,7 % (1,0 % au pas 1 585) |
| shuffle f = 1 | JS = 4,57 bits (4,22 au pas 1 585, après la bascule) | **pas de gradient 701** (pas d'env 1 425, JS 4,85), le plus tôt possible : minimum 500, puis 5 évaluations consécutives | **3,5 %** des contextes (2,4 % au pas 1 585) |

Les runs continuent (logs : `scratch/sw_base.log`, `scratch/sw_sh100.log`). Un troisième run, identique au run
shuffle mais avec la parade activée (`scratch/e5_samebg.yaml`, log `scratch/sw_sh100_samebg.log`), est en cours. Il
dira si la statistique de même fond reste sous le seuil tant que l'état n'est pas décodable. Sur ce CPU, il faut environ
45 min pour 600 pas.

Sous shuffle, la statistique de paires franchit le seuil de 3 bits alors que l'état de l'agent n'est pas décodable.
Le latent sépare les fonds (les affectations), pas les positions. La bascule déclenche alors le passage en
straight-through, la baisse du taux d'apprentissage, l'entraînement des têtes et la rampe de `beta_rep` sur un latent
qui ne code pas encore l'état. Le garde-fou d'effondrement (même statistique, 1,0 bit) ne verrait pas un effondrement
de la position tant que l'affectation reste codée. **Ma recommandation : activer la parade pour les runs shuffle**
(`e5.switch_pairs: same_background`, décision (17)), après avoir vu le run de contrôle ci-dessous. C'est une décision
de conception qui te revient : elle change la procédure d'entraînement des runs shuffle par rapport à E4, mais
seulement sur le critère de bascule. Sous f = 0,5, la moitié des épisodes sont canoniques, ce qui atténue le problème
sans le supprimer.

Logué désormais dans chaque run : le pas de bascule de l'estimateur (`e5.estimator_switch.env_step`, et
`switch_step` en pas de gradient), la statistique `batch_pair_js` à chaque checkpoint, et la décodabilité de l'état.
Pour les runs shuffle, elle est donnée par affectation à chaque checkpoint (`decode_accuracy`) ; pour la baseline et
l'immunisation, aux checkpoints ≤ 100 000 (`e5.decodability`).

**Après la fusion e4 → e5 (demain).** Si `e4_setup.py`, `run_e4.py` ou `measure_probes.py` ont changé sur `e4` après
3657b33, la fusion entrera en conflit avec les shims. Il faudra alors reporter ces changements dans les modules de la
bibliothèque (`two_zones_setup`, `two_zones_records`, `theme_probe_measure`), puis relancer :

- `tests/e4/test_e4_training_identity.py` (identité de l'entraînement) ;
- `tests/e5/test_e5_probe_identity.py`, avec `ORIGINAL` mis au dernier commit d'`e4` fusionné, pour comparer au code
  E4 le plus récent ;
- `tests/e5/test_e5_training.py` (f = 0 ≡ E4) ;
- `tests/e4` et `tests/e5`.
