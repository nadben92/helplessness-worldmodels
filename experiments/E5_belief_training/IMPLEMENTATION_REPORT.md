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

**Suite complète.** Une exécution unique de 37 min a donné 971 passés et **2 échecs** ; les noms ont été perdus à
cause d'un `tail`. Pendant cette exécution, je modifiais l'arborescence et un test E5 alors défectueux faisait partie
de la suite. Ce test (le compte des transitions de l'épisode coupé par la fin du run) a été corrigé depuis. J'ai
ensuite relancé tous les groupes séparément, tous verts :

| Groupe | Résultat |
|---|---|
| `tests/e3`, `envs`, `measures`, `evaluation`, `interventions`, `metrics`, `planning` | 605 passés, 4 ignorés |
| `agents`, `cluster`, tests de premier niveau, conformité et handoff E3, export E4 | 361 passés, 1 ignoré, 3 xfail |
| `tests/e4` | 45 passés |
| `tests/e5` | 36 + 9 + 5 passés |

Le second échec n'est donc pas identifié. Il ne se reproduit dans aucun groupe relancé.

**Effondrement du latent en mini.** Dans tous les runs mini, y compris la baseline B, qui suit exactement la procédure
d'E4, la statistique JS du switch reste à ~0,000 et la masse sur le vrai successeur à 0. Les I sont presque partout à
0. La config mini (latent 3×4, 600 pas, garde d'effondrement désactivée) ne sert qu'à exercer la chaîne : ses valeurs
ne mesurent rien. Rien n'indique un effet propre à E5, puisque la baseline B effondre de la même façon. Il faudra le
vérifier sur les premiers checkpoints GPU : la statistique JS figure dans les logs.

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

**Proposition.** Lire X dans l'export de la vague 1 d'E4 avant de lancer :

- si 231·X ≲ 10 % de la durée d'un run, garder `all_assignments_at: null` (toutes les affectations à chaque checkpoint) ;
- sinon, mettre `all_assignments_at: [10000, 25119, 63096, 100000, 150000, 200000, 250000]`. La dernière est toujours
  mesurée en plus, soit 8 points, ce qui ramène le surcoût à 88·X. L'affectation canonique reste mesurée à chaque
  checkpoint.

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
