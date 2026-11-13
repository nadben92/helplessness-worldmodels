# E8 — thérapie et rechute : rapport d'implémentation (2026-11-05)

Branche `e8` (depuis `v0.2-e7`), worktree `C:\nrw\wt-e8`, `PYTHONPATH=C:\nrw\wt-e8\src` (vérifié à chaque session :
`helplessness.__file__` = `C:\nrw\wt-e8\src\helplessness\__init__.py`). Consigne : `docs/e8/prompt_e8_therapie.md`.
Design : `docs/E8_therapy_relapse.md`. Décisions : `docs/e8/DECISIONS.md` (1)-(18). Formes attendues et prédictions,
commitées avant tout calcul : `docs/e8/EXPECTED_FORMS.md` (3e26fcb).

## 1. Documentation d'abord

Quatre commits avant toute ligne de code : la consigne copiée, le design et la ligne E8 d'`EXPERIMENTS.md` (avec le renvoi
en tête de `docs/archive/E8_immunisation.md`) (96b6470) ; les décisions (1)-(16) (388b99f) ; les formes attendues
(3e26fcb). Les décisions (17) et (18) ont été ajoutées pendant le codage, avant tout run d'E8.

## 2. Agents de départ : niveaux déclarés et empreintes

Les 10 fichiers de poids de l'inventaire d'E5 (`analysis_robustness/tables/inventory.csv`) ont été vérifiés le 5 novembre
par leur SHA-256, dans les deux copies locales (`C:\nrw\e6all` / `C:\nrw\e5_extends\extracted` et `C:\nrw\e7_650k_root`) :
tous conformes. Les poids à 100 000 sont absents en local (MISSING dans l'inventaire). Les empreintes complètes sont dans
`configs/e8/agents.yaml`, vérifiées au début de chaque run de thérapie (un écart arrête le run).

| Modèle | Graine | Checkpoint | SHA-256 | Niveau déclaré |
|---|---|---|---|---|
| base_A exposé / réf. | 2 | 300 000 | `fc813201…08dc71bd` / `b738b3fe…310d9d1b` | critère d'E4 atteint (E4) |
| base_B exposé / réf. | 102 | 300 000 | `d56d22ee…fd2df628` / `d1df54e6…72a902a5` | critère d'E4 atteint (E5, vague 2) |
| shuffle A exposé | 2 | 650 000 | `86da724c…fecba154` | sous le critère : masse min. 0,948 ; \|b_UD + log2 5\| = 0,181 (E5 (37)) |
| shuffle E exposé | 402 | 650 000 | `7653fcea…a948a975` | sous le critère : 0,935 ; 0,197 ; \|b_CD\| 0,136 ; \|b_neutre\| 0,101 |
| shuffle F exposé | 502 | 650 000 | `eed6acff…de84a5ca` | sous le critère : 0,917 ; 0,262 |
| shuffle réf. B | 102 | 650 000 | `7cf05785…814429ca` | masse min. 0,962 |

État du scheduler lu dans les fichiers de départ : bascule au pas de gradient 3 051 à 7 151, rampe de β_rep de 34 539 à
35 359 pas, fichiers au pas de gradient 179 846 (baselines) ou 389 846 (shuffle) : rampe terminée partout, lr déjà au
taux d'après la bascule. Les fichiers ne contiennent ni optimiseur, ni replay, ni état des flux.

## 3. Ce qui est implémenté

**Bibliothèque** (options par défaut = comportement antérieur, testées) :
- `pipeline.agent_run.run_one(fresh_start=FreshStart(...))` : départ depuis les poids d'un checkpoint (RSSM, tête but,
  critic et critic lent chargés, SHA-256 vérifié), replay vide puis préremplissage d'E5, optimiseurs neufs, compteurs à
  0, bascule fixée à 0 (lr d'après la bascule, straight-through et β_rep plein dès le pas 0 ; la bascule ne peut plus se
  déclencher, la garde d'effondrement reste active), flux d'entraînement enfants d'une racine propre, checkpoint 0 écrit
  avant le premier pas. Le bloc des checkpoints de `run_one` est devenu une fonction interne `write_checkpoint`,
  inchangée, appelée au même endroit.
- `streams/therapy.py` : racine de thérapie `[seed, 3]`, flux d'affectation commun `[e8.seed, 4]`, flux de
  `start_in_zone_frac`.
- `pipeline.episode_sources` : `TreatedAssignments` (les 3 affectations du quadrant traité, tirées comme le shuffle d'E5
  ; `all` redonne les tirages d'E5 à f = 1), `StartInZone`, `SingleWorld`.
- `pipeline.assignment_records.walks_by_assignment(slots_of=..., step1=...)` : un sous-ensemble de quadrants, et le
  résumé au pas 1 qui donne la validité.
- Déplacés depuis E5, E6 et E7 (41ee3dd, vérifiés bit pour bit) : `probing/critic_readout.py`,
  `decision_rule.critic_readouts`, `zone_phenotypes.localisation_split`, `two_zones_setup.apply_e5_procedure`.

**Expérience** (`experiments/E8_therapy_relapse/`, voir `README.md`) : `run_e8.py` (train / resume / probes),
`e8_therapy.py`, `probes_e8.py`, `check_identity.py`, `analyze_e8.py`, `batches.py`, `submit_kubeflow.py`,
`export_batch.py`, `chain_cpu.py`. Configs : `configs/e8/e8.yaml`, `agents.yaml`, `e8_mini.yaml`.

## 4. Tests et vérifications (5 novembre)

| Vérification | Résultat |
|---|---|
| Code d'E7 déplacé : D7 natural et lookahead réduits (base_A, `--n-reps 1`), avant / après | 29 et 52 tableaux identiques, lignes identiques |
| Fonctions de tableaux déplacées contre leurs anciens corps (`tests/e8/test_e8_library_moves.py`) | 4 / 4 |
| `tests/e6` après le déplacement | 35 / 35 |
| Thérapie sur la config mini (`tests/e8/test_e8_therapy.py`) : checkpoint 0 = fichier de départ, remises à zéro, pas de rebascule, lr = 0, environnements, tirages nw / all, reprise | 9 / 9 |
| Run court sur les vrais poids (base_A exposé et réf., 300 pas, CPU) | terminés ; statistique de la garde 5,5–6,4 bits (seuil d'effondrement 1,0) |
| **F0 (a), base_A au checkpoint 0** contre la campagne à 650 000 (`check_identity.py`) | **43 / 43 égalités exactes** : épisodes et lignes d'E6, sonde d'E4, tableaux de lecture de D7, part dirigée, b et localisation de l'oracle k = 1 |

**Bug signalé dans E7 (non corrigé : expérience terminée).** `run_e7_diagnostics.phase_episodes` (phase `lookahead`)
range les épisodes de l'exposé et de la référence sous la même clé (`none__0__k<k>`) : dans `episodes.npz`, les
épisodes oracle de l'exposé sont écrasés par ceux de la référence (part dirigée des tableaux conservés 0,9387 = la ligne
« reference » ; celle d'E8, 0,8752, est exactement la ligne « exposed »). Les lignes de `summary.json`, calculées avant
l'écriture, sont justes : aucun résultat publié ne change. F0 compare donc l'oracle sur ses lignes seulement.

**Coût mesuré d'un job de sondes** (base_A, image d'entraînement, exposé + référence, toutes les mesures, oracle
compris ; CPU chargé, 1 thread) : 439 s, contre ≈ 1 320 s estimées en (12). L'estimation des sondes est donc
pessimiste ; elle sera refaite sur le premier job shuffle (12 images).
| Non-régression E3/E4/E5 (identité d'entraînement d'E4 bit pour bit, entraînement / extension / affectations / sondes d'E5, boucle fermée d'E3), code courant | 97 / 97, aucun test sauté |
| Bout en bout mini (`tests/e8/test_e8_end_to_end.py` : train baseline et shuffle nw, sondes à 0 et au dernier checkpoint, analyse) | 3 / 3, **75 min** (point ouvert ci-dessous) |

## 5. Points ouverts

État au 5 novembre ; les points 2 et 3, et le point 5 pour shuffle F, sont traités en section 8.

1. **Jumeau tabulaire bloqué** (DECISIONS (14)) : les `counts.npz` des runs de départ ne sont pas en local. Existent-ils
   sur `/scratch` ?
2. **Bug d'E7** (section 4) : signalé, non corrigé.
3. **Durée du test bout en bout** (75 min, surtout la sonde de validité d'E4 à 150 marches sur les 12 images shuffle) : à
   réduire dans le test seulement (marches de validité réduites), si l'utilisateur le souhaite.
4. Vague 2 : budget et grille à fixer après la vague 1 ; `batches.LAUNCHABLE` ne contient que `e8_w1_base`.
5. F0 (b) et (c) (11 images non canoniques, enregistrements d'E5 sur GPU) et F0 (a) pour base_B, A, E, F : à faire sur
   les checkpoints 0 des runs de la vague concernée.

## 6. Coût

Entraînement (pod de 3 CPU + 1 GPU, comme E5 : ≈ 6 200 pas / h / run à 5 runs, 7 700–8 000 à 4) : vague 1, 4 runs ×
150 000 pas ≈ 19–20 h. Sondes : un job baseline (une image, deux modèles, oracle) = 439 s mesurées sur CPU local ; vague 1
= 2 paires × 14 checkpoints ≈ 3,4 h CPU (< 30 min sur 12 processus). Vague 2 : à mesurer sur le premier job shuffle
(estimation de (12), probablement pessimiste : ≈ 71–85 h CPU, ≈ 6–7 h sur 12 processus).

## 7. Commandes de la vague 1

Sur le cluster (notebook Kubeflow ; `nadim-launcher` arrêté pendant les runs ; après que l'utilisateur a poussé `e8`) :

```bash
git clone <origin> /scratch/helplessness-worldmodels-e8 && cd /scratch/helplessness-worldmodels-e8 && git checkout e8
python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w1_base --repo-dir /scratch/helplessness-worldmodels-e8 --dry-run
python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w1_base --repo-dir /scratch/helplessness-worldmodels-e8
python experiments/E8_therapy_relapse/submit_kubeflow.py --status <dossier superviseur affiché>
# si interrompu :
python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w1_base --phase resume --repo-dir /scratch/helplessness-worldmodels-e8 --dry-run
python experiments/E8_therapy_relapse/export_batch.py create --batch e8_w1_base
```

En local (sous Git Bash, `PYTHONPATH=C:/nrw/wt-e8/src`, depuis `C:\nrw\wt-e8`) :

```bash
python experiments/E8_therapy_relapse/export_batch.py verify C:/nrw/e8_w1/extracted
python experiments/E8_therapy_relapse/chain_cpu.py --batch e8_w1_base --batch-root C:/nrw/e8_w1/extracted --out-dir C:/nrw/e8_w1/probes --logs C:/nrw/e8_w1/logs --smoke
python experiments/E8_therapy_relapse/chain_cpu.py --batch e8_w1_base --batch-root C:/nrw/e8_w1/extracted --out-dir C:/nrw/e8_w1/probes --logs C:/nrw/e8_w1/logs --workers 12
python experiments/E8_therapy_relapse/check_identity.py --probes <probes__base_A_ref__k000000__*> --pair base_A --e6 <e6__base_A__*> --c12 <e7d7__base_A__natural__natural_training__*> --oracle <e7diag__base_A__lookahead__*>
python experiments/E8_therapy_relapse/analyze_e8.py --probes "C:/nrw/e8_w1/probes/E8_therapy_relapse/probes__*" --runs "C:/nrw/e8_w1/extracted/results/E8_therapy_relapse/grid12x12__*" --out results/E8_therapy_relapse/e8_w1_base --e6-cloud "C:/nrw/wt-e6/scratch/all/e6_*/E6_phenotypes_latent/e6__*"
```

## 8. Suite du 6 novembre : correctif d'E7, tests, jumeau, identité de shuffle F

**Correctif d'E7 (commit 565ea11, DECISIONS (19)).** `run_e7_diagnostics.episode_tag` donne aux tableaux de la phase
`lookahead` un suffixe de modèle (`none__0__k<k>__exposed`, `__reference`) ; les clés de l'induction et du softmax ne
changent pas ; aucun script d'analyse, de diapositives ou du papier ne lisait ces tableaux. Test :
`tests/e6/test_e7_lookahead_keys.py`. Rejeu réduit (base_A, `--n-reps 1`, CPU) avant / après : les 6 lignes de synthèse
identiques bit pour bit ; les tableaux de l'ancienne clé égaux à ceux de la nouvelle clé `reference` ; les nouveaux
tableaux `exposed` en diffèrent et redonnent exactement la part dirigée de la ligne « exposed », à k = 1, 5 et 10. Rien de
la campagne n'est relancé. Erratum d'une ligne : rapport d'E6/E7, section 16.

**L'oracle d'E8 n'hérite pas du bug (constaté).** `probing.critic_readout.oracle_q_fn` (le code déplacé) calcule Q et
n'écrit rien ; le bug était dans l'écriture d'E7, pas dans l'oracle. `probes_e8.py` ne fait tourner l'oracle que sur
l'exposé et l'écrit sous `<paire>__<image>__oracle__*`, une clé par paire et par image. Au checkpoint 0, la part dirigée
des tableaux oracle d'E8 est celle de la ligne « exposed » de la campagne (base_A : 0,8752, pas 0,9387 ; shuffle F :
0,8590, égale à la ligne). Garde-fou ajouté : un job de sondes s'arrête (`KeyError`) à toute seconde écriture d'une clé
de son `episodes.npz` ; les tableaux de la référence partagée sont écrits une fois par image (ses lignes restent une par
paire).

**Test de bout en bout.** Sonde de validité d'E4 à 10 marches au lieu de 150, dans le test seulement :
`test_only_validity_n_walks: 10` dans `configs/e8/e8_mini.yaml`, commenté « TESTS ONLY » ; `configs/e8/e8.yaml` n'a aucune
clé `test_only` (testé : `test_campaign_config_has_no_test_only_setting`), les runs réels gardent 150 marches. Le test
n'affirme que la tuyauterie (fichiers, clés, nombre d'affectations, tirages nw, statuts), aucun verdict qui dépende du
nombre de marches. Suite complète (`pytest`, worktree `e8`, machine au repos, CPU) : **1 179 passés, 6 sautés, 3 xfail, 0 échec, en 6 771 s (1 h 53)** ; le test de bout en bout en prend 2 461 s (41 min, contre 75 min avant la réduction), puis vient l'extension d'E5 (≈ 21 min). Les 6 sauts et 3 xfail sont antérieurs à E8 et dus à l'environnement ou à l'histoire d'E3 (inductor absent, handoff pilote d'E3 absent, `RUN_GPU.md` d'E3, xfail stricts du protocole d'E3) ; l'identité d'entraînement d'E4 bit pour bit a tourné.

**Jumeau tabulaire.** Non touché. Constat en passant (vérifié, rien reconstruit) : `replay_final.npz` n'est pas tronqué
— le replay d'un run E4/E5 est dimensionné pour tout le run et jamais écrasé (`ReplayBuffer.for_run`) ; celui de
l'exposé de base_B contient 333 410 pas = ses 300 000 transitions + les observations finales de ses 33 410 épisodes.
Des comptages tirés de lui seraient donc ceux du run. L'entrée DECISIONS « jumeau abandonné pour E8 » est préparée hors
du dépôt, non commitée (`C:\nrw\e8short\DECISIONS_draft_twin_abandoned.md`), en attendant la vérification de `/scratch`.

**Identité au checkpoint 0, shuffle F (`treated_quadrant: nw`).** Run court en local (CPU, 300 pas, checkpoints 0 et 300 ;
exposé F et référence B), sondes au checkpoint 0 sur les 12 images, contre la campagne à 650 000 :

| Vérification | Résultat |
|---|---|
| F0 (a) `check_identity.py` (épisodes et lignes d'E6, sonde d'E4, tableaux C1/C3 de D7, oracle k = 1 sur ses lignes ; exposé et référence) | **43 / 43 égalités exactes** |
| F0 (b) le code existant appelé directement sur le fichier de départ de F (`walks_by_assignment` d'E5 sans option, `run_model` d'E6 sans q_fn de lecture), contre les lignes et tableaux du job | **marches 12 / 12, épisodes d'E6 12 / 12 égaux bit pour bit** |
| F0 (c) enregistrements d'E5 (b au pas 1 par affectation, mesurés sur GPU au cluster) à 0,05 bit | **12 / 12** (écart max 0,043, affectation 2) |
| Validité d'E4 sur les 12 images | 12 / 12 valides ; b de −2,21 à −1,97 |
| F0 (d) poids du checkpoint 0 = fichier de départ (F et B ; SHA-256 du départ vérifié au chargement) | 61 / 61 tenseurs égaux (`torch.equal`) pour chacun ; le fichier du checkpoint 0, ré-sérialisé, a son propre SHA-256 |
| Tirages de la thérapie (run court) | affectations 0, 1, 2 seulement (UD en nw) ; épisodes 17 / 12 / 11 pour F, 17 / 12 / 12 pour la référence B, même suite d'affectations (flux commun ; B fait un épisode de plus en 300 pas) |

**Coût mesuré d'un job shuffle** (12 images, F + référence B, oracle compris, CPU, 1 thread) : 3 535 s. Vague 2 (3 paires
partageant la référence B, 14 checkpoints) : ≈ 33 h CPU, ≈ 3 h sur 12 processus, contre 71–85 h estimées en (12) — les
sondes coûtent nettement moins que l'entraînement ; aucune réduction proposée.

**Vague 2** : toujours bloquée (`batches.LAUNCHABLE = ("e8_w1_base",)`) jusqu'à la validation de la vague 1 et la
fixation de son budget par l'utilisateur.

## 9. Jumeau tabulaire : comptages de départ reconstruits (6 novembre, DECISIONS (20))

Le brouillon « jumeau abandonné » (hors dépôt, jamais commité) est supprimé. La règle et les formes ont été écrites et
commitées avant tout calcul (a371172).

**L'écart 333 410 / 300 000 (exposé base_B).** Ce n'est ni un préremplissage en plus ni un doublon. Chaque pas
d'environnement stocke son observation de départ (300 000, les 256 pas de préremplissage compris : ce sont des pas en
boucle fermée comme les autres, sans entraînement). Chaque épisode terminé stocke en plus sa dernière observation, avec
`a = 0` : 33 410 épisodes terminés, dont 32 369 au but et 1 041 tronqués à 36 transitions (`checkpoints.jsonl` :
33 410 épisodes, 32 369 succès). D'où 300 000 + 33 410 = 333 410. Le 33 411e épisode est en cours quand le run
s'arrête : son dernier pas stocké porte la 300 000e action, mais son successeur n'est pas stocké. Pour la référence :
339 938 = 300 000 + 39 938 (39 218 au but, 720 tronqués).

**Ce qui compte comme transition (s, a, s').** Chaque pas stocké qui n'est pas une fin d'épisode, avec sa propre action ;
s' est le pas stocké suivant du même épisode.
- La transition vers le but compte. Rien ne sort du but : son `a = 0` n'est jamais compté.
- La transition vers le 36e état d'un épisode tronqué compte. L'observation tronquée, elle, n'a pas de transition
  (la troncature est une limite de boucle).
- La réinitialisation n'est jamais une transition. Le préremplissage compte (le `N` de `run_one` le compte).
- La transition en suspens (dernier pas, successeur non stocké) est complétée par la dynamique du run lui-même
  (`transition_tensor()`, ligne vérifiée one-hot). C'est la seule utilisation de l'environnement. Exposé :
  (133, 3) → 132 ; référence : (122, 2) → 134.

**Code.** `twin/replay_counts.py` (nouveau, numpy seul) et `experiments/E8_therapy_relapse/start_counts.py`.
Le test `tests/twin/test_replay_counts.py` (3 / 3, 48 s) vérifie l'égalité exacte avec le `counts.npz` de `run_one` à
chaque checkpoint d'un run mini qui s'arrête en cours d'épisode, ainsi que les conventions sur des replays synthétiques.

**Formes, base_B** (CPU local, 6 s ; résumé : `results/E8_therapy_relapse/start_counts/forms_base_B.json` ; comptages
`N_<k>` à chaque checkpoint et `N_start` hors dépôt, `C:\nrw\e8_twin\base_B`) :

| Forme | Exposé | Référence |
|---|---|---|
| TW0 comptabilité : transitions = `env_steps` du run, ligne à ligne (stockés, fins = épisodes, terminaux = succès, débuts = épisodes + 1 en cours) | 300 000 = 299 999 + 1 en suspens ; 7 / 7 | 300 000 = 299 999 + 1 en suspens ; 7 / 7 |
| TW1 tapis roulant (zone nw, 25 cases) : toute transition comptée va au successeur du tapis, quelle que soit l'action | 20 954 transitions, 0 hors du tapis ; chaque case comptée sous les 5 actions, 1 successeur | — |
| TW2 même zone chez la référence : le successeur dépend de l'action | — | 52 264 transitions ; chaque case sous les 5 actions, 5 successeurs distincts, aucune action à deux successeurs |
| TW3 accord avec la dynamique du run, partout | 0 transition hors de P | 0 transition hors de P |

**base_A** : son `replay_final.npz` a été écrit sur le cluster (`save_final_replay: true` dans E4), mais l'export d'E4 ne
le prend pas. Il n'en existe aucune copie locale. Le jumeau de base_A attend ce fichier depuis `/scratch` ; la commande
est la même (`start_counts.py --pair base_A ...`). Rien n'a été lancé sur le cluster.

**Reste à faire pour le jumeau** (après la vague 1) : N_start + N_thérapie (le `steps.npz` / replay des runs de thérapie)
→ T̂ du mélange, contrôlé par la forme TW (`np.array_equal`), puis b_tab en nw en fonction des visites.

## 10. Jumeau : comptages de départ de base_A (6 novembre)

**Fichiers.** Les deux `replay_final.npz` de base_A (run E4 p1-b-d1, graine 2) ont été rapatriés de `/scratch` par
l'utilisateur dans `C:\nrw\tabular_twin_e4`, sous les noms `replay_final.npz` et `replay_final(1).npz`. Leurs noms ne
disaient pas lequel était lequel : ils ont été identifiés par leur contenu, contre le dernier enregistrement de
`checkpoints.jsonl` de chaque run. Les trois comptes coïncident sans ambiguïté :

| Fichier | Pas stockés | Épisodes terminés | Succès | Run |
|---|---|---|---|---|
| `replay_final.npz` (SHA-256 5d6c8aaa…) | 340 967 | 40 967 | 40 558 | référence |
| `replay_final(1).npz` (SHA-256 7826aa1d…) | 333 515 | 33 515 | 32 604 | exposé |

Comme pour base_B, chaque fichier a été copié (originaux conservés) sous le nom `replay_final.npz` dans le dossier de son
run, à côté de son `meta.yaml`, `config.yaml` et `checkpoints.jsonl` (`C:\nrw\e4_wave1\results\E4_generalization_gradient\`).

**Résultat** (`start_counts.py --pair base_A`, CPU local ; résumé
`results/E8_therapy_relapse/start_counts/forms_base_A.json` ; comptages hors dépôt, `C:\nrw\e8_twin\base_A`) : **toutes
les formes vérifiées**.

| Forme | Exposé | Référence |
|---|---|---|
| TW0 comptabilité | 333 515 stockés = 299 999 + 1 en suspens + 33 515 fins (32 604 au but, 911 tronquées) ; 300 000 transitions = `env_steps` ; 7 / 7 | 340 967 = 299 999 + 1 + 40 967 fins (40 558 au but, 409 tronquées) ; 300 000 ; 7 / 7 |
| Transition en suspens, complétée par la dynamique | (28, 0) → 40 | (13, 0) → 1 |
| TW1 tapis roulant (nw, 25 cases) | 22 254 transitions, 0 hors du tapis ; chaque case sous les 5 actions, 1 successeur | — |
| TW2 même zone chez la référence | — | 52 382 transitions ; chaque case sous les 5 actions, 5 successeurs distincts |
| TW3 accord avec la dynamique | 0 transition hors de P | 0 transition hors de P |

Les comptages de départ des deux baselines sont donc disponibles. Le jumeau proprement dit (N_départ + N_thérapie, forme
TW) viendra après la vague 1.

**Budget « planned 300000 » des logs de la vague 1 (lu dans le code, rien relancé).** 300 000 =
`budget.total_steps + prefill_steps` = 299 744 + 256, hérité tel quel de `configs/e5/e5.yaml` (lui-même hérité d'E4) ;
les 150 000 pas de la thérapie entrent par `max_env_steps`. Dans un run E8, ce budget prévu ne règle rien :
- epsilon (0,1), les taux d'apprentissage et `slow_tau` sont des constantes. Il n'y a aucun scheduler : le taux du
  modèle ne dépend que de la bascule (`phase_learning_rate`), et la bascule est à 0 ;
- la seule dépendance au budget prévu est la rampe de beta_rep (20 % des pas de gradient prévus restants), calculée au
  moment où la bascule se déclenche. En départ frais, `switch_step = 0` et `rep_warmup = 0` sont posés avant le pas 0, et
  la bascule ne peut pas se redéclencher (testé : `tests/e8/test_e8_therapy.py`). Ce code ne s'exécute donc jamais ;
- l'échéance de séparation (`switch_max_steps`, absolue et inactive une fois basculé) et le seuil tardif sont en pas
  absolus ;
- les checkpoints, la taille du replay et la fin du run se lisent sur les 150 000 pas ;
- `planned_env_steps` ne sert qu'au contrôle de cohérence d'une reprise et aux métadonnées (`results.json` :
  `planned_env_steps: 300000`, `truncated_by_max_env_steps: true`), qu'aucun script d'E8 ne lit.

## 11. Vague 1 (baselines) : analyse (8 novembre)

Code du cluster : `95c235b` (runs et sondes). Le seul commit local postérieur alors présent, `fa102ea`, ne touchait que ce
rapport et `forms_base_A.json` : rien de l'entraînement ni des mesures. Formes ajoutées avant tout calcul : EXPECTED_FORMS
section 4 (7905b01) ; décisions (21), (22) (f2d69c7).

### 11.1 Vérification de l'export

- `e8_w1_base__20261108T165312Z.tar.gz` extrait hors dépôt (`C:\nrw\e8_wave1\extracted`) ; `export_batch.py verify` :
  **262 fichiers, 0 problème**. Les 4 runs sont `done` à 150 000 pas (89 846 pas de gradient), et chacun a ses 14 checkpoints.
- Les 28 jobs de sondes sont présents, un par paire et par checkpoint, tous `done`, chacun avec 2 lignes (exposé,
  référence). Ils ont tourné sur **GPU** (`device: cuda`, torch 2.10.0+cu128).
- Les 8 fichiers rapatriés à part ont été copiés dans leurs dossiers de runs par `place_run_files.py` (2af568a, originaux
  conservés, SHA-256 vérifié après copie). Chaque dossier est identifié par le nom de dossier de run contenu dans le nom
  du fichier.
- **Sondes refaites sur CPU** (DECISIONS (21)). F0 (a) est définie sur le device de la campagne à 650 000. Les 28 jobs ont
  donc été relancés en local par `chain_cpu.py` : smoke d'abord (3,5 min), puis la tâche planifiée `E8_w1_cpu_chain`
  (12 processus, 52 min, 28/28 `done`). Sorties hors dépôt, dans `C:\nrw\e8_wave1\probes_cpu`. L'analyse lit ces reprises.

### 11.2 Formes de contrôle : toutes vérifiées

| Forme | Résultat |
|---|---|
| F0 (a) checkpoint 0 = campagne à 650 000, CPU, égalité exacte (`identity/F0a_base_*.json`) | **base_A 43 / 43, base_B 43 / 43** |
| F0 (c) base_B, I de l'exposé au pas 1 (marches canoniques) contre l'enregistrement d'E5 à 300 000 (0,000799), 0,05 bit | CPU 0,0111 (écart 0,010) : tient. Le job GPU du cluster redonne l'enregistrement exactement (écart 0,0). base_A n'a pas d'enregistrement local (prévu) |
| F0 (d) poids du checkpoint 0 = fichier de départ (RSSM, tête but, critic, critic lent) | 61 / 61 tenseurs égaux pour chacun des 4 modèles ; SHA-256 des départs conformes |
| H (a) pas de rebascule, pas d'effondrement | `switch_step` = 0 dans tous les enregistrements ; statistique de la garde ≥ 4,99 bits sur 1 797 évaluations par run (seuil 1,0) |
| H (b), (c) succès et masse par région dès 5 000 | succès 1,0 à tous les checkpoints ; masse min. hors nw de l'exposé ≥ 0,933 (exposés) et ≥ 0,958 (références). Transitoire unique au checkpoint 500 : couloir à 0,889 (base_A exp.) et 0,886 (base_B exp.) |
| R la référence reste contrôlable | **tient**, sans transitoire : I_réf 2,29–2,32 à tous les checkpoints ; C1, C3 et D dans ± 0,05 de leur valeur au checkpoint 0 partout |
| M l'exposé ne se dégrade pas en nw | **tient** : ρ (b, C3, D) = 1,00 / 0,95 / 0,70 (base_A) et 1,00 / 1,00 / 0,87 (base_B) ; drawdown max. 0,045 (C3 de base_A) |
| X GPU du cluster contre CPU local (exploratoire) | écart max. : b 0,081 (base_A, 2 000), C1 0,051, C3 0,040, D 0,050 (base_B, 500) ; écarts médians : b 0,012, D 0,002 |

### 11.3 Prédictions, telles que calculées (seuils d'EXPECTED_FORMS)

Abscisses (identiques pour les deux paires) : 500 / 1 000 / 2 000 / 3 500 / 10 000 / 25 000 pas d'environnement =
146 / 446 / 1 046 / 1 946 / 5 846 / 14 846 pas de gradient. Visites de nw entre parenthèses.

| | base_A | base_B |
|---|---|---|
| b(0) → b(150 000) (marches, pas 1) | −2,29 → −0,07 | −2,30 → −0,02 |
| t½ b | 3 500 (288) | 2 000 (227) |
| t½ C1 / C3 / D | 1 000 (87) / 2 000 (154) / 1 000 (87) | 500 (53) / 2 000 (227) / 2 000 (227) |
| b > −0,5 (franchissement) | 25 000 (2 842) | 10 000 (1 167) |
| g_D ≤ 0,05 (comportement au niveau de la référence) | 2 000 (154) | 3 500 (402) |
| t½ soutenus | égaux aux t½ (aucune rechute) | égaux aux t½ |

Aucun t½ n'est censuré.

- **P1 (la correction tient) : tient pour base_A et base_B.** Aucun t½ de b n'est censuré, b(T) > −0,5, et b ne retombe jamais
  sous −1,0 après son franchissement : la remontée est monotone (ρ = 1).
- **P2 : non décidable en vague 1** (section 4 d'EXPECTED_FORMS). Les retards t½(D) − t½(C3) valent **−1 000 pas (base_A :
  le comportement avant la lecture du planificateur)** et **0 (base_B)**. Aucun retard positif chez les baselines, ce qui
  est compatible avec un cache faible. Le verdict attend shuffle A, E et F.
- Décomposition croyance / cache (descriptive, `decomposition.csv`). Au checkpoint 0, effet croyance 0,39 et effet cache
  0,064 (base_A) ; 0,35 et 0,165 (base_B), soit les valeurs de la campagne d'E7. L'effet cache tombe à −0,009 dès 2 000
  chez base_A ; chez base_B, il vaut 0,086 à 2 000, puis −0,004 à 10 000. L'effet croyance est < 0,025 dès 2 000 (A), et
  < 0,006 dès 10 000 (B).
- **P3 (récupération en marche) : ne tient pas pour base_A ni pour base_B.** Les deux gains sont suffisants (D : 0,45 et
  0,51 ; b : 2,23 et 2,28). Mais s_D = 0,45 et 0,44 < 0,5 : la part dirigée monte sur trois intervalles de la grille
  (0–500, 500–1 000, 1 000–2 000), pas sur un seul. Et s_b = 0,23 et 0,30 < 0,5. La grille est fine à cet endroit : ce n'est
  pas une marche sous-échantillonnée (limite écrite en P3).

### 11.4 Résultats exploratoires (non prévus, marqués comme tels)

1. **Le comportement suit C1, C3 et la localisation, pas b.** C1, C3, D et la localisation par le décodeur sont tous revenus
   au niveau de la référence entre 1 000 et 3 500 pas (87 à 402 visites de nw). b, lui, est encore à −1,27 (A) et −1,05 (B)
   à 2 000, et ne passe −0,5 qu'à 10 000–25 000 pas. Dans le plan (b, D) (`b_vs_directed.png`), les points d'E8 sont sur le
   nuage d'E6 à 650 000 : D ≈ 0,92–0,94 dès b ≈ −1,1. C'est la leçon d'E7 : b sous-estime la récupération que l'agent
   utilise.
2. **Localisation** (`localisation_base_*.png`). Au checkpoint 0, l'exposé est mal localisé sur 39 % (A) et 29 % (B) des
   pas en zone. Il est alors dessiné en se à 97 % et 99 %, le régime « saut de quadrant » d'E6. L_dec dépasse 0,97 dès
   1 000 (A) et 0,99 dès 2 000 (B). Ensuite, il reste moins de 10 pas mal localisés par checkpoint. Depuis la vraie case,
   la part dirigée des pas mal localisés suit celle des pas bien localisés au checkpoint 0 (0,46 contre 0,50 pour A).
3. **La part de pas en zone récupère beaucoup plus tard que la part dirigée.** Elle passe de 0,46 à 0,73 (A) et de 0,48 à
   0,82 (B) à 150 000, pour 0,81–0,82 chez la référence. base_A n'y est pas encore à 150 000. La latence de récupération
   vaut 1 pas dès 1 000–2 000.
4. **La masse a priori sur le vrai successeur en nw** (enregistrement de santé, sur le replay de thérapie) remonte
   progressivement, comme b : 0,22 → 0,985 (A) et 0,18 → 0,991 (B).

### 11.5 Jumeau tabulaire (`twin/`, therapy_counts.py)

| Forme | base_A exp. / réf. | base_B exp. / réf. |
|---|---|---|
| TW4 `counts.npz` = comptages reconstruits depuis `replay_final.npz`, aux 14 checkpoints | égaux / égaux | égaux / égaux |
| TW0 comptabilité (150 000 transitions) | 171 892 = 149 999 + 1 en suspens + 21 892 fins / 171 928 = 149 999 + 1 + 21 928 | 171 749 = 150 000 + 0 + 21 749 / 171 808 = 150 000 + 0 + 21 808 |
| Transition en suspens, complétée par la dynamique de thérapie | (45, 1) → 46 / (99, 3) → 98 | aucune |
| TW2 en nw (contrôlable chez les deux) | 25 cases, 5 successeurs distincts | idem |
| TW3 accord avec la dynamique de thérapie | 0 hors de P | 0 hors de P |
| Mélange N_départ + N_thérapie (`np.array_equal`) | 14 / 14 | 14 / 14 |

Aucun épisode tronqué pendant la thérapie (succès 1,0).

**b_tab en nw** (jumeau d'E2, α = 1), à 150 000 :

- exposé : −2,23 → **−1,44** (base_A, 20 278 visites de nw contre 22 254 transitions de tapis au départ) et −2,22 →
  **−1,36** (base_B) ;
- référence : 0 exactement (la dynamique est déterministe, donc les tirages appariés redonnent les mêmes comptages).

Le world model franchit −0,5 après 1 100 à 2 800 visites. Le mélange des comptages est encore à −1,4 après 20 000 à
26 000 visites (`twin_overlay.png`).

Lecture interprétative, pas un contrôle apparié : l'écart tient en partie au design. Le world model repart d'un buffer
vide, alors que le jumeau garde toute son histoire. La croyance stockée dans les poids se réécrit en quelques milliers de
pas dès que l'expérience ancienne n'est plus rejouée. Un apprenant qui garde toute son histoire resterait bien plus
longtemps biaisé.

Piste future (non prévue dans E8) : une thérapie qui conserve l'ancien buffer, rejoué avec l'expérience nouvelle. Ce
serait l'analogue d'une rumination ; le jumeau en donne la borne de dilution.

### 11.6 Proposition pour la vague 2 (non débloquée : `batches.LAUNCHABLE` inchangé)

Lecture de la vague 1. Le comportement, C1 et C3 se corrigent en 1 000 à 3 500 pas (≈ 90–400 visites de nw). b franchit
−0,5 entre 10 000 et 25 000 pas. À 60 000, b est à −0,18 et −0,09. Les shuffle ont appris 3 à 5 fois plus lentement, et
leur cache est plus lourd (E7 : 0,22–0,24 chez A et E). Hypothèse : la thérapie partielle sur nw met T_inc en nw dans
chaque épisode, avec la même géométrie, donc le taux de visites de la zone traitée devrait être proche de celui des
baselines (≈ 0,13–0,18 par pas). Cette hypothèse est à vérifier au premier checkpoint.

- **Budget `max_env_steps` = 250 000.** Franchissement de b à 25 000 (le plus lent des baselines) × 5 = 125 000, × 2 de
  marge. P1 demande b(T) > −0,5 avec un plateau visible derrière, et P4 lit les non soignées à T. Alternative à 300 000 si
  l'on veut voir le plateau de A et E avec plus de marge.
- **Grille** : celle de la vague 1, prolongée : 0, 500, 1 000, 2 000, 3 500, 5 000, 7 500, 10 000, 15 000, 25 000, 40 000,
  60 000, 100 000, 150 000, 200 000, 250 000 (16 checkpoints). La partie dense du début reste nécessaire. Chez les
  baselines, tout le comportement s'est joué avant 3 500 pas ; ×3–5, cela donne 3 500–17 500 pas, que la grille couvre en
  6 points. C'est aussi ce qui permet de décider P3 sans l'ambiguïté d'un long intervalle.
- **Oracle** : 0, 2 000, 5 000, 10 000, 25 000, 60 000, 150 000, 250 000 (8 au lieu de 5). Chez les baselines, l'effet cache
  disparaît entre 2 000 et 10 000 pas : la grille de 5 points ne l'aurait pas vu décroître chez A et E. Coût négligeable
  (≈ 60 s par ensemble).
- **Coût d'entraînement** : 4 runs (A, E, F, référence B) × 250 000 pas. Au rythme de la vague 1 (≈ 7 900 pas / h / run,
  4 runs par pod), cela fait ≈ 32 h ; au rythme d'E5 (6 200 pas / h à 5 runs), ≈ 40 h. À 300 000 : 38–48 h.
- **Coût des mesures** : 12 affectations par checkpoint, 3 paires partageant la référence B. Le job shuffle mesuré (F + B,
  12 images) prend 3 535 s CPU, soit ≈ 2,4 h CPU par checkpoint pour les 3 paires (section 8). Sur 16 checkpoints, ≈ 38 h
  CPU. Durée réelle en local : la vague 1 a demandé ≈ 3× le temps nominal par job sous 12 processus concurrents. Avec 16
  jobs sur 12 processus (2 tours), compter **≈ 12–15 h** de chaîne CPU. F0 (a) impose le CPU pour le checkpoint 0 ; le
  reste pourrait tourner sur le GPU du cluster (écarts X ci-dessus), à décider.
- À faire avant le lancement, sur décision de l'utilisateur : `max_env_steps`, `checkpoints` et
  `probes.oracle_checkpoints` dans un bloc de vague 2 de `configs/e8/` ; `e8_w2_shuf_nw` dans `batches.LAUNCHABLE` ;
  `--dry-run` d'abord.

### 11.7 Figures et sorties (`results/E8_therapy_relapse/e8_w1_base/`)

- `figures/recovery_<paire>.png` : b, C1, C3 et D en fonction des pas d'environnement.
- `figures/recovery_3abscissae_<paire>.png` : les mêmes, en fonction des pas d'environnement, des pas de gradient et des
  visites de nw.
- `figures/localisation_<paire>.png` : L_dec, quadrant décodé des pas mal localisés (avec leur nombre), part dirigée depuis
  la vraie case.
- `figures/decomposition.png` : effets croyance et cache aux checkpoints de l'oracle.
- `figures/b_vs_directed.png` : points (b, D) de tous les checkpoints sur le nuage d'E6 à 650 000.
- `figures/twin_overlay.png` : b_tab du jumeau et b du world model en fonction des visites de nw.
- Tables : `curves.csv`, `recovery.csv`, `decomposition.csv`, `devices.csv`, `forms.json` (H, R, M, P1–P3, X),
  `identity/F0a_base_*.json`, `twin/forms_therapy_base_*.json`, `twin/twin_base_*.csv`.

### 11.8 En bref

- Formes de contrôle : F0 (43/43 exactes sur CPU pour les deux paires), H, R, M et TW0–TW4 tiennent toutes. Les sondes
  GPU du cluster s'écartent du CPU de 0,08 bit au plus.
- P1 tient chez base_A et base_B : b remonte de −2,3 à −0,07 et −0,02, de façon monotone, et franchit −0,5 à 25 000 et
  10 000 pas.
- Le comportement se corrige vite : D au niveau de la référence à 2 000 et 3 500 pas (150–400 visites de nw), avec C1, C3 et
  la localisation. b suit beaucoup plus lentement (exploratoire : b sous-estime la récupération utilisée).
- P2 n'est pas décidable sur les baselines. Retards de −1 000 et 0 pas, et effet cache résorbé entre 2 000 et 10 000 pas :
  compatible avec un cache faible.
- P3 ne tient pas : D monte sur trois intervalles fins de la grille (s_D 0,45 et 0,44), pas d'un coup.
- Le jumeau des comptages (dilution) reste à −1,4 à 150 000. La croyance stockée dans les poids se réécrit en quelques
  milliers de pas dès que l'expérience ancienne n'est plus rejouée (buffer vide) ; un apprenant qui garde toute son
  histoire resterait bien plus longtemps biaisé. Piste future : thérapie avec l'ancien buffer conservé (rumination).
- Ouvert : P2 et P4 (shuffle, vague 2) ; la part de pas en zone de base_A, pas encore revenue à 150 000 ; la vague 2
  (250 000 pas, 16 checkpoints, 8 points d'oracle proposés) attend la validation de l'utilisateur.

## 12. Préparation de la vague 2 (8 novembre)

L'utilisateur a validé la proposition de la section 11.6, avec des mesures en local sur CPU seulement. Décision (23)
(b462eca).

**Config et déblocage** (c13a9d8) :
- `configs/e8/e8_w2.yaml` reprend `e8.yaml` avec trois valeurs changées : `max_env_steps` 250 000, la grille de 16 points,
  `probes.oracle_checkpoints` en 8 points. Un test vérifie que la config résolue ne diffère que par ces trois valeurs.
- `e8.yaml`, avec lequel la vague 1 a tourné, n'est pas modifié.
- `batches.BATCH_CONFIG` associe la vague 2 à sa config, et `e8_w2_shuf_nw` entre dans `LAUNCHABLE` (ne, sw, se restent
  bloqués).
- `submit_kubeflow.py` (train) et `chain_cpu.py` (sondes) passent `--config` pour ce batch seulement : les lignes de
  commande de la vague 1 sont inchangées (testé). La reprise lit la config du run lui-même.

**Rien d'autre ne change pour l'entraînement par rapport à 95c235b** (`git diff 95c235b` sur `src/`, `configs/`,
`pyproject.toml`, `run_e8.py`, `e8_therapy.py`, `e8_setup.py`, `probes_e8.py`, `batches.py`, `submit_kubeflow.py`) :
- aucun fichier de `src/`, aucune config existante, ni `run_e8.py`, `e8_therapy.py`, `e8_setup.py`, `probes_e8.py` n'a
  changé ;
- seuls changent `batches.py` (la config par batch, `LAUNCHABLE`) et `submit_kubeflow.py` (`--config` transmis,
  docstring), plus le nouveau `e8_w2.yaml` ;
- le dry-run local du batch donne les 4 runs attendus (`--treated-quadrant nw --config configs/e8/e8_w2.yaml`), lançable.

**Formes de la vague 2** (4ce1508, commit seul, avant tout run) :
- P2 sur les shuffle : retards des baselines repris de la vague 1 ; un retard indéfini compte comme non positif en (a) et
  rend (b) non décidable.
- P4 chiffré : images valides à 0 et à T ; f_b soigné si b(0) ≤ −0,5 ; ensemble non soigné b(0) ≤ −1,0 pour f_b comme
  pour f_D ; TI défini si la moyenne soignée est ≥ 0,25 ; seuils 0,5 / 0,2 inchangés.
- Forme H sur l'enregistrement de santé par affectation, et taux de visites de nw.

L'analyse est alignée sur ces formes (60a7bbf). Le `forms.json` de la vague 1 est regénéré : valeurs inchangées, seule la
structure du bloc H change.

**Test de bout en bout** relancé. Le premier passage (`tests/e8` + `tests/twin` : 30 passés, 1 échec, 42 min) a fait
échouer `test_analysis_runs` sur un cas non prévu par la forme H. Dans le mini-run, la masse d'une région peut valoir
`None` : aucun contexte réel de cette région dans le replay. Ce n'est pas une mesure. Une telle région est désormais
rapportée sous `regions_not_measured`, sans compter comme échec (aucune en vague 1). Test ajouté. Résultat du second
passage : section 12.1.

**Tâche planifiée** `E8_w1_cpu_chain` supprimée. Le lanceur de la vague 2 est prêt hors dépôt
(`C:\nrw\e8_wave2\logs_cpu\launch.pyw` : batch `e8_w2_shuf_nw`, extraction attendue dans `C:\nrw\e8_wave2\extracted`,
sorties dans `C:\nrw\e8_wave2\probes_cpu`, 12 processus). Il n'est pas encore enregistré : l'archive n'existe pas.

**Durées estimées.**
- Entraînement : 4 runs × 250 000 pas, soit ≈ 32 h au rythme de la vague 1 (7 900 pas / h / run) et ≈ 40 h au rythme
  d'E5.
- Mesures : 16 jobs (3 paires et la référence B, 12 images, un checkpoint chacun), ≈ 2,4 h CPU nominales chacun. Sur 12
  processus : un premier tour de 12 jobs à ≈ 3× le nominal (concurrence mesurée en vague 1), soit ≈ 7 h ; puis 4 jobs moins
  chargés, ≈ 3–4 h. Total **≈ 10–15 h**, smoke compris.

### 12.1 Tests et commandes de la vague 2

Tests :
- premier passage de `tests/e8` + `tests/twin` : 30 passés, 1 échec, celui décrit plus haut ;
- après le correctif (28a2658), `tests/e8/test_e8_end_to_end.py` : **4 / 4 passés, 47 min** ;
- tests ciblés (`test_e8_wave1_analysis.py`, `test_e8_batches.py`, `test_place_run_files.py`) : **12 / 12**.

Commandes (le push est fait par l'utilisateur ; arrêter `nadim-launcher` avant la soumission) :

```bash
# local (Git Bash, C:\nrw\wt-e8)
git push origin e8

# cluster (terminal du notebook Kubeflow)
cd /scratch/helplessness-worldmodels-e8
git status --short --untracked-files=no      # doit être vide (submit_kubeflow refuse un fichier suivi modifié)
git checkout e8 && git pull --ff-only origin e8
git log --oneline -1                          # le dernier commit poussé
python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w2_shuf_nw --repo-dir /scratch/helplessness-worldmodels-e8 --dry-run
#   attendu : "launchable": true, 4 jobs (shuffle_A_exp, shuffle_E_exp, shuffle_F_exp, shuffle_B_ref),
#   chacun avec --treated-quadrant nw --config configs/e8/e8_w2.yaml
python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w2_shuf_nw --repo-dir /scratch/helplessness-worldmodels-e8
python experiments/E8_therapy_relapse/submit_kubeflow.py --status <dossier superviseur affiché>
# si interrompu :
python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w2_shuf_nw --phase resume --repo-dir /scratch/helplessness-worldmodels-e8 --dry-run
# à la fin (pas de --phase probes : mesures en local, DECISIONS (23)) :
python experiments/E8_therapy_relapse/export_batch.py create --batch e8_w2_shuf_nw

# local, après le transfert de l'archive dans C:\nrw\e8_wave2 (Git Bash, C:\nrw\wt-e8, PYTHONPATH=C:/nrw/wt-e8/src)
mkdir -p C:/nrw/e8_wave2/extracted && tar -xzf C:/nrw/e8_wave2/e8_w2_shuf_nw__<UTC>.tar.gz -C C:/nrw/e8_wave2/extracted
python experiments/E8_therapy_relapse/export_batch.py verify C:/nrw/e8_wave2/extracted
python experiments/E8_therapy_relapse/chain_cpu.py --batch e8_w2_shuf_nw --batch-root C:/nrw/e8_wave2/extracted --out-dir C:/nrw/e8_wave2/probes_cpu --logs C:/nrw/e8_wave2/logs_cpu --smoke
# puis la tâche planifiée E8_w2_cpu_chain (pythonw C:\nrw\e8_wave2\logs_cpu\launch.pyw), ≈ 10-15 h
```

### 12.2 Correction de P2 (censure), avant toute donnée de la vague 2

Sur la remarque de l'utilisateur, la règle « un retard indéfini compte comme non positif » de la section 5
d'EXPECTED_FORMS (4ce1508) traitait une censure comme une absence de retard. Elle est remplacée (0f197f7, commit seul) :
- part dirigée censurée et C3 rétabli : retard ≥ T − t½(C3), une borne inférieure positive ;
- C3 censuré et part dirigée rétablie : une borne supérieure négative ;
- les deux censurés : non décidable pour l'agent.

Les retards censurés sont rapportés comme bornes. (a) et (b) se décident sur ces intervalles : tenue, échec ou non
décidable. « Rien à récupérer » (écart < 0,10 au checkpoint 0) n'est pas une censure et garde sa règle.

`analyze_e8.p2` est aligné (`lag_interval`), avec un test pour chacun des trois cas. Le `forms.json` de la vague 1 et
toutes ses autres sorties sont identiques après la nouvelle exécution.

## 13. Vague 2 (shuffle A, E, F, référence B, thérapie partielle en nw) : analyse (12–13 novembre)

Résultats dans `results/E8_therapy_relapse/e8_w2_shuf_nw/`. Révisions écrites avant toute donnée de la vague 2 :
DECISIONS (24) (P2 (a) sur A et E seuls, a77a635) et (25) (P2 (b), agent léger sans rien à récupérer retiré, e4beaee).
Décision écrite après F0, avant R, H, M et les prédictions : (26) (bb808cb). Aucune règle d'EXPECTED_FORMS n'a changé
après la lecture de R et H. Ligne de résumé de `analyze_e8.py` vue en même temps que R (« P2 False, P4 transfer ») :
signalé à l'utilisateur, rien n'a été ajusté ensuite.

### 13.1 Export, runs et mesures

- Archive `e8_w2_shuf_nw__20261112T124924Z.tar.gz` : 163 315 840 octets, SHA-256 `9422bc4c…0952889` (égale à celle du
  cluster), 104 entrées (les 103 fichiers du manifeste et `MANIFEST.json`), extraite hors dépôt
  (`C:\nrw\e8_wave2\extracted`). `export_batch.py verify` : **103 fichiers, 0 problème**. Les 4 runs sont `done` à
  250 000 pas (149 846 pas de gradient), environ 32,5 h d'entraînement chacun sur le cluster.
- Sondes sur CPU local seulement (DECISIONS (23)) : smoke (1 job réduit, 21 min, supprimé ensuite), puis la tâche
  planifiée `E8_w2_cpu_chain` (12 processus) : **16 / 16 jobs `done`, 9 h 47** (13 h 57 → 23 h 44 UTC). Sorties hors
  dépôt dans `C:\nrw\e8_wave2\probes_cpu`. La tâche est supprimée.
- L'analyse lit les sondes CPU des deux vagues (retards des baselines pour P2) et les runs des deux vagues (H).

### 13.2 Formes de contrôle

| Forme | Résultat |
|---|---|
| F0 (a) checkpoint 0 = campagne à 650 000, CPU, égalité exacte | **F 43 / 43, E 43 / 43 ; A 20 / 20** sur les grandeurs de l'exposé seul (`--exposed-only`, décision (26)) |
| F0 (a) de A, comparaison complète (gardée : `identity/F0a_shuffle_A_full.json`) | 22 / 43 : tous les écarts portent sur la référence ou sur b. La campagne appariait A à la **référence A** (seed02), E8 l'apparie à la référence B (design). Erratum daté dans EXPECTED_FORMS §1. La référence B est vérifiée exacte par E et F |
| F0 (c) I de l'exposé au pas 1 contre les enregistrements d'E5 (GPU du cluster), 0,05 bit | F 12 / 12 (max 0,043) ; E 10 / 12 (affectations 6 : +0,053, 7 : +0,113) ; A 9 / 12 (2 : +0,089, 8 : +0,052, 11 : −0,075). **Écart de device CPU / GPU, sans verdict** (décision (26)) |
| H santé de l'entraînement | Les 4 runs `done` à 250 000, `switch_step` = 0 partout, garde ≥ 3,27 bits (seuil 1,0), succès ≥ 0,90. Référence B, A et E : **tiennent** (transitoires avant 5 000 : A à 1 000, couloir 0,877 et sw 0,856 ; E à 500, se 0,857 et couloir 0,893). **shuffle F : en échec tel qu'écrit** : masse sur le vrai successeur < 0,90 dans l'affectation 0, en ne et dans le couloir, de 5 000 à 25 000 pas (minimum 0,857) ; au-dessus de 0,90 dès 40 000. Ne bloque pas la lecture de F (utilisateur) ; fenêtre signalée en 13.6 |
| R la référence reste contrôlable | base_A, base_B (vague 1) : tiennent. **Référence B : tient sur les affectations soignées (0–2), échoue sur les non soignées (3–11)** (13.3) |
| M l'exposé ne se dégrade pas en nw | **tient** pour les 5 paires : ρ (b, C3, D) = 0,92 / 0,96 / 0,85 (A), 0,97 / 0,91 / 0,81 (E), 0,98 / 0,89 / 0,16 (F) ; drawdown max. 0,037 (b de E) |

### 13.3 R : oubli partiel des arrangements non entraînés (résultat)

La référence B s'entraîne 250 000 pas sur les 3 affectations de la thérapie (UD en nw) seulement.
- Affectations soignées (0–2) : I_réf de 2,24 à 2,30, masse ≥ 0,93, C1, C3 et D stables, à tous les checkpoints.
- Affectations non soignées (3–11) : I_réf passe sous log2 5 − 0,15 = 2,17 à partir de 10 000–25 000 pas.
  - **3, 6 et 9, durablement** : I_réf à T = 1,63, 2,07 et 2,13 (minimum 1,59 sur l'affectation 9 à 25 000) ; la masse
    de la référence sur le vrai successeur tombe jusqu'à 0,03 (affectation 3 à T). C3 sort des bornes à T sur 3 et 9
    (0,99 → 0,89 et 0,91), C1 sur 9 (1,00 → 0,91).
  - Les 6 autres : un creux entre 15 000 et 40 000 pas (I_réf 2,01–2,16), puis retour dans les bornes.
  - **3, 6 et 9 sont exactement les affectations où nw a le rôle CD** (la couleur de T_ctrl en nw) ; la thérapie ne montre
    nw qu'avec la couleur de T_inc (rôle UD). Dans les 6 autres, nw est neutre.
- D_réf reste dans ± 0,05 de sa valeur au checkpoint 0 sur toutes les images (moyenne non soignée 0,92–0,93 ; par image
  à T ≥ 0,89 : 3 : 0,93 → 0,89, 6 : 0,94 → 0,91, 9 : 0,93 → 0,90). Le comportement de la référence ne suit pas la dérive de
  sa croyance.
- **Validité des points** (critère d'E4 sur la référence : masse ≥ 0,90 et |I_réf − log2 5| ≤ 0,15 ; `validity.csv`,
  `validity_counts.csv`) : 120 points invalides sur 576 (3 paires × 12 images × 16 checkpoints). Aucun sur les
  affectations soignées. Affectation 3 invalide dès 15 000, 6 dès 5 000 (2 paires sur 3) puis 7 500, 9 dès 10 000, jusqu'à
  T. Les 6 autres : invalides à 15 000–40 000 seulement. **À 25 000, aucun point non soigné n'est valide.** b, f_b, TI_b,
  les courbes et les franchissements ne lisent que les points valides (c'est ce que faisait déjà l'analyse : `walks_valid`
  est ce critère) ; les moyennes non soignées portent donc sur un ensemble d'images qui change (9 jusqu'à 3 500, 0 à
  25 000, 6 dès 60 000).
- **Exploratoire, l'exposé** (`drift.csv`, `figures/validity_drift.png`) : même motif. Sur 3, 6 et 9, la masse de l'exposé
  à T vaut 0,23 / 0,44 / 0,33 (A), 0,00 / 0,66 / 0,71 (E), 0,55 / 0,32 / 0,23 (F), contre 0,94–0,98 sur les autres non
  soignées ; I_exp à T 1,67–2,14 contre 2,20–2,27. Les deux modèles perdent la prédiction des arrangements où nw porte la
  couleur de T_ctrl.

### 13.4 Prédictions, telles que calculées

Abscisses : 500 / 1 000 / 2 000 / 3 500 / 5 000 / 10 000 / 40 000 / 60 000 / 100 000 pas d'environnement = 146 / 446 /
1 046 / 1 946 / 2 846 / 5 846 / 23 846 / 35 846 / 59 846 pas de gradient. Courbes soignées : moyenne des 3 affectations
(toutes valides à tous les checkpoints).

**P1 tient chez A, E et F** (et chez base_A, base_B en vague 1). b (soigné) : −2,14 → +0,03 (A), −2,14 → −0,01 (E),
−2,05 → −0,03 (F) ; t½ de b à 1 000, 500 et 1 000 pas ; franchissement de −0,5 à 1 000, 2 000 et 2 000 ; pas de rechute.

**P2 en échec, sur (a) et sur (b).**

| | t½ C3 | t½ D | retard |
|---|---|---|---|
| shuffle A | 1 000 | 1 000 | **0** |
| shuffle E | 500 | 500 | **0** |
| shuffle F | 500 | 500 | 0 |
| base_A (vague 1) | 2 000 | 1 000 | −1 000 |
| base_B (vague 1) | 2 000 | 2 000 | 0 |

- (a) révisée (24) : A et E ont un retard de 0, non positif : **en échec**.
- (b) : min des bornes hautes de A et E (0) ≤ max des bornes basses des légers (0) : **en échec**. Aucun agent léger
  retiré (25).
- Règle d'origine (4 sur 5) : 0 positif, **en échec par construction** (retards des baselines −1 000 et 0).
- Résolution : un retard de 0 veut dire le même checkpoint (500 ou 1 000 pas) ; un retard plus fin que la grille n'est pas
  mesurable.
- Décomposition (descriptive ; l'oracle k = 1 ne tourne que sur l'image d'entraînement, assignment01, alors que D_réf et
  D_exp sont des moyennes soignées) : effet cache au checkpoint 0 = 0,284 (A), 0,223 (E), 0,075 (F), contre 0,064 (base_A)
  et 0,165 (base_B). À 2 000 pas, premier point d'oracle : −0,019 (A), −0,014 (E), −0,014 (F), contre −0,009 (base_A) et
  **0,086 (base_B)**. Rapport au dernier point d'oracle sur la valeur à 0 : −0,03 (A), −0,04 (E), −0,11 (F). La forme
  attendue (cache de A et E résorbé plus lentement) n'est pas observée : il est résorbé dès 2 000 pas, plus tôt que chez
  base_B.
- **Effet cache de A au checkpoint 0** (`cache_A.json`, image d'entraînement) : **0,279 avec la référence B** (D_réf 0,928,
  D_oracle1 0,649), **0,242 avec la référence A** de la campagne (D_réf 0,892), le 0,24 cité d'E7.

**P3 : tient chez F, ne tient pas chez A et E** (tient si s_D ≥ 0,5 et s_b < 0,5).

| | gain D | s_D | gain b | s_b | verdict |
|---|---|---|---|---|---|
| shuffle A | 0,55 | 0,43 | 2,17 | 0,42 | ne tient pas (s_D) |
| shuffle E | 0,43 | 0,81 | 2,13 | 0,51 | ne tient pas (s_b) |
| shuffle F | 0,42 | 0,92 | 2,01 | 0,48 | **tient** |

Limite écrite d'avance : la récupération soignée a lieu entre 0 et 2 000 pas, sur 3 intervalles de la grille. Le saut de D
chez E et F est l'intervalle 0–500 ; le s_b de E (0,51) est juste au-dessus du seuil.

**P4 : « transfer » (3 agents sur 3).**

| | TI_D | TI_b | images soignées / non soignées | verdict |
|---|---|---|---|---|
| shuffle A | 0,985 | 0,981 | 3 / 6 | transfer |
| shuffle E | 0,983 | 0,999 | 3 / 6 | transfer |
| shuffle F | 0,977 | 1,000 | 3 / 6 | transfer |

- Les 6 images non soignées qui entrent sont celles où nw est neutre (4, 5, 7, 8, 10, 11) ; 3, 6 et 9 (nw = CD) sont
  exclues, invalides à T. Le transfert n'est donc lu que sur les arrangements où nw est neutre.
- D_réf sur les non soignées (vérifié) : 0,92–0,93 en moyenne à tous les checkpoints valides, dans ± 0,05 de sa valeur à
  0 par image : g_D et TI_D ne sont pas tirés par une baisse de la référence.
- Franchissements (`transfer.csv`, `dose.csv`), en pas d'environnement / pas de gradient / visites de nw de l'exposé :

| | g_D ≤ 0,05, soignées | g_D ≤ 0,05, non soignées | b > −0,5, soignées | b > −0,5, non soignées |
|---|---|---|---|---|
| shuffle A | 2 000 / 1 046 / 232 | 3 500–10 000 / 1 946–5 846 / 489–1 677 | 1 000–2 000 / 446–1 046 / 125–232 | 10 000–60 000 / 5 846–35 846 / 1 677–10 201 |
| shuffle E | 1 000 / 446 / 71 | 40 000–60 000 / 23 846–35 846 / 6 443–9 654 | 2 000 / 1 046 / 203 | 15 000–60 000 / 8 846–35 846 / 2 190–9 654 |
| shuffle F | 500 / 146 / 58 | 40 000–100 000 / 23 846–59 846 / 6 632–16 910 | 2 000 / 1 046 / 252 | 40 000–60 000 / 23 846–35 846 / 6 632–10 097 |

- Toutes les images franchissent avant T (6 / 6 non soignées, 3 / 3 soignées, pour chaque agent). Aucun point non soigné
  n'étant valide à 25 000, un franchissement lu à 40 000 veut dire « entre 15 000 et 40 000 ».
- Pour D, le transfert demande 2 à 5 fois plus de pas que la récupération soignée chez A, 40 à 200 fois plus chez E et F.

### 13.5 Dose de thérapie (visites de nw)

Taux de visites de nw par pas (`visit_rates.csv`) :

| | 0–500 | 0–5 000 | 0–250 000 |
|---|---|---|---|
| shuffle A | 0,074 | 0,158 | 0,169 |
| shuffle E | 0,064 | 0,129 | 0,173 |
| shuffle F | 0,116 | 0,162 | 0,168 |
| référence B | 0,194 | 0,184 | 0,172 |

Les exposés évitent nw au début (6–12 % des pas contre 19 % pour la référence), puis rejoignent son taux vers 5 000 pas.
Malgré cela, la dose qui suffit est faible : t½ de D soigné à 125 (A), 32 (E) et 58 (F) visites, contre 87 (base_A) et
227 (base_B) en vague 1. Le transfert non soigné, lui, arrive après 489–1 677 visites de nw (A) et 6 443–16 910 (E, F).

### 13.6 Fenêtre de H pour shuffle F (5 000–25 000 pas)

Checkpoints concernés : 5 000, 7 500, 10 000, 15 000, 25 000 (colonne `in_F_H_window` de `dose.csv`, bande grise de
`figures/dose_crossings.png`). Ce qui tombe dedans : **les t½ non soignés de F** (b, C1, C3 et D à 10 000 ; courbes
moyennes sur 7 images valides). Hors de la fenêtre : les t½ soignés (500–1 000), les franchissements soignés (500 et
2 000), les franchissements P4 non soignés (40 000–100 000), P1 et TI (lus à T), le retard de P2 (au checkpoint 500).
L'échec de H porte sur l'enregistrement de santé de l'affectation 0 (soignée), en ne et dans le couloir, pas en nw.

### 13.7 Figures et sorties (`results/E8_therapy_relapse/e8_w2_shuf_nw/`)

- `figures/recovery_<paire>.png`, `recovery_3abscissae_<paire>.png`, `localisation_<paire>.png`, `decomposition.png`,
  `b_vs_directed.png` (`analyze_e8.py`, les 5 paires).
- `figures/validity_drift.png` : I et masse au pas 1 de la référence et de l'exposé, par image (paire A).
- `figures/dose_crossings.png` : franchissements g_D ≤ 0,05 par image, en pas et en visites de nw.
- Tables : `curves.csv`, `recovery.csv`, `decomposition.csv`, `transfer.csv`, `forms.json` (`analyze_e8.py`) ;
  `validity.csv`, `validity_counts.csv`, `drift.csv`, `dose.csv`, `visit_rates.csv`, `cache_A.json`
  (`w2_supplements.py`) ; `identity/F0a_shuffle_*.json`.

Commandes (Git Bash, `C:\nrw\wt-e8`, `PYTHONPATH=C:/nrw/wt-e8/src`) :

```bash
python experiments/E8_therapy_relapse/check_identity.py --probes <probes__shuffle_B_ref__k000000__*> --pair shuffle_A \
    --image assignment01 --e6 <e6__f1_A__*> --c12 <e7d7__f1_A__natural__natural_training__*> \
    --oracle <e7diag__f1_A__lookahead__*> --e5-records <run E5 seed02, experimental> --exposed-only \
    --out results/E8_therapy_relapse/e8_w2_shuf_nw/identity/F0a_shuffle_A.json
python experiments/E8_therapy_relapse/analyze_e8.py \
    --probes C:/nrw/e8_wave1/probes_cpu/E8_therapy_relapse/probes__* C:/nrw/e8_wave2/probes_cpu/E8_therapy_relapse/probes__* \
    --runs C:/nrw/e8_wave1/extracted/results/E8_therapy_relapse/grid12x12__* C:/nrw/e8_wave2/extracted/results/E8_therapy_relapse/grid12x12__* \
    --out results/E8_therapy_relapse/e8_w2_shuf_nw --e6-cloud "C:/nrw/wt-e6/scratch/all/e6_*/E6_phenotypes_latent/e6__*"
PYTHONPATH="C:/nrw/wt-e8/src;C:/nrw/wt-e8/experiments/E8_therapy_relapse" python experiments/E8_therapy_relapse/w2_supplements.py \
    --probes C:/nrw/e8_wave2/probes_cpu/E8_therapy_relapse/probes__* \
    --runs C:/nrw/e8_wave2/extracted/results/E8_therapy_relapse/grid12x12__* \
    --analysis results/E8_therapy_relapse/e8_w2_shuf_nw --campaign-e6-A <e6__f1_A__*>
```

### 13.8 En bref

- Formes : F0 exact (F, E 43/43 ; A 20/20 côté exposé, la campagne l'appariait à la référence A) ; M tient ; H tient sauf
  F (masse 0,86–0,89 en ne et couloir, affectation 0, 5 000–25 000 pas) ; F0 (c) : écart CPU / GPU jusqu'à 0,11 bit.
- R échoue sur les non soignées : la référence oublie en partie les arrangements non entraînés, surtout les 3 où nw porte
  la couleur de T_ctrl (I_réf jusqu'à 1,6, masse jusqu'à 0,03) ; l'exposé fait de même ; D_réf reste stable.
- P1 tient (A, E, F) : b soigné de −2,1 à ≈ 0, t½ en 500–1 000 pas, plus vite que les baselines.
- P2 échoue sur (a) et (b) : retard 0 chez A et E ; leur effet cache (0,28 et 0,22 au départ) est résorbé dès 2 000 pas.
- P3 tient chez F seulement ; la récupération soignée tient dans 3 intervalles de la grille.
- P4 : transfert chez les 3 agents (TI_D 0,98), lu sur les 6 arrangements où nw est neutre ; il demande 2 à 200 fois plus
  de pas que la récupération soignée (A le plus rapide).
- Les exposés évitent nw au début (6–12 % des pas contre 19 %), mais 30 à 125 visites suffisent à corriger D.
