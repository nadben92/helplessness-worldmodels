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
