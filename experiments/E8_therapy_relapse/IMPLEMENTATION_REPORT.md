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
