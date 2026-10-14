# Audit de préparation GPU — E3 (lecture seule)

Date : 2026-09-25. Branche `e2/online-tabular`, HEAD `724bf64` **plus un arbre de travail non commité** (voir B2). Environnement d'audit : Windows, Python 3.14.0, torch 2.14.0+cpu.
Aucune modification du code. Les vérifications ont été faites par lecture, grep, et quatre petits scripts CPU (dans le scratchpad, hors dépôt).

Convention : **[vérifié]** = lu dans le code ou mesuré/testé sur CPU ; **[supposé]** = déduit de la documentation PyTorch/CUDA ou du raisonnement, non testable ici (pas de GPU).

---

## 1. Résumé

**Le code peut-il tourner sur GPU en l'état ? Non.** Le cœur du modèle est probablement prêt, mais le point d'entrée ne l'est pas.

1. **`run_mini.py` ne place jamais le modèle sur un device** et la config n'a pas de champ `device` : `RSSM(rssm_cfg)` (`run_mini.py:336`) reste sur CPU, comme `templates_t` (`:206`). Sur l'A100, le run tournerait sur CPU **sans erreur ni avertissement**. `resolve_device` (`rssm.py:33`) existe mais n'est utilisé que par `profile_throughput.py`. [vérifié]
2. **Tout E3 est absent de GitHub** : 26 chemins non suivis et 4 modifiés (`git status`). L'advisor ne peut pas lancer ce qui n'est pas poussé. [vérifié]
3. **Rien de brut n'est sauvegardé** : sur un run réussi, seuls `MINI_REPORT.md` et 4 PNG sont écrits, à la toute fin, et ils écrasent des chemins fixes du dossier `docs/`. Pas de config résolue, pas de métriques brutes, pas de checkpoint, pas de reprise. Un crash après plusieurs heures perd tout, y compris les graines déjà terminées. [vérifié]
4. **À 10⁶ pas, deux points explosent** : le replay fait 15,6 Go de RAM par graine (311 Go pour 20), et `z_only_probe` déroule tout le buffer en un seul tenseur (≈ 62 Go en float32 pour les entrées, autant pour les images décodées). [lu + calculé]
5. **Le noyau (RSSM, pertes, pas d'entraînement, sondes) est indépendant du device** : chaque création de tenseur hérite d'un device d'entrée, chaque passage vers numpy est précédé de `.cpu()`. Déplacer le modèle et les tables suffirait probablement à le faire tourner sur CUDA. [lu ; non testé sur CUDA]
6. **Bonne nouvelle pour la mémoire** : le rendu est une fonction pure de l'état. L'observation stockée est *identique au bit près* à la ligne de la table pré-rendue de 144 images, sur 3000 pas × 2 graines (les deux rotations). Le buffer peut passer de 15,6 Go à ≈ 10 Mo par graine sans changer son contenu. [vérifié]
7. **Reproductibilité** : bit-exacte sur CPU (même machine, même nombre de threads) ; sur GPU elle ne le sera pas. Point délicat : sur A100, les convolutions cuDNN utilisent TF32 par défaut, alors que la phase de séparation dépend de très petits écarts de probabilités. [supposé]
8. **Ce que ça coûte** : 3 bloquants pour lancer sur GPU (B1 à B3), 4 de plus pour la pleine échelle (B4 à B7), tous de taille S à M sauf la reprise.

---

## 2. Constats par axe

Colonnes : constat | statut | localisation | correctif proposé | taille (S = quelques lignes, M = une demi-journée à une journée, L = plusieurs jours).

### Axe 1 — Gestion du device

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 1.1 | `resolve_device("auto"\|"cpu"\|"cuda")` existe et n'a aucun `.cuda()` en dur. Il n'est appelé que par `profile_throughput.py` (`--device`, défaut `cpu`). `run_mini.py` n'a ni argument `--device`, ni champ de config (`configs/e3/*.yaml` : aucun `device`), ni `.to(device)`. Le modèle et les tables restent sur CPU. [vérifié] | **problème (B1)** | `rssm.py:33-38` ; `profile_throughput.py:34` ; `run_mini.py:204-206,336` | Ajouter `runtime.device` dans `mini.yaml` et `--device` dans `run_mini.py` ; construire le RSSM **sur CPU avec la graine** puis `.to(device)` (les poids initiaux restent identiques à ceux du CPU) ; déplacer `background_t` et `templates_t` ; afficher le device résolu au démarrage. | S |
| 1.2 | Aucun `"cpu"` ni `.cuda()` en dur dans le code E3 hors `resolve_device`. Chaque passage vers numpy est précédé de `.detach().cpu()`. [vérifié : grep] | OK | `agent.py:111-112` ; `probe.py:63-70,113,125,145-146,175,249,341-356,389` ; `loop.py:396` ; `run_mini.py:301` | — | — |
| 1.3 | Créations de tenseurs : toutes celles du chemin modèle prennent leur device d'une entrée (`device=` explicite ou `*_like`). Pas de constante créée sans device dans les `forward`. [vérifié : grep des 20 sites] | OK | `rssm.py:157,183,234,282,331,342` ; `loop.py:53,295,332-333` ; `agent.py:82,103` ; `probe.py:56,115-116,140-141,171-181,247,296` | — | — |
| 1.4 | **Contrat de device incohérent** dans les sondes : `canonical_probe` déplace lui-même les templates (`probe.py:48`), alors que `canonical_separation` et `latent_usage_diagnostic` utilisent `templates.device` et passent `templates` tel quel à l'encodeur (`:349`, `:385`). Avec un modèle CUDA et des templates CPU (ce que fait `run_mini` aujourd'hui) : erreur de device. [supposé : par construction] | problème (lié à B1) | `probe.py:333-362,385`, `run_mini.py:206` | Dans ces fonctions : `device = next(rssm.parameters()).device` puis `templates.to(device)`, comme `canonical_probe`. | S |
| 1.5 | `ae_isolation.py` et `compare_*.py` n'ont aucun support de device (RSSM et templates CPU). Outils d'expérimentation, pas de chemin de lancement GPU prévu. [vérifié] | OK (hors périmètre) | `ae_isolation.py` | Aucun tant qu'on ne les lance pas sur GPU. | — |
| 1.6 | Mélanges CPU/GPU : indexation d'un tenseur GPU par des indices créés sur le même device (`probe.py:181`). `torch.where` avec masques dérivés de l'entrée (`rssm.py:335-339`). Rien de suspect à la lecture. [lu ; non testé sur CUDA] | OK (à vérifier au smoke test) | `probe.py:181-185` ; `rssm.py:328-339` | — | — |
| 1.7 | `torch.load` : **absent** (aucun chargement de checkpoint n'existe, voir 4.5). `torch.save` seulement à l'échec (`run_mini.py:93`). Pas de `torch.Generator` : tout l'aléa torch passe par le RNG global. `fork_rng()` dans `ae_isolation.py:130` (fork tous les devices CUDA si CUDA est initialisé ; sans effet ici). [vérifié] | OK (à traiter avec 4.5) | `run_mini.py:93` | Quand la reprise sera écrite : `map_location=device`. | S |
| 1.8 | **Environnement et transfert** : l'env renvoie du numpy `uint8` (H, W, 3). Le buffer est en numpy. Le transfert se fait à chaque pas d'entraînement dans `to_chw_float` : `torch.as_tensor(x, dtype=float32, device=device)`. La conversion en float32 se fait côté hôte avant la copie, donc on transfère 4× plus d'octets que nécessaire (16 Mo → 64 Mo pour un batch 16×64) ; pas de mémoire épinglée. Mesuré sur CPU : 24,7 ms pour la conversion d'un batch 16×64. [vérifié] | problème (souhaitable) | `loop.py:51-54` ; `probe.py:139,170,226` | Voir 2.4 : avec la table pré-rendue sur le device, le transfert disparaît. À défaut : transférer en `uint8` puis convertir sur le device. | S |

### Axe 2 — Mémoire du replay buffer

**Ce qui est stocké** (`replay.py:63-70`, RAM hôte, numpy) : `x` uint8 (72,72,3) = 15 552 o ; `a` int64 = 8 o ; `r` float32 = 4 o ; `terminal`, `truncated`, `is_first` bool = 3 o ; `s` int64 = 8 o. **Total 15 575 o/pas** (mesuré). Pas d'observation suivante : `x_{t+1}` est le pas suivant du buffer. Capacité exacte `budget + prefill` (`for_run`, mode strict), allouée d'un coup par `np.zeros`.

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 2.1 | 15,6 Go de RAM par graine à 10⁶ pas ; 62 Go pour 4 graines ; 311 Go pour 20 (table §3). Rien n'est en VRAM. [calculé sur la mesure de 15 575 o/pas] | **problème (B6, pleine échelle)** | `replay.py:63-70` | Table pré-rendue (2.3-2.4). | M |
| 2.2 | `chronological_windows` copie **tout** le buffer par indexation avancée (15,6 Go de plus à 10⁶ pas), puis `z_only_probe` convertit tout en float32 et déroule tout en un seul appel, en gardant aussi `decoded` : ≈ 62 Go + 62 Go sur le device, + 62 Go côté hôte pendant `to_chw_float`. Sans problème à 5 256 pas (0,3 Go), impossible à 10⁶. [lu + calculé] | **problème (B4, pleine échelle)** | `replay.py:127-146` ; `probe.py:158-178` ; `run_mini.py:438-439` | Traiter les fenêtres par blocs (p. ex. 512 séquences), n'accumuler que `post_probs`/`z` (petits), ne pas conserver `decoded`. | M |

**Question clé : le rendu est-il une fonction déterministe de l'état ?** **Oui.** [vérifié : lecture + test]

Entrées de la fonction de rendu (`allocentric_pixel_backend.py:93-101`, `render(s, layout, themes, k, grid_lines)`) :

| Entrée | Fixée par | Aléa ? |
|---|---|---|
| `s` (indice d'état, 0..143) | l'environnement (position de l'agent) | non |
| `layout` (`ZoneLayout` : affectation zone → catégorie de thème) | la rotation choisie par la graine (`rotation_for_seed_index`) | non (table fixe) |
| `themes` (couleurs par catégorie, couleur du but, couleur du marqueur de l'agent) | `build_themes(..., rng=default_rng(seed))`, appelé **une fois** au début du run (`run_mini.py:199`) | oui, mais tiré une seule fois par run |
| `k` (échelle, 6) et `grid_lines` (false) | la config | non |

Il n'y a **aucun RNG dans `render`, `_base_image`, `_draw_agent_marker` ni dans l'adaptateur** ; le backend lève une exception si `theme_seed` n'est pas `None` (pas de tirage de thème par épisode).
**Test :** pour les graines 0 et 5 (rotations 0 et 1), 3000 pas de la vraie collecte (`collect_and_count`), avec reset. `replay._x[:3000] == table[replay._s[:3000]]` : **égalité exacte sur tous les pas**, 139 et 138 états distincts visités sur 144. Deux appels de `render` avec les mêmes arguments sont identiques. La table complète fait 144×72×72×3 = **2,24 Mo**.

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 2.3 | **Faisabilité : oui.** Stocker seulement l'indice d'état (int16 suffit) et reconstruire `x` par lecture dans la table de 144 images placée sur le device. Buffer à 10⁶ pas : ≈ **10 Mo** par graine (10 o/pas) au lieu de 15,6 Go. | à faire (B6) | `replay.py` | voir 2.4 | M |
| 2.4 | Détail de ce que ça changerait et de ce que ça ne doit pas changer (ci-dessous). | — | — | — | — |
| 2.5 | Aucune réduction de capacité proposée ; le buffer garde chaque transition (condition B du jumeau tabulaire). | OK | — | — | — |

**Ce que la table pré-rendue changerait dans le code :**
- `ReplayBuffer` : supprimer `_x` (le tableau de 15,6 Go) ; recevoir la table `(144,H,W,3) uint8` à la construction ; `add()` garde son paramètre `x` mais **vérifie** `x == table[s]` puis l'oublie.
- `sample()` et `chronological_windows()` : reconstruire `x` par lecture dans la table. Pour éviter le transfert hôte → device, cette lecture doit se faire **sur le device** : `ModelBatch.x` passerait de `np.ndarray` à un tenseur (ou reste numpy pour les tests), et `to_chw_float` (`loop.py:51`, `probe.py:139,170,226`) devrait accepter les deux. La table peut être stockée déjà en float32 divisée par 255 : les valeurs sont **identiques bit à bit** à celles de `to_chw_float`.
- Gain de vitesse annexe : disparaissent la boucle Python de `sample` (9,3 ms pour 16×64), la conversion hôte (24,7 ms) et la copie de 64 Mo par pas.
- La collecte peut aussi utiliser la table (`render` seul coûte 0,47 ms sur les 0,75 ms mesurés par pas de collecte).

**Ce que ça ne doit pas changer :**
- **Le contenu logique du buffer** : chaque pas est stocké, capacité `budget + prefill` stricte, ordre chronologique et gestion du wraparound.
- **L'échantillonnage** : mêmes appels `rng.integers(0, n_valid_starts, size=batch)`, donc mêmes fenêtres pour une graine donnée.
- **Les tables de comptage** `N[s,a,s']` : calculées dans `collect_and_count` à partir de `info["state"]`, indépendamment du buffer (`run_mini.py:216-225`). Le jumeau tabulaire voit exactement les mêmes données.
- **L'étanchéité aux étiquettes** : `ModelBatch` n'a aucun champ `s` par conception (`replay.py`, tests `test_replay.py`). L'indice qui sert à lire la table est *numériquement* `s` ; il doit donc rester **privé au buffer**, et seul le tenseur d'images doit sortir dans `ModelBatch`. Sinon la règle « l'étiquette n'entre jamais dans la boucle de l'agent » devient invérifiable au niveau du type.
- **Précondition à faire échouer bruyamment** : cette équivalence n'est vraie que tant que l'observation est une fonction pure de `(s, config fixe du run)`. Si une variante E4-E6 introduit du bruit pixel, des thèmes par épisode, ou des perturbations de rendu, la table devient fausse. L'assertion `x == table[s]` dans `add()` (≈ 15 Ko comparés par pas, quelques µs) le détecte immédiatement et évite de stocker de mauvaises données en silence.

### Axe 3 — Reproductibilité

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 3.1 | **Graines** : `np.random.default_rng(seed)` (unique, partagé entre la collecte, l'échantillonnage du replay et les évaluations : `run_mini.py:314`) ; `torch.manual_seed(seed)` (`:318`, qui sème aussi CUDA d'après la doc PyTorch [supposé]) ; RNG de la dynamique séparé (`torus_env.py:95`) ; `pair_rng=[seed,4]` ; mesures `seed+1000/2000/3000` ; thèmes `default_rng(seed)`. Le module `random` n'est pas utilisé (grep). Aucun `torch.Generator` : tous les tirages latents (`Categorical.sample`, Gumbel, straight-through partiel) partagent le RNG global, donc toute mesure qui tire entre deux pas décale la suite (déjà vu avec l'instantané de `ae_isolation`). **Une graine suffit à reproduire un run sur la même machine et le même nombre de threads, sur CPU** (vérifié : reproduction déterministe de l'effondrement). **Pas d'un CPU à un GPU** : les flux aléatoires diffèrent. [vérifié / supposé] | à vérifier (4a) | `run_mini.py:314,318,347,418-423` | Flux séparés par composant, avec un `torch.Generator` dédié pour l'aléa latent (déjà prévu « Phase 4a » dans `run_mini.py:314`). Conforme à `CLAUDE.md` (RNG séparés). | M |
| 3.2 | **Opérations CUDA non déterministes** : aucune des habituelles dans le code torch (grep : pas de `scatter_add`, `index_add`, `index_put` avec accumulation, `bincount` torch, `interpolate`, `embedding`). **Seul suspect : le backward de `Conv2d`/`ConvTranspose2d` via cuDNN** (encodeur et décodeur), non déterministe sauf réglage explicite. `Categorical.sample` (multinomial) est déterministe à graine fixe sur un device donné. [supposé : doc PyTorch] | à vérifier (B7) | `rssm.py:95-98,128-134` | Politique 4b : `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `cudnn.benchmark=False` ; mesurer le surcoût ; décider avant les runs cluster. | S (décision) + M (validation) |
| 3.3 | **TF32 sur A100** : par défaut, les convolutions cuDNN utilisent TF32 (`cudnn.allow_tf32=True`) et les matmul non (`matmul.allow_tf32=False` depuis PyTorch 1.12). Pas d'AMP dans le code (float32 partout). La phase « probabilités » repose sur de très petits écarts (posterior à 23,6 bits sur 24, normes de logits de 0,7, cf. DECISIONS.md) : la précision réduite des convolutions peut changer la **dynamique**, pas seulement les derniers bits. [supposé] | à vérifier | (aucun réglage dans le code) | Fixer explicitement `allow_tf32=False` pour les deux drapeaux au démarrage et l'enregistrer dans les métadonnées ; mesurer le coût sur A100. | S |
| 3.4 | **Tables de comptage** : `N` est un `int64` numpy sur CPU (`run_mini.py:216-225`) ; `kernel_probe` utilise `np.bincount` numpy (`kernel_probe.py:67`). Aucune accumulation flottante sur GPU pour les comptages : à l'abri. Les mesures Monte Carlo sont en numpy avec RNG explicites ; elles restent déterministes *à entrées égales* (les probabilités, elles, viennent du device). [vérifié] | OK | `run_mini.py:216-225` ; `kernel_probe.py` | — | — |
| 3.5 | **Ce qui est enregistré par run** : `num_threads`, `torch_version`, `git_commit` (`run_mini.py:66-77`), dans le rapport et dans les diagnostics d'échec (qui contiennent aussi config, graine, erreur). **Manquent** : version de Python, version de CUDA/cuDNN, nom du GPU et device utilisé, drapeau « arbre de travail modifié » (`git rev-parse HEAD` seul donne le même hash pour un arbre sale), état des drapeaux TF32/déterminisme, et surtout la **config résolue copiée à côté des résultats** (exigée par `CLAUDE.md`). `helplessness.runs.make_run_dir`/`write_run_meta` existent et font exactement cela (E1+), mais `run_mini.py` ne les utilise pas. [vérifié] | problème (B3) | `run_mini.py:66-77,617-825` ; `src/helplessness/runs.py` | Utiliser `make_run_dir` ; y écrire `config.yaml` et un `meta.yaml` enrichi (python, torch, CUDA, cuDNN, GPU, dirty, flags). | M |

### Axe 4 — Lançabilité par un tiers

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 4.1 | **Commande de lancement** : `python experiments/E3_worldmodel_4zones/run_mini.py [--seeds ...]` (le docstring ne donne que la variante PowerShell `.venv\Scripts\python.exe`). Le chemin de config est codé en dur (`configs/e3/mini.yaml`) : **pas de `--config`, `--device`, `--out-dir`, `--max-steps`**. Aucune commande de balayage (graines × conditions) : le `Makefile` ne contient que des cibles E0. Prérequis implicites : `pip install -e ".[e3]"` *en mode éditable* (`experiment_loader.py:29` déduit `REPO_ROOT` de `__file__`) et lancement comme **script** (`import env`, `import probe` nus, `run_mini.py:36-37`, ne marchent pas avec `-m`). Le `README` d'E3 est périmé (« Phase 1 in progress »). [vérifié] | **problème (B1/B3)** | `run_mini.py:19,36-37,155,828-838` ; `Makefile` ; `experiments/E3_worldmodel_4zones/README.md` | Ajouter `--config/--device/--out-dir/--max-steps` ; un script `scripts/run_e3_sweep.sh` qui boucle sur les graines en processus séparés ; documenter la commande unique en bash et PowerShell ; mettre le README à jour. | M |
| 4.2 | **Dossiers de sortie** : `run_mini.py` écrit `docs/e3/MINI_REPORT.md` (`:823`) et `docs/e3/mini_report_figures/{loss_curves,iz_by_zone,separation_curves}.png` (`:619-621`) à **chemins fixes**, dans l'arbre suivi par git. Deux processus (deux graines en parallèle) ou deux relances **s'écrasent** ; aucun dossier par run. Les diagnostics d'échec, eux, sont bien horodatés (`results/.../mini_failures/seedNN__kind__stepN__UTC/`). Chemins tous construits avec `pathlib` depuis `REPO_ROOT`, sans séparateur ni chemin absolu propre à Windows. [vérifié] | **problème (B3)** | `run_mini.py:47-50,617-825` | `runs.make_run_dir` (refuse d'écraser, un dossier par run) ; le rapport markdown devient une sortie du run dans ce dossier. | M |
| 4.3 | **Logs** : uniquement `print` sur stdout (une ligne toutes les 500 pas) et un `logging.warning` dans `agent.py`. Pas de fichier de log, pas de TensorBoard ni W&B (absents de `pyproject.toml`). [vérifié] | à améliorer (souhaitable) | `run_mini.py:405-411` | Rediriger avec `2>&1 \| tee run.log` (aucun code) ; à terme un `metrics.csv` par run. | S / M |
| 4.4 | **Résultats bruts non persistés** : courbes de perte, courbe de séparation, `p_hat`, `N`, sondes ne vivent qu'en mémoire puis dans le texte du rapport. `write_report` n'est appelé qu'**après toutes les graines** (`run_mini.py:837-838`) : un crash à la graine 2 perd la graine 1. [vérifié] | **problème (B3)** | `run_mini.py:837-838` | Écrire un JSON/npz par graine à la fin de chaque graine, avant le rapport. | M |
| 4.5 | **Reprise après interruption : impossible.** Un seul `torch.save` (à l'échec, `run_mini.py:93`). `budget.checkpoints: [5000]` dans `mini.yaml:34` n'est lu par aucun code ; la grille de 16 checkpoints (décision de phase 0) n'est pas implémentée ; le pointeur `latest.json` de `runs.py:92-97` n'est pas utilisé par E3. Une reprise exacte demande aussi de sauver l'état des RNG numpy, l'optimiseur, la progression du switch (`SeparationSwitch`) et le replay (ou une recollecte déterministe). [vérifié] | problème (**B5**, pleine échelle ; souhaitable pour le mini) | `run_mini.py:93` ; `runs.py:92-97` | Checkpoints périodiques (modèle + Adam + états RNG + switch) dans `checkpoints/` avec `latest.json` ; option `--resume`. | L |
| 4.6 | **Scripts de lancement** : aucun `.ps1`/`.sh` ; les points d'entrée sont des scripts Python portables. Les seules traces Windows sont des chaînes d'usage dans des docstrings (`.venv\Scripts\python.exe`). Le `Makefile` (E0) est POSIX. [vérifié] | OK | docstrings de `run_mini.py`, `ae_isolation.py`, `compare_*.py` | Ajouter la variante bash dans les docstrings/README. | S |
| 4.7 | **Multiprocessing / DataLoader** : aucun (`num_workers`, `spawn`, `fork`, `Pool` absents du code E3). Les scripts ont `if __name__ == "__main__"`. Pas de piège `spawn`. [vérifié] | OK | — | — | — |

### Axe 5 — Dépendances

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 5.1 | `pyproject.toml` : `numpy>=1.24`, `matplotlib>=3.7`, `scipy>=1.10`, `pyyaml>=6.0` ; extra `e3 = ["torch>=2.2"]`. **Aucune version épinglée, pas de lockfile, pas de `requirements.txt`.** Ici : Python 3.14.0, torch 2.14.0+cpu ; le cluster peut donner une autre version de torch (réglages par défaut de TF32/cuDNN et numérique peuvent changer). Le commentaire de `pyproject.toml` promet « same version pin » mais aucun pin n'existe. [vérifié] | problème (souhaitable) | `pyproject.toml:29` | Après le smoke test, figer un `requirements-e3-lock.txt` (`pip freeze`) et épingler `torch==` ; l'enregistrer dans les métadonnées. | S |
| 5.2 | **Pas d'URL d'index CPU-only** dans `pyproject.toml` ni dans le `Makefile`. La commande `--index-url https://download.pytorch.org/whl/cpu` n'apparaît que dans `docs/e3/DECISIONS.md:104` (installation Windows). Sur Linux, `pip install -e ".[e3]"` résout la roue CUDA par défaut de PyPI. À confirmer avec l'advisor : version de CUDA/driver de l'A100 (cu118/cu121/cu124 selon le cas). [vérifié] | OK (à vérifier) | `pyproject.toml` | Noter dans le README la commande cluster exacte. | S |
| 5.3 | Aucune dépendance propre à Windows. `requires-python >= 3.10` : le code utilise `X \| None`, compatible. Python local 3.14 : un Python 3.10-3.12 côté cluster n'a pas été testé. `git` doit être dans le `PATH` (métadonnées). [vérifié / supposé] | à vérifier | — | Lancer `pytest tests/agents/worldmodel tests/measures tests/e3` sur l'environnement du cluster (CPU, quelques minutes). | S |

### Axe 6 — Taille des sorties

Aujourd'hui, un run **réussi** écrit moins de 1 Mo : le rapport (16 Ko) et les figures (80 Ko pour 2-3 PNG). Un **échec** écrit un checkpoint complet (modèle + Adam) : 6,7 Mo mesurés pour le modèle mini (576 211 paramètres × 12 o). Tout ce qui suit est donc **prospectif** (checkpoints, tables) : voir §3.

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 6.1 | `.gitignore` exclut `results/**/checkpoints/**` mais **pas** `results/.../mini_failures/**/checkpoint.pt` (6,7 Mo, actuellement non suivi, sera commité par un `git add`). | à corriger (souhaitable) | `.gitignore` | Ajouter `results/**/mini_failures/**/*.pt`. | S |
| 6.2 | Les tables de comptage ne sont pas écrites sur disque : `N` (144×5×144 int64) et `p_hat` (float64) pèsent chacune 829 Ko par graine si on les persiste. [calculé] | info | — | Les inclure dans le résultat brut par graine (4.4). | S |

### Axe 7 — Goulots probables (identification seulement)

| # | Constat | Statut | Localisation | Correctif proposé | Taille |
|---|---|---|---|---|---|
| 7.1 | **Boucle Python séquentielle sur T avec encodeur et décodeur appelés pas à pas** : `observation_step` encode `x_t` à chaque pas (`rssm.py:345`) et `unroll_observation` décode à chaque pas (`loop.py:298-305`). Ni l'un ni l'autre ne dépend de l'état récurrent (l'encodeur), ou seulement de `(h,z)` déjà calculés (le décodeur) : on pourrait encoder les B·T images d'un coup avant la boucle et décoder après. Sur GPU avec un petit modèle, on sera limité par le lancement des noyaux, pas par le calcul. [lu ; l'effet est supposé] | souhaitable | `rssm.py:345` ; `loop.py:298-305` | Encoder `(B·T)` en une passe, décoder `(B·T)` après la boucle (mêmes mathématiques ; l'ordre des réductions change donc pas bit-exact avec les runs CPU passés). | M |
| 7.2 | **Synchronisations dans la boucle chaude** : 8 `float(...)` par pas d'entraînement (7 pertes + `grad_norm`, `loop.py:393`), `.cpu().numpy()` de `post_probs` tous les 50 pas (`:396`), sondes toutes les 250 pas. Chaque `float()` force une synchronisation GPU. [vérifié] | souhaitable | `loop.py:393,396` | Renvoyer des tenseurs et ne synchroniser qu'aux pas de journalisation. | S |
| 7.3 | **Côté hôte** (mesuré sur CPU, un thread) : collecte 0,75 ms/pas dont **0,47 ms pour `render` seul** (double boucle de 144 cellules) : 10⁶ pas ≈ **12,5 min par graine**, soit ≈ 4,2 h pour 20 graines en séquentiel. `replay.sample` : 9,3 ms pour un batch 16×64 (boucle Python sur les fenêtres) ; `to_chw_float` : 24,7 ms. Par pas d'entraînement, l'hôte pèse donc ≈ 34 ms avant tout calcul GPU. [vérifié] | à traiter avec 2.4 | `run_mini.py:210-231` ; `replay.py:167` ; `loop.py:51-54` | Table pré-rendue (2.4). | M |
| 7.4 | **Plusieurs graines en parallèle** : possible par processus séparés (aucun état global partagé ; pas de dimension « graine » dans le modèle). Ce qui l'empêche aujourd'hui : (a) les chemins de sortie fixes qui s'écrasent (4.2) ; (b) 15,6 Go de RAM par processus (2.1) ; (c) `num_threads: 6` par processus (surcharge CPU) ; (d) une seule graine par processus devrait passer par `--seeds N`. [vérifié] | lié à B3/B6 | — | Après B3 et B6 : un processus par graine, `runtime.num_threads` adapté au nombre de cœurs. | (inclus) |
| 7.5 | Sondes coûteuses côté CPU (numpy float64) : `canonical_kernel_probe` fait 144×5 = 720 itérations avec transfert GPU → CPU à chaque itération (`probe.py:120-127`). Négligeable devant l'entraînement pour le mini, à regrouper si on le lance à chaque checkpoint. [lu] | souhaitable | `probe.py:105-130` | Regrouper les décodages par état. | S |

---

## 3. Estimations mémoire et disque

Hypothèses : 10⁶ pas + 256 de prefill ; observation 72×72×3 ; buffer actuel = 15 575 o/pas (mesuré).

### Replay (RAM hôte)

| | 1 graine | 4 graines | 20 graines |
|---|---|---|---|
| **Actuel** (pixels) | 15,6 Go | 62,3 Go | 311,6 Go |
| Table pré-rendue, dtypes actuels (a int64, s int64, r f32, 3 bool = 23 o/pas) | 23 Mo | 92 Mo | 460 Mo |
| Table pré-rendue, compact (s int16, a int8, r f32, 3 bool = 10 o/pas) | 10 Mo | 40 Mo | 200 Mo |
| + table de 144 images (2,24 Mo par graine, sur le device) | 2,2 Mo | 9 Mo | 45 Mo |

Si le buffer actuel était déplacé en VRAM : 15,6 Go par graine, soit 2 graines sur un A100 40 Go et 5 sur 80 Go. Ce n'est pas la conception actuelle (RAM hôte).

### Sondes sur tout le buffer (`z_only_probe`, état actuel)

| Nombre de pas | `x_seq` float32 (device) | + `decoded` (device) | + copie float32 côté hôte | + copie uint8 de `chronological_windows` |
|---|---|---|---|---|
| 5 256 (mini) | 0,3 Go | 0,3 Go | 0,3 Go | 0,1 Go |
| 100 000 | 6,2 Go | 6,2 Go | 6,2 Go | 1,6 Go |
| 1 000 256 | 62,2 Go | 62,2 Go | 62,2 Go | 15,6 Go |

### Modèle et checkpoints (mesurés par comptage de paramètres, float32)

| Configuration | Paramètres | Poids seuls | Poids + Adam | 16 checkpoints, poids seuls | 16 checkpoints, poids + Adam |
|---|---|---|---|---|---|
| mini (base 8, h/embed 64, 8×8) | 576 211 | 2,3 Mo | 6,9 Mo | 37 Mo | 111 Mo |
| défaut du spec (base 32, h/embed 256, 8×8) | 7 023 043 | 28,1 Mo | 84,3 Mo | 450 Mo | 1,35 Go |
| candidat 16×16 (base 32, h 256) | 9 161 539 | 36,6 Mo | 109,9 Mo | 586 Mo | 1,76 Go |
| candidat 32×32 (base 32, h 256) | 17 715 523 | 70,9 Mo | 212,6 Mo | 1,13 Go | 3,40 Go |

Les 16 checkpoints ne sont **pas encore écrits** par le code (4.5) ; ces chiffres sont ce que la grille de phase 0 produirait.

### Disque total par balayage (checkpoints uniquement ; métriques, figures et tables de comptage : < 5 Mo par graine)

| Configuration | 1 graine (poids seuls / avec Adam) | 4 graines | 20 graines |
|---|---|---|---|
| mini | 37 Mo / 111 Mo | 0,15 / 0,44 Go | 0,74 / 2,2 Go |
| défaut du spec | 0,45 / 1,35 Go | 1,8 / 5,4 Go | 9,0 / 27,0 Go |
| candidat 16×16 | 0,59 / 1,76 Go | 2,3 / 7,0 Go | 11,7 / 35,2 Go |
| candidat 32×32 | 1,13 / 3,40 Go | 4,5 / 13,6 Go | 22,7 / 68,0 Go |

### VRAM d'entraînement

Le modèle avec Adam pèse 84 Mo (défaut du spec). Les **activations** ne sont pas estimables sans GPU : à mesurer au smoke test avec `profile_throughput.py --device cuda` (batch 16×64, configuration `base_depth=32, h_dim=256`) et le snippet du §5, qui affichent `torch.cuda.max_memory_allocated()`.

---

## 4. Liste priorisée

### Bloquant pour un lancement GPU

| # | Point | Constat lié | Taille |
|---|---|---|---|
| **B1** | **Support du device dans `run_mini.py`** : champ `runtime.device` + `--device`, `.to(device)` du modèle (construit sur CPU avec la graine puis déplacé), de `background_t` et `templates_t`, contrat de device homogène des sondes (1.4), device affiché au démarrage. Sans cela, le run tourne sur CPU sans le dire. | 1.1, 1.4, 4.1 | S / M |
| **B2** | **Pousser E3 sur GitHub** : 26 chemins non suivis (le code du modèle, les mesures, les tests, `configs/e3/`, `experiments/E3_worldmodel_4zones/`) et 4 fichiers modifiés (`CLAUDE.md`, `pyproject.toml`, `docs/E2_online_4zones.md`, le README de `agents/worldmodel/`). Le tag `v0.1-offline` ne couvre pas E3. Donner à l'advisor un commit ou une branche précis, et compléter `.gitignore` (6.1). C'est ta décision : je n'ai rien commité. | État git | S |
| **B3** | **Sorties d'un run** : dossier par run via `runs.make_run_dir` (sans écrasement), config résolue et métadonnées d'environnement complètes (python, torch, CUDA, cuDNN, GPU, drapeau « sale »), résultats bruts écrits **à la fin de chaque graine**, rapport markdown écrit dans ce dossier et non dans `docs/`. Nécessaire pour tourner plusieurs graines en parallèle sans collision. | 3.5, 4.2, 4.4, 7.4 | M |

### Bloquant pour la pleine échelle (10⁶ pas), pas pour le mini

| # | Point | Constat lié | Taille |
|---|---|---|---|
| **B4** | `z_only_probe` par blocs : ne plus dérouler tout le buffer en un seul tenseur. | 2.2 | M |
| **B5** | Checkpoints périodiques (grille de 16) + reprise (`--resume`), avec états RNG, optimiseur et switch. | 4.5 | L |
| **B6** | Buffer à table pré-rendue (ou 16 Go de RAM par processus, 62 Go pour 4). | 2.1-2.4 | M |
| **B7** | Politique de déterminisme GPU décidée avant les runs cluster (4b) : `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, cuDNN, **TF32 désactivé explicitement**, enregistrés dans les métadonnées. | 3.2, 3.3 | S + M |

### Souhaitable

- Flux aléatoires séparés par composant, `torch.Generator` dédié à l'aléa latent (3.1) — M, déjà prévu en 4a.
- Encoder et décoder les B·T images en une passe (7.1) — M.
- Réduire les `float()` par pas (7.2) — S.
- Transfert en `uint8` puis conversion sur le device, à défaut de la table pré-rendue (1.8) — S.
- Figer `requirements-e3-lock.txt` et épingler `torch==` (5.1) — S.
- `--max-steps`, `--config`, `--out-dir` (4.1) ; script de balayage bash (4.1) — M.
- Log dans un fichier (`tee`) et `metrics.csv` (4.3) — S / M.
- Variante bash dans les docstrings, README d'E3 à jour (4.1, 4.6) — S.
- Lancer les tests CPU sur l'environnement du cluster (5.3) — S.
- Regrouper les décodages de `canonical_kernel_probe` (7.5) — S.

---

## 5. Protocole de smoke test sur Google Colab (GPU T4)

Objectif : vérifier, en quelques minutes, que le code tourne sur CUDA, que tout est sur `cuda`, et que la mémoire et les courbes sont plausibles. **Pas d'exigence d'égalité au bit près avec le CPU.**
Le protocole a deux étapes : l'étape 1 fonctionne **avec le code actuel** ; l'étape 2 suppose que **B1** est fait (les options citées n'existent pas encore).

### Étape 0 — Installation (Runtime → Type d'exécution → GPU T4)

```bash
!nvidia-smi
!python --version
!git clone https://github.com/nadben92/helplessness-worldmodels.git
%cd helplessness-worldmodels
!git checkout <BRANCHE_OU_COMMIT_CONTENANT_E3>      # cf. B2
!pip install -q -e ".[e3]"      # Colab a déjà un torch CUDA >= 2.2, pip le conserve
!python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.get_device_name(0))"
!pip freeze | grep -iE "^(torch|numpy|scipy|matplotlib|pyyaml)=="
```

Facultatif (quelques minutes, CPU) : `!python -m pytest tests/agents/worldmodel tests/measures tests/e3/test_e3_mini_report.py -q`.

### Étape 1 — Ce qui marche avec le code actuel

**1a. Pas d'entraînement et sonde sur CUDA (snippet testé sur CPU ici).** À coller dans une cellule Python :

```python
import time
import numpy as np
import torch
from helplessness.experiment_loader import load_experiment_module
from helplessness.agents.worldmodel.loop import WorldModelLossConfig, train_world_model_step
from helplessness.agents.worldmodel.replay import ModelBatch
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.get_device_name(0) if device.type == "cuda" else "-")
torch.manual_seed(0)
k, B, T = 6, 8, 16
side = 12 * k
rssm = RSSM(RSSMConfig(image_size=(side, side), base_depth=8, embed_dim=64, h_dim=64,
                       n_vars=8, n_classes=8, n_actions=5)).to(device)
assert all(p.device.type == device.type for p in rssm.parameters()), "parameters not on the target device"
opt = torch.optim.Adam(rssm.parameters(), lr=1e-3, eps=1e-8)
rng = np.random.default_rng(0)
mb = ModelBatch(
    x=rng.integers(0, 255, (B, T, side, side, 3), dtype=np.uint8), a=rng.integers(0, 5, (B, T)),
    r=np.zeros((B, T), np.float32), terminal=np.zeros((B, T), bool), truncated=np.zeros((B, T), bool),
    is_first=np.zeros((B, T), bool),
)
bg = torch.rand(3, side, side).to(device)
cfg = WorldModelLossConfig(k=k, w_patch=10.0)
train_world_model_step(rssm, opt, mb, bg, cfg, clip_norm=None)   # échauffement
if device.type == "cuda":
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for _ in range(20):
    out = train_world_model_step(rssm, opt, mb, bg, cfg, clip_norm=None)
if device.type == "cuda":
    torch.cuda.synchronize()
print(f"20 train steps: {20 / (time.perf_counter() - t0):.1f} steps/s | total {out['total']:.2f} | grad_norm {out['grad_norm']:.1f}")
assert all(np.isfinite(v) for v in out.values()), "non-finite loss"
if device.type == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "gpu_smoke_probe")
templates = torch.rand(144, 3, side, side).to(device)   # les sondes attendent des templates sur le device du modèle
print("canonical_separation:", {kk: round(v, 4) for kk, v in probe.canonical_separation(rssm, templates, n_eval_draws=1).items()})
print("OK: everything ran on", device)
```

Sortie de référence sur CPU : `20 train steps: 2.2 steps/s`, `canonical_separation` avec `acc_hard 0.0069`, dernière ligne `OK: everything ran on cpu`. Les données sont du bruit aléatoire : la perte est grande (≈ 2·10⁴), c'est normal ; le test vérifie les devices, pas l'apprentissage.

**1b. Configuration pleine taille du profil (batch 16×64, `base_depth=32, h_dim=256`) :**

```bash
!python experiments/E3_worldmodel_4zones/profile_throughput.py --device cuda --n-env-steps 500 --n-train-steps 20 --batch-size 16 --seq-len 64
!nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

Repère : la première ligne doit afficher `device: cuda`. Comparer le débit à `--device cpu`. C'est aussi la mesure de la VRAM d'entraînement (§3).

**1c. Détecter le lancement silencieux sur CPU** (le risque de B1). Lancer `run_mini.py` en l'état, sur une graine, avec un délai :

```bash
!timeout 300 python experiments/E3_worldmodel_4zones/run_mini.py --seeds 0 2>&1 | tee run_mini_colab.log
```

et, dans une autre cellule pendant qu'il tourne : `!nvidia-smi`. Attendu **aujourd'hui** : le processus n'apparaît pas dans `nvidia-smi` (0 Mo utilisés) — c'est la confirmation de B1. `timeout` tue le run sans écrire de rapport, mais les lignes de console (tous les 500 pas) restent dans le log.

### Étape 2 — Après B1

```bash
!timeout 480 python experiments/E3_worldmodel_4zones/run_mini.py --seeds 0 --device cuda 2>&1 | tee run_mini_colab.log
```

(ou `--max-steps 600` si cette option est ajoutée). Pendant ce temps : `!nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 5`.

### Ce qu'il faut vérifier

| # | Point | Attendu |
|---|---|---|
| 1 | Pas de crash | Aucune erreur de type `Expected all tensors to be on the same device`, aucune exception CUDA. |
| 2 | Tout sur `cuda` | 1a : les assertions passent et la dernière ligne dit `cuda`. Étape 2 : `nvidia-smi` montre le processus Python avec de la mémoire utilisée ; le log affiche le device résolu. |
| 3 | Mémoire | 1a (modèle mini) : quelques centaines de Mo au plus. 1b : noter le pic ; il doit tenir dans les 15 Go de la T4 (et donc largement dans un A100). |
| 4 | Valeur initiale | Au pas 0, `recon_patch` du log CPU pour la graine 0 était **74,539** (indépendant du lr). Le modèle étant construit sur CPU avec la même graine puis déplacé, le pas 0 sur GPU doit donner la même valeur à 10⁻³ près en relatif (TF32 mis à part) ; un grand écart indique un problème d'initialisation ou de données. |
| 5 | Allure des courbes | Comparer avec le log du run CPU de `run_mini.py` (à lr 1e-3) : `recon_patch` doit chuter d'un ordre de grandeur dans les 500 premiers pas ; la JS par paires reste ≈ 0 tant que la phase « probabilités » n'a pas séparé. Pas d'exigence sur les valeurs exactes, et la bascule (attendue vers le pas 1700-3100 d'après le test d'isolement) peut tomber hors d'une exécution de 8 minutes. |
| 6 | Débit | Le CPU du mini fait ≈ 2,4 pas/s. Noter le débit GPU affiché par 1a/1b ; s'il est de même ordre que le CPU, c'est le signe du goulot « lancement des noyaux » (7.1) plutôt qu'un problème de device. |
| 7 | Déterminisme (facultatif) | Lancer 1a deux fois : les deux sorties doivent être proches (pas bit-exactes tant que la politique 4b n'est pas décidée). |

Points de vigilance Colab : la RAM (≈ 12 Go) suffit pour le mini (replay de 82 Mo) mais **pas** pour un buffer de 10⁶ pas (15,6 Go) ; Python de Colab (3.10-3.12) diffère de celui de développement (3.14) ; la session expire, donc pas de run long sans B3/B5.
