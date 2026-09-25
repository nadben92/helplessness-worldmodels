# prompt_e3_worldmodel.md — Implémentation d'E3 (world model RSSM sur l'environnement à quatre zones)

## 0. Contexte, rôle, règles de travail

Tu implémentes **E3** du projet `helplessness-worldmodels` : le passage du tabulaire (E1, E2) à un world model RSSM de la lignée DreamerV3, sur la version pixelisée de l'environnement à quatre zones. **Ce code sera réutilisé tel quel en E4, E5 et E6.** La modularité, la configuration et les points d'accroche comptent autant que le résultat d'E3.

**Question scientifique.** Quand un agent apprend qu'une partie de son monde échappe à son contrôle, que devient cette croyance dans les parties où le contrôle existe ?

La contrôlabilité perçue est mesurée par `I(a; s'|s) = H(s'|s) − H(s'|s,a)`, sous une référence uniforme sur les actions. On l'exprime relativement à un environnement de référence apparié : `b = Î − Î_ref`.

E3 doit produire trois choses :
- un world model en boucle fermée dont on lit la croyance de transition ;
- le jumeau tabulaire correspondant ;
- les diagnostics qui disent si E4/E5 sont exécutables.

**Règles de travail (non négociables) :**

1. **Travail par phases** (section 11). À la fin de chaque phase :
   - tu t'arrêtes ;
   - tu lances les tests ;
   - tu écris un résumé court (ce qui est fait, ce qui dévie de la spec, ce qui reste ambigu) ;
   - tu attends mon feu vert.

   Ne jamais enchaîner la phase 4 vers la 5, ni la 5 vers la 6, sans validation explicite.
2. **Le code E1/E2 est gelé.** On l'importe, on ne le modifie jamais. Si un adaptateur est nécessaire, il vit dans le module E3 et tu me le signales.
3. **Nouveau module** `e3_worldmodel/`, nommé de façon cohérente avec la structure actuelle du repo (à vérifier en phase 0).
4. **Environnement** : Windows, PowerShell, Python dans `.venv`, PyTorch avec CUDA. Toutes les commandes que tu me donnes sont en PowerShell.
5. **Reproductibilité** : chaque run sauvegarde sa config complète, le hash git et ses graines à côté de ses résultats.
6. **Ambiguïtés** : en cas de doute sur la spec, tu demandes, tu n'improvises pas. Toute déviation doit être signalée explicitement.
7. **Tests de forme** : les tests de forme (section 9) sont décidés d'avance. Ils portent sur des formes (signes, monotonies, bornes, invariances), pas sur des valeurs plausibles. Ils sont implémentés comme tests automatisés ou comme sections de rapport.
8. **Frontière agent / instrument** : l'étiquette d'état `s` n'entre **jamais** dans la boucle de l'agent ni dans la représentation. Elle ne sert qu'aux mesures et aux logs.

---

## 1. Phase 0 — lecture du repo (aucun code)

Lis `CLAUDE.md` et le code d'E1/E2, puis écris `e3_worldmodel/PHASE0_REPORT.md`. Le rapport doit répondre précisément aux questions suivantes, en citant fichiers et fonctions :

1. **Structure** du repo et point d'entrée d'E2.
2. **Environnement tore 12×12** :
   - coordonnées des quatre zones 5×5, du couloir en croix et du but ;
   - ordre des actions, action `stay` ;
   - noyaux des quatre types de zones.
3. **Directions dans les dynamiques.** La dérive du tapis roulant (`away_from_goal`) et la direction fixe de l'incontrôlable-bruitée sont-elles définies **relativement au but** ou en absolu ? Ce point conditionne la permutation (section 2.2).
4. **Environnement de référence apparié** : sa construction et le test d'appariement.
5. **Politique E2** :
   - ε-greedy et départage des égalités ;
   - burn-in ou préremplissage éventuel ;
   - départs, troncature à H = 36, terminal au but ;
   - récompense (50 au but, 0 ailleurs ? coût par pas ?).
6. **Grille des 11 checkpoints.**
7. **Mesures** :
   - calcul de `b̂` ;
   - `Î_ref` appariée aux comptages ;
   - `Δ` évalué sous `P` ;
   - prior Dirichlet α = 1 restreint aux 5 successeurs atteignables ;
   - agrégation par zone.
8. **Schéma de graines** d'E2 : flux indépendants ou partagés entre expérimental et contrôle.

Stop après le rapport.

---

## 2. Environnement pixelisé

### 2.1 Dynamiques

Réutiliser les noyaux d'E2 par import. Aucune réimplémentation.

### 2.2 Permutation type de zone → emplacement

- Les quatre emplacements de zone sont équivalents sous le groupe des réflexions du tore qui **fixent la case but**. Une permutation des types de zone doit donc être équivalente à une réflexion.
- Pour cela, toute direction qui apparaît dans les dynamiques (dérive du tapis, direction de l'incontrôlable-bruitée) doit être exprimée dans le **repère relatif au but**. Si E2 les définit en absolu, écris un adaptateur dans `e3_worldmodel/` et signale-le.
- **Affectation par graine** : un carré latin 4×4 répété 5 fois pour 20 graines. Chaque type occupe chaque emplacement exactement 5 fois. La table graine → permutation est sauvegardée.
- Test exact T2 (section 9).

### 2.3 Environnement de référence

Il partage avec l'environnement expérimental :
- le layout ;
- la permutation ;
- les thèmes ;
- le but.

Seuls les types incontrôlables sont remplacés par leur contrepartie contrôlable :
- tapis roulant → contrôlable-déterministe ;
- incontrôlable-bruitée → contrôlable-bruitée.

C'est la même construction qu'en E2. Test d'appariement : pour tout (s, a), `H(s'|s,a)` et la taille du support sont identiques entre les deux environnements ; seule I diffère.

### 2.4 Rendu allocentrique

Signature : `render(s, layout, themes) -> uint8[H, W, 3]`.

- **Grille** : 12 × 12, k pixels par case (config, défaut k = 6, soit 72 × 72).
- **Éléments constants** :
  - couloir gris neutre constant ;
  - but dans une couleur constante dédiée ;
  - lignes de grille optionnelles (config).
- **Zones** : chaque zone a une couleur de thème uniforme en E3. L'interface `Theme` doit être extensible (textures) pour E4/E5.
- **Agent** : disque central de diamètre ≈ k/2, dans une couleur **hors palette** :
  - distance CIELAB ≥ seuil (config) à tous les thèmes, au couloir et au but ;
  - le reste de la case garde son thème visible, puisque c'est le patch local qui porte l'apparence de la zone.
- **Injectivité** : test que les 144 rendus d'un run sont distincts deux à deux.
- Fournir aussi `render_background(layout, themes)`, le rendu sans agent. Il sert au masque de perte (section 3.2).
- Fournir `render_counterfactual(s, layout, themes, override)` pour les diagnostics de la section 8. Ces images ne servent **jamais** à l'entraînement.

### 2.5 Thèmes

- **Générateur** : `sample_themes(n_zones, rng, min_dist, metric="cielab")`, avec une contrainte de distance minimale deux à deux.
  - En E3, tous les thèmes sont distincts, à une distance ≥ `min_dist`.
  - Prévoir dès maintenant le paramètre `pair_distance`, qui impose la distance entre une paire donnée. Il n'est pas utilisé en E3 mais servira en E4/E5.
- **Fixité** : les thèmes sont **fixes pendant tout un run**. Jamais de retirage par épisode : cela entraînerait une invariance à l'apparence, qui neutraliserait le mécanisme d'E4/E5.
- **Tirage** : les thèmes sont tirés par graine (contre-balancement thème → zone entre graines). Ils sont identiques entre l'environnement et sa référence pour une même graine.

---

## 3. World model (RSSM, DreamerV3 simplifié)

### 3.1 Architecture

- **Encodeur** : CNN à stride 2, profondeur de base 32, LayerNorm sur les canaux, SiLU. Sortie : embedding `e_t`.
- **Latent catégoriel** :
  - taille `n_vars × n_classes` configurable ; balayage de calibration sur {8×8, 16×16, 32×32} ;
  - **unimix 1 %** sur le prior et le posterior.
- **État déterministe** : `h_t = GRU(h_{t−1}, [z_{t−1}, onehot(a_{t−1})])`, avec une projection MLP en entrée comme dans V3. Taille `h_dim` configurable (défaut 256).
- **Prior** : `p(z_t | h_t)`, un MLP qui sort des logits de forme (n_vars, n_classes).
- **Posterior** : `q(z_t | h_t, e_t)`, un MLP sur la concaténation.
- **Décodeur** : `p(x_t | h_t, z_t)`, CNN transposé qui sort l'image moyenne.
- **Straight-through** : `z = onehot(sample) + probs − sg(probs)`.
- **Début d'épisode** : à chaque `is_first`, h prend sa valeur initiale apprise (comme dans V3), et z_{−1} = 0, a_{−1} = 0.
- **Hook de perturbation latente** : `latent_hook(h, z, context) -> (h, z)`.
  - Il est appelé avant le prior, le critic, la tête but et le décodeur ; identité par défaut.
  - Non utilisé en E3, mais il **doit exister et être documenté** : la manipulation latente causale est le point central du papier.

### 3.2 Pertes

- **Reconstruction** : MSE pondérée par pixel, `w = 1 + (w_patch − 1)·M`.
  - M est la dilatation d'une case de l'ensemble des pixels où `x ≠ render_background`.
  - Le masque est calculé **depuis l'image seule**, sans aucune étiquette d'état.
  - `w_patch` est configurable (défaut 10) et se calibre en phase 4.
- **Termes de KL** :
  - `dyn = KL(sg(q) ‖ p)` et `rep = KL(q ‖ sg(p))`, avec β_dyn = 0,5 et β_rep = 0,1 ;
  - KL calculée exactement, en sommant sur les variables ;
  - **free bits = 0** par défaut. Le paramètre reste dans la config (`free_bits`, `free_bits_mode ∈ {per_step, batch_mean}`) pour pouvoir le réactiver.
- **Tête but** : voir section 4.1. Aucun gradient ne remonte vers la représentation.
- **Aucune autre tête** : ni récompense, ni continuation séparée.

### 3.3 Optimisation

- Adam, lr = 1e-4, eps = 1e-8, clip de norme 1000 (config).
- Batch : 16 séquences × 64 pas.

### 3.4 Replay

- **Capacité** ≥ nombre total de pas du run, pour qu'aucune donnée ne soit oubliée (le jumeau B, lui, compte tout).
- **Échantillonnage** : fenêtres de 64 pas tirées uniformément. Elles peuvent traverser des frontières d'épisode ; `is_first` réinitialise h, z et a.
- **Contenu stocké à chaque pas** :
  - `x` et `a` ;
  - `r` et `terminal` ;
  - `truncated` et `is_first` ;
  - l'étiquette vraie `s` (jamais vue par le modèle).
- **Flags** : la troncature à H = 36 donne `truncated = True` et `terminal = False`. Seul le but est terminal.

### 3.5 Ratio d'entraînement

`replay_ratio` (pas rejoués par pas d'environnement) est configurable et se calibre en phase 4.

---

## 4. Agent (sans actor)

### 4.1 Tête but

`g(h, z) = p_but`, un MLP sur `sg([h, z])`.
- **Cible** : `1[r_t > 0]`, dérivée du signal de récompense et pas de l'étiquette d'état.
- **Perte** : BCE, sur les états posterior du replay.
- **Récompense et continuation** : `r = 50·p_but` et `c = 1 − p_but`.
- Aucun readout d'état dans la boucle de l'agent.

### 4.2 Critic

V(h, z) est un MLP sur `sg([h, z])`.
- **Perte** : régression MSE sur les λ-returns.
- **Cible** : calculée avec un critic lent, mis à jour par EMA (τ = 0,02).
- **Optimiseur** : lr = 3e-5, clip de norme 100.
- Aucun gradient du critic ne remonte vers le world model.

### 4.3 Politique ε-greedy sur anticipation d'un pas

Q se définit ainsi :

```
h'_a = GRU(h, z, a),  z'_k ~ p(· | h'_a),  k = 1..K
Q(h, z, a) = (1/K) Σ_k [ 50·p_but(h'_a, z'_k) + γ·(1 − p_but(h'_a, z'_k))·V(h'_a, z'_k) ]
```

avec γ = 0,9.

- **Choix de l'action** : avec probabilité ε = 0,1, action uniforme. Sinon, argmax **avec départage aléatoire** parmi les actions telles que `Q ≥ max Q − tol` (`tol` configurable, défaut 1e-3).
- **Logs** : fraction de pas départagés.
- **Nombre de tirages** : `K_act` (défaut 16) pour agir, `K_imag` (défaut 4) en imagination.
- **Filtrage à chaque pas réel** :
  - `h_t = GRU(h_{t−1}, z_{t−1}, a_{t−1})` ;
  - `z_t` tiré du posterior ;
  - réinitialisation à chaque début d'épisode.

### 4.4 Entraînement du critic en imagination

- **Départs** : tous les états posterior du batch, en stop-gradient.
- **Horizon** : 15 pas, avec la même politique ε-greedy (`K_imag`).
- **Notations** : on note s_0 le départ, s_1..s_15 les états imaginés, `r_k = 50·p_but(s_k)` et `c_k = 1 − p_but(s_k)`.
- **Retours** :

```
R^λ_k = r_{k+1} + γ·c_{k+1}·[ (1 − λ)·V_slow(s_{k+1}) + λ·R^λ_{k+1} ],   R^λ_15 = V_slow(s_15)
```

avec λ = 0,95.
- **Pondération de la perte** : `w_0 = c_0` et `w_k = w_{k−1}·c_k`. Un départ terminal ne compte pas.

### 4.5 Préremplissage

- Reproduire exactement le traitement d'E2 (burn-in à ε = 1 ou non), tel qu'identifié en phase 0.
- Si un préremplissage supplémentaire est nécessaire pour le world model (`prefill_steps` à ε = 1 avant la première mise à jour), il est compté dans le budget et logué.
- Signaler tout écart de comparabilité avec E2.

### 4.6 Paramètres d'E2 figés

- départs uniformes sur les états non terminaux ;
- troncature à H = 36 ;
- ε = 0,1 constant ;
- budget de 10⁶ pas d'environnement ;
- checkpoints sur la grille d'E2.

---

## 5. Mesures

Les mesures vivent dans un module `measure/`, sous forme de fonctions pures qui prennent un modèle gelé.

### 5.1 Sonde canonique

Pour chacun des 144 états s, visités ou non :
1. `x = render(s)` ;
2. h = h initial ;
3. `z0 = mode de q(· | h0, enc(x))` (loguer aussi l'entropie du posterior) ;
4. pour chaque action a : `h1_a = GRU(h0, z0, a)` et `p_a = prior(h1_a)`.

### 5.2 I en espace z (mesure principale, en bits)

- `H(z'|s,a) = Σ_i H(p_{a,i})`, exact.
- `m = (1/5)·Σ_a p_a`, c'est la référence uniforme sur les actions.
- `H(z'|s)` se calcule par Monte-Carlo **avec densités exactes** :
  - tirer M échantillons (a uniforme, puis z ~ p_a) ;
  - `log m(z) = logsumexp_a log p_a(z) − log 5` ;
  - `H = −moyenne(log m(z))`.

  Rapporter l'erreur standard MC. M est configurable (défaut 4096).
- `Î_z(s) = H(z'|s) − moyenne_a H(z'|s,a)`. Assert : `0 ≤ Î_z ≤ log₂5`, à la tolérance MC près.
- **Interdit** : estimer une entropie en espace z par comptage d'échantillons.

### 5.3 Noyau en espace s (secondaire)

1. Pour chaque couple (s, a), faire `K_s` tirages `z' ~ p_a`.
2. Décoder chacun : `x̂ = dec(h1_a, z')`.
3. Apparier chaque image décodée aux 144 gabarits `render(s')` du run, par distance L2 en pixels.
4. On obtient `p̂(s'|s,a)` comme la fréquence des argmin.
5. **Fantômes** :
   - la marge est `d(second gabarit) − d(premier gabarit)` ;
   - si la marge est sous un seuil configurable, le tirage est classé « fantôme », dans une catégorie séparée.
6. **Rapporter** :
   - le taux de fantômes par zone ;
   - `Î_s` et `H_s(s'|s,a)` ;
   - `L1(p̂, P)` et `L1(p̂, T̂_B)`.

### 5.4 Sonde « z seul »

Un classifieur linéaire z → s :
- entraîné sur les posteriors du replay ;
- évalué sur un split tenu à part.

Elle vérifie que z porte l'état.

### 5.5 Dispersion de contexte

Sur un échantillon d'états (h, z) réels du replay :
- calculer Î_z ;
- comparer au résultat de la sonde canonique pour le même s ;
- rapporter l'écart par zone.

### 5.6 b

- `b(s) = Î_A(s) − Î_ref(s)`, pour le même s, entre le run A et le run apparié dans l'environnement de référence.
- Agrégation par zone comme en E2 (réutiliser la fonction d'E2).

### 5.7 Δ sous P

1. La politique gloutonne est `π(s) = argmax_a Q(s, a)`, lue sur la sonde canonique avec `K_act` tirages et départage aléatoire.
2. Elle est évaluée **exactement sous le vrai noyau P**, par évaluation de politique tabulaire.
3. On en déduit `Δ(s) = V*_P(s) − V^π_P(s)`.

Δ n'est jamais évalué sous T̂ ni sous le modèle. Réutiliser la définition d'E2.

---

## 6. Conditions, runs, graines

- **E3-0 (calibration, sans boucle)** :
  - politique ε = 1, sans critic ni tête but nécessaires ;
  - départs uniformes, troncature à H = 36 ;
  - lancé dans l'environnement **et** dans sa référence.

  Il sert à : fixer le plancher `b0` par type de zone, faire le test du noyau bruité (T4), et réaliser le balayage de capacité.
- **A** : boucle fermée (section 4) dans l'environnement expérimental.
- **Ref** : même pipeline dans l'environnement de référence.
- **B** : jumeau tabulaire sur les transitions réelles de A.
  - comptages `N[s,a,s']` à chaque checkpoint ;
  - pipeline E2 importé : prior Dirichlet α = 1 sur les 5 successeurs, VI, `b̂` avec `Î_ref` appariée aux comptages, Δ sous P.

  Faire de même avec **B_ref** sur le run Ref, pour quantifier l'écart de comptages entre A et Ref.
- **E2** : null model, à partir des résultats existants.
- **Graines** : 20. Les flux RNG sont indépendants, dérivés par `np.random.SeedSequence(seed).spawn`, et séparés pour :
  - la stochasticité de l'environnement ;
  - la politique (ε, départage) ;
  - l'initialisation des réseaux ;
  - l'échantillonnage du replay ;
  - les tirages latents.

  Entre l'environnement et sa référence, **seuls la permutation et les thèmes sont partagés**, puisque ce sont eux qui définissent l'appariement. Tous les autres flux sont indépendants : des graines corrélées entre expérimental et contrôle ont été un bug réel en E1. Signaler si E2 faisait autrement.

---

## 7. Phase 4 — calibration (critères fixés avant les runs)

- **Hyperparamètres calibrés** :
  - `n_vars × n_classes` et `h_dim` ;
  - `w_patch` et `replay_ratio` ;
  - `prefill_steps`.

  La calibration se fait **uniquement sur E3-0**.
- **Critères d'adéquation** (préenregistrés ; ni `b` ni `α_app` ne servent jamais à choisir) :
  - **C-a** : précision de l'appariement aux gabarits sur les reconstructions posterior ≥ 99 %, sur les états visités ;
  - **C-b** : `L1(p̂_s, P)` moyen sur les couples avec `N[s,a] ≥ 100` ≤ 0,1 en fin d'E3-0 ;
  - **C-c** : précision de la sonde z seul ≥ 99 % (pas d'effondrement).
- **Règle de sélection** : la plus petite capacité qui satisfait C-a à C-c.
- **Rapport descriptif** : `α_app` et D2 (section 8) sont rapportés sur tout le balayage, à titre descriptif seulement. Si `α_app ≈ 0` à la capacité retenue, **arrêter et me le remonter**. Le remède sera une révision du rendu décidée par moi, pas une réduction de capacité.
- **Coût** :
  - profiler le débit (pas d'environnement par seconde, pas de gradient par seconde) ;
  - estimer le coût total de (A + Ref) × 20 graines ;
  - proposer `replay_ratio` en conséquence.
- **Livrables** : `CALIBRATION_REPORT.md` et ses figures. STOP. Je gèle ensuite la config dans `configs/e3_frozen.yaml`.

---

## 8. Diagnostic position / apparence (exécutabilité d'E4/E5)

À l'intérieur d'un run, position et apparence sont parfaitement confondues. La permutation entre graines ne les sépare qu'au niveau de la population. Le diagnostic intra-modèle passe donc par des rendus contrefactuels, hors distribution et jamais vus à l'entraînement.

- **D1-local** : pour s dans la zone L, rendre l'image où la case de l'agent prend le thème d'une autre zone L'.
- **D1-global** : échanger les thèmes des zones L et L' sur toute la carte.
- **Indice α_app** : sur chaque image contrefactuelle, passer la sonde canonique, qui donne `Î_z^swap(s)`, puis calculer

  ```
  α_app(s) = (Î_z^swap(s) − Î_L) / (Î_L' − Î_L)
  ```

  où `Î_L` et `Î_L'` sont les moyennes par zone de la sonde canonique sur les images normales. L'indice n'est défini que pour les paires telles que `|Î_L' − Î_L|` > seuil, c'est-à-dire avec un contraste de contrôlabilité. α_app = 0 signifie un code piloté par la position ; α_app = 1, un code piloté par l'apparence.
- **Même lecture en espace s** : le noyau prédit suit-il la dynamique de L ou celle de L' ?
- **Validité** : pour les images contrefactuelles, rapporter l'erreur de reconstruction et l'entropie du posterior. Signaler tout écart massif hors distribution.
- **D2** : distances latentes (Jensen-Shannon entre les probabilités du posterior canonique) entre cases voisines de part et d'autre d'une frontière zone/couloir ou zone/zone, comparées aux distances entre voisines à l'intérieur d'une zone.
- **Forme attendue, fixée d'avance** :
  - α_app ∈ [0, 1] ;
  - critère d'exécutabilité d'E4/E5 : l'IC bootstrap de α_app (local) sur les graines exclut 0 à la capacité retenue.

---

## 9. Tests de forme (préenregistrés)

- **T0 (unitaires)** :
  - gradient straight-through non nul vers l'encodeur ;
  - KL exacte contre estimation Monte-Carlo ;
  - plancher de l'unimix ;
  - masque de perte calculé sans étiquette ;
  - réinitialisation par `is_first` dans le replay ;
  - `truncated` distinct de `terminal`.
- **T1** :
  - injectivité du rendu ;
  - appariement env/ref (`H(s'|s,a)` et taille du support identiques, I différente).
- **T2 (symétrie)** :
  - pour chaque permutation, la réflexion associée transporte P sur P exactement ;
  - en E3-0, b par type de zone ne dépend pas de l'emplacement (test d'interaction sur les graines).
- **T3 (tapis roulant)** : Î_z décroît vers 0 à mesure que les données augmentent ; en espace s, la masse sur le successeur de dérive tend vers 1.
- **T4 (zones bruitées, E3-0)** : `p̂_s(issue fréquente)` → 0,9, avec un IC sur les graines compatible, dans les deux zones bruitées. Sinon, biais du terme de KL : à rapporter.
- **T5 (contrôlable-déterministe)** : Î_z croît avec les données, en restant ≤ log₂5.
- **T6** : la dispersion de contexte est petite devant l'écart entre zones.
- **T7 (couloir)** : b est compatible avec le plancher d'E3-0.
- **T8 (signe)** : dans les zones incontrôlables, b ≤ plancher.
- **T9** : Δ ≥ 0 partout. Un Δ < 0 signe un bug d'évaluation.
- **T10** : taux de fantômes rapporté, avec alerte au-dessus d'un seuil.

---

## 10. Logs et sorties

Dossier par run : `runs/e3/{condition}/{seed}/`.

- **Métadonnées** : `config.yaml`, hash git, graines, permutation, thèmes.
- **Transitions** : `transitions.parquet`, avec les colonnes seed, condition, episode, t, s, a, s_next, r, terminal, truncated.
- **À chaque checkpoint de la grille E2** :
  - poids du modèle ;
  - `N[s,a,s']` (npz) ;
  - résultats de la sonde canonique : Î_z par état avec erreur MC, noyau en espace s par couple, taux de fantômes ;
  - dispersion de contexte ;
  - Δ ;
  - α_app et D2.
- **Courbes d'entraînement** :
  - pertes par terme ;
  - KL par type de zone ;
  - erreur de reconstruction sur le patch de l'agent ;
  - fraction de départages ;
  - taux de succès et longueur d'épisode ;
  - visitation par zone.
- **Agrégation** : des scripts produisent des tables zone × condition × checkpoint et des figures directement comparables à celles d'E2.

---

## 11. Ordre des phases

| Phase | Contenu | Fin |
|---|---|---|
| P0 | Lecture du repo, `PHASE0_REPORT.md` | STOP |
| P1 | Environnement, rendu, thèmes, permutation, référence ; T1, T2 (partie exacte) | STOP |
| P2 | World model entraîné sur des données fixes à ε = 1 ; T0 ; profilage du débit | STOP |
| P3 | Module de mesures et diagnostics (sections 5 et 8), testé sur le modèle de P2 | STOP |
| P4 | Calibration E3-0 et balayage de capacité, `CALIBRATION_REPORT.md` | STOP, gel de la config |
| P5 | Agent (tête but, critic, politique, imagination), jumeau B, pipeline Ref ; smoke test sur 2 graines à budget réduit | STOP |
| P6 | Runs complets A + Ref + B sur 20 graines, après feu vert explicite | Rapport |

En fin de P6, mets à jour `CLAUDE.md` avec la description du module E3 et de ses points d'extension pour E4 à E6 : `pair_distance`, `Theme` à textures, `latent_hook`.
