# E4 — rapport final (2026-10-21)

E4 est terminée : 4 configurations (paire de couleurs × orientation, diagonale 1), chacune avec un modèle exposé et sa
référence, 300 000 pas, une graine par configuration. Ce rapport résume les analyses. Les valeurs sont rapportées
telles quelles ; l'interprétation reste à faire. Les tableaux complets sont dans `tables/`, les figures dans `figures/`.

| Configuration | Paire | T_inc | T_ctrl | Graine | Vague |
|---|---|---|---|---|---|
| p1-a-d1 | 1 | (227, 17, 15) | (22, 0, 244) | 1 | 1 |
| p1-b-d1 | 1 | (22, 0, 244) | (227, 17, 15) | 2 | 1 |
| p2-a-d1 | 2 | (17, 29, 14) | (208, 0, 252) | 5 | 2 |
| p2-b-d1 | 2 | (208, 0, 252) | (17, 29, 14) | 6 | 2 |

Mesure principale : b au **pas 1** dans la zone d'intérêt (croyance à la première rencontre, sur les 16 états de
bordure). Les pas suivants mesurent la récupération en cours d'épisode : les environnements de test sont entièrement
contrôlables, donc dès le pas 2 le modèle a observé une transition où son action a fonctionné.

## 1. Entraînement

- **Vague 1** (p1-a, p1-b) : 4 runs, 300 000 pas, diagnostics à chaque checkpoint. B = 300 000. Le critère de
  convergence recalculé avec b au pas 1 et la masse des quatre quadrants donne 175 000, à titre informatif.
- **Vague 2** (p2-a, p2-b) : 4 runs. Ils ont été figés par une panne de `/scratch`, puis repris en un seul pod
  (`--phase resume` sans `--resume`) : à 170 000 pas pour trois runs, 160 000 pour p2-b exposé. La reprise est continue :
  - pour p2-b exposé, les lignes 161 000 et 162 000 écrites avant la panne et après la reprise sont identiques ;
  - aucun pas n'est sauté ni compté deux fois ;
  - le code d'entraînement n'a pas changé entre le lancement et la reprise.
- **Santé au checkpoint 300 000** : succès à 1,00 pour les 8 runs ; masse sur le vrai successeur ≥ 0,985 dans chaque
  quadrant.
- **Contrôle du protocole** : b au pas 1 sur l'image d'entraînement concorde à 0,02 bit près avec le b de fin
  d'entraînement d'E3, dans toutes les zones.
- **Oubli de l'état initial** : l'écart de I_s dû à h0 passe sous 0,05 en 2 à 4 pas, avant n_pre = 32.
- **Masse du couloir** : elle est sous-estimée dans le diagnostic d'entraînement, parce que la moyenne inclut l'état but,
  qui est terminal. Elle est exclue du critère de convergence.

## 2. Sondes : cohérence et validité

**Cohérence.** Les sondes extrêmes du cluster (GPU) et les mêmes sondes mesurées en local sur CPU concordent :
|z| ≤ 1,45 sur 24 comparaisons (`tables/coherence_cluster_vs_local.csv`).

**Critère de validité.** Une sonde est valide si la référence prédit bien les transitions au pas 1 :
- masse sur le vrai successeur ≥ 0,90 ;
- |I_ref − log2 5| ≤ 0,15.

La masse et les fantômes au pas 1 ont été remesurés (`final_analysis.py validity`) ; I_ref vient des pas du cluster.

| Configuration | Sondes invalides (sur 100) |
|---|---|
| p1-a | 39 |
| p1-b | 52 |
| p2-a | 40 |
| p2-b | 72 |

| Axe / position | UD | CD | neutre |
|---|---|---|---|
| segment | 23/44 | 19/44 | 10/44 |
| arc centré en T_ctrl | 17/44 | 40/44 | 8/44 |
| arc centré en T_inc | 44/44 | 22/44 | 19/44 |
| position seule | 1/4 (p2-a) | – | – |

La validité dépend de la distance RVB entre la couleur sondée et la **couleur d'entraînement du quadrant sondé**
(`tables/validity_by_distance_to_training_color.csv`) :

| Distance | < 30 | 30–90 | 90–150 | 150–200 | 200–250 | ≥ 250 |
|---|---|---|---|---|---|---|
| Couples (image, zone) valides | 16/16 | 28/32 | 77/106 | 53/83 | 10/35 | 8/112 |

**Diagnostic de reconstruction** (`final_analysis.py diagnose`, `tables/diagnose_*.csv`). Dans toutes les configurations
et pour les deux modèles, le décodeur reconstruit le quadrant sondé dans sa **couleur d'entraînement**, jamais dans la
couleur de la sonde.

Pour p2-a, en thème seul et en position seule :
- la référence localise directement l'agent dans 0,84 et 0,82 des cas ;
- la règle du gabarit le plus proche, celle du noyau, échoue (0,00 et 0,02) ;
- la couleur d'entraînement du quadrant, (17, 29, 14), est à 86 en RVB de la couleur de l'agent, (100, 45, 0).

Dupliquer une couleur de quadrant casse la localisation de la référence dans p1-b, p2-b et en partie dans p2-a, mais
pas dans p1-a.

## 3. Gradients de b au pas 1 (sondes valides)

`tables/gradients_mean.csv` et `figures/gradients_by_axis_position.png`. k = 0 à 10 le long de l'axe ; entre parenthèses,
le nombre de configurations valides.

- **Segment, position UD** : de −2,31 (4) à −2,23 (3) pour k = 0 à 5, puis −2,13 (1) à k = 6. Au-delà, rien de valide.
- **Arc centré en T_ctrl, position UD** : −2,31 à −1,74 pour k = 0 à 5. La dispersion est forte : à k = 5, p2-b
  vaut −0,98 et p1-b −2,29.
- **Arc centré en T_inc, position UD** : aucune sonde valide.
- **Position CD** : ≈ 0 sur le segment pour k ≥ 3 ; −0,26 et −0,19 pour k = 1 et 2 (p1-a et p2-a).
- **Position neutre** : entre −0,35 et +0,02. Profil en U pour p1-a sur le segment (−0,29, puis ≈ 0, puis −0,32).
- **Position seule** (3 configurations valides) : −2,150 (p1-a), −2,296 (p1-b), −2,274 (p2-b).

**Thème seul :**

| Configuration | b au pas 1 | Valide |
|---|---|---|
| p1-a | −0,289 ± 0,015 | oui |
| p1-b | −0,021 ± 0,007 | oui |
| p2-a | −0,185 ± 0,022 | non |
| p2-b | −0,031 ± 0,007 | oui |

**Segment contre arc centré en T_ctrl**, à même distance de T_inc :
- en UD, différence ≤ 0,01 pour k ≤ 2 ; pour k = 3 à 5, la moyenne va de −0,33 à −0,45, et jusqu'à −1,29 pour p2-b ;
- en position neutre, différence entre −0,02 et +0,11.

## 4. Récupération et suffisance

**Récupération.** En UD, b passe d'environ −2,0 / −2,2 au pas 1 à environ −1,6 / −1,9 en plateau (pas 6 à 10) ;
l'écart va de +0,35 à +0,42 selon la configuration. En CD et en position neutre, l'écart est ≈ 0. Le thème seul de p1-a
reste à −0,28 / −0,31 sur les 25 pas.

**Erreur-type de b au pas 1** (192 sondes valides) :

| Marches | 25 | 50 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| Médiane | 0,025 | 0,019 | 0,012 | 0,010 | 0,008 | 0,006 |
| Maximum | 0,126 | 0,075 | 0,048 | 0,035 | 0,031 | 0,026 |

## 5. Reproduire

Il faut les deux archives de la campagne de sondes (`export_batch.py create`, vagues 1 et 2), extraites chacune dans son
propre dossier, puis `export_batch.py verify <dossier>`. Depuis la racine du dépôt :

```bash
A=experiments/E4_generalization_gradient
for c in p1-a-d1 p1-b-d1; do python $A/final_analysis.py validity --export <vague1> --configuration $c --n-walks 300 \
    --models reference experimental --out <work>/validity; done          # idem p2-* avec <vague2>
for c in p1-a-d1 p1-b-d1; do python $A/final_analysis.py diagnose --export <vague1> --configuration $c \
    --images 0 1 2 3 15 30 31 92 --out <work>/diagnose; done             # idem p2-*
python $A/final_analysis.py report --export <vague1> <vague2> --validity <work>/validity/*.csv \
    [--local-probes <dossiers de sondes locales>] --out <work>/report
```

Les valeurs de b et la validité sont déterministes. Les erreurs-types par bootstrap dépendent de l'ordre des tirages,
donc de la présence de `--local-probes`.
