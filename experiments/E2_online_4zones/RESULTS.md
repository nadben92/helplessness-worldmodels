# E2 — Résultats du sweep

Produit par `analysis.py`, relançable depuis les artefacts du sweep
(`results/E2_online_4zones_sweep/`), sans jamais relancer l'entraînement.
Faits uniquement — aucune interprétation scientifique.

## Paramètres figés

| paramètre | valeur |
|---|---|
| Départs | uniformes sur les 143 états non terminaux (état but exclu) |
| H (troncature) | 36 pas |
| ε | 0,1 (référence), 1,0 (pont E1) |
| α retenu comme référence | 1,0 (balayé sur {0,5 ; 1 ; 5} à ε=0,1) |
| Budget de référence | 10⁶ pas par graine |
| Checkpoints | 16, log-espacés, 1000 à 10⁶ |
| Graines | 20 par condition, carré latin 4 rotations × 5 graines |
| γ | 0,9 |
| Conditions | α∈{0,5 ; 1 ; 5} à ε=0,1 (online + apparié) ; α=1, ε=1 (online + apparié, pont E1) |

## Phase A — statuts

Voir `artifacts/phase_a_status.json` pour le détail complet. Résumé :

| vérification | statut |
|---|---|
| b̂ = 0 exact, couloir + contrôlable-déterministe, online et apparié | observé |
| T̂ online = T̂ apparié, tapis roulant + contrôlable-déterministe (via proxy L1) | observé |
| Δ ≥ 0 partout, après clamp | observé |
| variance non nulle de b̂ entre graines, zones bruitées | observé |
| N_start ≤ N, tout couple, tout checkpoint | observé |
| ε=1 : aucune différence appariée online-apparié (b̂, L1, Δ) | observé |
| tests du jumeau A-D, échantillon (2 graines × 4 conditions, checkpoint final) | observé (8/8) |

Contre-balancement : dispersion inter-rotation ≤ dispersion inter-graine dans toutes les zones à composante stochastique (voir `artifacts/phase_a_status.json`).

**Tout passe : Phase B produite sans interruption.**

## Figures et tableaux

Tous les PNG/PDF/CSV sont sous `artifacts/figures/`, répartis dans les sous-dossiers `png/`, `pdf/`, `csv/` (même nom de fichier dans chacun).

- **F1** (`F1_b_vs_budget_alpha{0.5,1.0,5.0}`) — b̂ par type de zone vs budget, online/apparié, une figure par α.
- **F2** (`F2_null_model_paired_diff`) — écart apparié online−apparié sur b̂/Δ/L1 par zone, α=1 ε=0,1. Figure centrale.
- **F3** (`F3a_l1_diff_by_zone_and_optimal_status`, `F3b_l1_diff_by_count_bin`) — localisation de l'écart L1 par statut d'action et par tranche de comptage.
- **F4** (`F4_N_total`, `F4_N_start_share`, `F4_N_min_per_state`, `F4_b_online`, `F4_delta_online`, `F4_b_diff`, `F4_delta_diff`, `F4_delta_diff_exterior`) — cartes spatiales, checkpoint final, moyenne sur graines, grille canonique (agrégation par type de zone, jamais par quadrant).
- **F5** (`F5a_alpha_contrast_vs_budget`, `F5b_alpha_contrast_final_checkpoint_table`, `F5c_b_vs_alpha_by_count_bin`) — effet du prior α.
- **F6** (`F6a_epsilon_bridge_paired_diff`, `F6b_visitation_comparison`, `F6c_e1_bridge_comparison`) — pont ε=1, comparaison à une courbe générative régénérée à α=1 (jamais aux artefacts α=0 d'E1).
- **F7** (`F7a_counts_profile`, `F7b_departure_share`, `F7c_episode_profile`) — profil de visite de référence (le dernier réutilise `episode_analysis.py` sans recalcul).
- **Tableau de synthèse** (`synthesis_table_b.csv`, `synthesis_table_delta_exterior.csv`) — checkpoint final, par zone × α : b̂ online, b̂ apparié, écart apparié + IC 95% + taille d'effet + p Wilcoxon ; Δ moyen extérieur (couloir) par α et condition.

## Formes prédites — observé / non observé

| prédiction | statut |
|---|---|
| b̂ = 0 exact, couloir et contrôlable-déterministe | observé |
| online et apparié identiques en distribution, deux zones déterministes | observé |
| écart adaptatif concentré sur l'action optimale sous P, zone contrôlable-bruitée | voir F3a |
| Δ extérieur ≈ 0, pas de différence online/apparié | voir tableau de synthèse (delta_exterior) — non nul à α=5 |
| aucun écart online−apparié à ε=1 | observé |
