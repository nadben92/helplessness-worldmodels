# MANIFEST v3

Script: `slides/figures_v3/make_v3.py` (reads `results/` read-only, nothing recomputed). Style as v2: default matplotlib, no title/legend/annotation; the only reference line is E7_bars' grey dashed chance line (2/5) on the directed-share panel of figure 1. Exposed `#E8412C`, reference `#1F6FEB`, oracle `#1DB954`, anti-oracle `#FF9F1C`, constrained fix `#6718AB`, free (latent) fix `#B98CE8` (E7_bars colour), induction `#FF2E88`. The v2 figures are untouched (they live on branch `slides`).

## Critic metric (used in both figures)
**C1** (D7, main z̄ = `draw`; report § 20, same definition in § 19/21): share of well-localised zone steps where argmax_a V(h'_a, z̄) is a directed action (`dr.argmax_directed`, `analyse_e7_d7.c_measures`). Chance ≈ 0.4. It is what the critic reads from h'_a at a common z̄; it is not a probe of b.
Per-agent values: natural exposed/reference = `slides/out_v3/data/e7h_c1_natural.csv` (copy of v2's `data_v2/`, extracted from D7's natural episodes by `_extract_e7h_c1.py`; training image). Constrained fix = `d7_correction.csv` `C1_corr` (zbar=draw) at each agent's recorded setting (`d7_identity.csv`). Induction = `slides/out_v3/data/e7d_induction_runs.csv` (copy of v2's, per-run mean C1, mode `reverse`).

## Figure 1 — `E7_bars_critic.png`
Panels: b (bits, step 1) · critic controllability (C1) · localisation (zone) · directed share (episodes, k = 1). Bar = mean of the 4 E4 agents, black dots = agents. Colours: exposed, latent fix (free, α=1), constrained fix (recorded D7 setting), oracle, reference. Values for b, loc, dir are the same as E7_bars (E6 `points.csv` training image; `dev_e4_cpu/analysis/checks.csv` for latent fix; D7 `runs/*corrected*/summary.json` for constrained; `d4_episodes.csv` k=1 for oracle dir; oracle b = 0 by construction, oracle loc = exposed's).
Points per condition in the C1 panel: exposed 4, constrained 4, reference 4, **latent fix 0 (C1 never computed → empty bar), oracle 0 (the oracle plans with true transitions, no critic reading → empty bar)**.
C1 means: exposed 0.40, constrained 0.49, reference 0.99.

## Figure 2 — `E7_curve_b_vs_critic.png`
y = directed share (k = 1, ε-greedy), shared. Left x = b (bits, step 1; anti-oracle: `b_eff`). Right x = C1. One point per (agent, condition/level), no mean. Left / right counts:

| Colour → condition | Left | Right | Sources |
|---|---|---|---|
| red: exposed natural | 4 | 4 | E6 `dev_e4_cpu_v2/analysis/points.csv`, C1 csv above |
| blue: reference natural | 4 | 4 | same (b = 0 by construction; overdrawn by green/pink at ≈(0, 0.94)) |
| green: oracle | 4 | 0 | `d4_episodes.csv` (exposed, k=1); b = 0 as in E7_bars |
| orange: anti-oracle | 44 | 0 | `E7_dissociation/dev_e4_cpu/analysis/t1_runs.csv`, reference, h_mode=mean, k=1, p=0..1 (11 levels × 4 agents) |
| light violet: free correction | 4 | 0 | `E7_latent_manipulation/dev_e4_cpu/analysis/checks.csv` (stepwise, α=1, training) |
| dark violet: constrained correction | 4 | 4 | D7 `runs/*corrected*/summary.json` (recorded setting), `d7_correction.csv` |
| pink: induction (B) | 8 | 8 | b and directed share: D7 `runs/*natural*/summary.json` (mode `reverse`, k=1, b_step1); C1: `e7d_induction_runs.csv` |

**Missing C1 (right panel)**: oracle, anti-oracle (no critic reading), free correction (never computed). The right panel therefore has 20 points, the left 72.
**Excluded**: anti-oracle at k=5 (44 points) and oracle at k=5, 10 (8) (`t1_runs.csv`, `d4_episodes.csv`); anti-oracle `h_mode=action` at k=1 (4, variant not the main one); anti-oracle exposed rows (8); induction `reverse_random` control runs; the 24 `reverse` induction runs absent from `e7d_induction_runs.csv` (too few well-localised steps, no C1) — dropped from both panels.

**Verdict line (neutral)** — Spearman ρ, all points: left (b) = 0.16 (n=72); right (C1) = 0.90 (n=20). Same 20 points on the left: ρ(b) = 0.71. So on identical points the alignment is higher on the right (0.90 vs 0.71); the all-points left value is dragged down by the anti-oracle (b from −2.2 to 0 at constant ≈0.94 directed share) and by the induction/correction points off the trend. The right panel does not contain those anti-oracle/oracle/free-correction points.
