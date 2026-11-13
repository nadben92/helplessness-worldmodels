# helplessness

Learned helplessness in recurrent world models.

## Question

When an agent learns that part of its world is beyond its control, how does that belief generalise, and does it
drive behaviour? The project studies this in a DreamerV3-style recurrent world model (RSSM) trained on a 12x12
pixel gridworld (a torus) where some zones ignore the agent's actions. Perceived controllability is read directly
from the model's prior as the mutual information I(a; s' | s) between the action and the predicted next state, and
compared with a reference model trained on the same environment where every zone is controllable:
**b = I - I_ref**. b near 0 means "I am in control here"; b near -log2 5 = -2.32 bits means "my actions do nothing
here". The work runs from tabular agents (E0-E2) to the world model (E3) and on to where the belief lives (E4-E5),
whether it drives behaviour (E6-E7), and how it can be corrected (E8).

## Main results

The numbers below are taken from the reports linked in each paragraph. Analyses that the reports mark as
exploratory, preliminary or post hoc are marked the same way here. Several reports are in French: the
E6/E7 implementation report, the E4 final report and E2's `RESULTS.md`.

### I. Measuring the belief (E0-E3)

- **E0** (5x5 gridworld, tabular agent): the transfer phenotype is carried entirely by the transition belief, and
  editing the belief reproduces it. Perceived controllability predicts its severity, not its topography. As a rescue,
  amplifying the belief fails where structure injection succeeds. `reports/week1_offline_report.md`.
- **E1** (12x12 torus, four 5x5 zones: controllable or not, deterministic or noisy): with offline data, b recovers each
  zone's analytic controllability. The data budget recommended for later work is n = 100 visits per state-action
  pair. `docs/E1_offline_4zones.md`, `paper/reports/e1_report.tex`.
- **E2** (online tabular agent): learning online biases the estimate only slightly, compared with a control that
  has the same visit counts. The bias is confined to the controllable-noisy zone and vanishes at epsilon = 1. This
  gives E3 its null model. `experiments/E2_online_4zones/RESULTS.md`.
- **E3** (the world model on pixels, 4 seeds, 300,000 steps): b = -2.26 [-2.29, -2.23] in the uncontrollable
  deterministic zone and -1.78 [-1.95, -1.61] in the uncontrollable noisy zone. In the controllable zones b is
  compatible with 0, with bounds of 0.21-0.29 bit. The hybrid-code rule fixed before the campaign is not satisfied,
  so b is not interpretable in the controllable zones under that rule. With N = 4, every statement is descriptive.
  `docs/E3_worldmodel_4zones.md`, `paper/reports/e3_report.tex`.

### II. Where the belief lives (E4-E5)

- **E4** (two deterministic zones, one controllable (CD) and one uncontrollable (UD); 4 colour configurations;
  zero-data probes in fully controllable test worlds; b at the first step in the zone).
  **The belief is attached to position.** At the old uncontrollable position, b stays near -2.3 bits whatever its
  colour (position only: -2.15 to -2.30 in the 3 valid configurations; along the RGB axes, -2.31 to -1.74 where probes
  are valid). Elsewhere b is close to 0 for most colours; theme only gives -0.29 in one configuration and -0.02 to
  -0.03 in the two other valid ones (the fourth, -0.19, is invalid). Within an episode, b recovers by +0.35 to +0.42
  bit in the old zone, then plateaus. Probe validity falls with the RGB distance to the quadrant's training colour:
  colours serve as landmarks of place.
  `results/E4_generalization_gradient/final/REPORT.md`.
- **E5** (shuffled zone assignments, so position no longer predicts controllability; shuffle agents A, E, F at
  650,000 steps, short of E4's convergence criterion).
  **When position no longer predicts controllability, the belief moves largely onto the zone's appearance**: the colour
  of the zone together with the arrangement of the other quadrants, the cue most predictive of the dynamics in
  training. Colour and arrangement cannot be separated with these agents: they are perfectly confounded during
  training, and novel colours, which would separate them, make the probes invalid or out of distribution.
  - The belief follows the uncontrollable quadrant wherever it is: lowest of the four quadrants in 12/12 training
    assignments for every agent.
  - Position only: -0.46 / -0.48 / -0.89, against -2.3 for the baselines.
  - Theme only: no valid measurement at 650,000; -0.91 / -0.92 at 475,000 (preliminary).
  - Colour against arrangement is not readable under the rule written before the theme-swap probes.
  - With the arrangement fixed, b rises by 0.5-1.9 bits with the colour alone at the start of the gradient
    (exploratory).
  - The robustness diagnostic (why F stays localised on the swap probes and A, E do not) ends "not decided" under its
    pre-written rule.
  - `experiments/E5_belief_training/analysis_650k/REPORT.md`, `analysis_swap/REPORT.md`,
    `analysis_swap/THEME_EVIDENCE.md`, `analysis_robustness/REPORT.md`.

### III. From belief to behaviour (E6-E7)

- **E6** asks whether behaviour in a fully controllable test world (share of steps that reduce the distance to the
  goal, starting inside the old zone) is a function of b. 8 agents: E4's four pairs (one of which is E5's baseline A),
  E5's baseline B and shuffle A, E, F. **In natural agents the relation is a step, not a slope.** Helpless pairs
  (b about -2) move at chance (directed share about 0.4); all others behave almost like the reference, including the
  pairs whose b is intermediate (retained only in shuffle agents: directed share 0.75-0.93, controllability seen by
  the critic about 0.90). Intermediate values of b are read by the probe but not used by the agent. Few intermediate
  pairs survive the localisation filter (A 4, E 3, F 1; two E4 pairs with an intermediate b fail it), and the filter
  reads position through the decoder, so it can exclude pairs in which the agent acts from its true cell.
- **E7** tests the causal role of the belief, with frozen weights.
  - **Oracle.** Replacing the agent's predicted transitions with the true ones in its one-step planner raises
    directed behaviour in all 8 agents (0.32-0.49 to 0.65-0.93; reference 0.89-0.94): the transition belief causes a
    large part of the deficit (48-98 % of the gap).
  - **Belief versus cached values** (at one step of lookahead, the agent's own decision rule). Belief effect = oracle
    minus exposed; cached-value effect = reference minus oracle. In the baselines the belief accounts for 68-98 % of
    the gap. In shuffle A and E the cached-value effect is as large as the belief effect; F resembles the baselines.
    Planning 5 steps ahead with true transitions bypasses the cached values (0.92-0.95, reference level).
  - **Correcting the recurrent state h** (re-applied at every step). Behaviour recovers in part: in shuffle A and E the
    correction recovers the whole belief part and exceeds the one-step oracle (directed share 0.83 and 0.79 against
    0.65 and 0.71); F comes close to it (0.81 against 0.86); in the baselines it stays below it (0.42-0.69 against
    0.75-0.93). It acts through the predicted transitions, not through the perceived colour. The controllability
    seen by the critic rises by 0.04 at most in 6 of the 8 agents; it rises in baseline A (0.54 to 0.74) and shuffle A
    (0.46 to 0.77).
  - **No persistence.** A correction applied once is gone from the next step, on real steps and in imagination,
    although the perturbation is still present in h. The same holds for the natural colour direction of h; for the
    natural recovery direction the result is indeterminate. The belief is re-evaluated at every step; that h carries
    only the context, from which the weights recompute it, is not established (the condition written for it is not
    met).
  - **No global controllability axis.** The natural direction of h that raises controllability is specific to each
    quadrant and does not transfer between quadrants.
  - **Method.** b and decoder-read localisation can be moved by random or transferred directions without raising the
    planner or behaviour; manipulations are judged by the planner and by behaviour.
  - Pre-written forms (section 21): 1, 3 and 5 hold; 2 and 4 do not (4 fails only for E: cached-value effect 0.22
    against a threshold of 0.20). Several section 22 forms are partial or not held; the report gives each verdict.
- Report for both: `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`, sections 16-22.

### IV. Correcting the belief (E8)

- **Question.** Can the belief be corrected by re-training the agent in a world where its old zone is controllable?
  How fast do the belief (b) and behaviour (the directed share D) recover, does behaviour lag the belief where the
  cached values are heavy (E7: cached-value effect 0.24 in shuffle A, 0.22 in E), and does a correction made in some
  contexts carry over to the untreated ones (the relapse test)?
- **Design.** Each agent restarts from its E4/E5 weights with an empty replay and fresh optimisers, beside a
  reference re-trained the same way. Wave 1: baselines A and B, full exposure (the old zone nw controllable),
  150,000 steps. Wave 2: shuffle A, E and F with reference B, partial therapy on the 3 layouts where the
  uncontrollable theme T_inc is in nw, 250,000 steps; transfer is read on the 9 untreated layouts. Probes on the
  local CPU at 14 and 16 checkpoints. Every threshold and prediction was written before the data
  (`docs/e8/EXPECTED_FORMS.md`); the revisions are dated, and those made after wave 1 were made before any wave 2
  data.
- **P1 holds in all 5 agents.** b in the treated zone rises from -2.3 to -0.07 and -0.02 in the baselines (150,000
  steps) and from -2.1 to about 0 in shuffle A, E and F (250,000 steps), with no fall back during the therapy. Its
  half-recovery takes 2,000-3,500 steps in the baselines and 500-1,000 in the shuffle agents.
- **Behaviour recovers first and fast.** D reaches the reference within 2,000-3,500 steps in the baselines (about
  150-400 visits of nw) and within 500-2,000 steps in the shuffle agents (58-232 visits), although the shuffle agents
  avoid nw at first (6-12 % of steps against 19 % for the reference).
- **P2 fails: the cached value does not slow the correction.** Behaviour does not lag the critic's reading in
  shuffle A or E (lag 0 at the grid's resolution); their cached-value effect (0.28 and 0.22 at the start) is gone by
  2,000 steps, earlier than baseline B's.
- **P3 holds in F only.** Behaviour recovers in one interval of the grid while b moves over several in F; not in A,
  E or the baselines. The treated recovery fits within 3 intervals of the grid.
- **P4: transfer, slower, in all 3 shuffle agents** (transfer index on D 0.98). D on the untreated layouts reaches
  the reference after 3,500-10,000 steps in A and 40,000-100,000 in E and F, 2 to 200 times later than on the treated
  ones. It is read on the 6 untreated layouts where nw is neutral only. It cannot be attributed to the colour alone:
  ne, se and sw were also controllable during the therapy, with other colours, so the transfer may go through the
  colour (T_inc treated in nw) or through the position (a quadrant already seen controllable). Exploratory: the
  predictions improve with it (the mass on the true successor rises from 0.20-0.28 to 0.93-0.98 and never falls), a
  transfer rather than a forgetting; in A, behaviour moves ahead of the critic's reading and of the mass.
- **Limits.**
  - The reference partly forgets the layouts it is not trained on: on the 3 untreated layouts where nw carries the
    colour of the controllable theme, I_ref falls to 1.6 bits and its mass on the true successor to 0.03, and the
    exposed model does the same. Those layouts fail the probes' validity check and are out of P4; no untreated point
    is valid at 25,000 steps.
  - The health check of training fails for shuffle F (mass 0.857-0.899 against a threshold of 0.90 in one treated layout, 5,000-25,000 steps); it
    is reported and does not block F's reading.
  - The therapy starts from an empty replay. The tabular twin, which keeps the old experience, is still at b_tab
    -1.4 after 150,000 steps: the fast recovery is a property of this design.
  - Shuffle A is paired with reference B, while E6/E7 paired it with reference A (cached-value effect 0.279 with B,
    0.242 with A).
  - t½ and lags are checkpoints of a log-spaced grid; a lag of 0 means the same checkpoint.
- **Not done (future work).** Waves 3-5, with the therapy on ne, sw and se (configured in
  `experiments/E8_therapy_relapse/batches.py`, not launched), which would help separate colour from position; a
  therapy that keeps the old replay buffer (an analogue of rumination).
- `docs/E8_therapy_relapse.md`, `docs/e8/` (decisions, expected forms), report
  `experiments/E8_therapy_relapse/IMPLEMENTATION_REPORT.md` (in French), sections 11-13.

## Repository layout

```
src/helplessness/        the shared library (pip install -e .)
  envs/                  environments: the frozen 5x5 gridworld (legacy/), the zoned torus (torus_env.py), dynamics,
                         appearance (themes, colour axes), observation backends (index / pixel), zone layouts
  agents/                tabular and online agents; worldmodel/ (the RSSM, training loop, closed-loop agent)
  measures/              arrays in, arrays out; controllability.py holds the single implementation of I(a; s'|s)
  probing/               torch-side measurement of a frozen world model (probes, walks, episodes, correction of h)
  pipeline/              world-model run pipeline (training worlds, checkpoints, frozen agents of E6/E7)
  interventions/, planning/, evaluation/, streams/, twin/, cluster/ (GPU cluster tooling), config.py, runs.py
configs/                 YAML only: shared blocks (zones, themes, dynamics) and one directory per experiment (e3/ ... e8/)
experiments/<EN_name>/   one directory per experiment: runners, analyses, batch tables, reports, its own README.md
docs/                    EXPERIMENTS.md (the tracker), one file per experiment, decisions (e3/, e5/, e6/, e8/), archive/
results/                 light outputs only (tables, JSON, small figures); weights and raw arrays are never versioned
reports/                 frozen outputs of E0 and E3
paper/                   LaTeX paper and reports, figure and number scripts
slides/                  presentation figures: scripts (figures*/, style*/) and outputs (out*/), MANIFEST*.md
tests/                   mirrors src/, plus tests/e3 ... tests/e8 for each experiment's orchestration
```

E0-E3 are finished and frozen: their code, configs, seeds and defaults never change. Tags `e0-final` ... `e3-final`
and `pre-reorg` mark their final state, and `v0.1-offline` marks E0's report. The conventions and invariants
(single source of truth for I(a; s'|s), separate random streams, terminated vs truncated) are in `CLAUDE.md`.

## Installation and reproduction

Python 3.10 or later. Development runs on Windows and on CPU; training runs on Linux on a GPU cluster (Kubeflow).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Linux: .venv/bin/python
# torch, needed from E3 on (the "e3" extra). Locally, the CPU wheel:
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
# on a GPU machine, the CUDA wheel matching the driver, same version pin
pytest                                              # the full suite takes a little over an hour on a laptop CPU
```

The editable install points at one checkout's `src/`. When working from another git worktree, put that worktree's
`src` first on `PYTHONPATH`.

**What runs where.**

| Step | Where | Entry point |
|---|---|---|
| E0-E2 (tabular) | local CPU | `experiments/E0_offline_5x5/README.md`, `E1_offline_4zones/README.md`, `E2_online_4zones/README.md` |
| E3-E5 training and probes | GPU cluster | `experiments/E{3,4,5}_*/submit_kubeflow.py --batch <batch> --repo-dir <checkout> [--dry-run]`; procedure in each `RUN_GPU.md` |
| CPU smoke runs of training | local CPU | e.g. `python experiments/E4_generalization_gradient/run_e4.py --config configs/e4/e4_mini.yaml --configuration p1-a-d1 --condition experimental --device cpu` (E5: `run_e5.py --config configs/e5/e5_mini.yaml`, see its README) |
| E4, E5 analyses | local CPU, from the batch exports | `experiments/E4_generalization_gradient/final_analysis.py`, `experiments/E5_belief_training/analysis_*/` (each has its own README or usage line) |
| E6, E7 (frozen weights) | local CPU | `python experiments/E6_phenotypes_latent/chain_all_cpu.py --runs-root <root> --logs <dir> [--workers 12]`, then `analyse_all.py` |
| E8 therapy runs | GPU cluster | `experiments/E8_therapy_relapse/submit_kubeflow.py --batch <batch> --repo-dir <checkout> [--dry-run]` |
| E8 probes and analyses | local CPU, from the batch exports | `experiments/E8_therapy_relapse/chain_cpu.py`, then `analyze_e8.py` and `w2_supplements.py`; commands in the report, sections 11-13 |

**Weights are never in the repository.** Training writes them to the cluster's storage. A batch export
(`export_batch.py`) or a handoff records each file with its SHA-256, and the local analyses check them. For
E6/E7, `--runs-root <root>` must hold the runs in the cluster's layout, as listed in `configs/e6/agents_e5.yaml`:

```
<root>/helplessness-worldmodels-e4/results/E4_generalization_gradient/<run>/checkpoints/agent_env300000.pt
<root>/helplessness-worldmodels-e5/results/E5_belief_training/<run>/checkpoints/agent_env650000.pt
```

Outputs larger than the light tables (raw per-walk arrays, episode arrays) go to `scratch/` or to a directory passed
as an argument; both are ignored by git. Code takes paths as arguments, with one exception:
`experiments/E3_worldmodel_4zones/submit_kubeflow.py` (frozen) hard-codes the `/scratch/...` paths of E3's cluster
runs.

## Status

| | Experiment | Status | Branch of origin | Report |
|---|---|---|---|---|
| E0 | Offline tabular, 5x5 | finished, frozen | `main` | `reports/week1_offline_report.md` |
| E1 | Offline tabular, 4 zones | finished, frozen | `main` | `paper/reports/e1_report.tex`, `docs/E1_offline_4zones.md` |
| E2 | Online tabular, 4 zones | finished, frozen | `main` | `experiments/E2_online_4zones/RESULTS.md` |
| E3 | World model, 4 zones | finished, frozen | `e3/worldmodel` | `paper/reports/e3_report.tex`, `docs/E3_worldmodel_4zones.md` |
| E4 | Generalisation gradient | finished | `e4` | `results/E4_generalization_gradient/final/REPORT.md` |
| E5 | Manipulating the belief through training | finished | `e5`, `e5-robust` | `experiments/E5_belief_training/analysis_*/REPORT.md` |
| E6 | Behavioural signature of the belief | finished | `e6` | `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md` |
| E7 | Causal tests: oracle, cached values, correcting h | finished | `e6` | same report, sections 16-22 |
| E8 | Therapy and relapse | finished | `e8` | `experiments/E8_therapy_relapse/IMPLEMENTATION_REPORT.md`, sections 11-13 |

Details, decisions and key commits: `docs/EXPERIMENTS.md`.

## Method

- **Expected shapes before data.** Every hypothesis, threshold and decision rule is written in the experiment's
  DECISIONS file or report and committed before the computation it governs. Verdicts are reported as computed. A
  correction made after seeing a result is recorded as such, and anything unplanned is labelled exploratory.
- **Probes validated by what the agent reads.** A probe counts only if the reference model predicts its transitions
  (mass on the true successor >= 0.90, I_ref within 0.15 bit of log2 5). From E5 on, it also requires that the
  model localise the agent at the first step (localisation >= 0.90).
- **Identical replays.** Dynamics are deterministic from E4 on and every random stream is separate. Reruns on the
  same machine reproduce earlier outputs bit for bit. Campaigns check this: for example, E6 first replays E4's agents
  and must find their earlier tables unchanged. Every change to the library is checked against the finished
  experiments' reduced runs.
- **One implementation per quantity.** I(a; s'|s) is computed in one place (`measures/controllability.py`). Every
  other path either calls it or is tested against it.

## Licence

See `LICENSE`.
