# helplessness

## Research question

This project studies the behavioural phenotypes of **perceived
uncontrollability** in a model-based reinforcement learning agent. An agent
is trained in environments with varying **controllability** -- the mutual
information between its actions and the resulting state transitions -- and
then evaluated, with learning disabled, in a fully controllable ("healthy")
environment. Because the agent is model-based, its transition belief `T_hat`
and reward belief `R_hat` are structurally separate, which makes perceived
controllability directly readable from the belief and, as it turns out,
directly editable. The long-term aim (see `CLAUDE.md`) is to train a neural
world model, extract the same phenotypes from its trajectories, and
causally intervene on its latent representation to identify what produces
them; this repository is the tabular rehearsal of that pipeline, in a
setting simple enough to verify every step against ground truth first.

## Paradigm

The design is a **yoked-control** paradigm in the sense of Maier & Seligman:
agents trained under different levels of environmental noise receive
statistically identical sensorimotor experience (verified empirically, see
`reports/week1_offline_report.md`) -- only the action-outcome contingency
differs. This isolates the causal pathway from *belief about control* to
*behaviour*, without confounding it with differences in exposure. Passivity
is treated as the default; what is measured is the acquisition (and, via
intervention, the manipulation) of the detection of control.

## Current status

**Week 1 of 6 months.** The offline phase (Phase 1: training under tunable
controllability; Phase 2: evaluation in the healthy environment; Phase 3:
component swaps to localise the phenotype in the belief) is complete,
including two causal interventions on the learned transition belief and
their comparison. See `reports/week1_offline_report.md` for the full
writeup. Next: an online model-based agent, to test the actual definitional
criterion of learned helplessness -- failure to *relearn* that control has
returned, which the offline design cannot probe by construction.

## Install and reproduce

```bash
pip install -e .
python scripts/regenerate_agents.py   # trains and saves the tabular agents (a few minutes)
```

That reproduces the offline sweep at both transition budgets
(`results/offline_sweep_5000.csv`, `results/offline_sweep_50000.csv`) and
the saved agents every other experiment depends on. From there, each
directory under `experiments/` is independently runnable -- see its
`README.md` for the exact commands and what it depends on. `pytest` from
the repo root runs the full test suite, including the model/simulation
agreement test that guarantees the environment's exact transition tensor
stays consistent with its step-by-step dynamics.

## Where things are

- `reports/` -- findings. Start with `week1_offline_report.md`.
- `experiments/` -- what was run to produce those findings: one directory
  per experiment, each with its own runner, config, and README stating what
  was tested and what came out.
- `src/helplessness/` -- the reusable library (environment, agents,
  planner, metrics, interventions, evaluation) that every experiment is
  built from. See `CONTRIBUTING.md` for the rule that keeps this directory
  from rotting into the experiments that use it.
- `CLAUDE.md` -- the scientific context and the invariants that must never
  be broken (Markov structure, terminated/truncated handling, RNG
  separation, and so on).

## A note on stability

This is research code from an ongoing internship. Interfaces will change as
the project moves from the tabular offline phase into an online agent and,
later, a neural world model -- pin a commit or tag (`v0.1-offline` marks the
end of the phase this README describes) if you need something that won't
move under you.
