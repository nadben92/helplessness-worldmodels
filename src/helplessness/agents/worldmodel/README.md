# World-model agent (E3+)

**Status: complete for E3 (finished, tag `e3-final`).** `rssm.py` (encoder,
decoder, GRU core, categorical latent with straight-through sampling and
unimix, `latent_hook` extension point), `losses.py` (masked reconstruction,
exact KL dyn/rep), `replay.py` (episode buffer, model/label batch split),
`loop.py` (data collection, world-model training step, separation switch),
`acceleration.py`, and the closed-loop agent of E3's Phase 5 (`agent.py`,
`heads.py`, `policy.py`, `imagination.py`, `closed_loop.py`). E4-E7 extend
it with new modules or options whose default reproduces E3's behaviour
exactly. The run pipeline built on it lives in `helplessness.pipeline`,
the torch-side measurement probes in `helplessness.probing`.

Whether the eventual agent class implements
`helplessness.agents.base.Agent.transition_distribution(s, a)` directly,
or via a decoded/kernel-probe bridge (see
`measures/kernel_probe.py`, spec section 5.3), is an open design
question noted in `docs/e3/PHASE0_REPORT.md` and left for Phase 5 --
spec section 5.2's PRINCIPAL controllability measure is computed
directly in z-space (the canonical probe), not through a materialised
`(S, A, S)` tensor, so `transition_distribution` may end up being a
secondary/diagnostic path rather than the primary one.
