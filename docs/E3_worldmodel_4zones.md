# E3 — World model validation, 4 zones

**Status: not started.** No world-model agent exists yet.

## What it tests

Whether a controllability axis emerges in a recurrent world model's latent
representation and is decodable, on the same 4-zone environment as E1/E2
(pixel version) -- the first step of the neural phase (see CLAUDE.md
roadmap step 2). Gated on E1 and E2 producing clean results first.

## Environment

Same 4 zones and dynamics as E1/E2, `PixelObservationBackend`
(`helplessness.envs.observation.pixel_backend.PixelObservationBackend`)
instead of the tabular index backend -- same underlying dynamics, only the
observation changes (see `helplessness.envs.torus_env.TorusZonedEnv`).

## Agent

Does not exist yet: `helplessness.agents.worldmodel/` (or equivalent) is
reserved but empty. Whatever gets built there must implement
`helplessness.agents.base.Agent` (specifically `transition_distribution`)
so `helplessness.measures.controllability.controllability_from_tensor` --
completely unchanged -- can be run on its belief exactly as it already
runs on the tabular agents' T_hat (see
`helplessness.agents.base.materialize_transition_tensor` and
`tests/agents/test_base.py`). How a world model actually produces a
comparable transition distribution over a large or continuous latent
space is an open design question for whoever builds this agent, not
something this restructuring resolves in advance.

## Configuration

Not yet defined.

## Caveat (from CLAUDE.md)

A successful linear decode from the latent does not prove the information
is used. Only intervention (E6) proves that.
