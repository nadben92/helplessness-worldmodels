# E4 — Generalization pilot

**Status: not started.** Depends on E3.

## What it tests

Pilot for the generalisation campaign (E5): one contrasted pair of zones
(e.g. controllable vs. uncontrollable, same appearance family) plus its
control (see `helplessness.envs.dynamics.pairing.derive_control` -- the
control is derived automatically from the pilot's own config, never a
second hand-written one).

## Environment

World-model / pixel version of the base environment, one contrasted
zone pair + its paired control, generated via `derive_control`.

## Agent

World model (same as E3).

## Configuration

Not yet defined -- depends on which contrast E3's results motivate testing
first.
