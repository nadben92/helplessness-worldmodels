"""One integer seed -> every independent random stream of an E3 run, by
np.random.SeedSequence(seed).spawn, as E2 does (spec section 6;
docs/e3/DECISIONS.md, 2026-09-25). No component receives the integer
seed itself.

The first children are E2's own streams, in E2's order
(experiments/E2_online_4zones/rng.py::STREAM_NAMES, imported): for the
same seed, E3's env_dynamics / explore / start_state are the very
generators E2's spawn_streams(seed) produces. E3 appends its own streams
after them:

  theme          the per-run theme draw (build_themes)
  network_init   the global torch RNG, seeded once just before the model is built
  replay_sample  training minibatch windows
  latent_sample  every latent draw of the training step (a torch.Generator)
  switch_pairs   the label-free batch-pair statistic of the estimator switch
  measure        parent of every measurement's own streams (measure_streams)

explore is the collection policy's action stream (E2's epsilon-greedy
action choice); env_dynamics seeds the environment exactly as E2 does:
seed=int(streams["env_dynamics"].integers(2**31)).

Conditions (spec 6; docs/e3/DECISIONS.md, 2026-09-28). The experimental
environment's streams derive from SeedSequence(seed), as above (E2's
own). The reference environment shares ONLY the permutation (a function
of the seed index) and the themes (the experimental seed's theme stream,
taken by run_mini.build_run); every other stream of a reference run
derives from its own root SeedSequence([seed, 1]), so no stream is
correlated between an environment and its reference.
"""

from __future__ import annotations

import numpy as np
import torch

from helplessness.experiment_loader import load_experiment_module

e2_rng = load_experiment_module("E2_online_4zones", "rng.py", "e3_rng_streams_e2_rng")

E3_ONLY_STREAMS = ("theme", "network_init", "replay_sample", "latent_sample", "switch_pairs", "measure")
STREAM_NAMES = tuple(e2_rng.STREAM_NAMES) + E3_ONLY_STREAMS
N_MEASURE_SLOTS = 32
CONDITIONS = ("experimental", "reference")
_CONDITION_ENTROPY = {"reference": 1}


def _root(seed: int, condition: str) -> np.random.SeedSequence:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if condition == "experimental":
        return np.random.SeedSequence(seed)
    return np.random.SeedSequence([int(seed), _CONDITION_ENTROPY[condition]])


def spawn_sequences(seed: int, condition: str = "experimental") -> dict[str, np.random.SeedSequence]:
    children = _root(seed, condition).spawn(len(STREAM_NAMES))
    return dict(zip(STREAM_NAMES, children))


def spawn_streams(seed: int, condition: str = "experimental") -> dict[str, np.random.Generator]:
    """One numpy Generator per name in STREAM_NAMES. Deterministic: every
    call returns fresh generators in the same initial state, so separate
    callers (build_run, run_one_seed) may each take the streams they own."""
    return {name: np.random.default_rng(seq) for name, seq in spawn_sequences(seed, condition).items()}


def env_seed(streams: dict[str, np.random.Generator]) -> int:
    """The environment's dynamics seed, drawn from env_dynamics exactly as
    E2's sweep.py / pilot.py do."""
    return int(streams["env_dynamics"].integers(2**31))


def torch_seed(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def torch_generator(seq: np.random.SeedSequence, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(torch_seed(seq))
    return gen


def measure_streams(seed: int, slot: int, device: torch.device, condition: str = "experimental") -> tuple[np.random.Generator, torch.Generator]:
    """The (numpy, torch) pair owned by measurement `slot`: child `slot`
    of the measure stream. A measurement never draws from a training
    stream, so measuring more or less often never changes the training
    trajectory."""
    if not 0 <= slot < N_MEASURE_SLOTS:
        raise ValueError(f"measurement slot must be in [0, {N_MEASURE_SLOTS}), got {slot}")
    child = spawn_sequences(seed, condition)["measure"].spawn(N_MEASURE_SLOTS)[slot]
    return np.random.default_rng(child), torch_generator(child, device)
