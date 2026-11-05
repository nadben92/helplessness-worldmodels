"""What the agent reads at each step of E6's episodes, beside its policy (docs/e6/DECISIONS.md (61), (62)), and the oracle
policy of E7's lookahead (E7 (52)): moved from E7's scripts on 2026-11-05 (docs/e8/DECISIONS.md (11)) so that E8 measures
its therapy checkpoints with the same code; bodies unchanged, E7's scripts import them back under their old names.

  readout_q_fn   D7's counterfactual q_fn (run_e7_d7.natural): at every step, lookahead.channel_q on a CLONE of the act
                 generator (the policy's own 16 prior draws) with the second z_bar (mean_zbar), V(h'_a, z_bar) for both
                 z_bar and Q with a common h are kept in a buffer, and the policy's own q_values is returned (checked
                 equal to channel_q's Q_normal): the episodes are E6's.
  oracle_q_fn    run_e7_diagnostics' lookahead q_fn: the policy on the oracle Q_k (true transitions, the agent's critic).
  assemble       the per-step buffers as (n, H, A) arrays (run_e7_d7.assemble).
  dissociation_seed, diagnostics_seed   the torch seeds of run_e7_dissociation.seed_of and run_e7_diagnostics.torch_seed.
"""

from __future__ import annotations

import numpy as np
import torch

from helplessness.agents.worldmodel.policy import q_values
from helplessness.probing.lookahead import channel_q, k_step_q

READOUT_KEYS = ("V_zbar", "V_zbar_mean", "Q_hbar")
DISSOCIATION_PURPOSE = {"alt": 10, "kernel": 11, "rollout": 12}  # run_e7_dissociation.PURPOSE
DIAGNOSTICS_PURPOSE = {"values": 0, "contexts": 1, "oracle": 2, "model": 3, "model_corrected": 4, "trace": 5, "lookahead": 6}  # run_e7_diagnostics.PURPOSE


def dissociation_seed(base: int, purpose: str, *extra) -> int:
    return int(np.random.SeedSequence([int(base), DISSOCIATION_PURPOSE[purpose], *[int(round(1000 * x)) for x in extra]])
               .generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def diagnostics_seed(seed: int, purpose: str, extra: int = 0) -> int:
    return int(np.random.SeedSequence([int(seed), DIAGNOSTICS_PURPOSE[purpose], int(extra)]).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def clone_generator(g: torch.Generator, device) -> torch.Generator:
    c = torch.Generator(device=device)
    c.set_state(g.get_state())
    return c


def assemble(buffer: list[dict], length: np.ndarray, shape: tuple, prefix: str) -> dict:
    """The per-call buffers (one per step, the active rows in order) as (n, H, A) arrays."""
    out = {}
    for k in buffer[0]:
        full = np.full(shape, np.nan)
        for t, b in enumerate(buffer):
            full[np.flatnonzero(length > t), t] = b[k]
        out[f"{prefix}{k}"] = full
    return out


def readout_q_fn(model, device, zgen: torch.Generator, buffer: list[dict]):
    """D7's natural q_fn for frozen_setup.run_model: appends READOUT_KEYS' arrays to `buffer` at every step and returns
    the policy's own Q (zgen: the z_bar draws' generator)."""
    def q_fn(h, z, s, g):
        ch = channel_q(model, h, z, clone_generator(g, device), zgen, mean_zbar=True)
        q = q_values(model.rssm, model.goal_head, model.critic, h, z, model.cfg.k_act, model.cfg, g)
        if not torch.equal(q, ch["Q_normal"]):
            raise RuntimeError("the counterfactual's Q differs from the policy's")
        buffer.append({k: ch[k].cpu().numpy() for k in READOUT_KEYS})
        return q
    return q_fn


def oracle_q_fn(model, tw, k: int, n_particles: int, k_inner: int, generator: torch.Generator, embeds):
    """run_e7_diagnostics' lookahead q_fn: the oracle Q_k (lookahead.k_step_q, variant "oracle") at every step."""
    def q_fn(h, z, s, _act_gen):
        return k_step_q(model, tw, h, z, s, k, "oracle", n_particles, k_inner, generator, embeds=embeds)["Q"][..., k - 1]
    return q_fn
