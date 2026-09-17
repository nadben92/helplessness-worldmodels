"""Etape B -- generative collection and the plug-in/Dirichlet estimators.

Generative model, never trajectories (per spec 3.2): exactly n draws for
each (s, a) pair, sampled directly from the true transition tensor's row --
no episode, no truncation, no exploration. Reuses
agents.tabular.TabularModelBasedAgent for T_hat/R_hat rather than
reimplementing plug-in estimation: alpha=0.0 (its Dirichlet prior weight)
reproduces the spec's primary MLE estimator exactly (verified: with every
(s,a) visited n>=1 times by construction, counts = N + 0 = N, so
T_hat = N/N.sum(axis=2) = N/n, and an unobserved successor gets exactly 0
-- see tests/experiments/e1/test_estimation.py). The Dirichlet robustness
variant (alpha in {0.5, 1}, restricted to the 5 geographically reachable
successors) reuses the same class's existing alpha_support mechanism,
unmodified.
"""

from __future__ import annotations

import numpy as np

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.envs.zones import geographic_support_mask


def spawn_exp_ctrl_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent RNG streams for the experimental and control
    environments' generative collection, both derived from the same
    integer seed via SeedSequence.spawn() -- NOT
    np.random.default_rng(seed) called twice with the same seed value.

    That naive pattern was a real bug (found by inspecting actual output,
    not anticipated): for a zone derive_control() leaves unchanged
    (already-controllable), P_exp[s,a] == P_ctrl[s,a] bit-for-bit, and
    numpy's Generator.multinomial's underlying stream-consumption pattern
    for a fixed (n, len(p)) turned out to stay in lockstep between the two
    separately-seeded-but-identical Generators across the ENTIRE state
    loop, not just the zones with identical P -- so N_exp and N_hat came
    out bit-identical in every controllable zone, and b measured exactly
    0 with zero variance there. That's true by construction, not by
    measurement, and it means no detection threshold was ever measured
    anywhere H0 (b=0) actually holds. SeedSequence.spawn() breaks the
    correlation regardless of the underlying algorithm's stream-
    consumption pattern -- the two children are statistically independent
    by construction, not merely "different seeds".
    """
    seed_seq = np.random.SeedSequence(seed)
    child_exp, child_ctrl = seed_seq.spawn(2)
    return np.random.default_rng(child_exp), np.random.default_rng(child_ctrl)


def collect_generative(P: np.ndarray, n_per_pair: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly n_per_pair draws for every (s, a) pair, sampled from P[s,a]
    -- n_states * n_actions * n_per_pair draws total, no episode/trajectory
    notion at all. Returns (S, A, S) int64 counts. Vectorized per (s,a) via
    a single multinomial draw (equivalent to n_per_pair iid categorical
    draws, exact, not an approximation) rather than a Python-level loop
    over individual samples -- n_per_pair reaches 1000 and there are 720
    (s,a) pairs on this layout, so the per-sample loop this replaces would
    run up to 720,000 times per (environment, seed, budget) combination.
    """
    n_states, n_actions, _ = P.shape
    counts = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
    for s in range(n_states):
        for a in range(n_actions):
            counts[s, a] = rng.multinomial(n_per_pair, P[s, a])
    return counts


def geographic_alpha_support(grid_shape: tuple[int, int]) -> np.ndarray:
    """(S, S) bool: state s's geographically reachable successors -- its 5
    torus neighbours (up/down/left/right/stay), the same 5 cells every
    DynamicsSpec in this registry ever puts mass on regardless of type.
    Used only for the Dirichlet robustness variant's alpha_support
    (spec 3.3: "restreint aux 5 successeurs geographiquement atteignables"),
    never the default estimator.

    Promoted to helplessness.envs.zones.geographic_support_mask (pure grid
    geometry, no dynamics/Env dependency, reused by E2's online agent) --
    this is a thin re-export kept for E1's own import path and byte-for-byte
    behaviour, never reimplemented here.
    """
    return geographic_support_mask(grid_shape)


def agent_from_counts(
    counts: np.ndarray,
    R_true: np.ndarray,
    alpha: float = 0.0,
    alpha_support: np.ndarray | None = None,
) -> TabularModelBasedAgent:
    """Builds a TabularModelBasedAgent directly from generative counts
    (bypassing a per-sample .observe() loop -- reward here is a fully
    known, deterministic function of the arrival state, R_true[s'], so
    R_sum/R_count can be assigned in closed form: every draw landing on s'
    contributes exactly R_true[s'], so R_hat[s'] == R_true[s'] exactly
    wherever s' was reached by any (s,a) pair, matching what n calls to
    .observe() would have produced).

    alpha=0.0 (default) is the primary plug-in MLE estimator (see module
    docstring); alpha in {0.5, 1} with alpha_support=geographic_alpha_support(...)
    is the robustness variant -- config-selectable, never the default.
    """
    n_states, n_actions, _ = counts.shape
    agent = TabularModelBasedAgent(n_states, n_actions, alpha=alpha, alpha_support=alpha_support)
    agent.N = counts.copy()
    arrivals_per_state = counts.sum(axis=(0, 1))  # (S,): total draws landing on each s', across every (s,a)
    agent.R_count = arrivals_per_state.copy()
    agent.R_sum = arrivals_per_state * R_true
    return agent
