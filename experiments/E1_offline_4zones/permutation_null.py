"""Two nulls for Ihat (and for b), answering two different questions --
neither is a correction of the other:

- The SHUFFLE null is conditional on the observed data: it takes the
  (action, successor) draws for every state in the zone, shuffles the
  action label (keeping which successor each draw produced fixed), and
  recomputes the zone-aggregated Ihat under the shuffled labels, 500
  times. This is an EXACT test of H0: "action has no effect on the
  successor distribution" -- use it to decide whether Ihat is
  distinguishable from zero. In a deterministic zone every draw for a
  given state has the SAME successor value regardless of action, so
  shuffling changes nothing and the null is degenerate at exactly 0 --
  the sanity check the spec calls out explicitly (tested).

- The BOOTSTRAP null is unconditional: for each action, independently,
  resample n successors WITH REPLACEMENT from THAT ACTION'S OWN observed
  empirical distribution (not pooled across actions) -- this preserves
  whatever action-dependence the data actually shows (real, in a
  controllable zone; spuriously small, in an uncontrollable one) and
  describes the spread Ihat/b would show on a FRESH dataset drawn from the
  same estimated process. Use it to quote a precision for E2/E4.

A small difference between the two is expected and invalidates neither.
Earlier text in this module claimed the shuffle null was "too narrow" and
attributed it to hypergeometric shrinkage relative to the bootstrap --
that claim is WITHDRAWN. A permutation test conditional on the observed
counts is exact, by construction; it is not an approximation the
bootstrap corrects.

An earlier version of the bootstrap POOLED all actions' draws into one
distribution before resampling (enforcing action-independence by
construction) -- that was the right choice for a null of I=0 (matches the
shuffle null's own hypothesis), but WRONG when reused for b's headline
detectability in the controllable zones, where the true per-action
dependence is real and should NOT be nulled out: pooling collapsed the
"fresh dataset" question into "what if this zone had no action effect",
which is a variance-of-a-statistic-near-its-null-boundary quantity
(O(1/n), matching Wilks'-theorem-style chi-squared asymptotics for MI
under independence) rather than the ordinary sampling variance of Ihat
around its own true value (O(1/sqrt(n))). The per-action resampling below
is the fix: each action keeps its own estimated distribution, so a
controllable zone's bootstrap null now scales like an ordinary estimator's
sampling noise, not like an independence-test statistic.

Performance note: both nulls are computed VECTORIZED across all
n_permutations draws in one batch of numpy calls per state (not a Python
loop calling one draw at a time) -- a per-permutation Python loop here was
the actual bottleneck in an earlier version (500 x 20 seeds x 5 budgets x
up to 12 (zone, quantity, method) combinations is 600,000 individual
resample-and-recount operations; looping in Python made that take the
better part of an hour, not because more compute was fundamentally
needed, but because numpy's per-call overhead was paid 600,000 times
instead of ~1,200). The per-STATE loop (<=25 states per zone) stays a
plain Python loop -- cheap at this size, and keeps the code readable.
"""

from __future__ import annotations

import numpy as np

from helplessness.measures.controllability import controllability_per_state


def _expand_state_samples(N_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N_s: (n_actions, n_states) counts for one state. Returns
    (actions, successors), one entry per individual draw -- the
    sample-level record the shuffle needs (aggregated counts alone don't
    carry which successor came from which action's draws once shuffled)."""
    actions_list = []
    successors_list = []
    for a in range(N_s.shape[0]):
        succ_for_a = np.repeat(np.arange(N_s.shape[1]), N_s[a])
        actions_list.append(np.full(succ_for_a.shape, a, dtype=np.int64))
        successors_list.append(succ_for_a)
    return np.concatenate(actions_list), np.concatenate(successors_list)


def _permuted_rows_batch(
    actions: np.ndarray, successors: np.ndarray, n_actions: int, n_states: int, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized counterpart of shuffling the action column n_permutations
    times: draws all n_permutations independent shuffles of `actions` in
    one batch (argsort of random keys, a standard vectorized-shuffle
    trick), then builds all n_permutations count tensors with a single
    np.add.at call instead of one per permutation. Returns
    (n_permutations, n_actions, n_states) plug-in (alpha=0) rows."""
    n_draws = actions.shape[0]
    if n_draws == 0:
        return np.zeros((n_permutations, n_actions, n_states))
    perm_idx = np.argsort(rng.random((n_permutations, n_draws)), axis=1)
    shuffled_actions = actions[perm_idx]  # (n_permutations, n_draws)

    # np.add.at is a slow, unbuffered scatter-add for arrays this size
    # (hundreds of thousands to millions of increments) -- np.bincount on
    # a single flattened index is a highly optimized C routine and is
    # dramatically faster for exactly this "count occurrences" pattern.
    perm_axis = np.repeat(np.arange(n_permutations), n_draws)
    succ_tiled = np.tile(successors, n_permutations)
    flat_idx = (perm_axis * n_actions + shuffled_actions.ravel()) * n_states + succ_tiled
    flat_counts = np.bincount(flat_idx, minlength=n_permutations * n_actions * n_states)
    counts = flat_counts.reshape(n_permutations, n_actions, n_states)

    row_sums = counts.sum(axis=2, keepdims=True)
    return counts / row_sums


def _zone_aggregate_i_under_permutation_batch(
    N: np.ndarray, zone_state_indices: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Returns (n_permutations,): the zone-aggregated Ihat under the
    shuffle null, for all n_permutations draws at once."""
    n_states_total, n_actions, _ = N.shape
    zone_size = zone_state_indices.size
    per_state_per_perm = np.empty((zone_size, n_permutations))
    for i, s in enumerate(zone_state_indices):
        actions, successors = _expand_state_samples(N[s])
        rows = _permuted_rows_batch(actions, successors, n_actions, n_states_total, n_permutations, rng)
        per_state_per_perm[i] = controllability_per_state(rows)
    return per_state_per_perm.mean(axis=0)


def permutation_null_i_zone(
    N: np.ndarray, zone_state_indices: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Null distribution (n_permutations values) of the zone-aggregated
    Ihat under the action-shuffle null, for ONE environment's raw counts
    N (S, A, S) -- the exact test of H0: I=0 (see module docstring)."""
    return _zone_aggregate_i_under_permutation_batch(N, zone_state_indices, n_permutations, rng)


def permutation_null_b_zone(
    N_exp: np.ndarray,
    N_ctrl: np.ndarray,
    zone_state_indices: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null distribution of b = Ihat_exp - Ihat_ctrl under the shuffle
    null (both environments shuffled independently, same n_permutations
    draws) -- see module docstring: b's null is not the same as either
    term's alone."""
    i_exp = _zone_aggregate_i_under_permutation_batch(N_exp, zone_state_indices, n_permutations, rng)
    i_ctrl = _zone_aggregate_i_under_permutation_batch(N_ctrl, zone_state_indices, n_permutations, rng)
    return i_exp - i_ctrl


def _resample_rows_per_action_batch(
    N_s: np.ndarray, n_per_action: int, n_actions: int, n_states: int, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized per-action bootstrap: for each action, draws all
    n_permutations count vectors in ONE rng.multinomial call (see the
    inline comment below on why multinomial, not choice). Returns
    (n_permutations, n_actions, n_states)."""
    counts = np.zeros((n_permutations, n_actions, n_states), dtype=np.int64)
    for a in range(n_actions):
        p_a = N_s[a] / N_s[a].sum()
        # rng.multinomial(n, p, size=n_permutations) draws n_permutations
        # independent Multinomial(n_per_action, p_a) COUNT vectors directly
        # -- rng.choice(..., p=...) instead draws n_per_action individual
        # categorical samples per permutation (a per-sample cumulative-
        # probability search), which is what actually made this the
        # bottleneck (profiled: ~4s of a ~5s call). multinomial() is the
        # right primitive since only the resulting counts are ever used.
        counts[:, a, :] = rng.multinomial(n_per_action, p_a, size=n_permutations)
    row_sums = counts.sum(axis=2, keepdims=True)
    return counts / row_sums


def per_state_i_under_bootstrap_batch(
    N: np.ndarray, zone_state_indices: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-state (NOT zone-averaged) Ihat under the per-action bootstrap:
    (zone_size, n_permutations). Exposed (not underscore-prefixed) for
    diagnostic use -- e.g. checking whether the zone-aggregate's log-log
    detectability slope differs from individual states' own slopes, to
    tell a real per-state effect apart from an aggregation-of-
    heterogeneous-rates artifact (see docs/E1_offline_4zones.md's slope
    investigation and report.py's per-state slope check)."""
    n_states_total, n_actions, _ = N.shape
    zone_size = zone_state_indices.size
    per_state_per_perm = np.empty((zone_size, n_permutations))
    for i, s in enumerate(zone_state_indices):
        n_per_action = int(N[s, 0].sum())  # every action has exactly n draws, by collect_generative's design
        rows = _resample_rows_per_action_batch(N[s], n_per_action, n_actions, n_states_total, n_permutations, rng)
        per_state_per_perm[i] = controllability_per_state(rows)
    return per_state_per_perm


def _zone_aggregate_i_under_bootstrap_batch(
    N: np.ndarray, zone_state_indices: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    return per_state_i_under_bootstrap_batch(N, zone_state_indices, n_permutations, rng).mean(axis=0)


def bootstrap_null_i_zone(
    N: np.ndarray, zone_state_indices: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Null distribution (n_permutations values) of the zone-aggregated
    Ihat under the per-action resample-with-replacement bootstrap, for
    ONE environment's raw counts N (S, A, S) -- the "fresh dataset"
    precision estimate, see module docstring."""
    return _zone_aggregate_i_under_bootstrap_batch(N, zone_state_indices, n_permutations, rng)


def bootstrap_null_b_zone(
    N_exp: np.ndarray,
    N_ctrl: np.ndarray,
    zone_state_indices: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap counterpart to permutation_null_b_zone(): both
    environments resampled per-action, independently, same n_permutations
    draws. In an already-controllable zone (derive_control is a no-op,
    both sides share the same true distribution) this describes the
    sampling variability of b around its true value of 0 -- the right
    precision estimate for the headline detectability table."""
    i_exp = _zone_aggregate_i_under_bootstrap_batch(N_exp, zone_state_indices, n_permutations, rng)
    i_ctrl = _zone_aggregate_i_under_bootstrap_batch(N_ctrl, zone_state_indices, n_permutations, rng)
    return i_exp - i_ctrl


def detectable_effect(null_values: np.ndarray, percentile: float = 95.0) -> float:
    """The smallest detectable effect at this budget: the given percentile
    of |null_values| -- a deliverable in its own right (spec 3.5), not an
    intermediate. abs() because both Ihat (>=0 always) and b (predicted
    negative) are compared by magnitude against a null centred at 0 (Ihat)
    or symmetric around 0 (b)."""
    return float(np.percentile(np.abs(null_values), percentile))
