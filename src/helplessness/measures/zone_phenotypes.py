"""Behavioural phenotypes in a zone of interest (E6, docs/E6_phenotypes_latent.md, docs/e6/DECISIONS.md): arrays in,
arrays and scalars out. Nothing here imports an environment, an agent or torch.

Episodes come as padded arrays (one row per episode):
  states   (n, H + 1) int, s_0 .. s_T then -1;    actions (n, H) int, a_0 .. a_{T-1} then -1;
  length   (n,) int, T (the transitions taken);   zone (n_states,) bool, the zone of interest;
  dist     (n_states,) the toric distance of every state to the goal.
A "step in the zone" is a transition taken FROM a state of the zone. The phenotypes:

  first_exit         T_exit: transitions before the first state outside the zone (an episode starts inside it);
                     censored at T when the episode never leaves the zone (truncation; the goal is outside every
                     quadrant). Summarised by the Kaplan-Meier median (kaplan_meier), with the censored share.
  directed           among the steps in the zone, the share that lowers the toric distance to the goal.
  directed_by_index  the directed share at the j-th step of the FIRST stay in the zone (j = 1 .. T_exit), pooled over
                     episodes, with the number of episodes behind each j.
  recovery_latency   the scalar summary of directed_by_index fixed before any agent was measured (docs/e6/DECISIONS.md
                     (35)): the step index j_theta at which the pooled directed share first reaches theta, linearly
                     interpolated between consecutive indices, theta = halfway between chance and the epsilon-greedy
                     ceiling (recovery_threshold); 1 when the first step already reaches it; censored (NaN, with the
                     flag) when no index with at least n_min episodes reaches it.
  zone_share         the share of the episode's steps taken in the zone.
  stay_share         among the steps in the zone, the share of `stay` actions.
  revisit_share      among the steps in the zone, the share whose next state is one of the k previous states
                     (s_{t+1} in {s_{t+1-k}, .., s_t}; a `stay` is a revisit).
"""

from __future__ import annotations

import numpy as np

STAY = 4  # envs.dynamics.ACTION_NAMES[4] (not imported: measures/ never imports an environment module)


def step_mask(states: np.ndarray, length: np.ndarray) -> np.ndarray:
    """(n, H) bool: the transitions that exist (t < T)."""
    H = states.shape[1] - 1
    return np.arange(H)[None, :] < np.asarray(length)[:, None]


def in_zone_steps(states: np.ndarray, length: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """(n, H) bool: transition t exists and is taken from a state of the zone."""
    s = np.clip(states[:, :-1], 0, None)
    return step_mask(states, length) & np.asarray(zone, dtype=bool)[s]


def directed_steps(states: np.ndarray, length: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(n, H) bool: transition t exists and lowers the distance to the goal."""
    d = np.asarray(dist)
    s, s_next = np.clip(states[:, :-1], 0, None), np.clip(states[:, 1:], 0, None)
    return step_mask(states, length) & (d[s_next] < d[s])


def first_exit(states: np.ndarray, length: np.ndarray, zone: np.ndarray) -> dict[str, np.ndarray]:
    """time (n,): transitions before the first state outside the zone; censored (n,) bool: the episode never left it
    (time = T then). Raises if an episode does not start in the zone."""
    zone = np.asarray(zone, dtype=bool)
    n, length = states.shape[0], np.asarray(length)
    if not zone[states[:, 0]].all():
        raise ValueError("every episode must start in the zone")
    time = np.empty(n, dtype=np.int64)
    censored = np.zeros(n, dtype=bool)
    for i in range(n):
        path = states[i, : length[i] + 1]
        out = np.flatnonzero(~zone[path])
        if out.size:
            time[i] = out[0]
        else:
            time[i], censored[i] = length[i], True
    return {"time": time, "censored": censored}


def kaplan_meier(time: np.ndarray, censored: np.ndarray) -> dict:
    """Kaplan-Meier survival of T_exit: the distinct event times, S(t) just after each, and the median (the first time
    with S <= 0.5; NaN when S never falls to 0.5, i.e. more than half censored before the last event)."""
    time, event = np.asarray(time), ~np.asarray(censored, dtype=bool)
    times = np.unique(time[event])
    surv, s = [], 1.0
    for t in times:
        at_risk = int((time >= t).sum())
        d = int(((time == t) & event).sum())
        s *= 1.0 - d / at_risk
        surv.append(s)
    surv = np.asarray(surv)
    below = np.flatnonzero(surv <= 0.5)
    median = float(times[below[0]]) if below.size else float("nan")
    return {"times": times, "survival": surv, "median": median}


def directed_share(states: np.ndarray, length: np.ndarray, zone: np.ndarray, dist: np.ndarray) -> dict[str, np.ndarray]:
    """Per episode: n_zone_steps, n_directed (in the zone)."""
    inz = in_zone_steps(states, length, zone)
    d = directed_steps(states, length, dist)
    return {"n_zone_steps": inz.sum(axis=1), "n_directed": (inz & d).sum(axis=1)}


def directed_by_index(states: np.ndarray, length: np.ndarray, zone: np.ndarray, dist: np.ndarray, max_index: int) -> dict[str, np.ndarray]:
    """The j-th step of the first stay in the zone, j = 1 .. max_index: n (episodes still in their first stay at j) and
    directed (how many of them take a directed step), pooled; share = directed / n (NaN where n = 0)."""
    exit_ = first_exit(states, length, zone)
    d = directed_steps(states, length, dist)
    n = np.zeros(max_index, dtype=np.int64)
    k = np.zeros(max_index, dtype=np.int64)
    for i in range(states.shape[0]):
        m = min(int(exit_["time"][i]), max_index)  # steps 0 .. T_exit - 1 are taken from the zone
        n[:m] += 1
        k[:m] += d[i, :m]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return {"index": np.arange(1, max_index + 1), "n": n, "directed": k, "share": share}


def recovery_threshold(epsilon: float, n_actions: int = 5, n_directed_actions: int = 2) -> float:
    """theta: halfway between chance (n_directed_actions / n_actions: inside a quadrant two of the five actions bring the
    agent closer to the goal) and the epsilon-greedy ceiling (1 - epsilon + epsilon * chance)."""
    chance = n_directed_actions / n_actions
    return 0.5 * (chance + (1.0 - epsilon + epsilon * chance))


def recovery_latency(share: np.ndarray, n: np.ndarray, theta: float, n_min: int) -> dict:
    """j_theta from directed_by_index's share and n (module docstring). Only indices with n >= n_min are read, in order,
    up to the first one below n_min."""
    share, n = np.asarray(share, dtype=float), np.asarray(n)
    usable = n >= n_min
    last = int(np.argmin(usable)) if not usable.all() else len(n)
    for j in range(last):
        if share[j] >= theta:
            if j == 0:
                return {"latency": 1.0, "censored": False, "max_index_read": last}
            lo, hi = share[j - 1], share[j]
            return {"latency": float(j + (theta - lo) / (hi - lo)), "censored": False, "max_index_read": last}
    return {"latency": float("nan"), "censored": True, "max_index_read": last}


def zone_share(states: np.ndarray, length: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """(n,): the share of each episode's steps taken in the zone."""
    return in_zone_steps(states, length, zone).sum(axis=1) / np.maximum(np.asarray(length), 1)


def stay_counts(states: np.ndarray, actions: np.ndarray, length: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """(n,): `stay` actions among each episode's steps in the zone."""
    return (in_zone_steps(states, length, zone) & (actions == STAY)).sum(axis=1)


def revisit_counts(states: np.ndarray, length: np.ndarray, zone: np.ndarray, k: int) -> np.ndarray:
    """(n,): steps in the zone whose next state is one of the k previous states (s_{t+1-k} .. s_t)."""
    inz = in_zone_steps(states, length, zone)
    out = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[0]):
        for t in np.flatnonzero(inz[i]):
            if states[i, t + 1] in states[i, max(0, t + 1 - k): t + 1]:
                out[i] += 1
    return out


def summarize(states: np.ndarray, actions: np.ndarray, length: np.ndarray, zone: np.ndarray, dist: np.ndarray, epsilon: float,
              k_revisit: int = 4, n_min: int = 20, max_index: int | None = None) -> dict:
    """Every phenotype of a set of episodes, pooled (the per-episode quantities are pooled as ratios of sums, so an
    episode weighs by its steps in the zone)."""
    max_index = states.shape[1] - 1 if max_index is None else max_index
    ex = first_exit(states, length, zone)
    km = kaplan_meier(ex["time"], ex["censored"])
    ds = directed_share(states, length, zone, dist)
    by = directed_by_index(states, length, zone, dist, max_index)
    theta = recovery_threshold(epsilon)
    rec = recovery_latency(by["share"], by["n"], theta, n_min)
    nz = ds["n_zone_steps"].sum()
    ratio = lambda a: float(a.sum() / nz) if nz else float("nan")  # noqa: E731
    return {"exit_median_km": km["median"], "exit_censored_share": float(ex["censored"].mean()),
            "exit_mean_naive": float(ex["time"].mean()),
            "directed_share": ratio(ds["n_directed"]),
            "recovery_latency": rec["latency"], "recovery_censored": rec["censored"], "recovery_theta": theta,
            "recovery_max_index_read": rec["max_index_read"],
            "zone_share": float(in_zone_steps(states, length, zone).sum() / max(int(np.asarray(length).sum()), 1)),
            "stay_share": ratio(stay_counts(states, actions, length, zone)),
            "revisit_share": ratio(revisit_counts(states, length, zone, k_revisit)),
            "n_episodes": int(states.shape[0]), "n_zone_steps": int(nz),
            "directed_by_index": by, "km": km}


PHENOTYPES = ("exit_median_km", "directed_share", "recovery_latency", "zone_share", "stay_share", "revisit_share")


def bootstrap(states: np.ndarray, actions: np.ndarray, length: np.ndarray, zone: np.ndarray, dist: np.ndarray, epsilon: float,
              rng: np.random.Generator, n_boot: int = 200, strata: np.ndarray | None = None, level: float = 0.95,
              **kwargs) -> dict[str, tuple[float, float]]:
    """Percentile intervals of every scalar phenotype, episodes resampled with replacement within each stratum (the start
    state: every start keeps its number of episodes). NaN draws (censored) are left out of the percentiles; the share of
    them is returned as <name>_nan_share."""
    n = states.shape[0]
    strata = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    draws = {name: [] for name in PHENOTYPES}
    for _ in range(n_boot):
        idx = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        out = summarize(states[idx], actions[idx], length[idx], zone, dist, epsilon, **kwargs)
        for name in PHENOTYPES:
            draws[name].append(out[name])
    q = (1.0 - level) / 2.0
    res = {}
    for name, values in draws.items():
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(v)
        res[name] = (float(np.quantile(v[ok], q)), float(np.quantile(v[ok], 1 - q))) if ok.any() else (float("nan"), float("nan"))
        res[f"{name}_nan_share"] = float(1.0 - ok.mean())
    return res
