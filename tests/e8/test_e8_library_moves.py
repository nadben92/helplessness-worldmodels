"""The E7 code moved to the library for E8 (docs/e8/DECISIONS.md (11)) gives what it gave in E7's scripts: the array
functions against verbatim copies of their old bodies, on random inputs; the moved names still importable from E7's
scripts. The torch side (readout and oracle q_fns) is checked bit for bit on reduced runs (IMPLEMENTATION_REPORT.md)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from helplessness.measures import decision_rule as dr
from helplessness.measures import zone_phenotypes as zp
from helplessness.probing import critic_readout as cr

N_STATES, A, N_COLS = 144, 5, 12


def _old_c_measures(V, Qh, s, directed, dist, succ, tol):  # analyse_e7_d7.c_measures before the move
    dirs = directed[s]
    rv = rankdata(V, axis=1)
    rd = rankdata(-dist[succ[s]], axis=1)
    rv, rd = rv - rv.mean(1, keepdims=True), rd - rd.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rv * rd).sum(1) / np.sqrt((rv ** 2).sum(1) * (rd ** 2).sum(1))
    return {"C1": dr.argmax_directed(V, dirs).astype(float), "C2": rho, "C3": dr.greedy_directed_share(Qh, dirs, tol)}


def _old_split(st, act, dec, in_zone, dist, directed):  # free_offsets_650k.py's loop body before the move
    s, s1 = st[:, :-1], st[:, 1:]
    zone = (s >= 0) & (s1 >= 0) & (act >= 0) & in_zone[np.clip(s, 0, None)]
    sv, s1v, av, dv = s[zone], s1[zone], act[zone], dec[zone].astype(int)
    good, directed_t = dv == sv, dist[s1v] < dist[sv]
    mis = ~good
    dd = dist[dv[mis]] - dist[sv[mis]]
    (r0, c0), (r1, c1) = np.divmod(sv[mis], N_COLS), np.divmod(dv[mis], N_COLS)
    err = np.abs(r0 - r1) + np.abs(c0 - c1)
    from_decoded = directed[dv[mis], av[mis]]
    return {"n_zone_steps": int(zone.sum()), "loc_cell": good.mean(),
            "mis_toward_goal": np.mean(dd < 0), "mis_same_distance": np.mean(dd == 0), "mis_away": np.mean(dd > 0),
            "mis_error_cells_mean": err.mean(), "mis_error_one_cell": np.mean(err == 1),
            "directed_localised": directed_t[good].mean(), "directed_mislocalised": directed_t[mis].mean(),
            "mis_directed_from_decoded": from_decoded.mean(), "mis_directed_both": np.mean(from_decoded & directed_t[mis])}


def _world(rng):
    succ = rng.integers(0, N_STATES, size=(N_STATES, A))
    dist = rng.integers(0, 12, size=N_STATES).astype(float)
    return succ, dist, dr.directed_action_mask(succ, dist)


def test_critic_readouts_equal_e7s_c_measures():
    rng = np.random.default_rng(0)
    succ, dist, directed = _world(rng)
    s = rng.integers(0, N_STATES, size=400)
    V = np.round(rng.normal(size=(400, A)), 1)  # ties included
    Qh = rng.normal(size=(400, A))
    new = dr.critic_readouts(V, Qh, directed[s], dist[succ[s]], 0.0751)
    old = _old_c_measures(V, Qh, s, directed, dist, succ, 0.0751)
    for k in ("C1", "C2", "C3"):
        np.testing.assert_array_equal(new[k], old[k])


def test_localisation_split_equals_free_offsets():
    rng = np.random.default_rng(1)
    succ, dist, directed = _world(rng)
    n, H = 30, 36
    states = rng.integers(0, N_STATES, size=(n, H + 1))
    actions = rng.integers(0, A, size=(n, H))
    length = rng.integers(5, H + 1, size=n)
    for i, T in enumerate(length):
        states[i, T + 1:], actions[i, T:] = -1, -1
    dec = np.where(rng.random((n, H)) < 0.7, states[:, :-1], rng.integers(0, N_STATES, size=(n, H))).astype(float)
    in_zone = np.zeros(N_STATES, dtype=bool)
    in_zone[rng.choice(N_STATES, 60, replace=False)] = True
    new = zp.localisation_split(states, actions, dec, in_zone, dist, directed, N_COLS)
    old = _old_split(states, actions, dec, in_zone, dist, directed)
    assert list(new) == list(old)
    for k in old:
        assert np.array_equal(np.asarray(new[k]), np.asarray(old[k]), equal_nan=True), k


def test_seeds_unchanged():
    # the values the E7 scripts drew before the move (run_e7_dissociation.seed_of, run_e7_diagnostics.torch_seed)
    expected_dis = int(np.random.SeedSequence([7107, 12, 0, 11000]).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))  # extras x 1000
    assert cr.dissociation_seed(7107, "rollout", 0.0, 11) == expected_dis
    expected_diag = int(np.random.SeedSequence([7107, 6, 1]).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    assert cr.diagnostics_seed(7107, "lookahead", 1) == expected_diag


def test_e7_scripts_keep_their_names():
    from _experiment_loader import load_experiment_module

    rd7 = load_experiment_module("E7_latent_manipulation", "run_e7_d7.py", "e8_check_rd7")
    rdis = load_experiment_module("E7_latent_manipulation", "run_e7_dissociation.py", "e8_check_rdis")
    rdiag = load_experiment_module("E7_latent_manipulation", "run_e7_diagnostics.py", "e8_check_rdiag")
    assert rd7.KEYS == ("V_zbar", "V_zbar_mean", "Q_hbar") and rd7.clone is cr.clone_generator and rd7.assemble is cr.assemble
    assert rdis.PURPOSE == {"alt": 10, "kernel": 11, "rollout": 12} and rdis.seed_of is cr.dissociation_seed
    assert rdiag.PURPOSE["lookahead"] == 6 and rdiag.torch_seed is cr.diagnostics_seed
