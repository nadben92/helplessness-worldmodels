"""w2_supplements.py: the validity table reads E4's criterion and counts invalid points per assignment; F's H window."""

from __future__ import annotations

from _experiment_loader import load_experiment_module

sup = load_experiment_module("E8_therapy_relapse", "w2_supplements.py", "e8_w2_supplements_under_test")


def _ref(pair, k, t, mass, i, valid):
    return {"pair": pair, "role": "reference", "image": f"assignment{k:02d}", "assignment": k, "treated": k < 3, "slot": "nw",
            "checkpoint": t, "walks_mass_step1": mass, "walks_I_step1_s1": i, "walks_valid": valid}


def test_validity_counts():
    rows = [_ref(p, 3, 250000, 0.03, 1.63, False) for p in sup.PAIRS] + [_ref(p, 0, 250000, 0.97, 2.29, True) for p in sup.PAIRS]
    val, counts = sup.validity(rows)
    assert len(val) == 6 and not any(v["mass_ok"] or v["I_ok"] for v in val if v["assignment"] == 3)
    by = {c["assignment"]: c for c in counts}
    assert by[3]["n_invalid"] == 3 and by[3]["n_mass_below"] == 3 and by[0]["n_invalid"] == 0


def test_F_window():
    assert sup.in_window(5000) and sup.in_window(25000) and not sup.in_window(40000) and not sup.in_window("> 250000")


def test_profile_crossings_read_valid_points_only():
    def pt(t, d, c3, mass, valid=True):
        return {"pair": "shuffle_A", "image": "assignment04", "nw_role": "neutral", "checkpoint": t, "valid": valid,
                "D_exp": d, "D_ref": 0.92, "C3_exp": c3, "C3_ref": 0.98, "L_dec_exp": 0.9, "L_dec_ref": 0.99,
                "mass_exp": mass, "mass_ref": 0.95, "I_exp": 1.0, "I_ref": 2.3}
    prof = [pt(0, 0.4, 0.4, 0.22), pt(5000, 0.9, 0.95, 0.95, valid=False), pt(10000, 0.9, 0.8, 0.5), pt(250000, 0.92, 0.97, 0.95)]
    rec = next(r for r in sup.profile_crossings(prof) if r["image"] == "assignment04")
    assert rec["D"] == 10000 and rec["C3"] == 250000 and rec["mass"] == 250000 and rec["I"] == "> 250000"
    assert abs(rec["mass_at_D"] - 0.5) < 1e-12 and abs(rec["C3_gap_at_D"] - 0.18) < 1e-12 and rec["mass_max_fall_below_0"] == 0.0
