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
