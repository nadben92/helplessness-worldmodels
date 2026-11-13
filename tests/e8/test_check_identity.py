"""check_identity.py's --exposed-only (DECISIONS (26)): off by default every comparison enters the check; on, only the
exposed model's own quantities (b and the reference's left out)."""

from __future__ import annotations

from _experiment_loader import load_experiment_module

check = load_experiment_module("E8_therapy_relapse", "check_identity.py", "test_e8_check_identity")


def test_default_compares_everything():
    keys = (*check.E6_ARRAYS, *check.ROW_FIELDS, *check.PROBE_FIELDS, *check.READOUT)
    assert all(check.compared(role, key, False) for role in ("exposed", "reference") for key in keys)


def test_exposed_only_leaves_out_the_reference_and_b():
    assert not any(check.compared("reference", key, True) for key in (*check.E6_ARRAYS, *check.ROW_FIELDS, *check.READOUT))
    assert not check.compared("exposed", "b_step1", True) and not check.compared("exposed", "b_step1_probe", True)
    assert all(check.compared("exposed", key, True) for key in ("states", "directed_share", "valid", "V_zbar", "loc_template_step1"))
