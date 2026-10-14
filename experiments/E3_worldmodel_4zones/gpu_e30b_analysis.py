"""Analysis of the second GPU batch (gpu_e30b), read-only on the extracted run directories
(results/E3_worldmodel_4zones/*gpu_e30b*): no measure is recomputed, nothing is trained. Everything
printed comes from each run's results.json, arrays.npz, diagnostics.jsonl and console.log.

Usage, from the repo root:  python experiments/E3_worldmodel_4zones/gpu_e30b_analysis.py [section ...]
(in reports/ until 2026-10-14; moved here, paths adjusted to the new depth, logic unchanged)
sections: table calib is traj sep consistency   (default: all)
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.controllability import controllability_per_state

ROOT = Path(__file__).resolve().parents[2] / "results" / "E3_worldmodel_4zones"
agg = load_experiment_module("E3_worldmodel_4zones", "aggregate_b.py", "analysis_aggregate_b")
CATS = list(agg.CATEGORIES)
SHORT = {"uncontrollable_deterministic": "ud", "controllable_noisy": "cn", "uncontrollable_noisy": "un", "controllable_deterministic": "cd"}
NOISY = ("controllable_noisy", "uncontrollable_noisy")
FIRST_BATCH_REF = "*control*worldmodel-gpu_e30__seed00*"


def runs() -> dict[str, dict]:
    out = {}
    for d in sorted(glob.glob(str(ROOT / "*worldmodel-gpu_e30b_*"))):
        d = Path(d)
        if not (d / "results.json").exists():
            continue
        m = re.search(r"pair-(\w+)__worldmodel-gpu_e30b_(win\d+)__seed(\d+)", d.name)
        role, win, seed = m.group(1), m.group(2), int(m.group(3))
        r = json.loads((d / "results.json").read_text())
        out[f"{win} s{seed} {'exp' if role == 'experimental' else 'ref'}"] = {
            "dir": d, "res": r, "arr": np.load(d / "arrays.npz"), "meta": yaml.safe_load((d / "meta.yaml").read_text()),
            "diag": [json.loads(line) for line in (d / "diagnostics.jsonl").read_text().splitlines()],
            "win": win, "seed": seed, "cond": "experimental" if role == "experimental" else "reference"}
    order = [f"{w} s{s} {c}" for w in ("win64", "win16") for s in (0, 5) for c in ("exp", "ref")]
    return {k: out[k] for k in order if k in out}


def t4(run: dict, zone: str) -> dict:
    """p_hat at the true frequent outcome of each (s, a) of the zone (real-context kernel), mean over pairs and se over states."""
    r, a = run["res"], run["arr"]
    zone_of = np.array(r["zone_of_state"])
    p_hat, P = a["real_p_hat"], a["P_true"]
    freq = P.argmax(-1)
    per_state = []
    for s in np.flatnonzero(zone_of == zone):
        if np.isnan(p_hat[s]).any():
            continue
        per_state.append(np.mean([p_hat[s, act, freq[s, act]] for act in range(5)]))
    per_state = np.array(per_state)
    truth = np.mean([P[s, act, freq[s, act]] for s in np.flatnonzero(zone_of == zone) for act in range(5)])
    return {"mean": float(per_state.mean()), "se": float(per_state.std(ddof=1) / np.sqrt(per_state.size)), "truth": float(truth)}


def theory(run: dict) -> dict[str, float]:
    zone_of = np.array(run["res"]["zone_of_state"])
    I = controllability_per_state(run["arr"]["P_true"])
    return {c: float(I[zone_of == c].mean()) for c in CATS}


def sec_table(R):
    print("\n## Per run: switch, calibration, I_s / I_z, T4, probes, gap, durations")
    for k, run in R.items():
        r, m = run["res"], run["meta"]
        proc, rc = r["training_procedure"], r["real_context"]
        cal = agg.calibration(r, run["arr"])
        th = theory(run)
        print(f"\n### {k}  ({run['dir'].name[-22:]}) class {r['zone_class']}; dynamics {r['zone_kinds']}")
        print(f"switch {proc['switch_step']} margin {proc['switch_deadline'] - proc['switch_step']} (deadline {proc['switch_deadline']}); "
              f"lr switch {proc['lr_switch_step']}; final step {r['final_step']}")
        print(f"C-a {cal['c_a']['value']:.4f} {'ok' if cal['c_a']['ok'] else 'FAIL'}; C-c {cal['c_c']['value']:.4f} {'ok' if cal['c_c']['ok'] else 'FAIL'}; "
              f"real-context acc {rc['acc_real_context']:.4f}; K {rc['kernel_draws']}")
        for c in CATS:
            v = cal["c_b"][c]
            iz, i_s = rc["I_z_by_zone"][c], rc["I_s_by_zone"][c]
            t = t4(run, c) if c in NOISY else None
            gap = (r["is_per_context"] or {}).get("by_zone", {}).get(c)
            print(f"  {SHORT[c]:2s} C-b L1_WM {v['l1_wm']:.3f} vs L1_B+.05 {v['l1_b'] + .05:.3f} floor {v['floor']:.3f} n{v['n_pairs']} {'ok' if v['ok'] else 'FAIL'} | "
                  f"I_s {i_s['mean']:.3f}±{i_s['se']:.3f} (bias {i_s['bias']:.3f}) I_z {iz['mean']:.3f}±{iz['se']:.3f} theory {th[c]:.3f} | "
                  + (f"T4 {t['mean']:.3f}±{t['se']:.3f} (true {t['truth']:.2f}) " if t else "T4 - ")
                  + (f"| gap pooled-per_ctx {gap['gap']:+.3f}±{gap['gap_se_over_states']:.3f} (per-ctx {gap['per_context']:.3f}, pooled {gap['pooled']:.3f})" if gap else ""))
        print(f"  probes (full replay) h->s {r['h_only_probe']['test_accuracy']:.4f}, z->s {r['z_only_probe']['test_accuracy']:.4f}; "
              f"(real contexts) h->s {rc['state_probe_h']:.4f}, z->s {rc['state_probe_z']:.4f}; post. entropy real {rc['posterior_entropy']:.2f}")
        diag_s = sum(d["seconds"] for d in run["diag"][:-1])
        t0, t1 = m["started_at"], m["updated_at"]
        print(f"  training {proc['training_seconds'] / 3600:.2f} h (incl. {len(run['diag']) - 1} diagnostic points, {diag_s / 3600:.2f} h); final diagnostic "
              f"{run['diag'][-1]['seconds']:.0f} s; meta {t0} -> {t1}; gpu peak final {r['gpu_memory_final']}")


def sec_calib(R):
    print("\n## Calibration summary (C-a >= .99, C-c >= .99, C-b per zone)")
    for k, run in R.items():
        cal = agg.calibration(run["res"], run["arr"])
        print(f"{k:14s} C-a {'ok ' if cal['c_a']['ok'] else 'FAIL'}{cal['c_a']['value']:.4f} C-c {'ok ' if cal['c_c']['ok'] else 'FAIL'}{cal['c_c']['value']:.4f} | "
              + " ".join(f"{SHORT[c]} {'ok' if v['ok'] else 'FAIL'}({v['l1_wm']:.2f}/{v['l1_b'] + .05:.2f})" for c, v in cal["c_b"].items()))


def sec_is(R):
    print("\n## I_s and I_z vs theory")
    for k, run in R.items():
        rc, th = run["res"]["real_context"], theory(run)
        print(f"{k:14s} " + " | ".join(f"{SHORT[c]} th {th[c]:.2f} I_s {rc['I_s_by_zone'][c]['mean']:.3f} I_z {rc['I_z_by_zone'][c]['mean']:.3f}" for c in CATS))


def sec_traj(R):
    print("\n## Trajectories (diagnostic points): I_s by zone | L1(WM,P) by zone | T4 preview cn/un | h->s, z->s (contexts) | acc real")
    for k, run in R.items():
        print(f"\n{k}")
        for d in run["diag"]:
            rc = d["real_context"]
            print(f"  {d['step']:6d} I_s " + "/".join(f"{rc['I_s_by_zone'][c]['mean']:.2f}" for c in CATS)
                  + " | L1 " + "/".join(f"{rc['l1_vs_p_by_zone'][c]:.3f}" for c in CATS)
                  + f" (floor {rc['l1_floor_by_zone']['controllable_noisy']:.3f}, K {rc['kernel_draws']})"
                  + " | T4 " + "/".join(f"{rc['frequent_outcome'][c]:.3f}" for c in NOISY)
                  + f" | probes {rc['state_probe_h']:.3f}/{rc['state_probe_z']:.3f} | acc {rc['acc_real_context']:.3f} | ent {rc['posterior_entropy']:.1f}")


def sec_sep(R):
    print("\n## Separation: batch-pair JS trajectory")
    for k, run in R.items():
        h = np.array(run["res"]["training_procedure"]["batch_pair_js"], dtype=float)
        first = lambda thr: int(h[h[:, 1] > thr][0, 0]) if (h[:, 1] > thr).any() else None
        sw = run["res"]["training_procedure"]["switch_step"]
        pre = h[h[:, 0] < sw]
        seg = lambda lo, hi: (float(np.median(h[(h[:, 0] >= lo) & (h[:, 0] < hi), 1])) if ((h[:, 0] >= lo) & (h[:, 0] < hi)).any() else float("nan"))
        print(f"{k:14s} switch {sw} | first JS>1e-3 {first(1e-3)} >0.1 {first(0.1)} >1 {first(1.0)} >3 {first(3.0)} | max before switch {pre[:, 1].max():.2f} | "
              f"median JS [0,1k) {seg(0, 1000):.1e} [3k,5k) {seg(3000, 5000):.1e} [10k,20k) {seg(10000, 20000):.2f} [40k,60k) {seg(40000, 60000):.2f}")
        cons = run["dir"] / "console.log"
        lines = [l for l in cons.read_text().splitlines() if re.search(r"step\s+\d+/", l)]
        pick = [l for l in lines if any(re.search(rf"step\s+{s}/", l) for s in (1000, 3000, 5000, 7000, 10000, 20000, 40000, 59000))]
        for l in pick:
            g = re.search(r"step\s+(\d+)/.*?batch-pair JS\s+([\d.]+).*?recon_patch ([\d.]+) \| grad_norm ([\d.e+]+)", l)
            if g:
                print(f"      {g.group(1):>6s}: JS {g.group(2)} recon_patch {g.group(3)} grad {g.group(4)}")


def sec_consistency(R):
    ref = glob.glob(str(ROOT / FIRST_BATCH_REF))
    print("\n## First-batch spec-network reference seed 0 vs win64 s0 ref")
    if ref:
        r0 = json.loads((Path(ref[0]) / "results.json").read_text())
        a0 = np.load(Path(ref[0]) / "arrays.npz")
        run0 = {"res": r0, "arr": a0}
        cal0 = agg.calibration(r0, a0)
        print("first batch (spec network, 16x64, 60000 steps):")
        for c in CATS:
            rc = r0["real_context"]
            t = t4(run0, c) if c in NOISY else None
            print(f"  {SHORT[c]} I_z {rc['I_z_by_zone'][c]['mean']:.3f} L1 {rc['l1_vs_p_by_zone'][c]:.3f} C-b {'ok' if cal0['c_b'][c]['ok'] else 'FAIL'}" + (f" T4 {t['mean']:.3f}" if t else ""))
        print(f"  C-a {cal0['c_a']['value']:.4f} C-c {cal0['c_c']['value']:.4f} switch {r0['training_procedure']['switch_step']}")
    for k in ("win64 s0 ref", "win16 s0 ref"):
        if k in R:
            print(k, "-> see table")


def _wall_hours(run: dict):
    """(training h, diagnostic points h, final measures min) from results.json, meta.yaml and console.log."""
    from datetime import datetime

    def parse(x):
        return datetime.strptime(str(x).replace("Z", "").replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S")

    r, m = run["res"], run["meta"]
    collection = float(re.search(r"collection done in ([\d.]+)s", (run["dir"] / "console.log").read_text()).group(1))
    tr = r["training_procedure"]["training_seconds"]
    wall = (parse(m["updated_at"]) - parse(m["started_at"])).total_seconds()
    diag = sum(d["seconds"] for d in run["diag"][:-1])
    return tr / 3600, diag / 3600, (wall - tr - collection) / 60


def sec_md(R):
    """Markdown tables of the report, generated from the data."""
    f2 = lambda x: f"{x:.3f}"
    print("### Table A -- switch, probes, durations\n")
    print("| run | switch (margin to 25,000) | h -> s / z -> s (full replay) | h -> s / z -> s (real contexts) | training h (of which 10 diagnostic points) | final measures min |")
    print("|---|---|---|---|---|---|")
    for k, run in R.items():
        r, rc = run["res"], run["res"]["real_context"]
        p = r["training_procedure"]
        tr, dg, fin = _wall_hours(run)
        print(f"| {k} | {p['switch_step']} ({p['switch_deadline'] - p['switch_step']}) | {r['h_only_probe']['test_accuracy']:.3f} / {r['z_only_probe']['test_accuracy']:.3f} | "
              f"{rc['state_probe_h']:.3f} / {rc['state_probe_z']:.3f} | {tr:.2f} ({dg:.2f}) | {fin:.0f} |")
    print("\n### Table B -- calibration (C-a, C-c >= 0.99; C-b: L1(WM, P) <= L1(B, P) + 0.05 over the pairs with N[s,a] >= 100; floor of the kernel's draws in brackets)\n")
    print("| run | C-a | C-c | C-b ud | C-b cn | C-b un | C-b cd |")
    print("|---|---|---|---|---|---|---|")
    for k, run in R.items():
        cal = agg.calibration(run["res"], run["arr"])
        cells = []
        for c in CATS:
            v = cal["c_b"][c]
            cells.append(f"{v['l1_wm']:.3f} vs {v['l1_b'] + 0.05:.3f} ({v['floor']:.3f}) **{'ok' if v['ok'] else 'FAIL'}**")
        print(f"| {k} | {cal['c_a']['value']:.4f} {'ok' if cal['c_a']['ok'] else '**FAIL**'} | {cal['c_c']['value']:.4f} {'ok' if cal['c_c']['ok'] else '**FAIL**'} | " + " | ".join(cells) + " |")
    print("\n### Table C -- I_s (MAIN) and I_z (secondary) by zone, real contexts, K = 1,358 (mean over the zone's states; se in brackets); theory from the run's own P\n")
    print("| run | I_s ud | I_s cn | I_s un | I_s cd | I_z ud | I_z cn | I_z un | I_z cd |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, run in R.items():
        rc, th = run["res"]["real_context"], theory(run)
        s_cells = [f"{rc['I_s_by_zone'][c]['mean']:.3f} [{rc['I_s_by_zone'][c]['se']:.3f}]" for c in CATS]
        z_cells = [f"{rc['I_z_by_zone'][c]['mean']:.3f} [{rc['I_z_by_zone'][c]['se']:.3f}]" for c in CATS]
        print(f"| {k} | " + " | ".join(s_cells + z_cells) + " |")
    ex, rf = theory(R["win64 s0 exp"]), theory(R["win64 s0 ref"])
    print(f"| theory, environment | " + " | ".join(f"{ex[c]:.3f}" for c in CATS) + " | (I_z has no closed form) | | | |")
    print(f"| theory, reference | " + " | ".join(f"{rf[c]:.3f}" for c in CATS) + " | | | | |")
    print("\n### Table D -- T4 (p_hat of the true frequent outcome, mean over the zone's (s, a), se over states) and I_s gap pooled - per context (4 states per zone, 10 contexts, 512 draws; se over states)\n")
    print("| run | T4 cn (true 0.90) | T4 un (true 0.90) | gap ud | gap cn | gap un | gap cd |")
    print("|---|---|---|---|---|---|---|")
    for k, run in R.items():
        t_cn, t_un = t4(run, "controllable_noisy"), t4(run, "uncontrollable_noisy")
        g = run["res"]["is_per_context"]["by_zone"]
        print(f"| {k} | {t_cn['mean']:.3f} [{t_cn['se']:.3f}] | {t_un['mean']:.3f} [{t_un['se']:.3f}] | "
              + " | ".join(f"{g[c]['gap']:+.3f} [{g[c]['gap_se_over_states']:.3f}]" for c in CATS) + " |")


SECTIONS = {"table": sec_table, "calib": sec_calib, "is": sec_is, "traj": sec_traj, "sep": sec_sep, "consistency": sec_consistency, "md": sec_md}

if __name__ == "__main__":
    R = runs()
    print(f"{len(R)} runs: {list(R)}")
    for name in (sys.argv[1:] or SECTIONS):
        SECTIONS[name](R)
