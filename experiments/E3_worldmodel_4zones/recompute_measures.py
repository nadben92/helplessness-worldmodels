"""Recomputes every measure of an E3-0 run from its directory alone, without
training again (docs/e3/DECISIONS.md, 2026-09-28): the saved weights
(checkpoints/weights_stepNNNNNNNN.pt at every checkpoint of the grid,
weights_final.pt, or failure/checkpoint.pt), the compact replay
(replay.npz) and the world (world.json: seed, condition, themes; the
pre-rendered images are rebuilt and checked against their saved hash).

The measures are run_mini.measure_final's -- the real-context main measure,
the canonical probe, the s-space kernels, the z-only probe, D1 and D2 --
with the same measurement streams, so the final checkpoint recomputed on
the same device and number of threads gives the run's own numbers
bit for bit (checked and printed when the run has its arrays.npz).

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/recompute_measures.py <run dir> [--checkpoint final|all|failure|<step> ...] [--device cpu|cuda]
        [--kernel-draws K]
Outputs: <run dir>/recomputed/<checkpoint>__<UTC>/ with results.json, arrays.npz, SUMMARY.md and
artifacts/ (plus REPORT.md, the run's report with the recomputed measures, when the run finished).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "recompute_run_mini")

COMPARED_ARRAYS = ("real_I_z", "real_I_z_se", "real_kl_per_state", "real_p_hat", "I_z", "p_hat", "N")


def checkpoint_files(run_dir: Path, which: list[str]) -> list[tuple[str, Path]]:
    ckpt = run_dir / "checkpoints"
    found = []
    for w in which:
        if w == "all":
            found += [(p.stem.replace("weights_", ""), p) for p in sorted(ckpt.glob("weights_step*.pt"))]
            if (ckpt / "weights_final.pt").exists():
                found.append(("final", ckpt / "weights_final.pt"))
        elif w == "final":
            found.append(("final", ckpt / "weights_final.pt"))
        elif w == "failure":
            found.append(("failure", run_dir / "failure" / "checkpoint.pt"))
        else:
            found.append((f"step{int(w):08d}", ckpt / f"weights_step{int(w):08d}.pt"))
    found = list(dict(found).items())  # each checkpoint once, first request order
    for label, path in found:
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path} does not exist")
    return found


def summary_lines(label: str, step: int, fields: dict) -> list[str]:
    rc = fields["real_context"]
    kinds = fields["zone_kinds"]
    lines = [f"# Recomputed measures -- {label} (gradient step {step})", "",
             f"Real contexts: {rc['n_contexts_total']} contexts, {rc['n_states_measured']} / 144 states measured "
             f"(not measured: {rc['not_measured_states'] or 'none'}).", "",
             "| zone | dynamics | I_s real, MAIN (mean [se], bias) | I_z real, secondary (mean [se], n measured) | KL real (bits) | "
             f"L1(WM, P) real | L1 floor (exact kernel, {rc['kernel_draws']} draws) | L1(B, P) | I_z canonical |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cat in fields["zone_categories"]:
        iz, i_s = rc["I_z_by_zone"][cat], rc["I_s_by_zone"][cat]
        lines.append(f"| {cat} | {kinds[cat]} | {i_s['mean']:.4f} [{i_s['se']:.4f}], {i_s['bias']:.4f} | "
                     f"{iz['mean']:.4f} [{iz['se']:.4f}], {iz['n_measured']}/{iz['n_states']} | "
                     f"{rc['kl_by_zone'][cat]:.4f} | {rc['l1_vs_p_by_zone'][cat]:.4f} | {rc['l1_floor_by_zone'][cat]:.4f} | "
                     f"{rc['l1_b_vs_p_by_zone'][cat]:.4f} | "
                     f"{fields['I_by_category'][cat]:.4f} |")
    lines += ["", "Template accuracies: " + ", ".join(f"{k} {v:.4f}" for k, v in fields["accuracies"].items()),
              f"z-only probe test accuracy: {fields['z_only_probe']['test_accuracy']:.4f}; h-only: "
              f"{fields['h_only_probe']['test_accuracy']:.4f}; on the real contexts h -> s {rc['state_probe_h']:.4f}, "
              f"z -> s {rc['state_probe_z']:.4f}",
              "p_hat(frequent outcome), real contexts: " + ", ".join(f"{k} {v:.4f}" for k, v in rc["frequent_outcome"].items()),
              f"D1 (real contexts): {sum(d['accepted'] for d in fields['d1'])} / {len(fields['d1'])} pairs accepted.",
              f"D2 (real contexts): boundary {fields['d2_boundary_mean']:.4f}, within-zone {fields['d2_within_zone_mean']:.4f}."]
    return lines


def recompute(run_dir: Path, which: list[str], device_arg: str | None, kernel_draws: int | None = None) -> list[Path]:
    """kernel_draws: overrides the run's K_s (measure.kernel_k_s and measure.real_context.kernel_draws) --
    e.g. to measure a run made before 2026-09-29 at the corrected K_s (docs/e3/DECISIONS.md); the output
    directory is then labelled __ks<K>, and no bit-for-bit comparison with the run's own arrays is made."""
    run_dir = Path(run_dir)
    cfg = run_mini.load_mini_config(run_dir / "config.yaml")
    if kernel_draws is not None:
        cfg["measure"]["kernel_k_s"] = kernel_draws
        cfg["measure"]["real_context"]["kernel_draws"] = kernel_draws
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    seed, condition = int(meta["seed"]), meta.get("condition", "experimental")
    if device_arg is not None:
        cfg["runtime"]["device"] = device_arg
    device = run_mini.configure_runtime(cfg)
    world, replay, N, _ = run_mini.load_run_data(run_dir, cfg, seed, condition)
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    if device.type == "cuda":
        replay.to_device(device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outputs = []
    for label, path in checkpoint_files(run_dir, which):
        out_dir = run_dir / "recomputed" / f"{label}{'' if kernel_draws is None else f'__ks{kernel_draws}'}__{stamp}"
        (out_dir / "artifacts").mkdir(parents=True)
        log = run_mini.RunLog(out_dir / "console.log", f"[recompute {label}]")
        state = torch.load(path, map_location="cpu", weights_only=False)
        rssm = run_mini.build_rssm(cfg)
        rssm.load_state_dict(state["rssm"])
        rssm.to(device)
        step = int(state["step"])
        log(f"{run_dir.name}: seed {seed} {condition}, weights {path.name} (gradient step {step}), device {device}")
        streams = run_mini.seed_streams(seed, device, condition)
        fields, arrays = run_mini.measure_final(rssm, world, replay, N, cfg, streams, out_dir, log, step)
        fields.pop("diagnostics_final")
        results = {"run_dir": str(run_dir), "checkpoint": label, "weights": path.name, "step": step, "seed": seed,
                   "condition": condition, "device": str(device), "kernel_draws_override": kernel_draws, **fields}
        (out_dir / "results.json").write_text(json.dumps(run_mini._jsonable(
            {k: v for k, v in results.items() if k not in run_mini.ARRAY_FIELDS}), indent=1), encoding="utf-8")
        np.savez_compressed(out_dir / "arrays.npz", **arrays)
        (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines(label, step, fields)) + "\n", encoding="utf-8")
        if label == "final" and (run_dir / "results.json").exists():
            report = run_mini.load_run_report(run_dir)
            report.update(fields)
            report["run_dir"] = str(out_dir)
            run_mini.write_report([report], out_dir, cfg_name=f"{run_dir.name} (recomputed)")
        if label == "final" and kernel_draws is None and (run_dir / "arrays.npz").exists():
            original = np.load(run_dir / "arrays.npz")
            same = {key: bool(np.array_equal(original[key], arrays[key], equal_nan=True)) for key in COMPARED_ARRAYS if key in original}
            log(f"identical to the run's own final arrays: {all(same.values())} {same}")
            (out_dir / "comparison_with_run.json").write_text(json.dumps(same, indent=1), encoding="utf-8")
        log(f"written to {out_dir}")
        outputs.append(out_dir)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", nargs="+", default=["final"], help="final (default), all, failure, or gradient steps")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--kernel-draws", type=int, default=None, help="override K_s (canonical and real-context kernels)")
    args = parser.parse_args(argv)
    recompute(args.run_dir, args.checkpoint, args.device, args.kernel_draws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
