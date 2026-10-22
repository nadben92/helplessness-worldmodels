"""Why the same-background switch statistic plateaus under shuffle (docs/e5/DECISIONS.md (28)): a DIAGNOSTIC of a run's
latent from its full state (checkpoints/resume.pt: the model and the replay as of that point), in the training regime
(windows drawn as training draws them, unrolled with the posterior PROBABILITIES, the estimator before the switch).
State labels are used here to explain the statistic, never by the training.

For each of the latent's variables (n_vars, n_classes each):
  js_same / js_diff   the mean Jensen-Shannon divergence of its posterior between pairs of items with the SAME background
                      (what the switch statistic sums) and with DIFFERENT backgrounds;
  sharpness           the mean of its posterior's largest probability (1 = a point mass, 1/n_classes = uniform);
  mi_assignment       the mutual information (bits) between its most probable class and the item's assignment;
  mi_state_given_a    the mutual information between its most probable class and the agent's state, within an
                      assignment (averaged over the assignments, weighted by their items).
And the statistic itself: as computed by the training (probabilities, same-background pairs), and "sharpened" (the same
pairs on the one-hot of each posterior's most probable class), the ceiling the run would reach with confident
posteriors on its current classes. Structural ceiling: variables whose class follows the assignment (high
mi_assignment) and not the state contribute nothing within same-background pairs.

Usage: python experiments/E5_belief_training/latent_ceiling.py <run dir> [--batches 200] [--device cpu] [--out <file.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.agents.worldmodel.loop import LatentEstimator, to_chw_float, unroll_observation
from helplessness.cluster.resume import load_for_check
from helplessness.measures.diagnostics import jensen_shannon_categorical
from helplessness.pipeline.checkpoints import build_rssm

N_STATES = 144


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in mutual information (bits) between two discrete label arrays."""
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(joint, (xi, yi), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def pair_js(probs: np.ndarray, groups: np.ndarray, same: bool, rng: np.random.Generator, n_pairs: int) -> np.ndarray:
    """(n_vars,) mean per-variable JS over random pairs of distinct items with the same (or different) group."""
    n = probs.shape[0]
    out, count = np.zeros(probs.shape[1]), 0
    for _ in range(50 * n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j or (groups[i] == groups[j]) != same:
            continue
        out += jensen_shannon_categorical(probs[i], probs[j])
        count += 1
        if count == n_pairs:
            break
    return out / max(count, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    # entirely on the CPU, the pickles inside included (the replay's CUDA copy of its table), with the notebook's older
    # torch / numpy: cluster.resume.load_for_check, as the resume check
    state, replay, _ = load_for_check(args.run / "checkpoints" / "resume.pt")
    device = torch.device(args.device)
    rssm = build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    rssm.to(device).eval()
    if device.type == "cpu":
        replay._device_table = replay._device_background = None  # gather from the host tables
    else:
        replay.to_device(device)
    rows, _ = replay.obs_rows_and_flags()
    tr = cfg["train"]
    rng = np.random.default_rng(0)
    probs_all, assign_all, state_all = [], [], []
    with torch.no_grad():
        for _ in range(args.batches):
            starts = rng.integers(0, len(replay) - tr["seq_len"] + 1, size=tr["batch_size"])
            batch, labels = replay.windows(starts, tr["seq_len"])
            out = unroll_observation(rssm, to_chw_float(batch.x, device), torch.as_tensor(batch.a, device=device),
                                     torch.as_tensor(batch.is_first, device=device), estimator=LatentEstimator("probs"), decode=False)
            probs_all.append(out["post_probs"].reshape(-1, *out["post_probs"].shape[2:]).cpu().numpy())
            r = rows[(starts[:, None] + np.arange(tr["seq_len"])[None, :]).reshape(-1)]
            assign_all.append(r // N_STATES)
            state_all.append(np.asarray(labels.s).reshape(-1))
    probs, assign, s = np.concatenate(probs_all), np.concatenate(assign_all), np.concatenate(state_all)
    onehot = np.eye(probs.shape[-1])[probs.argmax(-1)]
    unimix = float(cfg["model"]["unimix"])
    sharp = (1 - unimix) * onehot + unimix / probs.shape[-1]
    js_same, js_diff, js_same_sharp = [], [], []
    for b in range(args.batches):  # the statistic batch by batch, as training computes it
        sl = slice(b * tr["batch_size"] * tr["seq_len"], (b + 1) * tr["batch_size"] * tr["seq_len"])
        js_same.append(pair_js(probs[sl], assign[sl], True, rng, args.pairs))
        js_diff.append(pair_js(probs[sl], assign[sl], False, rng, args.pairs))
        js_same_sharp.append(pair_js(sharp[sl], assign[sl], True, rng, args.pairs))
    js_same, js_diff, js_same_sharp = (np.mean(x, axis=0) for x in (js_same, js_diff, js_same_sharp))
    cls = probs.argmax(-1)
    per_var = []
    for v in range(probs.shape[1]):
        within = [(np.mean(assign == k), mutual_information(cls[assign == k, v], s[assign == k])) for k in np.unique(assign)]
        per_var.append({"var": v, "js_same": float(js_same[v]), "js_diff": float(js_diff[v]), "js_same_sharpened": float(js_same_sharp[v]),
                        "sharpness": float(probs[:, v].max(-1).mean()), "classes_used": int(len(np.unique(cls[:, v]))),
                        "mi_assignment": mutual_information(cls[:, v], assign),
                        "mi_state_given_a": float(sum(w * m for w, m in within) / sum(w for w, _ in within))})
    summary = {"run": args.run.name, "env_step": int(state["progress"]["env_step"]), "grad_step": int(state["progress"]["grad_step"]),
               "switched": state["switch"]["switch_step"] is not None, "items": int(probs.shape[0]), "assignments_seen": int(len(np.unique(assign))),
               "statistic_same_background": float(js_same.sum()), "statistic_different_background": float(js_diff.sum()),
               "statistic_same_background_sharpened": float(js_same_sharp.sum()), "per_var": per_var}
    text = json.dumps(summary, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
