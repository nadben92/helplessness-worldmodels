"""Small, shared statistics helpers for the Phase-B figures/tables:
95% CI of a mean across seeds, and the paired (intra-seed) comparison
primitive (Wilcoxon signed-rank + effect size) every F2/F3/F5/F6 panel
reuses -- one implementation, never recomputed ad hoc per figure.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) -- a 95% CI of the mean using the
    t-distribution (n=20 seeds, not large enough for a normal
    approximation to be preferred over the exact small-sample t)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n < 2:
        return mean, float("nan"), float("nan")
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    tcrit = float(student_t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def paired_comparison(diffs: np.ndarray) -> dict:
    """diffs: one value per seed, the INTRA-SEED (online - matched)
    difference of some quantity, already averaged over whatever
    state/pair subset the comparison targets (per CLAUDE.md's own review
    note for this task: paired comparisons are always intra-seed, never
    two independent clouds).

    Returns mean, 95% CI of the mean, Wilcoxon signed-rank statistic/p
    (NaN if all diffs are exactly 0 -- wilcoxon() raises on an
    all-zero input), and a paired effect size (Cohen's d for paired
    samples: mean diff / sd of diffs -- standard, not "matched-pairs
    rank-biserial", to stay directly comparable to the mean/CI already
    reported alongside it)."""
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    mean, ci_low, ci_high = mean_ci95(diffs)
    sd = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    cohens_d = (mean / sd) if (n > 1 and sd > 0) else (0.0 if sd == 0 else float("nan"))

    if n == 0 or np.allclose(diffs, 0.0):
        stat, p = float("nan"), float("nan")
    else:
        stat, p = wilcoxon(diffs)
        stat, p = float(stat), float(p)

    return {
        "n": n, "mean": mean, "ci_low": ci_low, "ci_high": ci_high,
        "sd": sd, "cohens_d": cohens_d, "wilcoxon_stat": stat, "wilcoxon_p": p,
    }
