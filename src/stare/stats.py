"""
Cross-slide significance utilities (iteration 3).

The single-slide sweep cannot tell a real +0.003 PCC margin from noise. These pure functions turn
a set of PER-SLIDE paired metrics into an honest summary: the mean margin, a percentile bootstrap
CI over slides, the sign split, and an exact two-sided sign-test p-value. Kept dependency-light
(numpy + stdlib) and unit-tested offline so the CI math is verified without the HEST download.

Small-n caveat baked into the reporting: with ~7 slides a bootstrap CI is wide and the sign test is
weak — that is the honest state of the evidence, not a defect to paper over.
"""
from __future__ import annotations
from math import comb
import numpy as np


def bootstrap_ci(x: np.ndarray, B: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the MEAN of a 1-D sample (resample slides with replacement)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _sign_test_p(n_pos: int, n_effective: int) -> float:
    """Exact two-sided sign test p-value: P(|Binom(n,0.5) - n/2| >= |n_pos - n/2|). Zeros dropped."""
    n = n_effective
    if n == 0:
        return float("nan")
    k = n_pos
    dev = abs(k - n / 2)
    tail = sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= dev)
    return float(min(1.0, tail / (2 ** n)))


def paired_margin_summary(a_per_slide, b_per_slide, B: int = 5000, alpha: float = 0.05,
                          seed: int = 0) -> dict:
    """Summarize the paired margin (a - b) across slides. a,b are aligned per-slide metric arrays
    (e.g. ours vs mi_greedy pcc_top50, one value per slide). Returns mean, bootstrap CI, sign split,
    exact sign-test p, and whether the CI excludes 0 (the honest 'is it real?' flag)."""
    a = np.asarray(a_per_slide, dtype=float)
    b = np.asarray(b_per_slide, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    m = a - b
    n = m.size
    n_pos = int((m > 0).sum())
    n_neg = int((m < 0).sum())
    lo, hi = bootstrap_ci(m, B=B, alpha=alpha, seed=seed)
    return {
        "n_slides": n,
        "per_slide": [round(float(v), 4) for v in m],
        "mean": round(float(m.mean()), 4) if n else float("nan"),
        "median": round(float(np.median(m)), 4) if n else float("nan"),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci_excludes_zero": bool(n > 1 and (lo > 0 or hi < 0)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sign_p": round(_sign_test_p(n_pos, n_pos + n_neg), 4),
    }
