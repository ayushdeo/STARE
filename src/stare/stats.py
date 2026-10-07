"""
Cross-slide significance utilities (iteration 3).

The single-slide sweep cannot tell a real +0.003 PCC margin from noise. These pure functions turn
a set of PER-SLIDE paired metrics into an honest summary: the mean margin, a percentile bootstrap
CI over slides, the sign split, and an exact two-sided sign-test p-value. Kept dependency-light
(numpy + stdlib) and unit-tested offline so the CI math is verified without the HEST download.

Small-n caveat baked into the reporting: with ~7 slides a bootstrap CI is wide and the sign test is
weak — that is the honest state of the evidence, not a defect to paper over.

Revision (reviewer point 5): slides from the same study / patient are not independent, and many
margins are tested at once. Passing `clusters` (study per slide) and `strata` (tissue per slide) to
`paired_margin_summary` adds a cluster bootstrap CI, a cluster-level sign-flip (randomization) test,
and tissue-stratified estimates; `holm` adjusts a family of p-values. The slide-level fields are kept
unchanged so earlier results remain comparable.
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


def cluster_bootstrap_ci(x, clusters, B: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI for the pooled slide mean, resampling whole clusters with replacement."""
    x = np.asarray(x, dtype=float)
    keys = sorted(set(clusters))
    idx = [np.flatnonzero(np.asarray(clusters, dtype=object) == k) for k in keys]
    sums = np.array([x[i].sum() for i in idx]); cnts = np.array([len(i) for i in idx], dtype=float)
    k = len(keys)
    if k < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, k, size=(B, k))
    means = sums[draw].sum(1) / cnts[draw].sum(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def signflip_p(x, clusters=None, n_mc: int = 20000, seed: int = 0) -> float:
    """Two-sided randomization test of mean(x) == 0, flipping the sign of each CLUSTER jointly
    (valid when clusters are exchangeable under the null of a symmetric margin). Exact enumeration
    for <= 16 clusters, else Monte Carlo with the +1 correction."""
    x = np.asarray(x, dtype=float)
    clusters = list(range(x.size)) if clusters is None else list(clusters)
    keys = sorted(set(clusters), key=str)
    sums = np.array([x[np.asarray(clusters, dtype=object) == k].sum() for k in keys])
    k = len(keys)
    if k == 0:
        return float("nan")
    obs = abs(sums.sum())
    if k <= 16:
        signs = 1 - 2 * ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1)
        return float(np.mean(np.abs(signs @ sums) >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_mc, k))
    return float((1 + np.sum(np.abs(signs @ sums) >= obs - 1e-12)) / (n_mc + 1))


def holm(pvals) -> list:
    """Holm-Bonferroni adjusted p-values (same order as input; NaNs passed through)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.flatnonzero(~np.isnan(p))
    adj = np.full(p.shape, np.nan)
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[i]))
        adj[i] = running
    return [round(float(v), 4) if not np.isnan(v) else float("nan") for v in adj]


def stratified_summary(x, strata, clusters=None, B: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Per-stratum means and an equal-weight (mean of stratum means) estimate with a CI that
    resamples clusters (or slides) within each stratum."""
    x = np.asarray(x, dtype=float)
    strata = np.asarray(strata, dtype=object)
    clusters = np.asarray(list(range(x.size)) if clusters is None else clusters, dtype=object)
    names = sorted(set(strata.tolist()))
    by = {t: {"n": int((strata == t).sum()), "mean": round(float(x[strata == t].mean()), 4)} for t in names}
    rng = np.random.default_rng(seed)
    boots = np.zeros(B)
    for t in names:
        m = strata == t
        ck = sorted(set(clusters[m].tolist()), key=str)
        sums = np.array([x[m & (clusters == c)].sum() for c in ck])
        cnts = np.array([(m & (clusters == c)).sum() for c in ck], dtype=float)
        draw = rng.integers(0, len(ck), size=(B, len(ck)))
        boots += sums[draw].sum(1) / cnts[draw].sum(1)
    boots /= len(names)
    est = float(np.mean([x[strata == t].mean() for t in names]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"by_stratum": by, "stratified_mean": round(est, 4), "stratified_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "stratified_ci_excludes_zero": bool(lo > 0 or hi < 0)}


def paired_margin_summary(a_per_slide, b_per_slide, B: int = 5000, alpha: float = 0.05,
                          seed: int = 0, clusters=None, strata=None) -> dict:
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
    out = {
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
    if clusters is not None:
        clo, chi = cluster_bootstrap_ci(m, clusters, B=B, alpha=alpha, seed=seed)
        out["n_clusters"] = len(set(clusters))
        out["cluster_ci95"] = [round(clo, 4), round(chi, 4)]
        out["cluster_ci_excludes_zero"] = bool(clo > 0 or chi < 0)
        out["signflip_p"] = round(signflip_p(m, clusters, seed=seed), 4)   # cluster-level; feed to holm()
    if strata is not None:
        out.update(stratified_summary(m, strata, clusters, B=B, alpha=alpha, seed=seed))
    return out
