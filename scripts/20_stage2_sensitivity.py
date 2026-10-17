"""
Inference checks for the stage-2 feasible comparator (revision round 3, point 2; local, existing data only).

The feasible comparator is chosen per held-out study from the other 17 studies, so the 18 study margins share the
selection step. Two checks:

1. Selection-aware bootstrap. Each bootstrap replicate resamples studies with replacement and REPEATS the
   leave-one-study-out selection inside the replicate (copies of the held-out study are excluded from its own
   selection), so the CI reflects variability in which rule is chosen.
2. Fixed-baseline sensitivity. Every stage-2 primary margin is recomputed against a single rule fixed for all
   slides, for which the study-level sign-flip test is valid as specified: MI-greedy (the theory-motivated placement
   baseline named since stage 1) for both objectives, and k-means as a second fixed rule. Holm over the 18
   analogous margins.

Status: post hoc sensitivity analyses of stage-2 data. Writes results/stage2_sensitivity.json.
Run:  python -u scripts/20_stage2_sensitivity.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import config, stats

RES = Path(config.RESULTS_ROOT)
BUD = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
DET = ["0.020", "0.050", "0.100", "0.200"]


def main():
    R = json.loads((RES / "revision_runs.json").read_text(encoding="utf-8"))
    L, CL, POOL, A = R["slides"], np.asarray(R["grouping"]["clusters"], dtype=object), R["na_pool"], R["A"]
    studies = sorted(set(CL.tolist()), key=str)
    sidx = np.array([studies.index(c) for c in CL])
    col = lambda b, p, m: np.array([A[l][b][p][m] for l in L], float)
    rng = np.random.default_rng(config.CI_SEED)
    BOOT = config.CI_BOOTSTRAP

    def sel_aware_ci(policy_vals, pool_vals):
        """policy_vals (n,), pool_vals (r, n). Returns point estimate (LOSO) and selection-aware 95% CI."""
        k = len(studies)
        P = np.array([policy_vals[sidx == s].sum() for s in range(k)])
        S = np.array([[pv[sidx == s].sum() for s in range(k)] for pv in pool_vals])      # (r, k)
        N = np.array([(sidx == s).sum() for s in range(k)], float)

        def estimate(m):
            tot_S, tot_N = S @ m, N @ m                                                      # (r,), scalar
            other = (tot_S[:, None] - S * m[None, :]) / np.clip(tot_N - N * m, 1e-12, None)[None, :]
            choice = other.argmax(0)                                                         # (k,)
            comp = S[choice, np.arange(k)]
            return float((m * (P - comp)).sum() / (N @ m))
        point = estimate(np.ones(k))
        reps = np.array([estimate(np.bincount(rng.integers(0, k, k), minlength=k).astype(float)) for _ in range(BOOT)])
        lo, hi = np.percentile(reps, [2.5, 97.5])
        return point, (float(lo), float(hi))

    def fixed(x):
        lo, hi = stats.cluster_bootstrap_ci(x, CL, B=BOOT, alpha=0.05, seed=config.CI_SEED)
        return {"mean": float(np.mean(x)), "ci95": [lo, hi], "p": float(stats.signflip_p(x, CL))}

    rows = []
    for b in BUD:
        rg = col(b, "residual_guided", "pcc")
        pool = np.vstack([col(b, p, "pcc") for p in POOL])
        pt, ci = sel_aware_ci(rg, pool)
        rows.append({"q": "Q1", "margin": "residual - comparator (PCC)", "budget": b, "loso_point": pt, "loso_ci_selection_aware": ci,
                     "vs_mi_greedy": fixed(rg - col(b, "mi_greedy", "pcc")), "vs_kmeans": fixed(rg - col(b, "kmeans", "pcc"))})
    for b in DET:
        vg = col(b, "value_greedy", "recall")
        pool = np.vstack([col(b, p, "recall") for p in POOL])
        pt, ci = sel_aware_ci(vg, pool)
        rows.append({"q": "Q2", "margin": "local-variance - comparator (recall)", "budget": b, "loso_point": pt,
                     "loso_ci_selection_aware": ci, "vs_mi_greedy": fixed(vg - col(b, "mi_greedy", "recall")),
                     "vs_kmeans": fixed(vg - col(b, "kmeans", "recall"))})
    # Holm over the 18 analogous stage-2 primary margins with MI-greedy as the fixed comparator: the 10 rows above plus
    # the feedback and diversity margins, which do not involve the comparator and are unchanged
    extra = []
    for b in DET:
        vg = col(b, "value_greedy", "recall")
        extra.append(stats.signflip_p(vg - col(b, "value_greedy_perm", "recall"), CL))
        extra.append(stats.signflip_p(vg - col(b, "scr2st", "recall"), CL))
    pv = [r["vs_mi_greedy"]["p"] for r in rows] + extra
    holm = stats.holm(pv)
    for r, h in zip(rows, holm):
        r["vs_mi_greedy"]["p_holm18"] = h
    out = {"status": "post hoc sensitivity of stage-2 inference (revision round 3, point 2)", "rows": rows}
    (RES / "stage2_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[write] {RES / 'stage2_sensitivity.json'}")
    for r in rows:
        m, k = r["vs_mi_greedy"], r["vs_kmeans"]
        print(f"{r['q']} {r['budget']}: LOSO {r['loso_point']:+.4f} sel-aware CI [{r['loso_ci_selection_aware'][0]:+.4f}, {r['loso_ci_selection_aware'][1]:+.4f}] | "
              f"vs MI {m['mean']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}] p={m['p']:.2g} pH={m['p_holm18']:.2g} | "
              f"vs kmeans {k['mean']:+.4f} p={k['p']:.2g}")


if __name__ == "__main__":
    main()
