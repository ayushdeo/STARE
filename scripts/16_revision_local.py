"""
Revision analyses on existing results (reviewer round 2; local, no data needed).

1. Feasible reconstruction comparators (reviewer concern 1). The "best non-adaptive" comparator is the
   per-slide maximum, chosen with hindsight on that slide's outcome. Two comparators that could be chosen
   before profiling a new slide:
     * LOSO: for each held-out study, the non-adaptive policy with the highest mean PCC on the OTHER studies
       (per budget), applied to the held-out study's slides;
     * MI-greedy: the single theory-motivated placement rule [Krause 2008], fixed in advance.
   Applied to error-reduction and maximum-uncertainty (script 05 per-slide values).
2. Direct paired detection comparison (concern 4): value-conditioned minus non-adaptive diversity sampler,
   per slide. Both margins in competitors.json share the same per-slide best-coverage baseline, so their
   difference is exactly value_greedy - scr2st recall.
3. Power (concern 3): minimum detectable effect (MDE) for the study-clustered sign-flip test, by
   resampling studies from the observed per-slide margins re-centred to a hypothetical mean delta.
   Thresholds: alpha 0.05 unadjusted and 0.05/24 (Bonferroni bound for the 24-margin primary family; Holm's
   smallest-p threshold).

Status: these are post-review re-analyses of data already seen, not part of the pre-registered plan.
Writes results/revision_local.json.   Run:  python -u scripts/16_revision_local.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import config, stats

RES = Path(config.RESULTS_ROOT)
NON_ADAPTIVE = ["grid", "kmeans", "oneshot_spatialfinder", "var_greedy", "mi_greedy"]
B = config.CI_BOOTSTRAP


def summ(x, cl):
    x = np.asarray(x, float)
    lo, hi = stats.cluster_bootstrap_ci(x, cl, B=B, alpha=config.CI_ALPHA, seed=config.CI_SEED)
    return {"mean": round(float(x.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p": float(stats.signflip_p(x, cl)), "n_pos": int((x > 0).sum()), "n": int(x.size)}


def power_curve(x, cl, deltas, alphas, reps=400, flips=4000, seed=0):
    """Power of the two-sided cluster sign-flip test when the true mean margin is delta, holding the
    observed between-slide dispersion and study structure. Studies are resampled with replacement."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); cl = np.asarray(cl, dtype=object)
    keys = sorted(set(cl.tolist()), key=str)
    resid = [x[cl == k] - x.mean() for k in keys]
    signs = rng.choice([-1.0, 1.0], size=(flips, len(keys)))
    out = {}
    for d in deltas:
        hits = {a: 0 for a in alphas}
        for _ in range(reps):
            pick = rng.integers(0, len(keys), size=len(keys))
            sums = np.array([(resid[i] + d).sum() for i in pick])
            p = (1 + np.sum(np.abs(signs @ sums) >= abs(sums.sum()) - 1e-12)) / (flips + 1)
            for a in alphas:
                hits[a] += p < a
        out[round(float(d), 4)] = {str(a): hits[a] / reps for a in alphas}
    return out


def mde(curve, alpha, target=0.8):
    for d in sorted(curve):
        if curve[d][str(alpha)] >= target:
            return d
    return None


def main():
    out = {"status": "post-review re-analysis of existing results (not pre-registered)"}
    ci05 = json.loads((RES / "ci_multislide_graphflow.json").read_text(encoding="utf-8"))
    G = ci05["grouping"]; cl = np.asarray(G["clusters"], dtype=object); ps = ci05["per_slide"]

    # 1. feasible comparators ------------------------------------------------------------------
    comp = {}
    for b in ci05["budgets"]:
        P = {p: np.asarray(ps[p][b], float) for p in NON_ADAPTIVE + ["ours_closed_loop", "adaptive_uncertainty"]}
        loso_val, loso_pick = np.zeros(len(cl)), {}
        for s in sorted(set(cl.tolist()), key=str):
            held = cl == s
            choice = max(NON_ADAPTIVE, key=lambda p: P[p][~held].mean())
            loso_val[held] = P[choice][held]; loso_pick[s[:40]] = choice
        best = np.max(np.vstack([P[p] for p in NON_ADAPTIVE]), axis=0)
        comp[b] = {"loso_choice_counts": {p: list(loso_pick.values()).count(p) for p in NON_ADAPTIVE},
                   "comparator_mean": {"per_slide_best": round(float(best.mean()), 4),
                                       "loso": round(float(loso_val.mean()), 4),
                                       "mi_greedy": round(float(P["mi_greedy"].mean()), 4)}}
        for pol, name in [("ours_closed_loop", "error_reduction"), ("adaptive_uncertainty", "max_uncertainty")]:
            comp[b][name] = {"vs_loso": summ(P[pol] - loso_val, cl), "vs_mi_greedy": summ(P[pol] - P["mi_greedy"], cl),
                             "vs_per_slide_best": summ(P[pol] - best, cl)}
        # how much hindsight the per-slide maximum adds over a feasible rule
        comp[b]["hindsight_gap_best_minus_loso"] = summ(best - loso_val, cl)
    out["feasible_comparators_recon"] = comp

    # 2. direct paired detection comparison --------------------------------------------------------
    c10 = json.loads((RES / "competitors.json").read_text(encoding="utf-8"))
    assert c10["grouping"]["clusters"] == G["clusters"]
    direct = {}
    for b, r in c10["rows"].items():
        vg = np.asarray(r["detect_margin_vs_bestcover"]["value_greedy"]["per_slide"], float)
        dv = np.asarray(r["detect_margin_vs_bestcover"]["scr2st"]["per_slide"], float)
        vgr = np.asarray(r["recon_margin_vs_bestcover"]["value_greedy"]["per_slide"], float)
        dvr = np.asarray(r["recon_margin_vs_bestcover"]["scr2st"]["per_slide"], float)
        direct[b] = {"detect_value_minus_diversity": summ(vg - dv, cl),
                     "recon_value_minus_diversity": summ(vgr - dvr, cl)}
    out["direct_value_vs_diversity"] = direct

    # 3. power / MDE ----------------------------------------------------------------------------------
    det = json.loads((RES / "realdata_detection.json").read_text(encoding="utf-8"))
    alphas = [0.05, round(0.05 / 24, 5)]
    pw = {}
    for b in ["0.05", "0.10", "0.20"]:
        x = det["rows"][b]["value_greedy_detect_vs_bestcover"]["per_slide"]
        sd = float(np.std(x, ddof=1))
        deltas = np.round(np.linspace(0, 4 * sd, 41), 4)
        curve = power_curve(x, G["clusters"], deltas, alphas)
        pw[f"detect_{b}"] = {"observed_mean": round(float(np.mean(x)), 4), "slide_sd": round(sd, 4),
                             "mde80_unadjusted": mde(curve, alphas[0]), "mde80_bonferroni24": mde(curve, alphas[1]),
                             "power_at_observed_mean": {str(a): curve[min(curve, key=lambda d: abs(d - np.mean(x)))][str(a)] for a in alphas}}
    for b in ["0.100", "0.200"]:
        x = ci05["margins"][b]["ours_vs_best_nonadapt"]["per_slide"]
        sd = float(np.std(x, ddof=1))
        curve = power_curve(x, G["clusters"], np.round(np.linspace(0, 4 * sd, 41), 4), alphas)
        pw[f"recon_errred_{b}"] = {"observed_mean": round(float(np.mean(x)), 4), "slide_sd": round(sd, 4),
                                   "mde80_unadjusted": mde(curve, alphas[0]), "mde80_bonferroni24": mde(curve, alphas[1])}
    out["power"] = pw

    p = RES / "revision_local.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[write] {p}")
    for b, r in comp.items():
        print(f"budget {b}: comparators {r['comparator_mean']} | LOSO picks {r['loso_choice_counts']}")
        for name in ["error_reduction", "max_uncertainty"]:
            print(f"   {name:16s} vs LOSO {r[name]['vs_loso']['mean']:+.4f} {r[name]['vs_loso']['ci95']} p={r[name]['vs_loso']['p']:.2g} | "
                  f"vs MI {r[name]['vs_mi_greedy']['mean']:+.4f} p={r[name]['vs_mi_greedy']['p']:.2g} | vs best {r[name]['vs_per_slide_best']['mean']:+.4f}")
        print(f"   hindsight gap (best - LOSO) {r['hindsight_gap_best_minus_loso']['mean']:+.4f}")
    for b, r in direct.items():
        d = r["detect_value_minus_diversity"]
        print(f"detect value-diversity {b}: {d['mean']:+.4f} {d['ci95']} p={d['p']:.3g} pos {d['n_pos']}/{d['n']} | recon {r['recon_value_minus_diversity']['mean']:+.4f}")
    for k, v in pw.items():
        print(k, v)


if __name__ == "__main__":
    main()
