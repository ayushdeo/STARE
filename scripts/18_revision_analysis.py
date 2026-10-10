"""
Analysis of the round-2 revision experiments (local; reads results/revision_runs.json from script 17).
The analysis plan is pre-registered in docs/ITERATION_LOG.md (iteration 9).

Comparators
  LOSO_recon(b)  : for each held-out study, the non-adaptive policy (NA pool) with the highest mean PCC on the
                   other studies at budget b, applied to the held-out study's slides  -> feasible comparator
  LOSO_detect(b) : same, selected by mean ROI recall
  best_NA(b)     : per-slide maximum over the NA pool (hindsight benchmark, reported, not primary)

Primary family (18 margins; Holm; exact study-cluster sign-flip p):
  Q1  residual_guided - LOSO_recon            PCC     6 budgets
  Q2  value_greedy - value_greedy_perm        recall  2/5/10/20%   (does informative feedback help?)
      value_greedy - LOSO_detect              recall  2/5/10/20%
      value_greedy - scr2st                   recall  2/5/10/20%   (beyond a non-adaptive diversity design?)
Secondary (Holm within the secondary family): residual_guided - residual_guided_perm (PCC, 6);
  value_greedy - value_greedy_const (recall, 4); residual_guided - best_NA (PCC, 6); value_greedy - LOSO_recon
  (PCC, 4). Part B positive control (descriptive): residual_guided - LOSO_recon and its _perm, by alpha.
Every margin also gets the minimum detectable effect at 80% power (unadjusted and Holm-family bound).

Writes results/revision_summary.json.   Run:  python -u scripts/18_revision_analysis.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import config, stats

RES = Path(config.RESULTS_ROOT)
DET_BUDGETS = ["0.020", "0.050", "0.100", "0.200"]


def summ(x, cl):
    x = np.asarray(x, float)
    lo, hi = stats.cluster_bootstrap_ci(x, cl, B=config.CI_BOOTSTRAP, alpha=config.CI_ALPHA, seed=config.CI_SEED)
    return {"mean": round(float(x.mean()), 4), "ci95_unadjusted": [round(lo, 4), round(hi, 4)],
            "p": float(stats.signflip_p(x, cl)), "n_pos": int((x > 0).sum()), "n": int(x.size)}


def mde80(x, cl, alpha, reps=300, flips=3000, seed=0):
    """Smallest true mean margin with >= 80% power for the two-sided cluster sign-flip test at level alpha,
    keeping the observed dispersion and study structure (studies resampled)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); cl = np.asarray(cl, dtype=object)
    keys = sorted(set(cl.tolist()), key=str)
    resid = [x[cl == k] - x.mean() for k in keys]
    signs = rng.choice([-1.0, 1.0], size=(flips, len(keys)))
    sd = float(np.std(x, ddof=1)) or 1e-6
    for d in np.linspace(0, 5 * sd, 51):
        hit = 0
        for _ in range(reps):
            sums = np.array([(resid[i] + d).sum() for i in rng.integers(0, len(keys), size=len(keys))])
            p = (1 + np.sum(np.abs(signs @ sums) >= abs(sums.sum()) - 1e-12)) / (flips + 1)
            hit += p < alpha
        if hit / reps >= 0.8:
            return round(float(d), 4)
    return None


def table(runs, labels, budget_key, policy, metric):
    return np.array([runs[l][budget_key][policy][metric] for l in labels], float)


def loso(runs, labels, cl, bkey, pool, metric):
    vals = {p: table(runs, labels, bkey, p, metric) for p in pool}
    out, picks = np.zeros(len(labels)), {}
    for s in sorted(set(cl.tolist()), key=str):
        held = cl == s
        choice = max(pool, key=lambda p: vals[p][~held].mean())
        out[held] = vals[choice][held]; picks[choice] = picks.get(choice, 0) + 1
    best = np.max(np.vstack([vals[p] for p in pool]), axis=0)
    return out, best, picks


def main():
    d = json.loads((RES / "revision_runs.json").read_text(encoding="utf-8"))
    labels, cl = d["slides"], np.asarray(d["grouping"]["clusters"], dtype=object)
    pool, A, B = d["na_pool"], d["A"], d["B"]
    budgets = sorted(next(iter(A.values())).keys(), key=float)
    prim, sec, info = [], [], {}

    for b in budgets:
        lr, best, picks = loso(A, labels, cl, b, pool, "pcc")
        info[f"loso_recon_{b}"] = picks
        rg = table(A, labels, b, "residual_guided", "pcc")
        prim.append(("Q1 residual_guided - LOSO_recon (PCC)", b, rg - lr))
        sec.append(("residual_guided - residual_guided_perm (PCC)", b, rg - table(A, labels, b, "residual_guided_perm", "pcc")))
        sec.append(("residual_guided - best_NA hindsight (PCC)", b, rg - best))
        if b in DET_BUDGETS:
            ld, _, dp = loso(A, labels, cl, b, pool, "recall")
            info[f"loso_detect_{b}"] = dp
            vg = table(A, labels, b, "value_greedy", "recall")
            prim.append(("Q2 value_greedy - value_greedy_perm (recall)", b, vg - table(A, labels, b, "value_greedy_perm", "recall")))
            prim.append(("Q2 value_greedy - LOSO_detect (recall)", b, vg - ld))
            prim.append(("Q2 value_greedy - scr2st (recall)", b, vg - table(A, labels, b, "scr2st", "recall")))
            sec.append(("value_greedy - value_greedy_const (recall)", b, vg - table(A, labels, b, "value_greedy_const", "recall")))
            sec.append(("value_greedy - LOSO_recon (PCC)", b, table(A, labels, b, "value_greedy", "pcc") - lr))

    def finish(rows, family):
        out = [{"family": family, "margin": n, "budget": b, **summ(x, cl)} for n, b, x in rows]
        for r, h in zip(out, stats.holm([r["p"] for r in out])):
            r["p_holm"] = h
        a_fam = 0.05 / len(out)
        for r, (_, _, x) in zip(out, rows):
            r["mde80_unadjusted"] = mde80(x, cl, 0.05)
            r["mde80_family_bound"] = mde80(x, cl, a_fam)
        return out

    summary = {"status": "pre-registered round-2 analysis (ITERATION_LOG iteration 9)",
               "primary": finish(prim, "primary"), "secondary": finish(sec, "secondary"), "loso_choices": info}

    pc = []
    if B:
        for alpha in sorted(next(iter(B.values())).keys(), key=float):
            for b in sorted(next(iter(B.values()))[alpha].keys(), key=float):
                cells = {l: B[l][alpha][b] for l in labels}
                lr, _, _ = loso({l: {b: cells[l]} for l in labels}, labels, cl, b, pool, "pcc")
                rg = np.array([cells[l]["residual_guided"]["pcc"] for l in labels])
                rp = np.array([cells[l]["residual_guided_perm"]["pcc"] for l in labels])
                pc.append({"alpha": float(alpha), "budget": b,
                           "spiked_in_top50_mean": float(np.mean([cells[l]["_spiked_in_top50"] for l in labels])),
                           "residual_guided_minus_loso": summ(rg - lr, cl),
                           "residual_guided_perm_minus_loso": summ(rp - lr, cl),
                           "residual_guided_minus_perm": summ(rg - rp, cl)})
    summary["positive_control_B"] = pc

    p = RES / "revision_summary.json"
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[write] {p}")
    for r in summary["primary"]:
        star = "*" if r["p_holm"] < 0.05 else " "
        print(f"  P {r['margin'][:44]:44s} {r['budget']} {r['mean']:+.4f} {r['ci95_unadjusted']} p={r['p']:.2g} "
              f"pH={r['p_holm']:.2g}{star} MDE80={r['mde80_unadjusted']}/{r['mde80_family_bound']}")
    for r in summary["secondary"]:
        print(f"  S {r['margin'][:44]:44s} {r['budget']} {r['mean']:+.4f} {r['ci95_unadjusted']} p={r['p']:.2g} pH={r['p_holm']:.2g}")
    for r in pc:
        print(f"  B alpha={r['alpha']:g} b={r['budget']} spiked={r['spiked_in_top50_mean']:.1f} "
              f"rg-LOSO {r['residual_guided_minus_loso']['mean']:+.4f} {r['residual_guided_minus_loso']['ci95_unadjusted']} | "
              f"perm-LOSO {r['residual_guided_perm_minus_loso']['mean']:+.4f} | rg-perm {r['residual_guided_minus_perm']['mean']:+.4f}")
    print("LOSO choices:", info)


if __name__ == "__main__":
    main()
