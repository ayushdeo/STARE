"""
Analysis of the stage-3 experiments (registered in docs/ITERATION_LOG.md, iteration 11). Local; reads
results/stage3_runs.json (script 21) and results/revision_runs.json (stage 2: fixed-rule and stage-2 ablation
values on the same slides).

Primary family (20 margins, Holm): hybrid(f) minus the same fixed coverage rule run for the whole budget,
  reconstruction (PCC; f in {0.5, 0.8}; 6 budgets) and detection (recall; f in {0.5, 0.8}; 4 budgets).
  The comparator is one fixed rule per budget, so the study-level sign-flip test is valid as specified.
Secondary family (Holm within): hybrid minus its permuted-feedback version (mean over seeds), both objectives (20);
  rounds of 5 at 10, 20%: local-variance minus the fixed detection rule and minus its permuted version (4).
Descriptive: seed variation of the stage-2 feedback ablations (stage-2 seed 2021 plus the new seeds).
MDEs: unadjusted (descriptive) and at alpha/|family| (conservative bound for the Holm decision rule).

Writes results/stage3_summary.json.   Run:  python -u scripts/22_stage3_analysis.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import config, stats

RES = Path(config.RESULTS_ROOT)


def mde80(x, cl, alpha, reps=300, flips=3000, seed=0):
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
            hit += (1 + np.sum(np.abs(signs @ sums) >= abs(sums.sum()) - 1e-12)) / (flips + 1) < alpha
        if hit / reps >= 0.8:
            return round(float(d), 4)
    return None


def main():
    S3 = json.loads((RES / "stage3_runs.json").read_text(encoding="utf-8"))
    R2 = json.loads((RES / "revision_runs.json").read_text(encoding="utf-8"))
    L, CL = S3["slides"], np.asarray(S3["grouping"]["clusters"], dtype=object)
    assert L == R2["slides"]
    rec_rule, det_rule, fracs = S3["rec_rule"], S3["det_rule"], S3["fracs"]

    def v(src, part, b, name, m):
        if src == "A":
            return np.array([R2["A"][l][b][name][m] for l in L], float)
        return np.array([S3[part][l][b][name][m] for l in L], float)

    def summ(x):
        x = np.asarray(x, float)
        lo, hi = stats.cluster_bootstrap_ci(x, CL, B=config.CI_BOOTSTRAP, alpha=config.CI_ALPHA, seed=config.CI_SEED)
        return {"mean": round(float(x.mean()), 4), "ci95_unadjusted": [round(lo, 4), round(hi, 4)],
                "p": float(stats.signflip_p(x, CL)), "n_pos": int((x > 0).sum()), "n": int(x.size)}

    def perm_mean(part, b, base, m):
        names = [n for n in S3[part][L[0]][b] if n.startswith(base + "_perm")]
        return np.mean([v("S3", part, b, n, m) for n in names], axis=0), names

    prim, sec, desc = [], [], {}
    H = S3.get("H", {})
    if H:
        for b in H[L[0]]:
            for f in fracs:
                obj = [("rec", rec_rule[b], "pcc")] + ([("det", det_rule[b], "recall")] if f"hyb_det_f{f}" in H[L[0]][b] else [])
                for o, rule, m in obj:
                    hyb = v("S3", "H", b, f"hyb_{o}_f{f}", m)
                    prim.append((f"hybrid {o} f={f} - fixed {rule}", b, hyb - v("A", None, b, rule, m)))
                    pm, _ = perm_mean("H", b, f"hyb_{o}_f{f}", m)
                    sec.append((f"hybrid {o} f={f} - permuted (seed mean)", b, hyb - pm))
    R = S3.get("R", {})
    if R:
        for b in R[L[0]]:
            vg = v("S3", "R", b, "batch5_vg", "recall")
            sec.append((f"rounds of {S3['r_round']}: local-var - fixed {det_rule[b]}", b, vg - v("A", None, b, det_rule[b], "recall")))
            pm, _ = perm_mean("R", b, "batch5_vg", "recall")
            sec.append((f"rounds of {S3['r_round']}: local-var - permuted (seed mean)", b, vg - pm))
    S = S3.get("S", {})
    if S:
        for b in S[L[0]]:
            for base, a_name, a_perm, m in [("vg", "value_greedy", "value_greedy_perm", "recall"),
                                            ("rg", "residual_guided", "residual_guided_perm", "pcc")]:
                names = [n for n in S[L[0]][b] if n.startswith(base + "_perm")]
                if not names:
                    continue
                obs = v("A", None, b, a_name, m)
                per_seed = {"2021": obs - v("A", None, b, a_perm, m)}
                per_seed.update({n.split("_perm")[1]: obs - v("S3", "S", b, n, m) for n in names})
                means = {k: round(float(x.mean()), 4) for k, x in per_seed.items()}
                avg = obs - np.mean([v("A", None, b, a_perm, m)] + [v("S3", "S", b, n, m) for n in names], axis=0)
                desc[f"{a_name} - permuted, {b}"] = {"per_seed_mean": means, "across_seed_sd": round(float(np.std(list(means.values()), ddof=1)), 4),
                                                     "seed_averaged": summ(avg)}

    def finish(rows, fam):
        out = [{"family": fam, "margin": n, "budget": b, **summ(x)} for n, b, x in rows]
        for r, h in zip(out, stats.holm([r["p"] for r in out])):
            r["p_holm"] = h
        for r, (_, _, x) in zip(out, rows):
            r["mde80_unadjusted"] = mde80(x, CL, 0.05)
            r["mde80_family_bound"] = mde80(x, CL, 0.05 / max(len(rows), 1))
        return out

    summary = {"status": "stage-3 registered analysis (ITERATION_LOG iteration 11)",
               "primary": finish(prim, "primary"), "secondary": finish(sec, "secondary"), "seed_variation": desc}
    (RES / "stage3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[write] {RES / 'stage3_summary.json'}")
    for fam in ("primary", "secondary"):
        for r in summary[fam]:
            star = "*" if r["p_holm"] < 0.05 else " "
            print(f"  {fam[0].upper()} {r['margin'][:48]:48s} {r['budget']} {r['mean']:+.4f} {r['ci95_unadjusted']} "
                  f"p={r['p']:.2g} pH={r['p_holm']:.2g}{star} MDE={r['mde80_unadjusted']}/{r['mde80_family_bound']}")
    for k, d in desc.items():
        print(f"  D {k}: per seed {d['per_seed_mean']} sd {d['across_seed_sd']} | seed-avg {d['seed_averaged']['mean']:+.4f} "
              f"{d['seed_averaged']['ci95_unadjusted']} p={d['seed_averaged']['p']:.2g}")


if __name__ == "__main__":
    main()
