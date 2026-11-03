"""
Post hoc robustness checks on existing results (no new runs). Status: post hoc, descriptive.

1. Random baseline for detection. Under random selection of n = round(b N) of N regions, the expected recall of
   any fixed ROI set is exactly n / N, whatever the ROI fraction. Margins of the stage-2 policies over this
   expectation, per slide.
2. Interval sensitivity. Percentile intervals that resample few clusters can undercover (Cameron et al., 2008).
   For the registered primary margins of stages 2 and 3, the 95% interval obtained by inverting the exact study-level
   sign-flip test is reported next to the registered percentile bootstrap interval.
3. Development slide excluded. The imputer's hyperparameters were set on TENX174, one of the 22 slides. The stage-2
   and stage-3 primary margins are recomputed without it (21 slides, 17 studies), with the feasible design
   re-selected on the remaining studies and Holm within each family.

Inputs: results/revision_runs.json, results/stage3_runs.json, results/scale_audit.json (regions per slide).
Outputs: results/posthoc_robustness.json, paper/generated/robust_{random,intervals,devslide}.tex.
Run:  python -u scripts/26_posthoc_robustness.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import config, stats

ROOT = Path(__file__).resolve().parents[1]
RES, GEN = ROOT / "results", ROOT / "paper" / "generated"
J = lambda n: json.loads((RES / n).read_text(encoding="utf-8"))
pct = lambda b: f"{float(b) * 100:g}\\%"
BUD = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
DET = ["0.020", "0.050", "0.100", "0.200"]
DEV = "prostate:TENX174"
NAME = {"kmeans": "$k$-means", "mi_greedy": "MI-greedy", "var_greedy": "variance-greedy", "oneshot": "one-shot"}


def num(x, d=3):
    return f"${x:+.{d}f}$".replace("+", "{+}") if round(x, d) != 0 else f"${0:.{d}f}$"


def ci(c, d=3):
    return f"$[{c[0]:+.{d}f}, {c[1]:+.{d}f}]$".replace("+", "{+}")


def pf(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


class SignFlip:
    """Exact study-level sign-flip test and its inversion for one cluster structure."""

    def __init__(self, clusters):
        self.keys = sorted(set(clusters), key=str)
        self.idx = np.array([self.keys.index(c) for c in clusters])
        k = len(self.keys)
        pat = np.arange(2 ** k)
        self.signs = (1 - 2 * ((pat[:, None] >> np.arange(k)) & 1)).astype(np.float32)
        self.n = np.bincount(self.idx, minlength=k).astype(float)

    def p(self, x, delta=0.0):
        s = np.bincount(self.idx, weights=np.asarray(x, float) - delta, minlength=len(self.keys))
        return float(np.mean(np.abs(self.signs @ s) >= abs(s.sum()) - 1e-12))

    def interval(self, x, alpha=0.05, tol=1e-6):
        """{delta : p(x - delta) > alpha}, found by bisection outward from the mean on each side."""
        x = np.asarray(x, float)
        m, span = x.mean(), max(x.max() - x.min(), 1e-6)
        out = []
        for direction in (-1, 1):
            inner, outer = m, m + direction * span
            while self.p(x, outer) > alpha:
                outer += direction * span
            while abs(outer - inner) > tol:
                mid = (inner + outer) / 2
                if self.p(x, mid) > alpha:
                    inner = mid
                else:
                    outer = mid
            out.append(inner)
        return out[0], out[1]


def boot(x, cl):
    return stats.cluster_bootstrap_ci(np.asarray(x, float), np.asarray(cl, dtype=object), B=config.CI_BOOTSTRAP,
                                      alpha=config.CI_ALPHA, seed=config.CI_SEED)


def main():
    R2, S3 = J("revision_runs.json"), J("stage3_runs.json")
    L, CL = R2["slides"], list(R2["grouping"]["clusters"])
    assert S3["slides"] == L
    A, POOL, H = R2["A"], R2["na_pool"], S3["H"]
    audit = J("scale_audit.json")["rows"]
    N = np.array([audit[l.split(":")[1]]["n_spots_in_tissue"] for l in L], float)

    def col(b, p, m, keep):
        return np.array([A[l][b][p][m] for l, k in zip(L, keep) if k], float)

    def feasible(b, m, keep):
        """Leave-one-study-out choice among the non-adaptive pool, restricted to the kept slides."""
        cl = np.array([c for c, k in zip(CL, keep) if k], dtype=object)
        vals = {p: col(b, p, m, keep) for p in POOL}
        out = np.zeros(cl.size)
        for s in sorted(set(cl.tolist()), key=str):
            held = cl == s
            out[held] = vals[max(POOL, key=lambda p: vals[p][~held].mean())][held]
        return out

    def primary(keep):
        """Registered primary margins of stage 2 (18) and stage 3 (20) on the kept slides."""
        s2, s3 = [], []
        for b in BUD:
            s2.append(("RQ1: residual - feasible (PCC)", b, col(b, "residual_guided", "pcc", keep) - feasible(b, "pcc", keep)))
        for b in DET:
            lv = col(b, "value_greedy", "recall", keep)
            s2.append(("RQ2: local-variance - permuted (recall)", b, lv - col(b, "value_greedy_perm", "recall", keep)))
            s2.append(("RQ2: local-variance - feasible (recall)", b, lv - feasible(b, "recall", keep)))
            s2.append(("RQ2: local-variance - diversity (recall)", b, lv - col(b, "scr2st", "recall", keep)))
        for b in BUD:
            for f in S3["fracs"]:
                hyb = np.array([H[l][b][f"hyb_rec_f{f}"]["pcc"] for l, k in zip(L, keep) if k])
                s3.append((f"RQ3: reconstruction hybrid ({f * 100:g}% coverage) - {NAME[S3['rec_rule'][b]]}", b,
                           hyb - col(b, S3["rec_rule"][b], "pcc", keep)))
        for b in DET:
            for f in S3["fracs"]:
                hyb = np.array([H[l][b][f"hyb_det_f{f}"]["recall"] for l, k in zip(L, keep) if k])
                s3.append((f"RQ3: detection hybrid ({f * 100:g}% coverage) - {NAME[S3['det_rule'][b]]}", b,
                           hyb - col(b, S3["det_rule"][b], "recall", keep)))
        return {"stage 2": s2, "stage 3": s3}

    out = {"status": "post hoc robustness checks of existing results (descriptive)"}
    full = np.ones(len(L), bool)
    sf = SignFlip(CL)

    # 1. random baseline -----------------------------------------------------------------------------------------
    rows, rand = [], {}
    for b in DET:
        n = np.round(float(b) * N)
        exp_rand = n / N
        rand[b] = {"expected_random_recall": float(exp_rand.mean())}
        for p, lab in (("value_greedy", "local-variance"), ("scr2st", "diversity sampler"), ("mi_greedy", "MI-greedy")):
            x = col(b, p, "recall", full) - exp_rand
            rand[b][p] = {"mean": float(x.mean()), "ci95_boot": boot(x, CL), "ci95_signflip": sf.interval(x),
                          "p": sf.p(x)}
        fe = feasible(b, "recall", full) - exp_rand
        rand[b]["feasible"] = {"mean": float(fe.mean()), "ci95_boot": boot(fe, CL), "ci95_signflip": sf.interval(fe),
                               "p": sf.p(fe)}
        r = rand[b]
        rows.append(f"{pct(b)} & {r['expected_random_recall']:.3f} & " + " & ".join(
            f"{num(r[k]['mean'])} {ci(r[k]['ci95_signflip'])}" for k in ("value_greedy", "feasible", "scr2st", "mi_greedy")) + r" \\")
    out["random_baseline"] = rand
    (GEN / "robust_random.tex").write_text("\n".join([
        r"\begin{tabular}{lccccc}", r"\toprule",
        r" & expected recall & \multicolumn{4}{c}{recall minus expected random recall, $\Delta$ [95\% CI]} \\",
        r"\cmidrule(lr){3-6}",
        r"Budget & of random selection & local-variance & feasible design & diversity sampler & MI-greedy \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]), encoding="utf-8")

    # 2. interval sensitivity --------------------------------------------------------------------------------------
    rows, inter = [], []
    for fam, margins in primary(full).items():
        rows.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{fam.capitalize()} primary family}}}} \\")
        for name, b, x in margins:
            bci, sci = boot(x, CL), sf.interval(x)
            inter.append({"family": fam, "margin": name, "budget": b, "mean": float(x.mean()), "ci95_boot": bci,
                          "ci95_signflip": sci, "p": sf.p(x),
                          "excludes_zero_agree": bool((bci[0] > 0 or bci[1] < 0) == (sci[0] > 0 or sci[1] < 0))})
            rows.append(f"{name.split(': ', 1)[1].replace(' - ', ' $-$ ').replace('%', chr(92) + '%')} & {pct(b)} & "
                        f"{num(x.mean())} {ci(bci)} & {ci(sci)} \\\\")
    out["intervals"] = inter
    (GEN / "robust_intervals.tex").write_text("\n".join([
        r"\begin{tabular}{llcc}", r"\toprule",
        r"Margin & Budget & $\Delta$ [registered bootstrap 95\% CI] & sign-flip inversion 95\% CI \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]), encoding="utf-8")

    # 3. development slide excluded ---------------------------------------------------------------------------------
    keep = np.array([l != DEV for l in L])
    cl_k = [c for c, k in zip(CL, keep) if k]
    sf_k = SignFlip(cl_k)
    rows, dev = [], []
    full_m = primary(full)
    for fam, margins in primary(keep).items():
        ps = [sf_k.p(x) for _, _, x in margins]
        ph = stats.holm(ps)
        ph_full = stats.holm([sf.p(x) for _, _, x in full_m[fam]])
        rows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{fam.capitalize()} primary family}}}} \\")
        for (name, b, x), p, h, (_, _, xf), hf in zip(margins, ps, ph, full_m[fam], ph_full):
            sig_full, sig_k = hf < 0.05 and np.sign(xf.mean()), h < 0.05 and np.sign(x.mean())
            dev.append({"family": fam, "margin": name, "budget": b, "mean": float(x.mean()), "ci95_boot": boot(x, cl_k),
                        "p": p, "p_holm": h, "mean_all22": float(xf.mean()), "p_holm_all22": hf,
                        "conclusion_changes": bool(sig_full != sig_k)})
            rows.append(f"{name.split(': ', 1)[1].replace(' - ', ' $-$ ').replace('%', chr(92) + '%')} & {pct(b)} & "
                        f"{num(xf.mean())} & {num(x.mean())} {ci(boot(x, cl_k))} & {pf(h)} \\\\")
    out["development_slide_excluded"] = {"excluded": DEV, "n_slides": int(keep.sum()), "n_studies": len(set(cl_k)),
                                         "rows": dev, "any_conclusion_changes": any(r["conclusion_changes"] for r in dev)}
    (GEN / "robust_devslide.tex").write_text("\n".join([
        r"\begin{tabular}{llccc}", r"\toprule",
        r" & & all 22 slides & \multicolumn{2}{c}{without TENX174 (21 slides, 17 studies)} \\", r"\cmidrule(lr){4-5}",
        r"Margin & Budget & $\Delta$ & $\Delta$ [95\% CI] & $p_H$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]), encoding="utf-8")

    (RES / "posthoc_robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[write] {RES / 'posthoc_robustness.json'} and paper/generated/robust_*.tex")
    for b in DET:
        r = rand[b]
        print(f"  random {b}: E[recall]={r['expected_random_recall']:.3f}  LV-rand {r['value_greedy']['mean']:+.3f} "
              f"{np.round(r['value_greedy']['ci95_signflip'], 3)}  feasible-rand {r['feasible']['mean']:+.3f}  "
              f"MI-rand {r['mi_greedy']['mean']:+.3f} p={r['mi_greedy']['p']:.3f}")
    dis = [r for r in inter if not r["excludes_zero_agree"]]
    print(f"  intervals: {len(inter)} margins; bootstrap and sign-flip disagree on excluding zero for {len(dis)}")
    for r in dis:
        print("    ", r["margin"], r["budget"], np.round(r["ci95_boot"], 4), np.round(r["ci95_signflip"], 4))
    ch = [r for r in dev if r["conclusion_changes"]]
    print(f"  without TENX174: conclusions change for {len(ch)} margins")
    for r in ch:
        print("    ", r["margin"], r["budget"], f"all22 {r['mean_all22']:+.4f} pH {r['p_holm_all22']:.3f} -> {r['mean']:+.4f} pH {r['p_holm']:.3f}")


if __name__ == "__main__":
    main()
