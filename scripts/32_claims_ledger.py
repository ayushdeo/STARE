"""
Ledger of every registered primary margin across the four stages (Appendix O). Status: post hoc summary; no new
experiments. For each group of margins it counts how many had a positive or negative margin with Holm-adjusted p
below 0.05 within its registered family, and, as a sensitivity check, with Benjamini-Hochberg q below 0.05 over all
87 primary margins of the four stages together.

Inputs : results/power_summary.json, results/revision_summary.json, results/stage3_summary.json,
         results/stage4_summary.json.
Outputs: results/claims_ledger.json, paper/generated/claims_ledger.tex.
Run:     python -u scripts/32_claims_ledger.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import stats

ROOT = Path(__file__).resolve().parents[1]
RES, GEN = ROOT / "results", ROOT / "paper" / "generated"
J = lambda n: json.loads((RES / n).read_text(encoding="utf-8"))
ALPHA = 0.05

# the stage-1 primary family, as registered (scripts/14_power_summary.py)
S1_PRIMARY = [("ci_multislide_graphflow", "/ours_vs_best_nonadapt"),
              ("realdata_detection", "/value_greedy_detect_vs_bestcover"),
              ("realdata_detection", "/value_greedy_recon_vs_bestcover"),
              ("batched_contiguous", "")]


def bh(p):
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out.tolist()


def stage1():
    rows = [r for r in J("power_summary.json")["rows"]
            if any(r["experiment"] == e and (s == "" or r["margin"].endswith(s)) for e, s in S1_PRIMARY)]
    assert len(rows) == 24
    ph = stats.holm([r["signflip_p"] for r in rows])
    out = []
    for r, h in zip(rows, ph):
        e, m = r["experiment"], r["margin"]
        if e == "ci_multislide_graphflow":
            g = ("1", "error-reduction $-$ hindsight best design", "PCC")
        elif e == "realdata_detection":
            g = ("1", "local-variance $-$ hindsight coverage design", "recall" if "detect" in m else "PCC")
        else:
            g = ("1", "local-variance $-$ hindsight coverage, batched or contiguous", "recall" if "detect" in m else "PCC")
        out.append({"group": g, "family": "stage 1 primary (24)", "margin": m, "mean": r["mean"],
                    "p": r["signflip_p"], "p_holm": h})
    return out


def stage2():
    out = []
    for r in J("revision_summary.json")["primary"]:
        m = r["margin"]
        if "residual_guided" in m:
            g = ("2", "residual $-$ feasible design", "PCC")
        elif "perm" in m:
            g = ("2", "local-variance $-$ permuted feedback", "recall")
        elif "LOSO" in m:
            g = ("2", "local-variance $-$ feasible design", "recall")
        else:
            g = ("2", "local-variance $-$ diversity sampler", "recall")
        out.append({"group": g, "family": "stage 2 primary (18)", "margin": m, "budget": r["budget"],
                    "mean": r["mean"], "p": r["p"], "p_holm": r["p_holm"]})
    assert len(out) == 18
    return out


def stage3():
    out = []
    for r in J("stage3_summary.json")["primary"]:
        rec = r["margin"].startswith("hybrid rec")
        g = ("3", "hybrid $-$ its base design", "PCC" if rec else "recall")
        out.append({"group": g, "family": "stage 3 primary (20)", "margin": r["margin"], "budget": r["budget"],
                    "mean": r["mean"], "p": r["p"], "p_holm": r["p_holm"]})
    assert len(out) == 20
    return out


def stage4():
    s4, out = J("stage4_summary.json"), []
    for part, fam in (("E", "stage 4 common set (12)"), ("T", "stage 4 tiles (9)"), ("P", "stage 4 tuning (4)")):
        for r in s4[part]["primary"]:
            m = r["margin"]
            if part == "E":
                g = ("4", "residual $-$ feasible, common set" if m.startswith("residual")
                     else "50\\% hybrid $-$ base design, common set", "PCC")
            elif part == "T":
                g = ("4", "residual $-$ feasible, tiles" if m.startswith("residual")
                     else ("local-variance, 2 rounds $-$ feasible, tiles" if "2 rounds" in m
                           else "local-variance $-$ feasible, tiles"), "PCC" if "PCC" in m else "recall")
            else:
                g = ("4", "tuned local-variance $-$ feasible design", "recall")
            out.append({"group": g, "family": fam, "margin": m, "budget": r["budget"], "mean": r["mean"],
                        "p": r["p"], "p_holm": r["p_holm"]})
    assert len(out) == 25
    return out


def main():
    allm = stage1() + stage2() + stage3() + stage4()
    assert len(allm) == 87
    for r, q in zip(allm, bh([r["p"] for r in allm])):
        r["q_bh_all"] = q
    groups = {}
    for r in allm:
        groups.setdefault(r["group"], []).append(r)
    summary = []
    for (stage, comp, obj), rs in groups.items():
        cnt = lambda key, sign: sum(1 for r in rs if r[key] < ALPHA and np.sign(r["mean"]) == sign)
        summary.append({"stage": stage, "comparison": comp, "objective": obj, "n": len(rs),
                        "family": rs[0]["family"],
                        "holm_pos": cnt("p_holm", 1), "holm_neg": cnt("p_holm", -1),
                        "bh_pos": cnt("q_bh_all", 1), "bh_neg": cnt("q_bh_all", -1),
                        "mean_range": [min(r["mean"] for r in rs), max(r["mean"] for r in rs)]})
    (RES / "claims_ledger.json").write_text(json.dumps({"margins": allm, "groups": summary}, indent=1), encoding="utf-8")
    print("[write]", RES / "claims_ledger.json")

    f = lambda x: f"${x:+.3f}$".replace("+", "{+}") if round(x, 3) != 0 else "$0.000$"
    out = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & & & \multicolumn{2}{c}{Holm in family} & \multicolumn{2}{c}{BH over all 87} \\",
           r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"Stage & Comparison (objective) & Margins & Range of $\Delta$ & $+$ & $-$ & $+$ & $-$ \\", r"\midrule"]
    last = None
    for s in summary:
        if last is not None and s["stage"] != last:
            out.append(r"\midrule")
        last = s["stage"]
        out.append(f"{s['stage']} & {s['comparison']} ({s['objective']}) & {s['n']} & "
                   f"{f(s['mean_range'][0])} to {f(s['mean_range'][1])} & {s['holm_pos']} & {s['holm_neg']} & "
                   f"{s['bh_pos']} & {s['bh_neg']} \\\\")
    tot = lambda k: sum(s[k] for s in summary)
    out += [r"\midrule", f" & All primary margins & {len(allm)} & & {tot('holm_pos')} & {tot('holm_neg')} & "
            f"{tot('bh_pos')} & {tot('bh_neg')} \\\\", r"\bottomrule", r"\end{tabular}"]
    (GEN / "claims_ledger.tex").write_text("\n".join(out), encoding="utf-8")
    print("[write]", GEN / "claims_ledger.tex")
    for s in summary:
        print(f"{s['stage']} {s['comparison']:60s} {s['objective']:6s} n={s['n']:2d} "
              f"Holm +{s['holm_pos']} -{s['holm_neg']}  BH +{s['bh_pos']} -{s['bh_neg']}")


if __name__ == "__main__":
    main()
