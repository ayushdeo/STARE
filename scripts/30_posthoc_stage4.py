"""
Post hoc checks on the stage-4 tuned local-variance result (no new experiments on the slides). Status: post hoc.

1. Constant-feedback control for the tuned policy. Under constant feedback the local-variance term should be zero
   for every candidate, leaving beta times the z-scored coverage term, so the selection would not depend on beta or
   the kernel width. Checked here on synthetic slides: this holds at widths 0.1 and 0.2, but not at 0.05, because
   the policy clips the kernel sum at 1e-12 and a candidate far from every profiled region then receives a small
   artificial variance during the first picks. The stage-2 constant run (width 0.1) is therefore an approximate
   control for the tuned settings (width 0.05). If results/stage4_posthoc_runs.json (script 31, GPU machine) is
   present, the exact control of each study's chosen setting is used instead.
2. Tuned policy minus the diversity sampler, a non-adaptive design that was not in the feasible pool.
3. Tuned policy minus a feasible design chosen from the pool with the diversity sampler added.
4. Selection-aware bootstrap: each replicate resamples studies and repeats both leave-one-study-out choices (the
   tuned setting and the feasible design), excluding copies of the held-out study from its own choices.
5. Holm over all 25 stage-4 primary margins instead of within each registered family.

Inputs : results/stage4_runs.json, results/revision_runs.json, results/stage4_summary.json.
Outputs: results/stage4_posthoc.json, paper/generated/s4_posthoc.tex.
Run:     python -u scripts/30_posthoc_stage4.py
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
DET = ["0.020", "0.050", "0.100", "0.200"]
POOL = ["grid", "kmeans", "var_greedy", "mi_greedy", "oneshot"]
DIV = "scr2st"                       # the diversity sampler's key in the stage-2 runs


def invariance_check():
    """Constant-feedback selections of local-variance for all 12 tuning settings on synthetic slides."""
    from stare.simulator import Slide, AcquisitionSimulator
    from stare.imputer import build_imputer
    from stare.policies import build_policy
    from stare.policies.feedback import FeedbackAblation
    same = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        side = 40
        xy = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2) * 100.0
        R, G = xy.shape[0], 60
        lat = rng.normal(size=(R, 8))
        ex = np.maximum(lat @ rng.normal(size=(8, G)) + rng.normal(scale=0.3, size=(R, G)), 0)
        s = Slide(tissue="t", region_coords=xy + rng.uniform(0, 5, xy.shape), image_embeddings=rng.normal(size=(R, 64)),
                  expression=np.log1p(ex), hvg_idx=np.argsort(-ex.var(0))[:50], patient_id=f"p{seed}")
        sel = {}
        for beta in (0.25, 0.5, 1, 2):
            for w in (0.05, 0.1, 0.2):
                sim = AcquisitionSimulator(s, build_imputer("graphflow"), 0.10); st = sim.reset()
                pol = FeedbackAblation(build_policy("value_greedy", beta=beta, lengthscale_frac=w), "constant")
                while not sim.done:
                    st = sim.reveal(pol.select(st))
                sel[(beta, w)] = list(sim.profiled)
        for (beta, w), v in sel.items():
            same.append({"seed": seed, "beta": beta, "width": w, "identical_to_registered": v == sel[(0.5, 0.1)]})
    by_w = {w: all(r["identical_to_registered"] for r in same if r["width"] == w) for w in (0.05, 0.1, 0.2)}
    return {"slides": 3, "settings": 12, "identical_by_width": {str(w): bool(v) for w, v in by_w.items()}, "runs": same}


def main():
    S4, R2, SUM = J("stage4_runs.json"), J("revision_runs.json"), J("stage4_summary.json")
    EXACT = J("stage4_posthoc_runs.json")["P_const"] if (RES / "stage4_posthoc_runs.json").exists() else None
    L, A = S4["slides"], R2["A"]
    assert R2["slides"] == L
    CL = np.asarray(S4["grouping"]["clusters"], dtype=object)
    studies = sorted(set(CL.tolist()), key=str)
    sidx = np.array([studies.index(c) for c in CL])
    k = len(studies)
    N = np.array([(sidx == s).sum() for s in range(k)], float)
    rng = np.random.default_rng(config.CI_SEED)

    def by_study(v):
        return np.array([v[sidx == s].sum() for s in range(k)])

    def loso(vals, m=None):
        """Per study, the option with the best mean on the other studies (study weights m). Returns (k,) choices."""
        m = np.ones(k) if m is None else m
        names = list(vals)
        Ssum = np.array([by_study(vals[n]) for n in names])                       # (r, k)
        tot, totN = Ssum @ m, N @ m
        other = (tot[:, None] - Ssum * m[None, :]) / np.clip(totN - N * m, 1e-12, None)[None, :]
        return [names[i] for i in other.argmax(0)], Ssum

    def margin_loso(pol, comp, m):
        """Slide-weighted mean of (LOSO choice among pol) - (LOSO choice among comp) for study weights m."""
        cp, Sp = loso(pol, m); cc, Sc = loso(comp, m)
        ip, ic = {n: i for i, n in enumerate(pol)}, {n: i for i, n in enumerate(comp)}
        d = np.array([Sp[ip[cp[s]], s] - Sc[ic[cc[s]], s] for s in range(k)])
        return float((m * d).sum() / (N @ m))

    def per_slide(pol, comp):
        cp, _ = loso(pol); cc, _ = loso(comp)
        return np.array([pol[cp[sidx[i]]][i] - comp[cc[sidx[i]]][i] for i in range(len(L))])

    def summarize(name, rows):
        ps = [float(stats.signflip_p(x, list(CL))) for _, x in rows]
        out = []
        for (b, x), p, h in zip(rows, ps, stats.holm(ps)):
            lo, hi = stats.cluster_bootstrap_ci(x, CL, B=config.CI_BOOTSTRAP, alpha=config.CI_ALPHA, seed=config.CI_SEED)
            out.append({"margin": name, "budget": b, "mean": float(x.mean()), "ci95_boot": [lo, hi], "p": p, "p_holm4": h,
                        "n_pos": int((x > 0).sum())})
        return out

    res = {"status": "post hoc (scripts/30_posthoc_stage4.py)", "invariance": invariance_check(),
           "constant_control": "exact (script 31)" if EXACT is not None else "approximate: stage-2 constant run, width 0.1"}
    rows_c, rows_d, rows_f6, selaware, picks6 = [], [], [], [], {}
    for b in DET:
        P = {c: np.array([S4["P"][l][b][c]["recall"] for l in L], float) for c in S4["P"][L[0]][b]}
        D = {p: np.array([A[l][b][p]["recall"] for l in L], float) for p in POOL}
        D6 = dict(D, **{DIV: np.array([A[l][b][DIV]["recall"] for l in L], float)})
        assert np.array_equal(P["local_variance:0.5:0.1"], np.array([A[l][b]["value_greedy"]["recall"] for l in L], float))
        cp, _ = loso(P)
        tuned = np.array([P[cp[sidx[i]]][i] for i in range(len(L))])
        if EXACT is not None:
            const = np.array([EXACT[l][b][cp[sidx[i]] + "_const"]["recall"] for i, l in enumerate(L)], float)
        else:
            const = np.array([A[l][b]["value_greedy_const"]["recall"] for l in L], float)
        rows_c.append((b, tuned - const))
        rows_d.append((b, tuned - D6[DIV]))
        rows_f6.append((b, per_slide(P, D6)))
        c6, _ = loso(D6)
        picks6[b] = {n: c6.count(n) for n in set(c6)}
        for label, comp in (("five-design pool", D), ("with diversity sampler", D6)):
            reps = [margin_loso(P, comp, np.bincount(rng.integers(0, k, k), minlength=k).astype(float))
                    for _ in range(config.CI_BOOTSTRAP)]
            lo, hi = np.percentile(reps, [2.5, 97.5])
            selaware.append({"comparator": label, "budget": b, "point": margin_loso(P, comp, np.ones(k)),
                             "ci95_selection_aware": [float(lo), float(hi)]})
    res["tuned_minus_constant"] = summarize("tuned local-variance - constant feedback", rows_c)
    res["tuned_minus_diversity"] = summarize("tuned local-variance - diversity sampler", rows_d)
    res["tuned_minus_feasible6"] = summarize("tuned local-variance - feasible incl. diversity sampler", rows_f6)
    res["feasible6_picks"] = picks6
    res["selection_aware"] = selaware

    prim = [(part, r) for part in ("E", "T", "P") for r in SUM[part]["primary"]]
    hs = stats.holm([r["p"] for _, r in prim])
    res["holm_all_stage4_primary"] = [{"part": part, "margin": r["margin"], "budget": r["budget"], "p": r["p"],
                                       "p_holm_family": r["p_holm"], "p_holm_25": h} for (part, r), h in zip(prim, hs)]
    (RES / "stage4_posthoc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[write]", RES / "stage4_posthoc.json")

    def num(x):
        return f"${x:+.3f}$".replace("+", "{+}") if round(x, 3) != 0 else "$0.000$"

    def ci(c):
        e = lambda x: "0.000" if round(x, 3) == 0 else f"{x:+.3f}"
        return f"$[{e(c[0])}, {e(c[1])}]$".replace("+", "{+}")

    pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    sa = {(r["comparator"], r["budget"]): r for r in selaware}
    h25 = {r["budget"]: r["p_holm_25"] for r in res["holm_all_stage4_primary"] if r["part"] == "P"}
    out = [r"\begin{tabular}{lcccc}", r"\toprule", r"Comparison & 2\% & 5\% & 10\% & 20\% \\", r"\midrule"]

    def line(title, cells):
        out.append(title + " & " + " & ".join(cells) + r" \\")
    ctitle = "$-$ constant feedback" if EXACT is not None else "$-$ constant feedback (width 0.1, approximate)"
    # the pool with the diversity sampler chose it for every study at every budget, so that row equals tuned - diversity
    assert all(v == {DIV: k} for v in picks6.values())
    for key, title in (("tuned_minus_constant", ctitle), ("tuned_minus_feasible6", "$-$ diversity sampler (feasible pool + diversity)")):
        rs = res[key]
        line(title + r", $\Delta$ [95\% CI]", [f"{num(r['mean'])} {ci(r['ci95_boot'])}" for r in rs])
        line(r"\quad $p_H$ (4 budgets)", [pf(r["p_holm4"]) for r in rs])
    out.append(r"\midrule")
    line(r"$-$ feasible (registered), selection-aware 95\% CI", [ci(sa[("five-design pool", b)]["ci95_selection_aware"]) for b in DET])
    line(r"\quad $p_H$ over all 25 stage-4 primary margins", [pf(h25[b]) for b in DET])
    line(r"$-$ feasible with diversity sampler, selection-aware 95\% CI",
         [ci(sa[("with diversity sampler", b)]["ci95_selection_aware"]) for b in DET])
    (GEN / "s4_posthoc.tex").write_text("\n".join(out + [r"\bottomrule", r"\end{tabular}"]), encoding="utf-8")
    print("[write]", GEN / "s4_posthoc.tex")
    print(json.dumps({"identical_by_width": res["invariance"]["identical_by_width"], "control": res["constant_control"],
                      "picks": res["feasible6_picks"]}))
    for key in ("tuned_minus_constant", "tuned_minus_diversity", "tuned_minus_feasible6"):
        for r in res[key]:
            print(f"{r['margin']:58s} {r['budget']} {r['mean']:+.3f} [{r['ci95_boot'][0]:+.3f}, {r['ci95_boot'][1]:+.3f}] "
                  f"p {r['p']:.4f} pH4 {r['p_holm4']:.4f} n_pos {r['n_pos']}")
    for r in selaware:
        print("sel-aware", r["comparator"], r["budget"], round(r["point"], 4), [round(x, 4) for x in r["ci95_selection_aware"]])
    print("Holm over 25 (P):", {b: round(h, 4) for b, h in h25.items()})


if __name__ == "__main__":
    main()
