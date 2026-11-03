"""
Registered analysis of the stage-4 experiments (docs/STAGE4_PLAN.md). Local; reads results/stage4_runs.json
(script 27) and results/revision_runs.json (stage-2 runs, for the tuned-policy comparison of Part P).

Families (Holm within each; exact study-level sign-flip tests; bootstrap and sign-flip-inversion 95% intervals;
minimum detectable effect at 80% power and alpha / family size):
  E primary (12):   residual - feasible (held-out PCC), 6 budgets; reconstruction hybrid - same fixed design
                    (held-out PCC), 6 budgets.
  E secondary (30): feasible - residual in held-out MSE (positive = residual better), 6; residual - residual with
                    constant feedback (PCC), 6; reconstruction hybrid - its constant-feedback version (PCC), 6;
                    local-variance - constant feedback (recall), 4; detection hybrid - fixed design (recall), 4;
                    detection hybrid - constant feedback (recall), 4.
  T primary (9):    at 12, 24, 48 tiles: residual - feasible (held-out tile PCC); local-variance - feasible
                    (tile-ROI recall); local-variance in 2 rounds - feasible (recall).
  T secondary (9):  residual - constant feedback (PCC); local-variance - constant feedback (recall);
                    local-variance in 4 rounds - feasible (recall); each at 3 budgets.
  P primary (4):    tuned local-variance - feasible detection design (recall), 2/5/10/20%; the setting is chosen per
                    held-out study by mean recall on the other studies.
  Descriptive:      local-variance - feasible in held-out PCC (E); recall above random expectation (T);
                    ground-truth greedy - best non-adaptive design in held-out PCC (U).
"Feasible" = the non-adaptive design with the best mean on the other studies, chosen per held-out study.

Writes results/stage4_summary.json.   Run:  python -u scripts/28_stage4_analysis.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import config, stats

RES = Path(config.RESULTS_ROOT)
POOL = ["grid", "kmeans", "var_greedy", "mi_greedy", "oneshot"]
BUD = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
DET = ["0.020", "0.050", "0.100", "0.200"]
TB = ["12", "24", "48"]


def signflip_interval(x, cl, alpha=0.05, tol=1e-6):
    keys = sorted(set(cl), key=str)
    idx = np.array([keys.index(c) for c in cl])
    pat = np.arange(2 ** len(keys))
    signs = (1 - 2 * ((pat[:, None] >> np.arange(len(keys))) & 1)).astype(np.float32)

    def p(d):
        s = np.bincount(idx, weights=x - d, minlength=len(keys))
        return float(np.mean(np.abs(signs @ s) >= abs(s.sum()) - 1e-12))
    if 2.0 / len(pat) > alpha:  # too few studies for any shift to be rejected
        return [None, None]
    m, span, out = x.mean(), max(x.max() - x.min(), 1e-6), []
    for direction in (-1, 1):
        inner, outer = m, m + direction * span
        for _ in range(1000):
            if p(outer) <= alpha:
                break
            outer += direction * span
        else:
            out.append(None)
            continue
        while abs(outer - inner) > tol:
            mid = (inner + outer) / 2
            inner, outer = (mid, outer) if p(mid) > alpha else (inner, mid)
        out.append(inner)
    return out


def mde80(x, cl, alpha, reps=300, flips=3000, seed=0):
    rng = np.random.default_rng(seed)
    cl = np.asarray(cl, dtype=object)
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


def summarize(rows, cl, family):
    out = []
    for name, b, x in rows:
        x = np.asarray(x, float)
        lo, hi = stats.cluster_bootstrap_ci(x, np.asarray(cl, dtype=object), B=config.CI_BOOTSTRAP,
                                            alpha=config.CI_ALPHA, seed=config.CI_SEED)
        out.append({"family": family, "margin": name, "budget": b, "mean": float(x.mean()), "ci95_boot": [lo, hi],
                    "ci95_signflip": signflip_interval(x, list(cl)), "p": float(stats.signflip_p(x, list(cl))),
                    "n_pos": int((x > 0).sum()), "n": int(x.size)})
    for r, h in zip(out, stats.holm([r["p"] for r in out])):
        r["p_holm"] = h
    for r, (_, _, x) in zip(out, rows):
        r["mde80_family"] = mde80(np.asarray(x, float), cl, 0.05 / max(len(rows), 1))
    return out


def feasible(vals, cl):
    """Per held-out study, the pool design with the best mean on the other studies (vals: name -> per-slide array)."""
    cl = np.asarray(cl, dtype=object)
    out = np.zeros(cl.size)
    names = [p for p in POOL if p in vals]
    for s in sorted(set(cl.tolist()), key=str):
        held = cl == s
        out[held] = vals[max(names, key=lambda p: vals[p][~held].mean())][held]
    return out


def main():
    S4 = json.loads((RES / "stage4_runs.json").read_text(encoding="utf-8"))
    L, CL = S4["slides"], list(S4["grouping"]["clusters"])
    summary = {"status": "registered stage-4 analysis (docs/STAGE4_PLAN.md)"}

    def v(part, b, name, m, labels=L):
        return np.array([S4[part][l][b][name][m] for l in labels], float)

    # ---------------------------------------------------------------- Part E
    if "E" in S4 and all(l in S4["E"] for l in L):
        prim, sec, desc = [], [], []
        for b in BUD:
            fe_pcc = feasible({p: v("E", b, p, "pcc") for p in POOL}, CL)
            fe_mse = feasible({p: -v("E", b, p, "mse") for p in POOL}, CL)          # best = lowest MSE
            rg = v("E", b, "residual", "pcc")
            prim.append(("residual - feasible (held-out PCC)", b, rg - fe_pcc))
            prim.append((f"reconstruction hybrid - {S4['rec_rule'][b]} (held-out PCC)", b,
                         v("E", b, "hyb_rec", "pcc") - v("E", b, S4["rec_rule"][b], "pcc")))
            sec.append(("feasible - residual (held-out MSE)", b, -fe_mse - v("E", b, "residual", "mse")))
            sec.append(("residual - constant feedback (PCC)", b, rg - v("E", b, "residual_const", "pcc")))
            sec.append(("reconstruction hybrid - constant feedback (PCC)", b,
                        v("E", b, "hyb_rec", "pcc") - v("E", b, "hyb_rec_const", "pcc")))
            desc.append(("residual - permuted feedback (PCC)", b, rg - v("E", b, "residual_perm", "pcc")))
        for b in DET:
            lv = v("E", b, "local_variance", "recall")
            sec.append(("local-variance - constant feedback (recall)", b, lv - v("E", b, "local_variance_const", "recall")))
            sec.append((f"detection hybrid - {S4['det_rule'][b]} (recall)", b,
                        v("E", b, "hyb_det", "recall") - v("E", b, S4["det_rule"][b], "recall")))
            sec.append(("detection hybrid - constant feedback (recall)", b,
                        v("E", b, "hyb_det", "recall") - v("E", b, "hyb_det_const", "recall")))
            fe_pcc = feasible({p: v("E", b, p, "pcc") for p in POOL}, CL)
            desc.append(("local-variance - feasible (held-out PCC)", b, v("E", b, "local_variance", "pcc") - fe_pcc))
        summary["E"] = {"primary": summarize(prim, CL, "E primary"), "secondary": summarize(sec, CL, "E secondary"),
                        "descriptive": summarize(desc, CL, "E descriptive")}

    # ---------------------------------------------------------------- Part T
    if "T" in S4:
        prim, sec, desc = [], [], []
        for n in TB:
            keep = [l for l in L if n in S4["T"].get(l, {})]
            if len(keep) < len(L):
                summary.setdefault("T_notes", []).append(f"{n} tiles: {len(L) - len(keep)} slide(s) too small, excluded")
            cl = [c for l, c in zip(L, CL) if l in keep]
            g = lambda name, m: v("T", n, name, m, keep)
            fe_pcc = feasible({p: g(p, "pcc") for p in POOL}, cl)
            fe_rec = feasible({p: g(p, "recall") for p in POOL}, cl)
            prim += [(f"residual - feasible (held-out tile PCC)", n, g("residual", "pcc") - fe_pcc),
                     (f"local-variance - feasible (tile-ROI recall)", n, g("local_variance", "recall") - fe_rec),
                     (f"local-variance, 2 rounds - feasible (recall)", n, g("local_variance@r2", "recall") - fe_rec)]
            sec += [(f"residual - constant feedback (PCC)", n, g("residual", "pcc") - g("residual_const", "pcc")),
                    (f"local-variance - constant feedback (recall)", n, g("local_variance", "recall") - g("local_variance_const", "recall")),
                    (f"local-variance, 4 rounds - feasible (recall)", n, g("local_variance@r4", "recall") - fe_rec)]
            n_sel = np.array([S4["T"][l][n]["_n_tiles"] for l in keep], float) * (1 - S4["holdout"])
            desc.append((f"local-variance - random expectation (recall)", n, g("local_variance", "recall") - float(n) / n_sel))
            desc.append((f"feasible - random expectation (recall)", n, fe_rec - float(n) / n_sel))
        if not summary.get("T_notes"):
            summary["T"] = {"primary": summarize(prim, CL, "T primary"), "secondary": summarize(sec, CL, "T secondary"),
                            "descriptive": summarize(desc, CL, "T descriptive")}
        else:
            summary["T"] = {"status": "some slides lacked a tile budget; see T_notes (analysed per budget in a revision)"}

    # ---------------------------------------------------------------- Part P
    if "P" in S4 and all(l in S4["P"] for l in L):
        R2 = json.loads((RES / "revision_runs.json").read_text(encoding="utf-8"))
        assert R2["slides"] == L
        A = R2["A"]
        rows = []
        cl = np.asarray(CL, dtype=object)
        for b in DET:
            configs = sorted(S4["P"][L[0]][b])
            vals = {c: v("P", b, c, "recall") for c in configs}
            tuned = np.zeros(len(L))
            for s in sorted(set(CL), key=str):
                held = cl == s
                best = max(configs, key=lambda c: vals[c][~held].mean())
                tuned[held] = vals[best][held]
            fe = feasible({p: np.array([A[l][b][p]["recall"] for l in L], float) for p in POOL}, CL)
            rows.append(("tuned local-variance - feasible (recall)", b, tuned - fe))
        summary["P"] = {"primary": summarize(rows, CL, "P primary")}

    # ---------------------------------------------------------------- Part U
    if "U" in S4:
        rows = []
        for lab, cells in S4["U"].items():
            for b, cell in cells.items():
                if "greedy_truth" not in cell or b not in S4.get("E", {}).get(lab, {}):
                    continue
                e = S4["E"][lab][b]
                best = max(POOL, key=lambda p: e[p]["pcc"])
                rows.append({"slide": lab, "budget": b, "greedy_truth_pcc": cell["greedy_truth"]["pcc"],
                             "best_design": best, "best_design_pcc": e[best]["pcc"],
                             "residual_pcc": e["residual"]["pcc"],
                             "headroom_over_best_design": cell["greedy_truth"]["pcc"] - e[best]["pcc"]})
        summary["U"] = rows

    (RES / "stage4_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[write] {RES / 'stage4_summary.json'}")
    for part in ("E", "T", "P"):
        for fam, rows in summary.get(part, {}).items():
            if not isinstance(rows, list):
                continue
            for r in rows:
                star = "*" if r["p_holm"] < 0.05 else " "
                print(f"  {part} {fam[:4]} {r['margin'][:46]:46s} {r['budget']:>5} {r['mean']:+.4f} "
                      f"boot {np.round(r['ci95_boot'], 4)} sf {[None if v is None else round(v, 4) for v in r['ci95_signflip']]} pH={r['p_holm']:.3f}{star} "
                      f"MDE={r['mde80_family']}")
    for r in summary.get("U", []):
        print(f"  U {r['slide']} {r['budget']}: greedy {r['greedy_truth_pcc']:.3f} vs {r['best_design']} "
              f"{r['best_design_pcc']:.3f} (headroom {r['headroom_over_best_design']:+.3f}); residual {r['residual_pcc']:.3f}")


if __name__ == "__main__":
    main()
