"""
Tables and figure for the stage-3 analyses (coverage-plus-feedback hybrids, batched acquisition with feedback
controls) and for the post hoc sensitivity of stage-2 inference. Numbers-locked: reads only result JSONs.

Inputs : results/stage3_runs.json (script 21), results/stage3_summary.json (script 22, registered Holm p-values and
         MDEs), results/revision_runs.json (script 17; fixed-rule and stage-2 values on the same slides),
         results/stage2_sensitivity.json (script 20).
Outputs: paper/generated/s3_*.tex, paper/generated/s2_sensitivity.tex, paper/fig_s3.pdf, paper/generated/facts_s3.json.

Run after 22:  python -u scripts/23_paper_assets_stage3.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from stare import config, stats

ROOT = Path(__file__).resolve().parents[1]
RES, PAPER = ROOT / "results", ROOT / "paper"
GEN = PAPER / "generated"; GEN.mkdir(parents=True, exist_ok=True)
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b"     # validated categorical slots 1-3
J = lambda n: json.loads((RES / n).read_text(encoding="utf-8"))
pct = lambda b: f"{float(b) * 100:g}\\%"
NAME = {"kmeans": "$k$-means", "mi_greedy": "MI-greedy", "var_greedy": "variance-greedy", "oneshot": "one-shot"}


def num(x, d=3):
    return f"${x:+.{d}f}$".replace("+", "{+}") if round(x, d) != 0 else f"${0:.{d}f}$"


def ci(c, d=3):
    return f"$[{c[0]:+.{d}f}, {c[1]:+.{d}f}]$".replace("+", "{+}")


def pf(p):
    return "--" if p is None or np.isnan(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write(name, lines):
    (GEN / name).write_text("\n".join(lines), encoding="utf-8"); print(f"[write] {GEN / name}")


S3, SUM, R2, SEN = J("stage3_runs.json"), J("stage3_summary.json"), J("revision_runs.json"), J("stage2_sensitivity.json")
L, CL = S3["slides"], np.asarray(S3["grouping"]["clusters"], dtype=object)
assert L == R2["slides"]
REC_RULE, DET_RULE, FRACS = S3["rec_rule"], S3["det_rule"], S3["fracs"]
BUD, DET = list(REC_RULE), list(DET_RULE)
facts = {}


def v(src, b, name, m):
    return np.array([src[l][b][name][m] for l in L], float)


def perm_mean(src, b, base, m):
    names = [n for n in src[L[0]][b] if n.startswith(base + "_perm")]
    return np.mean([v(src, b, n, m) for n in names], axis=0)


def ms(x):
    x = np.asarray(x, float)
    lo, hi = stats.cluster_bootstrap_ci(x, CL, B=config.CI_BOOTSTRAP, alpha=config.CI_ALPHA, seed=config.CI_SEED)
    return {"mean": float(x.mean()), "ci": (lo, hi), "p": float(stats.signflip_p(x, CL))}


def row(fam, start, b):
    for r in SUM[fam]:
        if r["margin"].startswith(start) and r["budget"] == b:
            return r
    raise KeyError((fam, start, b))


H, RB = S3["H"], S3["R"]

# ---------------------------------------------------------------- primary: hybrid - same fixed rule
rows = []
for o, rule, m, buds in (("rec", REC_RULE, "pcc", BUD), ("det", DET_RULE, "recall", DET)):
    if o == "det":
        rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{'Reconstruction (PCC); feedback: residual policy' if o == 'rec' else 'Detection (ROI recall); feedback: local-variance policy'}}}}} \\")
    for b in buds:
        cells = []
        for f in FRACS:
            r = row("primary", f"hybrid {o} f={f} - fixed", b)
            facts[f"prim_{o}_f{f}_{b}"] = r
            cells.append(f"{num(r['mean'])} {ci(r['ci95_unadjusted'])} & {pf(r['p_holm'])} & {r['mde80_family_bound']:.3f}")
        rows.append(f"{pct(b)} & {NAME[rule[b]]} & " + " & ".join(cells) + r" \\")
write("s3_primary.tex", [r"\begin{tabular}{llcccccc}", r"\toprule",
      r" & & \multicolumn{3}{c}{coverage for 50\% of budget} & \multicolumn{3}{c}{coverage for 80\% of budget} \\",
      r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
      r"Budget & fixed rule & $\Delta$ [95\% CI] & $p_H$ & MDE$_H$ & $\Delta$ [95\% CI] & $p_H$ & MDE$_H$ \\", r"\midrule",
      *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- absolute values: fixed, hybrids, pure policy
rows = []
for o, rule, m, buds, pure in (("rec", REC_RULE, "pcc", BUD, "residual_guided"), ("det", DET_RULE, "recall", DET, "value_greedy")):
    if o == "det":
        rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{'Reconstruction (PCC)' if o == 'rec' else 'Detection (ROI recall)'}}}}} \\")
    for b in buds:
        vals = [v(R2["A"], b, rule[b], m).mean()]
        for f in FRACS:
            vals += [v(H, b, f"hyb_{o}_f{f}", m).mean(), perm_mean(H, b, f"hyb_{o}_f{f}", m).mean()]
        vals.append(v(R2["A"], b, pure, m).mean())
        facts[f"abs_{o}_{b}"] = dict(zip(["fixed", "hyb05", "hyb05_perm", "hyb08", "hyb08_perm", "pure"], vals))
        rows.append(f"{pct(b)} & " + " & ".join(f"{x:.3f}" for x in vals) + r" \\")
write("s3_abs.tex", [r"\begin{tabular}{lcccccc}", r"\toprule",
      r" & & \multicolumn{2}{c}{hybrid, 50\% coverage} & \multicolumn{2}{c}{hybrid, 80\% coverage} & \\",
      r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
      r"Budget & fixed rule & observed & permuted & observed & permuted & feedback policy alone \\", r"\midrule",
      *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- secondary family (appendix)
rows = []
for r in SUM["secondary"]:
    n = r["margin"]
    if n.startswith("hybrid"):
        o, f = n.split()[1], n.split("f=")[1].split()[0]
        lab = f"{'reconstruction' if o == 'rec' else 'detection'}, hybrid ({float(f) * 100:g}\\% coverage) $-$ permuted"
    elif "permuted" in n:
        lab = r"rounds of 5: local-variance $-$ permuted"
    else:
        lab = rf"rounds of 5: local-variance $-$ {NAME[n.split('fixed ')[1]]}"
    rows.append(f"{lab} & {pct(r['budget'])} & {num(r['mean'], 4)} {ci(r['ci95_unadjusted'], 4)} & {pf(r['p_holm'])} \\\\")
write("s3_secondary.tex", [r"\begin{tabular}{llcc}", r"\toprule", r"Margin & Budget & $\Delta$ [95\% CI] & $p_H$ \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- batched acquisition (rounds of 5)
rows = []
for b in RB[L[0]]:
    obs, pm = v(RB, b, "batch5_vg", "recall"), perm_mean(RB, b, "batch5_vg", "recall")
    seq, fx = v(R2["A"], b, "value_greedy", "recall"), v(R2["A"], b, DET_RULE[b], "recall")
    a, c = row("secondary", "rounds of 5: local-var - fixed", b), row("secondary", "rounds of 5: local-var - permuted", b)
    facts[f"batch5_{b}"] = {"obs": obs.mean(), "perm": pm.mean(), "seq": seq.mean(), "fixed": fx.mean(), "vs_fixed": a, "vs_perm": c}
    rows.append(f"{pct(b)} & {NAME[DET_RULE[b]]} & {fx.mean():.3f} & {seq.mean():.3f} & {obs.mean():.3f} & {pm.mean():.3f} & "
                f"{num(a['mean'])} {ci(a['ci95_unadjusted'])} & {pf(a['p_holm'])} & {num(c['mean'])} {ci(c['ci95_unadjusted'])} & {pf(c['p_holm'])} \\\\")
write("s3_batched.tex", [r"\begin{tabular}{llcccccccc}", r"\toprule",
      r" & & \multicolumn{4}{c}{ROI recall} & \multicolumn{2}{c}{rounds of 5 $-$ fixed rule} & \multicolumn{2}{c}{rounds of 5 $-$ permuted} \\",
      r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
      r"Budget & fixed rule & fixed & one at a time & rounds of 5 & rounds of 5, perm. & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p_H$ \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- stage-2 inference sensitivity (appendix)
rows = []
for q, lab in (("Q1", "Reconstruction (PCC): residual policy"), ("Q2", "Detection (ROI recall): local-variance policy")):
    if q == "Q2":
        rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{lab}}}}} \\")
    for r in SEN["rows"]:
        if r["q"] != q:
            continue
        mi = r["vs_mi_greedy"]
        facts[f"sens_{q}_{r['budget']}"] = r
        rows.append(f"{pct(r['budget'])} & {num(r['loso_point'])} & {ci(r['loso_ci_selection_aware'])} & "
                    f"{num(mi['mean'])} {ci(mi['ci95'])} & {pf(mi['p'])} & {pf(mi['p_holm18'])} \\\\")
write("s2_sensitivity.tex", [r"\begin{tabular}{lccccc}", r"\toprule",
      r" & \multicolumn{2}{c}{$-$ feasible rule} & \multicolumn{3}{c}{$-$ MI-greedy (fixed)} \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
      r"Budget & $\Delta$ & selection-aware 95\% CI & $\Delta$ [95\% CI] & $p$ & $p_H$ \\", r"\midrule",
      *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- figure: hybrid margins over the fixed rule
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555", "axes.labelcolor": "#222222",
                     "legend.frameon": False, "lines.linewidth": 1.6, "pdf.fonttype": 42})


def series(ax, pts, color, label, marker, off=1.0):
    xs = np.array([p[0] for p in pts]) * off; m = np.array([p[1] for p in pts])
    lo = np.array([p[2][0] for p in pts]); hi = np.array([p[2][1] for p in pts])
    ax.errorbar(xs, m, yerr=[m - lo, hi - m], color=color, marker=marker, ms=4.5, capsize=2, elinewidth=1, label=label)


fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4), constrained_layout=True)
for ax, (o, rule, m, buds, ttl, ylab) in zip(axs, (("rec", REC_RULE, "pcc", BUD, "(a) Reconstruction", r"$\Delta$ PCC vs fixed rule"),
                                                   ("det", DET_RULE, "recall", DET, "(b) ROI detection", r"$\Delta$ ROI recall vs fixed rule"))):
    obs5, obs8, prm5 = [], [], []
    for b in buds:
        x, fx = float(b) * 100, v(R2["A"], b, rule[b], m)
        r5, r8 = row("primary", f"hybrid {o} f=0.5 - fixed", b), row("primary", f"hybrid {o} f=0.8 - fixed", b)
        p5 = ms(perm_mean(H, b, f"hyb_{o}_f0.5", m) - fx)                     # descriptive
        facts[f"perm_minus_fixed_{o}_f0.5_{b}"] = p5
        obs5.append((x, r5["mean"], r5["ci95_unadjusted"])); obs8.append((x, r8["mean"], r8["ci95_unadjusted"]))
        prm5.append((x, p5["mean"], p5["ci"]))
    series(ax, obs5, C1, "50% coverage, observed", "o")
    series(ax, obs8, C3, "80% coverage, observed", "^", 1.07)
    series(ax, prm5, C2, "50% coverage, permuted", "s", 0.93)
    ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xscale("log")
    xt = [float(b) * 100 for b in buds]; ax.set_xticks(xt); ax.set_xticklabels([f"{x:g}" for x in xt])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("budget (% of regions)"); ax.set_ylabel(ylab); ax.set_title(ttl, loc="left", fontsize=8.5)
axs[0].legend(fontsize=6.3, loc="lower right")
fig.savefig(PAPER / "fig_s3.pdf"); fig.savefig(ROOT / "docs" / "figures" / "fig_s3.png", dpi=200)
print(f"[write] {PAPER / 'fig_s3.pdf'}")


def ser(x):
    if isinstance(x, dict):
        return {k: ser(y) for k, y in x.items()}
    if isinstance(x, (tuple, list, np.ndarray)):
        return [ser(y) for y in x]
    return float(x) if isinstance(x, (np.floating, float)) else (int(x) if isinstance(x, np.integer) else x)


(GEN / "facts_s3.json").write_text(json.dumps(ser(facts), indent=1), encoding="utf-8")
print(f"[write] {GEN / 'facts_s3.json'}")
