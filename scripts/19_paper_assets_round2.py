"""
Tables and the main figure for the round-2 analyses (feasible comparator, reconstruction-aligned value-conditioned
policy, feedback ablations, semi-synthetic positive control). Numbers-locked: reads only result JSONs.

Inputs : results/revision_runs.json (script 17), results/revision_summary.json (script 18, pre-registered
         Holm p-values and MDEs), results/ci_multislide_graphflow.json (script 05; sequential fixed designs),
         results/competitors.json (script 10; ground-truth detection ordering).
Outputs: paper/generated/r2_*.tex, paper/fig_r2.pdf, paper/generated/facts_r2.json.

Run after 18:  python -u scripts/19_paper_assets_round2.py
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


def num(x, d=3):
    return f"${x:+.{d}f}$".replace("+", "{+}") if round(x, d) != 0 else f"${0:.{d}f}$"


def ci(c, d=3):
    return f"$[{c[0]:+.{d}f}, {c[1]:+.{d}f}]$".replace("+", "{+}")


def pf(p):
    return "--" if p is None or np.isnan(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write(name, lines):
    (GEN / name).write_text("\n".join(lines), encoding="utf-8"); print(f"[write] {GEN / name}")


R = J("revision_runs.json"); S = J("revision_summary.json"); C05 = J("ci_multislide_graphflow.json"); C10 = J("competitors.json")
L, CL = R["slides"], np.asarray(R["grouping"]["clusters"], dtype=object)
assert C05["slides"] == L and C10["slides"] == L
POOL, A, B = R["na_pool"], R["A"], R["B"]
BUD = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
DET = ["0.020", "0.050", "0.100", "0.200"]
B10 = {"0.020": "0.02", "0.050": "0.05", "0.100": "0.10", "0.200": "0.20"}
facts = {}


def col(b, p, m):
    return np.array([A[l][b][p][m] for l in L], float)


def loso(b, metric):
    vals = {p: col(b, p, metric) for p in POOL}
    out = np.zeros(len(L))
    for s in sorted(set(CL.tolist()), key=str):
        held = CL == s
        out[held] = vals[max(POOL, key=lambda p: vals[p][~held].mean())][held]
    return out, np.max(np.vstack([vals[p] for p in POOL]), axis=0)


def ms(x):
    x = np.asarray(x, float)
    lo, hi = stats.cluster_bootstrap_ci(x, CL, B=config.CI_BOOTSTRAP, alpha=config.CI_ALPHA, seed=config.CI_SEED)
    return {"mean": float(x.mean()), "ci": (lo, hi), "p": float(stats.signflip_p(x, CL)), "n_pos": int((x > 0).sum())}


def pre(margin, b, fam="primary"):
    for r in S[fam]:
        if r["margin"] == margin and r["budget"] == b:
            return r
    raise KeyError((margin, b))


ER = {b: np.asarray(C05["per_slide"]["ours_closed_loop"][b], float) for b in BUD}
MU = {b: np.asarray(C05["per_slide"]["adaptive_uncertainty"][b], float) for b in BUD}

# ---------------------------------------------------------------- Q1 absolute PCC
rows = []
for b in BUD:
    fe, best = loso(b, "pcc")
    vals = [fe.mean(), best.mean(), ER[b].mean(), MU[b].mean(), col(b, "residual_guided", "pcc").mean(),
            col(b, "residual_guided_perm", "pcc").mean(), col(b, "value_greedy", "pcc").mean()]
    facts[f"q1abs_{b}"] = dict(zip(["feasible", "hindsight", "errred", "maxunc", "rg", "rg_perm", "localvar"], vals))
    rows.append(f"{pct(b)} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
write("r2_q1_abs.tex", [r"\begin{tabular}{lccccccc}", r"\toprule",
      r" & \multicolumn{2}{c}{Non-adaptive} & \multicolumn{2}{c}{Sequential fixed design} & \multicolumn{3}{c}{Value-conditioned} \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
      r"Budget & feasible & per-slide best & error-red. & max-unc. & residual & residual, perm. & local-var. \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- Q1 margins
rows, fig_a = [], {"rg": [], "er": [], "lv": []}
for b in BUD:
    fe, best = loso(b, "pcc")
    p1 = pre("Q1 residual_guided - LOSO_recon (PCC)", b)
    p2 = pre("residual_guided - residual_guided_perm (PCC)", b, "secondary")
    er, mu = ms(ER[b] - fe), ms(MU[b] - fe)
    lv = ms(col(b, "value_greedy", "pcc") - fe)
    facts[f"q1_{b}"] = {"rg": p1, "rg_perm": p2, "er": er, "mu": mu, "lv": lv}
    fig_a["rg"].append((float(b), p1["mean"], p1["ci95_unadjusted"])); fig_a["er"].append((float(b), er["mean"], er["ci"]))
    if b not in ("0.005", "0.010"):
        fig_a["lv"].append((float(b), lv["mean"], lv["ci"]))
    rows.append(f"{pct(b)} & {num(p1['mean'])} {ci(p1['ci95_unadjusted'])} & {pf(p1['p_holm'])} & "
                f"{num(p2['mean'])} {ci(p2['ci95_unadjusted'])} & {pf(p2['p_holm'])} & "
                f"{num(er['mean'])} {ci(er['ci'])} & {pf(er['p'])} \\\\")
write("r2_q1_margins.tex", [r"\begin{tabular}{lcccccc}", r"\toprule",
      r" & \multicolumn{2}{c}{residual $-$ feasible} & \multicolumn{2}{c}{residual $-$ residual, perm.} & \multicolumn{2}{c}{error-red. $-$ feasible} \\",
      r" & \multicolumn{2}{c}{(primary)} & \multicolumn{2}{c}{(secondary)} & \multicolumn{2}{c}{(post hoc)} \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
      r"Budget & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p$ \\", r"\midrule",
      *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- Q2 margins
rows, fig_b = [], {"perm": [], "div": [], "fe": []}
for b in DET:
    a = pre("Q2 value_greedy - value_greedy_perm (recall)", b)
    f = pre("Q2 value_greedy - LOSO_detect (recall)", b)
    d = pre("Q2 value_greedy - scr2st (recall)", b)
    facts[f"q2_{b}"] = {"perm": a, "feasible": f, "diversity": d,
                        "const": pre("value_greedy - value_greedy_const (recall)", b, "secondary")}
    for k, r in (("perm", a), ("div", d), ("fe", f)):
        fig_b[k].append((float(b), r["mean"], r["ci95_unadjusted"]))
    rows.append(f"{pct(b)} & {num(a['mean'])} {ci(a['ci95_unadjusted'])} & {pf(a['p_holm'])} & {a['mde80_family_bound']:.3f} & "
                f"{num(f['mean'])} {ci(f['ci95_unadjusted'])} & {pf(f['p_holm'])} & "
                f"{num(d['mean'])} {ci(d['ci95_unadjusted'])} & {pf(d['p_holm'])} \\\\")
write("r2_q2_margins.tex", [r"\begin{tabular}{lcccccccc}", r"\toprule",
      r" & \multicolumn{3}{c}{local-var. $-$ permuted feedback} & \multicolumn{2}{c}{local-var. $-$ feasible} & \multicolumn{2}{c}{local-var. $-$ diversity} \\",
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
      r"Budget & $\Delta$ [95\% CI] & $p_H$ & MDE$_H$ & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p_H$ \\", r"\midrule",
      *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- Q2 absolute recall
rows = []
for b in DET:
    fe, _ = loso(b, "recall")
    vals = [fe.mean(), col(b, "kmeans", "recall").mean(), col(b, "mi_greedy", "recall").mean(),
            col(b, "scr2st", "recall").mean(), col(b, "value_greedy", "recall").mean(),
            col(b, "value_greedy_perm", "recall").mean(), col(b, "value_greedy_const", "recall").mean(),
            C10["rows"][B10[b]]["detection_recall_means"]["oracle"]]
    facts[f"q2abs_{b}"] = dict(zip(["feasible", "kmeans", "mi", "diversity", "value", "perm", "const", "oracle"], vals))
    rows.append(f"{pct(b)} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
write("r2_q2_abs.tex", [r"\begin{tabular}{lcccccccc}", r"\toprule",
      r" & \multicolumn{4}{c}{Non-adaptive} & \multicolumn{3}{c}{Value-conditioned (local variance)} & \\",
      r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
      r"Budget & feasible & $k$-means & MI-greedy & diversity & observed & permuted & constant & oracle \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- positive control (Part B)
rows, fig_c = [], {"0.050": [], "0.100": []}
for r in S["positive_control_B"]:
    a, b = r["alpha"], r["budget"]
    x, y, z = r["residual_guided_minus_loso"], r["residual_guided_perm_minus_loso"], r["residual_guided_minus_perm"]
    facts[f"spike_{a:g}_{b}"] = r
    fig_c[b].append((a, x["mean"], x["ci95_unadjusted"]))
    rows.append(f"{a:g} & {pct(b)} & {int(round(r['spiked_in_top50_mean']))} & {num(x['mean'])} {ci(x['ci95_unadjusted'])} & "
                f"{num(y['mean'])} {ci(y['ci95_unadjusted'])} & {num(z['mean'])} {ci(z['ci95_unadjusted'])} \\\\")
write("r2_spike.tex", [r"\begin{tabular}{cccccc}", r"\toprule",
      r"$\alpha$ & Budget & spiked genes in top 50 & residual $-$ feasible & residual, perm. $-$ feasible & residual $-$ residual, perm. \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- appendix: comparator choices and hindsight gap
rows = []
for b in BUD:
    fe, best = loso(b, "pcc"); g = ms(best - fe)
    ch = S["loso_choices"][f"loso_recon_{b}"]
    dch = S["loso_choices"].get(f"loso_detect_{b}", {})
    fmt = lambda c: ", ".join(f"{k.replace('_', '-')} ({v})" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
    facts[f"hindsight_gap_{b}"] = g
    rows.append(f"{pct(b)} & {fmt(ch)} & {fmt(dch) if dch else '--'} & {num(g['mean'])} {ci(g['ci'])} \\\\")
write("r2_loso.tex", [r"\begin{tabular}{llll}", r"\toprule",
      r"Budget & feasible choice, reconstruction & feasible choice, detection & per-slide best $-$ feasible (PCC) \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- appendix: stage-2 secondary family
rows = []
for r in S["secondary"]:
    name = (r["margin"].replace("residual_guided_perm", "residual, perm.").replace("residual_guided", "residual")
            .replace("value_greedy_const", "local-var., constant").replace("value_greedy", "local-var.")
            .replace("LOSO_recon", "feasible").replace("best_NA hindsight", "per-slide best").replace(" - ", " $-$ "))
    rows.append(f"{name} & {pct(r['budget'])} & {num(r['mean'])} {ci(r['ci95_unadjusted'])} & {pf(r['p_holm'])} \\\\")
write("r2_secondary.tex", [r"\begin{tabular}{llcc}", r"\toprule", r"Margin & Budget & $\Delta$ [95\% CI] & $p_H$ \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])

# ---------------------------------------------------------------- main figure
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555", "axes.labelcolor": "#222222",
                     "legend.frameon": False, "lines.linewidth": 1.6, "pdf.fonttype": 42})


def series(ax, pts, color, label, marker, off=1.0):
    xs = np.array([p[0] for p in pts]) * off; m = np.array([p[1] for p in pts])
    lo = np.array([p[2][0] for p in pts]); hi = np.array([p[2][1] for p in pts])
    ax.errorbar(xs, m, yerr=[m - lo, hi - m], color=color, marker=marker, ms=4.5, capsize=2, elinewidth=1, label=label)


fig, ax3 = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)
ax = ax3[0]
series(ax, [(x * 100, m, c) for x, m, c in fig_a["rg"]], C1, "residual (value-cond.)", "o")
series(ax, [(x * 100, m, c) for x, m, c in fig_a["er"]], GREY, "error-red. (fixed design)", "D", 1.06)
series(ax, [(x * 100, m, c) for x, m, c in fig_a["lv"]], C2, "local-variance (value-cond.)", "s", 0.94)
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xscale("log")
xt = [0.5, 1, 2, 5, 10, 20]; ax.set_xticks(xt); ax.set_xticklabels([f"{x:g}" for x in xt])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(-0.235, 0.105); ax.set_xlabel("budget (% of regions)"); ax.set_ylabel(r"$\Delta$ PCC vs feasible rule")
ax.set_title("(a) Reconstruction", loc="left", fontsize=8.5); ax.legend(fontsize=6.3, loc="upper left")
ax = ax3[1]
series(ax, [(x * 100, m, c) for x, m, c in fig_b["perm"]], C1, "vs permuted feedback", "o")
series(ax, [(x * 100, m, c) for x, m, c in fig_b["fe"]], GREY, "vs feasible rule", "D", 1.06)
series(ax, [(x * 100, m, c) for x, m, c in fig_b["div"]], C3, "vs diversity sampler", "^", 0.94)
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xscale("log")
xt = [2, 5, 10, 20]; ax.set_xticks(xt); ax.set_xticklabels([f"{x:g}" for x in xt])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("budget (% of regions)"); ax.set_ylabel(r"$\Delta$ ROI recall of local-variance")
ax.set_title("(b) ROI detection", loc="left", fontsize=8.5); ax.legend(fontsize=6.3, loc="upper left")
ax = ax3[2]
series(ax, fig_c["0.050"], C1, "5% budget", "o")
series(ax, fig_c["0.100"], C2, "10% budget", "s")
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xticks([0, 1, 2])
ax.set_xlabel(r"spike amplitude $\alpha$"); ax.set_ylabel(r"$\Delta$ PCC, residual $-$ feasible")
ax.set_title("(c) Spiked positive control", loc="left", fontsize=8.5); ax.legend(fontsize=6.3, loc="lower left")
fig.savefig(PAPER / "fig_r2.pdf"); fig.savefig(ROOT / "docs" / "figures" / "fig_r2.png", dpi=200)
print(f"[write] {PAPER / 'fig_r2.pdf'}")


def ser(v):
    if isinstance(v, dict):
        return {k: ser(x) for k, x in v.items()}
    if isinstance(v, (tuple, list, np.ndarray)):
        return [ser(x) for x in v]
    return float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, np.integer) else v)


(GEN / "facts_r2.json").write_text(json.dumps(ser(facts), indent=1), encoding="utf-8")
print(f"[write] {GEN / 'facts_r2.json'}")
