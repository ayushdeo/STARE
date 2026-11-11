"""
Tables for the stage-4 analyses (common held-out evaluation set, constant-feedback controls, 300 um tiles, tuned
local-variance, ground-truth greedy headroom). Numbers-locked: reads only result JSONs.

Inputs : results/stage4_summary.json (script 28, registered Holm p-values and MDEs), results/stage4_runs.json
         (script 27; tuned settings, descriptive).
Outputs: paper/generated/s4_*.tex, paper/generated/facts_s4.json.

Run after 28:  python -u scripts/29_paper_assets_stage4.py
"""
import os, sys, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, PAPER = ROOT / "results", ROOT / "paper"
GEN = PAPER / "generated"; GEN.mkdir(parents=True, exist_ok=True)
J = lambda n: json.loads((RES / n).read_text(encoding="utf-8"))
pct = lambda b: f"{float(b) * 100:g}\\%"
NAME = {"kmeans": "$k$-means", "mi_greedy": "MI-greedy", "var_greedy": "variance-greedy", "oneshot": "one-shot"}
BUD = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
DET = ["0.020", "0.050", "0.100", "0.200"]


def num(x, d=3):
    return f"${x:+.{d}f}$".replace("+", "{+}") if round(x, d) != 0 else f"${0:.{d}f}$"


def ci(c, d=3):
    end = lambda x: f"{0:.{d}f}" if round(x, d) == 0 else f"{x:+.{d}f}"
    return f"$[{end(c[0])}, {end(c[1])}]$".replace("+", "{+}")


def pf(p):
    return "--" if p is None or np.isnan(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write(name, lines):
    (GEN / name).write_text("\n".join(lines), encoding="utf-8"); print(f"[write] {GEN / name}")


def label(m):
    """Display name for a registered margin string."""
    s = m.split(" (")[0]
    s = s.replace("reconstruction hybrid", "50\\% reconstruction hybrid").replace("detection hybrid", "50\\% detection hybrid")
    for k, v in NAME.items():
        s = s.replace(" - " + k, " - " + v)
    return s.replace(" - ", " $-$ ")


S, R = J("stage4_summary.json"), J("stage4_runs.json")
facts = {}


def find(rows, start, b):
    for r in rows:
        if r["margin"].startswith(start) and r["budget"] == b:
            return r
    return None


def cell(r):
    return "-- & --" if r is None else f"{num(r['mean'])} {ci(r['ci95_boot'])} & {pf(r['p_holm'])}"


# ---------------------------------------------------------------- main text: the three registered primary questions
EP, PP = S["E"]["primary"], S["P"]["primary"]
rows = []
for b in BUD:
    res, hyb, tun = find(EP, "residual - feasible", b), find(EP, "reconstruction hybrid", b), find(PP, "tuned", b)
    base = NAME[hyb["margin"].split(" - ")[1].split(" (")[0]]
    rows.append(f"{pct(b)} & {cell(res)} & {base} & {cell(hyb)} & {cell(tun)} \\\\")
    for k, r in (("res", res), ("hyb", hyb), ("tuned", tun)):
        if r is not None:
            facts[f"{k}_{b}"] = {x: r[x] for x in ("mean", "ci95_boot", "ci95_signflip", "p_holm", "mde80_family", "n_pos")}
write("s4_main.tex", [r"\begin{tabular}{lccccccc}", r"\toprule",
      r" & \multicolumn{2}{c}{residual $-$ feasible (PCC)} & \multicolumn{3}{c}{50\% hybrid $-$ base design (PCC)}"
      r" & \multicolumn{2}{c}{tuned local-variance $-$ feasible (recall)} \\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
      r"Budget & $\Delta$ [95\% CI] & $p_H$ & base design & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p_H$ \\",
      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


# ---------------------------------------------------------------- appendix: long-format families
def long_table(name, groups, unit):
    out = [r"\begin{tabular}{llccc}", r"\toprule", r"Margin & Budget & $\Delta$ [95\% CI] & $p_H$ & MDE$_H$ \\"]
    for title, rs in groups:
        out += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\"]
        for r in rs:
            b = f"{r['budget']} tiles" if unit == "tiles" else pct(r["budget"])
            mde = "--" if r["mde80_family"] is None else f"{r['mde80_family']:.3f}"
            out.append(f"{label(r['margin'])} & {b} & {num(r['mean'])} {ci(r['ci95_boot'])} & {pf(r['p_holm'])} & {mde} \\\\")
    write(name, out + [r"\bottomrule", r"\end{tabular}"])


ES = S["E"]["secondary"]
order = lambda rs, key: sorted([r for r in rs if r["margin"].startswith(key)], key=lambda r: float(r["budget"]))
long_table("s4_E_secondary.tex", [
    ("Reconstruction (held-out PCC; MSE row: feasible $-$ residual, positive favours residual)",
     order(ES, "feasible - residual") + order(ES, "residual - constant") + order(ES, "reconstruction hybrid - constant")),
    ("Detection (ROI recall)",
     order(ES, "local-variance - constant") + [r for r in order(ES, "detection hybrid -") if "constant" not in r["margin"]]
     + order(ES, "detection hybrid - constant"))], "frac")
TP, TS = S["T"]["primary"], S["T"]["secondary"]
long_table("s4_T.tex", [("Primary (9 margins)", TP), ("Secondary (9 margins)", TS)], "tiles")

ED = S["E"]["descriptive"]
for r in ED:
    facts[f"desc_{r['margin'].split(' (')[0]}_{r['budget']}"] = {x: r[x] for x in ("mean", "ci95_boot", "p")}
for r in S["T"]["descriptive"]:
    facts[f"Tdesc_{r['margin'].split(' (')[0]}_{r['budget']}"] = {x: r[x] for x in ("mean", "ci95_boot", "p")}

# ---------------------------------------------------------------- appendix: ground-truth greedy headroom (Part U)
SL = {"prostate:TENX174": "TENX174 (prostate)", "kidney:TENX173": "TENX173 (kidney)", "lung:TENX171": "TENX171 (lung)",
      "colon:TENX155": "TENX155 (colon)"}
out = [r"\begin{tabular}{llccccc}", r"\toprule",
       r"Slide & Budget & greedy bound & best fixed design & PCC & residual & bound $-$ best fixed \\", r"\midrule"]
for r in S["U"]:
    out.append(f"{SL[r['slide']]} & {pct(r['budget'])} & {r['greedy_truth_pcc']:.3f} & {NAME[r['best_design']]} & "
               f"{r['best_design_pcc']:.3f} & {r['residual_pcc']:.3f} & {num(r['headroom_over_best_design'])} \\\\")
write("s4_U.tex", out + [r"\bottomrule", r"\end{tabular}"])
facts["U_headroom_range"] = [min(r["headroom_over_best_design"] for r in S["U"]), max(r["headroom_over_best_design"] for r in S["U"])]

# ---------------------------------------------------------------- tuned settings chosen per held-out study (descriptive)
L, CL = R["slides"], np.asarray(R["grouping"]["clusters"], dtype=object)
picks = {}
for b in DET:
    configs = sorted(R["P"][L[0]][b])
    vals = {c: np.array([R["P"][l][b][c]["recall"] for l in L], float) for c in configs}
    cnt = collections.Counter(max(configs, key=lambda c: vals[c][CL != s].mean()) for s in sorted(set(CL), key=str))
    picks[b] = dict(cnt)
facts["P_picks"] = picks
(GEN / "facts_s4.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
print("[write]", GEN / "facts_s4.json")
for b, p in picks.items():
    print(b, p)
