"""
Generate every table and figure of the manuscript from the committed result JSONs (numbers-locked).

Reads results/*.json (n=22 expanded run, synthetic controls, SOFisher native, n7_legacy for the
descriptive imputer comparison) and writes:
  paper/generated/*.tex   LaTeX tables, \\input by paper/manuscript.tex
  paper/fig_*.pdf         figures
Margins are summarized with the study-cluster bootstrap CI and the exact cluster-level sign-flip test
(src/stare/stats.py); "p_H" is Holm-adjusted over the pre-registered 24-margin primary family
(results/power_summary.json, scripts/14_power_summary.py). Local, no data needed.

Run:  python -u scripts/14_power_summary.py && python -u scripts/15_paper_assets.py
"""
import os, sys, json, shutil
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
GEN = PAPER / "generated"
GEN.mkdir(parents=True, exist_ok=True)
NON_ADAPTIVE = ["grid", "kmeans", "oneshot_spatialfinder", "var_greedy", "mi_greedy"]
HD = {"TENX174", "TENX173", "TENX171", "TENX170", "TENX163", "TENX156", "TENX155"}
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b"     # validated categorical slots 1-3


def J(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def num(x, d=3):
    return f"${x:+.{d}f}$".replace("+", "{+}") if x != 0 else f"${0:.{d}f}$"


def ci(c, d=3):
    return f"$[{c[0]:+.{d}f}, {c[1]:+.{d}f}]$".replace("+", "{+}")


def pfmt(p):
    if p is None or np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def margin_stats(per_slide, grouping):
    x = np.asarray(per_slide, float)
    cl = grouping["clusters"]
    lo, hi = stats.cluster_bootstrap_ci(x, cl, B=config.CI_BOOTSTRAP, alpha=config.CI_ALPHA, seed=config.CI_SEED)
    return {"mean": float(x.mean()), "ci": (lo, hi), "p": stats.signflip_p(x, cl), "x": x}


def write(name, body):
    (GEN / name).write_text(body, encoding="utf-8")
    print(f"[write] {GEN / name}")


PS = J("power_summary.json")
HOLM = {(r["experiment"], r["margin"]): r.get("holm_p_primary") for r in PS["rows"]}

ci05, det09, comp10, hvg12, bat13 = (J("ci_multislide_graphflow.json"), J("realdata_detection.json"),
                                     J("competitors.json"), J("hvg_sensitivity.json"), J("batched_contiguous.json"))
G = ci05["grouping"]
assert det09["grouping"] == G and bat13["grouping"] == G and comp10["grouping"] == G, "slide sets differ"
N, K = len(ci05["slides"]), len(set(G["clusters"]))
BUD05 = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
BUD09 = ["0.02", "0.05", "0.10", "0.20"]
pct = lambda b: f"{float(b) * 100:g}\\%"
facts = {"n_slides": N, "n_clusters": K}

# ---------------------------------------------------------------- Table: absolute reconstruction
ps = ci05["per_slide"]
rows = []
for b in BUD05:
    best = np.max(np.vstack([ps[p][b] for p in NON_ADAPTIVE]), axis=0)
    b9 = f"{float(b):.2f}"
    vg = det09["rows"][b9]["reconstruction_pcc_means"]["value_greedy"] if b9 in det09["rows"] else None
    cells = [best.mean(), np.mean(ps["mi_greedy"][b]), np.mean(ps["kmeans"][b]),
             np.mean(ps["ours_closed_loop"][b]), np.mean(ps["adaptive_uncertainty"][b])]
    rows.append(f"{pct(b)} & " + " & ".join(f"{v:.3f}" for v in cells) + f" & {'--' if vg is None else f'{vg:.3f}'} \\\\")
    facts[f"best_na_{b}"] = best.mean()
    for pol in ["adaptive_uncertainty", "ours_closed_loop"]:
        facts[f"{pol}_vs_bestNA_{b}"] = margin_stats(np.asarray(ps[pol][b]) - best, G)
    facts[f"closed_vs_ablate_{b}"] = margin_stats(ci05["margins"][b]["closed_vs_ablate"]["per_slide"], G)
write("tab_recon_abs.tex", "\n".join([
    r"\begin{tabular}{lcccccc}", r"\toprule",
    r" & \multicolumn{3}{c}{Non-adaptive} & \multicolumn{3}{c}{Adaptive} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"Budget & best per slide & MI-greedy & $k$-means & error-red. & max-unc. & value-cond. \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Table: primary margins
rows = []
for b in BUD05:
    m = margin_stats(ci05["margins"][b]["ours_vs_best_nonadapt"]["per_slide"], G)
    hp = HOLM[("ci_multislide_graphflow", f"margins/{b}/ours_vs_best_nonadapt")]
    b9 = f"{float(b):.2f}"
    cells = [f"{num(m['mean'])} {ci(m['ci'])}", pfmt(hp)]
    facts[f"er_{b}"] = m
    for key in ["value_greedy_recon_vs_bestcover", "value_greedy_detect_vs_bestcover"]:
        if b9 in det09["rows"]:
            mm = margin_stats(det09["rows"][b9][key]["per_slide"], G)
            cells += [f"{num(mm['mean'])} {ci(mm['ci'])}", pfmt(HOLM[("realdata_detection", f"rows/{b9}/{key}")])]
            facts[f"{key}_{b9}"] = mm
        else:
            cells += ["--", "--"]
    rows.append(f"{pct(b)} & " + " & ".join(cells) + r" \\")
write("tab_primary.tex", "\n".join([
    r"\begin{tabular}{lcccccc}", r"\toprule",
    r" & \multicolumn{4}{c}{Reconstruction (PCC, top-50 genes)} & \multicolumn{2}{c}{Detection (ROI recall)} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
    r" & \multicolumn{2}{c}{error-reduction $-$ best non-adapt.} & \multicolumn{2}{c}{value-cond. $-$ best coverage} & \multicolumn{2}{c}{value-cond. $-$ best coverage} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"Budget & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p_H$ \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Table: physical realism (13)
SETTINGS = [("1", "Sequential (1 region per round)"), ("5", "Rounds of 5 regions"), ("25", "Rounds of 25 regions"),
            ("oneshot", "Single round (whole budget)")]
rows = []
for key, label in SETTINGS:
    r = bat13["partA_batched"][key]
    cells = []
    for t in ["recon_margin", "detect_margin"]:
        mm = margin_stats(r[t]["per_slide"], G)
        cells += [f"{num(mm['mean'])} {ci(mm['ci'])}", pfmt(HOLM[("batched_contiguous", f"partA_batched/{key}/{t}")])]
        facts[f"bat_{key}_{t}"] = mm
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
r = bat13["partB_contiguous"]
cells = []
for t in ["recon_margin", "detect_margin"]:
    mm = margin_stats(r[t]["per_slide"], G)
    cells += [f"{num(mm['mean'])} {ci(mm['ci'])}", pfmt(HOLM[("batched_contiguous", f"partB_contiguous/{t}")])]
    facts[f"contig_{t}"] = mm
rows.append(f"Contiguous blocks of {bat13['block_k']} regions & " + " & ".join(cells) + r" \\")
write("tab_realism.tex", "\n".join([
    r"\begin{tabular}{lcccc}", r"\toprule",
    r" & \multicolumn{2}{c}{Reconstruction (PCC)} & \multicolumn{2}{c}{Detection (ROI recall)} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"Acquisition setting & $\Delta$ [95\% CI] & $p_H$ & $\Delta$ [95\% CI] & $p_H$ \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Table: absolute detection recall (10)
POL = [("kmeans", "$k$-means"), ("grid", "grid"), ("mi_greedy", "MI-greedy"), ("scr2st", "diversity$^\\dagger$"),
       ("sofisher", "ROI-seeking$^\\dagger$"), ("value_greedy", "value-cond."), ("oracle", "oracle$^\\ast$")]
rows = []
for b in BUD09:
    rm = comp10["rows"][b]["detection_recall_means"]
    rows.append(f"{pct(b)} & " + " & ".join(f"{rm[p]:.3f}" for p, _ in POL) + r" \\")
    for p in ["scr2st", "sofisher"]:
        facts[f"{p}_detect_{b}"] = margin_stats(comp10["rows"][b]["detect_margin_vs_bestcover"][p]["per_slide"], G)
        facts[f"{p}_recon_{b}"] = margin_stats(comp10["rows"][b]["recon_margin_vs_bestcover"][p]["per_slide"], G)
write("tab_detect_abs.tex", "\n".join([
    r"\begin{tabular}{l" + "c" * len(POL) + "}", r"\toprule",
    r" & \multicolumn{3}{c}{Non-adaptive coverage} & \multicolumn{2}{c}{Competitor objectives} & & \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
    "Budget & " + " & ".join(l for _, l in POL) + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Appendix: slide list
audit = J("scale_audit.json")["rows"]
cand = J("sample_candidates.json")["picks"]
study = {p["id"]: p["study"] for t in cand.values() for p in t}
study_ids = {s: i + 1 for i, s in enumerate(dict.fromkeys(G["clusters"]))}
rows = []
for lab, cl in zip(ci05["slides"], G["clusters"]):
    t, sid = lab.split(":")
    a = audit[sid]
    tech = "Visium HD" if sid in HD else "Visium"
    title = study.get(sid, cl).replace("&", r"\&").replace("_", r"\_")
    rows.append(f"{t} & {sid} & {tech} & {a['n_spots_in_tissue']} & {a['px_um']:.3f} & {a['nn_pitch_true_um']:.0f} & "
                f"{study_ids[cl]} & {title[:58]}{'...' if len(title) > 58 else ''} \\\\")
write("tab_slides.tex", "\n".join([
    r"\begin{tabular}{lllrrrrl}", r"\toprule",
    r"Tissue & HEST ID & Platform & Spots & $\mu$m/px & Pitch ($\mu$m) & Cluster & Study (HEST dataset title) \\",
    r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Appendix: platform and tissue splits
ids = [s.split(":")[1] for s in ci05["slides"]]
hd = np.array([i in HD for i in ids])
strata = np.array(G["strata"])
SPLITS = [("Recon., error-red. (1\\%)", ci05["margins"]["0.010"]["ours_vs_best_nonadapt"]["per_slide"]),
          ("Recon., value-cond. (10\\%)", det09["rows"]["0.10"]["value_greedy_recon_vs_bestcover"]["per_slide"]),
          ("Detect., value-cond. (10\\%)", det09["rows"]["0.10"]["value_greedy_detect_vs_bestcover"]["per_slide"]),
          ("Detect., value-cond. (20\\%)", det09["rows"]["0.20"]["value_greedy_detect_vs_bestcover"]["per_slide"]),
          ("Detect., rounds of 5 (10\\%)", bat13["partA_batched"]["5"]["detect_margin"]["per_slide"])]
tissues = ["prostate", "kidney", "lung", "colon"]
rows = []
for label, x in SPLITS:
    x = np.asarray(x, float)
    cells = []
    for mask in [hd, ~hd]:
        c = [G["clusters"][i] for i in np.flatnonzero(mask)]
        cells.append(f"{num(x[mask].mean())} ({pfmt(stats.signflip_p(x[mask], c))})")
    cells += [num(x[strata == t].mean()) for t in tissues]
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
n_hd_cl = len({G["clusters"][i] for i in np.flatnonzero(hd)})
n_v_cl = len({G["clusters"][i] for i in np.flatnonzero(~hd)})
facts.update(n_hd=int(hd.sum()), n_hd_cl=n_hd_cl, n_vis=int((~hd).sum()), n_vis_cl=n_v_cl)
write("tab_split.tex", "\n".join([
    r"\begin{tabular}{lcccccc}", r"\toprule",
    rf" & HD ($n$={hd.sum()}, {n_hd_cl} cl.) & Visium ($n$={(~hd).sum()}, {n_v_cl} cl.) & " +
    " & ".join(f"{t} ($n$={(strata == t).sum()})" for t in tissues) + r" \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Appendix: HVG sensitivity
rows = []
for b in BUD09:
    cells = []
    for gs in ["slide_hvg", "ref_hvg", "all_genes"]:
        mm = margin_stats(hvg12["rows"][b][gs]["margin_vs_bestcover"]["per_slide"], G)
        cells.append(f"{num(mm['mean'])} {ci(mm['ci'])}")
    rows.append(f"{pct(b)} & " + " & ".join(cells) + r" \\")
write("tab_hvg.tex", "\n".join([
    r"\begin{tabular}{lccc}", r"\toprule",
    r"Budget & per-slide top-50 HVG & reference-subset top-50 HVG & all filtered genes \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Appendix: imputer comparison (legacy, human HD only)
EXCL = {"TENX186", "TENX175"}            # non-human per HEST v1.3 metadata (excluded from every analysis)
rows = []
for b in BUD05:
    cells = []
    for imp in ["knn", "kriging", "graphflow"]:
        L = json.loads((RES / "n7_legacy" / f"ci_multislide_{imp}.json").read_text(encoding="utf-8"))
        keep = [i for i, s in enumerate(L["slides"]) if s.split(":")[1] not in EXCL]
        x = np.asarray(L["margins"][b]["ours_vs_best_nonadapt"]["per_slide"], float)[keep]
        cells.append(f"{num(x.mean())} ({int((x > 0).sum())}/{len(x)})")
    rows.append(f"{pct(b)} & " + " & ".join(cells) + r" \\")
facts["legacy_n"] = len(keep)
write("tab_imputers_legacy.tex", "\n".join([
    r"\begin{tabular}{lccc}", r"\toprule",
    r"Budget & kNN & kriging & graphflow \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- Figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "xtick.color": "#555555",
                     "ytick.color": "#555555", "axes.labelcolor": "#222222", "legend.frameon": False,
                     "lines.linewidth": 1.6, "pdf.fonttype": 42})


def band(ax, xs, ms, color, label, marker, offset=1.0):
    xs = np.asarray(xs) * offset
    mean = [m["mean"] for m in ms]
    lo = [m["ci"][0] for m in ms]; hi = [m["ci"][1] for m in ms]
    ax.errorbar(xs, mean, yerr=[np.subtract(mean, lo), np.subtract(hi, mean)], color=color, marker=marker,
                ms=4.5, capsize=2, elinewidth=1, label=label)


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
ax = axes[0]
xs5 = [float(b) * 100 for b in BUD05]; xs9 = [float(b) * 100 for b in BUD09]
band(ax, xs5, [facts[f"er_{b}"] for b in BUD05], C1, "error-reduction $-$ best non-adapt.", "o")
band(ax, xs9, [facts[f"value_greedy_recon_vs_bestcover_{b}"] for b in BUD09], C2, "value-cond. $-$ best coverage", "s", 1.06)
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xticks(xs5); ax.set_xticklabels([f"{x:g}" for x in xs5])
ax.set_xlabel("budget (% of regions)"); ax.set_ylabel(r"$\Delta$ PCC (top-50 genes)"); ax.set_title("(a) Reconstruction", loc="left", fontsize=8.5)
ax.set_ylim(-0.235, 0.075)
ax.legend(fontsize=6.5, loc="upper left", handlelength=1.6)
ax = axes[1]
band(ax, xs9, [facts[f"value_greedy_detect_vs_bestcover_{b}"] for b in BUD09], C2, "value-cond. (adaptive)", "s")
band(ax, xs9, [facts[f"scr2st_detect_{b}"] for b in BUD09], C3, "diversity (non-adaptive)", "^", 1.06)
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xticks(xs9); ax.set_xticklabels([f"{x:g}" for x in xs9])
ax.set_xlabel("budget (% of regions)"); ax.set_ylabel(r"$\Delta$ ROI recall vs best coverage"); ax.set_title("(b) ROI detection", loc="left", fontsize=8.5)
ax.legend(fontsize=6.5, loc="upper left")
ax = axes[2]
labs = ["1", "5", "25", "all", "blocks"]
ms = [facts[f"bat_{k}_detect_margin"] for k in ["1", "5", "25", "oneshot"]] + [facts["contig_detect_margin"]]
xs = np.arange(len(ms))
mean = [m["mean"] for m in ms]
ax.errorbar(xs, mean, yerr=[[m["mean"] - m["ci"][0] for m in ms], [m["ci"][1] - m["mean"] for m in ms]],
            color=C2, marker="s", ms=4.5, capsize=2, elinewidth=1, ls="none")
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xticks(xs); ax.set_xticklabels(labs)
ax.set_xlabel("regions per round (10% budget)"); ax.set_ylabel(r"$\Delta$ ROI recall vs best coverage")
ax.set_title("(c) Batched acquisition", loc="left", fontsize=8.5)
for a in axes[:2]:
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
fig.savefig(PAPER / "fig_main.pdf"); fig.savefig(ROOT / "docs" / "figures" / "fig_main.png", dpi=200)
print(f"[write] {PAPER / 'fig_main.pdf'}")

pc, dc = J("positive_control.json"), J("detection_control.json")
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), constrained_layout=True)
ax = axes[0]
ells = [str(e) for e in pc["rough_ells"]]
for key, col, lab, mk in [("oracle_vs_bestcover", GREY, "oracle", "D"), ("greedy_ig_vs_bestcover", C1, "error-reduction", "o")]:
    m = [pc["rows"][e][key] for e in ells]
    mean = [r["mean"] for r in m]
    ax.errorbar([float(e) for e in ells], mean, yerr=[[r["mean"] - r["ci95"][0] for r in m], [r["ci95"][1] - r["mean"] for r in m]],
                color=col, marker=mk, ms=4, capsize=2, elinewidth=1, label=lab)
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xlabel("structure correlation length (regions)")
ax.set_ylabel(r"$\Delta$ PCC vs best coverage"); ax.set_title("(a) Synthetic reconstruction", loc="left", fontsize=8.5)
ax.legend(fontsize=6.5, loc="lower right")
ax = axes[1]
al = [str(a) for a in dc["alphas"]]
for key, col, lab, mk in [("value_greedy_vs_bestcover", C2, "value-cond.", "s"), ("greedy_ig_vs_bestcover", C1, "error-reduction", "o")]:
    m = [dc["rows"][a][key] for a in al]
    mean = [r["mean"] for r in m]
    ax.errorbar([float(a) for a in al], mean, yerr=[[r["mean"] - r["ci95"][0] for r in m], [r["ci95"][1] - r["mean"] for r in m]],
                color=col, marker=mk, ms=4, capsize=2, elinewidth=1, label=lab)
ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.set_xlabel(r"ROI signal amplitude $\alpha$")
ax.set_ylabel(r"$\Delta$ ROI recall vs best coverage"); ax.set_title("(b) Synthetic detection", loc="left", fontsize=8.5)
ax.legend(fontsize=6.5, loc="upper left")
fig.savefig(PAPER / "fig_controls.pdf"); fig.savefig(ROOT / "docs" / "figures" / "fig_controls.png", dpi=200)
print(f"[write] {PAPER / 'fig_controls.pdf'}")

shutil.copy(ROOT / "docs" / "figures" / "figure4_sofisher_native.pdf", PAPER / "fig_sofisher_native.pdf")

# ---------------------------------------------------------------- facts for the text (checked against the prose)
def ser(v):
    if isinstance(v, dict):
        return {k: ser(x) for k, x in v.items() if k != "x"}
    if isinstance(v, (tuple, list, np.ndarray)):
        return [ser(x) for x in v]
    return float(v) if isinstance(v, (np.floating, float)) else v
(GEN / "facts.json").write_text(json.dumps(ser(facts), indent=1), encoding="utf-8")
print(f"[write] {GEN / 'facts.json'}  (n={N}, clusters={K})")
