"""
Numbers-locked results table generator (paper Table 1 + verdicts).

Reads every results/sweep_<tissue>_<imputer>.json that exists and emits docs/RESULTS_TABLE.md.
EVERY number is pulled from the JSON at print time and rounded here — nothing is hand-transcribed,
so the table cannot drift from the runs. Imputers with no JSON yet (e.g. graphflow before the 4070
run) render as PENDING rows, so this is a working STUB now that auto-fills on the next `git pull`.

The headline claim sentence is COMPUTED from the verdicts (no-gain vs gap), so it stays honest:
it will only say "adaptivity provides no gain" if the data actually shows that for every imputer.

Run:  python -u scripts/04_make_results_table.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
from stare import config

TISSUE = config.FEASIBILITY_TISSUE
IMPUTERS = config.SWEEP_IMPUTERS                      # [knn, kriging, graphflow]
BUDGETS = [f"{b:.3f}" for b in config.SWEEP_BUDGET_FRACTIONS]
RESULTS = Path(config.RESULTS_ROOT)
OUT = Path("docs/RESULTS_TABLE.md")

IMPUTER_LABEL = {
    "knn":       "kNN (stationary)",
    "kriging":   "regression-kriging (learned, linear, non-stationary)",
    "graphflow": "GraphFlow (STFlow-class: nonlinear + spatial message passing)",
}
# display order + labels; ★ marks the principled non-adaptive opponent
POLICY_ROWS = [
    ("random",                "Random (floor)"),
    ("grid",                  "Grid"),
    ("kmeans",                "k-means (image)"),
    ("oneshot_spatialfinder", "One-shot (SpatialFinder-style)"),
    ("var_greedy",            "Variance-greedy [Krause, non-adaptive]"),
    ("mi_greedy",             "MI-greedy [Krause, non-adaptive] ★"),
    ("adaptive_uncertainty",  "Adaptive-uncertainty (ours, closed-loop)"),
    ("ours_closed_loop",      "Error-reduction (ours, closed-loop)"),
    ("ablate_no_measured",    "└ ablation: measured signal OFF"),
]


def load(imputer):
    p = RESULTS / f"sweep_{TISSUE}_{imputer}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_ci(imputer):
    p = RESULTS / f"ci_multislide_{imputer}.json"
    return json.loads(p.read_text()) if p.exists() else None


CI_CONTRASTS = [
    ("ours_vs_best_nonadapt", "ours − best non-adaptive (the decisive test)"),
    ("ours_vs_mi",            "ours − MI-greedy"),
    ("adaptive_vs_var",       "adaptive − variance-greedy"),
    ("closed_vs_ablate",      "ours − ablation (measured off)"),
]


def ci_table(cid):
    """Cross-slide margin table for one imputer's ci_multislide JSON."""
    budgets = cid["budgets"]
    lines = [f"_slides ({len(cid['slides'])}): {', '.join(cid['slides'])}_\n",
             "| Contrast | " + " | ".join(f"{float(b)*100:g}%" for b in budgets) + " |",
             "|---|" + "---|" * len(budgets)]
    for key, label in CI_CONTRASTS:
        cells = []
        for b in budgets:
            m = cid["margins"].get(b, {}).get(key)
            if not m:
                cells.append("—"); continue
            star = " ✔" if m["ci_excludes_zero"] and m["mean"] > 0 else ""
            cells.append(f"{m['mean']:+.3f} [{m['ci95'][0]:+.3f},{m['ci95'][1]:+.3f}]{star}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    win = cid.get("win_budgets_ours_vs_best_nonadapt")
    lines.append(f"\n_✔ = 95% CI excludes 0 (positive). WIN budgets (ours > best non-adaptive): "
                 f"**{win if win else 'NONE'}**_\n")
    return "\n".join(lines)


def cell(v, std=None):
    if v is None:
        return "—"
    return f"{v:.3f} ±{std:.3f}" if std is not None else f"{v:.3f}"


def metric_table(data, metric):
    """One imputer's policy×budget table for `metric`; bold the best value per budget column."""
    R = {n: data["results"].get(n, {}) for n, _ in POLICY_ROWS}
    # best (max) value per budget among real methods (exclude the ablation row from the bold pick)
    best = {}
    for b in BUDGETS:
        vals = [R[n].get(b, {}).get(metric) for n, _ in POLICY_ROWS if n != "ablate_no_measured"]
        vals = [x for x in vals if isinstance(x, (int, float))]
        best[b] = max(vals) if vals else None
    lines = ["| Policy | " + " | ".join(f"{float(b)*100:g}%" for b in BUDGETS) + " |",
             "|---|" + "---|" * len(BUDGETS)]
    for name, label in POLICY_ROWS:
        cells = []
        for b in BUDGETS:
            d = R[name].get(b, {})
            v = d.get(metric)
            s = d.get(metric + "_std") if name == "random" else None
            txt = cell(v, s)
            if isinstance(v, (int, float)) and best[b] is not None and abs(v - best[b]) < 1e-12 and name != "ablate_no_measured":
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict_table(data):
    v = data.get("verdicts", {})
    lines = ["| Budget | beats MI-greedy? | closed-loop helps? | adaptivity gap (adaptive−var) |",
             "|---|---|---|---|"]
    for b in BUDGETS:
        vb = v.get(b, {})
        bm = vb.get("beats_mi_greedy"); clh = vb.get("closed_loop_helps"); ag = vb.get("adaptivity_gap")
        d_mi = vb.get("ours_minus_mi"); d_cl = vb.get("closed_minus_nomeasured")
        bm_s = "—" if bm is None else (f"yes (+{d_mi})" if bm else f"no ({d_mi})")
        cl_s = "—" if clh is None else (f"yes (+{d_cl})" if clh else f"no ({d_cl})")
        ag_s = "—" if ag is None else (f"+{ag}" if ag > 0 else f"{ag}")
        lines.append(f"| {float(b)*100:g}% | {bm_s} | {cl_s} | {ag_s} |")
    return "\n".join(lines)


def compute_claim(loaded):
    """Honest, data-derived headline. Only asserts 'no gain' if EVERY present imputer shows it."""
    present = {k: d for k, d in loaded.items() if d is not None}
    if not present:
        return "_No result files found yet — run scripts/03_budget_sweep.py on the 4070._"
    any_core = any(d.get("core_pass_budgets") for d in present.values())
    any_clh = any(vb.get("closed_loop_helps") for d in present.values() for vb in d.get("verdicts", {}).values())
    any_adapt = any(vb.get("adaptive_beats_nonadaptive") for d in present.values()
                    for vb in d.get("verdicts", {}).values())
    names = ", ".join(present.keys())
    pend = [k for k in IMPUTERS if loaded.get(k) is None]
    pend_s = f" _(pending: {', '.join(pend)})_" if pend else ""
    if not any_core and not any_clh and not any_adapt:
        return (f"**Across every imputer tested ({names}) and every budget, no closed-loop policy beats "
                f"the non-adaptive MI-greedy placement, conditioning on measured expression never "
                f"improves reconstruction (closed-loop-helps = False everywhere), and adaptive placement "
                f"never beats its non-adaptive twin. Adaptivity provides no measurable gain — consistent "
                f"with fixed-kernel sensor-placement theory [Krause et al., 2008].**{pend_s}")
    parts = []
    if any_core: parts.append("some budget passes the core gate (closed loop beats MI-greedy AND helps)")
    if any_clh:  parts.append("closed-loop-helps is True in ≥1 cell")
    if any_adapt: parts.append("adaptive placement beats its non-adaptive twin in ≥1 cell")
    return (f"**Mixed result across ({names}): " + "; ".join(parts) +
            f". Inspect the tables below before claiming a negative result.**{pend_s}")


def main():
    loaded = {imp: load(imp) for imp in IMPUTERS}
    md = []
    md.append("# Results — Numbers-Locked Table (auto-generated)\n")
    md.append("> Generated by `scripts/04_make_results_table.py` directly from "
              "`results/sweep_" + TISSUE + "_*.json`. Do not edit by hand; re-run the script after "
              "each sweep. Metric = **PCC over top-50 HVG** (primary) and PCC-mean (full "
              "transcriptome, secondary), scored on UNPROFILED regions.\n")
    r0 = next((d for d in loaded.values() if d), None)
    if r0:
        md.append(f"Tissue **{TISSUE}** · sample-derived regions **{r0.get('regions','?')}** · "
                  f"encoder **{r0.get('encoder','?')}** · budgets = fraction of regions profiled.\n")
    md.append("## Headline\n")
    md.append(compute_claim(loaded) + "\n")

    md.append("## Table 1 — PCC-top50 by policy × budget (per imputer)\n")
    md.append("Best value per budget column in **bold** (ablation excluded from the bold pick). "
              "★ = the principled non-adaptive opponent the closed loop must beat.\n")
    for imp in IMPUTERS:
        d = loaded[imp]
        md.append(f"### {imp} — {IMPUTER_LABEL.get(imp, imp)}\n")
        if d is None:
            md.append(f"> **PENDING** — `results/sweep_{TISSUE}_{imp}.json` not present yet. "
                      f"Run the sweep on the 4070; this section auto-fills on next `git pull`.\n")
            continue
        md.append(metric_table(d, "pcc_top50") + "\n")

    md.append("## Table 2 — Verdicts by imputer × budget\n")
    for imp in IMPUTERS:
        d = loaded[imp]
        md.append(f"### {imp}\n")
        if d is None:
            md.append("> PENDING.\n"); continue
        md.append(verdict_table(d) + "\n")
        cp = d.get("core_pass_budgets")
        md.append(f"_core-pass budgets (beat MI-greedy AND closed-loop-helps): "
                  f"{cp if cp else 'NONE'}_\n")

    md.append("## Table 3 — PCC-mean (full transcriptome, secondary) by policy × budget\n")
    for imp in IMPUTERS:
        d = loaded[imp]
        md.append(f"### {imp}\n")
        if d is None:
            md.append("> PENDING.\n"); continue
        md.append(metric_table(d, "pcc_mean") + "\n")

    # Cross-slide CI tables (iteration 3) — the significance layer that decides the claim.
    ci_loaded = {imp: load_ci(imp) for imp in IMPUTERS}
    md.append("## Table 4 — Cross-slide margins with 95% CI (the significance test)\n")
    md.append("Paired margin across all pooled slides (mean [CI]); the closed loop only *wins* where "
              "`ours − best non-adaptive` is positive with a CI excluding 0 (✔). Source: "
              "`results/ci_multislide_*.json` via `scripts/05_multislide_ci.py`.\n")
    if not any(ci_loaded.values()):
        md.append("> **PENDING** — no `ci_multislide_*.json` yet. Run "
                  "`python -u scripts/05_multislide_ci.py` on the 4070 (pools "
                  f"{', '.join(config.CI_TISSUES)}); this section auto-fills on next `git pull`.\n")
    else:
        for imp in IMPUTERS:
            cid = ci_loaded[imp]
            md.append(f"### {imp}\n")
            md.append((ci_table(cid) if cid else "> PENDING.\n"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"[write] {OUT}")
    present = [k for k, v in loaded.items() if v is not None]
    pending = [k for k, v in loaded.items() if v is None]
    print(f"[imputers] present={present or 'none'}  pending={pending or 'none'}")


if __name__ == "__main__":
    main()
