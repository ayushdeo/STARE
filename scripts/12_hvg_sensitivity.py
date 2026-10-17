"""
HVG-leakage sensitivity (revision point 8; RUN ON 4070).

Concern: the top-50 HVGs are selected from each COMPLETE slide, which is unavailable in a prospective
partially-observed setting -- the evaluation target leaks full-slide information. We test whether the
reconstruction result (adaptive does not beat strong coverage) is robust to how the evaluation gene set
is chosen. For each slide we score the SAME policy runs under several gene-set definitions:

  * slide_hvg : top-50 HVG from the full slide (current, leaky)         -- baseline for comparison
  * ref_hvg   : top-50 HVG from a fixed random 15%% reference subset     -- prospective pilot, no eval leak
  * all_genes : every gene passing the 10%% filter (PCC-mean)            -- no selection at all, zero leak

Each policy is run once per (slide, budget); the imputer reconstructs ALL genes, so we re-score different
gene subsets without re-running. We report, across slides, the margin (value_greedy - best coverage) under
EACH gene-set definition, with a bootstrap CI. If the margin stays <= 0 under every definition, the
leak-free gene choice does not change the conclusion. Writes results/hvg_sensitivity.json.

Run:  python -u scripts/12_hvg_sensitivity.py
"""
import os, sys, json, pickle
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import numpy as np

from stare import config, metrics, stats
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.slides import load_ci_slides
from stare.policies import build_policy

IMPUTER = "graphflow"
BUDGETS = [0.02, 0.05, 0.10, 0.20]
COVERAGE = ["kmeans", "grid", "mi_greedy"]
ADAPTIVE = "value_greedy"
REF_FRAC = 0.15
BOOT = 5000


def log(m): print(m, flush=True)


# study clusters + tissue strata of the loaded slides (set in main; used by every margin summary)
GROUPING = {}




def gene_sets(slide, n=config.N_HVG, seed=0):
    """Dict of eval gene-index arrays (or None for all genes)."""
    expr = slide.expression
    rng = np.random.default_rng(seed)
    R = slide.n_regions
    ref = rng.choice(R, size=max(n, int(REF_FRAC * R)), replace=False)
    ref_hvg = np.argsort(expr[ref].var(0))[::-1][:n].astype(int)
    return {"slide_hvg": np.asarray(slide.hvg_idx, dtype=int),
            "ref_hvg": ref_hvg,
            "all_genes": None}


def score_policy(slide, imputer, policy_name, bf):
    """Run policy once; return (y_true, y_pred) over unprofiled regions (all genes)."""
    sim = AcquisitionSimulator(slide, imputer, bf)
    pol = build_policy(policy_name, **({"beta": 0.5} if policy_name == ADAPTIVE else {}))
    st = sim.reset()
    while not sim.done:
        st = sim.reveal(pol.select(st))
    state = sim._state()
    unpro = state.unprofiled
    y_true = slide.expression[unpro]
    y_pred = imputer.impute(state)
    return y_true, y_pred


def pcc_over(y_true, y_pred, idx):
    r = metrics.per_gene_pcc(y_true, y_pred)
    r = r if idx is None else r[idx]
    return float(np.nanmean(r))


def main():
    imputer = build_imputer(IMPUTER)
    slides, labels, group = load_ci_slides(log=log)
    GROUPING.update(clusters=group["clusters"], strata=group["strata"])
    log(f"=== HVG sensitivity | slides={labels} imputer={IMPUTER} budgets={BUDGETS} ===")
    GS = ["slide_hvg", "ref_hvg", "all_genes"]
    rows = {}
    for bf in BUDGETS:
        # per geneset: pcc per policy per slide
        acc = {gs: {p: [] for p in COVERAGE + [ADAPTIVE]} for gs in GS}
        for si, slide in enumerate(slides):
            gsets = gene_sets(slide, seed=config.CI_SEED)
            for p in COVERAGE + [ADAPTIVE]:
                yt, yp = score_policy(slide, imputer, p, bf)
                for gs in GS:
                    acc[gs][p].append(pcc_over(yt, yp, gsets[gs]))
            log(f"  [{bf:.2f}] slide {si+1}/{len(slides)} {labels[si]}")
        rows[f"{bf:.2f}"] = {}
        for gs in GS:
            best_cov = np.maximum.reduce([acc[gs][p] for p in COVERAGE])
            m = stats.paired_margin_summary(np.array(acc[gs][ADAPTIVE]), best_cov, BOOT, config.CI_ALPHA, config.CI_SEED, **GROUPING)
            rows[f"{bf:.2f}"][gs] = {"adaptive_mean": float(np.mean(acc[gs][ADAPTIVE])),
                                     "best_cover_mean": float(np.mean(best_cov)),
                                     "margin_vs_bestcover": m}
            log(f"    {gs:10s} adaptive-bestcover = {m['mean']:+.3f} CI={m['ci95']} excl0={m['ci_excludes_zero']}")

    out = Path(config.RESULTS_ROOT) / "hvg_sensitivity.json"
    out.write_text(json.dumps({"imputer": IMPUTER, "slides": labels, "grouping": GROUPING, "region_mode": config.REGION_MODE, "budgets": BUDGETS,
                               "ref_frac": REF_FRAC, "gene_sets": GS, "rows": rows}, indent=2), encoding="utf-8")
    log(f"[write] {out}")
    log("\nExpected: the reconstruction margin stays <= 0 under slide_hvg, ref_hvg, AND all_genes -> the "
        "full-slide HVG selection does not drive the result (no leakage artifact).")


if __name__ == "__main__":
    main()
